import logging
from pathlib import Path

import napari
import napari.layers
import numpy as np
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FileEdit,
    Label,
    LineEdit,
    PushButton,
    RadioButtons,
)
from ngio import open_ome_zarr_container, open_ome_zarr_plate
from ngio.common import Roi, compute_masking_roi
from ngio.ome_zarr_meta import PixelSize
from ngio.tables import MaskingRoiTable, RoiTable, open_tables_container
from ngio.utils import StoreOrGroup, fractal_fsspec_store
from qtpy.QtCore import QTimer  # type: ignore[attr-defined]

from napari_ome_zarr_navigator.util import (
    ZarrSelector,
    calculate_well_positions,
)

logger = logging.getLogger(__name__)

# Maps UI display names to the string identifiers accepted by ngio's add_table()
_BACKEND_MAP: dict[str, str] = {
    "CSV": "csv",
    "JSON": "json",
    "Parquet": "parquet",
    "AnnData": "anndata",
}

_MODE_EMPTY = "Initialize empty ROI layer"
_MODE_MASK = "Masking ROI layer"


class ROIAnnotator(Container):
    def __init__(
        self,
        viewer: napari.Viewer,
        extra_widgets=None,
    ):
        self._viewer = viewer
        self.store: StoreOrGroup | None = None
        self.is_local: bool = True
        self.translation: tuple[float, float] = (0.0, 0.0)
        self.layer_base_name: str = ""
        self.image_extent: tuple[float, float] | None = None  # (y_size, x_size) in µm
        self._post_save_callbacks: list = []
        self._suppress_layer_refresh: bool = False
        self._image_is_3d: bool = False
        self._image_is_time_series: bool = False
        self._shapes_ndim: int = 2
        self._shapes_z_scale: float = 1.0  # µm per z-voxel, used for 3D shapes layer
        self._shapes_t_scale: float = (
            1.0  # seconds per frame, used for time-resolved shapes layer
        )

        self._mode_selector = RadioButtons(
            label="Mode",
            choices=[_MODE_EMPTY, _MODE_MASK],
            value=_MODE_EMPTY,  # type: ignore[call-arg]
            orientation="vertical",
        )
        self._init_layer_btn = PushButton(text="Initialize ROI Layer")

        self._label_layer_picker = ComboBox(
            label="Label layer",
            choices=["(no label layers)"],
            value="(no label layers)",
        )
        self._label_layer_picker.hide()
        self._label_layer_picker.changed.connect(self._on_label_layer_selected)

        self._save_section_label = Label(value="─── Save ROI Table ───")

        self._save_type_picker = ComboBox(
            label="Save as",
            choices=["Masking ROI Table", "ROI Table"],
            value="Masking ROI Table",
            tooltip=(
                "Masking ROI Table links to a label image in the OME-Zarr.\n"
                "Use ROI Table if the label has not been saved back to the OME-Zarr."
            ),
        )
        self._save_type_picker.hide()
        self._save_type_picker.changed.connect(self._on_save_type_changed)

        self._reference_label_picker = ComboBox(
            label="Reference label",
            choices=["(no label images)"],
            value="(no label images)",
            tooltip=(
                "Label image in the OME-Zarr that this masking ROI table refers to.\n"
                "Must match the label used to compute the ROIs."
            ),
        )
        self._reference_label_picker.hide()

        self._table_name = LineEdit(label="Table name", value="interactive_ROI_table")
        self._backend_picker = ComboBox(
            label="Backend",
            choices=list(_BACKEND_MAP.keys()),
            value="CSV",
        )
        self._backend_picker.tooltip = "ngio table backend to store the ROI table in"
        self._overwrite_box = CheckBox(value=False, text="Overwrite existing table")
        self._remote_save_folder = FileEdit(
            label="Save to folder",
            mode="d",  # type: ignore[arg-type]
            tooltip=(
                "The plugin cannot write back to remote OME-Zarr stores. "
                "Choose a local folder to save the ROI table instead."
            ),
        )
        self._remote_save_folder.hide()
        self._save_btn = PushButton(text="Save ROI Table", enabled=False)
        self._save_btn.tooltip = "Select a Zarr image to enable saving"

        self._mode_selector.changed.connect(self._on_mode_changed)
        self._init_layer_btn.clicked.connect(self._on_main_btn_clicked)
        self._save_btn.clicked.connect(self.save_roi_table)
        self._remote_save_folder.changed.connect(self._update_save_btn_state)
        self._viewer.layers.events.inserted.connect(self._on_viewer_layers_changed)
        self._viewer.layers.events.removed.connect(self._on_viewer_layers_changed)

        widgets = [
            self._mode_selector,
            self._label_layer_picker,
            self._init_layer_btn,
            self._save_section_label,
            self._save_type_picker,
            self._reference_label_picker,
            self._table_name,
            self._backend_picker,
            self._overwrite_box,
            self._remote_save_folder,
            self._save_btn,
        ]
        if extra_widgets:
            widgets = extra_widgets + widgets

        super().__init__(widgets=widgets)

    def _on_mode_changed(self, value: str):
        if value == _MODE_MASK:
            self._init_layer_btn.text = "Calculate masking ROI table"
            self._label_layer_picker.show()
            self._save_type_picker.show()
            self._refresh_label_layer_picker()
            self._refresh_reference_label_picker()
            if self._save_type_picker.value == "Masking ROI Table":
                self._reference_label_picker.show()
                self._table_name.value = self._default_mask_table_name()
            else:
                self._reference_label_picker.hide()
                self._table_name.value = "interactive_ROI_table"
        else:
            self._init_layer_btn.text = "Initialize ROI Layer"
            self._label_layer_picker.hide()
            self._save_type_picker.hide()
            self._reference_label_picker.hide()
            self._table_name.value = "interactive_ROI_table"
        self._update_save_btn_state()

    def _on_main_btn_clicked(self):
        if self._mode_selector.value == _MODE_MASK:
            self.calculate_masking_roi_table()
        else:
            self.initialize_roi_layer()

    def _on_viewer_layers_changed(self, _=None):
        if self._mode_selector.value == _MODE_MASK and not self._suppress_layer_refresh:
            # Defer one Qt cycle so the layer list is fully updated before we scan.
            QTimer.singleShot(0, self._refresh_label_layer_picker)

    def _refresh_label_layer_picker(self):
        if self._suppress_layer_refresh:
            return
        current = self._label_layer_picker.value
        layers = [
            layer.name
            for layer in self._viewer.layers
            if isinstance(layer, napari.layers.Labels)
        ]
        new_choices = layers if layers else ["(no label layers)"]
        self._label_layer_picker.choices = new_choices
        if current in layers:
            self._label_layer_picker.value = current
        elif layers:
            self._label_layer_picker.value = layers[0]

    def _on_label_layer_selected(self, _=None):
        if self._mode_selector.value != _MODE_MASK:
            return
        if self._save_type_picker.value == "Masking ROI Table":
            self._table_name.value = self._default_mask_table_name()
            self._refresh_reference_label_picker()

    def _default_mask_table_name(self) -> str:
        name = self._label_layer_picker.value or "label"
        if name == "(no label layers)":
            name = "label"
        name = name.replace(" ", "")
        return f"{name}_masking_ROI_table"

    def _on_save_type_changed(self, value: str):
        if value == "Masking ROI Table":
            self._reference_label_picker.show()
            self._table_name.value = self._default_mask_table_name()
        else:
            self._reference_label_picker.hide()
            self._table_name.value = "interactive_ROI_table"

    def _refresh_reference_label_picker(self):
        """Repopulate reference label choices from the OME-Zarr container."""
        if self.store is None:
            self._reference_label_picker.choices = ["(no label images)"]
            return
        try:
            labels = open_ome_zarr_container(
                self.store,
                mode="r",
                cache=True,  # type: ignore[arg-type]
            ).list_labels()
        except Exception:  # noqa: BLE001
            labels = []
        choices = labels if labels else ["(no label images)"]
        self._reference_label_picker.choices = choices
        if labels:
            guess = self._guess_reference_label(self._label_layer_picker.value, labels)
            self._reference_label_picker.value = guess or labels[0]

    def _guess_reference_label(
        self, layer_name: str, available: list[str]
    ) -> str | None:
        """Match a viewer label layer name to an OME-Zarr label image name.

        Tries progressively shorter suffixes (splitting on '_') so that a layer
        named 'B03_nuclei' matches the 'nuclei' label in the OME-Zarr.
        """
        if layer_name in available:
            return layer_name
        parts = layer_name.split("_")
        # Try "nuclei", then "01_nuclei", then "B03_01_nuclei" (shortest suffix first)
        for i in range(len(parts) - 1, 0, -1):
            suffix = "_".join(parts[i:])
            if suffix in available:
                return suffix
        return None

    def _release_calculate_suppress(self):
        self._suppress_layer_refresh = False
        if self._mode_selector.value == _MODE_MASK:
            self._refresh_label_layer_picker()

    def calculate_masking_roi_table(self):
        self._suppress_layer_refresh = True
        try:
            self._calculate_masking_roi_table_inner()
        finally:
            QTimer.singleShot(0, self._release_calculate_suppress)

    def _calculate_masking_roi_table_inner(self):
        layer_name = self._label_layer_picker.value
        if not layer_name or layer_name == "(no label layers)":
            logger.warning("No label layer selected.")
            return

        label_layers = [
            layer
            for layer in self._viewer.layers
            if isinstance(layer, napari.layers.Labels) and layer.name == layer_name
        ]
        if not label_layers:
            logger.warning("Label layer '%s' not found in viewer.", layer_name)
            return
        label_layer = label_layers[0]

        # Load highest res label layer in multiscale labels case
        data = (
            label_layer.data[0]
            if isinstance(label_layer.data, list)
            else label_layer.data
        )

        ndim = data.ndim
        is_3d = self._image_is_3d
        is_time_series = self._image_is_time_series

        # Validate dimensionality against stored metadata:
        #   yx=2, zyx=3, tyx=3, singleton-z-yx=3 (e.g. (1,64,64)), tzyx=4
        valid = (
            (ndim == 2)
            or (ndim == 3 and is_3d and not is_time_series)
            or (ndim == 3 and not is_3d)  # tyx or singleton-z yx
            or (ndim == 4 and is_3d and is_time_series)
        )
        if not valid:
            logger.warning("Label layer has unsupported dimensionality (%dD).", ndim)
            return

        # Pixel size for the spatial (non-time) compute, read from the label layer's
        # actual napari scale so it reflects the loaded resolution.
        if is_3d:
            pixel_size = PixelSize(
                z=float(label_layer.scale[-3]),
                y=float(label_layer.scale[-2]),
                x=float(label_layer.scale[-1]),
            )
            spatial_axes: list[str] = ["z", "y", "x"]
        else:
            pixel_size = PixelSize(
                x=float(label_layer.scale[-1]),
                y=float(label_layer.scale[-2]),
                z=1.0,
            )
            spatial_axes = ["y", "x"]

        # Convert from label-local frame to image-local frame.
        # label.translate and self.translation are both in world µm.
        offset_y = float(label_layer.translate[-2]) - self.translation[0]
        offset_x = float(label_layer.translate[-1]) - self.translation[1]

        # Keep z scale in sync with the label layer actually being used.
        self._shapes_z_scale = float(pixel_size.z)
        self.initialize_roi_layer()
        shapes_layer = next(
            (
                layer
                for layer in self._viewer.layers
                if isinstance(layer, napari.layers.Shapes)
                and layer.name == self._shapes_layer_name
            ),
            None,
        )
        if shapes_layer is None:
            return

        if is_time_series:
            n_t = data.shape[0]
        elif ndim == 3 and not is_3d:
            data = data[0]  # strip singleton leading axis (e.g. (1, Y, X) → (Y, X))
            n_t = 1
        else:
            n_t = 1

        rectangles = []
        label_ints = []
        z_starts: list[float] = []
        z_lengths: list[float] = []
        t_starts: list[float] = []
        t_lengths: list[float] = []

        for t_idx in range(n_t):
            frame_data = np.array(data[t_idx]) if is_time_series else np.array(data)
            try:
                frame_rois = compute_masking_roi(frame_data, pixel_size, spatial_axes)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to compute masking ROIs (t=%d): %s", t_idx, exc)
                continue

            for i, roi in enumerate(frame_rois):
                y_start = roi["y"].start
                y_length = roi["y"].length
                x_start = roi["x"].start
                x_length = roi["x"].length
                if (
                    y_start is None
                    or y_length is None
                    or x_start is None
                    or x_length is None
                ):
                    continue
                y_s = y_start + offset_y
                y_e = y_s + y_length
                x_s = x_start + offset_x
                x_e = x_s + x_length

                if is_3d:
                    z_start_w = roi["z"].start or 0.0
                    z_len_w = roi["z"].length or pixel_size.z
                    # Place at z centroid voxel; true extent stored in properties
                    z_voxel = (z_start_w + z_len_w / 2.0) / pixel_size.z
                    z_starts.append(z_start_w)
                    z_lengths.append(z_len_w)

                if is_time_series:
                    t_starts.append(t_idx * self._shapes_t_scale)
                    t_lengths.append(self._shapes_t_scale)

                # Build shape coordinate array for the appropriate ndim
                if is_time_series and is_3d:
                    rect = np.array(
                        [
                            [t_idx, z_voxel, y_s, x_s],
                            [t_idx, z_voxel, y_s, x_e],
                            [t_idx, z_voxel, y_e, x_e],
                            [t_idx, z_voxel, y_e, x_s],
                        ]
                    )
                elif is_time_series:
                    rect = np.array(
                        [
                            [t_idx, y_s, x_s],
                            [t_idx, y_s, x_e],
                            [t_idx, y_e, x_e],
                            [t_idx, y_e, x_s],
                        ]
                    )
                elif is_3d:
                    rect = np.array(
                        [
                            [z_voxel, y_s, x_s],
                            [z_voxel, y_s, x_e],
                            [z_voxel, y_e, x_e],
                            [z_voxel, y_e, x_s],
                        ]
                    )
                else:
                    rect = np.array([[y_s, x_s], [y_s, x_e], [y_e, x_e], [y_e, x_s]])

                rectangles.append(rect)
                label_ints.append(roi.label if roi.label is not None else i)

        if rectangles:
            shapes_layer.add_rectangles(rectangles)
            props: dict[str, np.ndarray] = {
                "label_int": np.array(label_ints, dtype=int),
            }
            if is_3d:
                props["z_start"] = np.array(z_starts, dtype=float)
                props["z_length"] = np.array(z_lengths, dtype=float)
            if is_time_series:
                props["t_start"] = np.array(t_starts, dtype=float)
                props["t_length"] = np.array(t_lengths, dtype=float)
            shapes_layer.properties = props  # type: ignore[assignment]

        total = len(rectangles)
        if total:
            if is_time_series:
                logger.info(
                    "Found %d masking ROIs across %d time point(s) in label layer '%s'.",
                    total,
                    n_t,
                    layer_name,
                )
            else:
                logger.info(
                    "Found %d masking ROIs in label layer '%s'.", total, layer_name
                )
        else:
            logger.warning("No objects found in label layer '%s'.", layer_name)

    @property
    def _shapes_layer_name(self) -> str:
        return f"{self.layer_base_name}ROIs"

    def _update_save_btn_state(self):
        if self.store is None:
            self._save_btn.enabled = False
            self._remote_save_folder.hide()
            self._save_btn.tooltip = "Select a Zarr image to enable saving"
        elif self.is_local:
            self._remote_save_folder.hide()
            self._save_btn.enabled = True
            self._save_btn.tooltip = ""
        else:
            self._remote_save_folder.show()
            folder = str(self._remote_save_folder.value)
            folder_chosen = bool(folder and folder != ".")
            self._save_btn.enabled = folder_chosen
            self._save_btn.tooltip = (
                ""
                if folder_chosen
                else "Choose a local folder above to save the ROI table"
            )

    def initialize_roi_layer(self):
        name = self._shapes_layer_name
        for layer in list(self._viewer.layers):
            if isinstance(layer, napari.layers.Shapes) and layer.name == name:
                self._viewer.layers.remove(layer)

        ty = float(self.translation[0])
        tx = float(self.translation[1])
        if self._image_is_time_series and self._image_is_3d:
            ndim = 4  # tzyx
            translate = (0.0, 0.0, ty, tx)
            scale = (self._shapes_t_scale, self._shapes_z_scale, 1.0, 1.0)
        elif self._image_is_time_series:
            ndim = 3  # tyx
            translate = (0.0, ty, tx)
            scale = (self._shapes_t_scale, 1.0, 1.0)
        elif self._image_is_3d:
            ndim = 3  # zyx
            translate = (0.0, ty, tx)
            scale = (self._shapes_z_scale, 1.0, 1.0)
        else:
            ndim = 2  # yx
            translate = (ty, tx)
            scale = None
        self._shapes_ndim = ndim

        kw: dict = {
            "name": name,
            "ndim": ndim,
            "translate": translate,
            "edge_color": "red",
            "face_color": "transparent",
            "edge_width": 3,
        }
        if scale is not None:
            kw["scale"] = scale

        shapes_layer = self._viewer.add_shapes(**kw)
        self._viewer.layers.selection.active = shapes_layer
        shapes_layer.mode = "add_rectangle"

    def _extract_shape_coords(
        self,
        index: int,
        shape_data,
        shapes_layer: napari.layers.Shapes,
        z_starts_prop,
        z_lengths_prop,
        t_starts_prop,
        t_lengths_prop,
    ) -> (
        tuple[float, float, float, float, float, float, float | None, float | None]
        | None
    ):
        """Extract clipped spatial coords + z/t extent for one shape.

        Returns (x_s, x_len, y_s, y_len, z_s, z_len, t_s, t_len) or None when
        the shape is degenerate after clipping to image_extent.
        """
        coords = np.array(shape_data)
        y_start_raw = float(np.min(coords[:, -2]))
        y_end_raw = float(np.max(coords[:, -2]))
        x_start_raw = float(np.min(coords[:, -1]))
        x_end_raw = float(np.max(coords[:, -1]))

        if self.image_extent is not None:
            y_start = max(0.0, y_start_raw)
            y_end = min(self.image_extent[0], y_end_raw)
            x_start = max(0.0, x_start_raw)
            x_end = min(self.image_extent[1], x_end_raw)
        else:
            y_start, y_end = y_start_raw, y_end_raw
            x_start, x_end = x_start_raw, x_end_raw

        y_len = y_end - y_start
        x_len = x_end - x_start
        if y_len <= 0 or x_len <= 0:
            return None

        # Z: prefer stored properties; fall back to vertex coord when 3D
        if z_starts_prop is not None and z_lengths_prop is not None:
            z_start = float(z_starts_prop[index])
            z_len = float(z_lengths_prop[index])
        elif self._image_is_3d:
            z_scale = (
                float(shapes_layer.scale[-3]) if len(shapes_layer.scale) >= 3 else 1.0
            )
            z_start = float(coords[0, -3]) * z_scale
            z_len = z_scale
        else:
            z_start, z_len = 0.0, 1.0

        # T: prefer stored properties; fall back to vertex coord when time-resolved
        if t_starts_prop is not None and t_lengths_prop is not None:
            t_start: float | None = float(t_starts_prop[index])
            t_len: float | None = float(t_lengths_prop[index])
        elif self._image_is_time_series:
            t_scale = (
                float(shapes_layer.scale[0]) if len(shapes_layer.scale) >= 1 else 1.0
            )
            t_start = float(coords[0, 0]) * t_scale
            t_len = t_scale
        else:
            t_start = None
            t_len = None

        return x_start, x_len, y_start, y_len, z_start, z_len, t_start, t_len

    def _shapes_to_rois(
        self, shapes_layer: napari.layers.Shapes
    ) -> tuple[list[Roi], int]:
        """Convert rectangle shapes to ngio Roi objects.

        shapes.data[i] contains 4 corner points in the layer's local coordinate
        frame (i.e. image-relative world coordinates in µm, because the shapes
        layer translate matches the image layer translate). Tuples passed to
        Roi.from_values are (start, length).

        Shapes are clipped to [0, image_extent] when image_extent is known.
        ROIs that become empty after clipping are skipped; the count is returned
        so callers can include it in a single consolidated log message.
        """
        z_starts_prop = shapes_layer.properties.get("z_start", None)
        z_lengths_prop = shapes_layer.properties.get("z_length", None)
        t_starts_prop = shapes_layer.properties.get("t_start", None)
        t_lengths_prop = shapes_layer.properties.get("t_length", None)

        rois = []
        skipped = 0
        for i, (shape_data, shape_type) in enumerate(
            zip(shapes_layer.data, shapes_layer.shape_type, strict=False)
        ):
            if shape_type != "rectangle":
                skipped += 1
                continue

            result = self._extract_shape_coords(
                i,
                shape_data,
                shapes_layer,
                z_starts_prop,
                z_lengths_prop,
                t_starts_prop,
                t_lengths_prop,
            )
            if result is None:
                skipped += 1
                continue

            x_start, x_len, y_start, y_len, z_start, z_len, t_start, t_len = result
            slices: dict = {
                "x": (x_start, x_len),
                "y": (y_start, y_len),
                "z": (z_start, z_len),
            }
            if t_start is not None:
                slices["t"] = (t_start, t_len)

            roi = Roi.from_values(
                slices=slices,
                name=f"roi_{len(rois)}",
                space="world",
            )
            rois.append(roi)

        return rois, skipped

    def save_roi_table(self):
        if self._mode_selector.value == _MODE_MASK:
            self._save_shapes_masking_roi_table()
        else:
            self._save_shapes_roi_table()

    def _get_shapes_layer_for_save(self) -> "napari.layers.Shapes | None":
        """Shared pre-flight: validate store, find shapes layer, return it or None."""
        if self.store is None:
            logger.warning("No Zarr store set. Cannot save ROI table.")
            return None
        name = self._shapes_layer_name
        matching = [
            layer
            for layer in self._viewer.layers
            if isinstance(layer, napari.layers.Shapes) and layer.name == name
        ]
        if not matching:
            logger.warning(
                "No ROI layer '%s' found. Click 'Initialize ROI Layer' first.", name
            )
            return None
        shapes_layer = matching[0]
        if len(shapes_layer.data) == 0:
            logger.warning("No shapes in the ROI layer. Draw some rectangles first.")
            return None
        return shapes_layer

    def _do_save_table(self, table) -> bool:
        """Dispatch to local or remote save. Returns True on success."""
        return (
            self._do_save_local(table) if self.is_local else self._do_save_remote(table)
        )

    def _do_save_local(self, table) -> bool:
        """Write table into the local OME-Zarr container."""
        table_name = self._table_name.value.strip()
        if not table_name:
            logger.warning("Table name cannot be empty.")
            return False
        backend = _BACKEND_MAP[self._backend_picker.value]
        overwrite = self._overwrite_box.value
        # Check before opening in write mode to give a clear message rather than
        # a cryptic zarr read-only error.  cache=False on both opens avoids the
        # cached read-only container being returned for the append open.
        existing_tables = open_ome_zarr_container(
            self.store,  # type: ignore[arg-type]
            mode="r",
            cache=False,
        ).list_tables()
        if table_name in existing_tables and not overwrite:
            logger.warning(
                "Table '%s' already exists. "
                "Enable 'Overwrite existing table' to replace it.",
                table_name,
            )
            return False
        self._save_btn.enabled = False
        self._save_btn.text = "Saving..."
        try:
            container = open_ome_zarr_container(self.store, mode="a", cache=False)  # type: ignore[arg-type]
            container.add_table(
                name=table_name,
                table=table,
                backend=backend,
                overwrite=overwrite,
            )
            for cb in self._post_save_callbacks:
                cb()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to save table: %s", exc)
            return False
        finally:
            self._save_btn.text = "Save ROI Table"
            self._update_save_btn_state()

    def _do_save_remote(self, table) -> bool:
        """Write table to a user-chosen local folder (remote store is read-only)."""
        table_name = self._table_name.value.strip()
        if not table_name:
            logger.warning("Table name cannot be empty.")
            return False
        folder = str(self._remote_save_folder.value)
        if not folder or folder == ".":
            logger.warning("Select a local folder to save the ROI table.")
            return False
        backend = _BACKEND_MAP[self._backend_picker.value]
        dest = str(Path(folder) / "tables")
        self._save_btn.enabled = False
        self._save_btn.text = "Saving..."
        try:
            tc = open_tables_container(dest, mode="w")
            tc.add(table_name, table, backend=backend, overwrite=True)
            logger.info("Saved ROI table to '%s'.", dest)
            for cb in self._post_save_callbacks:
                cb()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to save table: %s", exc)
            return False
        finally:
            self._save_btn.text = "Save ROI Table"
            self._update_save_btn_state()

    def _save_shapes_roi_table(self, _fallback_note: str | None = None):
        shapes_layer = self._get_shapes_layer_for_save()
        if shapes_layer is None:
            return
        rois, skipped = self._shapes_to_rois(shapes_layer)
        if not rois:
            logger.warning("No valid rectangles found. Only rectangles are supported.")
            return
        table_name = self._table_name.value.strip()
        if not self._do_save_table(RoiTable(rois=rois)):
            return
        suffix = f" ({_fallback_note})" if _fallback_note else ""
        if skipped:
            logger.warning(
                "Saved %d ROI(s) to table '%s'; %d shape(s) skipped"
                " (not rectangles or out of bounds).%s",
                len(rois),
                table_name,
                skipped,
                suffix,
            )
        elif _fallback_note:
            logger.warning(
                "Saved %d ROI(s) to table '%s'%s.", len(rois), table_name, suffix
            )
        else:
            logger.info("Saved %d ROI(s) to table '%s'.", len(rois), table_name)

    def _save_shapes_masking_roi_table(self):
        shapes_layer = self._get_shapes_layer_for_save()
        if shapes_layer is None:
            logger.warning("No shapes layer found. Cannot save masking ROI table.")
            return

        # If save type was switched to "ROI Table", delegate
        if self._save_type_picker.value == "ROI Table":
            self._save_shapes_roi_table()
            return

        ref = self._reference_label_picker.value
        if ref == "(no label images)":
            self._save_shapes_roi_table(
                _fallback_note=(
                    "no label image found in the OME-Zarr, saved as ROI Table instead; "
                    "to save as Masking ROI Table first save the label to the OME-Zarr"
                )
            )
            return

        # Read per-shape properties stored when calculate_masking_roi_table ran.
        # napari keeps properties in sync with shape deletions.
        label_ints_prop = shapes_layer.properties.get("label_int", None)
        z_starts_prop = shapes_layer.properties.get("z_start", None)
        z_lengths_prop = shapes_layer.properties.get("z_length", None)
        t_starts_prop = shapes_layer.properties.get("t_start", None)
        t_lengths_prop = shapes_layer.properties.get("t_length", None)

        rois = []
        skipped = 0
        for i, (shape_data, shape_type) in enumerate(
            zip(shapes_layer.data, shapes_layer.shape_type, strict=False)
        ):
            if shape_type != "rectangle":
                skipped += 1
                continue

            result = self._extract_shape_coords(
                i,
                shape_data,
                shapes_layer,
                z_starts_prop,
                z_lengths_prop,
                t_starts_prop,
                t_lengths_prop,
            )
            if result is None:
                skipped += 1
                continue

            x_start, x_len, y_start, y_len, z_start, z_len, t_start, t_len = result
            slices: dict = {
                "x": (x_start, x_len),
                "y": (y_start, y_len),
                "z": (z_start, z_len),
            }
            if t_start is not None:
                slices["t"] = (t_start, t_len)

            label_int = int(label_ints_prop[i]) if label_ints_prop is not None else i
            roi = Roi.from_values(
                slices=slices,
                name=str(label_int),
                label=label_int,
                space="world",
            )
            rois.append(roi)

        if not rois:
            logger.warning("No valid rectangles found. Only rectangles are supported.")
            return

        table_name = self._table_name.value.strip()
        if not self._do_save_table(MaskingRoiTable(rois=rois, reference_label=ref)):
            return
        if skipped:
            logger.warning(
                "Saved %d masking ROI(s) to table '%s'; %d shape(s) skipped"
                " (not rectangles or out of bounds).",
                len(rois),
                table_name,
                skipped,
            )
        else:
            logger.info("Saved %d masking ROI(s) to table '%s'.", len(rois), table_name)


class ROIAnnotatorImage(ROIAnnotator):
    """Standalone ROI Annotator registered as a napari plugin widget."""

    def __init__(
        self,
        viewer: napari.Viewer,
        zarr_url: str | None = None,
        token: str | None = None,
        source: str = "File",
    ):
        self.zarr_selector = ZarrSelector()

        extra: list = [self.zarr_selector]
        if zarr_url:
            self._info_label = Label(value=f"Image: {Path(zarr_url).name}")
            self._info_label.tooltip = zarr_url
            extra = [self._info_label] + extra

        super().__init__(viewer=viewer, extra_widgets=extra)
        self.zarr_selector.on_change(self._on_selector_changed)

        if zarr_url:
            self.zarr_selector.configure(source=source, url=zarr_url, token=token)
            self.zarr_selector.hide()
            self._on_selector_changed()
        else:
            self._btn_launch_loader = PushButton(text="Load ROIs")
            self._btn_launch_loader.clicked.connect(self._launch_roi_loader)
            self.append(self._btn_launch_loader)

    def _launch_roi_loader(self):
        from napari_ome_zarr_navigator.roi_loader import ROILoaderImage

        loader = ROILoaderImage(
            viewer=self._viewer,
            zarr_url=self.zarr_selector.url,
            token=self.zarr_selector.token,
            source=self.zarr_selector.source,
        )
        self._viewer.window.add_dock_widget(
            widget=loader,
            name="ROI Loader",
            tabify=True,
            allowed_areas=["right"],
        )

    def _on_selector_changed(self):
        url = self.zarr_selector.url
        if url in ("", ".", None):
            self.store = None
            self.is_local = True
            self._update_save_btn_state()
            return

        if self.zarr_selector.source == "File":
            self.store = url
            self.is_local = True
        else:
            self.store = fractal_fsspec_store(
                url, fractal_token=self.zarr_selector.token
            )
            self.is_local = False

        try:
            container = open_ome_zarr_container(self.store, mode="r", cache=True)
            self._image_is_3d = container.is_3d
            self._image_is_time_series = container.is_time_series
            img = container.get_image(path=container.level_paths[0])
            axes = img.axes
            y_idx = axes.index("y")
            x_idx = axes.index("x")
            self.image_extent = (
                float(img.shape[y_idx] * img.pixel_size.y),
                float(img.shape[x_idx] * img.pixel_size.x),
            )
            self._shapes_z_scale = float(img.pixel_size.z) if self._image_is_3d else 1.0
            self._shapes_t_scale = (
                float(img.pixel_size.t) if self._image_is_time_series else 1.0
            )
        except Exception:  # noqa: BLE001
            self.image_extent = None
            self._image_is_3d = False
            self._image_is_time_series = False
            self._shapes_z_scale = 1.0
            self._shapes_t_scale = 1.0

        if self._mode_selector.value == _MODE_MASK:
            self._refresh_reference_label_picker()
        self._update_save_btn_state()


class ROIAnnotatorPlate(ROIAnnotator):
    """ROI Annotator launched from the Plate Browser for a specific well."""

    def __init__(
        self,
        viewer: napari.Viewer,
        plate_store: StoreOrGroup,
        row: str,
        col: str,
        plate_browser,
        is_plate: bool,
        plate_id: str = "",
        is_local: bool = True,
    ):
        self._zarr_picker = ComboBox(label="Image")
        self.plate_store = plate_store
        self.plate = open_ome_zarr_plate(store=self.plate_store, cache=True, mode="r")
        self.row = row
        self.col = col
        self.plate_id = plate_id

        plate_name = Path(str(plate_id)).name if plate_id else str(plate_store)
        self._info_label = Label(value=f"Well: {row}{col}  |  {plate_name}")
        self._info_label.tooltip = str(plate_id) if plate_id else str(plate_store)

        super().__init__(
            viewer=viewer, extra_widgets=[self._info_label, self._zarr_picker]
        )

        self.layer_base_name = f"{row}{col}_"
        self.is_local = is_local

        translation, bottom_right = calculate_well_positions(
            plate_store=plate_store, row=row, col=col, is_plate=is_plate
        )
        self.translation = (float(translation[0]), float(translation[1]))
        self.image_extent = (
            float(bottom_right[0] - translation[0]),
            float(bottom_right[1] - translation[1]),
        )

        self._zarr_picker.changed.connect(self._on_image_selected)
        zarr_images = sorted(self._get_available_images())
        self._zarr_picker.choices = zarr_images
        self._zarr_picker._default_choices = zarr_images
        if zarr_images:
            self._on_image_selected()

    def _get_available_images(self) -> list[str]:
        well = self.plate.get_well(row=self.row, column=self.col)
        return well.paths()

    def _on_image_selected(self):
        image_path = self._zarr_picker.value
        if image_path is None:
            self.store = None
            self._update_save_btn_state()
            return
        # Path string so zarr can open in any mode at write time.
        # plate.get_image_store() returns a read-only Group (plate is mode="r"),
        # which zarr refuses to reopen in append mode.
        self.store = f"{self.plate_id}/{self.row}/{self.col}/{image_path}"
        try:
            container = open_ome_zarr_container(
                self.store,  # type: ignore[arg-type]
                mode="r",
                cache=True,
            )
            self._image_is_3d = container.is_3d
            self._image_is_time_series = container.is_time_series
            if self._image_is_3d or self._image_is_time_series:
                img = container.get_image()
                self._shapes_z_scale = (
                    float(img.pixel_size.z) if self._image_is_3d else 1.0
                )
                self._shapes_t_scale = (
                    float(img.pixel_size.t) if self._image_is_time_series else 1.0
                )
            else:
                self._shapes_z_scale = 1.0
                self._shapes_t_scale = 1.0
        except Exception:  # noqa: BLE001
            self._image_is_3d = False
            self._image_is_time_series = False
            self._shapes_z_scale = 1.0
            self._shapes_t_scale = 1.0
        if self._mode_selector.value == _MODE_MASK:
            self._refresh_reference_label_picker()
        self._update_save_btn_state()
