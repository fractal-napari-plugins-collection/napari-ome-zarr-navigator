import logging
from pathlib import Path

import napari
import napari.layers
import numpy as np
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    Label,
    LineEdit,
    PushButton,
    RadioButtons,
)
from ngio import open_ome_zarr_container, open_ome_zarr_plate
from ngio.common import Roi, compute_masking_roi
from ngio.ome_zarr_meta import PixelSize
from ngio.tables import MaskingRoiTable, RoiTable
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

        self._table_name = LineEdit(label="Table name", value="interactive_ROIs")
        self._backend_picker = ComboBox(
            label="Backend",
            choices=list(_BACKEND_MAP.keys()),
            value="CSV",
        )
        self._backend_picker.tooltip = "ngio table backend to store the ROI table in"
        self._overwrite_box = CheckBox(value=False, text="Overwrite existing table")
        self._save_btn = PushButton(text="Save ROI Table", enabled=False)
        # FIXME: Update tooltip when adding remote Zarr support
        self._save_btn.tooltip = "Select a local Zarr image to enable saving"

        self._mode_selector.changed.connect(self._on_mode_changed)
        self._init_layer_btn.clicked.connect(self._on_main_btn_clicked)
        self._save_btn.clicked.connect(self.save_roi_table)
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
                self._table_name.value = "interactive_ROIs"
        else:
            self._init_layer_btn.text = "Initialize ROI Layer"
            self._label_layer_picker.hide()
            self._save_type_picker.hide()
            self._reference_label_picker.hide()
            self._table_name.value = "interactive_ROIs"
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
        if self._save_type_picker.value == "Masking ROI Table":
            self._table_name.value = self._default_mask_table_name()
            self._refresh_reference_label_picker()

    def _default_mask_table_name(self) -> str:
        name = self._label_layer_picker.value or "label"
        if name == "(no label layers)":
            name = "label"
        return f"{name}_masking_ROI_table"

    def _on_save_type_changed(self, value: str):
        if value == "Masking ROI Table":
            self._reference_label_picker.show()
            self._table_name.value = self._default_mask_table_name()
        else:
            self._reference_label_picker.hide()
            self._table_name.value = "interactive_ROIs"

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
        if ndim == 2:
            pixel_size = PixelSize(
                x=float(label_layer.scale[-1]),
                y=float(label_layer.scale[-2]),
                z=1.0,
            )
            axes_order = ["y", "x"]
        elif ndim == 3:
            # FIXME: Handle time dimension: 3 dim could be tyx
            pixel_size = PixelSize(
                z=float(label_layer.scale[-3]),
                y=float(label_layer.scale[-2]),
                x=float(label_layer.scale[-1]),
            )
            axes_order = ["z", "y", "x"]
        else:
            # FIXME: Handle time dimension (tzyx)
            logger.warning("Label layer has unsupported dimensionality (%dD).", ndim)
            return

        try:
            rois = compute_masking_roi(np.array(data), pixel_size, axes_order)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to compute masking ROIs: %s", exc)
            return

        # Convert from label-local frame to image-local frame.
        # label.translate and self.translation are both in world µm.
        offset_y = float(label_layer.translate[-2]) - self.translation[0]
        offset_x = float(label_layer.translate[-1]) - self.translation[1]

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

        rectangles = []
        label_ints = []
        for i, roi in enumerate(rois):
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
            # TODO: Handle 3D dimensionality of shape layers
            rectangles.append(
                np.array([[y_s, x_s], [y_s, x_e], [y_e, x_e], [y_e, x_s]])
            )
            # Carry the integer label value so MaskingRoiTable gets correct indices
            label_ints.append(roi.label if roi.label is not None else i)

        if rectangles:
            shapes_layer.add_rectangles(rectangles)
            shapes_layer.properties = {"label_int": np.array(label_ints, dtype=int)}  # type: ignore[assignment]

        if rois:
            logger.info(
                "Found %d masking ROIs in label layer '%s'.", len(rois), layer_name
            )
        else:
            logger.warning("No objects found in label layer '%s'.", layer_name)

    @property
    def _shapes_layer_name(self) -> str:
        return f"{self.layer_base_name}ROIs"

    def _update_save_btn_state(self):
        enabled = bool(self.is_local and self.store is not None)
        self._save_btn.enabled = enabled
        if not self.is_local:
            # FIXME: Update tooltip when adding remote Zarr support
            self._save_btn.tooltip = (
                "Saving is only supported for local (file) Zarr stores. "
                "HTTP stores are read-only."
            )
        elif self.store is None:
            # FIXME: Update tooltip when adding remote Zarr support
            self._save_btn.tooltip = "Select a local Zarr image to enable saving"
        else:
            self._save_btn.tooltip = ""

    def initialize_roi_layer(self):
        name = self._shapes_layer_name
        for layer in list(self._viewer.layers):
            if isinstance(layer, napari.layers.Shapes) and layer.name == name:
                self._viewer.layers.remove(layer)

        translate = (float(self.translation[0]), float(self.translation[1]))
        shapes_layer = self._viewer.add_shapes(
            name=name,
            ndim=2,
            translate=translate,
            edge_color="red",
            face_color="transparent",
            edge_width=3,
        )

        self._viewer.layers.selection.active = shapes_layer
        shapes_layer.mode = "add_rectangle"

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
        rois = []
        skipped = 0
        for shape_data, shape_type in zip(
            shapes_layer.data, shapes_layer.shape_type, strict=False
        ):
            if shape_type != "rectangle":
                skipped += 1
                continue

            coords = np.array(shape_data)  # (4, 2): [[y0,x0], ...]
            y_start_raw = float(np.min(coords[:, 0]))
            y_end_raw = float(np.max(coords[:, 0]))
            x_start_raw = float(np.min(coords[:, 1]))
            x_end_raw = float(np.max(coords[:, 1]))

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
                skipped += 1
                continue

            roi = Roi.from_values(
                slices={
                    "x": (x_start, x_len),
                    "y": (y_start, y_len),
                    "z": (0.0, 1.0),
                },
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
        if not self.is_local:
            logger.warning(
                "Saving ROI tables is only supported for local (file) stores."
            )
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
        """Write table to zarr, fire post-save callbacks. Returns True on success."""
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

    def _save_shapes_roi_table(self):
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
        if skipped:
            logger.warning(
                "Saved %d ROI(s) to table '%s'; %d shape(s) skipped"
                " (not rectangles or out of bounds).",
                len(rois),
                table_name,
                skipped,
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
            logger.warning(
                "No label image found in the OME-Zarr. Saving as ROI Table instead. "
                "To save as Masking ROI Table, first save the label to the OME-Zarr."
            )
            self._save_shapes_roi_table()
            return

        # Read per-shape label integers stored when calculate_masking_roi_table ran.
        # napari keeps properties in sync with shape deletions.
        label_ints_prop = shapes_layer.properties.get("label_int", None)

        rois = []
        skipped = 0
        for i, (shape_data, shape_type) in enumerate(
            zip(shapes_layer.data, shapes_layer.shape_type, strict=False)
        ):
            if shape_type != "rectangle":
                skipped += 1
                continue

            coords = np.array(shape_data)
            y_start_raw = float(np.min(coords[:, 0]))
            y_end_raw = float(np.max(coords[:, 0]))
            x_start_raw = float(np.min(coords[:, 1]))
            x_end_raw = float(np.max(coords[:, 1]))

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
                skipped += 1
                continue

            label_int = int(label_ints_prop[i]) if label_ints_prop is not None else i
            roi = Roi.from_values(
                slices={"x": (x_start, x_len), "y": (y_start, y_len), "z": (0.0, 1.0)},
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
            img = container.get_image(path=container.level_paths[0])
            axes = img.axes
            y_idx = axes.index("y")
            x_idx = axes.index("x")
            self.image_extent = (
                float(img.shape[y_idx] * img.pixel_size.y),
                float(img.shape[x_idx] * img.pixel_size.x),
            )
        except Exception:  # noqa: BLE001
            self.image_extent = None

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
        if self._mode_selector.value == _MODE_MASK:
            self._refresh_reference_label_picker()
        self._update_save_btn_state()
