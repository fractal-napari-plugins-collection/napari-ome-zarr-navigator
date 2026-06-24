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
from ngio.common import Roi
from ngio.tables import RoiTable
from ngio.utils import StoreOrGroup, fractal_fsspec_store

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
_MODE_MASK = "Masking ROI layer (not yet implemented)"


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

        self._mode_selector = RadioButtons(
            label="Mode",
            choices=[_MODE_EMPTY, _MODE_MASK],
            value=_MODE_EMPTY,  # type: ignore[call-arg]
            orientation="vertical",
        )
        self._init_layer_btn = PushButton(text="Initialize ROI Layer")

        self._save_section_label = Label(value="─── Save ROI Table ───")
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
        self._init_layer_btn.clicked.connect(self.initialize_roi_layer)
        self._save_btn.clicked.connect(self.save_roi_table)

        widgets = [
            self._mode_selector,
            self._init_layer_btn,
            self._save_section_label,
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
            logger.info("Masking ROI mode is not yet implemented.")
            self._mode_selector.value = _MODE_EMPTY

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

    def _shapes_to_rois(self, shapes_layer: napari.layers.Shapes) -> list[Roi]:
        """Convert rectangle shapes to ngio Roi objects.

        shapes.data[i] contains 4 corner points in the layer's local coordinate
        frame (i.e. image-relative world coordinates in µm, because the shapes
        layer translate matches the image layer translate). Tuples passed to
        Roi.from_values are (start, length).

        Shapes are clipped to [0, image_extent] when image_extent is known.
        ROIs that become empty after clipping are skipped with a warning.
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

        if skipped > 0:
            logger.warning(
                "Skipped %d shape(s) (non-rectangle or outside image bounds).",
                skipped,
            )
        return rois

    def save_roi_table(self):
        if self.store is None:
            logger.warning("No Zarr store set. Cannot save ROI table.")
            return
        if not self.is_local:
            logger.warning(
                "Saving ROI tables is only supported for local (file) stores."
            )
            return

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
            return

        shapes_layer = matching[0]
        if len(shapes_layer.data) == 0:
            logger.warning("No shapes in the ROI layer. Draw some rectangles first.")
            return

        rois = self._shapes_to_rois(shapes_layer)
        if not rois:
            logger.warning("No valid rectangles found. Only rectangles are supported.")
            return

        table_name = self._table_name.value.strip()
        if not table_name:
            logger.warning("Table name cannot be empty.")
            return

        backend = _BACKEND_MAP[self._backend_picker.value]
        overwrite = self._overwrite_box.value

        # Check for existing table before opening in write mode so we can give
        # a clear message rather than a cryptic zarr read-only error.
        # cache=False on both opens: the cached read-only container would
        # otherwise be returned for the append open too, causing a read-only error.
        existing_tables = open_ome_zarr_container(
            self.store, mode="r", cache=False
        ).list_tables()
        if table_name in existing_tables and not overwrite:
            logger.warning(
                "Table '%s' already exists. "
                "Enable 'Overwrite existing table' to replace it.",
                table_name,
            )
            return

        self._save_btn.enabled = False
        self._save_btn.text = "Saving..."
        try:
            container = open_ome_zarr_container(self.store, mode="a", cache=False)
            roi_table = RoiTable(rois=rois)
            container.add_table(
                name=table_name,
                table=roi_table,
                backend=backend,
                overwrite=overwrite,
            )
            logger.info("Saved %d ROI(s) to table '%s'.", len(rois), table_name)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to save ROI table: %s", exc)
        finally:
            self._save_btn.text = "Save ROI Table"
            self._update_save_btn_state()


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
        self._update_save_btn_state()
