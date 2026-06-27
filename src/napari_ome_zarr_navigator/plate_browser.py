import logging
from contextlib import suppress

import napari
import napari.layers
import numpy as np
from magicgui.widgets import (
    ComboBox,
    Container,
    FloatSpinBox,
    PushButton,
    Select,
)
from ngio import open_ome_zarr_container
from ngio.utils import fractal_fsspec_store

from napari_ome_zarr_navigator.condition_filter import ConditionTableFilter
from napari_ome_zarr_navigator.plate_manager import PlateManager
from napari_ome_zarr_navigator.roi_loader import (
    ROILoaderPlate,
    remove_existing_label_layers,
)
from napari_ome_zarr_navigator.roi_loading_utils import orchestrate_load_roi
from napari_ome_zarr_navigator.util import (
    LoaderButtonController,
    ZarrSelector,
    calculate_well_positions,
)
from napari_ome_zarr_navigator.well_utils import (
    WELL_LAYER_PATTERN,
    get_row_cols,
)

logger = logging.getLogger(__name__)


class PlateBrowser(Container):
    def __init__(self, viewer: napari.Viewer):
        self.viewer = viewer
        self._zarr_selector = ZarrSelector(label="Plate")
        self._plate_mgr = PlateManager(self._zarr_selector)
        self.well = Select(label="Wells", enabled=True, allow_multiple=False)
        self.select_well = PushButton(text="➡ Go to well", enabled=False)
        self.zoom_level = FloatSpinBox(value=0.25, min=0.01, step=0.01)
        self.btn_load_roi = PushButton(text="Select ROI to load", enabled=False)
        self.btn_load_default_roi = PushButton(
            text="Load selected ROI for additional well(s)",
            enabled=False,
            tooltip="Once you've loaded a ROI with the 'Select ROI to load' "
            "button, this allows you to load the same ROI for different wells.",
        )
        self.btn_annotate_roi = PushButton(
            text="Annotate ROIs",
            enabled=False,
            tooltip="Open the ROI Annotator to draw ROIs on the selected well.",
        )
        self.roi_annotator = None
        self.roi_annotator_widget = None
        self._condition_table_source = ComboBox(
            label="Filter by condition",
            choices=[
                "No",
                "Plate-based condition table",
                "Image-based condition table (slow)",
            ],
            value="No",
        )
        self.roi_loader = None
        self.roi_widget = None
        self.default_zarr_image_subgroup = None
        self.default_roi_table = None
        self.default_roi_name = None
        self.default_channels = None
        self.default_level = None
        self.default_labels = None
        self.default_features = None
        self.remove_old_labels = False
        self.default_image_loading_mode = "Multi-resolution (lazy)"
        self.default_label_loading_mode = "Fixed resolution"
        self.default_label_level = "0"
        self._cond_filter = ConditionTableFilter(self._plate_mgr)

        self._default_btn_ctrl = LoaderButtonController(
            self.btn_load_default_roi,
            ready_label="Load selected ROI for additional well(s)",
        )

        super().__init__(
            widgets=[
                self._zarr_selector,
                self.well,
                self._condition_table_source,
                self._cond_filter.filter_container,
                Container(
                    widgets=[self.zoom_level, self.select_well],
                    label="Zoom level",
                    layout="horizontal",
                ),
                self.btn_load_roi,
                self.btn_load_default_roi,
                self.btn_annotate_roi,
            ],
        )
        self.viewer.layers.selection.events.changed.connect(self.get_zarr_url)
        # initialize_filters already calls filter_df internally — only one
        # callback per source is needed to avoid redundant downstream filtering
        self._zarr_selector.on_change(self.initialize_filters)
        self.select_well.clicked.connect(self.go_to_well)
        self.btn_load_roi.clicked.connect(self.launch_load_roi)
        self.btn_load_default_roi.clicked.connect(self.load_default_roi)
        self.btn_annotate_roi.clicked.connect(self.launch_roi_annotator)
        self._condition_table_source.changed.connect(
            self.on_condition_table_source_changed
        )
        # initialize_filters calls filter_df — no need to connect filter_df
        # separately here
        self._cond_filter.condition_name_selector.changed.connect(
            self.initialize_filters
        )
        self._cond_filter.signals.wells_changed.connect(self._update_well_choices)
        self.viewer.layers.events.removed.connect(self.check_empty_layerlist)

    def initialize_filters(self):
        # Check if the zarr_url is empty & finish early
        if (
            self._zarr_selector.url is None
            or self._zarr_selector.url == "."
            or self._zarr_selector.url == ""
        ):
            self._plate_mgr.clear()
            self.select_well.enabled = False
            self.btn_load_roi.enabled = False
            self.btn_load_default_roi.enabled = False
            self.btn_annotate_roi.enabled = False
            self.well.choices = []
            self.well._default_choices = []
            return

        self._plate_mgr.open_zarr_plate()
        if self._plate_mgr.zarr_plate:
            source = self._condition_table_source.value
            if source == "Plate-based condition table":
                df = self._cond_filter.load_plate_condition_table(
                    self._plate_mgr.zarr_plate,
                    table_name=self._cond_filter.condition_name_selector.value,
                )
            elif source == "Image-based condition table (slow)":
                df = self._cond_filter.load_image_condition_table(
                    self._plate_mgr.zarr_plate,
                    table_name=self._cond_filter.condition_name_selector.value,
                )
            else:
                df = None
            if df is not None:
                self._cond_filter.setup_filters(df)
            else:
                self._cond_filter.reset()

            self.select_well.enabled = True
            self.btn_load_roi.enabled = True
            self.btn_annotate_roi.enabled = True
        else:
            self.select_well.enabled = False
            self.btn_load_roi.enabled = False
            self.btn_annotate_roi.enabled = False

    def _update_well_choices(self, wells: list) -> None:
        with suppress(Exception):
            self.well.choices = ()
        self.well.choices = wells
        self.well._default_choices = wells

    def check_empty_layerlist(self):
        if len(self.viewer.layers) == 0:
            self._zarr_selector.set_url("")
            self.select_well.enabled = False
            self.btn_load_roi.enabled = False
            self.btn_annotate_roi.enabled = False
            self.well.choices = []
            self.well._default_choices = []
            self._cond_filter.filter_container.clear()
            self._cond_filter.filter_container.visible = False

    def get_zarr_url(self):
        # Only auto-populate the URL from layers explicitly loaded by
        # napari-ome-zarr. All other sources (our ROI loader, raw arrays,
        # other plugins) are ignored to prevent overwriting the plate URL or
        # losing the HTTP token.
        active = self.viewer.layers.selection.active
        if active and active.as_layer_data_tuple()[-1] == "image":
            if active.source.reader_plugin != "napari-ome-zarr":
                return
            path = active.source.path
            if path and path != self._zarr_selector.url:
                self._zarr_selector.set_url(path)
            if "sample_path" in active.metadata:
                self._zarr_selector.set_url(active.metadata["sample_path"])

    def launch_load_roi(self):
        assert self._plate_mgr.plate_store is not None
        wells = get_row_cols(self.well.value)
        if len(wells) != 1:
            logger.info("Please select a single well.")
        else:
            if self.roi_widget:
                with suppress(RuntimeError):
                    self.viewer.window.remove_dock_widget(self.roi_widget)  # type: ignore[arg-type]

            is_local = self._zarr_selector._source_selector.value == "File"
            self.roi_loader = ROILoaderPlate(
                self.viewer,
                self._plate_mgr.plate_store,
                wells[0][0],
                wells[0][1],
                self,
                self._plate_mgr.is_plate,
                plate_id=self._zarr_selector.url,
                is_local=is_local,
            )
            self.roi_widget = self.viewer.window.add_dock_widget(
                widget=self.roi_loader,
                name="ROI Loader",
                tabify=True,
                allowed_areas=["right"],
            )
            self._wire_annotator_loader()

    def launch_roi_annotator(self):
        from napari_ome_zarr_navigator.roi_annotator import ROIAnnotatorPlate

        assert self._plate_mgr.plate_store is not None
        wells = get_row_cols(self.well.value)
        if len(wells) != 1:
            logger.info("Please select a single well.")
            return

        if self.roi_annotator_widget:
            with suppress(RuntimeError):
                self.viewer.window.remove_dock_widget(self.roi_annotator_widget)  # type: ignore[arg-type]

        is_local = self._zarr_selector._source_selector.value == "File"
        self.roi_annotator = ROIAnnotatorPlate(
            self.viewer,
            self._plate_mgr.plate_store,
            wells[0][0],
            wells[0][1],
            self,
            self._plate_mgr.is_plate,
            plate_id=self._zarr_selector.url,
            is_local=is_local,
        )
        self.roi_annotator_widget = self.viewer.window.add_dock_widget(
            widget=self.roi_annotator,
            name="ROI Annotator",
            tabify=True,
            allowed_areas=["right"],
        )
        self._wire_annotator_loader()

    def _wire_annotator_loader(self):
        """Connect annotator's post-save to loader's refresh (plate case only)."""
        if self.roi_annotator is None or self.roi_loader is None:
            return
        import weakref

        loader_ref = weakref.ref(self.roi_loader)

        def _auto_refresh():
            loader = loader_ref()
            if loader is not None:
                loader.refresh_roi_tables()

        self.roi_annotator._post_save_callbacks = [_auto_refresh]

    def load_default_roi(self):
        assert self._plate_mgr.plate_store is not None
        wells = get_row_cols(self.well.value)

        # Loop over all selected wells
        for well in wells:
            if self.remove_old_labels:
                remove_existing_label_layers(self.viewer)
            layer_base_name = f"{well[0]}{well[1]}_"
            # Calculate translations
            translation, _ = calculate_well_positions(
                plate_store=self._plate_mgr.plate_store,
                row=well[0],
                col=well[1],
            )
            # Create the Zarr object
            zarr_url = f"{str(self._zarr_selector.url)}/{well[0]}/{well[1]}/{self.default_zarr_image_subgroup}"
            if self._zarr_selector._source_selector.value == "File":
                store = zarr_url
            else:
                store = fractal_fsspec_store(
                    zarr_url, fractal_token=self._zarr_selector.token
                )
            ome_zarr_container = open_ome_zarr_container(store)
            assert (
                self.default_roi_table is not None
                and self.default_roi_name is not None
                and self.default_level is not None
                and self.default_channels is not None
                and self.default_labels is not None
                and self.default_features is not None
            ), "load_default_roi called before defaults were set"
            orchestrate_load_roi(
                ome_zarr_container=ome_zarr_container,
                viewer=self.viewer,
                roi_table=self.default_roi_table,
                roi_name=self.default_roi_name,
                layer_base_name=layer_base_name,
                level=self.default_level,  # type: ignore[arg-type]  # asserted non-None above
                channels=self.default_channels,
                labels=self.default_labels,
                features=self.default_features,
                translation=tuple(translation),  # type: ignore[arg-type]
                set_state_fn=self._default_btn_ctrl.set_state,
                zarr_id=zarr_url,
                multiscale_image=(
                    self.default_image_loading_mode == "Multi-resolution (lazy)"
                ),
                multiscale_labels=(
                    self.default_label_loading_mode == "Multi-resolution (lazy)"
                ),
                label_level=self.default_label_level,
            )

    def go_to_well(self):
        wells = get_row_cols(self.well.value)

        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Shapes) and WELL_LAYER_PATTERN.match(
                layer.name
            ):
                self.viewer.layers.remove(layer)

        if len(wells) > 0:
            rec = None
            for well in wells:
                (
                    top_left_corner,
                    bottom_right_corner,
                ) = calculate_well_positions(
                    plate_store=self._plate_mgr.plate_store,
                    row=well[0],
                    col=well[1],
                    is_plate=self._plate_mgr.is_plate,
                )
                rec = np.array([top_left_corner, bottom_right_corner])
                self.viewer.add_shapes(
                    rec,
                    shape_type="rectangle",
                    edge_width=5,
                    edge_color="white",
                    face_color="transparent",
                    name=f"{well[0]}{well[1]}",
                )

            if rec is not None:
                self.viewer.camera.center = rec.mean(axis=0)
            self.viewer.camera.zoom = self.zoom_level.value
        else:
            logger.info("Please select at least one well")

    def on_condition_table_source_changed(self):
        if not self._plate_mgr.zarr_plate:
            return
        self._cond_filter.clear_condition_tables()
        self._cond_filter.reset()
        source = self._condition_table_source.value
        if source == "Plate-based condition table":
            self._cond_filter.init_condition_tables(self._plate_mgr.zarr_plate)
        elif source == "Image-based condition table (slow)":
            self._cond_filter.init_image_condition_tables(self._plate_mgr.zarr_plate)

    def update_defaults(
        self,
        zarr_image_subgroup,
        roi_table,
        roi_name,
        channels,
        level,
        labels,
        features,
        remove_old_labels,
        image_loading_mode="Multi-resolution (lazy)",
        label_loading_mode="Fixed resolution",
        label_level="0",
    ):
        self.default_zarr_image_subgroup = zarr_image_subgroup
        self.default_roi_table = roi_table
        self.default_roi_name = roi_name
        self.default_channels = channels
        self.default_level = level
        self.default_labels = labels
        self.default_features = features
        self.remove_old_labels = remove_old_labels
        self.default_image_loading_mode = image_loading_mode
        self.default_label_loading_mode = label_loading_mode
        self.default_label_level = label_level
        self.btn_load_default_roi.enabled = True
