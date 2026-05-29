import logging

import napari
import ngio
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    PushButton,
    Select,
)
from napari.qt.threading import thread_worker
from ngio import open_ome_zarr_container, open_ome_zarr_plate
from ngio.utils import (
    NgioFileNotFoundError,
    NgioValidationError,
    StoreOrGroup,
    fractal_fsspec_store,
)

from napari_ome_zarr_navigator.roi_loading_utils import (
    ROILoaderSignals,
    orchestrate_load_roi,
    remove_existing_label_layers,
)
from napari_ome_zarr_navigator.util import (
    LoaderButtonController,
    LoaderState,
    NapariHandler,
    ZarrSelector,
    calculate_well_positions,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Sentinel values shown in the UI when no ROI tables are found.
# Whole-image loading is not yet implemented; these keep the UI from crashing.
_NO_ROI_TABLE = "(none — load whole image)"
_NO_ROI_NAME = "(whole image)"


class ROILoader(Container):
    def __init__(
        self,
        viewer: napari.Viewer,
        extra_widgets=None,
    ):
        self._viewer = viewer
        self.setup_logging()
        self.channel_dict = {}
        self.channel_names_dict = {}
        self.labels_dict = {}
        # Used to identify the zarr image in the feature table
        self.zarr_id = ""

        # Translation to move position of ROIs loaded
        self.translation = (0, 0)
        self.layer_base_name = ""

        self._roi_table_picker = ComboBox(label="ROI Table")
        self._roi_picker = ComboBox(label="ROI")
        self._channel_picker = Select(
            label="Channels",
        )
        self._level_picker = ComboBox(label="Image Level")
        self._label_picker = Select(
            label="Labels",
        )
        self._feature_picker = Select(
            label="Features",
        )
        self._remove_old_labels_box = CheckBox(
            value=False, text="Remove existing labels"
        )
        self._run_button = PushButton(value=False, text="Load ROI")
        self._ome_zarr_container: ngio.OmeZarrContainer | None = None

        self.image_changed_event = ROILoaderSignals()

        self._btn_ctrl = LoaderButtonController(self._run_button)
        self._btn_ctrl.set_state(LoaderState.INITIALIZING)

        # Initialize possible choices
        self.image_changed_event.image_changed.connect(self._start_initialization)
        self._roi_table_picker.changed.connect(self.update_roi_selection)
        self._run_button.clicked.connect(self.run)

        widgets = [
            self._roi_table_picker,
            self._roi_picker,
            self._level_picker,
            self._channel_picker,
            self._label_picker,
            self._feature_picker,
            self._remove_old_labels_box,
            self._run_button,
        ]
        if extra_widgets:
            widgets = extra_widgets + widgets

        super().__init__(widgets=widgets)

    @property
    def ome_zarr_container(self):
        return self._ome_zarr_container

    @ome_zarr_container.setter
    def ome_zarr_container(self, value: ngio.OmeZarrContainer | None) -> None:
        if self._ome_zarr_container != value:
            self._ome_zarr_container = value
            self.image_changed_event.image_changed.emit(self._ome_zarr_container)

    @property
    def state(self) -> LoaderState | None:
        return self._btn_ctrl.current_state

    @state.setter
    def state(self, new: LoaderState) -> None:
        self._btn_ctrl.set_state(new)

    def setup_logging(self):
        for handler in logger.root.handlers[:]:
            logging.root.removeHandler(handler)
        # Create a custom handler for napari
        napari_handler = NapariHandler()
        napari_handler.setLevel(logging.INFO)

        # Optionally, set a formatter for the handler
        # formatter = logging.Formatter(
        #     '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        # )
        # napari_handler.setFormatter(formatter)

        logger.addHandler(napari_handler)

    def _start_initialization(self, *_):
        # 3 steps: (1) ROI-tables list, (2) ROI-names (dependent on 1), (3) image-attrs
        self._btn_ctrl.begin_init(n_steps=3)

        # (1) Start ROI-tables lookup:
        @thread_worker
        def _get_tables():
            return self.get_roi_tables()

        tbl_worker = _get_tables()  # type: ignore[call-arg]
        tbl_worker.returned.connect(self._on_tables_ready)
        tbl_worker.start()

        # (3) Kick off image-attrs right away (synchronous), then mark done:
        self.update_available_image_attrs(self.ome_zarr_container)
        self._btn_ctrl.on_step_done()

    def get_roi_tables(self) -> list[str]:
        """
        List available ROI tables

        Returns the list of available ROI tables in the current OME-Zarr
        container. Regular ROI tables are listed first, followed by masking
        ROI tables. If the container is None, returns a list with an
        empty string.
        """
        if self.ome_zarr_container is not None:
            # Use custom listing of ROI tables to ensure masking ROI tables
            # come last in the list (to avoid )
            roi = self.ome_zarr_container.list_tables(
                filter_types="roi_table",
            )
            masking_roi = self.ome_zarr_container.list_tables(
                filter_types="masking_roi_table",
            )
            tables = roi + masking_roi
            return tables if tables else [_NO_ROI_TABLE]
        else:
            return [_NO_ROI_TABLE]

    def _on_tables_ready(self, table_list):
        self._apply_roi_table_choices_update(table_list)
        self._btn_ctrl.on_step_done()  # step 1 done

        if self._roi_table_picker.value == _NO_ROI_TABLE:
            # No ROI tables found — skip worker, use sentinel name
            self._apply_roi_choices_update([_NO_ROI_NAME])
            self._btn_ctrl.on_step_done()  # step 2 done
        else:
            roi_worker = self._fetch_rois(self._roi_table_picker.value)  # type: ignore[call-arg]
            roi_worker.returned.connect(self._apply_roi_choices_update)
            roi_worker.returned.connect(self._btn_ctrl.on_step_done)  # step 2 done
            roi_worker.start()

    @thread_worker
    def _fetch_rois(self, table_name: str) -> list[str | None]:
        """
        Worker that returns the list of ROI names for the given table.
        """
        if self.ome_zarr_container is not None:
            ngio_table = self.ome_zarr_container.get_generic_roi_table(name=table_name)
            return [r.name for r in ngio_table.rois()]
        else:
            return [""]

    def update_roi_selection(self):
        """
        Called when the user picks a new ROI table.
        Disables the Load button until the names have loaded.
        """
        if not self.ome_zarr_container:
            self._apply_roi_choices_update([_NO_ROI_NAME])
            self.state = LoaderState.INITIALIZING
            return

        if self._roi_table_picker.value == _NO_ROI_TABLE:
            self._apply_roi_choices_update([_NO_ROI_NAME])
            self.state = LoaderState.READY
            return

        self._btn_ctrl.begin_init(n_steps=1)
        worker = self._fetch_rois(self._roi_table_picker.value)  # type: ignore[call-arg]
        worker.returned.connect(self._apply_roi_choices_update)
        worker.returned.connect(self._btn_ctrl.on_step_done)
        worker.start()

    def _apply_roi_choices_update(self, roi_list):
        """
        Update the list of available ROIs in the dropdown
        """
        self._roi_picker.choices = roi_list
        self._roi_picker._default_choices = roi_list
        self.image_changed_event.roi_choices_updated.emit(roi_list)

    def _apply_roi_table_choices_update(self, table_list):
        """
        Update the list of available ROI tables in the dropdown menu
        """
        self._roi_table_picker.choices = table_list
        self._roi_table_picker._default_choices = table_list
        self.image_changed_event.roi_tables_updated.emit(table_list)

    def _on_state_change(self, new_state: LoaderState):
        self._btn_ctrl.set_state(new_state)
        if new_state is LoaderState.READY:
            self.image_changed_event.load_finished.emit()

    def update_available_image_attrs(self, new_zarr_img: ngio.OmeZarrContainer | None):
        if new_zarr_img is not None:
            channels = new_zarr_img.channel_labels
            levels = sorted(new_zarr_img.level_paths)
            try:
                labels = new_zarr_img.list_labels()
            except NgioValidationError:
                labels = []
            # ngio version now strictly only loads feature tables
            try:
                features = new_zarr_img.tables_container.list(
                    filter_types="feature_table"
                )
            except NgioValidationError:
                features = []
            self.set_available_image_attrs(channels, levels, labels, features)
        else:
            # If zarr image was set to None, reset all selectors
            self.set_available_image_attrs([""], [""], [""], [""])

    def set_available_image_attrs(self, channels, levels, labels, features):
        self._channel_picker.choices = channels
        self._channel_picker._default_choices = channels
        # Set pyramid levels
        self._level_picker.choices = levels
        self._level_picker._default_choices = levels

        # Initialize available label images
        self._label_picker.choices = labels
        self._label_picker._default_choices = labels

        # Initialize available features
        self._feature_picker.choices = features
        self._feature_picker._default_choices = features

    def reset_widgets(self):
        """Clear out all dropdowns & go back to an uninitialized state."""
        # ROI tables + names
        for picker in (self._roi_table_picker, self._roi_picker):
            picker.choices = []
            picker._default_choices = []
        # channels, levels, labels, features
        for picker in (
            self._channel_picker,
            self._level_picker,
            self._label_picker,
            self._feature_picker,
        ):
            picker.choices = []
            picker._default_choices = []
        # and disable the run button
        self.state = LoaderState.INITIALIZING

    @thread_worker
    def _load_container(self, store):
        """Threaded open_ome_zarr_container."""
        try:
            return open_ome_zarr_container(store, mode="r", cache=True)
        except NgioFileNotFoundError as exc:
            self._on_image_container_error(exc)
            return None

    def _on_image_container_ready(self, container):
        # on main thread, assign and kick off your init sequence
        self.ome_zarr_container = container

    def _on_image_container_error(self, exc: Exception):
        logger.warning(f"Error while loading image: {exc}")
        self.ome_zarr_container = None
        self.reset_widgets()

    def run(self):
        if not self.ome_zarr_container:
            return
        rt, rn = self._roi_table_picker.value, self._roi_picker.value

        if rt == _NO_ROI_TABLE:
            # TODO: implement whole-image loading path
            logger.info(
                "No ROI table found in this image. "
                "Whole-image loading is not yet implemented."
            )
            return
        channels = self._channel_picker.value
        level = self._level_picker.value
        labels = self._label_picker.value
        features = self._feature_picker.value
        translation = self.translation

        if len(channels) < 1 and len(labels) < 1:
            logger.info(
                "No channel or labels selected. "
                "Select the channels/labels you want to load"
            )
            return

        if len(labels) < 1 and len(features) > 0:
            logger.info(
                "No labels selected to attach features to. "
                "Select the labels you want to load."
            )
            return

        if self._remove_old_labels_box.value:
            remove_existing_label_layers(self._viewer)

        orchestrate_load_roi(
            self.ome_zarr_container,
            self._viewer,
            roi_table=rt,
            roi_name=rn,
            layer_base_name=self.layer_base_name,
            level=level,
            channels=channels,
            labels=labels,
            features=features,
            translation=translation,
            set_state_fn=self._on_state_change,
            zarr_id=self.zarr_id,
        )


class ROILoaderImage(ROILoader):
    def __init__(
        self,
        viewer: napari.Viewer,
        zarr_url: str | None = None,
        token: str | None = None,
    ):
        self.zarr_selector = ZarrSelector()

        super().__init__(
            viewer=viewer,
            extra_widgets=[self.zarr_selector],
        )

        self.zarr_selector.on_change(self.update_image_selection)

        # Set initial value if provided
        if zarr_url:
            if token:
                self.zarr_selector.set_url(zarr_url, token=token)
            else:
                self.zarr_selector.set_url(zarr_url)

    def update_image_selection(self):
        source = self.zarr_selector.source
        self.zarr_url = self.zarr_selector.url
        token = self.zarr_selector.token
        self.zarr_id = self.zarr_url

        if self.zarr_url in ("", ".", None):
            self._ome_zarr_container = None
            self.reset_widgets()
            return

        if source == "File":
            store = self.zarr_url
        else:
            store = fractal_fsspec_store(self.zarr_url, fractal_token=token)

        worker = self._load_container(store)  # type: ignore[call-arg]
        worker.returned.connect(self._on_image_container_ready)
        worker.start()


class ROILoaderPlate(ROILoader):
    def __init__(
        self,
        viewer: napari.Viewer,
        plate_store: StoreOrGroup,
        row: str,
        col: str,
        plate_browser,
        is_plate: bool,
        plate_id: str = "",
    ):
        self._zarr_picker = ComboBox(label="Image")
        self.plate_store = plate_store
        self.plate = open_ome_zarr_plate(store=self.plate_store, cache=True, mode="r")
        self.row = row
        self.col = col
        self.plate_browser = plate_browser
        self.plate_id = plate_id
        super().__init__(
            viewer=viewer,
            extra_widgets=[
                self._zarr_picker,
            ],
        )
        self.layer_base_name = f"{row}{col}_{self.layer_base_name}"
        self._zarr_picker.changed.connect(self.update_image_selection)
        zarr_images = sorted(self.get_available_ome_zarr_images())
        self._zarr_picker.choices = zarr_images
        self._zarr_picker._default_choices = zarr_images

        self._run_button.clicked.connect(self._update_defaults)

        # Calculate base translation for a given well
        self.translation, _ = calculate_well_positions(
            plate_store=plate_store, row=row, col=col, is_plate=is_plate
        )

        # # Handle defaults for plate loading
        # if "well_ROI_table" in self._roi_table_picker.choices:
        #     self._roi_table_picker.value = "well_ROI_table"

    def get_available_ome_zarr_images(self):
        well = self.plate.get_well(row=self.row, column=self.col)
        return well.paths()

    def update_image_selection(self):
        self.zarr_id = (
            f"{self.plate_id}/{self.row}/{self.col}/{self._zarr_picker.value}"
        )
        image_store = self.plate.get_image_store(
            row=self.row, column=self.col, image_path=self._zarr_picker.value
        )
        worker = self._load_container(image_store)  # type: ignore[call-arg]
        worker.returned.connect(self._on_image_container_ready)
        worker.start()

    def _update_defaults(self):
        "Updates Plate Browser defaults when ROIs are loaded"
        self.plate_browser.update_defaults(
            zarr_image_subgroup=self._zarr_picker.value,
            roi_table=self._roi_table_picker.value,
            roi_name=self._roi_picker.value,
            channels=self._channel_picker.value,
            level=self._level_picker.value,
            labels=self._label_picker.value,
            features=self._feature_picker.value,
            remove_old_labels=self._remove_old_labels_box.value,
        )
