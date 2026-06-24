import logging
from pathlib import Path

import napari
import ngio
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    Label,
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
    ZarrSelector,
    calculate_well_positions,
)

logger = logging.getLogger(__name__)

# Sentinel values shown in the UI when no ROI tables are found.
# Whole-image loading is not yet implemented; these keep the UI from crashing.
_NO_ROI_TABLE = "(none — load whole image)"
_NO_ROI_NAME = "(whole image)"

_MODE_MULTISCALE = "Multi-resolution (lazy)"
_MODE_FIXED = "Fixed resolution"


class ROILoader(Container):
    def __init__(
        self,
        viewer: napari.Viewer,
        extra_widgets=None,
    ):
        self._viewer = viewer
        # Used to identify the zarr image in the feature table
        self.zarr_id = ""

        # Translation to move position of ROIs loaded
        self.translation = (0, 0)
        self.layer_base_name = ""

        # Resolution maps: display string → ngio level path
        self._image_res_map: dict[str, str] = {}
        self._label_res_map: dict[str, str] = {}

        self._roi_table_picker = ComboBox(label="ROI Table")
        self._roi_picker = ComboBox(label="ROI")
        self._channel_picker = Select(
            label="Channels",
        )
        self._image_loading_mode = ComboBox(
            label="Image loading",
            choices=[_MODE_MULTISCALE, _MODE_FIXED],
            value=_MODE_MULTISCALE,
        )
        self._label_loading_mode = ComboBox(
            label="Label loading",
            choices=[_MODE_FIXED, _MODE_MULTISCALE],
            value=_MODE_FIXED,
        )
        self._label_picker = Select(
            label="Labels",
        )
        self._feature_picker = Select(
            label="Features",
        )

        # Advanced settings (collapsed by default)
        self._level_picker = ComboBox(label="Image resolution")
        self._level_picker.enabled = False  # disabled in default Multi-resolution mode
        self._label_level_picker = ComboBox(label="Label resolution")
        self._remove_old_labels_box = CheckBox(
            value=False, text="Remove existing labels"
        )
        self._advanced_toggle = PushButton(text="▶ Advanced settings")
        self._advanced_container = Container(
            widgets=[
                self._image_loading_mode,
                self._level_picker,
                self._label_loading_mode,
                self._label_level_picker,
                self._remove_old_labels_box,
            ],
        )
        self._advanced_container.visible = False

        self._run_button = PushButton(value=False, text="Load ROI")
        self._ome_zarr_container: ngio.OmeZarrContainer | None = None

        self.image_changed_event = ROILoaderSignals()

        self._btn_ctrl = LoaderButtonController(self._run_button)
        self._btn_ctrl.set_state(LoaderState.INITIALIZING)

        # Wire up events
        self.image_changed_event.image_changed.connect(self._start_initialization)
        self._roi_table_picker.changed.connect(self.update_roi_selection)
        self._image_loading_mode.changed.connect(self._on_image_mode_changed)
        self._label_loading_mode.changed.connect(self._on_label_mode_changed)
        self._advanced_toggle.clicked.connect(self._toggle_advanced)
        self._run_button.clicked.connect(self.run)

        widgets = [
            self._roi_table_picker,
            self._roi_picker,
            self._channel_picker,
            self._label_picker,
            self._feature_picker,
            self._advanced_toggle,
            self._advanced_container,
            self._run_button,
        ]
        if extra_widgets:
            widgets = extra_widgets + widgets

        super().__init__(widgets=widgets)

    # ------------------------------------------------------------------
    # Advanced settings toggle
    # ------------------------------------------------------------------

    def _toggle_advanced(self):
        vis = not self._advanced_container.visible
        self._advanced_container.visible = vis
        self._advanced_toggle.text = (
            "▼ Advanced settings" if vis else "▶ Advanced settings"
        )

    def _on_image_mode_changed(self):
        self._level_picker.enabled = self._image_loading_mode.value == _MODE_FIXED

    def _on_label_mode_changed(self):
        self._label_level_picker.enabled = self._label_loading_mode.value == _MODE_FIXED

    # ------------------------------------------------------------------
    # Container / state properties
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Initialization sequence
    # ------------------------------------------------------------------

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
            roi = self.ome_zarr_container.list_tables(
                filter_types="roi_table",
            )
            masking_roi = self.ome_zarr_container.list_tables(
                filter_types="masking_roi_table",
            )
            tables = roi + masking_roi
            return tables + [_NO_ROI_TABLE]
        else:
            return [_NO_ROI_TABLE]

    def _on_tables_ready(self, table_list):
        self._apply_roi_table_choices_update(table_list)
        self._btn_ctrl.on_step_done()  # step 1 done

        if self._roi_table_picker.value == _NO_ROI_TABLE:
            self._apply_roi_choices_update([_NO_ROI_NAME])
            self._btn_ctrl.on_step_done()  # step 2 done
        else:
            roi_worker = self._fetch_rois(self._roi_table_picker.value)  # type: ignore[call-arg]
            roi_worker.returned.connect(self._apply_roi_choices_update)
            roi_worker.returned.connect(self._btn_ctrl.on_step_done)  # step 2 done
            roi_worker.start()

    @thread_worker
    def _fetch_rois(self, table_name: str) -> list[str | None]:
        """Worker that returns the list of ROI names for the given table."""
        if self.ome_zarr_container is not None:
            ngio_table = self.ome_zarr_container.get_generic_roi_table(name=table_name)
            return [r.name for r in ngio_table.rois()]
        else:
            return [""]

    def update_roi_selection(self):
        """Called when the user picks a new ROI table."""
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
        self._roi_picker.choices = roi_list
        self._roi_picker._default_choices = roi_list
        self.image_changed_event.roi_choices_updated.emit(roi_list)

    def _apply_roi_table_choices_update(self, table_list):
        # Disconnect update_roi_selection while setting choices: changing choices
        # may change the picker's value, which would fire update_roi_selection
        # mid-init, calling begin_init(1) and resetting the step counter.  That
        # causes on_step_done() to declare READY before the _fetch_rois workers
        # launched by _on_tables_ready have finished, leaving dangling callbacks
        # that crash on already-deleted Qt widgets at teardown.
        self._roi_table_picker.changed.disconnect(self.update_roi_selection)
        try:
            self._roi_table_picker.choices = table_list
            self._roi_table_picker._default_choices = table_list
        finally:
            self._roi_table_picker.changed.connect(self.update_roi_selection)
        self.image_changed_event.roi_tables_updated.emit(table_list)

    def _on_state_change(self, new_state: LoaderState):
        self._btn_ctrl.set_state(new_state)
        if new_state is LoaderState.READY:
            self.image_changed_event.load_finished.emit()

    # ------------------------------------------------------------------
    # Image attributes & resolution maps
    # ------------------------------------------------------------------

    @staticmethod
    def _build_res_map(
        level_paths: list[str],
        get_fn,
    ) -> tuple[list[str], dict[str, str]]:
        """Build (res_strings, {res_str: level_path}) from ngio pixel_size metadata."""
        res_map: dict[str, str] = {}
        res_strings: list[str] = []
        for lv in level_paths:
            obj = get_fn(lv)
            ps = obj.pixel_size
            unit = ps.space_unit or "µm"
            s = f"{ps.x:.3g} {unit}"
            res_map[s] = lv
            res_strings.append(s)
        return res_strings, res_map

    def update_available_image_attrs(self, new_zarr_img: ngio.OmeZarrContainer | None):
        if new_zarr_img is not None:
            channels = new_zarr_img.channel_labels
            level_paths = new_zarr_img.level_paths  # ngio order: finest→coarsest
            try:
                labels = new_zarr_img.list_labels()
            except NgioValidationError:
                labels = []
            try:
                features = new_zarr_img.tables_container.list(
                    filter_types="feature_table"
                )
            except NgioValidationError:
                features = []

            # Build image resolution display strings
            image_res_strings, image_res_map = self._build_res_map(
                level_paths, lambda lv: new_zarr_img.get_image(path=lv)
            )

            # Build label resolution display strings from the first label
            if labels:
                first_lbl = labels[0]
                lbl_sample = new_zarr_img.get_label(name=first_lbl, path=level_paths[0])
                lbl_paths = lbl_sample.meta.paths  # finest→coarsest via ngio meta
                label_res_strings, label_res_map = self._build_res_map(
                    lbl_paths,
                    lambda lv: new_zarr_img.get_label(name=first_lbl, path=lv),
                )
            else:
                label_res_strings, label_res_map = image_res_strings, image_res_map

            self.set_available_image_attrs(
                channels,
                image_res_strings,
                image_res_map,
                label_res_strings,
                label_res_map,
                labels,
                features,
            )
        else:
            self.set_available_image_attrs([], [], {}, [], {}, [], [])

    def set_available_image_attrs(
        self,
        channels: list[str],
        image_res_strings: list[str],
        image_res_map: dict[str, str],
        label_res_strings: list[str],
        label_res_map: dict[str, str],
        labels: list[str],
        features: list[str],
    ):
        self._image_res_map = image_res_map
        self._label_res_map = label_res_map

        self._channel_picker.choices = channels
        self._channel_picker._default_choices = channels

        self._level_picker.choices = image_res_strings
        self._level_picker._default_choices = image_res_strings

        self._label_level_picker.choices = label_res_strings
        self._label_level_picker._default_choices = label_res_strings
        # Default to finest resolution (first in ngio's ordered list)
        if label_res_strings:
            self._label_level_picker.value = label_res_strings[0]

        self._label_picker.choices = labels
        self._label_picker._default_choices = labels

        self._feature_picker.choices = features
        self._feature_picker._default_choices = features

    def reset_widgets(self):
        """Clear out all dropdowns & go back to an uninitialized state."""
        for picker in (self._roi_table_picker, self._roi_picker):
            picker.choices = []
            picker._default_choices = []
        for picker in (
            self._channel_picker,
            self._level_picker,
            self._label_level_picker,
            self._label_picker,
            self._feature_picker,
        ):
            picker.choices = []
            picker._default_choices = []
        self._image_res_map = {}
        self._label_res_map = {}
        self.state = LoaderState.INITIALIZING

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    @thread_worker
    def _load_container(self, store):
        """Threaded open_ome_zarr_container."""
        try:
            return open_ome_zarr_container(store, mode="r", cache=True)
        except NgioFileNotFoundError as exc:
            self._on_image_container_error(exc)
            return None

    def _on_image_container_ready(self, container):
        self.ome_zarr_container = container

    def _on_image_container_error(self, exc: Exception):
        logger.warning(f"Error while loading image: {exc}")
        self.ome_zarr_container = None
        self.reset_widgets()

    def run(self):
        if not self.ome_zarr_container:
            return
        rt, rn = self._roi_table_picker.value, self._roi_picker.value

        whole_image = rt == _NO_ROI_TABLE
        channels = self._channel_picker.value
        labels = self._label_picker.value
        features = self._feature_picker.value
        translation = self.translation

        multiscale_image = self._image_loading_mode.value == _MODE_MULTISCALE
        multiscale_labels = self._label_loading_mode.value == _MODE_MULTISCALE

        # Resolve display strings to ngio level paths
        level_str = self._level_picker.value
        label_level_str = self._label_level_picker.value
        if level_str is None or label_level_str is None:
            logger.info("Image or label level not set; cannot load ROI")
            return
        level = self._image_res_map.get(level_str, level_str)
        label_level = self._label_res_map.get(label_level_str, label_level_str)

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
            level=level,  # type: ignore[arg-type]  # dict.get(str, str) → str
            channels=channels,
            labels=labels,
            features=features,
            translation=translation,
            set_state_fn=self._on_state_change,
            zarr_id=self.zarr_id,
            multiscale_image=multiscale_image,
            multiscale_labels=multiscale_labels,
            label_level=label_level,  # type: ignore[arg-type]
            whole_image=whole_image,
        )


class ROILoaderImage(ROILoader):
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

        super().__init__(
            viewer=viewer,
            extra_widgets=extra,
        )

        self.zarr_selector.on_change(self.update_image_selection)

        if zarr_url:
            self.zarr_selector.configure(source=source, url=zarr_url, token=token)
            self.zarr_selector.hide()
            self.update_image_selection()
        else:
            self._btn_launch_annotator = PushButton(text="Annotate ROIs interactively")
            self._btn_launch_annotator.clicked.connect(self._launch_roi_annotator)
            self.append(self._btn_launch_annotator)

    def _launch_roi_annotator(self):
        from napari_ome_zarr_navigator.roi_annotator import ROIAnnotatorImage

        annotator = ROIAnnotatorImage(
            viewer=self._viewer,
            zarr_url=self.zarr_selector.url,
            token=self.zarr_selector.token,
            source=self.zarr_selector.source,
        )
        self._viewer.window.add_dock_widget(
            widget=annotator,
            name="ROI Annotator",
            tabify=True,
            allowed_areas=["right"],
        )

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

        plate_name = Path(str(plate_id)).name if plate_id else str(plate_store)
        self._info_label = Label(value=f"Well: {row}{col}  |  {plate_name}")
        self._info_label.tooltip = str(plate_id) if plate_id else str(plate_store)

        super().__init__(
            viewer=viewer,
            extra_widgets=[self._info_label, self._zarr_picker],
        )
        self.layer_base_name = f"{row}{col}_{self.layer_base_name}"
        self._zarr_picker.changed.connect(self.update_image_selection)
        zarr_images = sorted(self.get_available_ome_zarr_images())
        self._zarr_picker.choices = zarr_images
        self._zarr_picker._default_choices = zarr_images

        self._run_button.clicked.connect(self._update_defaults)

        self.translation, _ = calculate_well_positions(
            plate_store=plate_store, row=row, col=col, is_plate=is_plate
        )

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
        level = self._image_res_map.get(
            self._level_picker.value, self._level_picker.value
        )
        label_level = self._label_res_map.get(
            self._label_level_picker.value, self._label_level_picker.value
        )
        self.plate_browser.update_defaults(
            zarr_image_subgroup=self._zarr_picker.value,
            roi_table=self._roi_table_picker.value,
            roi_name=self._roi_picker.value,
            channels=self._channel_picker.value,
            level=level,
            labels=self._label_picker.value,
            features=self._feature_picker.value,
            remove_old_labels=self._remove_old_labels_box.value,
            image_loading_mode=self._image_loading_mode.value,
            label_loading_mode=self._label_loading_mode.value,
            label_level=label_level,
        )
