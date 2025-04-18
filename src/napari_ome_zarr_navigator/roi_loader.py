import logging
import re
from enum import Enum, auto

import napari
import napari.layers
import ngio
import numpy as np
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    PushButton,
    Select,
)
from napari.qt.threading import thread_worker
from napari.utils.colormaps import Colormap
from ngio import open_ome_zarr_container, open_ome_zarr_plate
from ngio.common import Roi
from ngio.utils import (
    NgioValidationError,
    StoreOrGroup,
    fractal_fsspec_store,
)
from qtpy.QtCore import QObject, QTimer, Signal

from napari_ome_zarr_navigator.util import (
    NapariHandler,
    ZarrSelector,
    calculate_well_positions,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LoaderState(Enum):
    INITIALIZING = auto()
    READY = auto()
    LOADING = auto()


class ROILoader(Container):
    def __init__(
        self,
        viewer: napari.viewer.Viewer,
        extra_widgets=None,
    ):
        self._viewer = viewer
        self.setup_logging()
        self.channel_dict = {}
        self.channel_names_dict = {}
        self.labels_dict = {}

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
        self._ome_zarr_container: ngio.OmeZarrContainer = None

        self.image_changed_event = ROILoaderSignals()

        # Add timers for state changes
        self._init_timer = QTimer(singleShot=True, interval=150)
        self._init_timer.timeout.connect(self._enter_initializing)

        # State of the load button
        self._state = None
        self.state = LoaderState.INITIALIZING

        # Initialize possible choices
        self.image_changed_event.image_changed.connect(
            self._start_initialization
        )
        self._roi_table_picker.changed.connect(self.update_roi_selection)
        self._run_button.clicked.connect(self.run)

        widgets = [
            self._roi_table_picker,
            self._roi_picker,
            self._channel_picker,
            self._level_picker,
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
    def ome_zarr_container(self, value) -> ngio.OmeZarrContainer:
        if self._ome_zarr_container != value:
            self._ome_zarr_container = value
            self.image_changed_event.image_changed.emit(
                self._ome_zarr_container
            )

    @property
    def state(self) -> LoaderState:
        return self._state

    @state.setter
    def state(self, new: LoaderState):
        self._state = new
        if new is LoaderState.INITIALIZING:
            self._run_button.enabled = False
            self._run_button.text = "Initializing"
        elif new is LoaderState.READY:
            self._run_button.enabled = True
            self._run_button.text = "Load ROI"
        elif new is LoaderState.LOADING:
            self._run_button.enabled = False
            self._run_button.text = "Loading"

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

    def _begin_init(self):
        """Generic disable + debounce for any initializing step."""
        self._run_button.enabled = False
        # restart the 150 ms timer
        if self._init_timer.isActive():
            self._init_timer.stop()
        self._init_timer.start()

    def _start_initialization(self, *_):
        # 1) disable & schedule “Initializing” after 150 ms
        self._begin_init()

        # 2) schedule that, after 150 ms, if we're still not done,
        #    switch text → “Initializing”
        if self._init_timer.isActive():
            self._init_timer.stop()
        self._init_timer.start()

        # We’ll have 3 “done” events:
        # 1) ROI‐tables list, 2) ROI‐names list (dependent on 1), 3) image‐attrs
        self._init_pending = 3

        # (1) Start ROI‐tables lookup:
        @thread_worker
        def _get_tables():
            if self.ome_zarr_container is not None:
                return self.ome_zarr_container.list_roi_tables()
            else:
                return [""]

        tbl_worker = _get_tables()
        tbl_worker.returned.connect(self._on_tables_ready)
        tbl_worker.start()

        # (3) Kick off image‐attrs right away (synchronous), then mark done:
        self.update_available_image_attrs(self.ome_zarr_container)
        self._on_init_step_done()

    def _on_tables_ready(self, table_list):
        # apply ROI‐tables dropdown
        self._apply_roi_table_choices_update(table_list)
        # (1) done:
        self._on_init_step_done()

        # (2) now that tables are populated, fetch ROI‐names:
        roi_worker = self._fetch_rois(self._roi_table_picker.value)
        roi_worker.returned.connect(self._apply_roi_choices_update)
        roi_worker.returned.connect(self._on_init_step_done)
        roi_worker.start()

    def _on_init_step_done(self, *_):
        self._init_pending -= 1
        if self._init_pending == 0:
            # cancel any pending button label‐change
            if self._init_timer.isActive():
                self._init_timer.stop()
            self.state = LoaderState.READY

    def _enter_initializing(self):
        """Only flip the text after a delay to avoid flickering"""
        if getattr(self, "_init_pending", 0) > 0:
            self.state = LoaderState.INITIALIZING

    @thread_worker
    def _fetch_rois(self, table_name: str) -> list[str]:
        """
        Worker that returns the list of ROI names for the given table.
        """
        if self.ome_zarr_container is not None:
            ngio_table = self.ome_zarr_container.get_table(
                name=table_name, check_type="generic_roi_table"
            )
            return [r.name for r in ngio_table.rois()]
        else:
            return [""]

    def update_roi_selection(self):
        """
        Called when the user picks a new ROI table.
        Disables the Load button until the names have loaded.
        """
        if not self.ome_zarr_container:
            self._apply_roi_choices_update([""])
            self.state = LoaderState.INITIALIZING
            return

        # 1) disable & debounce
        self._begin_init()

        # 2) fetch table names exactly like in _start_initialization
        worker = self._fetch_rois(self._roi_table_picker.value)
        worker.returned.connect(self._apply_roi_choices_update)

        # 3) when done, cancel timer + go ready
        def _on_rois_returned(_):
            if self._init_timer.isActive():
                self._init_timer.stop()
            self.state = LoaderState.READY

        worker.returned.connect(_on_rois_returned)
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

    def update_available_image_attrs(self, new_zarr_img):
        if new_zarr_img:
            channels = self.ome_zarr_container.image_meta.channel_labels
            levels = self.ome_zarr_container.levels_paths
            try:
                labels = self.ome_zarr_container.list_labels()
            except NgioValidationError:
                labels = []
            # ngio version now strictly only loads feature tables
            try:
                features = self.ome_zarr_container.tables_container.list(
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
        return open_ome_zarr_container(store, mode="r", cache=True)

    def _on_image_container_ready(self, container):
        # on main thread, assign and kick off your init sequence
        self.ome_zarr_container = container

    def _on_image_container_error(self, exc: Exception):
        logger.error(f"Error while loading image: {exc}")
        self.ome_zarr_container = None
        self.reset_widgets()

    def run(self):
        # TODO: Refactor to use thread worker
        # (but keep it callable from img_browser class)
        self.state = LoaderState.LOADING
        roi_table = self._roi_table_picker.value
        roi_name = self._roi_picker.value
        level = self._level_picker.value
        channels = self._channel_picker.value
        labels = self._label_picker.value
        features = self._feature_picker.value
        blending = None
        if len(channels) < 1 and len(labels) < 1:
            logger.info(
                "No channel or labels selected. "
                "Select the channels/labels you want to load"
            )
            return

        if self._remove_old_labels_box.value:
            remove_existing_label_layers(self._viewer)

        load_roi(
            ome_zarr_container=self.ome_zarr_container,
            viewer=self._viewer,
            roi_table=roi_table,
            roi_name=roi_name,
            layer_base_name=self.layer_base_name,
            channels=channels,
            level=level,
            labels=labels,
            features=features,
            translation=self.translation,
            blending=blending,
        )
        self.state = LoaderState.READY


class ROILoaderImage(ROILoader):
    def __init__(
        self,
        viewer: napari.viewer.Viewer,
        zarr_url: str = None,
        token: str = None,
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

        if self.zarr_url in ("", ".", None):
            self._ome_zarr_container = None
            self.reset_widgets()
            return

        if source == "File":
            store = self.zarr_url
        else:
            store = fractal_fsspec_store(self.zarr_url, fractal_token=token)

        worker = self._load_container(store)
        worker.returned.connect(self._on_image_container_ready)
        worker.errored.connect(self._on_image_container_error)
        worker.start()


class ROILoaderPlate(ROILoader):
    def __init__(
        self,
        viewer: napari.viewer.Viewer,
        plate_store: StoreOrGroup,
        row: str,
        col: str,
        image_browser,
        is_plate: bool,
    ):
        self._zarr_picker = ComboBox(label="Image")
        self.plate_store = plate_store
        self.plate = open_ome_zarr_plate(
            store=self.plate_store, cache=True, mode="r", parallel_safe=False
        )
        self.row = row
        self.col = col
        self.image_browser = image_browser
        super().__init__(
            viewer=viewer,
            extra_widgets=[
                self._zarr_picker,
            ],
        )
        self.layer_base_name = f"{row}{col}_{self.layer_base_name}"
        self._zarr_picker.changed.connect(self.update_image_selection)
        zarr_images = self.get_available_ome_zarr_images()
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
        image_store = self.plate.get_image_store(
            row=self.row, column=self.col, image_path=self._zarr_picker.value
        )
        worker = self._load_container(image_store)
        worker.returned.connect(self._on_image_container_ready)
        worker.errored.connect(self._on_image_container_error)
        worker.start()

    def _update_defaults(self):
        "Updates Image Browser default when ROIs are loaded"
        self.image_browser.update_defaults(
            zarr_image_subgroup=self._zarr_picker.value,
            roi_table=self._roi_table_picker.value,
            roi_name=self._roi_picker.value,
            channels=self._channel_picker.value,
            level=self._level_picker.value,
            labels=self._label_picker.value,
            features=self._feature_picker.value,
            remove_old_labels=self._remove_old_labels_box.value,
        )


class ROILoaderSignals(QObject):
    image_changed = Signal(object)
    roi_choices_updated = Signal(list)
    roi_tables_updated = Signal(list)

    def __init__(self):
        super().__init__()


def load_roi(
    ome_zarr_container: ngio.OmeZarrContainer,
    viewer,
    roi_table: str,
    roi_name: str,
    layer_base_name: str,
    channels: list = None,
    level: str = "0",
    labels: list = None,
    features: list = None,
    translation: tuple = (0, 0),
    blending: str = None,
    lazy: bool = False,
):
    """
    Load images, labels & tables of a given ROI & add to viewer

    Args:
        ome_zarr_container: OME-Zarr object to be loaded
        viewer: napari viewer object
        roi_table: Name of the ROI table to load a ROI from
            (e.g. "well_ROI_table")
        roi_name: Name of the ROI within the roi_table to load
        layer_base_name: Base name for the layers to be added
        channels: List of intensity channels to load
        level: Resolution level to load
        labels: List of labels to load
        translation: Translation to apply to all loaded ROIs (typically the
            translation to shift it to the correct well in a plate setting)
        blending: Blending for the first intensity image to be used
        lazy: Whether to use dask for lazy loading

    """
    # Get translation within the larger image based on the ROI table
    label_layers = {}

    ngio_roi_table = ome_zarr_container.get_table(
        roi_table, check_type="generic_roi_table"
    )
    curr_roi = ngio_roi_table.get(roi_name)

    roi_translation = (
        translation[0] + curr_roi.y,
        translation[1] + curr_roi.x,
    )

    # Set layer names
    if len(ngio_roi_table.rois()) == 1:
        layer_base_name = layer_base_name
    else:
        layer_base_name = f"{layer_base_name}{roi_name}_"

    # Load intensity images
    img_pixel_size = None
    if channels:
        for channel in channels:
            add_intensity_roi(
                ome_zarr_container,
                viewer,
                channel,
                roi=curr_roi,
                level=level,
                blending=blending,
                translate=roi_translation,
                layer_name=f"{layer_base_name}{channel}",
                lazy=lazy,
            )
            blending = "additive"

        img_pixel_size = ome_zarr_container.get_image(path=level).pixel_size
    # Load labels
    for label in labels:
        if img_pixel_size:
            ngio_label = ome_zarr_container.get_label(
                name=label,
                pixel_size=img_pixel_size,
                strict=False,
            )
        else:
            ngio_label = ome_zarr_container.get_label(name=label, path=level)
        if lazy:
            label_roi = ngio_label.get_roi(roi=curr_roi, mode="dask").squeeze()
        else:
            label_roi = np.squeeze(
                ngio_label.get_roi(roi=curr_roi, mode="numpy")
            )

        # FIXME: load only relevant pixel size in case image is 2D

        if len(label_roi.shape) == 3:
            z, y, x = ngio_label.pixel_size.zyx
            scale_label = (z, y, x)
        elif len(label_roi.shape) == 2:
            y, x = ngio_label.pixel_size.yx
            scale_label = (y, x)
        else:
            raise NotImplementedError(
                "ROI loading has not been implemented for ROIs of shape "
                f"{len(label_roi.shape)} yet."
            )

        layer_name = f"{layer_base_name}{label}"
        if layer_name in viewer.layers:
            logger.info(f"{layer_name} is already loaded")
            label_layers[label] = viewer.layers[layer_name]
        else:
            label_layers[label] = viewer.add_labels(
                label_roi,
                scale=scale_label,
                name=layer_name,
                translate=roi_translation,
            )

    # Load features
    for table_name in features:
        label_layer = find_matching_label_layer(
            ome_zarr_container, table_name, label_layers
        )
        add_feature_table_to_layer(
            ome_zarr_container,
            table_name,
            label_layer,
            roi_name,
        )


def add_intensity_roi(
    ome_zarr_container: ngio.OmeZarrContainer,
    viewer,
    channel: str,
    roi: Roi,
    level: str,
    blending: str,
    translate: tuple[float, float],
    layer_name: str = "",
    lazy: bool = False,
):
    channel_index = ome_zarr_container.image_meta.get_channel_idx(channel)
    ngio_img = ome_zarr_container.get_image(path=level)
    if lazy:
        # FIXME: Better handling of dropping channel dimension
        img_roi = ngio_img.get_roi(
            roi=roi, c=channel_index, mode="dask"
        ).squeeze()
    else:
        # FIXME: Better handling of dropping channel dimension
        img_roi = np.squeeze(
            ngio_img.get_roi(roi=roi, c=channel_index, mode="numpy")
        )

    if len(img_roi.shape) == 3:
        z, y, x = ngio_img.pixel_size.zyx
        scale_img = (z, y, x)
    elif len(img_roi.shape) == 2:
        y, x = ngio_img.pixel_size.yx
        scale_img = (y, x)
    else:
        raise NotImplementedError(
            "ROI loading has not been implemented for ROIs of shape "
            f"{len(img_roi.shape)} yet."
        )

    if not np.any(img_roi):
        return

    # Get channel omero metadata

    channel_visualisation = ome_zarr_container.image_meta.channels[
        channel_index
    ].channel_visualisation
    try:
        # Colormap creation needs to have this black initial color for
        # background
        colormap = Colormap(
            ["#000000", f"#{channel_visualisation.color}"],
            name=channel_visualisation.color,
        )
    except AttributeError:
        colormap = None
    try:
        rescaling = (
            channel_visualisation.start,
            channel_visualisation.end,
        )
    except AttributeError:
        rescaling = None

    if layer_name in viewer.layers:
        logger.info(f"{layer_name} is already loaded")
    else:
        viewer.add_image(
            img_roi,
            scale=scale_img,
            blending=blending,
            contrast_limits=rescaling,
            colormap=colormap,
            name=layer_name,
            translate=translate,
        )
    # TODO: Optionally return some values as well? e.g. if info is needed
    # by label loading


def find_matching_label_layer(
    ome_zarr_container: ngio.OmeZarrContainer,
    table_name: str,
    label_layers: list,
):
    """
    Finds the matching label layer for a feature table
    """
    ngio_table = ome_zarr_container.get_table(
        table_name, check_type="feature_table"
    )
    target_label_name = ngio_table._meta.region.path.split("/")[-1]

    if target_label_name not in label_layers:
        target_label_name = list(label_layers.keys())[0]
        logger.info(
            f"The label {target_label_name} that {table_name} would be "
            "matched to where not loaded. Attaching the features to the"
            f"first selected label layer ({target_label_name})"
        )

    return label_layers[target_label_name]


def add_feature_table_to_layer(
    ome_zarr_container: ngio.OmeZarrContainer,
    feature_table: str,
    label_layer,
    roi_name: str,
):
    # FIXME: Add case where label layer already contains some columns
    features_df = ome_zarr_container.get_table(
        feature_table,
        check_type="feature_table",
    ).dataframe
    # Cast to numpy array in case the data is lazily loaded as dask
    labels_current_layer = np.unique(np.array(label_layer.data))[1:]

    features_df.index = features_df.index.astype(int)
    features_df = features_df.loc[features_df.index.isin(labels_current_layer)]
    features_df = features_df.reset_index()
    # Drop duplicate columns
    features_df = features_df.loc[:, ~features_df.columns.duplicated()].copy()
    # FIXME: Get OME Zarr container store {ome_zarr_container.store}
    features_df["roi_id"] = f"Test:ROI_{roi_name}"

    # To display correct
    features_df["index"] = features_df["label"]
    label_layer.features = features_df


def remove_existing_label_layers(viewer):
    for layer in viewer.layers:
        # FIXME: Generalize well name catching
        if type(layer) == napari.layers.Labels and re.match(
            r"[A-Z][a-z]*\d+_*", layer.name
        ):
            viewer.layers.remove(layer)
