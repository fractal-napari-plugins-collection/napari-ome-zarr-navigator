import logging

import napari
import numpy as np
import zarr
from fractal_tasks_core.ngff import load_NgffWellMeta
from magicgui.widgets import (
    ComboBox,
    Container,
    FileEdit,
    PushButton,
    Select,
)
from napari.qt.threading import thread_worker
from napari.utils.colormaps import Colormap

from napari_ome_zarr_navigator.ome_zarr_image import OMEZarrImage
from napari_ome_zarr_navigator.util import calculate_well_positions
from napari_ome_zarr_navigator.utils_roi_loader import (
    NapariHandler,
    read_table,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ROILoader(Container):
    def __init__(
        self,
        viewer: napari.viewer.Viewer,
        zarr_url: str = None,
        extra_widgets=None,
    ):
        self._viewer = viewer
        self.setup_logging()
        self.zarr_url = zarr_url
        self.channel_dict = {}
        self.channel_names_dict = {}
        self.labels_dict = {}
        self.label_layers = {}

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
        self._run_button = PushButton(value=False, text="Load ROI")

        self._ome_zarr_image: OMEZarrImage = None
        self.image_changed = ImageEvent()
        # Initialize possible choices

        # Update selections & bind buttons
        self.image_changed.connect(self.update_roi_table_choices)
        self.image_changed.connect(self.update_available_image_attrs)
        self._roi_table_picker.changed.connect(self.update_roi_selection)
        self._run_button.clicked.connect(self.run)

        widgets = [
            self._roi_table_picker,
            self._roi_picker,
            self._channel_picker,
            self._level_picker,
            self._label_picker,
            self._feature_picker,
            self._run_button,
        ]
        if extra_widgets:
            widgets = extra_widgets + widgets

        super().__init__(widgets=widgets)

    @property
    def ome_zarr_image(self):
        return self._ome_zarr_image

    @ome_zarr_image.setter
    def ome_zarr_image(self, value) -> OMEZarrImage:
        if self._ome_zarr_image != value:
            self._ome_zarr_image = value
            self.image_changed.emit(self._ome_zarr_image)

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

    def update_roi_selection(self):
        @thread_worker
        def get_roi_choices():
            try:
                roi_table = read_table(
                    self.zarr_url, self._roi_table_picker.value
                )
                new_choices = list(roi_table.obs_names)
                return new_choices
            except zarr.errors.PathNotFoundError:
                return [""]

        if self.ome_zarr_image:
            worker = get_roi_choices()
            worker.returned.connect(self._apply_roi_choices_update)
            worker.start()
        else:
            self._apply_roi_choices_update([""])

    def _apply_roi_choices_update(self, roi_list):
        """
        Update the list of available ROIs in the dropdown
        """
        self._roi_picker.choices = roi_list
        self._roi_picker._default_choices = roi_list

    def update_roi_table_choices(self, event):
        @thread_worker
        def threaded_get_table_list(table_type: str = None, strict=False):
            return self.ome_zarr_image.get_tables_list(
                table_type=table_type,
                strict=strict,
            )

        if self.ome_zarr_image:
            worker = threaded_get_table_list(
                table_type="ROIs",
                strict=False,
            )
            worker.returned.connect(self._apply_roi_table_choices_update)
            worker.start()
        else:
            self._apply_roi_table_choices_update([""])

    def _apply_roi_table_choices_update(self, table_list):
        """
        Update the list of available ROI tables in the dropdown menu
        """
        self._roi_table_picker.choices = table_list
        self._roi_table_picker._default_choices = table_list

    def update_available_image_attrs(self, new_zarr_img):
        if new_zarr_img:
            channels = self.ome_zarr_image.get_channel_list()
            levels = self.ome_zarr_image.get_pyramid_levels()
            labels = self.ome_zarr_image.get_labels_list()
            features = self.ome_zarr_image.get_tables_list(
                table_type="feature_table"
            )
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

    def add_intensity_roi(
        self,
        channel: str,
        roi_table: str,
        roi_name: str,
        level: str,
        blending: str,
        translate: tuple[float, float],
        layer_name: str = "",
    ):
        img_roi, scale_img = self.ome_zarr_image.load_intensity_roi(
            roi_table=roi_table,
            roi_name=roi_name,
            channel=channel,
            level_path=level,
        )
        if not np.any(img_roi):
            return

        # Get channel omero metadata
        omero = self.ome_zarr_image.get_omero_metadata(channel)
        try:
            # Colormap creation needs to have this black initial color for
            # background
            colormap = Colormap(
                ["#000000", f"#{omero.color}"],
                name=omero.color,
            )
        except AttributeError:
            colormap = None
        try:
            rescaling = (
                omero.window.start,
                omero.window.end,
            )
        except AttributeError:
            rescaling = None

        if layer_name in self._viewer.layers:
            logger.info(f"{layer_name} is already loaded")
        else:
            self._viewer.add_image(
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

    def run(self):
        # TODO: Handle case of this function being slow: Threadworker?
        roi_table = self._roi_table_picker.value
        roi_name = self._roi_picker.value
        level = self._level_picker.value
        channels = self._channel_picker.value
        labels = self._label_picker.value
        if len(channels) < 1 and len(labels) < 1:
            logger.info(
                "No channel or labels selected. "
                "Select the channels/labels you want to load"
            )
            return
        blending = None

        # Get translation within the larger image based on the ROI table
        roi_df = self.ome_zarr_image.read_table(roi_table).to_df()
        roi_translation = (
            self.translation[0] + roi_df.loc[roi_name, "y_micrometer"],
            self.translation[1] + roi_df.loc[roi_name, "x_micrometer"],
        )
        # scale_img = None

        # Set layer names
        if roi_df.shape[0] == 1:
            layer_base_name = self.layer_base_name
        else:
            layer_base_name = f"{self.layer_base_name}{roi_name}_"

        # Load intensity images
        for channel in channels:
            self.add_intensity_roi(
                channel,
                roi_table,
                roi_name,
                level,
                blending,
                translate=roi_translation,
                layer_name=f"{layer_base_name}{channel}",
            )
            blending = "additive"

        # Load labels
        # TODO: handle case of no intensity image being present =>
        # level choice for labels?
        for label in labels:
            label_roi, scale_label = self.ome_zarr_image.load_label_roi(
                roi_table=roi_table,
                roi_name=roi_name,
                label=label,
                level_path_img=level,
            )

            layer_name = f"{layer_base_name}{label}"
            if layer_name in self._viewer.layers:
                logger.info(f"{layer_name} is already loaded")
            else:
                self.label_layers[label] = self._viewer.add_labels(
                    label_roi,
                    scale=scale_label,
                    name=layer_name,
                    translate=roi_translation,
                )

        # Load features
        features = self._feature_picker.value
        for table_name in features:
            # FIXME: Check if no label type or no match
            label_layer = self.find_matching_label_layer(table_name)
            self.add_feature_table_to_layer(
                table_name,
                label_layer,
                roi_name,
            )

    def find_matching_label_layer(self, table_name):
        """
        Finds the matching label layer for a feature table
        """
        table_attrs = self.ome_zarr_image.get_table_attrs(table_name)
        try:
            target_label_name = table_attrs["region"]["path"].split("/")[-1]
        except KeyError:
            target_label_name = list(self.label_layers.keys())[0]
            logger.info(
                f"Table {table_name} did not have region metadata to match"
                "it to the correct label image. Attaching the features to the"
                f"first selected label layer ({target_label_name})"
            )

        if target_label_name not in self.label_layers:
            target_label_name = list(self.label_layers.keys())[0]
            logger.info(
                f"The label {target_label_name} that {table_name} would be "
                "matched to where not loaded. Attaching the features to the"
                f"first selected label layer ({target_label_name})"
            )

        return self.label_layers[target_label_name]

    def add_feature_table_to_layer(self, feature_table, label_layer, roi_name):
        # FIXME: Add case where label layer already contains some columns
        feature_ad = self.ome_zarr_image.read_table(
            table_name=feature_table,
        )
        if "label" in feature_ad.obs:
            # Cast to numpy array in case the data is lazily loaded as dask
            labels_current_layer = np.unique(np.array(label_layer.data))[1:]
            shared_labels = list(
                set(feature_ad.obs["label"].astype(int))
                & set(labels_current_layer)
            )
            features_roi = feature_ad[
                feature_ad.obs["label"].astype(int).isin(shared_labels)
            ]
            features_df = features_roi.to_df()
            # Drop duplicate columns
            features_df = features_df.loc[
                :, ~features_df.columns.duplicated()
            ].copy()
            features_df["label"] = feature_ad.obs["label"].astype(int)
            features_df["roi_id"] = f"{self.zarr_url}:ROI_{roi_name}"
            features_df.set_index("label", inplace=True, drop=False)
            # To display correct
            features_df["index"] = features_df["label"]
            label_layer.features = features_df
        else:
            logger.info(
                f"Table {feature_table} does not have a label obs "
                "column, can't be loaded as features for the "
                f"layer {label_layer}"
            )


class ROILoaderImage(ROILoader):
    def __init__(self, viewer: napari.viewer.Viewer, zarr_url: str = None):
        self._zarr_url_picker = FileEdit(label="Zarr URL", mode="d")
        super().__init__(
            viewer=viewer,
            extra_widgets=[
                self._zarr_url_picker,
            ],
        )
        self._zarr_url_picker.changed.connect(self.update_image_selection)
        if zarr_url:
            self._zarr_url_picker.value = zarr_url

    def update_image_selection(self):
        self.zarr_url = self._zarr_url_picker.value
        try:
            self.ome_zarr_image = OMEZarrImage(self.zarr_url)
        except ValueError:
            self.ome_zarr_image = None


class ROILoaderPlate(ROILoader):
    def __init__(
        self, viewer: napari.viewer.Viewer, plate_url: str, row: str, col: str
    ):
        self._zarr_picker = ComboBox(label="Image")
        self.plate_url = plate_url.rstrip("/")
        self.row = row
        self.col = col
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

        # Calculate base translation for a given well
        self.translation, _ = calculate_well_positions(
            plate_url=plate_url,
            row=row,
            col=col,
        )

        # # Handle defaults for plate loading
        # if "well_ROI_table" in self._roi_table_picker.choices:
        #     self._roi_table_picker.value = "well_ROI_table"

    def get_available_ome_zarr_images(self):
        well_url = f"{self.plate_url}/{self.row}/{self.col}"
        well_meta = load_NgffWellMeta(well_url)
        return [image.path for image in well_meta.well.images]

    def update_image_selection(self):
        self.zarr_url = (
            f"{self.plate_url}/{self.row}/{self.col}/{self._zarr_picker.value}"
        )
        try:
            self.ome_zarr_image = OMEZarrImage(self.zarr_url)
        except ValueError:
            self.ome_zarr_image = None


class ImageEvent:
    def __init__(self):
        self.handlers = []

    def connect(self, handler):
        self.handlers.append(handler)

    def emit(self, *args, **kwargs):
        for handler in self.handlers:
            handler(*args, **kwargs)
