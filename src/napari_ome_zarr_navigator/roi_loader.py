import logging

import napari
import numpy as np
import zarr
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
from napari_ome_zarr_navigator.utils_roi_loader import (
    read_table,
)

logger = logging.getLogger(__name__)


class ROILoader(Container):
    def __init__(self, viewer: napari.viewer.Viewer, zarr_url=None):
        self._viewer = viewer
        self.channel_dict = {}
        self.channel_names_dict = {}
        self.labels_dict = {}
        self.label_layers = {}
        self._zarr_url_picker = FileEdit(label="Zarr URL", mode="d")
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
        self._zarr_url_picker.changed.connect(self.update_image_selection)
        self.image_changed.connect(self.update_roi_table_choices)
        self.image_changed.connect(self.update_available_image_attrs)
        self._roi_table_picker.changed.connect(self.update_roi_selection)
        self._run_button.clicked.connect(self.run)

        if zarr_url:
            self.ome_zarr_image = OMEZarrImage(zarr_url)

        super().__init__(
            widgets=[
                self._zarr_url_picker,
                self._roi_table_picker,
                self._roi_picker,
                self._channel_picker,
                self._level_picker,
                self._label_picker,
                self._feature_picker,
                self._run_button,
            ]
        )

    @property
    def ome_zarr_image(self):
        return self._ome_zarr_image

    @ome_zarr_image.setter
    def ome_zarr_image(self, value) -> OMEZarrImage:
        if self._ome_zarr_image != value:
            self._ome_zarr_image = value
            self.image_changed.emit(self._ome_zarr_image)

    def update_roi_selection(self):
        @thread_worker
        def get_roi_choices():
            try:
                roi_table = read_table(
                    self._zarr_url_picker.value, self._roi_table_picker.value
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

    def update_image_selection(self):
        zarr_url = self._zarr_url_picker.value
        try:
            self.ome_zarr_image = OMEZarrImage(zarr_url)
        except ValueError:
            self.ome_zarr_image = None

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
    ):
        img_roi, scale_img = self.ome_zarr_image.load_intensity_roi(
            roi_table=roi_table,
            roi_name=roi_name,
            channel=channel,
            level_path=level,  # FIXME: pass
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

        self._viewer.add_image(
            img_roi,
            scale=scale_img,
            blending=blending,
            contrast_limits=rescaling,
            colormap=colormap,
            name=channel,
        )
        # TODO: Optionally return some values as well? e.g. if info is needed
        # by label loading

    def run(self):
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
        # scale_img = None

        # Load intensity images
        for channel in channels:
            self.add_intensity_roi(
                channel, roi_table, roi_name, level, blending
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
            # if not np.any(label_roi):
            #     show_info(
            #         "Could not load this ROI. Did you correctly set the "
            #         "`Reset ROI Origin`?"
            #     )
            #     return
            self.label_layers[label] = self._viewer.add_labels(
                label_roi, scale=scale_label, name=label
            )


class ImageEvent:
    def __init__(self):
        self.handlers = []

    def connect(self, handler):
        self.handlers.append(handler)

    def emit(self, *args, **kwargs):
        for handler in self.handlers:
            handler(*args, **kwargs)

    # def run(self):
    #     # Load features
    #     # Initially a bearbones implementation that only works when a single
    #     # label image is also loaded at that moment
    #     features = self._feature_picker.value
    #     if len(features) > 0:
    #         if len(labels) != 1:
    #             show_info(
    #                 "Not implemented yet: Please select exactly one label "
    #                 "image to load features for"
    #             )
    #             return
    #         else:
    #             # TODO: Implement loading multiple features at once
    #             # (and mapping them to the correct labels)
    #             if len(features) > 1:
    #                 show_info(
    #                     "Not implemented yet: Please select exactly one "
    #                     "feature to load"
    #                 )
    #                 return
    #             else:
    #                 # Actual feature loading
    #                 feature_table = features[0]
    #                 label_layer = label_layers[0]
    #                 self.add_feature_table_to_layer(
    #                     feature_table,
    #                     label_layer,
    #                     roi_name,
    #                 )

    # def add_feature_table_to_layer(self, feature_table, label_layer, roi_name):
    #     feature_ad = load_features(
    #         zarr_url=self._zarr_url_picker.value,
    #         feature_table=feature_table,
    #     )
    #     if "label" in feature_ad.obs:
    #         # TODO: Only load the feature for the ROI,
    #         # not the whole table
    #         labels_current_layer = np.unique(label_layer.data)[1:]
    #         shared_labels = list(
    #             set(feature_ad.obs["label"].astype(int))
    #             & set(labels_current_layer)
    #         )
    #         features_roi = feature_ad[
    #             feature_ad.obs["label"].astype(int).isin(shared_labels)
    #         ]
    #         features_df = features_roi.to_df()
    #         # Drop duplicate columns
    #         features_df = features_df.loc[
    #             :, ~features_df.columns.duplicated()
    #         ].copy()
    #         features_df["label"] = feature_ad.obs["label"].astype(int)
    #         features_df[
    #             "roi_id"
    #         ] = f"{self._zarr_url_picker.value}:ROI_{roi_name}"
    #         features_df.set_index("label", inplace=True, drop=False)
    #         # To display correct
    #         features_df["index"] = features_df["label"]
    #         label_layer.features = features_df
    #     else:
    #         show_info(
    #             f"Table {feature_table} does not have a label obs "
    #             "column, can't be loaded as features for the "
    #             f"layer {label_layer}"
    #         )
