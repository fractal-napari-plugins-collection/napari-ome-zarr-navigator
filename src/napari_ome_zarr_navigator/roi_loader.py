import napari
import zarr
from magicgui.widgets import (
    ComboBox,
    Container,
    FileEdit,
    PushButton,
    Select,
)
from napari.qt.threading import thread_worker

from napari_ome_zarr_navigator.ome_zarr_image import OMEZarrImage
from napari_ome_zarr_navigator.utils_roi_loader import (
    read_table,
)


class ROILoader(Container):
    def __init__(self, viewer: napari.viewer.Viewer, zarr_url=None):
        self._viewer = viewer
        self.channel_dict = {}
        self.channel_names_dict = {}
        self.labels_dict = {}
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

    def run(self):
        print("run")

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


class ImageEvent:
    def __init__(self):
        self.handlers = []

    def connect(self, handler):
        self.handlers.append(handler)

    def emit(self, *args, **kwargs):
        for handler in self.handlers:
            handler(*args, **kwargs)

    # def run(self):
    #     roi_table = self._roi_table_picker.value
    #     roi_name = self._roi_picker.value
    #     level = self._level_picker.value
    #     channels = self._channel_picker.value
    #     labels = self._label_picker.value
    #     if len(channels) < 1 and len(labels) < 1:
    #         show_info(
    #             "No channel or labels selected. "
    #             "Select the channels/labels you want to load"
    #         )
    #         return
    #     blending = None
    #     scale_img = None

    #     # Load intensity images
    #     for channel in channels:
    #         img_roi, scale_img = load_intensity_roi(
    #             zarr_url=self._zarr_url_picker.value,
    #             roi_of_interest=roi_name,
    #             channel_index=self.channel_names_dict[channel],
    #             level=level,
    #             roi_table=roi_table,    #         )
    #         if not np.any(img_roi):
    #             show_info(
    #                 "Could not load this ROI. Did you correctly set the "
    #                 "`Reset ROI Origin`?"
    #             )
    #             return

    #         channel_meta = self.channel_dict[self.channel_names_dict[channel]]
    #         colormap = Colormap(
    #             ["#000000", f"#{channel_meta['color']}"],
    #             name=channel_meta["color"],
    #         )
    #         try:
    #             rescaling = (
    #                 channel_meta["window"]["start"],
    #                 channel_meta["window"]["end"],
    #             )
    #         except KeyError:
    #             rescaling = None

    #         self._viewer.add_image(
    #             img_roi,
    #             scale=scale_img,
    #             blending=blending,
    #             contrast_limits=rescaling,
    #             colormap=colormap,
    #             name=channel,
    #         )
    #         blending = "additive"

    #     # Load labels
    #     label_layers = []
    #     for label in labels:
    #         label_roi, scale_label = load_label_roi(
    #             zarr_url=self._zarr_url_picker.value,
    #             roi_of_interest=roi_name,
    #             label_name=label,
    #             target_scale=scale_img,
    #             roi_table=roi_table,
    #         )
    #         if not np.any(label_roi):
    #             show_info(
    #                 "Could not load this ROI. Did you correctly set the "
    #                 "`Reset ROI Origin`?"
    #             )
    #             return
    #         label_layers.append(
    #             self._viewer.add_labels(
    #                 label_roi, scale=scale_label, name=label
    #             )
    #         )

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

    # def update_roi_tables(self):
    #     """
    #     Handles updating the list of available ROI tables
    #     """
    #     # Uses the `_default_choices` to avoid having choices reset.
    #     # See https://github.com/pyapp-kit/magicgui/issues/306
    #     # roi_table = self._get_roi_table_choices()
    #     roi_tables = self._get_table_choices(type="ROIs")
    #     self._roi_table_picker.choices = roi_tables
    #     self._roi_table_picker._default_choices = roi_tables
    #     self.update_roi_selection()

    # def update_roi_selection(self):
    #     """
    #     Updates all selections that depend on which ROI table was selected
    #     """
    #     # Uses the `_default_choices` to avoid having choices reset.
    #     # See https://github.com/pyapp-kit/magicgui/issues/306
    #     new_rois = self._get_roi_choices()
    #     self._roi_picker.choices = new_rois
    #     self._roi_picker._default_choices = new_rois
    #     channels = self._get_channel_choices()
    #     self._channel_picker.choices = channels
    #     self._channel_picker._default_choices = channels
    #     levels = self._get_level_choices()
    #     self._level_picker.choices = levels
    #     self._level_picker._default_choices = levels

    #     # Initialize available label images
    #     labels = self._get_label_choices()
    #     self._label_picker.choices = labels
    #     self._label_picker._default_choices = labels

    #     # Initialize available features
    #     features = self._get_table_choices(type="features")
    #     self._feature_picker.choices = features
    #     self._feature_picker._default_choices = features

    # def _get_roi_choices(self):
    #     if not self._roi_table_picker.value:
    #         # When no roi table is provided.
    #         # E.g. during bug with self._roi_table_picker reset
    #         return [""]
    #     try:
    #         roi_table = read_table(
    #             self._zarr_url_picker.value, self._roi_table_picker.value
    #         )
    #         new_choices = list(roi_table.obs_names)
    #         return new_choices
    #     except zarr.errors.PathNotFoundError:
    #         new_choices = [""]
    #         return new_choices
