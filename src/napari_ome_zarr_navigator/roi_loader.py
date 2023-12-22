import napari
from magicgui.widgets import (
    ComboBox,
    Container,
    FileEdit,
    PushButton,
    Select,
)

from napari_ome_zarr_navigator.utils_roi_loader import threaded_get_table_list


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

        # Initialize possible choices
        self.update_roi_selection()

        # Update selections & bind buttons
        self._zarr_url_picker.changed.connect(self.update_roi_table_choices)
        self._run_button.clicked.connect(self.run)
        self._roi_table_picker.changed.connect(self.update_roi_selection)

        if zarr_url:
            self._zarr_url_picker.value = zarr_url

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

    def update_roi_selection(self):
        print("update_roi_selection")

    def update_roi_table_choices(self):
        worker = threaded_get_table_list(
            zarr_url=self._zarr_url_picker.value,
            table_type="ROIs",
            strict=False,
        )
        worker.returned.connect(self.apply_roi_table_choices_update)
        worker.start()

    def apply_roi_table_choices_update(self, table_list):
        """
        Update the list of available ROI tables in the dropdown menu
        """
        self._roi_table_picker.choices = table_list
        self._roi_table_picker._default_choices = table_list
        # Update the list of options that depend on the ROI table selection
        self.update_roi_selection()

    def run(self):
        print("run")

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

    # def _get_channel_choices(self):
    #     self.channel_dict = get_channel_dict(self._zarr_url_picker.value)
    #     self.channel_names_dict = {}
    #     for channel_index in self.channel_dict.keys():
    #         channel_name = self.channel_dict[channel_index]["label"]
    #         self.channel_names_dict[channel_name] = channel_index
    #     return list(self.channel_names_dict.keys())

    # def _get_label_choices(self):
    #     self.label_dict = get_label_dict(
    #         Path(self._zarr_url_picker.value) / "labels"
    #     )
    #     return list(self.label_dict.values())

    # def _get_table_choices(self, type):
    #     # TODO: Once we have relevant metadata, allow this function to only
    #     # load ROI tables or only feature tables => type features or ROIs
    #     self.label_dict = get_feature_dict(
    #         Path(self._zarr_url_picker.value) / "tables"
    #     )
    #     potential_tables = list(self.label_dict.values())
    #     if type == "ROIs":
    #         return [table for table in potential_tables if "ROI" in table]
    #     else:
    #         return [table for table in potential_tables if "ROI" not in table]

    # def _get_level_choices(self):
    #     try:
    #         metadata = get_metadata(self._zarr_url_picker.value)
    #         dataset = 0  # FIXME, hard coded in case multiple multiscale
    #         # datasets would be present & multiscales is a list
    #         nb_levels = len(metadata.attrs["multiscales"][dataset]["datasets"])
    #         return list(range(nb_levels))
    #     except KeyError:
    #         # This happens when no valid OME-Zarr file is selected, thus no
    #         # metadata file is found & no levels can be set
    #         return [""]
