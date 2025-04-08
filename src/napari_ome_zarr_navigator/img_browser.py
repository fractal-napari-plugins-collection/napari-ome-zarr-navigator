import logging
import re
from contextlib import suppress

import napari
import numpy as np
import pandas as pd
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FloatSpinBox,
    ProgressBar,
    PushButton,
    Select,
)
from ngio import open_ome_zarr_container, open_ome_zarr_plate
from ngio.utils import (
    NgioFileNotFoundError,
    NgioValidationError,
    fractal_fsspec_store,
)

from napari_ome_zarr_navigator.roi_loader import (
    ROILoaderPlate,
    load_roi,
    remove_existing_label_layers,
)
from napari_ome_zarr_navigator.util import (
    ZarrSelector,
    calculate_well_positions,
)

logger = logging.getLogger(__name__)


class ImgBrowser(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        self.viewer = viewer
        self._zarr_selector = ZarrSelector()
        self.filters = []
        self.well = Select(label="Wells", enabled=True, allow_multiple=False)
        self.select_well = PushButton(text="➡ Go to well", enabled=False)
        self.zoom_level = FloatSpinBox(value=0.25, min=0.01, step=0.01)
        self.btn_load_roi = PushButton(
            text="Select ROI to load", enabled=False
        )
        self.btn_load_default_roi = PushButton(
            text="Load selected ROI for additional well(s)",
            enabled=False,
            tooltip="Once you've loaded a ROI with the 'Select ROI to load' "
            "button, this allows you to load the same ROI for different wells.",
        )
        self._load_condition_tables = CheckBox(
            value=False, text="Load condition tables"
        )
        self.roi_loader = None
        self.roi_widget = None
        self.filter_widget = None
        self.progress = ProgressBar(visible=False)
        self.default_zarr_image_subgroup = None
        self.default_roi_table = None
        self.default_roi_name = None
        self.default_channels = None
        self.default_level = None
        self.default_labels = None
        self.default_features = None
        self.remove_old_labels = False
        self.zarr_plate = None
        self.plate_store = None
        self.df = None  # Dataframe for condition table

        super().__init__(
            widgets=[
                self._zarr_selector,
                self.well,
                self.progress,
                Container(
                    widgets=[self.zoom_level, self.select_well],
                    label="Zoom level",
                    layout="horizontal",
                ),
                self._load_condition_tables,
                self.btn_load_roi,
                self.btn_load_default_roi,
            ],
        )
        self.viewer.layers.selection.events.changed.connect(self.get_zarr_url)
        self.filter_names = None
        self._zarr_selector.on_change(self.initialize_filters)
        self._zarr_selector.on_change(self.filter_df)
        self.select_well.clicked.connect(self.go_to_well)
        self.btn_load_roi.clicked.connect(self.launch_load_roi)
        self.btn_load_default_roi.clicked.connect(self.load_default_roi)
        self._load_condition_tables.changed.connect(self.initialize_filters)
        self._load_condition_tables.changed.connect(self.filter_df)
        self.viewer.layers.events.removed.connect(self.check_empty_layerlist)

    def open_zarr_plate(self):
        if self._zarr_selector._source_selector.value == "File":
            self.plate_store = self._zarr_selector.url
        else:
            self.plate_store = fractal_fsspec_store(
                self._zarr_selector.url,
                fractal_token=self._zarr_selector.token,
            )
        try:
            self.zarr_plate = open_ome_zarr_plate(
                self.plate_store, cache=True, parallel_safe=False, mode="r"
            )
        except NgioFileNotFoundError:
            self.zarr_plate = None
        except NgioValidationError as e:
            self.zarr_plate = None
            msg = (
                "No valid Zarr plate found at the provided URL. Verify the "
                "URL to ensure it points to the root of the plate or "
                f"check the validation error: \n {e}"
            )
            logger.info(msg)
            napari.utils.notifications.show_info(msg)

    def initialize_filters(self):
        # Check if the zarr_url is empty & finish early
        if (
            self._zarr_selector.url is None
            or self._zarr_selector.url == "."
            or self._zarr_selector.url == ""
        ):
            self.is_plate = False
            self.select_well.enabled = False
            self.btn_load_roi.enabled = False
            return

        self.open_zarr_plate()
        if self.zarr_plate:
            self.is_plate = True
            if self._load_condition_tables.value:
                self.df = self.load_condition_table()
            if self.df is not None:
                self.set_filtered_wells_for_selection()
            else:
                self.set_all_wells_for_selection()

            self.select_well.enabled = True
            self.btn_load_roi.enabled = True
        else:
            self.is_plate = False
            self.select_well.enabled = False
            self.btn_load_roi.enabled = False

    def set_all_wells_for_selection(self):
        wells = []
        dfs = []
        # FIXME: Refactor to not load well objects, just well row/col metadata
        # based on plate object
        for well_path in self.zarr_plate.get_wells():
            row = well_path.split("/")[0]
            col = well_path.split("/")[1]
            wells.append(f"{row}{col}")
            dfs.append(pd.DataFrame({"row": [row], "col": [col]}))
        wells_str = sorted(wells)
        self.well.choices = wells_str
        self.well._default_choices = wells_str
        self.df = pd.concat(dfs, ignore_index=True)
        self.filter_names = None
        self.well.value = wells_str[0]

    def set_filtered_wells_for_selection(self):
        self.df_without_pk = self.df.drop(
            columns=["row", "col"]
        )  # .apply(pd.to_numeric, errors="ignore")
        # Casting to numeric with error ignore is deprecated.
        # What would the point of it be??
        self.df = pd.concat(
            (self.df[["row", "col"]], self.df_without_pk), axis=1
        )
        self.filter_names = self.df_without_pk.columns

        self.filters = Container(
            widgets=[
                Container(
                    widgets=[
                        ComboBox(
                            choices=self.df_without_pk[filter_name]
                            .sort_values()
                            .unique(),
                            enabled=False,
                        ),
                        CheckBox(label=filter_name, value=False),
                    ],
                    layout="horizontal",
                )
                for filter_name in self.filter_names
            ],
            layout="vertical",
        )
        # TODO: Improve adding/removing behavior of dockwidget
        if self.filter_widget:
            with suppress(RuntimeError):
                self.viewer.window.remove_dock_widget(self.filter_widget)

        self.filter_widget = self.viewer.window.add_dock_widget(
            widget=self.filters,
            name="Filters",
        )

        for i in range(len(self.filter_names)):
            self.filters[i][1].changed.connect(self.toggle_filter(i))
            self.filters[i][0].changed.connect(self.filter_df)
            self.filters[i][1].changed.connect(self.filter_df)

    def toggle_filter(self, i):
        def toggle_on_change():
            self.filters[i][0].enabled = not self.filters[i][0].enabled

        return toggle_on_change

    def check_empty_layerlist(self):
        if len(self.viewer.layers) == 0:
            self._zarr_selector.set_url("")
            self.select_well.enabled = False
            self.btn_load_roi.enabled = False
            self.df = pd.DataFrame()
            self.well.choices = []
            self.well._default_choices = []
            if self.filter_names is not None:
                self.viewer.window.remove_dock_widget(
                    widget=self.filter_widget
                )

    def get_zarr_url(self):
        # When the user adds new image layers, check if they have zarr_urls in
        # the path or have a sample_path. If so, update the plugin with it.
        active = self.viewer.layers.selection.active
        if active and active.as_layer_data_tuple()[-1] == "image":
            path = self.viewer.layers.selection.active.source.path
            if path:
                self._zarr_selector.set_url(path)
            if "sample_path" in self.viewer.layers.selection.active.metadata:
                self._zarr_selector.set_url(
                    self.viewer.layers.selection.active.metadata["sample_path"]
                )

    def filter_df(self):
        if self.filter_names is not None:
            and_filter = pd.DataFrame(
                [
                    (
                        self.df.iloc[:, i + 2] == self.filters[i][0].value
                        if self.filters[i][1].value
                        else self.df.iloc[:, i + 2] == self.df.iloc[:, i + 2]
                    )
                    for i in range(len(self.filter_names))
                ]
            ).min()

            tbl = self.df.loc[and_filter]
            wells = (
                (tbl["row"] + tbl["col"].astype(str)).sort_values().unique()
            )
            self.well.choices = wells
            self.well._default_choices = wells
            if len(wells) > 0:
                self.well.value = wells[0]

    def launch_load_roi(self):
        wells = get_row_cols(self.well.value)
        if len(wells) != 1:
            msg = "Please select a single well."
            logger.info(msg)
            napari.utils.notifications.show_info(msg)
        else:
            if self.roi_widget:
                with suppress(RuntimeError):
                    self.viewer.window.remove_dock_widget(self.roi_widget)

            self.roi_loader = ROILoaderPlate(
                self.viewer,
                self.plate_store,
                wells[0][0],
                wells[0][1],
                self,
                self.is_plate,
            )
            self.roi_widget = self.viewer.window.add_dock_widget(
                widget=self.roi_loader,
                name="ROI Loader",
                tabify=True,
                allowed_areas=["right"],
            )

    def load_default_roi(self):
        wells = get_row_cols(self.well.value)

        # Loop over all selected wells
        for well in wells:
            if self.remove_old_labels:
                remove_existing_label_layers(self.viewer)
            layer_base_name = f"{well[0]}{well[1]}_"
            # Calculate translations
            translation, _ = calculate_well_positions(
                plate_store=self._zarr_selector.url,
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
            load_roi(
                ome_zarr_container=ome_zarr_container,
                viewer=self.viewer,
                roi_table=self.default_roi_table,
                roi_name=self.default_roi_name,
                layer_base_name=layer_base_name,
                channels=self.default_channels,
                level=self.default_level,
                labels=self.default_labels,
                features=self.default_features,
                translation=translation,
                blending=None,
            )

    def go_to_well(self):
        wells = get_row_cols(self.well.value)

        for layer in self.viewer.layers:
            if type(layer) == napari.layers.Shapes and re.match(
                r"([A-Z][a-z]*)(\d+)", layer.name
            ):
                self.viewer.layers.remove(layer)

        if len(wells) > 0:
            for well in wells:
                (
                    top_left_corner,
                    bottom_right_corner,
                ) = calculate_well_positions(
                    plate_url=self._zarr_selector.url,
                    row=well[0],
                    col=well[1],
                    is_plate=self.is_plate,
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

            self.viewer.camera.center = rec.mean(axis=0)
            self.viewer.camera.zoom = self.zoom_level.value
        else:
            msg = "Please select at least one well"
            logger.info(msg)
            napari.utils.notifications.show_info(msg)

    def load_condition_table(self, table_name="condition"):
        wells = self.zarr_plate.get_wells()
        self.progress = ProgressBar(visible=False)
        self.progress.visible = True
        self.progress.min = 0
        self.progress.max = len(wells)
        self.progress.value = 0
        all_tables = []
        for well_path, well in self.zarr_plate.get_wells().items():
            # Currently only loads condition tables from the first image
            image = well.get_image(image_path=well.paths()[0])
            try:
                all_tables.append(image.get_table(name=table_name).dataframe)
            except KeyError:
                logger.info(
                    f'The table "{table_name}" was not found in well {well_path}'
                )

            self.progress.value += 1
        self.progress.visible = False
        if len(all_tables) == 0:
            msg = "No condition table is present in the OME-Zarr."
            logger.info(msg)
            napari.utils.notifications.show_info(msg)
            return None
        else:
            return pd.concat(all_tables, ignore_index=True)

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
    ):
        self.default_zarr_image_subgroup = zarr_image_subgroup
        self.default_roi_table = roi_table
        self.default_roi_name = roi_name
        self.default_channels = channels
        self.default_level = level
        self.default_labels = labels
        self.default_features = features
        self.remove_old_labels = remove_old_labels
        self.btn_load_default_roi.enabled = True


# Deprecated searching for plate zarr url for now, assuming the user inputs it.
# def parse_zarr_url(zarr_url: Union[str, Path]) -> dict:
#     """Parse the OME-Zarr URL into a dictionary with the root URL, row, column and dataset

#     Args:
#         zarr_url: Path to the OME-Zarr

#     Returns:
#         Dictionary with root URL, row, column and dataset
#     """
#     zarr_dict = {
#         "root": None,
#         "row_alpha": None,
#         "row": None,
#         "col": None,
#         "well": None,
#         "dataset": None,
#     }
#     if zarr_url:
#         parts = [
#             p.replace("\\", "") for p in Path(zarr_url).parts
#         ]  # remove backslash escape character
#         root_idx = None
#         for i, p in enumerate(parts):
#             if p.endswith(".zarr"):
#                 root_idx = i
#                 zarr_dict["root"] = Path(*parts[0 : i + 1])
#             if root_idx and i == root_idx + 1:
#                 zarr_dict["row_alpha"] = p
#                 zarr_dict["row"] = alpha_to_numeric(p)
#             if root_idx and i == root_idx + 2:
#                 zarr_dict["col"] = int(p)
#                 zarr_dict["well"] = zarr_dict["row_alpha"] + p
#             if root_idx and i == root_idx + 3:
#                 zarr_dict["dataset"] = int(p)
#     return zarr_dict


def get_row_cols(well_list):
    """
    Given a well list, provide a list of rows & columns

    The well list i

    """
    matches = [re.match(r"([A-Z][a-z]*)(\d+)", well) for well in well_list]
    wells = [(m.group(1), m.group(2)) for m in matches]
    return wells
