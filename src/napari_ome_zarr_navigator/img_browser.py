import logging
import re
from contextlib import suppress
from pathlib import Path
from typing import Union

import anndata as ad
import napari
import numpy as np
import pandas as pd
import zarr
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FileEdit,
    FloatSpinBox,
    ProgressBar,
    PushButton,
    Select,
)
from zarr.errors import PathNotFoundError

from napari_ome_zarr_navigator.roi_loader import ROILoaderPlate
from napari_ome_zarr_navigator.util import (
    alpha_to_numeric,
    calculate_well_positions,
)

logger = logging.getLogger(__name__)
logging.getLogger("ome_zarr").setLevel(logging.WARN)


class ImgBrowser(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        self.viewer = viewer
        self.zarr_dir = FileEdit(
            label="OME-ZARR URL", mode="d", filter="*.zarr"
        )
        self.filters = []
        self.well = Select(label="Wells", enabled=True, allow_multiple=False)
        self.select_well = PushButton(text="➡ Go to well", enabled=False)
        self.zoom_level = FloatSpinBox(value=0.25, min=0.01, step=0.01)
        self.btn_load_roi = PushButton(
            text="Send to ROI Loader ⤵️", enabled=False
        )
        self.roi_loader = None
        self.roi_widget = None
        self.filter_widget = None
        self.progress = ProgressBar(visible=False)

        super().__init__(
            widgets=[
                self.zarr_dir,
                self.well,
                self.progress,
                Container(
                    widgets=[self.zoom_level, self.select_well],
                    label="Zoom level",
                    layout="horizontal",
                ),
                self.btn_load_roi,
            ],
        )
        self.viewer.layers.selection.events.changed.connect(self.get_zarr_url)
        self.zarr_dir.changed.connect(self.initialize_filters)
        self.zarr_dir.changed.connect(self.filter_df)
        self.select_well.clicked.connect(self.go_to_well)
        self.btn_load_roi.clicked.connect(self.load_roi)
        self.viewer.layers.events.removed.connect(self.check_empty_layerlist)

    def initialize_filters(self):
        self.zarr_dict = parse_zarr_url(self.zarr_dir.value)
        self.zarr_root = self.zarr_dict["root"]
        if self.zarr_root:
            adt = self.load_table()
            if adt:
                self.df = adt.to_df()
                self.df_without_pk = self.df.drop(
                    columns=["row", "col"]
                ).apply(pd.to_numeric, errors="ignore")
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
                if self.filter_widget:
                    with suppress(RuntimeError):
                        self.viewer.window.remove_dock_widget(
                            self.filter_widget
                        )

                self.filter_widget = self.viewer.window.add_dock_widget(
                    widget=self.filters,
                    name="Filters",
                )

                for i in range(len(self.filter_names)):
                    self.filters[i][1].changed.connect(self.toggle_filter(i))
                    self.filters[i][0].changed.connect(self.filter_df)
                    self.filters[i][1].changed.connect(self.filter_df)
            else:
                msg = "No condition table is present in the OME-ZARR."
                logger.info(msg)
                napari.utils.notifications.show_info(msg)
                wells = _validate_wells(self.zarr_dir.value, None)
                wells_str = sorted([f"{w[0]}{w[1]}" for w in wells])
                self.well.choices = wells_str
                self.well._default_choices = wells_str
                self.df = pd.DataFrame(
                    {
                        "row": [w[0] for w in wells],
                        "col": [w[1] for w in wells],
                    }
                )
                self.filter_names = None

            self.select_well.enabled = True
            self.btn_load_roi.enabled = True

    def toggle_filter(self, i):
        def toggle_on_change():
            self.filters[i][0].enabled = not self.filters[i][0].enabled

        return toggle_on_change

    def check_empty_layerlist(self):
        if len(self.viewer.layers) == 0:
            self.zarr_dir.value = ""
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
        active = self.viewer.layers.selection.active
        if active and active.as_layer_data_tuple()[-1] == "image":
            path = self.viewer.layers.selection.active.source.path
            if path:
                self.zarr_dir.value = Path(path)
            if "sample_path" in self.viewer.layers.selection.active.metadata:
                self.zarr_dir.value = Path(
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

    def load_roi(self):
        matches = [
            re.match(r"([A-Z]+)(\d+)", well) for well in self.well.value
        ]
        row_alpha = [m.group(1) for m in matches]
        col_str = [m.group(2) for m in matches]
        if len(row_alpha) != 1 or len(col_str) != 1:
            msg = "Please select a single well."
            logger.info(msg)
            napari.utils.notifications.show_info(msg)
        else:
            if self.roi_widget:
                with suppress(RuntimeError):
                    self.viewer.window.remove_dock_widget(self.roi_widget)

            self.roi_loader = ROILoaderPlate(
                self.viewer, str(self.zarr_root), row_alpha[0], col_str[0]
            )
            self.roi_widget = self.viewer.window.add_dock_widget(
                widget=self.roi_loader,
                name="ROI Loader",
                tabify=True,
                allowed_areas=["right"],
            )

    def go_to_well(self):
        matches = [
            re.match(r"([A-Z]+)(\d+)", well) for well in self.well.value
        ]
        wells = [(m.group(1), m.group(2)) for m in matches]

        for layer in self.viewer.layers:
            if type(layer) == napari.layers.Shapes and re.match(
                r"([A-Z]+)(\d+)", layer.name
            ):
                self.viewer.layers.remove(layer)

        if len(wells) > 0:
            for well in wells:
                (
                    top_left_corner,
                    bottom_right_corner,
                ) = calculate_well_positions(
                    plate_url=self.zarr_root, row=well[0], col=well[1]
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

    def load_table(self):
        wells = _validate_wells(self.zarr_root, self.zarr_dict["well"])
        wells_str = [f"{w[0]}{w[1]}" for w in wells]
        tbl = []
        n = len(wells)
        self.progress.visible = True
        self.progress.min = 0
        self.progress.max = n
        self.progress.value = 0
        name = "condition"
        dataset = 0
        while wells:
            row_alpha, col = wells.pop()
            try:
                tbl.append(
                    ad.read_zarr(
                        f"{self.zarr_root}/{row_alpha}/{col}/{dataset}/tables/{name}"
                    )
                )
            except PathNotFoundError:
                logger.info(
                    f'The table "{name}" was not found in well {row_alpha}{col}'
                )
            self.progress.value += 1
        self.progress.visible = False
        if tbl:
            if len(wells_str) > 1:
                return ad.concat(tbl, keys=wells_str, index_unique="-")
            else:
                return ad.concat(tbl)


def parse_zarr_url(zarr_url: Union[str, Path]) -> dict:
    """Parse the OME-ZARR URL into a dictionary with the root URL, row, column and dataset

    Args:
        zarr_url: Path to the OME-ZARR

    Returns:
        Dictionary with root URL, row, column and dataset
    """
    zarr_dict = {
        "root": None,
        "row_alpha": None,
        "row": None,
        "col": None,
        "well": None,
        "dataset": None,
    }
    if zarr_url:
        parts = [
            p.replace("\\", "") for p in Path(zarr_url).parts
        ]  # remove backslash escape character
        root_idx = None
        for i, p in enumerate(parts):
            if p.endswith(".zarr"):
                root_idx = i
                zarr_dict["root"] = Path(*parts[0 : i + 1])
            if root_idx and i == root_idx + 1:
                zarr_dict["row_alpha"] = p
                zarr_dict["row"] = alpha_to_numeric(p)
            if root_idx and i == root_idx + 2:
                zarr_dict["col"] = int(p)
                zarr_dict["well"] = zarr_dict["row_alpha"] + p
            if root_idx and i == root_idx + 3:
                zarr_dict["dataset"] = int(p)
    return zarr_dict


def _validate_wells(
    zarr_url: Union[str, Path], wells: Union[str, list[str], None]
) -> set[tuple[str, int]]:
    """Check that wells are formatted correctly

    Args:
        zarr_url: Path to the OME-ZARR
        wells: A single well, a list of wells on a plate or None

    Returns:
        A unique set of alphanumeric tuples describing the wells
    """
    if wells is not None:
        wells = [wells] if isinstance(wells, str) else wells
        matches = [re.match(r"([A-Z]+)(\d+)", well) for well in wells]
        wells = {(m.group(1), m.group(2)) for m in matches}
    else:
        with zarr.open(zarr_url) as metadata:
            matches = [
                re.match(r"([A-Z]+)/(\d+)", well["path"])
                for well in metadata.attrs["plate"]["wells"]
            ]
            wells = {(m.group(1), m.group(2)) for m in matches}
    return wells
