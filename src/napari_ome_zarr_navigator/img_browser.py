from typing import TYPE_CHECKING
import napari

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union
from zarr.errors import PathNotFoundError
import anndata as ad
import zarr
import re
import logging


from napari_ome_zarr_navigator.util import alpha_to_numeric
from fractal_tasks_core.roi import convert_ROI_table_to_indices

from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FileEdit,
    PushButton,
    Select,
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
        self.select_well = PushButton(text="Go to well", enabled=False)

        super().__init__(
            widgets=[self.zarr_dir, self.well, self.select_well],
        )
        self.viewer.layers.selection.events.changed.connect(self.get_zarr_url)
        self.zarr_dir.changed.connect(self.initialize_filters)
        self.zarr_dir.changed.connect(self.filter_df)
        self.select_well.clicked.connect(self.go_to_well)
        self.viewer.layers.events.removed.connect(self.check_empty_layerlist)

    def initialize_filters(self):
        zarr_dict = parse_zarr_url(self.zarr_dir.value)
        self.zarr_root = zarr_dict["root"]
        if self.zarr_root:
            adt = load_table(
                self.zarr_root,
                "condition",
                zarr_dict["well"],
            )
            if adt:
                self.df = adt.to_df()
                self.filter_names = self.df.columns.drop(["row", "col"])
                self.filters = Container(
                    widgets=[
                        Container(
                            widgets=[
                                ComboBox(
                                    choices=sorted(self.df[filter].unique()),
                                    enabled=False,
                                ),
                                CheckBox(label=filter, value=False),
                            ],
                            layout="horizontal",
                        )
                        for filter in self.filter_names
                    ],
                    layout="vertical",
                )
                self.filter_widget = self.viewer.window.add_dock_widget(
                    widget=self.filters, name="Filters"
                )

                for i in range(len(self.filter_names)):
                    self.filters[i][1].changed.connect(self.toggle_filter(i))
                    self.filters[i][0].changed.connect(self.filter_df)
                    self.filters[i][1].changed.connect(self.filter_df)
            else:
                msg = "No condition table is present in the OME-ZARR."
                logger.info(msg)
                napari.utils.notifications.show_info(msg)
                wells = _validate_wells(None, self.zarr_dir.value)
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

    def toggle_filter(self, i):
        def toggle_on_change():
            self.filters[i][0].enabled = not self.filters[i][0].enabled

        return toggle_on_change

    def check_empty_layerlist(self):
        if len(self.viewer.layers) == 0:
            self.zarr_dir.value = ""
            self.select_well.enabled = False
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
            wells = (tbl["row"] + tbl["col"].astype(str)).sort_values()
            self.well.choices = wells
            self.well._default_choices = wells

    def go_to_well(self):
        row_min = alpha_to_numeric(self.df["row"].min())
        col_min = int(self.df["col"].min())
        matches = [
            re.match(r"([A-Z]+)(\d+)", well) for well in self.well.value
        ]
        wells = [(m.group(1), m.group(2)) for m in matches]
        dataset = 0
        level = 0

        for row_alpha, col_str in wells:
            row = alpha_to_numeric(row_alpha)
            col = int(col_str)
            roi_tbl = load_table(
                self.zarr_root, "well_ROI_table", f"{row_alpha}{col_str}"
            )
            with zarr.open(
                f"{self.zarr_root}/{row_alpha}/{col_str}/{dataset}"
            ) as metadata:
                img_scale = metadata.attrs["multiscales"][dataset]["datasets"][
                    level
                ]["coordinateTransformations"][0]["scale"]

            r = convert_ROI_table_to_indices(roi_tbl, img_scale[-3:])[0]

            rec = np.array(
                [
                    [
                        (col - col_min) * (r[3] - r[2]) + r[2],
                        (row - row_min) * (r[5] - r[4]) + r[4],
                    ],
                    [
                        (col - col_min) * (r[3] - r[2]) + r[3],
                        (row - row_min) * (r[5] - r[4]) + r[5],
                    ],
                ]
            )

            for layer in self.viewer.layers:
                if type(layer) == napari.layers.Shapes:
                    self.viewer.layers.remove(layer)
            self.viewer.add_shapes(
                rec,
                shape_type="rectangle",
                edge_width=10,
                edge_color="white",
                face_color="transparent",
                name=f"{row_alpha}{col_str}",
            )
            self.viewer.camera.center = rec.mean(axis=0)
            self.viewer.camera.zoom = 0.25


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


def load_table(
    zarr_url: Union[str, Path],
    name: str,
    wells: Union[str, list[str]] = None,
    dataset: int = 0,
) -> ad.AnnData:
    """Load an Anndata table from a OME-ZARR URL

    Args:
        zarr_url: Path to the OME-ZARR
        name: Name of the table
        wells: A single well or a list of wells on a plate
        dataset: Index of the dataset

    Returns:
        An Anndata table with columns x/y/z_micrometer, len_x/y/z_micrometer and x/y_micrometer_original
    """
    wells = _validate_wells(wells, zarr_url)
    wells_str = [f"{w[0]}{w[1]}" for w in wells]
    tbl = list()
    while wells:
        row_alpha, col = wells.pop()
        try:
            tbl.append(
                ad.read_zarr(
                    f"{zarr_url}/{row_alpha}/{col}/{dataset}/tables/{name}"
                )
            )
        except PathNotFoundError:
            logger.info(
                f'The table "{name}" was not found in well {row_alpha}{col}'
            )
    if tbl:
        if len(wells_str) > 1:
            return ad.concat(tbl, keys=wells_str, index_unique="-")
        else:
            return ad.concat(tbl)


def _validate_wells(
    wells: Union[str, list[str]], zarr_url: Union[str, Path]
) -> set[tuple[str, int]]:
    """Check that wells are formatted correctly

    Args:
        wells: A single well or a list of wells on a plate
        zarr_url: Path to the OME-ZARR

    Returns:
        A unique set of alphanumeric tuples describing the wells
    """
    if wells is not None:
        wells = [wells] if isinstance(wells, str) else wells
        matches = [re.match(r"([A-Z]+)(\d+)", well) for well in wells]
        wells = set([(m.group(1), m.group(2)) for m in matches])
    else:
        with zarr.open(zarr_url) as metadata:
            matches = [
                re.match(r"([A-Z]+)/(\d+)", well["path"])
                for well in metadata.attrs["plate"]["wells"]
            ]
            wells = set([(m.group(1), m.group(2)) for m in matches])
    return wells
