from typing import TYPE_CHECKING

import pandas as pd
from pathlib import Path
from typing import Union
from zarr.errors import PathNotFoundError
import anndata as ad
import zarr
import re


from napari_ome_zarr_navigator.util import alpha_to_numeric, numeric_to_alpha


from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FileEdit,
    PushButton,
    RadioButtons,
    Select,
    SpinBox,
)

if TYPE_CHECKING:
    import napari


class ImgBrowser(Container):
    # TODO: if no condition table is available in the OME-zarr only load the col/rows into the Select widget, otherwise load the filters
    def __init__(self, viewer: "napari.viewer.Viewer"):
        self.viewer = viewer
        self.zarr_dir = FileEdit(
            label="OME-ZARR URL", mode="d", filter="*.zarr"
        )
        self.selection_mode = RadioButtons(
            choices=["Drug/Conc", "Row/Col"],
            label="Mode",
            value="Drug/Conc",
        )

        self.drug = ComboBox(
            label="Drug",
        )
        self.concentration = ComboBox(
            label="Conc. (nM)",
        )
        self.row_alpha = ComboBox(
            label="Row",
        )
        self.col = ComboBox(label="Column")
        self.well = ComboBox(
            label="Well",
        )
        self.display_drugs = Select(
            label="content", enabled=True, allow_multiple=False
        )
        self.select_well = PushButton(text="Select well", enabled=False)
        self.new_layers = CheckBox(label="Add Image/Labels as new layer(s)")
        self.drug_layout = pd.DataFrame(
            {"row": [], "col": [], "drug": [], "concentration": [], "unit": []}
        )

        self.well = Select(label="content", enabled=True, allow_multiple=False)

        # super().__init__(
        #     widgets=[
        #         self.zarr_dir,
        #         self.drug,
        #         self.concentration,
        #         self.row_alpha,
        #         self.col,
        #         self.well,
        #         self.display_drugs,
        #         self.new_layers,
        #         self.select_well,
        #     ]
        # )

        super().__init__(
            widgets=[self.zarr_dir, self.row_alpha, self.col, self.well],
        )
        self.drug.changed.connect(self.set_doses)
        # self.select_well.clicked.connect(self.load_zarr)
        self.zarr_dir.changed.connect(self.select_layer)
        self.row_alpha.changed.connect(self.set_drugs)
        self.col.changed.connect(self.set_drugs)
        self.drug.changed.connect(self.set_row_col)
        self.concentration.changed.connect(self.set_row_col)
        self.viewer.layers.events.removed.connect(self.check_empty_layerlist)
        self.viewer.layers.selection.events.changed.connect(self.get_zarr_url)

        self.zarr_dir.changed.connect(self.initialize_filters)

    def initialize_filters(self):
        zarr_dict = parse_zarr_url(self.zarr_dir.value)
        self.zarr_root = zarr_dict["root"]
        try:
            self.df = (
                load_table(
                    self.zarr_root,
                    "condition",
                    zarr_dict["well"],
                )
                .to_df()
                .astype({"col": "int"})
            )
        except PathNotFoundError:
            napari.utils.notifications.show_warning(
                f"Some wells don't have a conditions table associated."
            )
        else:
            filter_names = self.df.columns.drop(["row", "col"])
            self.filters = Container(
                widgets=[
                    Container(
                        widgets=[
                            ComboBox(
                                choices=self.df[filter].unique(), enabled=False
                            ),
                            CheckBox(label=filter, value=False),
                        ],
                        layout="horizontal",
                    )
                    for filter in filter_names
                ],
                layout="vertical",
            )
            # print(self.filters[0][0])
            self.viewer.window.add_dock_widget(
                widget=self.filters, name="Filters"
            )

        for i in range(self.df.shape[1]):
            self.filters[i][1].changed.connect(self.toggle_filter(i))

    def toggle_filter(self, i):
        def toggle_on_change():
            self.filters[i][0].enabled = not self.filters[i][0].enabled

        return toggle_on_change

    def check_empty_layerlist(self):
        if len(self.viewer.layers) == 0:
            self.zarr_dir.value = ""

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

    def select_layer(self):
        zarr_dict = parse_zarr_url(self.zarr_dir.value)
        self.zarr_root = zarr_dict["root"]
        if zarr_dict["root"]:
            try:
                self.drug_layout = (
                    load_table(
                        self.zarr_root,
                        "condition",
                        zarr_dict["well"],
                    )
                    .to_df()
                    .astype({"col": "int"})
                )
            except PathNotFoundError:
                napari.utils.notifications.show_warning(
                    f"Some wells don't have a conditions table associated."
                )
            else:
                self.row_alpha.choices = self.drug_layout["row"].unique()
                self.row_alpha._default_choices = self.drug_layout[
                    "row"
                ].unique()
                self.col.choices = self.drug_layout["col"].unique()
                self.col._default_choices = self.drug_layout["col"].unique()
                self.drug.choices = self.drug_layout["drug"].unique()
                self.drug._default_choices = self.drug_layout["drug"].unique()
                if self.zarr_root == self.zarr_dir.value:
                    self.select_well.enabled = True
                    self.new_layers.enabled = True
                else:
                    self.new_layers.enabled = False
                    self.select_well.enabled = False
        else:
            self.row_alpha.choices = []
            self.row_alpha._default_choices = []
            self.col.choices = []
            self.col._default_choices = []
            self.drug.choices = []
            self.drug._default_choices = []
            self.set_doses()
            self.set_row_col()
            self.set_drugs()
            self.select_well.enabled = False

    def set_doses(self):
        if not self.drug_layout.empty:
            concentrations = self.drug_layout.query(
                "drug == @self.drug.value"
            )["concentration"].unique()
            self.concentration.choices = concentrations
            self.concentration._default_choices = concentrations

    def set_row_col(self):
        if not self.drug_layout.empty:
            tbl = self.drug_layout.query(
                "(drug == @self.drug.value) & (concentration == @self.concentration.value)"
            )
            well = tbl["row"] + tbl["col"].astype(str)
            self.well.choices = well
            self.well._default_choices = well

    def set_drugs(self):
        if not self.drug_layout.empty:
            tbl = self.drug_layout.query(
                "(row == @self.row_alpha.value) & (col == @self.col.value)"
            )
            drug_concentrations = (
                tbl["drug"] + " (" + tbl["concentration"].astype(str) + " nM)"
            )
            self.display_drugs.choices = drug_concentrations
            self.display_drugs._default_choices = drug_concentrations

    def toggle_selection_mode(self):
        if self.selection_mode.value == "Row/Col":
            self.row_alpha.visible = True
            self.col.visible = True
            self.drug.visible = False
            self.concentration.visible = False
            self.well.visible = False
            self.display_drugs.visible = True
        else:
            self.row_alpha.visible = False
            self.col.visible = False
            self.drug.visible = True
            self.concentration.visible = True
            self.display_drugs.visible = False
            self.well.visible = True

    # def load_zarr(self):
    #     if self.selection_mode.value == "Row/Col":
    #         well = f"{self.row_alpha.value}{self.col.value}"
    #         row_alpha = self.row_alpha.value
    #         row = operio.util.alpha_to_numeric(row_alpha)
    #         col = self.col.value
    #     else:
    #         well = self.well.value
    #         m = re.match(r"(\w+)(\d+)", well)
    #         row_alpha = m.group(1)
    #         row = operio.util.alpha_to_numeric(row_alpha)
    #         col = int(m.group(2))
    #     roi_url, roi_idx = operio.io.get_roi(
    #         self.zarr_root,
    #         row_alpha,
    #         row,
    #         "well_ROI_table",
    #         level=0,
    #     )
    #     if self.new_layers.value:
    #         img = operio.io.load_intensity_roi(roi_url, roi_idx)
    #         labels = operio.io.load_label_roi(roi_url, roi_idx)
    #         self.viewer.add_image(img, name=f"{row_alpha}{col}")
    #         label_layer = self.viewer.add_labels(
    #             labels, name=f"{row_alpha}{col}_label"
    #         )
    #         operio.io.napari.add_features_to_labels(
    #             label_layer, self.zarr_root, row_alpha, col
    #         )
    #     else:
    #         self.draw_well(roi_idx, row, col, well)

    # def draw_well(self, roi_idx: pd.DataFrame, row: int, col: int, well: str):
    #     r = roi_idx.iloc[0]
    #     row_min = operio.util.alpha_to_numeric(self.row_alpha.choices[0])
    #     col_min = self.col.min

    #     rec = np.array(
    #         [
    #             [
    #                 (row - row_min) * (r.e_x - r.s_x) + r.s_x,
    #                 (col - col_min) * (r.e_y - r.s_y) + r.s_y,
    #             ],
    #             [
    #                 (row - row_min) * (r.e_x - r.s_x) + r.e_x,
    #                 (col - col_min) * (r.e_y - r.s_y) + r.e_y,
    #             ],
    #         ]
    #     )

    #     for layer in self.viewer.layers:
    #         if type(layer) == napari.layers.Shapes:
    #             self.viewer.layers.remove(layer)
    #     self.viewer.add_shapes(
    #         rec,
    #         shape_type="rectangle",
    #         edge_width=5,
    #         edge_color="white",
    #         face_color="transparent",
    #         name=well,
    #     )
    #     self.viewer.camera.center = rec.mean(axis=0)
    #     self.viewer.camera.zoom = 0.25


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
    """Load an Anndata table from a zarr url

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
            print(f"No condition table was found for well {row_alpha}{col}")
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
        # wells = set([(m.group(1), int(m.group(2))) for m in matches])
        wells = set([(m.group(1), m.group(2)) for m in matches])
    else:
        with zarr.open(zarr_url) as metadata:
            matches = [
                re.match(r"([A-Z]+)/(\d+)", well["path"])
                for well in metadata.attrs["plate"]["wells"]
            ]
            # wells = set([(m.group(1), int(m.group(2))) for m in matches])
            wells = set([(m.group(1), m.group(2)) for m in matches])
    return wells
