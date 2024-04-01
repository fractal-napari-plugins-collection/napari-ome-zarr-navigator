import logging
from functools import lru_cache
from pathlib import Path

import anndata as ad
import dask.array as da
import numpy as np
import zarr
from fractal_tasks_core.ngff.specs import Multiscale
from fractal_tasks_core.ngff.zarr_utils import load_NgffImageMeta
from fractal_tasks_core.roi import convert_ROI_table_to_indices
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class OMEZarrImage:
    """
    Class that represents a given OME-Zarr image.

    Helper class provides read access & info about the Zarr image. One of the
    complexities to be aware of: Some information, like the ROI table, is
    specific to the whole Zarr image. But the image metadata and the label
    metadata don't need to match (will have different multiscales).

    """

    def __init__(self, zarr_url) -> None:
        self.zarr_url = zarr_url
        if not self.is_zarr_dataset(zarr_url):
            raise ValueError
        try:
            self.image_meta = load_NgffImageMeta(zarr_url)
        except ValidationError as e:
            raise ValueError(
                "The provided Zarr is not a valid OME-Zarr image. Loading its"
                "metadata triggered the following error ValidationError: "
                f"{e.json()}"
            ) from e
        with zarr.open(zarr_url, mode="r") as zarr_group:
            self.zarr_subgroups = list(zarr_group.group_keys())

    def get_channel_list(self):
        channel_names = []
        for channel in self.image_meta.omero.channels:
            channel_names.append(channel.label)
        return channel_names

    def get_pyramid_levels(self):
        pyramid_levels = []
        for ds in self.image_meta.multiscale.datasets:
            pyramid_levels.append(ds.path)
        return pyramid_levels

    def get_labels_list(self):
        if "labels" not in self.zarr_subgroups:
            return [""]
        with zarr.open(self.zarr_url, mode="r") as zarr_group:
            return list(zarr_group.labels.group_keys())

    def get_tables_list(self, table_type: str = None, strict: bool = False):
        """
        Find the list of tables in the Zarr file

        Optionally match a table type and only return the names of those tables.

        Params:
            table_type (str): The type of table to look for. Special handling for
                "ROIs" => matches both "roi_table" & "masking_roi_table".
            strict (bool): If True, only return tables that have a type attribute.
                If False, also include tables without a type attribute.
        """
        if "tables" not in self.zarr_subgroups:
            return [""]
        with zarr.open(self.zarr_url, mode="r") as zarr_group:
            all_tables = list(zarr_group.tables.group_keys())

        if not table_type:
            return all_tables
        else:
            return self.filter_tables_by_type(all_tables, table_type, strict)

    def filter_tables_by_type(
        self, all_tables, table_type: str = None, strict: bool = False
    ):
        tables_list = []
        for table_name in all_tables:
            with zarr.open(self.zarr_url, mode="r").tables[
                table_name
            ] as table:
                table_attrs = table.attrs.asdict()
                if "type" in table_attrs:
                    if table_type == "ROIs":
                        roi_table_types = ["roi_table", "masking_roi_table"]
                        if table_attrs["type"] in roi_table_types:
                            tables_list.append(table_name)
                    elif table_attrs["type"] == table_type:
                        tables_list.append(table_name)
                else:
                    # If there are tables without types, let the users choose
                    # from all tables
                    logger.warning(
                        f"Table {table_name} had no type attribute."
                    )
                    if not strict:
                        tables_list.append(table_name)
        return tables_list

    @staticmethod
    def is_zarr_dataset(path):
        """
        Check if the given path contains a Zarr dataset.

        Parameters:
        - path: str, the path to the dataset.

        Returns:
        - bool, True if the path contains a Zarr dataset, False otherwise.
        """
        if path == Path("."):
            # Handle the scenario where no path was set
            return False

        try:
            # Attempt to open the path as a Zarr dataset
            zarr.open(path, mode="r")
            return True
        except ValueError:
            # ValueError indicates that it is not a Zarr dataset
            print(f"No Zarr found at {path}")
            return False
        except PermissionError:
            # Handle the case where the path exists but there's no read permission
            print(f"Permission denied when accessing the path: {path}")
            return False
        except FileNotFoundError:
            # Handle the case where the path does not exist
            print(f"Path does not exist: {path}")
            return False

    @staticmethod
    @lru_cache(maxsize=16)
    def read_table(zarr_url, roi_table: str):
        with zarr.open(zarr_url, mode="r").tables[roi_table] as table:
            return ad.read_zarr(table)

    @staticmethod
    def _get_image_scale_zyx(level: str, multiscales: Multiscale):
        def get_scale_and_index_for_path(level, dataset_list):
            """
            Returns the scale & the index of the level.
            """
            for i, dataset in enumerate(dataset_list):
                if dataset.path == level:
                    if len(dataset.coordinateTransformations) > 1:
                        raise NotImplementedError(
                            "OMEZarrImage has only been implemented for Zarrs "
                            "with a single coordinate transformation"
                        )
                    # Only handles datasets with a single scale
                    return dataset.coordinateTransformations[0].scale, i
            raise ValueError(f"Level {level} not available in {dataset_list}.")

        if len(multiscales) > 1:
            raise NotImplementedError(
                "OMEZarrImage has only been implemented for Zarrs with a "
                "single multiscales"
            )

        scale, level_index = get_scale_and_index_for_path(
            str(level), multiscales[0].datasets
        )
        if len(scale) < 3:
            raise NotImplementedError(
                "OMEZarrImage has only been implemented for OME-Zarrs that "
                "contain the zyx dimensions last. This Zarr had: "
                f"{multiscales[0].axes}"
            )

        if (
            multiscales[0].axes[-3].name == "z"
            and multiscales[0].axes[-2].name == "y"
            and multiscales[0].axes[-1].name == "x"
        ):
            return scale[-3:], level_index
        elif (
            multiscales[0].axes[-3].name == "c"
            and multiscales[0].axes[-2].name == "y"
            and multiscales[0].axes[-1].name == "x"
        ):
            logger.warning(
                "Processing a cyx image. This has not been tested well."
            )
            return scale[-3:], level_index
        else:
            raise NotImplementedError(
                "OMEZarrImage has only been implemented for OME-Zarrs that "
                "contain the zyx dimensions last. This Zarr had: "
                f"{multiscales[0].axes}"
            )

    def get_roi_indices(
        self,
        roi_table: str,
        level_index: int,
        pixel_size_zyx: tuple,
    ):
        """
        Get the indices of a specific ROI from a ROI table.

        Args:
        - roi_table: Name of the ROI table to load
        - level: Resolution level to load
        - pixel_size_zyx: pixel size as a tuple z, y, x

        """
        # Get the ROI table
        roi_an = self.read_table(self.zarr_url, roi_table)

        # TODO: Switch to a more robust way of loading indices when the
        # dimensionality of the image can vary. This only works for 3D images
        # (all Yokogawa images are saved as 3D images) and
        # by accident for 2D MD images (if they are multichannel)
        # See issue 420 on fractal-tasks-core
        indices_list = convert_ROI_table_to_indices(
            roi_an,
            full_res_pxl_sizes_zyx=pixel_size_zyx,
            level=level_index,
        )

        return indices_list

    def load_zarr_array_index_based(
        self,
        zarr_url: str,
        roi_table: str,
        roi_index: int,
        channel_index: int,
        multiscales: Multiscale,
        level_path: str = "0",
    ):
        """
        Load an intensity ROI based on indices

        Args:
        - roi_table: Name of the ROI table to load
        - roi_index: Index
        - level: Resolution level to load
        """
        img_scale, level_index = self._get_image_scale_zyx(
            level_path,
            multiscales=multiscales,
        )
        s_z, e_z, s_y, e_y, s_x, e_x = self.get_roi_indices(
            roi_table,
            level_index,
            pixel_size_zyx=img_scale,
        )[roi_index][:]

        # Load data
        # TODO: make this more axes-robust (e.g. able to handle 5D axes)
        img_data_zyx = da.from_zarr(f"{zarr_url}/{level_path}")[channel_index]

        if len(img_data_zyx.shape) == 2:
            img_roi = img_data_zyx[s_y:e_y, s_x:e_x]
            img_scale = img_scale[1:]
        else:
            img_roi = img_data_zyx[s_z:e_z, s_y:e_y, s_x:e_x]

        return np.array(img_roi), img_scale

    def load_intensity_roi(
        self,
        roi_table: str,
        roi_name: str,
        channel: str,
        level_path: str = "0",
    ):
        roi_an = self.read_table(self.zarr_url, roi_table)
        roi_index = roi_an.obs.index.get_loc(roi_name)

        # This assumes the order of the omero channels will match the order
        # of the channels in the Zarr array.
        channels = self.get_channel_list()
        # This triggers a ValueError if the channel is not found in the list
        channel_index = channels.index(channel)

        return self.load_zarr_array_index_based(
            zarr_url=self.zarr_url,
            roi_table=roi_table,
            roi_index=roi_index,
            channel_index=channel_index,
            multiscales=self.image_meta.multiscales,
            level_path=level_path,
        )
