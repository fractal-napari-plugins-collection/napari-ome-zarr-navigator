import logging
from pathlib import Path

import zarr
from fractal_tasks_core.ngff.zarr_utils import load_NgffImageMeta

logger = logging.getLogger(__name__)


class OMEZarrImage:
    """
    Class that represents a given OME-Zarr image

    Helper class provides read access & info about the Zarr image

    """

    def __init__(self, zarr_url) -> None:
        self.zarr_url = zarr_url
        if not self.is_zarr_dataset(zarr_url):
            raise ValueError

        self.image_meta = load_NgffImageMeta(zarr_url)
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
