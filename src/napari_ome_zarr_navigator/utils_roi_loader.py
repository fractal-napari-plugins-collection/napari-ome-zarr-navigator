import logging
from functools import lru_cache
from pathlib import Path

import anndata as ad
import zarr
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info


@lru_cache(maxsize=16)
def get_zattrs(zarr_url):
    with zarr.open(zarr_url) as zarr_attrs:
        return dict(zarr_attrs.attrs)


@lru_cache(maxsize=16)
def read_table(zarr_url: str, roi_table: str):
    with zarr.open(zarr_url, mode="r").tables[roi_table] as table:
        return ad.read_zarr(table)


def get_table_list(zarr_url, table_type: str = None, strict: bool = False):
    """
    Find the list of tables in the Zarr file

    Optionally match a table type and only return the names of those tables.

    Params:
        zarr_url (str): The path to the zarr file (e.g. "path/to/file.zarr/B/03/0")
        table_type (str): The type of table to look for. Special handling for
            "ROIs" => matches both "roi_table" & "masking_roi_table".
        strict (bool): If True, only return tables that have a type attribute.
            If False, also include tables without a type attribute.
    """
    table_folder = Path(zarr_url) / "tables"
    if not table_folder.exists():
        return []
    table_meta_dict = get_zattrs(table_folder)
    table_list = []

    if not table_type:
        return table_meta_dict["tables"]

    for table_name in table_meta_dict["tables"]:
        table_attrs = get_zattrs(table_folder / table_name)
        if "type" in table_attrs:
            if table_type == "ROIs":
                roi_table_types = ["roi_table", "masking_roi_table"]
                if table_attrs["type"] in roi_table_types:
                    table_list.append(table_name)
            elif table_attrs["type"] == table_type:
                table_list.append(table_name)
        else:
            # If there are tables without types, let the users choose from all
            # tables
            logging.warning(f"Table {table_name} had no type attribute.")
            if not strict:
                table_list.append(table_name)
    return table_list


@thread_worker
def threaded_get_table_list(zarr_url, table_type: str = None, strict=False):
    return get_table_list(
        zarr_url=zarr_url,
        table_type=table_type,
        strict=strict,
    )


class NapariHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        show_info(log_entry)
