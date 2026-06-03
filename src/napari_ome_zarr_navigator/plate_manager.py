import logging

import pandas as pd
from napari.utils.notifications import show_info
from ngio import open_ome_zarr_plate
from ngio.utils import (
    NgioFileNotFoundError,
    NgioValidationError,
    fractal_fsspec_store,
)

from napari_ome_zarr_navigator.util import ZarrSelector
from napari_ome_zarr_navigator.well_utils import well_sort_key

logger = logging.getLogger(__name__)


class PlateManager:
    """Owns the zarr plate connection and well enumeration.

    Decoupled from the napari viewer and filter logic.
    """

    def __init__(self, zarr_selector: ZarrSelector) -> None:
        self._zarr_selector = zarr_selector
        self._zarr_plate = None
        self._plate_store = None

    @property
    def zarr_plate(self):
        return self._zarr_plate

    @property
    def plate_store(self):
        return self._plate_store

    @property
    def is_plate(self) -> bool:
        return self._zarr_plate is not None

    def clear(self) -> None:
        """Reset plate state (e.g. when the URL is cleared)."""
        self._zarr_plate = None
        self._plate_store = None

    def open_zarr_plate(self) -> None:
        """Open the plate from the current ZarrSelector URL."""
        if self._zarr_selector._source_selector.value == "File":
            self._plate_store = self._zarr_selector.url
        else:
            self._plate_store = fractal_fsspec_store(
                self._zarr_selector.url,
                fractal_token=self._zarr_selector.token,
            )
        try:
            self._zarr_plate = open_ome_zarr_plate(
                self._plate_store,
                cache=True,
                mode="r",  # type: ignore[arg-type]
            )
        except NgioFileNotFoundError:
            self._zarr_plate = None
        except NgioValidationError as e:
            self._zarr_plate = None
            msg = (
                "No valid Zarr plate found at the provided URL. Verify the "
                "URL to ensure it points to the root of the plate or "
                f"check the validation error: \n {e}"
            )
            logger.info(msg)
            show_info(msg)

    def get_plate_wells(self, filters=None) -> tuple[list[str], pd.DataFrame]:
        """Return sorted well names and a row/col DataFrame.

        Args:
            filters: optional set of (row, int_col) tuples to restrict results
        """
        assert self._zarr_plate is not None
        wells = []
        dfs = []
        if filters is not None:
            for well_path in self._zarr_plate.wells_paths():
                row, col = well_path.split("/")
                col_int = int(col)
                if (row, col_int) in filters:
                    wells.append(f"{row}{col}")
                    dfs.append(pd.DataFrame({"row": [row], "col": [col]}))
        else:
            for well_path in self._zarr_plate.wells_paths():
                row, col = well_path.split("/")
                wells.append(f"{row}{col}")
                dfs.append(pd.DataFrame({"row": [row], "col": [col]}))
        wells_str = sorted(wells, key=well_sort_key)
        if not dfs:
            return wells_str, pd.DataFrame(
                {"row": pd.Series(dtype=str), "col": pd.Series(dtype=str)}
            )
        return wells_str, pd.concat(dfs, ignore_index=True)
