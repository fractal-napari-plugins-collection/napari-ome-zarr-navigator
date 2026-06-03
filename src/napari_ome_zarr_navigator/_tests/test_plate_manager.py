"""Unit tests for PlateManager — no napari, no Qt, no network required."""

from unittest.mock import MagicMock

from napari_ome_zarr_navigator.plate_manager import PlateManager


def make_plate_manager(wells_paths: list[str]) -> PlateManager:
    """Build a PlateManager with a mocked zarr_plate.

    wells_paths: list of "ROW/COL" strings, e.g. ["B/03", "C/04"]
    """
    pm = PlateManager.__new__(PlateManager)
    pm._zarr_selector = MagicMock()
    pm._zarr_plate = MagicMock()
    pm._zarr_plate.wells_paths.return_value = wells_paths
    pm._plate_store = None
    return pm


# ---------------------------------------------------------------------------
# get_plate_wells — no filter
# ---------------------------------------------------------------------------


class TestGetPlateWellsNoFilter:
    def test_returns_all_wells(self):
        pm = make_plate_manager(["B/03", "C/04", "B/05"])
        wells, _ = pm.get_plate_wells()
        assert set(wells) == {"B03", "C04", "B05"}

    def test_returns_sorted_wells(self):
        pm = make_plate_manager(["C/04", "B/05", "B/03"])
        wells, _ = pm.get_plate_wells()
        assert wells == ["B03", "B05", "C04"]

    def test_returns_dataframe_with_row_col(self):
        pm = make_plate_manager(["B/03"])
        _, df = pm.get_plate_wells()
        assert "row" in df.columns
        assert "col" in df.columns
        assert df["row"].iloc[0] == "B"
        assert df["col"].iloc[0] == "03"


# ---------------------------------------------------------------------------
# get_plate_wells — with filter
# ---------------------------------------------------------------------------


class TestGetPlateWellsWithFilter:
    def test_subset_filter_returns_only_matching_wells(self):
        pm = make_plate_manager(["B/03", "C/04", "B/05"])
        wells, _ = pm.get_plate_wells(filters={("B", 3)})
        assert wells == ["B03"]

    def test_filter_matching_multiple_wells(self):
        pm = make_plate_manager(["B/03", "B/05", "C/04"])
        wells, _ = pm.get_plate_wells(filters={("B", 3), ("B", 5)})
        assert wells == ["B03", "B05"]

    def test_empty_filter_set_returns_empty_list(self):
        pm = make_plate_manager(["B/03", "C/04"])
        wells, df = pm.get_plate_wells(filters=set())
        assert wells == []
        assert df.empty

    def test_filter_no_matches_returns_empty(self):
        pm = make_plate_manager(["B/03", "C/04"])
        wells, df = pm.get_plate_wells(filters={("Z", 99)})
        assert wells == []
        assert df.empty

    def test_filtered_result_is_sorted(self):
        pm = make_plate_manager(["C/04", "B/03", "B/05"])
        wells, _ = pm.get_plate_wells(filters={("C", 4), ("B", 3)})
        assert wells == ["B03", "C04"]


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_resets_zarr_plate(self):
        pm = make_plate_manager(["B/03"])
        assert pm.zarr_plate is not None
        pm.clear()
        assert pm.zarr_plate is None

    def test_clear_resets_plate_store(self):
        pm = make_plate_manager(["B/03"])
        pm._plate_store = "some/path"
        pm.clear()
        assert pm.plate_store is None

    def test_is_plate_false_after_clear(self):
        pm = make_plate_manager(["B/03"])
        pm.clear()
        assert pm.is_plate is False
