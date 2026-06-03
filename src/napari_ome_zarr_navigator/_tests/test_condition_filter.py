"""Unit tests for ConditionTableFilter — requires Qt (qtbot), no network."""

from unittest.mock import MagicMock

import pandas as pd

from napari_ome_zarr_navigator.condition_filter import ConditionTableFilter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A minimal condition table with two filterable columns.
# col is int to match what get_plate_wells returns after int() conversion.
CONDITION_DF = pd.DataFrame(
    {
        "row": ["B", "B", "C"],
        "col": [3, 3, 4],
        "drug": ["drugA", "drugB", "drugA"],
        "dose": [1, 2, 1],
    }
)

# Wells returned by the mock for each (row, int_col) combo
WELL_MAP = {("B", 3): "B03", ("C", 4): "C04"}
ALL_WELLS = ["B03", "C04"]


def make_condition_filter() -> ConditionTableFilter:
    """
    Build a ConditionTableFilter whose PlateManager mock returns wells
    from WELL_MAP when filters match, or ALL_WELLS when filters=None.
    """
    mock_pm = MagicMock()

    def _get_plate_wells(filters=None):
        if filters is None:
            return (
                ALL_WELLS,
                pd.DataFrame({"row": ["B", "C"], "col": ["03", "04"]}),
            )
        matched = sorted(WELL_MAP[key] for key in filters if key in WELL_MAP)
        rows = [w[0] for w in matched]
        cols = [w[1:] for w in matched]
        return matched, pd.DataFrame({"row": rows, "col": cols})

    mock_pm.get_plate_wells.side_effect = _get_plate_wells
    return ConditionTableFilter(mock_pm)


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_emits_all_wells(self, qtbot):
        cf = make_condition_filter()
        received = []
        cf.signals.wells_changed.connect(received.append)

        cf.reset()

        assert received == [ALL_WELLS]

    def test_reset_clears_filter_state(self, qtbot):
        cf = make_condition_filter()
        cf.setup_filters(CONDITION_DF)
        cf.reset()

        assert cf.df is None
        assert cf._filter_names is None

    def test_reset_hides_filter_container(self, qtbot):
        cf = make_condition_filter()
        cf.setup_filters(CONDITION_DF)
        cf.reset()

        assert not cf.filter_container.visible


# ---------------------------------------------------------------------------
# _filter_df() — no active filters
# ---------------------------------------------------------------------------


class TestFilterDfNoActiveFilters:
    def test_emits_all_wells_when_no_checkbox_enabled(self, qtbot):
        cf = make_condition_filter()
        cf.setup_filters(CONDITION_DF)

        received = []
        cf.signals.wells_changed.connect(received.append)
        cf._filter_df()

        assert received == [ALL_WELLS]


# ---------------------------------------------------------------------------
# _filter_df() — single active filter
# ---------------------------------------------------------------------------


class TestFilterDfSingleFilter:
    def _enable_filter(self, cf, filter_index, value):
        """Enable the checkbox for filter_index and set the combo to value."""
        # filter_container[0] is condition_name_selector;
        # filter widgets start at index 1
        fw = cf.filter_container[filter_index + 1]
        combo, check = fw[0], fw[1]
        check.value = True
        combo.value = value

    def test_single_filter_drug_a(self, qtbot):
        cf = make_condition_filter()
        cf.setup_filters(CONDITION_DF)
        self._enable_filter(cf, 0, "drugA")  # drug is first filter column

        received = []
        cf.signals.wells_changed.connect(received.append)
        cf._filter_df()

        # drugA rows: B/3 and C/4 → both wells
        assert received == [["B03", "C04"]]

    def test_single_filter_drug_b(self, qtbot):
        cf = make_condition_filter()
        cf.setup_filters(CONDITION_DF)
        self._enable_filter(cf, 0, "drugB")  # only B/3

        received = []
        cf.signals.wells_changed.connect(received.append)
        cf._filter_df()

        assert received == [["B03"]]


# ---------------------------------------------------------------------------
# _filter_df() — two active filters (AND logic)
# ---------------------------------------------------------------------------


class TestFilterDfTwoFilters:
    def test_and_logic_narrows_results(self, qtbot):
        cf = make_condition_filter()
        cf.setup_filters(CONDITION_DF)

        # drug = drugA AND dose = 1 → B/3 and C/4 both qualify
        drug_fw = cf.filter_container[1]
        dose_fw = cf.filter_container[2]
        drug_fw[1].value = True
        drug_fw[0].value = "drugA"
        dose_fw[1].value = True
        dose_fw[0].value = 1

        received = []
        cf.signals.wells_changed.connect(received.append)
        cf._filter_df()

        assert received == [["B03", "C04"]]

    def test_and_logic_excludes_partial_match(self, qtbot):
        cf = make_condition_filter()
        cf.setup_filters(CONDITION_DF)

        # drug = drugA AND dose = 2 → no row has both (drugA rows have dose=1)
        drug_fw = cf.filter_container[1]
        dose_fw = cf.filter_container[2]
        drug_fw[1].value = True
        drug_fw[0].value = "drugA"
        dose_fw[1].value = True
        dose_fw[0].value = 2

        received = []
        cf.signals.wells_changed.connect(received.append)
        cf._filter_df()

        assert received == [[]]


# ---------------------------------------------------------------------------
# _filter_df() — empty result (regression: was crashing with ValueError)
# ---------------------------------------------------------------------------


class TestFilterDfEmptyResult:
    def test_no_matching_rows_emits_empty_list(self, qtbot):
        cf = make_condition_filter()
        cf.setup_filters(CONDITION_DF)

        # Enable drug filter with a value that matches no rows
        drug_fw = cf.filter_container[1]
        drug_fw[1].value = True
        # Temporarily add a non-existent value by overriding choices
        drug_fw[0].choices = ["drugA", "drugB", "drugX"]
        drug_fw[0].value = "drugX"

        received = []
        cf.signals.wells_changed.connect(received.append)
        cf._filter_df()  # must not raise ValueError

        assert received == [[]]


# ---------------------------------------------------------------------------
# Regression: deactivating a filter must restore ALL plate wells, not just
# the wells that appear in the condition table.
#
# Setup: plate has 3 wells (B03, C04, D05).  The condition table covers only
# B03 and C04 (D05 has no assigned conditions).  When every checkbox is off,
# `_filter_df` must return all 3 plate wells — it should call
# get_plate_wells(filters=None) rather than passing the condition-table
# row-set as the filter.
# ---------------------------------------------------------------------------

# Two condition columns, but only 2 of the 3 plate wells have conditions.
PARTIAL_CONDITION_DF = pd.DataFrame(
    {
        "row": ["B", "C"],
        "col": ["03", "04"],
        "timepoint": ["day 0", "day 6"],
        "drug": ["drugA", "drugB"],
    }
)
PARTIAL_WELL_MAP = {("B", 3): "B03", ("C", 4): "C04", ("D", 5): "D05"}
ALL_PLATE_WELLS = ["B03", "C04", "D05"]


def make_condition_filter_partial() -> ConditionTableFilter:
    """Plate has 3 wells; condition table only covers B03 and C04."""
    mock_pm = MagicMock()

    def _get_plate_wells(filters=None):
        if filters is None:
            return (
                ALL_PLATE_WELLS,
                pd.DataFrame({"row": ["B", "C", "D"], "col": ["03", "04", "05"]}),
            )
        matched = sorted(
            PARTIAL_WELL_MAP[key] for key in filters if key in PARTIAL_WELL_MAP
        )
        rows = [w[0] for w in matched]
        cols = [w[1:] for w in matched]
        return matched, pd.DataFrame({"row": rows, "col": cols})

    mock_pm.get_plate_wells.side_effect = _get_plate_wells
    return ConditionTableFilter(mock_pm)


class TestFilterDeactivationRestoresAllWells:
    def test_deactivating_single_filter_restores_all_plate_wells(self):
        """Disabling all checkboxes should emit ALL plate wells, not condition-table wells."""
        cf = make_condition_filter_partial()
        cf.setup_filters(PARTIAL_CONDITION_DF)

        # Enable timepoint filter "day 0" → only B03 has day 0
        timepoint_fw = cf.filter_container[1]
        timepoint_fw[1].value = True
        timepoint_fw[0].value = "day 0"

        received = []
        cf.signals.wells_changed.connect(received.append)

        # Deactivate the filter — should restore ALL 3 plate wells, not just
        # the 2 that appear in the condition table.
        timepoint_fw[1].value = False

        assert received[-1] == ALL_PLATE_WELLS

    def test_deactivating_via_filter_df_restores_all_plate_wells(self):
        """Direct _filter_df call with all checkboxes off returns all plate wells."""
        cf = make_condition_filter_partial()
        cf.setup_filters(PARTIAL_CONDITION_DF)

        received = []
        cf.signals.wells_changed.connect(received.append)

        cf._filter_df()  # all checkboxes are off by default

        assert received[-1] == ALL_PLATE_WELLS
