"""Unit tests for well_utils — no napari, no Qt, no network required."""

import pandas as pd
import pytest

from napari_ome_zarr_navigator.util import alpha_to_numeric, numeric_to_alpha
from napari_ome_zarr_navigator.well_utils import (
    get_row_cols,
    parse_condition_table,
    parse_well_name,
    well_sort_key,
)

# ---------------------------------------------------------------------------
# alpha_to_numeric / numeric_to_alpha round-trip
# ---------------------------------------------------------------------------


class TestAlphaNumeric:
    def test_alpha_to_numeric_uppercase(self):
        assert alpha_to_numeric("A") == 1
        assert alpha_to_numeric("Z") == 26

    def test_alpha_to_numeric_lowercase(self):
        assert alpha_to_numeric("a") == 1
        assert alpha_to_numeric("b") == 2

    def test_numeric_to_alpha_upper(self):
        assert numeric_to_alpha(1) == "A"
        assert numeric_to_alpha(26) == "Z"

    def test_numeric_to_alpha_lower(self):
        assert numeric_to_alpha(1, upper=False) == "a"
        assert numeric_to_alpha(26, upper=False) == "z"

    def test_round_trip(self):
        for i in range(1, 27):
            assert alpha_to_numeric(numeric_to_alpha(i)) == i


# ---------------------------------------------------------------------------
# parse_well_name
# ---------------------------------------------------------------------------


class TestParseWellName:
    def test_simple(self):
        assert parse_well_name("A01") == ("A", "01")

    def test_multi_letter(self):
        assert parse_well_name("Ba011") == ("Ba", "011")

    def test_no_padding(self):
        assert parse_well_name("B3") == ("B", "3")

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_well_name("invalid")

        with pytest.raises(ValueError):
            parse_well_name("123")


# ---------------------------------------------------------------------------
# well_sort_key
# ---------------------------------------------------------------------------


class TestWellSortKey:
    def test_returns_letter_int_tuple(self):
        assert well_sort_key("A01") == ("A", 1)
        assert well_sort_key("B03") == ("B", 3)

    def test_multi_letter(self):
        assert well_sort_key("Ba011") == ("Ba", 11)

    def test_fallback_for_invalid(self):
        assert well_sort_key("invalid") == ("invalid", 0)

    def test_sorting_order(self):
        wells = ["B02", "A10", "A01", "B01"]
        assert sorted(wells, key=well_sort_key) == ["A01", "A10", "B01", "B02"]


# ---------------------------------------------------------------------------
# get_row_cols
# ---------------------------------------------------------------------------


class TestGetRowCols:
    def test_single_string(self):
        assert get_row_cols("A01") == [("A", "01")]

    def test_list(self):
        assert get_row_cols(["A01", "B02"]) == [("A", "01"), ("B", "02")]

    def test_multi_letter(self):
        assert get_row_cols(["Ba011"]) == [("Ba", "011")]

    def test_invalid_entries_skipped(self):
        assert get_row_cols(["A01", "invalid", "B02"]) == [
            ("A", "01"),
            ("B", "02"),
        ]


# ---------------------------------------------------------------------------
# parse_condition_table
# ---------------------------------------------------------------------------


class TestParseConditionTable:
    def test_already_has_row_col(self):
        df = pd.DataFrame({"row": ["A"], "col": ["01"], "condition": ["ctrl"]})
        result = parse_condition_table(df)
        assert list(result.columns) == ["row", "col", "condition"]

    def test_renames_column_to_col(self):
        df = pd.DataFrame({"row": ["A"], "column": ["01"], "condition": ["ctrl"]})
        result = parse_condition_table(df)
        assert "col" in result.columns
        assert "column" not in result.columns

    def test_parses_well_column(self):
        df = pd.DataFrame({"well": ["A01", "B02"], "condition": ["ctrl", "kd"]})
        result = parse_condition_table(df)
        assert "well" not in result.columns
        assert list(result["row"]) == ["A", "B"]
        assert list(result["col"]) == ["01", "02"]

    def test_well_with_existing_row_col(self):
        df = pd.DataFrame(
            {
                "well": ["A01"],
                "row": ["A"],
                "col": ["01"],
                "condition": ["ctrl"],
            }
        )
        result = parse_condition_table(df)
        assert "well" not in result.columns
        assert result["row"].iloc[0] == "A"
        assert result["col"].iloc[0] == "01"

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"well": ["A01"], "condition": ["ctrl"]})
        original_cols = list(df.columns)
        parse_condition_table(df)
        assert list(df.columns) == original_cols
