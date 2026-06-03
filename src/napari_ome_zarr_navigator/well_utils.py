"""Utility functions for well-name parsing and condition-table normalization."""

import re

import pandas as pd

# Single canonical pattern for well names like "A01", "Ba011", "B3"
WELL_PATTERN = re.compile(r"^([A-Za-z]+)(\d+)$")

# Pattern for matching well-name-prefixed layer names (e.g. "A01_labels")
WELL_LAYER_PATTERN = re.compile(r"^[A-Za-z]+\d+")


def parse_well_name(well: str) -> tuple[str, str]:
    """Parse a well name into its letter and digit parts.

    Args:
        well: Well name string, e.g. "A01" or "Ba011".

    Returns:
        Tuple of (letters, digits), e.g. ("A", "01").

    Raises:
        ValueError: If the well name doesn't match the expected pattern.
    """
    m = WELL_PATTERN.match(well)
    if not m:
        raise ValueError(f"Cannot parse well name: {well!r}")
    return m.group(1), m.group(2)


def well_sort_key(well: str) -> tuple[str, int]:
    """Return a sort key for a well name so wells sort by letter then number.

    Falls back to (well, 0) for non-standard names so sorting never raises.
    """
    m = WELL_PATTERN.match(well)
    if m:
        return m.group(1), int(m.group(2))
    return well, 0


def get_row_cols(well_list: "str | list[str]") -> list[tuple[str, str]]:
    """Parse a well or list of wells into (row_letters, col_digits) tuples.

    Args:
        well_list: A single well name string or a list of well name strings.

    Returns:
        List of (letters, digits) tuples, e.g. [("A", "01"), ("B", "02")].
    """
    if isinstance(well_list, str):
        well_list = [well_list]
    results = []
    for well in well_list:
        m = WELL_PATTERN.match(well)
        if m:
            results.append((m.group(1), m.group(2)))
    return results


def parse_condition_table(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a condition table so it always has "row" and "col" columns.

    Handles these input shapes:
    - Already has "row" and "col" columns → returned as-is (minus "well").
    - Has "column" instead of "col" → renamed to "col".
    - Has "well" column (e.g. "A01") → parsed into "row" + "col", then dropped.

    Args:
        df: Raw condition table DataFrame.

    Returns:
        DataFrame with at least "row" and "col" columns.
    """
    df = df.copy()

    if "column" in df.columns and "col" not in df.columns:
        df = df.rename(columns={"column": "col"})

    if "well" in df.columns:
        if "row" not in df.columns or "col" not in df.columns:
            parsed = df["well"].str.extract(WELL_PATTERN)
            if "row" not in df.columns:
                df["row"] = parsed[0]
            if "col" not in df.columns:
                df["col"] = parsed[1]
        df = df.drop(columns=["well"])

    return df
