#!/usr/bin/env python3
import string

import dask.array as da
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.ngff.zarr_utils import load_NgffPlateMeta


def alpha_to_numeric(alpha: str) -> int:
    """Return the position of a single character in the alphabet

    Args:
        alpha: Single alphabet character

    Returns:
        Integer position in the alphabet
    """
    return ord(alpha.upper()) - 64


def numeric_to_alpha(numeric: int, upper: bool = True) -> str:
    """Return the upper or lowercase character for a given position in the alphabet

    Args:
        numeric: Integer position in the alphabet

    Returns:
        Single alphabet character
    """
    if upper:
        string.ascii_uppercase[numeric - 1]
    else:
        string.ascii_lowercase[numeric - 1]


def calculate_well_positions(plate_url, row, col, is_plate=True):
    dataset = 0
    level = 0
    zarr_url = f"{plate_url}/{row}/{col}/{dataset}"
    shape = da.from_zarr(f"{zarr_url}/{level}").shape[-2:]
    image_meta = load_NgffImageMeta(zarr_url)
    scale = image_meta.get_pixel_sizes_zyx(level=level)[-2:]
    plate_meta = load_NgffPlateMeta(plate_url)
    rows = [x.name for x in plate_meta.plate.rows]
    cols = [x.name for x in plate_meta.plate.columns]

    row = rows.index(row)
    col = cols.index(col)

    if is_plate:
        top_left_corner = [
            row * scale[0] * shape[0],
            col * scale[1] * shape[1],
        ]
    else:
        top_left_corner = [0, 0]
    bottom_right_corner = [
        top_left_corner[0] + scale[0] * shape[0],
        top_left_corner[1] + scale[1] * shape[1],
    ]

    return top_left_corner, bottom_right_corner
