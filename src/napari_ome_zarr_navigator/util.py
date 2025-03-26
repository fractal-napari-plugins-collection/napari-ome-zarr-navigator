#!/usr/bin/env python3
import logging
import string

# from fractal_tasks_core.ngff import load_NgffImageMeta
# from fractal_tasks_core.ngff.zarr_utils import load_NgffPlateMeta
from napari.utils.notifications import show_info
from ngio import open_omezarr_container, open_omezarr_plate


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
    zarr_plate = open_omezarr_plate(
        plate_url, cache=True, parallel_safe=False, mode="r"
    )
    # Load the first image of the selected well to get shape & pixel sizes
    # Makes the assumption that all images in all wells will have the same shapes
    image_path = f"{plate_url}/{zarr_plate.get_image_path(row, col, zarr_plate.get_well(row, col).paths()[0])}"
    ome_zarr_container = open_omezarr_container(image_path)
    level = ome_zarr_container.levels_paths[0]
    ome_zarr_image = ome_zarr_container.get_image(path=level)
    shape = ome_zarr_image.shape[-2:]
    scale = (ome_zarr_image.pixel_size.y, ome_zarr_image.pixel_size.x)

    row_i = zarr_plate.rows.index(row)
    col_i = zarr_plate.columns.index(col)

    if is_plate:
        top_left_corner = [
            row_i * scale[0] * shape[0],
            col_i * scale[1] * shape[1],
        ]
    else:
        top_left_corner = [0, 0]
    bottom_right_corner = [
        top_left_corner[0] + scale[0] * shape[0],
        top_left_corner[1] + scale[1] * shape[1],
    ]

    return top_left_corner, bottom_right_corner


class NapariHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        show_info(log_entry)
