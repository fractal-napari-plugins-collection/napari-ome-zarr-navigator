#!/usr/bin/env python3
import string
import napari
from zarr.errors import PathNotFoundError
import anndata as ad
import pandas as pd


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


def add_features_to_labels(
    zarr_url: str,
    labels_layer: napari.layers.Labels,
    feature_name: str = "regionprops",
):
    """Add features to napari labels layer

    Args:
        zarr_url: Path to the OME-ZARR
        labels_layer: napari labels layer
        feature_name: Folder name of the measured regionprobs features
    """
    try:
        ann_tbl = ad.read_zarr(f"{zarr_url}/tables/{feature_name}/")
        labels_layer.features = ann_tbl.to_df()
        labels_layer.predictions = ann_tbl.obs
    except PathNotFoundError:
        pass
