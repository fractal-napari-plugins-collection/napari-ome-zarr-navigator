from pathlib import Path

from napari_ome_zarr_navigator import _TEST_DATA_DIR
from napari_ome_zarr_navigator._sample_data import (
    load_ome_zarr_from_zenodo,
)


def test_load_zenodo_data():
    doi = "10.5281/zenodo.11262587"
    zarr_url = "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr"
    doi_path = doi_path = Path(_TEST_DATA_DIR).joinpath(doi.replace("/", "_"))
    load_ome_zarr_from_zenodo(doi, zarr_url)
    assert doi_path.is_dir()
