from __future__ import annotations

from pathlib import Path
from typing import Union
import re
import requests
import urllib
import hashlib
import wget
import shutil
import logging
import numpy as np

from napari.types import LayerDataTuple

from ome_zarr.io import parse_url
from ome_zarr.reader import Reader, Node

from napari_ome_zarr_navigator import _TEST_DATA_DIR

logging.getLogger("ome_zarr").setLevel(logging.WARN)


def load_ome_zarr_from_zenodo(doi: str, zarr_url: Union[str, Path]):
    doi_path = Path(_TEST_DATA_DIR).joinpath(doi.replace("/", "_"))
    zarr_path = doi_path.joinpath(zarr_url)
    if not doi_path.is_dir():
        download_from_zenodo(doi, directory=doi_path)
        shutil.unpack_archive(zarr_path.with_suffix(".zarr.zip"), doi_path)
    reader = Reader(parse_url(zarr_path))
    zarr_group = list(reader())[0]
    return zarr_group


def download_from_zenodo(
    doi: str,
    overwrite: bool = False,
    directory: Union[str, Path] = Path(),
    access_token: str = None,
):
    record_id = re.match(r".*zenodo.(\w+)", doi).group(1)
    url = "https://zenodo.org/api/records/" + record_id
    js = requests.get(url).json()
    doi = js["metadata"]["doi"]
    print("Title: " + js["metadata"]["title"])
    print("Publication date: " + js["metadata"]["publication_date"])
    print("DOI: " + js["metadata"]["doi"])
    print(
        "Total file size: {:.1f} MB".format(
            sum(f["size"] / 10**6 for f in js["files"])
        )
    )
    doi_path = Path(directory)
    try:
        doi_path.mkdir(exist_ok=overwrite, parents=True)
    except FileExistsError:
        print(f"{doi_path} exists. Don't overwrite.")
        return
    for file in js["files"]:
        file_path = Path(doi_path).joinpath(file["key"])
        algorithm, checksum = file["checksum"].split(":")
        try:
            link = urllib.parse.unquote(file["links"]["self"])
            wget.download(
                f"{link}?access_token={access_token}", str(directory)
            )
            check_passed, returned_checksum = verify_checksum(
                file_path, algorithm, checksum
            )
            if check_passed:
                print(f"\nChecksum is correct. ({checksum})")
            else:
                print(
                    f"\nChecksum is incorrect! ({checksum} got: {returned_checksum})"
                )
        except urllib.error.HTTPError:
            pass


def verify_checksum(filename: Union[str, Path], algorithm, original_checksum):
    h = hashlib.new(algorithm)
    with open(filename, "rb") as f:
        h.update(f.read())
        returned_checksum = h.hexdigest()
    if returned_checksum == original_checksum:
        return True, returned_checksum
    else:
        return False, returned_checksum


def hiPSC_zarr() -> list[LayerDataTuple]:
    doi = "10.5281_zenodo.11262587"
    zarr_url = "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr"
    return load_zarr(doi, zarr_url)


def load_zarr(doi: str, zarr_url: Union[str, Path]) -> list[LayerDataTuple]:
    ome_zarr = load_ome_zarr_from_zenodo(doi, zarr_url)
    if ome_zarr:

        return [
            (
                ome_zarr.data,
                {
                    "name": ome_zarr.metadata["name"],
                    "channel_axis": 0,
                    "contrast_limits": ome_zarr.metadata["contrast_limits"],
                    "colormap": ome_zarr.metadata["colormap"],
                    "metadata": {"sample_path": ome_zarr.zarr.path},
                    "scale": np.array([1.0, 0.1625, 0.1625]),
                },
                "image",
            )
        ]
    else:
        return [(None,)]
