from __future__ import annotations

import hashlib
import logging
import re
import shutil
import urllib.parse
from pathlib import Path
from typing import Union

import requests
from napari.types import LayerDataTuple
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader

from napari_ome_zarr_navigator import _TEST_DATA_DIR

logger = logging.getLogger(__name__)
logging.getLogger("ome_zarr").setLevel(logging.WARN)


def load_ome_zarr_from_zenodo(doi: str, zarr_url: Union[str, Path]):
    doi_path = Path(_TEST_DATA_DIR).joinpath(doi.replace("/", "_"))
    zarr_path = doi_path.joinpath(zarr_url)
    if not doi_path.is_dir():
        download_from_zenodo(doi, directory=doi_path)
        shutil.unpack_archive(zarr_path.with_suffix(".zarr.zip"), doi_path)
    reader = Reader(parse_url(zarr_path))  # type: ignore[arg-type]
    zarr_group = list(reader())[0]
    return zarr_group


def download_from_zenodo(
    doi: str,
    overwrite: bool = False,
    directory: Union[str, Path] = Path(),
    access_token: str | None = None,
):
    m = re.match(r".*zenodo.(\w+)", doi)
    assert m is not None, f"Could not parse Zenodo DOI: {doi}"
    record_id = m.group(1)
    url = "https://zenodo.org/api/records/" + record_id
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    js = response.json()
    doi_path = Path(directory)
    try:
        doi_path.mkdir(exist_ok=overwrite, parents=True)
    except FileExistsError:
        logger.warning(
            "%s already exists, skipping download (pass overwrite=True to replace).",
            doi_path,
        )
        return
    total_mb = sum(f["size"] for f in js["files"]) / 10**6
    # Single message: the package logger turns every INFO record into a napari
    # notification, so per-field messages would show up as several popups.
    logger.info(
        "Downloading Zenodo record '%s' (DOI %s, published %s, %.1f MB) to %s",
        js["metadata"]["title"],
        js["metadata"]["doi"],
        js["metadata"]["publication_date"],
        total_mb,
        doi_path,
    )
    for file in js["files"]:
        file_path = doi_path / file["key"]
        algorithm, checksum = file["checksum"].split(":")
        link = urllib.parse.unquote(file["links"]["self"])
        _download_file(link, file_path, access_token=access_token)
        check_passed, returned_checksum = verify_checksum(
            file_path, algorithm, checksum
        )
        if check_passed:
            logger.debug("Checksum OK for %s (%s)", file_path.name, checksum)
        else:
            logger.warning(
                "Checksum mismatch for %s: expected %s, got %s. "
                "The download may be corrupted.",
                file_path.name,
                checksum,
                returned_checksum,
            )


def _download_file(
    url: str, destination: Path, access_token: str | None = None
) -> None:
    """Stream ``url`` into ``destination``.

    Zenodo file links end in ``/content``, so the target filename must come from
    the record's file key (``destination``), not from the URL.
    """
    params = {"access_token": access_token} if access_token else None
    with requests.get(url, params=params, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)


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
    doi = "10.5281_zenodo.20429951"
    zarr_url = "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr"
    return load_zarr(doi, zarr_url)


def leukemia_zarr() -> list[LayerDataTuple]:
    doi = "10.5281_zenodo.8322727"
    zarr_url = "operetta_plate.zarr"
    return load_zarr(doi, zarr_url)


def load_zarr(doi: str, zarr_url: Union[str, Path]) -> list[LayerDataTuple]:
    ome_zarr = load_ome_zarr_from_zenodo(doi, zarr_url)
    if ome_zarr:
        return [  # type: ignore[return-value]
            (
                ome_zarr.data,
                {
                    "name": ome_zarr.metadata["name"],
                    "channel_axis": 0,
                    "contrast_limits": ome_zarr.metadata["contrast_limits"],
                    "colormap": ome_zarr.metadata["colormap"],
                    "metadata": {"sample_path": ome_zarr.zarr.path},
                    "scale": ome_zarr.metadata["coordinateTransformations"][0][0][
                        "scale"
                    ][-3:],
                },
                "image",
            )
        ]
    else:
        return []
