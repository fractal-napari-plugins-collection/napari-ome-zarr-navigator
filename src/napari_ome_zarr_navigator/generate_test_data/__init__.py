#!/usr/bin/env python3
from pathlib import Path
from typing import Union
import re
import requests
import urllib
import hashlib
import wget
import shutil


from napari_ome_zarr_navigator import _TEST_DATA_DIR


def load_ome_zarr_from_zenodo(doi: str, zarr_url: Union[str, Path]):
    doi_path = Path(_TEST_DATA_DIR).joinpath(doi.replace("/", "_"))
    zarr_path = doi_path.joinpath(zarr_url)
    if not doi_path.is_dir():
        download_from_zenodo(doi, directory=doi_path)
        shutil.unpack_archive(zarr_path.with_suffix(".zarr.zip"), doi_path)


def download_from_zenodo(
    doi: str,
    overwrite: bool = False,
    directory: Union[str, Path] = Path(),
    access_token: str = None,
):
    record_id = re.match(r".*/zenodo.(\w+)", doi).group(1)
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
        doi_path.mkdir(exist_ok=overwrite)
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
