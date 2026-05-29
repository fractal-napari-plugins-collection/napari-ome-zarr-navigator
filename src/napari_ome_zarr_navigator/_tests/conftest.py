import os
import shutil
from pathlib import Path

import pandas as pd
import pooch
import pytest
from ngio import create_empty_plate, create_synthetic_ome_zarr
from ngio.hcs._plate import OmeZarrPlate
from ngio.tables import ConditionTable


@pytest.fixture(scope="session")
def testdata_path() -> Path:
    TEST_DIR = Path(__file__).parent.parent.parent.parent
    return TEST_DIR / "test_data/"


@pytest.fixture(scope="session")
def zenodo_zarr(testdata_path: Path) -> list[str]:
    """
    This takes care of multiple steps:

    1. Download/unzip two Zarr containers (3D and MIP) from Zenodo, via pooch
    2. Copy the two Zarr containers into tests/data
    3. Modify the Zarrs in tests/data, to add whatever is not in Zenodo
    """

    # 1 Download Zarrs from Zenodo
    DOI = "10.5281/zenodo.20429951"
    DOI_slug = DOI.replace("/", "_").replace(".", "_")
    rootfolder = testdata_path / DOI_slug

    registry = {
        "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr.zip": None,
        "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr.zip": None,
    }
    folders = [rootfolder / plate[:-4] for plate in registry]

    base_url = f"doi:{DOI}"
    POOCH = pooch.create(
        pooch.os_cache("pooch") / DOI_slug,
        base_url,
        registry=registry,
        retry_if_failed=10,
        allow_updates=False,
    )

    for ind, file_name in enumerate(
        [
            "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr",
            "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr",
        ]
    ):
        # 1) Download/unzip a single Zarr from Zenodo
        file_paths = POOCH.fetch(
            f"{file_name}.zip", processor=pooch.Unzip(extract_dir=file_name)
        )
        zarr_full_path = file_paths[0].split(file_name)[0] + file_name
        # print(zarr_full_path)
        folder = folders[ind]

        # 2) Copy the downloaded Zarr into tests/data
        if os.path.isdir(str(folder)):
            shutil.rmtree(str(folder))
        shutil.copytree(Path(zarr_full_path) / file_name, folder)
    return [str(f) for f in folders]


# ---------------------------------------------------------------------------
# Synthetic OME-Zarr plate fixtures (no network, fast)
# ---------------------------------------------------------------------------


def _build_synthetic_plate(plate_dir: Path) -> OmeZarrPlate:
    """Create a 2-well (B/03, C/04) synthetic plate at *plate_dir*."""
    plate = create_empty_plate(store=str(plate_dir), name="test_plate")
    for row, col_str in [("B", "03"), ("C", "04")]:
        image_dir = plate_dir / row / col_str / "0"
        create_synthetic_ome_zarr(
            store=str(image_dir),
            shape=(1, 64, 64),
            table_backend="csv",
        )
        plate.add_image(row=row, column=col_str, image_path="0")
    return plate


@pytest.fixture
def synthetic_plate_path(tmp_path: Path) -> str:
    """2-well synthetic plate (B03, C04) with no condition tables."""
    plate_dir = tmp_path / "test_plate.zarr"
    _build_synthetic_plate(plate_dir)
    return str(plate_dir)


@pytest.fixture
def synthetic_plate_with_conditions_path(tmp_path: Path) -> str:
    """2-well synthetic plate (B03, C04) with a plate-level condition table."""
    plate_dir = tmp_path / "test_plate_cond.zarr"
    plate = _build_synthetic_plate(plate_dir)
    condition_df = pd.DataFrame(
        {
            "well": ["B03", "C04"],
            "row": ["B", "C"],
            "column": ["03", "04"],
            "differentiation_timepoint": ["day 0", "day 6"],
        }
    )
    plate.add_table(
        name="differentiation_timepoint",
        table=ConditionTable(condition_df),
        backend="csv",
        overwrite=True,
    )
    return str(plate_dir)
