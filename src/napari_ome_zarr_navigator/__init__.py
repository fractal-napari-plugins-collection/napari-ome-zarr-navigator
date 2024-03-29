from pathlib import Path

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
from .img_browser import (
    ImgBrowser,
)
from .roi_loader import (
    ROILoader,
)

__all__ = ("ROILoader", "ImgBrowser")
FILE = Path(__file__).resolve()
_PACKAGE_DIR = FILE.parents[2]
_MODULE_DIR = FILE.parent
_TEST_DIR = _MODULE_DIR.joinpath("_tests")
_TEST_DATA_DIR = _PACKAGE_DIR.joinpath("test_data")
