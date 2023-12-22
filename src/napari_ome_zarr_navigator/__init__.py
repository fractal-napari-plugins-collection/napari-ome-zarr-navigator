try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
from ._sample_data import make_sample_data
from ._widget import (
    ExampleQWidget,
    ImageThreshold,
    threshold_autogenerate_widget,
    threshold_magic_widget,
)

__all__ = (
    "make_sample_data",
    "ExampleQWidget",
    "ImageThreshold",
    "threshold_autogenerate_widget",
    "threshold_magic_widget",
)
