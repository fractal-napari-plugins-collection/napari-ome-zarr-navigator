from pathlib import Path

import napari.layers
import numpy as np
import pytest

from napari_ome_zarr_navigator.plate_browser import PlateBrowser
from napari_ome_zarr_navigator.roi_loader import (
    _NO_ROI_TABLE,
    ROILoader,
    ROILoaderImage,
    ROILoaderPlate,
)


def _init_roi_loader_image(viewer, zarr_url, qtbot):
    """Create a ROILoaderImage and wait for init signals."""
    roi_loader = ROILoaderImage(viewer, zarr_url=str(zarr_url))
    with qtbot.waitSignal(
        roi_loader.image_changed_event.roi_tables_updated, timeout=5000
    ):
        pass
    with qtbot.waitSignal(
        roi_loader.image_changed_event.roi_choices_updated, timeout=5000
    ):
        pass
    return roi_loader


def test_roi_loader_init(make_napari_viewer):
    viewer = make_napari_viewer()
    _ = viewer.add_image(np.random.random((100, 100)))
    _ = ROILoader(viewer)


def test_plate(make_napari_viewer, zenodo_zarr, qtbot):
    viewer = make_napari_viewer()
    plate_url = zenodo_zarr[1]
    plate_browser = PlateBrowser(viewer=viewer)
    roi_loader = ROILoaderPlate(
        viewer,
        plate_store=plate_url,
        row="B",
        col="03",
        plate_browser=plate_browser,
        is_plate=True,
    )
    with qtbot.waitSignal(
        roi_loader.image_changed_event.roi_tables_updated, timeout=5000
    ):
        pass

    with qtbot.waitSignal(
        roi_loader.image_changed_event.roi_choices_updated, timeout=5000
    ):
        pass

    assert roi_loader._zarr_picker.choices == ("0",)
    assert roi_loader._zarr_picker.value == "0"

    # Set parameters (level picker now shows resolution strings; default Multi-resolution
    # mode means run() uses dask pyramid regardless of level picker value)
    roi_loader._roi_table_picker.value = "FOV_ROI_table"
    roi_loader._roi_picker.value = "FOV_1"
    roi_loader._channel_picker.value = ["DAPI"]
    roi_loader._label_picker.value = ["nuclei"]
    roi_loader._feature_picker.value = ["measurements"]
    roi_loader._remove_old_labels_box.value = False

    roi_loader.run()
    with qtbot.waitSignal(roi_loader.image_changed_event.load_finished, timeout=10000):
        pass

    image_layers = [
        layer for layer in viewer.layers if isinstance(layer, napari.layers.Image)
    ]
    label_layers = [
        layer for layer in viewer.layers if isinstance(layer, napari.layers.Labels)
    ]
    assert len(image_layers) == 1
    assert len(label_layers) == 1
    # Images load as multiscale pyramid by default; finest level is (2160, 2560)
    assert image_layers[0].multiscale
    assert image_layers[0].data[0].shape == (2160, 2560)
    # Labels load as fixed resolution by default
    assert not label_layers[0].multiscale
    assert label_layers[0].data.shape == (540, 640)
    assert "label" in label_layers[0].features.columns


def test_roi_loader(make_napari_viewer, zenodo_zarr, qtbot):
    viewer = make_napari_viewer()
    zarr_url = Path(zenodo_zarr[1]) / "B" / "03" / "0"
    roi_loader = _init_roi_loader_image(viewer, zarr_url, qtbot)

    assert str(roi_loader.zarr_selector._file_picker.value) == str(zarr_url)

    # Check parameters
    assert roi_loader._roi_table_picker.value == "FOV_ROI_table"
    assert roi_loader._roi_picker.value == "FOV_1"
    assert roi_loader._channel_picker.choices == ("DAPI", "nanog", "Lamin B1")
    # Level picker now shows physical resolution strings (e.g. "0.163 micrometer")
    assert roi_loader._level_picker.value is not None
    assert roi_loader._level_picker.value == roi_loader._level_picker.choices[0]
    assert roi_loader._label_picker.choices == ("nuclei",)
    assert roi_loader._feature_picker.choices == ("measurements", "measurements_csv")
    assert not roi_loader._remove_old_labels_box.value
    # Whole-image sentinel appears last even when ROI tables exist
    assert roi_loader._roi_table_picker.choices[-1] == _NO_ROI_TABLE

    roi_loader._roi_table_picker.value = "FOV_ROI_table"
    roi_loader._roi_picker.value = "FOV_1"
    roi_loader._channel_picker.value = ["DAPI"]
    roi_loader._label_picker.value = ["nuclei"]
    roi_loader._feature_picker.value = ["measurements"]
    roi_loader._remove_old_labels_box.value = False

    roi_loader.run()
    with qtbot.waitSignal(roi_loader.image_changed_event.load_finished, timeout=10000):
        pass

    image_layers = [
        layer for layer in viewer.layers if isinstance(layer, napari.layers.Image)
    ]
    label_layers = [
        layer for layer in viewer.layers if isinstance(layer, napari.layers.Labels)
    ]
    assert len(image_layers) == 1
    assert len(label_layers) == 1
    # Images load as multiscale pyramid by default; finest level is (2160, 2560)
    assert image_layers[0].multiscale
    assert image_layers[0].data[0].shape == (2160, 2560)
    # Labels load as fixed resolution by default
    assert not label_layers[0].multiscale
    assert label_layers[0].data.shape == (540, 640)
    assert "label" in label_layers[0].features.columns


def test_roi_loader_fixed_resolution(make_napari_viewer, zenodo_zarr, qtbot):
    viewer = make_napari_viewer()
    zarr_url = Path(zenodo_zarr[1]) / "B" / "03" / "0"
    roi_loader = _init_roi_loader_image(viewer, zarr_url, qtbot)

    roi_loader._image_loading_mode.value = "Fixed resolution"
    assert roi_loader._level_picker.enabled

    roi_loader._roi_table_picker.value = "FOV_ROI_table"
    roi_loader._roi_picker.value = "FOV_1"
    roi_loader._channel_picker.value = ["DAPI"]
    roi_loader._label_picker.value = ["nuclei"]
    roi_loader._remove_old_labels_box.value = False

    roi_loader.run()
    with qtbot.waitSignal(roi_loader.image_changed_event.load_finished, timeout=10000):
        pass

    image_layers = [
        layer for layer in viewer.layers if isinstance(layer, napari.layers.Image)
    ]
    label_layers = [
        layer for layer in viewer.layers if isinstance(layer, napari.layers.Labels)
    ]
    assert len(image_layers) == 1
    assert len(label_layers) == 1
    # Fixed resolution: single array; image at finest level, labels at their default level
    assert not image_layers[0].multiscale
    assert image_layers[0].data.shape == (2160, 2560)
    assert not label_layers[0].multiscale
    assert label_layers[0].data.shape == (540, 640)


def test_roi_loader_whole_image(make_napari_viewer, zenodo_zarr, qtbot):
    viewer = make_napari_viewer()
    zarr_url = Path(zenodo_zarr[1]) / "B" / "03" / "0"
    roi_loader = _init_roi_loader_image(viewer, zarr_url, qtbot)

    # Sentinel must appear last even when ROI tables are present
    assert roi_loader._roi_table_picker.choices[-1] == _NO_ROI_TABLE
    roi_loader._roi_table_picker.value = _NO_ROI_TABLE
    roi_loader._channel_picker.value = ["DAPI"]
    roi_loader._label_picker.value = []
    roi_loader._remove_old_labels_box.value = False

    roi_loader.run()
    with qtbot.waitSignal(roi_loader.image_changed_event.load_finished, timeout=10000):
        pass

    image_layers = [
        layer for layer in viewer.layers if isinstance(layer, napari.layers.Image)
    ]
    assert len(image_layers) == 1
    # Whole image in multiscale mode; wider than a single FOV (2 sites = 5120 px wide)
    assert image_layers[0].multiscale
    assert image_layers[0].data[0].shape == (2160, 5120)


@pytest.mark.parametrize("roi_table_picker_value", ["FOV_ROI_table", _NO_ROI_TABLE])
def test_roi_table_choices_always_include_sentinel(
    make_napari_viewer, zenodo_zarr, qtbot, roi_table_picker_value
):
    """_NO_ROI_TABLE always appears as the last choice regardless of available tables."""
    viewer = make_napari_viewer()
    zarr_url = Path(zenodo_zarr[1]) / "B" / "03" / "0"
    roi_loader = _init_roi_loader_image(viewer, zarr_url, qtbot)
    assert roi_loader._roi_table_picker.choices[-1] == _NO_ROI_TABLE
