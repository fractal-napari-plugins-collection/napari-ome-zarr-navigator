from pathlib import Path

import numpy as np

from napari_ome_zarr_navigator.img_browser import ImgBrowser
from napari_ome_zarr_navigator.roi_loader import (
    ROILoader,
    ROILoaderImage,
    ROILoaderPlate,
)


def test_roi_loader_init(make_napari_viewer):
    viewer = make_napari_viewer()
    _ = viewer.add_image(np.random.random((100, 100)))
    _ = ROILoader(viewer)


def test_plate(make_napari_viewer, zenodo_zarr, qtbot):
    viewer = make_napari_viewer()
    plate_url = zenodo_zarr[1]
    image_browser = ImgBrowser(viewer=viewer)
    roi_loader = ROILoaderPlate(
        viewer,
        plate_store=plate_url,
        row="B",
        col="03",
        image_browser=image_browser,
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

    # Set parameters
    roi_loader._roi_table_picker.value = "FOV_ROI_table"
    roi_loader._roi_picker.value = "FOV_1"
    roi_loader._channel_picker.value = ["DAPI"]
    roi_loader._level_picker.value = "0"
    roi_loader._label_picker.value = ["nuclei"]
    roi_loader._feature_picker.value = ["measurements"]
    roi_loader._remove_old_labels_box.value = False

    roi_loader.run()
    with qtbot.waitSignal(
        roi_loader.image_changed_event.load_finished, timeout=10000
    ):
        pass

    # TODO: Test that layers got added and that label layer has features


def test_roi_loader(make_napari_viewer, zenodo_zarr, qtbot):
    viewer = make_napari_viewer()
    zarr_url = Path(zenodo_zarr[1]) / "B" / "03" / "0"
    print(zarr_url)
    roi_loader = ROILoaderImage(
        viewer,
        zarr_url=str(zarr_url),
    )
    with qtbot.waitSignal(
        roi_loader.image_changed_event.roi_tables_updated, timeout=5000
    ):
        pass

    with qtbot.waitSignal(
        roi_loader.image_changed_event.roi_choices_updated, timeout=5000
    ):
        pass

    assert str(roi_loader.zarr_selector._file_picker.value) == str(zarr_url)

    # Check parameters
    assert roi_loader._roi_table_picker.value == "FOV_ROI_table"
    assert roi_loader._roi_picker.value == "FOV_1"
    assert roi_loader._channel_picker.choices == ("DAPI", "nanog", "Lamin B1")
    assert roi_loader._level_picker.value == "0"
    assert roi_loader._label_picker.choices == ("nuclei",)
    assert roi_loader._feature_picker.choices == ("measurements",)
    assert not roi_loader._remove_old_labels_box.value

    # Set parameters
    roi_loader._roi_table_picker.value = "FOV_ROI_table"
    roi_loader._roi_picker.value = "FOV_1"
    roi_loader._channel_picker.value = ["DAPI"]
    roi_loader._level_picker.value = "0"
    roi_loader._label_picker.value = ["nuclei"]
    roi_loader._feature_picker.value = ["measurements"]
    roi_loader._remove_old_labels_box.value = False

    roi_loader.run()
    with qtbot.waitSignal(
        roi_loader.image_changed_event.load_finished, timeout=10000
    ):
        pass

    # TODO: Test that layers got added and that label layer has features
