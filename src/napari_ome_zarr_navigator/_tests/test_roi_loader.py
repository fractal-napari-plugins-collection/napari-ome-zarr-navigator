import numpy as np

from napari_ome_zarr_navigator.img_browser import ImgBrowser
from napari_ome_zarr_navigator.roi_loader import (
    ROILoader,
    ROILoaderPlate,
)


def test_roi_loader_init(make_napari_viewer):
    viewer = make_napari_viewer()
    _ = viewer.add_image(np.random.random((100, 100)))
    _ = ROILoader(viewer)


def test_plate(make_napari_viewer, zenodo_zarr, qtbot):
    viewer = make_napari_viewer()
    plate_url = zenodo_zarr[0]
    image_browser = ImgBrowser(viewer=viewer)
    roi_loader = ROILoaderPlate(
        viewer,
        plate_url=plate_url,
        row="B",
        col="03",
        image_browser=image_browser,
        is_plate=True,
    )
    with qtbot.waitSignal(
        roi_loader.image_changed_event.roi_tables_updated, timeout=100
    ):
        pass

    with qtbot.waitSignal(
        roi_loader.image_changed_event.roi_choices_updated, timeout=100
    ):
        pass

    assert roi_loader._zarr_picker.choices == ("0",)
    assert roi_loader._zarr_picker.value == "0"
    # time.sleep(5)
    # roi_loader.run()
    # print(roi_loader.ome_zarr_container)
