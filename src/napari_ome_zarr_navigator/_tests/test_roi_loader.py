import numpy as np

from napari_ome_zarr_navigator.roi_loader import (
    ROILoader,
)


def test_image_threshold_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    _ = viewer.add_image(np.random.random((100, 100)))
    _ = ROILoader(viewer)
