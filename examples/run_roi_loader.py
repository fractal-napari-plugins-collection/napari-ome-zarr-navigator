import numpy as np

import napari

from napari_ome_zarr_navigator.roi_loader import (
    ROILoader,
)


def run_roi_loader():
    viewer = napari.Viewer()
    # _ = viewer.add_image(np.random.random((100, 100)))
    roi_widget = ROILoader(viewer)
    viewer.window.add_dock_widget(roi_widget)
    viewer.show(block=True)


if __name__ == '__main__':
    run_roi_loader()