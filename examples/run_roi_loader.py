import napari
from napari_ome_zarr_navigator.roi_loader import ROILoaderPlate


def run_roi_loader():
    viewer = napari.Viewer()
    # _ = viewer.add_image(np.random.random((100, 100)))
    # zarr_url = "/Users/joel/Documents/Code/napari-ome-zarr-navigator/test_data/10_5281_zenodo_11262587/20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr/B/03/0"
    # roi_widget = ROILoader(viewer, zarr_url=zarr_url)
    # roi_widget = ROILoaderImage(viewer, zarr_url=zarr_url)
    plate_url = "/Users/joel/Documents/Code/napari-ome-zarr-navigator/test_data/10_5281_zenodo_11262587/20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr"
    roi_widget = ROILoaderPlate(viewer, plate_url=plate_url, row="B", col="03")
    viewer.window.add_dock_widget(roi_widget)
    viewer.show(block=True)


if __name__ == "__main__":
    run_roi_loader()
