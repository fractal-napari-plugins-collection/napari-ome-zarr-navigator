import napari

from napari_ome_zarr_navigator.roi_loader import ROILoader


def main():
    # Create a viewer
    viewer = napari.Viewer()

    zarr_url = "/Users/joel/Library/CloudStorage/Dropbox/Joel/BioVisionCenter/Code/napari-ome-zarr-navigator/test_data/20231222_navigator_test_data/20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr/B/03/0"
    # zarr_url = None
    roi_loader_widget = ROILoader(viewer, zarr_url=zarr_url)

    viewer.window.add_dock_widget(roi_loader_widget)

    viewer.show(block=True)


if __name__ == "__main__":
    main()
