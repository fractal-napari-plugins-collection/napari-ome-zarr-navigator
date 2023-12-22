import napari

from napari_ome_zarr_navigator.roi_loader import ROILoader


def main():
    # Create a viewer
    viewer = napari.Viewer()

    zarr_url = "/Users/joel/Library/CloudStorage/Dropbox/Joel/BioVisionCenter/Code/napari-ome-zarr-navigator/test_data/20231222_navigator_test_data/20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr/B/03/0"
    roi_loader_widget = ROILoader(viewer, zarr_url=zarr_url)

    viewer.window.add_dock_widget(roi_loader_widget)
    # Add the ROI Loader plugin
    # viewer.window.add_plugin_dock_widget(
    #     roi_loader_widget, area="right", name="ROI Loader"
    # )

    viewer.show(block=True)


if __name__ == "__main__":
    main()
