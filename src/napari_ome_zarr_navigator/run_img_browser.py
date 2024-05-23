import napari

from napari_ome_zarr_navigator.img_browser import ImgBrowser

from napari_ome_zarr_navigator import _TEST_DATA_DIR


def main():
    viewer = napari.Viewer()
    roi_loader_widget = ImgBrowser(
        viewer,
    )
    viewer.window.add_dock_widget(roi_loader_widget, name="OME-ZARR navigator")
    viewer.show(block=True)


if __name__ == "__main__":
    main()
