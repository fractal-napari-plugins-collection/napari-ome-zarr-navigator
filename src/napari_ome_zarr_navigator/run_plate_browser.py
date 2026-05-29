import napari

from napari_ome_zarr_navigator.plate_browser import PlateBrowser


def main():
    viewer = napari.Viewer()
    plate_browser_widget = PlateBrowser(
        viewer,
    )
    viewer.window.add_dock_widget(plate_browser_widget, name="OME-ZARR navigator")
    viewer.show(block=True)


if __name__ == "__main__":
    main()
