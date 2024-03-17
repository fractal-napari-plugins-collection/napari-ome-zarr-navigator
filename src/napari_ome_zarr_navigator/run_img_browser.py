import napari

from napari_ome_zarr_navigator.img_browser import ImgBrowser

from napari_ome_zarr_navigator import _TEST_DATA_DIR


def main():
    zarr_url = "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr/B/03/0"
    viewer = napari.Viewer()
    zarr_url = "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr/B/03/0"
    roi_loader_widget = ImgBrowser(
        viewer,
        # zarr_url=_TEST_DATA_DIR.joinpath("10.5281_zenodo.10424292", zarr_url),
    )

    viewer.window.add_dock_widget(roi_loader_widget, name="OME-ZARR navigator")
    viewer.show(block=True)


if __name__ == "__main__":
    main()
