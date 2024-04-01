import pytest

from napari_ome_zarr_navigator.ome_zarr_image import OMEZarrImage


@pytest.fixture(scope="function")
def ome_zarr_image_2d(zenodo_zarr) -> OMEZarrImage:
    plate_zarr_url_2D = zenodo_zarr[1]
    zarr_url_2D = f"{plate_zarr_url_2D}/B/03/0"
    zarr_img_2D = OMEZarrImage(zarr_url_2D)
    return zarr_img_2D


@pytest.fixture(scope="function")
def ome_zarr_image_3d(zenodo_zarr) -> OMEZarrImage:
    plate_zarr_url_3D = zenodo_zarr[0]
    zarr_url_3D = f"{plate_zarr_url_3D}/B/03/0"
    zarr_img_3D = OMEZarrImage(zarr_url_3D)
    return zarr_img_3D


def test_OMEZarrImage_init(ome_zarr_image_2d, ome_zarr_image_3d):
    assert len(ome_zarr_image_2d.image_meta.multiscales) == 1
    assert ome_zarr_image_2d.zarr_subgroups == ["labels", "tables"]
    assert len(ome_zarr_image_3d.image_meta.multiscales) == 1
    assert ome_zarr_image_3d.zarr_subgroups == ["labels", "tables"]


def test_channel_list(ome_zarr_image_2d, ome_zarr_image_3d):
    assert ome_zarr_image_2d.get_channel_list() == [
        "DAPI",
        "nanog",
        "Lamin B1",
    ]
    assert ome_zarr_image_3d.get_channel_list() == [
        "DAPI",
        "nanog",
        "Lamin B1",
    ]


def test_pyramid_levels(ome_zarr_image_2d, ome_zarr_image_3d):
    assert ome_zarr_image_2d.get_pyramid_levels() == ["0", "1", "2", "3", "4"]
    assert ome_zarr_image_3d.get_pyramid_levels() == ["0", "1", "2", "3", "4"]


def test_labels_list(ome_zarr_image_2d, ome_zarr_image_3d):
    # TODO: Add a test case without labels => new test data?
    assert ome_zarr_image_2d.get_labels_list() == ["nuclei"]
    assert ome_zarr_image_3d.get_labels_list() == ["nuclei"]


def test_tables_list(ome_zarr_image_2d):
    assert ome_zarr_image_2d.get_tables_list() == [
        "FOV_ROI_table",
        "condition",
        "measurements",
        "well_ROI_table",
    ]
    # Only ROI tables
    assert ome_zarr_image_2d.get_tables_list(table_type="ROIs") == [
        "FOV_ROI_table",
        "well_ROI_table",
    ]
    assert ome_zarr_image_2d.get_tables_list(table_type="roi_table") == [
        "FOV_ROI_table",
        "well_ROI_table",
    ]

    # Feature tables
    assert ome_zarr_image_2d.get_tables_list(table_type="feature_table") == [
        "measurements"
    ]

    # Condition table
    assert ome_zarr_image_2d.get_tables_list(table_type="condition") == [
        "condition"
    ]

    # Check for other table types
    assert (
        len(ome_zarr_image_2d.get_tables_list(table_type="masking_roi_table"))
        == 0
    )
    assert len(ome_zarr_image_2d.get_tables_list(table_type="abcd")) == 0

    # TODO: Add test for table without a type & strict parsing
    # print(ome_zarr_image_2d.get_tables_list(strict=False))


def test_wrong_zarr_urls(zenodo_zarr):
    non_existing_path = "/path/does/not/exist"
    with pytest.raises(ValueError):
        OMEZarrImage(non_existing_path)

    zarr_but_not_image = f"{zenodo_zarr[1]}/B/03/"
    expected_error = (
        "The provided Zarr is not a valid OME-Zarr image. Loading its"
        "metadata triggered the following error ValidationError: "
    )
    with pytest.raises(ValueError) as exc_info:
        OMEZarrImage(zarr_but_not_image)
    assert expected_error in str(exc_info.value)
    assert "multiscales" in str(exc_info.value)
    assert "field required" in str(exc_info.value)


def test_roi_loading_from_indices(ome_zarr_image_2d):
    img, scale = ome_zarr_image_2d.load_zarr_array_index_based(
        zarr_url=ome_zarr_image_2d.zarr_url,
        roi_table="FOV_ROI_table",
        roi_index=0,
        subset=0,
        multiscale=ome_zarr_image_2d.image_meta.multiscale,
        level_path="2",
    )
    assert scale == [1.0, 0.65, 0.65]
    assert img.shape == (1, 540, 640)


def test_roi_loading(ome_zarr_image_2d):
    img, scale = ome_zarr_image_2d.load_intensity_roi(
        roi_table="FOV_ROI_table",
        roi_name="FOV_1",
        channel="DAPI",
        level_path="2",
    )
    assert scale == [1.0, 0.65, 0.65]
    assert img.shape == (1, 540, 640)


def test_label_roi_loading_direct_level(ome_zarr_image_2d):
    img, scale = ome_zarr_image_2d.load_label_roi(
        roi_table="FOV_ROI_table",
        roi_name="FOV_1",
        label="nuclei",
        level_path_label="0",
    )
    assert scale == [1.0, 0.65, 0.65]
    assert img.shape == (1, 540, 640)

    img, scale = ome_zarr_image_2d.load_label_roi(
        roi_table="FOV_ROI_table",
        roi_name="FOV_1",
        label="nuclei",
        level_path_label="1",
    )
    assert scale == [1.0, 1.3, 1.3]
    assert img.shape == (1, 270, 320)


def test_label_roi_loading_relative_level(ome_zarr_image_2d):
    img, scale = ome_zarr_image_2d.load_label_roi(
        roi_table="FOV_ROI_table",
        roi_name="FOV_1",
        label="nuclei",
        level_path_img="0",
    )
    assert scale == [1.0, 0.65, 0.65]
    assert img.shape == (1, 540, 640)

    img, scale = ome_zarr_image_2d.load_label_roi(
        roi_table="FOV_ROI_table",
        roi_name="FOV_1",
        label="nuclei",
        level_path_img="1",
    )
    assert scale == [1.0, 0.65, 0.65]
    assert img.shape == (1, 540, 640)

    img, scale = ome_zarr_image_2d.load_label_roi(
        roi_table="FOV_ROI_table",
        roi_name="FOV_1",
        label="nuclei",
        level_path_img="2",
    )
    assert scale == [1.0, 0.65, 0.65]
    assert img.shape == (1, 540, 640)

    img, scale = ome_zarr_image_2d.load_label_roi(
        roi_table="FOV_ROI_table",
        roi_name="FOV_1",
        label="nuclei",
        level_path_img="3",
    )
    assert scale == [1.0, 1.3, 1.3]
    assert img.shape == (1, 270, 320)
