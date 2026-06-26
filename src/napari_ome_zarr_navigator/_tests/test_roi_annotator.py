from pathlib import Path

import napari.layers
import numpy as np
import pytest
from ngio import create_synthetic_ome_zarr, open_ome_zarr_container

from napari_ome_zarr_navigator.plate_browser import PlateBrowser
from napari_ome_zarr_navigator.roi_annotator import (
    _MODE_EMPTY,
    _MODE_MASK,
    ROIAnnotator,
    ROIAnnotatorImage,
    ROIAnnotatorPlate,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_image_path(tmp_path: Path) -> str:
    image_dir = tmp_path / "test_image.zarr"
    create_synthetic_ome_zarr(
        store=str(image_dir),
        shape=(1, 64, 64),
        table_backend="csv",
    )
    return str(image_dir)


# ---------------------------------------------------------------------------
# ROIAnnotator base class
# ---------------------------------------------------------------------------


def test_roi_annotator_init(make_napari_viewer):
    viewer = make_napari_viewer()
    annotator = ROIAnnotator(viewer)
    assert annotator.store is None
    assert annotator.is_local is True
    assert annotator.translation == (0.0, 0.0)
    assert annotator.layer_base_name == ""
    assert not annotator._save_btn.enabled


def test_shapes_layer_name(make_napari_viewer):
    viewer = make_napari_viewer()
    annotator = ROIAnnotator(viewer)
    assert annotator._shapes_layer_name == "ROIs"

    annotator.layer_base_name = "B03_"
    assert annotator._shapes_layer_name == "B03_ROIs"


def test_initialize_roi_layer_creates_shapes(make_napari_viewer):
    viewer = make_napari_viewer()
    annotator = ROIAnnotator(viewer)
    annotator.initialize_roi_layer()

    shapes_layers = [
        layer for layer in viewer.layers if isinstance(layer, napari.layers.Shapes)
    ]
    assert len(shapes_layers) == 1
    assert shapes_layers[0].name == "ROIs"
    assert shapes_layers[0].mode == "add_rectangle"


def test_initialize_roi_layer_replaces_existing(make_napari_viewer):
    viewer = make_napari_viewer()
    annotator = ROIAnnotator(viewer)
    annotator.initialize_roi_layer()
    annotator.initialize_roi_layer()  # second call should replace

    shapes_layers = [
        layer for layer in viewer.layers if isinstance(layer, napari.layers.Shapes)
    ]
    assert len(shapes_layers) == 1


def test_initialize_roi_layer_translation(make_napari_viewer):
    viewer = make_napari_viewer()
    annotator = ROIAnnotator(viewer)
    annotator.translation = (100.0, 200.0)
    annotator.initialize_roi_layer()

    shapes_layers = [
        layer for layer in viewer.layers if isinstance(layer, napari.layers.Shapes)
    ]
    assert len(shapes_layers) == 1
    np.testing.assert_array_equal(shapes_layers[0].translate, [100.0, 200.0])


# ---------------------------------------------------------------------------
# Mode guard
# ---------------------------------------------------------------------------


def test_mask_mode_shows_label_picker(make_napari_viewer):
    viewer = make_napari_viewer()
    annotator = ROIAnnotator(viewer)
    # In headless mode visible is always False; isHidden() is the reliable check
    assert annotator._label_layer_picker.native.isHidden()

    annotator._mode_selector.value = _MODE_MASK

    assert annotator._mode_selector.value == _MODE_MASK
    assert not annotator._label_layer_picker.native.isHidden()
    assert annotator._init_layer_btn.text == "Calculate masking ROI table"


def test_mask_mode_reverts_on_empty(make_napari_viewer):
    viewer = make_napari_viewer()
    annotator = ROIAnnotator(viewer)
    annotator._mode_selector.value = _MODE_MASK
    annotator._mode_selector.value = _MODE_EMPTY

    assert annotator._label_layer_picker.native.isHidden()
    assert annotator._init_layer_btn.text == "Initialize ROI Layer"
    assert annotator._table_name.value == "interactive_ROIs"


# ---------------------------------------------------------------------------
# Save button state
# ---------------------------------------------------------------------------


def test_save_btn_disabled_when_no_store(make_napari_viewer):
    viewer = make_napari_viewer()
    annotator = ROIAnnotator(viewer)
    assert not annotator._save_btn.enabled


def test_save_btn_enabled_for_local_store(make_napari_viewer, synthetic_image_path):
    viewer = make_napari_viewer()
    annotator = ROIAnnotator(viewer)
    annotator.store = synthetic_image_path
    annotator.is_local = True
    annotator._update_save_btn_state()
    assert annotator._save_btn.enabled


def test_save_btn_disabled_for_http_store(make_napari_viewer):
    viewer = make_napari_viewer()
    annotator = ROIAnnotator(viewer)
    # Simulate HTTP store (non-local): folder picker shown, button disabled until folder chosen
    annotator.store = "https://example.com/image.zarr"
    annotator.is_local = False
    annotator._update_save_btn_state()
    assert not annotator._save_btn.enabled
    assert not annotator._remote_save_folder.native.isHidden()


def test_remote_save_writes_to_folder(make_napari_viewer, tmp_path):
    """_do_save_remote writes a Zarr table group to the chosen local folder."""
    from ngio.tables import open_tables_container

    viewer = make_napari_viewer()
    annotator = ROIAnnotator(viewer)
    annotator.store = "https://example.com/image.zarr"
    annotator.is_local = False

    annotator.initialize_roi_layer()
    rect = np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=float)
    viewer.layers["ROIs"].add_rectangles([rect])

    annotator._table_name.value = "remote_test_table"
    annotator._remote_save_folder.value = str(tmp_path)  # type: ignore[assignment]
    annotator._update_save_btn_state()
    assert annotator._save_btn.enabled

    annotator.save_roi_table()

    dest = str(tmp_path / "remote_test_table")
    tc = open_tables_container(dest, mode="r")
    assert "remote_test_table" in tc.list()


# ---------------------------------------------------------------------------
# _shapes_to_rois coordinate conversion
# ---------------------------------------------------------------------------


def test_shapes_to_rois_basic(make_napari_viewer):
    """Rectangle at (y=10, x=20) with size (y=30, x=40) → correct ngio ROI."""
    viewer = make_napari_viewer()
    annotator = ROIAnnotator(viewer)

    # napari rectangle: 4 corners [[y0,x0],[y1,x1],[y2,x2],[y3,x3]]
    rect = np.array(
        [
            [10.0, 20.0],
            [10.0, 60.0],
            [40.0, 60.0],
            [40.0, 20.0],
        ]
    )
    shapes_layer = viewer.add_shapes(rect, shape_type="rectangle", name="ROIs")

    rois, _ = annotator._shapes_to_rois(shapes_layer)
    assert len(rois) == 1
    roi = rois[0]
    assert roi.name == "roi_0"

    x_slice = roi["x"]
    y_slice = roi["y"]
    z_slice = roi["z"]

    assert x_slice.start == pytest.approx(20.0)
    assert x_slice.length == pytest.approx(40.0)
    assert y_slice.start == pytest.approx(10.0)
    assert y_slice.length == pytest.approx(30.0)
    assert z_slice.start == pytest.approx(0.0)
    assert z_slice.length == pytest.approx(1.0)


def test_shapes_to_rois_skips_non_rectangles(make_napari_viewer):
    viewer = make_napari_viewer()
    annotator = ROIAnnotator(viewer)

    ellipse = np.array(
        [
            [0.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [10.0, 0.0],
        ]
    )
    rect = np.array(
        [
            [5.0, 5.0],
            [5.0, 15.0],
            [15.0, 15.0],
            [15.0, 5.0],
        ]
    )
    shapes_layer = viewer.add_shapes(name="ROIs")
    shapes_layer.add_ellipses([ellipse])
    shapes_layer.add_rectangles([rect])

    rois, _ = annotator._shapes_to_rois(shapes_layer)
    assert len(rois) == 1
    assert rois[0].name == "roi_0"


def test_shapes_to_rois_multiple(make_napari_viewer):
    viewer = make_napari_viewer()
    annotator = ROIAnnotator(viewer)

    rects = [
        np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=float),
        np.array([[20, 30], [20, 50], [40, 50], [40, 30]], dtype=float),
    ]
    shapes_layer = viewer.add_shapes(rects, shape_type="rectangle", name="ROIs")

    rois, _ = annotator._shapes_to_rois(shapes_layer)
    assert len(rois) == 2
    assert rois[0].name == "roi_0"
    assert rois[1].name == "roi_1"


# ---------------------------------------------------------------------------
# Clipping to image bounds
# ---------------------------------------------------------------------------


def test_shapes_to_rois_clips_negative_start(make_napari_viewer):
    """ROI with negative start is clipped to 0; length adjusted accordingly."""
    viewer = make_napari_viewer()
    annotator = ROIAnnotator(viewer)
    annotator.image_extent = (100.0, 100.0)

    # Rectangle from (-10, -5) to (20, 30) → clips to (0, 0)-(20, 30)
    rect = np.array([[-10.0, -5.0], [-10.0, 30.0], [20.0, 30.0], [20.0, -5.0]])
    shapes_layer = viewer.add_shapes(rect, shape_type="rectangle", name="ROIs")

    rois, _ = annotator._shapes_to_rois(shapes_layer)
    assert len(rois) == 1
    assert rois[0]["x"].start == pytest.approx(0.0)
    assert rois[0]["x"].length == pytest.approx(30.0)
    assert rois[0]["y"].start == pytest.approx(0.0)
    assert rois[0]["y"].length == pytest.approx(20.0)


def test_shapes_to_rois_clips_far_edge(make_napari_viewer):
    """ROI extending past image far edge is clipped to image_extent."""
    viewer = make_napari_viewer()
    annotator = ROIAnnotator(viewer)
    annotator.image_extent = (50.0, 60.0)

    # Rectangle from (30, 40) to (80, 90) → clips to (30, 40)-(50, 60)
    rect = np.array([[30.0, 40.0], [30.0, 90.0], [80.0, 90.0], [80.0, 40.0]])
    shapes_layer = viewer.add_shapes(rect, shape_type="rectangle", name="ROIs")

    rois, _ = annotator._shapes_to_rois(shapes_layer)
    assert len(rois) == 1
    assert rois[0]["y"].start == pytest.approx(30.0)
    assert rois[0]["y"].length == pytest.approx(20.0)  # 50 - 30
    assert rois[0]["x"].start == pytest.approx(40.0)
    assert rois[0]["x"].length == pytest.approx(20.0)  # 60 - 40


def test_shapes_to_rois_skips_fully_outside(make_napari_viewer):
    """ROI entirely outside image bounds is skipped."""
    viewer = make_napari_viewer()
    annotator = ROIAnnotator(viewer)
    annotator.image_extent = (50.0, 50.0)

    # Entirely below y=0
    rect = np.array([[-30.0, 0.0], [-30.0, 10.0], [-10.0, 10.0], [-10.0, 0.0]])
    shapes_layer = viewer.add_shapes(rect, shape_type="rectangle", name="ROIs")

    rois, _ = annotator._shapes_to_rois(shapes_layer)
    assert len(rois) == 0


def test_shapes_to_rois_no_clipping_without_extent(make_napari_viewer):
    """When image_extent is None, coordinates are passed through unchanged."""
    viewer = make_napari_viewer()
    annotator = ROIAnnotator(viewer)
    assert annotator.image_extent is None

    rect = np.array([[-5.0, -5.0], [-5.0, 200.0], [200.0, 200.0], [200.0, -5.0]])
    shapes_layer = viewer.add_shapes(rect, shape_type="rectangle", name="ROIs")

    rois, _ = annotator._shapes_to_rois(shapes_layer)
    assert len(rois) == 1
    assert rois[0]["x"].start == pytest.approx(-5.0)
    assert rois[0]["x"].length == pytest.approx(205.0)


# ---------------------------------------------------------------------------
# End-to-end save
# ---------------------------------------------------------------------------


def test_save_roi_table(make_napari_viewer, synthetic_image_path, tmp_path):
    viewer = make_napari_viewer()
    annotator = ROIAnnotator(viewer)
    annotator.store = synthetic_image_path
    annotator.is_local = True
    annotator._update_save_btn_state()

    # Initialize shapes layer and add a rectangle
    annotator.initialize_roi_layer()
    rect = np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=float)
    shapes_layer = viewer.layers["ROIs"]
    shapes_layer.add_rectangles([rect])

    annotator._table_name.value = "test_roi_table"
    annotator._overwrite_box.value = True
    annotator.save_roi_table()

    # Verify the table was written
    container = open_ome_zarr_container(synthetic_image_path, mode="r")
    tables = container.list_tables(filter_types="roi_table")
    assert "test_roi_table" in tables


# ---------------------------------------------------------------------------
# ROIAnnotatorImage standalone
# ---------------------------------------------------------------------------


def test_roi_annotator_image_init(make_napari_viewer):
    viewer = make_napari_viewer()
    annotator = ROIAnnotatorImage(viewer)
    assert annotator.store is None
    assert annotator.layer_base_name == ""
    assert not annotator._save_btn.enabled


def test_roi_annotator_image_file_source(make_napari_viewer, synthetic_image_path):
    viewer = make_napari_viewer()
    annotator = ROIAnnotatorImage(viewer)

    annotator.zarr_selector._source_selector.value = "File"
    annotator.zarr_selector._file_picker.value = synthetic_image_path  # type: ignore[assignment]
    annotator._on_selector_changed()

    assert annotator.store == synthetic_image_path
    assert annotator.is_local is True
    assert annotator._save_btn.enabled


# ---------------------------------------------------------------------------
# ROIAnnotatorPlate (plate context)
# ---------------------------------------------------------------------------


def test_roi_annotator_plate_init(make_napari_viewer, synthetic_plate_path):
    viewer = make_napari_viewer()
    plate_browser = PlateBrowser(viewer=viewer)
    annotator = ROIAnnotatorPlate(
        viewer=viewer,
        plate_store=synthetic_plate_path,
        row="B",
        col="03",
        plate_browser=plate_browser,
        is_plate=True,
        is_local=True,
    )
    assert annotator.layer_base_name == "B03_"
    assert annotator.is_local is True
    assert annotator.store is not None
    assert annotator._save_btn.enabled
    # Translation should be (0,0) for the first well in a 2-well plate
    assert annotator.translation[0] == pytest.approx(0.0)
    assert annotator.translation[1] == pytest.approx(0.0)


def test_roi_annotator_plate_second_well_translation(
    make_napari_viewer, synthetic_plate_path
):
    viewer = make_napari_viewer()
    plate_browser = PlateBrowser(viewer=viewer)
    annotator = ROIAnnotatorPlate(
        viewer=viewer,
        plate_store=synthetic_plate_path,
        row="C",
        col="04",
        plate_browser=plate_browser,
        is_plate=True,
        is_local=True,
    )
    # Second well should have non-zero translation
    assert annotator.translation[0] > 0 or annotator.translation[1] > 0


# ---------------------------------------------------------------------------
# _guess_reference_label
# ---------------------------------------------------------------------------


def test_guess_reference_label_exact(make_napari_viewer):
    annotator = ROIAnnotator(make_napari_viewer())
    assert annotator._guess_reference_label("nuclei", ["nuclei", "other"]) == "nuclei"


def test_guess_reference_label_prefix_strip(make_napari_viewer):
    annotator = ROIAnnotator(make_napari_viewer())
    assert (
        annotator._guess_reference_label("B03_nuclei", ["nuclei", "other"]) == "nuclei"
    )


def test_guess_reference_label_no_match(make_napari_viewer):
    annotator = ROIAnnotator(make_napari_viewer())
    assert annotator._guess_reference_label("B03_nuclei", ["cells"]) is None


# ---------------------------------------------------------------------------
# _default_mask_table_name
# ---------------------------------------------------------------------------


def test_default_mask_table_name_simple(make_napari_viewer):
    viewer = make_napari_viewer()
    annotator = ROIAnnotator(viewer)
    viewer.add_labels(np.zeros((10, 10), dtype=int), name="nuclei")
    annotator._label_layer_picker.choices = ["nuclei"]
    annotator._label_layer_picker.value = "nuclei"
    assert annotator._default_mask_table_name() == "nuclei_masking_ROI_table"


def test_default_mask_table_name_spaces_stripped(make_napari_viewer):
    viewer = make_napari_viewer()
    annotator = ROIAnnotator(viewer)
    viewer.add_labels(np.zeros((10, 10), dtype=int), name="Label - 1")
    annotator._label_layer_picker.choices = ["Label - 1"]
    annotator._label_layer_picker.value = "Label - 1"
    assert annotator._default_mask_table_name() == "Label-1_masking_ROI_table"


def test_default_mask_table_name_no_label(make_napari_viewer):
    annotator = ROIAnnotator(make_napari_viewer())
    annotator._label_layer_picker.choices = ["(no label layers)"]
    annotator._label_layer_picker.value = "(no label layers)"
    assert annotator._default_mask_table_name() == "label_masking_ROI_table"


# ---------------------------------------------------------------------------
# initialize_roi_layer — ndim variants
# ---------------------------------------------------------------------------


def test_initialize_roi_layer_3d(make_napari_viewer):
    viewer = make_napari_viewer()
    annotator = ROIAnnotator(viewer)
    annotator._image_is_3d = True
    annotator._shapes_z_scale = 0.5
    annotator.initialize_roi_layer()
    layer = next(
        layer for layer in viewer.layers if isinstance(layer, napari.layers.Shapes)
    )
    assert layer.ndim == 3
    assert float(layer.scale[0]) == pytest.approx(0.5)
    assert float(layer.translate[0]) == pytest.approx(0.0)


def test_initialize_roi_layer_time_series(make_napari_viewer):
    viewer = make_napari_viewer()
    annotator = ROIAnnotator(viewer)
    annotator._image_is_time_series = True
    annotator._shapes_t_scale = 30.0
    annotator.initialize_roi_layer()
    layer = next(
        layer for layer in viewer.layers if isinstance(layer, napari.layers.Shapes)
    )
    assert layer.ndim == 3
    assert float(layer.scale[0]) == pytest.approx(30.0)


def test_initialize_roi_layer_tzyx(make_napari_viewer):
    viewer = make_napari_viewer()
    annotator = ROIAnnotator(viewer)
    annotator._image_is_3d = True
    annotator._image_is_time_series = True
    annotator._shapes_z_scale = 0.5
    annotator._shapes_t_scale = 30.0
    annotator.initialize_roi_layer()
    layer = next(
        layer for layer in viewer.layers if isinstance(layer, napari.layers.Shapes)
    )
    assert layer.ndim == 4
    np.testing.assert_allclose(layer.scale, [30.0, 0.5, 1.0, 1.0])


# ---------------------------------------------------------------------------
# _shapes_to_rois — 3D z-coordinate recovery
# ---------------------------------------------------------------------------


def test_shapes_to_rois_3d_z_from_vertex(make_napari_viewer):
    """Vertex z coord x z_scale used when no z properties stored."""
    viewer = make_napari_viewer()
    annotator = ROIAnnotator(viewer)
    annotator._image_is_3d = True
    # z_voxel=2, scale[0]=0.5 → z_start = 2*0.5 = 1.0, z_len = 0.5
    rect = np.array([[2, 10, 20], [2, 10, 60], [2, 40, 60], [2, 40, 20]], dtype=float)
    shapes_layer = viewer.add_shapes(rect, shape_type="rectangle", name="ROIs", ndim=3)
    shapes_layer.scale = [0.5, 1.0, 1.0]
    rois, _ = annotator._shapes_to_rois(shapes_layer)
    assert len(rois) == 1
    assert rois[0]["z"].start == pytest.approx(1.0)
    assert rois[0]["z"].length == pytest.approx(0.5)


def test_shapes_to_rois_3d_z_from_properties(make_napari_viewer):
    """Stored z_start/z_length properties override vertex-derived z."""
    viewer = make_napari_viewer()
    annotator = ROIAnnotator(viewer)
    annotator._image_is_3d = True
    # z_voxel=1 → vertex would give z_start=0.5 if scale=0.5, but props say 0/2.0
    rect = np.array([[1, 10, 20], [1, 10, 60], [1, 40, 60], [1, 40, 20]], dtype=float)
    shapes_layer = viewer.add_shapes(rect, shape_type="rectangle", name="ROIs", ndim=3)
    shapes_layer.scale = [0.5, 1.0, 1.0]
    shapes_layer.properties = {
        "z_start": np.array([0.0]),
        "z_length": np.array([2.0]),
    }
    rois, _ = annotator._shapes_to_rois(shapes_layer)
    assert len(rois) == 1
    assert rois[0]["z"].start == pytest.approx(0.0)
    assert rois[0]["z"].length == pytest.approx(2.0)


def test_shapes_to_rois_t_from_properties(make_napari_viewer):
    """Stored t_start/t_length properties are included in the ROI slices."""
    viewer = make_napari_viewer()
    annotator = ROIAnnotator(viewer)
    annotator._image_is_time_series = True
    annotator._shapes_t_scale = 10.0
    # t_idx=0 stored in vertex[0][0], but properties override
    rect = np.array([[0, 10, 20], [0, 10, 60], [0, 40, 60], [0, 40, 20]], dtype=float)
    shapes_layer = viewer.add_shapes(rect, shape_type="rectangle", name="ROIs", ndim=3)
    shapes_layer.scale = [10.0, 1.0, 1.0]
    shapes_layer.properties = {
        "t_start": np.array([30.0]),
        "t_length": np.array([10.0]),
    }
    rois, _ = annotator._shapes_to_rois(shapes_layer)
    assert len(rois) == 1
    assert rois[0]["t"].start == pytest.approx(30.0)
    assert rois[0]["t"].length == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# End-to-end: calculate + save masking ROI table
# ---------------------------------------------------------------------------


def test_calculate_and_save_masking_roi_table(make_napari_viewer, synthetic_image_path):
    """Full round-trip: compute masking ROIs from 'nuclei' label, save, verify."""
    viewer = make_napari_viewer()
    annotator = ROIAnnotator(viewer)
    annotator.store = synthetic_image_path
    annotator.is_local = True
    annotator._update_save_btn_state()

    # Load the nuclei label into the viewer
    container = open_ome_zarr_container(synthetic_image_path, mode="r")
    lbl = container.get_label("nuclei")
    label_data = np.array(lbl.get_array())
    ps = lbl.pixel_size
    viewer.add_labels(label_data, name="nuclei", scale=(ps.z, ps.y, ps.x))

    # Switch to mask mode and compute
    annotator._mode_selector.value = _MODE_MASK
    annotator._label_layer_picker.choices = ["nuclei"]
    annotator._label_layer_picker.value = "nuclei"
    annotator.calculate_masking_roi_table()

    # nuclei label in create_synthetic_ome_zarr(shape=(1,64,64)) has 6 objects
    shapes = [
        layer for layer in viewer.layers if isinstance(layer, napari.layers.Shapes)
    ]
    assert len(shapes) == 1
    assert len(shapes[0].data) == 6

    # Save and verify the table was written
    annotator._reference_label_picker.choices = ["nuclei"]
    annotator._reference_label_picker.value = "nuclei"
    annotator._table_name.value = "test_masking_table"
    annotator._overwrite_box.value = True
    annotator.save_roi_table()

    container2 = open_ome_zarr_container(synthetic_image_path, mode="r")
    assert "test_masking_table" in container2.list_tables()
