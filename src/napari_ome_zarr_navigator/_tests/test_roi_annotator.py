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


def test_mode_2_reverts_to_mode_1(make_napari_viewer):
    viewer = make_napari_viewer()
    annotator = ROIAnnotator(viewer)
    annotator._mode_selector.value = _MODE_MASK
    # Should have been reverted
    assert annotator._mode_selector.value == _MODE_EMPTY


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
    # Simulate HTTP store (non-local)
    annotator.store = "https://example.com/image.zarr"
    annotator.is_local = False
    annotator._update_save_btn_state()
    assert not annotator._save_btn.enabled


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

    rois = annotator._shapes_to_rois(shapes_layer)
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

    rois = annotator._shapes_to_rois(shapes_layer)
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

    rois = annotator._shapes_to_rois(shapes_layer)
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

    rois = annotator._shapes_to_rois(shapes_layer)
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

    rois = annotator._shapes_to_rois(shapes_layer)
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

    rois = annotator._shapes_to_rois(shapes_layer)
    assert len(rois) == 0


def test_shapes_to_rois_no_clipping_without_extent(make_napari_viewer):
    """When image_extent is None, coordinates are passed through unchanged."""
    viewer = make_napari_viewer()
    annotator = ROIAnnotator(viewer)
    assert annotator.image_extent is None

    rect = np.array([[-5.0, -5.0], [-5.0, 200.0], [200.0, 200.0], [200.0, -5.0]])
    shapes_layer = viewer.add_shapes(rect, shape_type="rectangle", name="ROIs")

    rois = annotator._shapes_to_rois(shapes_layer)
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
