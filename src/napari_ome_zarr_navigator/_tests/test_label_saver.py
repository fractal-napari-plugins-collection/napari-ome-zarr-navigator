from pathlib import Path

import numpy as np
import pytest
from ngio import create_synthetic_ome_zarr, open_ome_zarr_container

from napari_ome_zarr_navigator.label_saver import (
    _WM_APPEND,
    _WM_NEW,
    _WM_OVERWRITE,
    _WM_RESET,
    LabelSaverImage,
)

# ---------------------------------------------------------------------------
# Fixtures
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


@pytest.fixture
def synthetic_timeseries_image_path(tmp_path: Path) -> str:
    """3-timepoint 2D image (t=3, y=64, x=64)."""
    image_dir = tmp_path / "test_image_t.zarr"
    create_synthetic_ome_zarr(
        store=str(image_dir),
        shape=(3, 64, 64),
        axes_names=["t", "y", "x"],
        table_backend="csv",
    )
    return str(image_dir)


def _make_full_label_layer(viewer, image_path, name="seg"):
    """Add a full-size label layer whose scale matches the synthetic image."""
    container = open_ome_zarr_container(image_path, mode="r")
    img = container.get_image()
    label_data = np.zeros((64, 64), dtype=np.uint32)
    label_data[10:30, 10:30] = 1
    return viewer.add_labels(
        label_data, name=name, scale=(img.pixel_size.y, img.pixel_size.x)
    )


def _base_kwargs(image_path, label_layer, write_mode=_WM_NEW, axes_str="yx"):
    """Shared keyword arguments for _do_save_impl."""
    return {
        "zarr_url": image_path,
        "label_array": np.asarray(label_layer.data),
        "layer_scale": tuple(label_layer.scale),
        "layer_translate": tuple(float(v) for v in label_layer.translate),
        "label_name": label_layer.name,
        "axes_str": axes_str,
        "write_mode": write_mode,
        "save_masking_roi": False,
        "masking_roi_table_name": "",
        "masking_roi_backend": "csv",
    }


# ---------------------------------------------------------------------------
# Widget instantiation
# ---------------------------------------------------------------------------


def test_label_saver_standalone_init(make_napari_viewer):
    viewer = make_napari_viewer()
    saver = LabelSaverImage(viewer=viewer)
    assert not saver._btn_save.enabled
    assert not saver._advanced_container.visible
    assert not saver._masking_roi_table_name.visible
    assert not saver._masking_roi_backend.visible
    assert saver._zarr_url is None


def test_label_saver_prepopulated_init(make_napari_viewer, synthetic_image_path):
    viewer = make_napari_viewer()
    saver = LabelSaverImage(
        viewer=viewer,
        zarr_url=synthetic_image_path,
        source="File",
    )
    assert not saver._zarr_selector.visible
    assert saver._zarr_url == synthetic_image_path
    assert saver._source == "File"


def test_label_saver_save_button_disabled_without_name(make_napari_viewer):
    viewer = make_napari_viewer()
    saver = LabelSaverImage(viewer=viewer)
    assert not saver._btn_save.enabled
    saver._label_name.value = "my_label"
    assert saver._btn_save.enabled
    saver._label_name.value = ""
    assert not saver._btn_save.enabled


def test_label_saver_masking_roi_table_name_autofill(make_napari_viewer):
    viewer = make_napari_viewer()
    saver = LabelSaverImage(viewer=viewer)
    saver._label_name.value = "nuclei"
    assert saver._masking_roi_table_name.value == "nuclei_masking_ROI_table"


def test_label_saver_masking_roi_toggle_text_unchanged(make_napari_viewer):
    viewer = make_napari_viewer()
    saver = LabelSaverImage(viewer=viewer)
    saver._on_save_masking_roi_changed(True)
    saver._on_save_masking_roi_changed(False)


def test_label_saver_advanced_toggle_text(make_napari_viewer):
    viewer = make_napari_viewer()
    saver = LabelSaverImage(viewer=viewer)
    assert "▶" in saver._advanced_toggle.text
    saver._toggle_advanced()
    assert "▼" in saver._advanced_toggle.text
    saver._toggle_advanced()
    assert "▶" in saver._advanced_toggle.text


def test_label_saver_layer_picker_updates(make_napari_viewer):
    viewer = make_napari_viewer()
    saver = LabelSaverImage(viewer=viewer)
    assert list(saver._layer_picker.choices) == []

    label_data = np.zeros((64, 64), dtype=np.uint8)
    viewer.add_labels(label_data, name="my_seg")
    assert "my_seg" in saver._layer_picker.choices

    viewer.layers.remove(viewer.layers["my_seg"])
    assert "my_seg" not in saver._layer_picker.choices


def test_label_saver_write_mode_choices(make_napari_viewer):
    viewer = make_napari_viewer()
    saver = LabelSaverImage(viewer=viewer)
    choices = list(saver._write_mode.choices)
    assert _WM_NEW in choices
    assert _WM_OVERWRITE in choices
    assert _WM_RESET in choices
    assert _WM_APPEND in choices


# ---------------------------------------------------------------------------
# Validation unit tests
# ---------------------------------------------------------------------------


def test_validate_full_matching_extents(synthetic_image_path):
    container = open_ome_zarr_container(synthetic_image_path, mode="r")
    img = container.get_image(path=container.level_paths[0])

    label_array = np.zeros((64, 64), dtype=np.uint8)
    layer_scale = (img.pixel_size.y, img.pixel_size.x)
    ok, msg = LabelSaverImage._validate_full(label_array, layer_scale, "yx", img)
    assert ok, msg


def test_validate_full_mismatched_extents(synthetic_image_path):
    container = open_ome_zarr_container(synthetic_image_path, mode="r")
    img = container.get_image(path=container.level_paths[0])

    label_array = np.zeros((32, 32), dtype=np.uint8)
    layer_scale = (img.pixel_size.y, img.pixel_size.x)
    ok, msg = LabelSaverImage._validate_full(label_array, layer_scale, "yx", img)
    assert not ok
    assert "extent" in msg.lower()


def test_validate_full_timepoint_mismatch(synthetic_timeseries_image_path):
    container = open_ome_zarr_container(synthetic_timeseries_image_path, mode="r")
    img = container.get_image(path=container.level_paths[0])
    label_array = np.zeros((2, 64, 64), dtype=np.uint8)
    layer_scale = (img.pixel_size.t, img.pixel_size.y, img.pixel_size.x)
    ok, msg = LabelSaverImage._validate_full(label_array, layer_scale, "tyx", img)
    assert not ok
    assert "timepoint" in msg.lower()


def test_validate_tz_passes_when_no_tz(synthetic_image_path):
    container = open_ome_zarr_container(synthetic_image_path, mode="r")
    img = container.get_image(path=container.level_paths[0])
    # 2D label smaller than image: _validate_tz ignores yx extents
    label_array = np.zeros((32, 32), dtype=np.uint8)
    ok, msg = LabelSaverImage._validate_tz(label_array, "yx", img)
    assert ok, msg


def test_validate_tz_fails_on_timepoint_mismatch(synthetic_timeseries_image_path):
    container = open_ome_zarr_container(synthetic_timeseries_image_path, mode="r")
    img = container.get_image(path=container.level_paths[0])
    label_array = np.zeros((2, 32, 32), dtype=np.uint8)  # t=2, but image has t=3
    ok, msg = LabelSaverImage._validate_tz(label_array, "tyx", img)
    assert not ok
    assert "timepoint" in msg.lower()


# ---------------------------------------------------------------------------
# Axes inference unit tests
# ---------------------------------------------------------------------------


def test_infer_axes_2d(make_napari_viewer, synthetic_image_path):
    viewer = make_napari_viewer()
    saver = LabelSaverImage(viewer=viewer)
    container = open_ome_zarr_container(synthetic_image_path, mode="r")
    result = saver._infer_axes(2, container)
    assert result == "yx"


def test_infer_axes_3d_zyx(make_napari_viewer, tmp_path):
    viewer = make_napari_viewer()
    saver = LabelSaverImage(viewer=viewer)
    image_dir = tmp_path / "test_3d.zarr"
    create_synthetic_ome_zarr(
        store=str(image_dir),
        shape=(4, 64, 64),
        axes_names=["z", "y", "x"],
        table_backend="csv",
    )
    container = open_ome_zarr_container(str(image_dir), mode="r")
    result = saver._infer_axes(3, container)
    assert result == "zyx"


# ---------------------------------------------------------------------------
# _compute_write_region unit tests
# ---------------------------------------------------------------------------


def test_compute_write_region_full_image(synthetic_image_path):
    container = open_ome_zarr_container(synthetic_image_path, mode="r")
    img = container.get_image(path=container.level_paths[0])
    label_array = np.zeros((64, 64), dtype=np.uint32)
    layer_scale = (img.pixel_size.y, img.pixel_size.x)
    y0, x0, y1, x1, full_h, full_w, is_partial = LabelSaverImage._compute_write_region(
        layer_translate=(0.0, 0.0),
        layer_scale=layer_scale,
        plate_translation=(0.0, 0.0),
        label_array=label_array,
        axes_str="yx",
        img=img,
    )
    assert y0 == 0 and x0 == 0
    assert y1 == full_h and x1 == full_w
    assert not is_partial


def test_compute_write_region_sub_roi(synthetic_image_path):
    container = open_ome_zarr_container(synthetic_image_path, mode="r")
    img = container.get_image(path=container.level_paths[0])
    pxy = img.pixel_size.y
    pxx = img.pixel_size.x
    label_array = np.zeros((32, 32), dtype=np.uint32)
    layer_scale = (pxy, pxx)
    layer_translate = (10 * pxy, 20 * pxx)
    y0, x0, y1, x1, full_h, full_w, is_partial = LabelSaverImage._compute_write_region(
        layer_translate=layer_translate,
        layer_scale=layer_scale,
        plate_translation=(0.0, 0.0),
        label_array=label_array,
        axes_str="yx",
        img=img,
    )
    assert y0 == 10 and x0 == 20
    assert y1 == 42 and x1 == 52
    assert is_partial


def test_compute_write_region_plate_offset_subtracted(synthetic_image_path):
    container = open_ome_zarr_container(synthetic_image_path, mode="r")
    img = container.get_image(path=container.level_paths[0])
    pxy = img.pixel_size.y
    pxx = img.pixel_size.x
    label_array = np.zeros((32, 32), dtype=np.uint32)
    layer_scale = (pxy, pxx)
    plate_offset = (100.0, 200.0)
    roi_offset = (10 * pxy, 20 * pxx)
    layer_translate = (plate_offset[0] + roi_offset[0], plate_offset[1] + roi_offset[1])
    y0, x0, *_ = LabelSaverImage._compute_write_region(
        layer_translate=layer_translate,
        layer_scale=layer_scale,
        plate_translation=plate_offset,
        label_array=label_array,
        axes_str="yx",
        img=img,
    )
    assert y0 == 10 and x0 == 20


# ---------------------------------------------------------------------------
# Integration tests — Mode: New label (initialise)
# ---------------------------------------------------------------------------


def test_mode_new_creates_label(make_napari_viewer, tmp_path):
    image_dir = tmp_path / "test.zarr"
    create_synthetic_ome_zarr(
        store=str(image_dir), shape=(1, 64, 64), table_backend="csv"
    )
    viewer = make_napari_viewer()
    layer = _make_full_label_layer(viewer, str(image_dir))
    saver = LabelSaverImage(viewer=viewer)
    assert saver._do_save_impl(**_base_kwargs(str(image_dir), layer, _WM_NEW)) is True
    assert layer.name in open_ome_zarr_container(str(image_dir), mode="r").list_labels()


def test_mode_new_fails_if_label_exists(make_napari_viewer, tmp_path):
    image_dir = tmp_path / "test.zarr"
    create_synthetic_ome_zarr(
        store=str(image_dir), shape=(1, 64, 64), table_backend="csv"
    )
    viewer = make_napari_viewer()
    layer = _make_full_label_layer(viewer, str(image_dir))
    saver = LabelSaverImage(viewer=viewer)
    kwargs = _base_kwargs(str(image_dir), layer, _WM_NEW)
    assert saver._do_save_impl(**kwargs) is True
    assert saver._do_save_impl(**kwargs) is False


def test_mode_new_sub_roi_writes_partial(make_napari_viewer, tmp_path):
    image_dir = tmp_path / "test.zarr"
    create_synthetic_ome_zarr(
        store=str(image_dir), shape=(1, 64, 64), table_backend="csv"
    )
    viewer = make_napari_viewer()
    container = open_ome_zarr_container(str(image_dir), mode="r")
    img = container.get_image()
    pxy, pxx = img.pixel_size.y, img.pixel_size.x
    sub_data = np.ones((32, 32), dtype=np.uint32)
    sub_layer = viewer.add_labels(
        sub_data, name="sub_seg", scale=(pxy, pxx), translate=(10 * pxy, 10 * pxx)
    )
    saver = LabelSaverImage(viewer=viewer)
    assert (
        saver._do_save_impl(**_base_kwargs(str(image_dir), sub_layer, _WM_NEW)) is True
    )
    saved = np.array(
        open_ome_zarr_container(str(image_dir), mode="r")
        .get_label("sub_seg")
        .get_array()
    )
    assert np.all(saved[0, 10:42, 10:42] == 1)
    assert np.all(saved[0, 0:10, :] == 0)


# ---------------------------------------------------------------------------
# Integration tests — Mode: Overwrite full label
# ---------------------------------------------------------------------------


def test_mode_overwrite_replaces_existing(make_napari_viewer, tmp_path):
    image_dir = tmp_path / "test.zarr"
    create_synthetic_ome_zarr(
        store=str(image_dir), shape=(1, 64, 64), table_backend="csv"
    )
    viewer = make_napari_viewer()
    layer = _make_full_label_layer(viewer, str(image_dir))
    saver = LabelSaverImage(viewer=viewer)
    kwargs = _base_kwargs(str(image_dir), layer, _WM_OVERWRITE)
    assert saver._do_save_impl(**kwargs) is True
    assert saver._do_save_impl(**kwargs) is True  # second save also succeeds


def test_mode_overwrite_fails_on_extent_mismatch(make_napari_viewer, tmp_path):
    image_dir = tmp_path / "test.zarr"
    create_synthetic_ome_zarr(
        store=str(image_dir), shape=(1, 64, 64), table_backend="csv"
    )
    viewer = make_napari_viewer()
    container = open_ome_zarr_container(str(image_dir), mode="r")
    img = container.get_image()
    sub_data = np.zeros((32, 32), dtype=np.uint32)
    sub_layer = viewer.add_labels(
        sub_data, name="sub", scale=(img.pixel_size.y, img.pixel_size.x)
    )
    saver = LabelSaverImage(viewer=viewer)
    assert (
        saver._do_save_impl(**_base_kwargs(str(image_dir), sub_layer, _WM_OVERWRITE))
        is False
    )


# ---------------------------------------------------------------------------
# Integration tests — Mode: Reset existing label
# ---------------------------------------------------------------------------


def test_mode_reset_recreates_label(make_napari_viewer, tmp_path):
    image_dir = tmp_path / "test.zarr"
    create_synthetic_ome_zarr(
        store=str(image_dir), shape=(1, 64, 64), table_backend="csv"
    )
    viewer = make_napari_viewer()
    layer = _make_full_label_layer(viewer, str(image_dir))
    saver = LabelSaverImage(viewer=viewer)
    assert saver._do_save_impl(**_base_kwargs(str(image_dir), layer, _WM_NEW)) is True
    assert saver._do_save_impl(**_base_kwargs(str(image_dir), layer, _WM_RESET)) is True


def test_mode_reset_fails_if_label_missing(make_napari_viewer, tmp_path):
    image_dir = tmp_path / "test.zarr"
    create_synthetic_ome_zarr(
        store=str(image_dir), shape=(1, 64, 64), table_backend="csv"
    )
    viewer = make_napari_viewer()
    layer = _make_full_label_layer(viewer, str(image_dir))
    saver = LabelSaverImage(viewer=viewer)
    assert (
        saver._do_save_impl(**_base_kwargs(str(image_dir), layer, _WM_RESET)) is False
    )


# ---------------------------------------------------------------------------
# Integration tests — Mode: Append to existing label
# ---------------------------------------------------------------------------


def test_mode_append_patches_existing(make_napari_viewer, tmp_path):
    image_dir = tmp_path / "test.zarr"
    create_synthetic_ome_zarr(
        store=str(image_dir), shape=(1, 64, 64), table_backend="csv"
    )
    viewer = make_napari_viewer()
    container = open_ome_zarr_container(str(image_dir), mode="r")
    img = container.get_image()
    pxy, pxx = img.pixel_size.y, img.pixel_size.x

    full_layer = _make_full_label_layer(viewer, str(image_dir), name="lbl")
    saver = LabelSaverImage(viewer=viewer)
    assert (
        saver._do_save_impl(**_base_kwargs(str(image_dir), full_layer, _WM_NEW)) is True
    )

    sub_data = np.ones((20, 20), dtype=np.uint32)
    sub_layer = viewer.add_labels(
        sub_data, name="lbl_sub", scale=(pxy, pxx), translate=(10 * pxy, 10 * pxx)
    )
    # Pass label_name="lbl" explicitly so it targets the existing OME-Zarr label
    kwargs = _base_kwargs(str(image_dir), sub_layer, _WM_APPEND)
    kwargs["label_name"] = "lbl"
    assert saver._do_save_impl(**kwargs) is True

    saved = np.array(
        open_ome_zarr_container(str(image_dir), mode="r").get_label("lbl").get_array()
    )
    assert np.all(saved[0, 10:30, 10:30] == 1)
    assert np.all(saved[0, 0:10, :] == 0)


def test_mode_append_fails_if_label_missing(make_napari_viewer, tmp_path):
    image_dir = tmp_path / "test.zarr"
    create_synthetic_ome_zarr(
        store=str(image_dir), shape=(1, 64, 64), table_backend="csv"
    )
    viewer = make_napari_viewer()
    layer = _make_full_label_layer(viewer, str(image_dir))
    saver = LabelSaverImage(viewer=viewer)
    assert (
        saver._do_save_impl(**_base_kwargs(str(image_dir), layer, _WM_APPEND)) is False
    )


# ---------------------------------------------------------------------------
# Masking ROI table save
# ---------------------------------------------------------------------------


def test_label_saver_saves_with_masking_roi_table(make_napari_viewer, tmp_path):
    image_dir = tmp_path / "test_save_masking.zarr"
    create_synthetic_ome_zarr(
        store=str(image_dir), shape=(1, 64, 64), table_backend="csv"
    )
    viewer = make_napari_viewer()
    container = open_ome_zarr_container(str(image_dir), mode="r")
    img = container.get_image()

    label_data = np.zeros((64, 64), dtype=np.uint32)
    label_data[10:30, 10:30] = 1
    label_data[40:60, 40:60] = 2
    label_layer = viewer.add_labels(
        label_data, name="my_seg", scale=(img.pixel_size.y, img.pixel_size.x)
    )
    saver = LabelSaverImage(viewer=viewer)

    success = saver._do_save_impl(
        zarr_url=str(image_dir),
        label_array=np.asarray(label_layer.data),
        layer_scale=tuple(label_layer.scale),
        layer_translate=(0.0, 0.0),
        label_name="my_seg",
        axes_str="yx",
        write_mode=_WM_NEW,
        save_masking_roi=True,
        masking_roi_table_name="my_seg_masking_ROI_table",
        masking_roi_backend="csv",
    )
    assert success
    result = open_ome_zarr_container(str(image_dir), mode="r")
    assert "my_seg" in result.list_labels()
    assert "my_seg_masking_ROI_table" in result.list_tables()


# ---------------------------------------------------------------------------
# _validate_tz rejects timepoint mismatch in partial write modes
# ---------------------------------------------------------------------------


def test_partial_mode_rejects_timepoint_mismatch(make_napari_viewer, tmp_path):
    image_dir = tmp_path / "test_t.zarr"
    create_synthetic_ome_zarr(
        store=str(image_dir),
        shape=(3, 64, 64),
        axes_names=["t", "y", "x"],
        table_backend="csv",
    )
    viewer = make_napari_viewer()
    container = open_ome_zarr_container(str(image_dir), mode="r")
    img = container.get_image()
    label_data = np.zeros((2, 32, 32), dtype=np.uint32)
    label_layer = viewer.add_labels(
        label_data,
        name="lbl",
        scale=(img.pixel_size.t, img.pixel_size.y, img.pixel_size.x),
    )
    saver = LabelSaverImage(viewer=viewer)
    kwargs = {
        "zarr_url": str(image_dir),
        "label_array": np.asarray(label_layer.data),
        "layer_scale": tuple(label_layer.scale),
        "layer_translate": (0.0, 0.0, 0.0),
        "label_name": "lbl",
        "axes_str": "tyx",
        "write_mode": _WM_NEW,
        "save_masking_roi": False,
        "masking_roi_table_name": "",
        "masking_roi_backend": "csv",
    }
    assert saver._do_save_impl(**kwargs) is False
