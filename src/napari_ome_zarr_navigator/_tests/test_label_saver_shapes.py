"""
Parametrised regression tests for LabelSaverImage across OME-Zarr axes/shapes.

Four write patterns are tested against every image configuration:
  1. Full label (no c axis) matching all non-channel axes.
  2. 2D (yx-only) label for images where every non-yx/c axis is a singleton.
  3. 2x downsampled label (mode "Overwrite full label").
  4. Partial-region append: pixels outside the sub-region must stay unchanged.

Tests for images with a channel axis currently fail because ngio's
``derive_label(shape=...)`` validation compares the provided shape against the
full reference image shape including c, but labels never have a c axis.
Those failures reproduce the ngio bug and are expected to become green once a
fix is in place.
"""

import numpy as np
import pytest
from ngio import create_synthetic_ome_zarr, open_ome_zarr_container

from napari_ome_zarr_navigator.label_saver import (
    _WM_APPEND,
    _WM_NEW,
    _WM_OVERWRITE,
    LabelSaverImage,
)

# ---------------------------------------------------------------------------
# Image configurations
# ---------------------------------------------------------------------------

# (shape, axes_names)
_ALL_CONFIGS = [
    # tczyx — 4 z/t singleton combos
    ((1, 1, 1, 32, 32), ["t", "c", "z", "y", "x"]),
    ((1, 1, 2, 32, 32), ["t", "c", "z", "y", "x"]),
    ((2, 1, 1, 32, 32), ["t", "c", "z", "y", "x"]),
    ((2, 1, 2, 32, 32), ["t", "c", "z", "y", "x"]),
    # tzyx — 4 z/t singleton combos
    ((1, 1, 32, 32), ["t", "z", "y", "x"]),
    ((1, 2, 32, 32), ["t", "z", "y", "x"]),
    ((2, 1, 32, 32), ["t", "z", "y", "x"]),
    ((2, 2, 32, 32), ["t", "z", "y", "x"]),
    # czyx — 2 z singleton combos
    ((1, 1, 32, 32), ["c", "z", "y", "x"]),
    ((1, 2, 32, 32), ["c", "z", "y", "x"]),
    # cyx, yx
    ((1, 32, 32), ["c", "y", "x"]),
    ((32, 32), ["y", "x"]),
]


def _config_id(shape, axes_names):
    return "".join(axes_names) + "_" + "x".join(str(s) for s in shape)


def _label_axes(axes_names):
    """Image axes without c → label axes string."""
    return "".join(a for a in axes_names if a != "c")


def _label_shape(image_shape, axes_names):
    """Image shape without c → label shape tuple."""
    return tuple(s for s, a in zip(image_shape, axes_names, strict=False) if a != "c")


def _label_scale(img, label_axes_str):
    """Per-axis pixel sizes for a label (reading from img.pixel_size)."""
    return tuple(float(getattr(img.pixel_size, a)) for a in label_axes_str)


def _is_all_nonyx_singleton(image_shape, axes_names):
    """True when every non-yx/c axis has size 1 (yx label expansion is safe)."""
    return all(
        s == 1
        for s, a in zip(image_shape, axes_names, strict=False)
        if a not in ("y", "x", "c")
    )


def _make_zarr(tmp_path, shape, axes_names, name="test_image.zarr"):
    image_dir = tmp_path / name
    create_synthetic_ome_zarr(
        store=str(image_dir),
        shape=shape,
        axes_names=axes_names,
        table_backend="csv",
    )
    return str(image_dir)


def _save_kwargs(image_path, layer, label_name, axes_str, write_mode):
    return {
        "zarr_url": image_path,
        "label_array": np.asarray(layer.data),
        "layer_scale": tuple(layer.scale),
        "layer_translate": tuple(float(v) for v in layer.translate),
        "label_name": label_name,
        "axes_str": axes_str,
        "write_mode": write_mode,
        "save_masking_roi": False,
        "masking_roi_table_name": "",
        "masking_roi_backend": "csv",
    }


# ---------------------------------------------------------------------------
# Parametrize lists
# ---------------------------------------------------------------------------

_ALL_PARAMS = [
    pytest.param(shape, axes, id=_config_id(shape, axes))
    for shape, axes in _ALL_CONFIGS
]

_SINGLETON_PARAMS = [
    pytest.param(shape, axes, id=_config_id(shape, axes))
    for shape, axes in _ALL_CONFIGS
    if _is_all_nonyx_singleton(shape, axes)
]

# ---------------------------------------------------------------------------
# Pattern 1: Full label (no c), matching all non-channel axes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("image_shape,axes_names", _ALL_PARAMS)
def test_save_full_label_no_channel(
    make_napari_viewer, tmp_path, image_shape, axes_names
):
    """A label with all image axes except c can be saved for any image shape."""
    image_path = _make_zarr(tmp_path, image_shape, axes_names)
    img = open_ome_zarr_container(image_path, mode="r").get_image()

    laxes = _label_axes(axes_names)
    lshape = _label_shape(image_shape, axes_names)
    lscale = _label_scale(img, laxes)

    viewer = make_napari_viewer()
    layer = viewer.add_labels(
        np.zeros(lshape, dtype=np.uint32), name="seg", scale=lscale
    )
    saver = LabelSaverImage(viewer=viewer)
    assert saver._do_save_impl(**_save_kwargs(image_path, layer, "seg", laxes, _WM_NEW))
    assert "seg" in open_ome_zarr_container(image_path, mode="r").list_labels()


# ---------------------------------------------------------------------------
# Pattern 2: yx-only label with singleton expansion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("image_shape,axes_names", _SINGLETON_PARAMS)
def test_save_yx_label_singleton_expansion(
    make_napari_viewer, tmp_path, image_shape, axes_names
):
    """A 2D yx label is auto-expanded to match all-singleton non-yx axes."""
    image_path = _make_zarr(tmp_path, image_shape, axes_names)
    img = open_ome_zarr_container(image_path, mode="r").get_image()

    viewer = make_napari_viewer()
    lscale = (img.pixel_size.y, img.pixel_size.x)
    layer = viewer.add_labels(
        np.zeros((32, 32), dtype=np.uint32), name="seg2d", scale=lscale
    )
    saver = LabelSaverImage(viewer=viewer)
    assert saver._do_save_impl(
        **_save_kwargs(image_path, layer, "seg2d", "yx", _WM_NEW)
    )
    assert "seg2d" in open_ome_zarr_container(image_path, mode="r").list_labels()


# ---------------------------------------------------------------------------
# Pattern 3: 2× downsampled label (mode Overwrite full label)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("image_shape,axes_names", _ALL_PARAMS)
def test_save_downsampled_label(make_napari_viewer, tmp_path, image_shape, axes_names):
    """A label at 2× coarser yx resolution can be saved with mode Overwrite."""
    image_path = _make_zarr(tmp_path, image_shape, axes_names)
    img = open_ome_zarr_container(image_path, mode="r").get_image()

    laxes = _label_axes(axes_names)
    # Build downsampled shape and scale: halve y/x, keep other dims unchanged
    lshape = []
    lscale = []
    for a in laxes:
        img_size = img.shape[list(img.axes).index(a)]
        psize = float(getattr(img.pixel_size, a))
        if a in ("y", "x"):
            lshape.append(img_size // 2)
            lscale.append(psize * 2)
        else:
            lshape.append(img_size)
            lscale.append(psize)

    viewer = make_napari_viewer()
    layer = viewer.add_labels(
        np.zeros(tuple(lshape), dtype=np.uint32), name="seg_down", scale=tuple(lscale)
    )
    saver = LabelSaverImage(viewer=viewer)
    assert saver._do_save_impl(
        **_save_kwargs(image_path, layer, "seg_down", laxes, _WM_OVERWRITE)
    )
    assert "seg_down" in open_ome_zarr_container(image_path, mode="r").list_labels()


# ---------------------------------------------------------------------------
# Pattern 4: Partial-region append — pixels outside sub-region stay unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("image_shape,axes_names", _ALL_PARAMS)
def test_partial_write_preserves_outside(
    make_napari_viewer, tmp_path, image_shape, axes_names
):
    """After appending a 16×16 patch, pixels outside the patch remain zero."""
    image_path = _make_zarr(tmp_path, image_shape, axes_names)
    img = open_ome_zarr_container(image_path, mode="r").get_image()

    laxes = _label_axes(axes_names)
    lshape = _label_shape(image_shape, axes_names)
    lscale = _label_scale(img, laxes)
    pxy, pxx = img.pixel_size.y, img.pixel_size.x

    viewer = make_napari_viewer()
    saver = LabelSaverImage(viewer=viewer)

    # Step 1: create full-size zeros label
    full_layer = viewer.add_labels(
        np.zeros(lshape, dtype=np.uint32), name="seg", scale=lscale
    )
    assert saver._do_save_impl(
        **_save_kwargs(image_path, full_layer, "seg", laxes, _WM_NEW)
    ), "Initial full-label save failed"

    # Step 2: append a 16×16 ones patch at pixel offset y=8, x=8
    patch_shape = tuple(
        16 if a in ("y", "x") else img.shape[list(img.axes).index(a)] for a in laxes
    )
    patch_translate = tuple(
        8.0 * pxy if a == "y" else (8.0 * pxx if a == "x" else 0.0) for a in laxes
    )
    patch_layer = viewer.add_labels(
        np.ones(patch_shape, dtype=np.uint32),
        name="patch",
        scale=lscale,
        translate=patch_translate,
    )
    kwargs = _save_kwargs(image_path, patch_layer, "seg", laxes, _WM_APPEND)
    assert saver._do_save_impl(**kwargs), "Partial append failed"

    # Step 3: verify patch region is 1, strip above patch is still 0
    saved = np.array(
        open_ome_zarr_container(image_path, mode="r").get_label("seg").get_array()
    )
    assert np.all(saved[..., 8:24, 8:24] == 1), "Patch region should be 1"
    assert np.all(saved[..., 0:8, :] == 0), "Region above patch should remain 0"
