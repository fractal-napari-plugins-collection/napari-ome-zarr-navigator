"""
Tests for LabelSaverImage remote-save path.

These tests exercise the private ngio.images._label.derive_label API so that
a ngio version bump that removes or renames that symbol fails here immediately,
giving early notice to update the remote-save implementation.
"""

from pathlib import Path

import numpy as np
from ngio import create_synthetic_ome_zarr, open_label, open_ome_zarr_container

from napari_ome_zarr_navigator._label_save_utils import (
    _WM_EDIT,
    _WM_NEW,
    _WM_RESET,
)
from napari_ome_zarr_navigator.label_saver import LabelSaverImage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_zarr(path, shape=(1, 1, 32, 32), axes_names=("t", "c", "y", "x")):
    create_synthetic_ome_zarr(
        store=str(path),
        shape=shape,
        axes_names=list(axes_names),
        table_backend="csv",
    )
    return str(path)


def _base_kwargs(
    zarr_url, layer, write_mode=_WM_NEW, output_folder=None, axes_str="yx"
):
    return {
        "store": zarr_url,
        "label_array": np.asarray(layer.data),
        "layer_scale": tuple(layer.scale),
        "layer_translate": tuple(float(v) for v in layer.translate),
        "label_name": layer.name,
        "axes_str": axes_str,
        "write_mode": write_mode,
        "save_masking_roi": False,
        "masking_roi_table_name": "",
        "masking_roi_backend": "csv",
        "output_folder": output_folder,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_remote_save_new_label(make_napari_viewer, tmp_path):
    """NEW mode: creates output_folder/labels/<name>/ with valid OME-Zarr label."""
    image_path = _make_zarr(tmp_path / "image.zarr")
    output_folder = str(tmp_path / "output")

    img = open_ome_zarr_container(image_path, mode="r").get_image()
    viewer = make_napari_viewer()
    layer = viewer.add_labels(
        np.zeros((32, 32), dtype=np.uint32),
        name="seg",
        scale=(img.pixel_size.y, img.pixel_size.x),
    )
    saver = LabelSaverImage(viewer=viewer)

    result = saver._do_save_impl(
        **_base_kwargs(image_path, layer, output_folder=output_folder)
    )

    assert result, "Remote NEW save failed"
    label_dir = Path(output_folder) / "labels" / "seg"
    assert (label_dir / ".zattrs").exists(), "Label .zattrs not written"
    lbl = open_label(str(label_dir))
    arr = lbl.get_array()
    assert arr.shape[-2:] == (32, 32)


def test_remote_save_new_label_does_not_write_to_remote(make_napari_viewer, tmp_path):
    """Remote save must not modify the remote OME-Zarr container."""
    image_path = _make_zarr(tmp_path / "image.zarr")
    output_folder = str(tmp_path / "output")

    viewer = make_napari_viewer()
    layer = viewer.add_labels(np.zeros((32, 32), dtype=np.uint32), name="seg")
    saver = LabelSaverImage(viewer=viewer)

    saver._do_save_impl(**_base_kwargs(image_path, layer, output_folder=output_folder))

    remote_labels = open_ome_zarr_container(image_path, mode="r").list_labels()
    assert "seg" not in remote_labels, (
        "Label was unexpectedly written to remote container"
    )


def test_remote_save_new_label_fails_if_already_exists(make_napari_viewer, tmp_path):
    """NEW mode fails gracefully when the label already exists at the output folder."""
    image_path = _make_zarr(tmp_path / "image.zarr")
    output_folder = str(tmp_path / "output")

    img = open_ome_zarr_container(image_path, mode="r").get_image()
    viewer = make_napari_viewer()
    layer = viewer.add_labels(
        np.zeros((32, 32), dtype=np.uint32),
        name="seg",
        scale=(img.pixel_size.y, img.pixel_size.x),
    )
    saver = LabelSaverImage(viewer=viewer)

    kwargs = _base_kwargs(image_path, layer, output_folder=output_folder)
    assert saver._do_save_impl(**kwargs), "First save should succeed"
    assert not saver._do_save_impl(**kwargs), (
        "Second NEW save should fail (already exists)"
    )


def test_remote_save_reset_overwrites(make_napari_viewer, tmp_path):
    """RESET mode overwrites an existing label at the output folder."""
    image_path = _make_zarr(tmp_path / "image.zarr")
    output_folder = str(tmp_path / "output")

    img = open_ome_zarr_container(image_path, mode="r").get_image()
    viewer = make_napari_viewer()
    psy, psx = img.pixel_size.y, img.pixel_size.x
    data = np.zeros((32, 32), dtype=np.uint32)
    data[8:16, 8:16] = 1
    layer = viewer.add_labels(data, name="seg", scale=(psy, psx))
    saver = LabelSaverImage(viewer=viewer)

    # First save (NEW)
    saver._do_save_impl(**_base_kwargs(image_path, layer, output_folder=output_folder))

    # Reset with all-zeros; use a distinct layer name to avoid napari auto-rename
    zeros_layer = viewer.add_labels(
        np.zeros((32, 32), dtype=np.uint32), name="seg_zeros", scale=(psy, psx)
    )
    kwargs = _base_kwargs(
        image_path, zeros_layer, write_mode=_WM_RESET, output_folder=output_folder
    )
    kwargs["label_name"] = "seg"  # write to the existing "seg" label
    result = saver._do_save_impl(**kwargs)
    assert result, "RESET save failed"
    lbl = open_label(str(Path(output_folder) / "labels" / "seg"))
    assert np.all(lbl.get_array() == 0), "RESET should have written all zeros"


def test_remote_save_edit_patches_existing(make_napari_viewer, tmp_path):
    """EDIT mode patches the loaded region of an existing local label."""
    image_path = _make_zarr(tmp_path / "image.zarr")
    output_folder = str(tmp_path / "output")

    img = open_ome_zarr_container(image_path, mode="r").get_image()
    viewer = make_napari_viewer()
    psy, psx = img.pixel_size.y, img.pixel_size.x
    saver = LabelSaverImage(viewer=viewer)

    # Step 1: create full-size zeros label
    full_layer = viewer.add_labels(
        np.zeros((32, 32), dtype=np.uint32), name="seg", scale=(psy, psx)
    )
    assert saver._do_save_impl(
        **_base_kwargs(image_path, full_layer, output_folder=output_folder)
    ), "Initial save failed"

    # Step 2: patch a 16×16 region with ones; distinct name to avoid napari auto-rename
    patch_layer = viewer.add_labels(
        np.ones((16, 16), dtype=np.uint32),
        name="patch",
        scale=(psy, psx),
        translate=(8 * psy, 8 * psx),
    )
    kwargs = _base_kwargs(
        image_path, patch_layer, write_mode=_WM_EDIT, output_folder=output_folder
    )
    kwargs["label_name"] = "seg"  # edit the existing "seg" label
    result = saver._do_save_impl(**kwargs)
    assert result, "EDIT save failed"

    lbl = open_label(str(Path(output_folder) / "labels" / "seg"))
    arr = lbl.get_array()
    assert np.all(arr[..., 8:24, 8:24] == 1), "Patch region should be 1"
    assert np.all(arr[..., 0:8, :] == 0), "Region above patch should remain 0"


def test_remote_save_edit_fails_if_not_exists(make_napari_viewer, tmp_path):
    """EDIT mode fails gracefully when the label doesn't exist at the output folder."""
    image_path = _make_zarr(tmp_path / "image.zarr")
    output_folder = str(tmp_path / "output")

    viewer = make_napari_viewer()
    layer = viewer.add_labels(np.zeros((32, 32), dtype=np.uint32), name="seg")
    saver = LabelSaverImage(viewer=viewer)

    result = saver._do_save_impl(
        **_base_kwargs(
            image_path, layer, write_mode=_WM_EDIT, output_folder=output_folder
        )
    )
    assert not result, "EDIT should fail when label doesn't exist"


def test_remote_save_with_masking_roi(make_napari_viewer, tmp_path):
    """Remote save with masking ROI writes table to output_folder/tables/."""
    image_path = _make_zarr(tmp_path / "image.zarr")
    output_folder = str(tmp_path / "output")

    img = open_ome_zarr_container(image_path, mode="r").get_image()
    viewer = make_napari_viewer()
    data = np.zeros((32, 32), dtype=np.uint32)
    data[4:8, 4:8] = 1
    layer = viewer.add_labels(
        data, name="seg", scale=(img.pixel_size.y, img.pixel_size.x)
    )
    saver = LabelSaverImage(viewer=viewer)

    kwargs = _base_kwargs(image_path, layer, output_folder=output_folder)
    kwargs["save_masking_roi"] = True
    kwargs["masking_roi_table_name"] = "seg_masking_ROI_table"
    result = saver._do_save_impl(**kwargs)

    assert result, "Remote save with masking ROI failed"
    tables_path = Path(output_folder) / "tables"
    assert tables_path.is_dir(), "tables/ folder not created"
    assert any(tables_path.iterdir()), "tables/ folder is empty"


def test_remote_save_unavailable_guard(make_napari_viewer, tmp_path, monkeypatch):
    """When the private API guard is False, save fails with an informative error."""
    import napari_ome_zarr_navigator.label_saver as ls_module

    monkeypatch.setattr(ls_module, "_NGIO_REMOTE_LABEL_AVAILABLE", False)

    image_path = _make_zarr(tmp_path / "image.zarr")
    output_folder = str(tmp_path / "output")

    viewer = make_napari_viewer()
    layer = viewer.add_labels(np.zeros((32, 32), dtype=np.uint32), name="seg")
    saver = LabelSaverImage(viewer=viewer)

    result = saver._do_save_impl(
        **_base_kwargs(image_path, layer, output_folder=output_folder)
    )
    assert not result, "Should return False when derive_label is unavailable"
