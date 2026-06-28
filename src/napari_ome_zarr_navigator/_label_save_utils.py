"""Pure calculation helpers for LabelSaverImage.

All functions here are stateless — they operate only on numpy arrays and ngio
image objects, with no dependency on napari or magicgui widgets.
"""

import numpy as np

_WM_NEW = "Save as new label"
_WM_EDIT = "Edit existing label"
_WM_RESET = "Reset existing label"
_WRITE_MODES = [_WM_NEW, _WM_EDIT, _WM_RESET]

_SCALE_REL_TOL = 1e-4  # 0.01% — allows float rounding, rejects factor-2 mismatches


def _validate_scale_matches_existing_label(
    layer_scale: tuple,
    axes_str: str,
    label_name: str,
    container,
) -> tuple[bool, str]:
    """Check that layer_scale matches the full-resolution pixel size of an existing label.

    Only checks y and x axes — those are the ones that control pixel indexing.
    Returns (True, "") on a match, (False, message) on a mismatch.
    """
    try:
        label_obj = container.get_label(label_name)
    except Exception:  # noqa: BLE001
        return True, ""

    axes_list = list(axes_str)
    for ax in ("y", "x"):
        if ax not in axes_list:
            continue
        layer_ax = abs(float(layer_scale[axes_list.index(ax)]))
        label_ax = abs(float(getattr(label_obj.pixel_size, ax)))
        if label_ax > 0 and abs(layer_ax - label_ax) / label_ax > _SCALE_REL_TOL:
            return (
                False,
                f"Label layer scale ({layer_ax:.6g} along '{ax}') does not "
                f"match the full pixel size of '{label_name}' ({label_ax:.6g}). "
                f"To edit this label, work at the resolution of '{label_name}'.",
            )
    return True, ""


def _expand_label_dims(
    label_array: np.ndarray,
    layer_scale: tuple,
    axes_str: str,
    img,
) -> tuple[np.ndarray, tuple, str]:
    """Expand label with singleton dims to match the image's non-channel axes.

    If the image has extra axes (e.g. z=1) not present in the label, they are
    inserted as singleton dimensions.  Raises ValueError when a missing axis has
    size > 1 (can't auto-expand).
    """
    img_non_c_axes = [a for a in img.axes if a != "c"]
    if label_array.ndim == len(img_non_c_axes):
        return label_array, layer_scale, axes_str

    label_axes_list = list(axes_str)
    layer_scale_list = list(layer_scale)

    for i, a in enumerate(img_non_c_axes):
        if a not in label_axes_list:
            img_a_idx = list(img.axes).index(a)
            img_a_size = img.shape[img_a_idx]
            if img_a_size == 1:
                label_array = np.expand_dims(label_array, axis=i)
                label_axes_list.insert(i, a)
                layer_scale_list.insert(i, float(getattr(img.pixel_size, a)))
            else:
                raise ValueError(
                    f"Cannot save a {len(axes_str)}D label for an image whose "
                    f"'{a}' axis has size {img_a_size}. "
                    "The label must cover the same number of dimensions as the image."
                )

    return label_array, tuple(layer_scale_list), "".join(label_axes_list)


def _insert_c_dim(shape: tuple, img) -> tuple:
    """Insert the c dimension into *shape* when the reference image has a c axis.

    ngio's ``derive_label`` validates ``len(shape) == len(ref_image.shape)``
    *before* it applies ``channels_policy='squeeze'``, so a label shape
    (z, y, x) is rejected for a (c, z, y, x) image even though the c axis
    is removed by the policy immediately afterwards.

    Workaround: insert the image's c size at the correct axis position so
    the length check passes; ngio then squeezes it out, leaving a label
    without a channel axis.  For images without c the shape is returned
    unchanged.

    See https://github.com/BioVisionCenter/ngio/issues/195
    """
    if "c" not in img.axes:
        return shape
    c_idx = list(img.axes).index("c")
    c_size = img.shape[c_idx]
    return shape[:c_idx] + (c_size,) + shape[c_idx:]


def _validate_tz(
    label_array: np.ndarray,
    axes_str: str,
    img,
) -> tuple[bool, str]:
    """Check z plane count and timepoint count (no yx extent check)."""
    axes_list = list(axes_str)
    img_axes = list(img.axes)

    if "z" in img_axes and "z" in axes_list:
        img_z = img.shape[img_axes.index("z")]
        lbl_z = label_array.shape[axes_list.index("z")]
        if img_z != lbl_z:
            return (
                False,
                f"Label has {lbl_z} z-plane(s) but the image has {img_z}. "
                "The number of z-planes must match.",
            )

    if "t" in img_axes and "t" in axes_list:
        img_t = img.shape[img_axes.index("t")]
        lbl_t = label_array.shape[axes_list.index("t")]
        if img_t != lbl_t:
            return (
                False,
                f"Label has {lbl_t} timepoint(s) but the image has {img_t}. "
                "The number of timepoints must match.",
            )

    return True, ""


def _extract_pixel_sizes(layer_scale: tuple, axes_str: str, img) -> tuple:
    """Return (pixelsize_yx, z_spacing, time_spacing) for derive_label."""
    axes_list = list(axes_str)
    pixelsize_yx = None
    z_spacing = None
    time_spacing = None
    if "y" in axes_list and "x" in axes_list:
        y_idx = axes_list.index("y")
        x_idx = axes_list.index("x")
        pixelsize_yx = (
            abs(float(layer_scale[y_idx])),
            abs(float(layer_scale[x_idx])),
        )
    if "z" in axes_list:
        z_spacing = abs(float(layer_scale[axes_list.index("z")]))
    if "t" in axes_list:
        time_spacing = float(img.pixel_size.t)
    return pixelsize_yx, z_spacing, time_spacing


def _compute_write_region(
    layer_translate: tuple,
    layer_scale: tuple,
    plate_translation: tuple,
    label_array: np.ndarray,
    axes_str: str,
    img,
) -> tuple[int, int, int, int, int, int, bool]:
    """Compute pixel offset and full-image dims for sub-ROI write modes.

    Returns (y0, x0, y1, x1, full_h, full_w, is_partial).
    layer_translate includes the plate/well offset; plate_translation is
    subtracted to get the image-relative world offset.
    """
    axes_list = list(axes_str)
    img_axes = list(img.axes)

    y_scale = abs(float(layer_scale[axes_list.index("y")])) if "y" in axes_list else 1.0
    x_scale = abs(float(layer_scale[axes_list.index("x")])) if "x" in axes_list else 1.0

    y_world = float(layer_translate[-2]) - float(plate_translation[0])
    x_world = float(layer_translate[-1]) - float(plate_translation[1])
    y0_px = round(y_world / y_scale)
    x0_px = round(x_world / x_scale)

    img_y = img.shape[img_axes.index("y")] if "y" in img_axes else 0
    img_x = img.shape[img_axes.index("x")] if "x" in img_axes else 0
    full_h = round(img_y * img.pixel_size.y / y_scale)
    full_w = round(img_x * img.pixel_size.x / x_scale)

    label_h = label_array.shape[axes_list.index("y")] if "y" in axes_list else full_h
    label_w = label_array.shape[axes_list.index("x")] if "x" in axes_list else full_w

    y1_px = min(y0_px + label_h, full_h)
    x1_px = min(x0_px + label_w, full_w)

    if y0_px < 0 or x0_px < 0 or y0_px >= full_h or x0_px >= full_w:
        raise ValueError(
            f"Label offset (y={y0_px}, x={x0_px}) is outside the full image "
            f"({full_h}×{full_w} px). Cannot write label."
        )

    is_partial = y0_px != 0 or x0_px != 0 or y1_px != full_h or x1_px != full_w
    return y0_px, x0_px, y1_px, x1_px, full_h, full_w, is_partial


def _apply_write_region(
    label_obj,
    label_array: np.ndarray,
    y0: int,
    x0: int,
    y1: int,
    x1: int,
    full_shape: tuple,
    is_partial: bool,
    write_mode: str,
) -> None:
    """Write label_array into label_obj at the computed region, then consolidate."""
    if write_mode == _WM_EDIT:
        label_obj.set_array(label_array, y=slice(y0, y1), x=slice(x0, x1))
    else:
        if is_partial:
            label_obj.set_array(np.zeros(full_shape, dtype=label_array.dtype))
            label_obj.set_array(label_array, y=slice(y0, y1), x=slice(x0, x1))
        else:
            label_obj.set_array(label_array)
    label_obj.consolidate()
