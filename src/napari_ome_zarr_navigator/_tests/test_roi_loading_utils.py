"""Unit tests for pure helper functions in roi_loading_utils."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from napari_ome_zarr_navigator.roi_loading_utils import (
    MASKED_METADATA_KEY,
    POSITION_METADATA_KEY,
    RoiTableInfo,
    _compute_scale,
    _roi_translate,
    find_matching_label_layer_index,
    is_masked_layer,
    is_shifted_layer,
    roi_table_info,
)


def _make_pixel_size(z=1.0, y=0.325, x=0.325):
    ps = MagicMock()
    ps.zyx = (z, y, x)
    ps.yx = (y, x)
    ps.tzyx = (1.0, z, y, x)
    return ps


def _make_ngio_img(z=1.0, y=0.325, x=0.325):
    img = MagicMock()
    img.pixel_size = _make_pixel_size(z, y, x)
    return img


# ---------------------------------------------------------------------------
# _compute_scale
# ---------------------------------------------------------------------------


class TestComputeScale:
    def test_2d(self):
        img = _make_ngio_img(y=0.325, x=0.325)
        assert _compute_scale(img, ndim=2) == (0.325, 0.325)

    def test_3d(self):
        img = _make_ngio_img(z=1.0, y=0.325, x=0.325)
        assert _compute_scale(img, ndim=3) == (1.0, 0.325, 0.325)

    def test_4d(self):
        img = _make_ngio_img(z=1.0, y=0.325, x=0.325)
        result = _compute_scale(img, ndim=4)
        assert len(result) == 4

    def test_unsupported_ndim_raises(self):
        img = _make_ngio_img()
        with pytest.raises(NotImplementedError):
            _compute_scale(img, ndim=5)


# ---------------------------------------------------------------------------
# find_matching_label_layer_index
# ---------------------------------------------------------------------------


class TestFindMatchingLabelLayerIndex:
    def _make_feature_table(self, reference_label: str):
        ft = MagicMock()
        ft.reference_label = reference_label
        return ft

    def test_exact_match_first_label(self):
        ft = self._make_feature_table("nuclei")
        assert find_matching_label_layer_index(ft, ["nuclei", "cells"]) == 0

    def test_exact_match_second_label(self):
        ft = self._make_feature_table("cells")
        assert find_matching_label_layer_index(ft, ["nuclei", "cells"]) == 1

    def test_fallback_to_zero_when_not_found(self):
        ft = self._make_feature_table("organoids")
        result = find_matching_label_layer_index(ft, ["nuclei", "cells"])
        assert result == 0

    def test_single_label_match(self):
        ft = self._make_feature_table("nuclei")
        assert find_matching_label_layer_index(ft, ["nuclei"]) == 0


# ---------------------------------------------------------------------------
# _compute_scale integration (uses real zarr from zenodo fixture)
# ---------------------------------------------------------------------------


def test_compute_scale_from_real_zarr(zenodo_zarr):
    """Verify _compute_scale returns sensible values from a real ngio image."""
    import ngio

    zarr_url = Path(zenodo_zarr[1]) / "B" / "03" / "0"
    container = ngio.open_ome_zarr_container(str(zarr_url))
    img = container.get_image()
    scale = _compute_scale(img, ndim=2)
    assert len(scale) == 2
    assert all(s > 0 for s in scale)


# ---------------------------------------------------------------------------
# ROI positioning: original position and registration translation
# ---------------------------------------------------------------------------


def _roi(name, y, x, z=0.0, **extras):
    """World-space ROI with optional *_original / translation_* extra columns."""
    from ngio import Roi

    return Roi.from_values(
        name=name,
        slices={"x": (x, 100.0), "y": (y, 50.0), "z": (z, 1.0)},
        space="world",
        **extras,
    )


# Two FOVs that overlap by 50 µm in x at acquisition, stored side by side
# (shifted apart) at their regular positions, as Fractal's converter does.
FOV_1 = _roi(
    "FOV_1", 0.0, 0.0, y_micrometer_original=-1000.0, x_micrometer_original=-2000.0
)
FOV_2 = _roi(
    "FOV_2",
    0.0,
    100.0,
    y_micrometer_original=-1000.0,
    x_micrometer_original=-1950.0,
    translation_y=3.0,
    translation_x=-1.5,
)
ORIGIN = {"y": -1000.0, "x": -2000.0}


class TestRoiTableInfo:
    def test_original_origin_and_translation(self):
        info = roi_table_info([FOV_1, FOV_2])
        assert info == RoiTableInfo(original_origin=ORIGIN, has_translation=True)

    def test_table_without_position_columns(self):
        assert roi_table_info([_roi("well", 0.0, 0.0)]) == RoiTableInfo()

    def test_minimum_ignores_nan_and_missing_rows(self):
        nan_roi = _roi(
            "nan",
            0.0,
            0.0,
            y_micrometer_original=float("nan"),
            x_micrometer_original=-9999.0,
        )
        info = roi_table_info([FOV_2, nan_roi, _roi("plain", 0.0, 0.0)])
        assert info.original_origin == {"y": -1000.0, "x": -1950.0}

    def test_z_origin_only_from_rois_with_z(self):
        with_z = _roi(
            "z",
            0.0,
            0.0,
            y_micrometer_original=0.0,
            x_micrometer_original=0.0,
            z_micrometer_original=-5.0,
        )
        info = roi_table_info([FOV_1, with_z])
        assert info.original_origin == {"y": -1000.0, "x": -2000.0, "z": -5.0}


class TestRoiTranslate:
    def test_regular_position_unchanged(self):
        assert _roi_translate(FOV_2, (10.0, 20.0)) == ((10.0, 120.0), None)

    def test_original_position_relative_to_table_minimum(self):
        translate, tag = _roi_translate(FOV_2, (10.0, 20.0), origin=ORIGIN)
        # Original overlap of 50 µm is restored, inside the well offset (10, 20).
        assert translate == (10.0, 70.0)
        assert tag == {"original": {"y": -1000.0, "x": -1950.0}}

    def test_original_missing_falls_back_to_regular(self):
        plain = _roi("plain", 5.0, 7.0)
        assert _roi_translate(plain, (0.0, 0.0), origin=ORIGIN) == ((5.0, 7.0), None)

    def test_translation_added_to_regular_position(self):
        translate, tag = _roi_translate(FOV_2, (0.0, 0.0), add_translation=True)
        assert translate == (3.0, 98.5)
        assert tag == {"translation": {"z": 0.0, "y": 3.0, "x": -1.5}}

    def test_translation_missing_axes_count_as_zero(self):
        only_y = _roi("t", 0.0, 0.0, translation_y=2.0)
        assert _roi_translate(only_y, (0.0, 0.0), add_translation=True)[0] == (2.0, 0.0)

    def test_without_translation_columns_stays_regular(self):
        assert _roi_translate(FOV_1, (0.0, 0.0), add_translation=True) == (
            (0.0, 0.0),
            None,
        )

    def test_original_and_translation_combined(self):
        translate, tag = _roi_translate(
            FOV_2, (10.0, 20.0), origin=ORIGIN, add_translation=True
        )
        assert translate == (13.0, 68.5)
        assert tag is not None
        assert set(tag) == {"original", "translation"}

    @pytest.mark.parametrize(
        ("ndim", "has_z_axis", "expected"),
        [
            (3, True, (4.0, 0.0, 0.0)),  # zyx
            (4, True, (0.0, 4.0, 0.0, 0.0)),  # tzyx
            (2, False, (0.0, 0.0)),  # z squeezed away: stays 2D
        ],
    )
    def test_z_offset_only_with_z_axis(self, ndim, has_z_axis, expected):
        roi = _roi(
            "z",
            0.0,
            0.0,
            y_micrometer_original=0.0,
            x_micrometer_original=0.0,
            z_micrometer_original=-1.0,
        )
        origin = {"y": 0.0, "x": 0.0, "z": -5.0}
        translate, _ = _roi_translate(
            roi, (0.0, 0.0), origin=origin, ndim=ndim, has_z_axis=has_z_axis
        )
        assert translate == expected


def test_is_shifted_layer():
    assert is_shifted_layer(
        MagicMock(metadata={POSITION_METADATA_KEY: {"original": {}}})
    )
    assert not is_shifted_layer(MagicMock(metadata={}))


def test_original_position_of_real_fractal_fov_table(zenodo_zarr):
    """Fractal stores x/y_micrometer as original minus the per-well minimum, so
    non-overlapping FOVs land at their regular position."""
    import ngio

    zarr_url = Path(zenodo_zarr[1]) / "B" / "03" / "0"
    container = ngio.open_ome_zarr_container(str(zarr_url))
    rois = container.get_generic_roi_table("FOV_ROI_table").rois()
    info = roi_table_info(rois)
    assert info.original_origin is not None
    assert not info.has_translation
    fov_2 = next(r for r in rois if r.name == "FOV_2")
    translate, tag = _roi_translate(fov_2, (0.0, 0.0), origin=info.original_origin)
    assert translate == pytest.approx((0.0, 416.0))
    assert tag is not None and "original" in tag


def test_roi_table_info_flags_masking_tables():
    assert roi_table_info([FOV_1], is_masking_table=True).is_masking_table
    assert not roi_table_info([FOV_1]).is_masking_table


def test_is_masked_layer():
    assert is_masked_layer(MagicMock(metadata={MASKED_METADATA_KEY: {"label": 6}}))
    assert not is_masked_layer(MagicMock(metadata={POSITION_METADATA_KEY: {}}))
