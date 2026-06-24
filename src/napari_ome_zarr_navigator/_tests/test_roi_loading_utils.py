"""Unit tests for pure helper functions in roi_loading_utils."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from napari_ome_zarr_navigator.roi_loading_utils import (
    _compute_scale,
    find_matching_label_layer_index,
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
