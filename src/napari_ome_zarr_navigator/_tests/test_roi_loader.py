from pathlib import Path

import napari.layers
import numpy as np
import pytest

from napari_ome_zarr_navigator.plate_browser import PlateBrowser
from napari_ome_zarr_navigator.roi_loader import (
    _NO_ROI_TABLE,
    ROILoader,
    ROILoaderImage,
    ROILoaderPlate,
)


def _init_roi_loader_image(viewer, zarr_url, qtbot):
    """Create a ROILoaderImage and wait for init signals."""
    roi_loader = ROILoaderImage(viewer, zarr_url=str(zarr_url))
    with qtbot.waitSignal(
        roi_loader.image_changed_event.roi_tables_updated, timeout=5000
    ):
        pass
    with qtbot.waitSignal(
        roi_loader.image_changed_event.roi_choices_updated, timeout=5000
    ):
        pass
    return roi_loader


def test_roi_loader_init(make_napari_viewer):
    viewer = make_napari_viewer()
    _ = viewer.add_image(np.random.random((100, 100)))
    _ = ROILoader(viewer)


def test_plate(make_napari_viewer, zenodo_zarr, qtbot):
    viewer = make_napari_viewer()
    plate_url = zenodo_zarr[1]
    plate_browser = PlateBrowser(viewer=viewer)
    roi_loader = ROILoaderPlate(
        viewer,
        plate_store=plate_url,
        row="B",
        col="03",
        plate_browser=plate_browser,
        is_plate=True,
    )
    with qtbot.waitSignal(
        roi_loader.image_changed_event.roi_tables_updated, timeout=5000
    ):
        pass

    with qtbot.waitSignal(
        roi_loader.image_changed_event.roi_choices_updated, timeout=5000
    ):
        pass

    assert roi_loader._zarr_picker.choices == ("0",)
    assert roi_loader._zarr_picker.value == "0"

    # Set parameters (level picker now shows resolution strings; default Multi-resolution
    # mode means run() uses dask pyramid regardless of level picker value)
    roi_loader._roi_table_picker.value = "FOV_ROI_table"
    roi_loader._roi_picker.value = "FOV_1"
    roi_loader._channel_picker.value = ["DAPI"]
    roi_loader._label_picker.value = ["nuclei"]
    roi_loader._feature_picker.value = ["measurements"]
    roi_loader._remove_old_labels_box.value = False

    roi_loader.run()
    with qtbot.waitSignal(roi_loader.image_changed_event.load_finished, timeout=10000):
        pass

    image_layers = [
        layer for layer in viewer.layers if isinstance(layer, napari.layers.Image)
    ]
    label_layers = [
        layer for layer in viewer.layers if isinstance(layer, napari.layers.Labels)
    ]
    assert len(image_layers) == 1
    assert len(label_layers) == 1
    # Images load as multiscale pyramid by default; finest level is (2160, 2560)
    assert image_layers[0].multiscale
    assert image_layers[0].data[0].shape == (2160, 2560)
    # Labels load as fixed resolution by default
    assert not label_layers[0].multiscale
    assert label_layers[0].data.shape == (540, 640)
    assert "label" in label_layers[0].features.columns


def test_roi_loader(make_napari_viewer, zenodo_zarr, qtbot):
    viewer = make_napari_viewer()
    zarr_url = Path(zenodo_zarr[1]) / "B" / "03" / "0"
    roi_loader = _init_roi_loader_image(viewer, zarr_url, qtbot)

    assert str(roi_loader.zarr_selector._file_picker.value) == str(zarr_url)

    # Check parameters
    assert roi_loader._roi_table_picker.value == "FOV_ROI_table"
    assert roi_loader._roi_picker.value == "FOV_1"
    assert roi_loader._channel_picker.choices == ("DAPI", "nanog", "Lamin B1")
    # Level picker now shows physical resolution strings (e.g. "0.163 micrometer")
    assert roi_loader._level_picker.value is not None
    assert roi_loader._level_picker.value == roi_loader._level_picker.choices[0]
    assert roi_loader._label_picker.choices == ("nuclei",)
    assert roi_loader._feature_picker.choices == ("measurements", "measurements_csv")
    assert not roi_loader._remove_old_labels_box.value
    # Whole-image sentinel appears last even when ROI tables exist
    assert roi_loader._roi_table_picker.choices[-1] == _NO_ROI_TABLE

    roi_loader._roi_table_picker.value = "FOV_ROI_table"
    roi_loader._roi_picker.value = "FOV_1"
    roi_loader._channel_picker.value = ["DAPI"]
    roi_loader._label_picker.value = ["nuclei"]
    roi_loader._feature_picker.value = ["measurements"]
    roi_loader._remove_old_labels_box.value = False

    roi_loader.run()
    with qtbot.waitSignal(roi_loader.image_changed_event.load_finished, timeout=10000):
        pass

    image_layers = [
        layer for layer in viewer.layers if isinstance(layer, napari.layers.Image)
    ]
    label_layers = [
        layer for layer in viewer.layers if isinstance(layer, napari.layers.Labels)
    ]
    assert len(image_layers) == 1
    assert len(label_layers) == 1
    # Images load as multiscale pyramid by default; finest level is (2160, 2560)
    assert image_layers[0].multiscale
    assert image_layers[0].data[0].shape == (2160, 2560)
    # Labels load as fixed resolution by default
    assert not label_layers[0].multiscale
    assert label_layers[0].data.shape == (540, 640)
    assert "label" in label_layers[0].features.columns


def test_roi_loader_fixed_resolution(make_napari_viewer, zenodo_zarr, qtbot):
    viewer = make_napari_viewer()
    zarr_url = Path(zenodo_zarr[1]) / "B" / "03" / "0"
    roi_loader = _init_roi_loader_image(viewer, zarr_url, qtbot)

    roi_loader._image_loading_mode.value = "Fixed resolution"
    assert roi_loader._level_picker.enabled

    roi_loader._roi_table_picker.value = "FOV_ROI_table"
    roi_loader._roi_picker.value = "FOV_1"
    roi_loader._channel_picker.value = ["DAPI"]
    roi_loader._label_picker.value = ["nuclei"]
    roi_loader._remove_old_labels_box.value = False

    roi_loader.run()
    with qtbot.waitSignal(roi_loader.image_changed_event.load_finished, timeout=10000):
        pass

    image_layers = [
        layer for layer in viewer.layers if isinstance(layer, napari.layers.Image)
    ]
    label_layers = [
        layer for layer in viewer.layers if isinstance(layer, napari.layers.Labels)
    ]
    assert len(image_layers) == 1
    assert len(label_layers) == 1
    # Fixed resolution: single array; image at finest level, labels at their default level
    assert not image_layers[0].multiscale
    assert image_layers[0].data.shape == (2160, 2560)
    assert not label_layers[0].multiscale
    assert label_layers[0].data.shape == (540, 640)


def test_roi_loader_whole_image(make_napari_viewer, zenodo_zarr, qtbot):
    viewer = make_napari_viewer()
    zarr_url = Path(zenodo_zarr[1]) / "B" / "03" / "0"
    roi_loader = _init_roi_loader_image(viewer, zarr_url, qtbot)

    # Sentinel must appear last even when ROI tables are present
    assert roi_loader._roi_table_picker.choices[-1] == _NO_ROI_TABLE
    roi_loader._roi_table_picker.value = _NO_ROI_TABLE
    roi_loader._channel_picker.value = ["DAPI"]
    roi_loader._label_picker.value = []
    roi_loader._remove_old_labels_box.value = False

    roi_loader.run()
    with qtbot.waitSignal(roi_loader.image_changed_event.load_finished, timeout=10000):
        pass

    image_layers = [
        layer for layer in viewer.layers if isinstance(layer, napari.layers.Image)
    ]
    assert len(image_layers) == 1
    # Whole image in multiscale mode; wider than a single FOV (2 sites = 5120 px wide)
    assert image_layers[0].multiscale
    assert image_layers[0].data[0].shape == (2160, 5120)


@pytest.mark.parametrize("roi_table_picker_value", ["FOV_ROI_table", _NO_ROI_TABLE])
def test_roi_table_choices_always_include_sentinel(
    make_napari_viewer, zenodo_zarr, qtbot, roi_table_picker_value
):
    """_NO_ROI_TABLE always appears as the last choice regardless of available tables."""
    viewer = make_napari_viewer()
    zarr_url = Path(zenodo_zarr[1]) / "B" / "03" / "0"
    roi_loader = _init_roi_loader_image(viewer, zarr_url, qtbot)
    assert roi_loader._roi_table_picker.choices[-1] == _NO_ROI_TABLE


def test_refresh_sees_labels_and_tables_written_elsewhere(
    make_napari_viewer, tmp_path, qtbot
):
    """The loader's cache=True container must pick up writes from other handles.

    Label saver and ROI annotator write through their own containers and then
    call refresh_labels / refresh_roi_tables on the loader.
    """
    from ngio import create_synthetic_ome_zarr, open_ome_zarr_container

    zarr_url = tmp_path / "synthetic.zarr"
    create_synthetic_ome_zarr(str(zarr_url), shape=(1, 64, 64), overwrite=True)
    viewer = make_napari_viewer()
    roi_loader = _init_roi_loader_image(viewer, zarr_url, qtbot)
    assert "new_label" not in roi_loader._label_picker.choices

    writer = open_ome_zarr_container(str(zarr_url), mode="a", cache=False)
    label = writer.derive_label("new_label")
    writer.add_table("new_ROI_table", label.build_masking_roi_table())

    roi_loader.refresh_labels()
    assert "new_label" in roi_loader._label_picker.choices

    with qtbot.waitSignal(
        roi_loader.image_changed_event.roi_tables_updated, timeout=5000
    ):
        roi_loader.refresh_roi_tables()
    assert "new_ROI_table" in roi_loader._roi_table_picker.choices


# ---------------------------------------------------------------------------
# ROI position options: original position and registration translation
# ---------------------------------------------------------------------------


def _make_image_with_position_columns(path: Path) -> tuple[float, float]:
    """Synthetic image whose FOV_ROI_table carries *_original and translation_*.

    The two FOVs sit side by side at their regular positions but overlapped by a
    quarter of their width at acquisition. FOV_2 also has a registration
    translation. Returns the FOV size (height, width) in µm.
    """
    from ngio import Roi, RoiTable, create_synthetic_ome_zarr

    container = create_synthetic_ome_zarr(
        store=str(path), shape=(1, 64, 128), table_backend="csv"
    )
    ps = container.get_image().pixel_size
    h, w = 64 * ps.y, 64 * ps.x

    def fov(name, x, x_original, **extras):
        return Roi.from_values(
            name=name,
            slices={"x": (x, w), "y": (0.0, h), "z": (0.0, 1.0)},
            space="world",
            y_micrometer_original=-300.0,
            x_micrometer_original=x_original,
            **extras,
        )

    table = RoiTable(
        rois=[
            fov("FOV_1", 0.0, -500.0),
            fov("FOV_2", w, -500.0 + 0.75 * w, translation_y=2.0, translation_x=-1.0),
        ]
    )
    container.add_table("FOV_ROI_table", table, backend="csv", overwrite=True)
    return h, w


def _select_roi_table(roi_loader, qtbot, name: str, expected_roi: str) -> None:
    """Select a ROI table and wait until its ROIs are listed."""
    roi_loader._roi_table_picker.value = name
    qtbot.waitUntil(
        lambda: expected_roi in roi_loader._roi_picker.choices, timeout=10000
    )


def _load_fov_2(viewer, roi_loader, qtbot, original: bool, translation: bool):
    viewer.layers.clear()
    roi_loader._roi_picker.value = "FOV_2"
    roi_loader._channel_picker.value = [roi_loader._channel_picker.choices[0]]
    roi_loader._label_picker.value = ["nuclei"]
    roi_loader._original_position_box.value = original
    roi_loader._add_translation_box.value = translation
    with qtbot.waitSignal(roi_loader.image_changed_event.load_finished, timeout=10000):
        roi_loader.run()
    image = next(lyr for lyr in viewer.layers if isinstance(lyr, napari.layers.Image))
    label = next(lyr for lyr in viewer.layers if isinstance(lyr, napari.layers.Labels))
    return image, label


def test_roi_position_options_availability(make_napari_viewer, tmp_path, qtbot):
    zarr_url = tmp_path / "positions.zarr"
    _make_image_with_position_columns(zarr_url)
    roi_loader = _init_roi_loader_image(make_napari_viewer(), zarr_url, qtbot)

    _select_roi_table(roi_loader, qtbot, "FOV_ROI_table", "FOV_2")
    assert roi_loader._original_position_box.enabled
    assert roi_loader._add_translation_box.enabled

    roi_loader._original_position_box.value = True
    _select_roi_table(roi_loader, qtbot, "well_ROI_table", "image")
    assert not roi_loader._original_position_box.enabled
    assert not roi_loader._add_translation_box.enabled

    # The choice survives switching back to a table with the columns
    _select_roi_table(roi_loader, qtbot, "FOV_ROI_table", "FOV_2")
    assert roi_loader._original_position_box.enabled
    assert roi_loader._original_position_box.value


@pytest.mark.parametrize(
    ("original", "translation", "expected", "tag_keys"),
    [
        (False, False, lambda h, w: (0.0, w), None),
        (True, False, lambda h, w: (0.0, 0.75 * w), {"original"}),
        (False, True, lambda h, w: (2.0, w - 1.0), {"translation"}),
        (True, True, lambda h, w: (2.0, 0.75 * w - 1.0), {"original", "translation"}),
    ],
)
def test_roi_position_options_place_layers(
    make_napari_viewer, tmp_path, qtbot, original, translation, expected, tag_keys
):
    from napari_ome_zarr_navigator.roi_loading_utils import POSITION_METADATA_KEY

    zarr_url = tmp_path / "positions.zarr"
    h, w = _make_image_with_position_columns(zarr_url)
    viewer = make_napari_viewer()
    roi_loader = _init_roi_loader_image(viewer, zarr_url, qtbot)
    _select_roi_table(roi_loader, qtbot, "FOV_ROI_table", "FOV_2")

    image, label = _load_fov_2(viewer, roi_loader, qtbot, original, translation)

    for layer in (image, label):
        assert tuple(layer.translate[-2:]) == pytest.approx(expected(h, w))
        tag = layer.metadata.get(POSITION_METADATA_KEY)
        assert (set(tag) if tag else None) == tag_keys


# Content-based checks: the loaded layers must show the same pixels wherever
# they overlap in world coordinates, which validates the positions end to end.

_PS = 0.5  # µm per pixel


def _write_image(path: Path, image, fovs) -> None:
    """Single-channel 2D OME-Zarr with a FOV_ROI_table of (name, x_px, extras)."""
    from ngio import Roi, RoiTable, create_ome_zarr_from_array

    h = image.shape[0]
    container = create_ome_zarr_from_array(
        str(path),
        image[None],
        pixelsize=_PS,
        axes_names=["c", "y", "x"],
        channels_meta=["DAPI"],
        levels=1,
    )
    rois = [
        Roi.from_values(
            name=name,
            slices={"x": (x_px * _PS, 64 * _PS), "y": (0.0, h * _PS), "z": (0.0, 1.0)},
            space="world",
            **extras,
        )
        for name, x_px, extras in fovs
    ]
    container.add_table(
        "FOV_ROI_table", RoiTable(rois=rois), backend="csv", overwrite=True
    )


def _load_image_roi(viewer, roi_loader, qtbot, roi, original=False, translation=False):
    """Load one ROI (single channel, fixed resolution) and return the new layer."""
    before = set(viewer.layers)
    roi_loader._image_loading_mode.value = "Fixed resolution"
    roi_loader._roi_picker.value = roi
    roi_loader._channel_picker.value = ["DAPI"]
    roi_loader._label_picker.value = []
    roi_loader._original_position_box.value = original
    roi_loader._add_translation_box.value = translation
    with qtbot.waitSignal(roi_loader.image_changed_event.load_finished, timeout=10000):
        roi_loader.run()
    (layer,) = [lyr for lyr in viewer.layers if lyr not in before]
    return layer


def _world_overlap(a, b):
    """Pixels of two 2D layers (same scale) on their overlap in world space."""
    offset = np.round(
        (np.asarray(b.translate) - np.asarray(a.translate)) / np.asarray(a.scale)
    ).astype(int)  # pixel i of b sits on pixel i + offset of a
    a_data, b_data = np.asarray(a.data), np.asarray(b.data)
    lo = np.maximum(0, -offset)
    hi = np.minimum(b_data.shape, np.asarray(a_data.shape) - offset)
    if np.any(hi <= lo):
        return None
    b_part = b_data[lo[0] : hi[0], lo[1] : hi[1]]
    a_part = a_data[
        lo[0] + offset[0] : hi[0] + offset[0], lo[1] + offset[1] : hi[1] + offset[1]
    ]
    return a_part, b_part


def test_original_position_restores_fov_overlap(make_napari_viewer, tmp_path, qtbot):
    """Two FOVs overlapping by 16 px at acquisition, stored side by side."""
    scene = np.random.default_rng(0).integers(1, 60000, (64, 112), dtype=np.uint16)
    stored = np.concatenate([scene[:, 0:64], scene[:, 48:112]], axis=1)
    zarr_url = tmp_path / "overlap.zarr"
    stage = {"y_micrometer_original": 2000.0}
    _write_image(
        zarr_url,
        stored,
        [
            ("FOV_1", 0, {**stage, "x_micrometer_original": 1000.0}),
            ("FOV_2", 64, {**stage, "x_micrometer_original": 1000.0 + 48 * _PS}),
        ],
    )
    viewer = make_napari_viewer()
    roi_loader = _init_roi_loader_image(viewer, zarr_url, qtbot)
    _select_roi_table(roi_loader, qtbot, "FOV_ROI_table", "FOV_2")

    fov_1 = _load_image_roi(viewer, roi_loader, qtbot, "FOV_1", original=True)
    fov_2 = _load_image_roi(viewer, roi_loader, qtbot, "FOV_2", original=True)
    overlap = _world_overlap(fov_1, fov_2)
    assert overlap is not None
    a_part, b_part = overlap
    assert a_part.shape == (64, 16)
    np.testing.assert_array_equal(a_part, b_part)

    # At the regular position the FOVs are only adjacent
    viewer.layers.clear()
    fov_1 = _load_image_roi(viewer, roi_loader, qtbot, "FOV_1")
    fov_2 = _load_image_roi(viewer, roi_loader, qtbot, "FOV_2")
    assert _world_overlap(fov_1, fov_2) is None


def test_registration_translation_aligns_acquisitions(
    make_napari_viewer, tmp_path, qtbot
):
    """The acquisition's content is shifted by (dy, dx) px. Fractal's
    registration (checked against fractal-tasks-core 2.1.0, both methods) stores
    translation = -(dy, dx) * pixel size, and adding it must re-align the layers."""
    dy, dx = 7, -11
    scene = np.random.default_rng(1).integers(1, 60000, (100, 120), dtype=np.uint16)
    ref = scene[20:84, 30:94]
    acq = scene[
        20 - dy : 84 - dy, 30 - dx : 94 - dx
    ]  # acq[y, x] == ref[y - dy, x - dx]
    shift = {"translation_y": -dy * _PS, "translation_x": -dx * _PS}
    _write_image(tmp_path / "ref.zarr", ref, [("FOV_1", 0, {})])
    _write_image(tmp_path / "acq.zarr", acq, [("FOV_1", 0, shift)])

    viewer = make_napari_viewer()
    ref_loader = _init_roi_loader_image(viewer, tmp_path / "ref.zarr", qtbot)
    acq_loader = _init_roi_loader_image(viewer, tmp_path / "acq.zarr", qtbot)
    assert acq_loader._add_translation_box.enabled
    assert not ref_loader._add_translation_box.enabled

    ref_layer = _load_image_roi(viewer, ref_loader, qtbot, "FOV_1")
    unshifted = _load_image_roi(viewer, acq_loader, qtbot, "FOV_1")
    overlap = _world_overlap(ref_layer, unshifted)
    assert overlap is not None
    assert not np.array_equal(*overlap)

    aligned = _load_image_roi(viewer, acq_loader, qtbot, "FOV_1", translation=True)
    overlap = _world_overlap(ref_layer, aligned)
    assert overlap is not None
    a_part, b_part = overlap
    assert a_part.shape == (64 - abs(dy), 64 - abs(dx))
    np.testing.assert_array_equal(a_part, b_part)


def test_additional_wells_use_each_wells_own_table(tmp_path):
    """'Load selected ROI for additional well(s)' takes the minimum original
    position from each well's own ROI table."""
    from types import SimpleNamespace

    from ngio import open_ome_zarr_container

    zarr_url = tmp_path / "well.zarr"
    image = np.ones((64, 128), dtype=np.uint16)
    _write_image(
        zarr_url,
        image,
        [
            (
                "FOV_1",
                0,
                {
                    "y_micrometer_original": -5.0,
                    "x_micrometer_original": 30.0,
                    "translation_x": 1.0,
                },
            ),
            (
                "FOV_2",
                64,
                {"y_micrometer_original": -9.0, "x_micrometer_original": 60.0},
            ),
        ],
    )
    container = open_ome_zarr_container(str(zarr_url))
    defaults = SimpleNamespace(
        default_roi_table="FOV_ROI_table",
        default_original_position=True,
        default_add_translation=True,
        default_masked=False,
    )
    kwargs = PlateBrowser._default_roi_kwargs(defaults, container)  # type: ignore[arg-type]
    assert kwargs == {
        "original_origin": {"y": -9.0, "x": 30.0},
        "add_translation": True,
    }

    defaults.default_original_position = False
    defaults.default_add_translation = False
    assert PlateBrowser._default_roi_kwargs(defaults, container) == {}  # type: ignore[arg-type]

    defaults.default_masked = True
    assert PlateBrowser._default_roi_kwargs(defaults, container) == {"masked": True}  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Masked loading from masking ROI tables
# ---------------------------------------------------------------------------

_MASKING_TABLE = "masking_roi_table_nuclei"


def test_masked_option_only_for_masking_tables(make_napari_viewer, zenodo_zarr, qtbot):
    zarr_url = Path(zenodo_zarr[1]) / "B" / "03" / "0"
    roi_loader = _init_roi_loader_image(make_napari_viewer(), zarr_url, qtbot)

    _select_roi_table(roi_loader, qtbot, "FOV_ROI_table", "FOV_2")
    assert not roi_loader._masked_box.enabled
    _select_roi_table(roi_loader, qtbot, _MASKING_TABLE, "6")
    assert roi_loader._masked_box.enabled


@pytest.mark.parametrize("image_mode", ["Fixed resolution", "Multi-resolution (lazy)"])
def test_masked_loading_keeps_only_the_object(
    make_napari_viewer, zenodo_zarr, qtbot, image_mode
):
    """Object 6's bounding box also contains objects 20 and 35; masking keeps
    only object 6 in images, labels and features, at the same position."""
    from napari_ome_zarr_navigator.roi_loading_utils import MASKED_METADATA_KEY

    zarr_url = Path(zenodo_zarr[1]) / "B" / "03" / "0"
    viewer = make_napari_viewer()
    roi_loader = _init_roi_loader_image(viewer, zarr_url, qtbot)
    _select_roi_table(roi_loader, qtbot, _MASKING_TABLE, "6")

    def load(masked):
        viewer.layers.clear()
        roi_loader._image_loading_mode.value = image_mode
        roi_loader._roi_picker.value = "6"
        roi_loader._channel_picker.value = ["DAPI"]
        roi_loader._label_picker.value = ["nuclei"]
        roi_loader._feature_picker.value = ["measurements"]
        roi_loader._masked_box.value = masked
        with qtbot.waitSignal(
            roi_loader.image_changed_event.load_finished, timeout=10000
        ):
            roi_loader.run()
        image = next(
            lyr for lyr in viewer.layers if isinstance(lyr, napari.layers.Image)
        )
        label = next(
            lyr for lyr in viewer.layers if isinstance(lyr, napari.layers.Labels)
        )
        image_data = np.asarray(image.data[0] if image.multiscale else image.data)
        return image, image_data, label

    plain_image, plain_pixels, plain_label = load(masked=False)
    assert set(np.unique(np.asarray(plain_label.data))) == {0, 6, 20, 35}
    assert MASKED_METADATA_KEY not in plain_image.metadata

    image, pixels, label = load(masked=True)
    assert set(np.unique(np.asarray(label.data))) == {0, 6}
    assert set(label.features["label"]) == {6}
    assert pixels.shape == plain_pixels.shape
    kept = pixels != 0
    assert 0 < kept.mean() < 1
    np.testing.assert_array_equal(pixels[kept], plain_pixels[kept])
    np.testing.assert_allclose(image.translate, plain_image.translate)
    np.testing.assert_allclose(label.translate, plain_label.translate)
    for layer in (image, label):
        assert layer.metadata[MASKED_METADATA_KEY] == {
            "roi_table": _MASKING_TABLE,
            "label": 6,
        }
