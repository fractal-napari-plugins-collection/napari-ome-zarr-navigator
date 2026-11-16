# Utils for threaded loading of ROIs & adding them to the viewer.
import logging
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import napari
import napari.layers
import ngio
import numpy as np
from napari.layers._source import Source
from napari.qt.threading import thread_worker
from napari.utils.colormaps import Colormap
from qtpy.QtCore import QObject, Signal  # type: ignore[attr-defined]

from napari_ome_zarr_navigator.util import (
    LoaderState,
)
from napari_ome_zarr_navigator.well_utils import WELL_LAYER_PATTERN

logger = logging.getLogger(__name__)


def _compute_scale(ngio_img, ndim: int) -> tuple:
    """Return a scale tuple from ngio pixel_size matching the array dimensionality."""
    # TODO: cleaner handling of tyx edge-case
    if ndim == 3:
        z, y, x = ngio_img.pixel_size.zyx
        return (z, y, x)
    elif ndim == 2:
        y, x = ngio_img.pixel_size.yx
        return (y, x)
    elif ndim == 4:
        return ngio_img.pixel_size.tzyx
    else:
        raise NotImplementedError(
            f"ROI loading has not been implemented for ROIs of shape {ndim}D yet."
        )


# ---------------------------------------------------------------------------
# ROI positioning: original position and registration translation
# ---------------------------------------------------------------------------

# layer.metadata key marking layers that were not loaded at their regular
# position. The Label Saver and ROI Annotator refuse such layers, because they
# derive the on-disk position from the layer's translate.
POSITION_METADATA_KEY = "navigator_position"


@dataclass(frozen=True)
class RoiPositionInfo:
    """Optional position columns a ROI table provides (computed once per table).

    Attributes:
        original_origin: Per-axis minimum of the ``*_micrometer_original``
            columns over the table (keys "y", "x" and, if present, "z"), or
            None when the table has no original positions.
        has_translation: Whether any ROI has ``translation_x/y/z`` columns.
    """

    original_origin: dict[str, float] | None = None
    has_translation: bool = False


def is_shifted_layer(layer) -> bool:
    """Whether a layer was loaded at its original position or with translation."""
    return POSITION_METADATA_KEY in (getattr(layer, "metadata", None) or {})


SHIFTED_LAYER_MESSAGE = (
    "Label layer '%s' was loaded at a shifted position (original position or "
    "registration translation), so it doesn't line up with the OME-Zarr image. "
    "Reload the ROI without these ROI Loader options to %s."
)


def _finite_extra(roi, key: str) -> float | None:
    """Return a finite float from a ROI's extra columns, or None (missing/NaN)."""
    raw = (roi.model_extra or {}).get(key)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _original_coords(roi) -> dict[str, float] | None:
    """The ROI's original position in µm, or None without finite x and y."""
    y = _finite_extra(roi, "y_micrometer_original")
    x = _finite_extra(roi, "x_micrometer_original")
    if y is None or x is None:
        return None
    coords = {"y": y, "x": x}
    z = _finite_extra(roi, "z_micrometer_original")
    if z is not None:
        coords["z"] = z
    return coords


def _translation_coords(roi) -> dict[str, float] | None:
    """The ROI's registration translation in µm (missing axes count as 0).

    Returns None when the ROI has none of the translation columns, e.g. in the
    reference acquisition of a Fractal multiplexing registration.
    """
    values = {ax: _finite_extra(roi, f"translation_{ax}") for ax in ("z", "y", "x")}
    if all(v is None for v in values.values()):
        return None
    return {ax: v if v is not None else 0.0 for ax, v in values.items()}


def roi_position_info(rois: Iterable) -> RoiPositionInfo:
    """Summarise which optional position columns a ROI table's ROIs carry."""
    rois = list(rois)
    originals = [c for roi in rois if (c := _original_coords(roi)) is not None]
    origin = None
    if originals:
        origin = {ax: min(c[ax] for c in originals) for ax in ("y", "x")}
        z_values = [c["z"] for c in originals if "z" in c]
        if z_values:
            origin["z"] = min(z_values)
    has_translation = any(_translation_coords(roi) is not None for roi in rois)
    return RoiPositionInfo(original_origin=origin, has_translation=has_translation)


def _roi_translate(
    roi,
    translation: tuple[float, float],
    *,
    origin: dict[str, float] | None = None,
    add_translation: bool = False,
    ndim: int = 2,
    has_z_axis: bool = False,
) -> tuple[tuple[float, ...], dict | None]:
    """Compute the napari ``translate`` for a loaded ROI.

    Position = ``translation`` (well offset in plate mode) + base + optional
    registration translation, where base is the ROI start, or, when ``origin``
    is given, the ROI's original position relative to the table minimum. A ROI
    without original coordinates keeps its regular base.

    Returns:
        The translate tuple (y, x), or with a leading z term (padded to
        ``ndim``) when the data has a z axis and the z offset is non-zero, and a
        tag dict ``{"original": ..., "translation": ...}`` with the values used,
        or None when the ROI is at its regular position.
    """
    tag: dict[str, dict[str, float]] = {}
    original = _original_coords(roi) if origin is not None else None
    if origin is not None and original is not None:
        y = original["y"] - origin["y"]
        x = original["x"] - origin["x"]
        z = original["z"] - origin["z"] if "z" in original and "z" in origin else 0.0
        tag["original"] = original
    else:
        y, x, z = roi["y"].start, roi["x"].start, 0.0
    if add_translation:
        shift = _translation_coords(roi)
        if shift is not None:
            y, x, z = y + shift["y"], x + shift["x"], z + shift["z"]
            tag["translation"] = shift
    yx = (translation[0] + y, translation[1] + x)
    if has_z_axis and z != 0.0:
        return (0.0,) * (ndim - 3) + (z,) + yx, (tag or None)
    return yx, (tag or None)


def _layer_translate(
    ome_zarr_container: ngio.OmeZarrContainer,
    roi,
    translation: tuple[float, float],
    ndim: int,
    original_origin: dict[str, float] | None,
    add_translation: bool,
) -> tuple[tuple[float, ...], dict | None]:
    """``_roi_translate`` for loaded data of dimensionality ``ndim``.

    ``roi`` is None for whole-image loading, which ignores the position options.
    """
    if roi is None:
        return tuple(translation), None
    # A z axis survives .squeeze() for zyx (3D) and tzyx (4D) data.
    has_z_axis = ome_zarr_container.is_3d and (
        ndim == 4 or (ndim == 3 and not ome_zarr_container.is_time_series)
    )
    return _roi_translate(
        roi,
        translation,
        origin=original_origin,
        add_translation=add_translation,
        ndim=ndim,
        has_z_axis=has_z_axis,
    )


@thread_worker
def fetch_single_image(
    ome_zarr_container: ngio.OmeZarrContainer,
    roi_table: str,
    roi_name: str,
    layer_base_name: str,
    level: str,
    channel: str,
    translation: tuple[int, int],
    multiscale: bool = False,
    whole_image: bool = False,
    original_origin: dict[str, float] | None = None,
    add_translation: bool = False,
):
    """Load exactly one channel and return kwargs for viewer.add_image.

    When whole_image=True, loads the full image array (no ROI cropping).
    When multiscale=True, returns a list of dask arrays (finest→coarsest) for
    napari's multiscale rendering. When False, returns a numpy array at level.
    ``original_origin`` / ``add_translation`` select the ROI position, see
    ``_roi_translate``.
    """
    curr_roi = None
    if not whole_image:
        # TODO: Refactor ROI table loading to happen before the parallelized fetch
        ngio_table = ome_zarr_container.get_generic_roi_table(roi_table)
        curr_roi = ngio_table.get(roi_name)

    if multiscale:
        level_paths = ome_zarr_container.level_paths  # ngio order: finest→coarsest
        data = []
        for lv in level_paths:
            ngio_img = ome_zarr_container.get_image(path=lv)
            if curr_roi is None:
                arr = ngio_img.get_as_dask(channel_selection=channel).squeeze()  # type: ignore[attr-defined]
            else:
                arr = ngio_img.get_roi_as_dask(
                    roi=curr_roi, channel_selection=channel
                ).squeeze()  # type: ignore[attr-defined]
            data.append(arr)
        ngio_img0 = ome_zarr_container.get_image(path=level_paths[0])
        scale = _compute_scale(ngio_img0, data[0].ndim)
        if not whole_image and not np.any(data[0]):
            return None
    else:
        ngio_img = ome_zarr_container.get_image(path=level)
        if curr_roi is None:
            arr = ngio_img.get_as_numpy(channel_selection=channel).squeeze()  # type: ignore[attr-defined]
        else:
            arr = ngio_img.get_roi_as_numpy(
                roi=curr_roi, channel_selection=channel
            ).squeeze()  # type: ignore[attr-defined]
            if not np.any(arr):
                return None
        data = arr
        scale = _compute_scale(ngio_img, arr.ndim)

    roi_translation, position_tag = _layer_translate(
        ome_zarr_container,
        curr_roi,
        translation,
        ndim=len(scale),
        original_origin=original_origin,
        add_translation=add_translation,
    )

    # build colormap + contrast_limits
    idx = ome_zarr_container.get_channel_idx(channel_label=channel)
    vis = ome_zarr_container.meta.channels_meta.channels[  # type: ignore[attr-defined]
        idx
    ].channel_visualisation
    try:
        cmap = Colormap(["#000000", f"#{vis.color}"], name=vis.color)
    except AttributeError:
        cmap = None
    try:
        clims = (vis.start, vis.end)
    except AttributeError:
        clims = None

    layer_kwargs = {
        "data": data,
        "scale": scale,
        "contrast_limits": clims,
        "colormap": cmap,
        "name": f"{layer_base_name}{channel}",
        "translate": roi_translation,
    }
    if position_tag is not None:
        layer_kwargs["metadata"] = {POSITION_METADATA_KEY: position_tag}
    return layer_kwargs


@thread_worker
def fetch_labels_and_features(
    ome_zarr_container: ngio.OmeZarrContainer,
    roi_table: str,
    roi_name: str,
    layer_base_name: str,
    level: str,
    labels: list[str],
    features: list[str],
    translation: tuple[int, int],
    zarr_id: str = "",
    multiscale_labels: bool = False,
    label_level: str = "0",
    whole_image: bool = False,
    original_origin: dict[str, float] | None = None,
    add_translation: bool = False,
):
    """
    Load all label masks + feature tables for one ROI (or the whole image).
    Returns a dict with keys "labels" (a list of args for add_labels)
    and "features" (a list of {layer_name, df} to attach).

    When whole_image=True, loads the full label array without ROI cropping.
    When multiscale_labels=True, label data is a list of dask arrays
    (finest→coarsest). When False, loads a numpy array at label_level.
    ``original_origin`` / ``add_translation`` select the ROI position, see
    ``_roi_translate``.
    """
    curr_roi = None
    if not whole_image:
        ngio_table = ome_zarr_container.get_generic_roi_table(roi_table)
        curr_roi = ngio_table.get(roi_name)

    result = {"labels": [], "features": []}

    # 1) load labels
    # TODO: edge-case tyx, see above
    for lbl in labels:
        if multiscale_labels:
            level_paths = ome_zarr_container.level_paths
            data = []
            for lv in level_paths:
                ngio_lbl = ome_zarr_container.get_label(name=lbl, path=lv)
                if curr_roi is None:
                    arr = ngio_lbl.get_as_dask().squeeze()  # type: ignore[attr-defined]
                else:
                    arr = ngio_lbl.get_roi_as_dask(roi=curr_roi).squeeze()  # type: ignore[attr-defined]
                data.append(arr)
            ngio_lbl0 = ome_zarr_container.get_label(name=lbl, path=level_paths[0])
            scale = _compute_scale(ngio_lbl0, data[0].ndim)
        else:
            ngio_lbl = ome_zarr_container.get_label(name=lbl, path=label_level)
            if curr_roi is None:
                arr = ngio_lbl.get_as_numpy().squeeze()  # type: ignore[attr-defined]
            else:
                arr = ngio_lbl.get_roi_as_numpy(roi=curr_roi).squeeze()  # type: ignore[attr-defined]
            data = arr
            scale = _compute_scale(ngio_lbl, arr.ndim)

        roi_translation, position_tag = _layer_translate(
            ome_zarr_container,
            curr_roi,
            translation,
            ndim=len(scale),
            original_origin=original_origin,
            add_translation=add_translation,
        )
        label_kwargs = {
            "data": data,
            "scale": scale,
            "name": f"{layer_base_name}{lbl}",
            "translate": roi_translation,
        }
        if position_tag is not None:
            label_kwargs["metadata"] = {POSITION_METADATA_KEY: position_tag}
        result["labels"].append(label_kwargs)

    # 2) load features
    for tbl in features:
        feat_tbl = ome_zarr_container.get_feature_table(tbl)
        df = feat_tbl.dataframe.copy()
        df.index = df.index.astype(int)
        lbl_idx = find_matching_label_layer_index(feature_table=feat_tbl, labels=labels)
        labels_arr = result["labels"][lbl_idx]["data"]
        # For multiscale labels, use the finest level; for dask arrays, compute
        # to resolve chunk sizes before calling np.unique.
        arr_for_ids = labels_arr[0] if isinstance(labels_arr, list) else labels_arr
        if hasattr(arr_for_ids, "compute"):
            arr_for_ids = arr_for_ids.compute()
        lbl_ids = np.unique(arr_for_ids)[1:]
        df = df.loc[df.index.isin(lbl_ids)].reset_index()
        df = df.loc[:, ~df.columns.duplicated()].copy()
        df["index"] = df["label"]
        df["roi_id"] = f"{zarr_id}:{roi_name}"

        result["features"].append(
            {"layer_name": result["labels"][lbl_idx]["name"], "df": df}
        )

    return result


def find_matching_label_layer_index(
    feature_table: ngio.FeatureTable,
    labels: list[str],
):
    """
    Finds the matching label layer for a feature table
    """
    reference_label = feature_table.reference_label
    if reference_label not in labels:
        logger.info(
            f"The label {reference_label} that was referenced"
            "in a feature table to be loaded could not be found."
            "Attaching the features to the first selected label layer "
            f"{labels[0]} instead."
        )
        return 0

    return labels.index(reference_label)


def apply_roi_data_to_viewer(
    viewer: napari.Viewer,
    results: dict,
    channels: list[str],
    base_name: str,
    blending_int: str | None = None,
    set_state_fn: Callable | None = None,
    zarr_id: str = "",
) -> None:
    """Add pre-fetched ROI data to the napari viewer.

    Args:
        viewer: napari viewer to add layers to
        results: dict with keys "images" (list of add_image kwargs) and
            "lbl_feats" (dict with "labels" and "features" lists)
        channels: ordered list of channel names — determines layer add order
        base_name: layer name prefix used to match images by name
        blending_int: blending mode for the first intensity channel
        set_state_fn: optional callable to restore button state to READY
        zarr_id: zarr URL set as layer.source.path for traceability
    """
    # add intensity images in user-selected order
    for i, ch in enumerate(channels):
        blending = blending_int if i == 0 else "additive"
        name = f"{base_name}{ch}"
        kw = next(
            (item for item in results["images"] if item and item["name"] == name),
            None,
        )
        if kw is None:
            # worker returned None (empty/all-zero ROI) — skip this channel
            continue
        layer = viewer.add_image(**kw, blending=blending)
        if zarr_id and isinstance(layer, napari.layers.Image):
            layer._set_source(
                Source(path=zarr_id, reader_plugin="napari-ome-zarr-navigator")
            )

    # add labels
    for kw in results["lbl_feats"]["labels"]:
        layer = viewer.add_layer(napari.layers.Labels(**kw))
        if zarr_id:
            layer._set_source(
                Source(path=zarr_id, reader_plugin="napari-ome-zarr-navigator")
            )

    # attach feature tables
    for feat in results["lbl_feats"]["features"]:
        viewer.layers[feat["layer_name"]].features = feat["df"]

    if set_state_fn is not None:
        set_state_fn(LoaderState.READY)


def orchestrate_load_roi(
    ome_zarr_container: ngio.OmeZarrContainer,
    viewer: napari.Viewer,
    roi_table: str,
    roi_name: str,
    layer_base_name: str,
    level: str,
    channels: list[str],
    labels: list[str],
    features: list[str],
    translation: tuple[int, int],
    blending_int: str | None = None,
    set_state_fn: Callable | None = None,
    zarr_id: str = "",
    multiscale_image: bool = False,
    multiscale_labels: bool = False,
    label_level: str = "0",
    whole_image: bool = False,
    original_origin: dict[str, float] | None = None,
    add_translation: bool = False,
):
    """
    Load images, labels & tables of a given ROI & add to viewer

    Orchestrate parallel fetch of single-channel images and labels+features,
    then add them in order and restore button state.
    `set_state_fn` is a callable: set_state_fn(LoaderState.LOADING or READY).

    Args:
        ome_zarr_container: OME-Zarr object to be loaded
        viewer: napari viewer object
        roi_table: Name of the ROI table to load a ROI from
            (e.g. "well_ROI_table")
        roi_name: Name of the ROI within the roi_table to load
        layer_base_name: Base name for the layers to be added
        channels: List of intensity channels to load
        level: Resolution level to load (used when multiscale_image=False)
        labels: List of labels to load
        translation: Translation to apply to all loaded ROIs (typically the
            translation to shift it to the correct well in a plate setting)
        blending_int: Blending for the first intensity image to be used
        set_state_fn: is a callable that can be used to update button states.
            Needs to support set_state_fn(LoaderState.LOADING or READY) if
            provided.
        zarr_id: A unique identifier for the OME-Zarr image that features get
            loaded from. The zarr_id is used as a column for the features that
            allows a user to map back the features to a given OME-Zarr. Also
            set as layer.source.path for traceability.
        multiscale_image: When True, load all pyramid levels as a dask pyramid
            (list of dask arrays). When False, load a single array at `level`.
        multiscale_labels: When True, load labels as a dask pyramid. When
            False, load labels at `label_level` as a numpy array.
        label_level: Level path for fixed-resolution label loading (e.g. "0").
        whole_image: When True, skip ROI cropping and load the full image/labels.
        original_origin: When given, place the ROI at its original position
            (``*_micrometer_original`` columns) relative to this per-axis table
            minimum, instead of its regular position. See ``roi_position_info``.
        add_translation: When True, shift the ROI by its registration
            translation (``translation_x/y/z`` columns).

    """
    # 1) disable & show "Loading"
    if set_state_fn is not None:
        set_state_fn(LoaderState.LOADING)

    results = {"images": [], "lbl_feats": None}
    total_workers = len(channels) + 1  # one per channel + one for labels/features
    finished_count = {"value": 0}

    if whole_image or roi_table == "well_ROI_table" or roi_table == "image_ROI_table":
        base_name = layer_base_name
    else:
        base_name = f"{layer_base_name}{roi_name}_"

    def worker_done() -> None:
        finished_count["value"] += 1
        if finished_count["value"] < total_workers:
            return
        # All workers finished (some may have errored) — apply whatever loaded.
        if results["lbl_feats"] is None:
            results["lbl_feats"] = {"labels": [], "features": []}
        if original_origin is not None and not whole_image:
            layer_kwargs = [kw for kw in results["images"] if kw]
            layer_kwargs += results["lbl_feats"]["labels"]
            if any(
                "original"
                not in ((kw.get("metadata") or {}).get(POSITION_METADATA_KEY) or {})
                for kw in layer_kwargs
            ):
                logger.warning(
                    "ROI '%s' in '%s' has no x/y_micrometer_original; "
                    "loaded at its regular position.",
                    roi_name,
                    roi_table,
                )
        apply_roi_data_to_viewer(
            viewer, results, channels, base_name, blending_int, set_state_fn, zarr_id
        )

    def make_channel_error_handler(ch_name: str) -> Callable[[BaseException], None]:
        def on_error(exc: BaseException) -> None:
            logger.error("Channel '%s' load failed: %s", ch_name, exc, exc_info=exc)
            worker_done()

        return on_error

    def on_labels_error(exc: BaseException) -> None:
        logger.error("Labels/features load failed: %s", exc, exc_info=exc)
        worker_done()

    # 2) launch one worker per channel
    for ch in channels:
        w = fetch_single_image(  # type: ignore[call-arg]
            ome_zarr_container,
            roi_table,  # type: ignore
            roi_name,
            base_name,
            level,
            ch,
            translation,
            multiscale=multiscale_image,
            whole_image=whole_image,
            original_origin=original_origin,
            add_translation=add_translation,
        )
        w.returned.connect(
            lambda img_kw: (results["images"].append(img_kw), worker_done())
        )
        w.errored.connect(make_channel_error_handler(ch))
        w.start()

    # 3) launch labels+features worker
    w2 = fetch_labels_and_features(  # type: ignore[call-arg]
        ome_zarr_container,
        roi_table,  # type: ignore
        roi_name,
        base_name,
        level,
        labels,
        features,
        translation,
        zarr_id=zarr_id,
        multiscale_labels=multiscale_labels,
        label_level=label_level,
        whole_image=whole_image,
        original_origin=original_origin,
        add_translation=add_translation,
    )
    w2.returned.connect(
        lambda lf: (results.__setitem__("lbl_feats", lf), worker_done())
    )
    w2.errored.connect(on_labels_error)
    w2.start()


class ROILoaderSignals(QObject):
    image_changed = Signal(object)
    roi_choices_updated = Signal(list)
    roi_tables_updated = Signal(list)
    load_finished = Signal()

    def __init__(self):
        super().__init__()


def remove_existing_label_layers(viewer):
    for layer in viewer.layers:
        if isinstance(layer, napari.layers.Labels) and WELL_LAYER_PATTERN.match(
            layer.name
        ):
            viewer.layers.remove(layer)
