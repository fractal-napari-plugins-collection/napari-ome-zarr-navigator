# Utils for threaded loading of ROIs & adding them to the viewer.
import logging
from collections.abc import Callable

import napari
import napari.layers
import ngio
import ngio.tables
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
):
    """Load exactly one channel and return kwargs for viewer.add_image.

    When whole_image=True, loads the full image array (no ROI cropping).
    When multiscale=True, returns a list of dask arrays (finest→coarsest) for
    napari's multiscale rendering. When False, returns a numpy array at level.
    """
    if whole_image:
        roi_translation = translation
    else:
        # TODO: Refactor ROI table loading to happen before the parallelized fetch
        ngio_table = ome_zarr_container.get_generic_roi_table(roi_table)
        curr_roi = ngio_table.get(roi_name)
        roi_translation = (
            translation[0] + curr_roi["y"].start,  # type: ignore[index]
            translation[1] + curr_roi["x"].start,  # type: ignore[index]
        )

    if multiscale:
        level_paths = ome_zarr_container.level_paths  # ngio order: finest→coarsest
        data = []
        for lv in level_paths:
            ngio_img = ome_zarr_container.get_image(path=lv)
            if whole_image:
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
        if whole_image:
            arr = ngio_img.get_as_numpy(channel_selection=channel).squeeze()  # type: ignore[attr-defined]
        else:
            arr = ngio_img.get_roi_as_numpy(
                roi=curr_roi, channel_selection=channel
            ).squeeze()  # type: ignore[attr-defined]
            if not np.any(arr):
                return None
        data = arr
        scale = _compute_scale(ngio_img, arr.ndim)

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

    return {
        "data": data,
        "scale": scale,
        "contrast_limits": clims,
        "colormap": cmap,
        "name": f"{layer_base_name}{channel}",
        "translate": roi_translation,
    }


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
):
    """
    Load all label masks + feature tables for one ROI (or the whole image).
    Returns a dict with keys "labels" (a list of args for add_labels)
    and "features" (a list of {layer_name, df} to attach).

    When whole_image=True, loads the full label array without ROI cropping.
    When multiscale_labels=True, label data is a list of dask arrays
    (finest→coarsest). When False, loads a numpy array at label_level.
    """
    if whole_image:
        roi_translation = translation
    else:
        ngio_table = ome_zarr_container.get_generic_roi_table(roi_table)
        curr_roi = ngio_table.get(roi_name)
        roi_translation = (
            translation[0] + curr_roi["y"].start,  # type: ignore[index]
            translation[1] + curr_roi["x"].start,  # type: ignore[index]
        )

    result = {"labels": [], "features": []}

    # 1) load labels
    # TODO: edge-case tyx, see above
    for lbl in labels:
        if multiscale_labels:
            level_paths = ome_zarr_container.level_paths
            data = []
            for lv in level_paths:
                ngio_lbl = ome_zarr_container.get_label(name=lbl, path=lv)
                if whole_image:
                    arr = ngio_lbl.get_as_dask().squeeze()  # type: ignore[attr-defined]
                else:
                    arr = ngio_lbl.get_roi_as_dask(roi=curr_roi).squeeze()  # type: ignore[attr-defined]
                data.append(arr)
            ngio_lbl0 = ome_zarr_container.get_label(name=lbl, path=level_paths[0])
            scale = _compute_scale(ngio_lbl0, data[0].ndim)
        else:
            ngio_lbl = ome_zarr_container.get_label(name=lbl, path=label_level)
            if whole_image:
                arr = ngio_lbl.get_as_numpy().squeeze()  # type: ignore[attr-defined]
            else:
                arr = ngio_lbl.get_roi_as_numpy(roi=curr_roi).squeeze()  # type: ignore[attr-defined]
            data = arr
            scale = _compute_scale(ngio_lbl, arr.ndim)

        result["labels"].append(
            {
                "data": data,
                "scale": scale,
                "name": f"{layer_base_name}{lbl}",
                "translate": roi_translation,
            }
        )

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
    feature_table: ngio.tables.FeatureTable,
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
        layer = viewer.add_labels(**kw)
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
