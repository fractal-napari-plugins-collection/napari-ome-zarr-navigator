# Utils for threaded loading of ROIs & adding them to the viewer.
import logging
import re
from typing import Optional

import napari
import napari.layers
import ngio
import ngio.tables
import numpy as np
from napari.qt.threading import thread_worker
from napari.utils.colormaps import Colormap
from qtpy.QtCore import QObject, Signal

from napari_ome_zarr_navigator.util import (
    LoaderState,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@thread_worker
def fetch_single_image(
    ome_zarr_container: ngio.OmeZarrContainer,
    roi_table: str,
    roi_name: str,
    layer_base_name: str,
    level: str,
    channel: str,
    translation: tuple[int, int],
    lazy: bool = False,
):
    """Load exactly one channel's ROI and return kwargs for viewer.add_image."""
    # TODO: Refactor ROI table loading to happen before the parallelized fetch
    ngio_table = ome_zarr_container.get_table(
        roi_table, check_type="generic_roi_table"
    )
    curr_roi = ngio_table.get(roi_name)
    roi_translation = (
        translation[0] + curr_roi.y,
        translation[1] + curr_roi.x,
    )

    idx = ome_zarr_container.image_meta.get_channel_idx(channel)
    ngio_img = ome_zarr_container.get_image(path=level)

    if lazy:
        arr = ngio_img.get_roi(roi=curr_roi, c=idx, mode="dask").squeeze()
    else:
        arr = np.squeeze(ngio_img.get_roi(roi=curr_roi, c=idx, mode="numpy"))

    if not np.any(arr):
        # nothing to display
        return None

    # compute scale
    if arr.ndim == 3:
        z, y, x = ngio_img.pixel_size.zyx
        scale = (z, y, x)
    elif arr.ndim == 2:
        y, x = ngio_img.pixel_size.yx
        scale = (y, x)
    else:
        raise NotImplementedError(
            "ROI loading has not been implemented for ROIs of shape "
            f"{len(arr.shape)} yet."
        )

    # build colormap + contrast_limits
    vis = ome_zarr_container.image_meta.channels[idx].channel_visualisation
    try:
        cmap = Colormap(["#000000", f"#{vis.color}"], name=vis.color)
    except AttributeError:
        cmap = None
    try:
        clims = (vis.start, vis.end)
    except AttributeError:
        clims = None

    return {
        "data": arr,
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
    lazy: bool = False,
    zarr_id: str = "",
):
    """
    Load all label masks + feature tables for one ROI.
    Returns a dict with keys "labels" (a list of args for add_labels)
    and "features" (a list of {layer_name, df} to attach).
    """
    ngio_table = ome_zarr_container.get_table(
        roi_table, check_type="generic_roi_table"
    )
    curr_roi = ngio_table.get(roi_name)
    roi_translation = (
        translation[0] + curr_roi.y,
        translation[1] + curr_roi.x,
    )

    result = {"labels": [], "features": []}

    # 1) load labels
    img_meta = ome_zarr_container.get_image(path=level).pixel_size
    for lbl in labels:
        ngio_lbl = ome_zarr_container.get_label(
            name=lbl, pixel_size=img_meta, strict=False
        )
        if lazy:
            arr = ngio_lbl.get_roi(roi=curr_roi, mode="dask").squeeze()
        else:
            arr = np.squeeze(ngio_lbl.get_roi(roi=curr_roi, mode="numpy"))

        if arr.ndim == 3:
            z, y, x = ngio_lbl.pixel_size.zyx
            scale = (z, y, x)
        elif arr.ndim == 2:
            y, x = ngio_lbl.pixel_size.yx
            scale = (y, x)
        else:
            raise NotImplementedError(
                "ROI loading has not been implemented for ROIs of shape "
                f"{len(arr.shape)} yet."
            )

        result["labels"].append(
            {
                "data": arr,
                "scale": scale,
                "name": f"{layer_base_name}{lbl}",
                "translate": roi_translation,
            }
        )

    # 2) load features
    for tbl in features:
        feat_tbl = ome_zarr_container.get_table(
            tbl, check_type="feature_table"
        )
        df = feat_tbl.dataframe.copy()
        df.index = df.index.astype(int)
        lbl_idx = find_matching_label_layer_index(
            feature_table=feat_tbl, labels=labels
        )
        labels_arr = result["labels"][lbl_idx]["data"]
        lbl_ids = np.unique(labels_arr)[1:]
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
    blending_int: Optional[str] = None,
    set_state_fn: Optional[callable] = None,
    lazy: bool = False,
    zarr_id: str = "",
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
        level: Resolution level to load
        labels: List of labels to load
        translation: Translation to apply to all loaded ROIs (typically the
            translation to shift it to the correct well in a plate setting)
        blending_int: Blending for the first intensity image to be used
        set_state_fn: is a callable that can be used to update button states.
            Needs to support set_state_fn(LoaderState.LOADING or READY) if
            provided.
        lazy: Whether to use dask for lazy loading. True will load the data
            lazily, False will load the data immediately.
        zarr_id: A unique identifier for the OME-Zarr image that features get
            loaded from. The zarr_id is used as a column for the features that
            allows a user to map back the features to a given OME-Zarr.

    """
    # 1) disable & show "Loading"
    if set_state_fn is not None:
        set_state_fn(LoaderState.LOADING)

    results = {"images": [], "lbl_feats": None}

    if roi_table == "well_ROI_table" or roi_table == "image_ROI_table":
        base_name = f"{layer_base_name}"
    else:
        base_name = f"{layer_base_name}{roi_name}_"

    def try_finalize():
        # wait until images for all channels + lbl_feats present
        if (
            len(results["images"]) != len(channels)
            or results["lbl_feats"] is None
        ):
            return

        # add intensity images in user‐selected order
        for i, ch in enumerate(channels):
            blending = blending_int if i == 0 else "additive"
            name = f"{base_name}{ch}"
            kw = next(
                item
                for item in results["images"]
                if item and item["name"] == name
            )
            viewer.add_image(**kw, blending=blending)

        # add labels
        for kw in results["lbl_feats"]["labels"]:
            viewer.add_labels(**kw)

        # attach feature tables
        for feat in results["lbl_feats"]["features"]:
            viewer.layers[feat["layer_name"]].features = feat["df"]

        # back to READY
        if set_state_fn is not None:
            set_state_fn(LoaderState.READY)

    # 2) launch one worker per channel
    for ch in channels:
        w = fetch_single_image(
            ome_zarr_container,
            roi_table,
            roi_name,
            base_name,
            level,
            ch,
            translation,
            lazy=lazy,
        )
        w.returned.connect(
            lambda img_kw: (results["images"].append(img_kw), try_finalize())
        )
        w.start()

    # 3) launch labels+features worker
    w2 = fetch_labels_and_features(
        ome_zarr_container,
        roi_table,
        roi_name,
        base_name,
        level,
        labels,
        features,
        translation,
        lazy=lazy,
        zarr_id=zarr_id,
    )
    w2.returned.connect(
        lambda lf: (results.__setitem__("lbl_feats", lf), try_finalize())
    )
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
        # FIXME: Generalize well name catching
        if type(layer) == napari.layers.Labels and re.match(
            r"[A-Z][a-z]*\d+_*", layer.name
        ):
            viewer.layers.remove(layer)
