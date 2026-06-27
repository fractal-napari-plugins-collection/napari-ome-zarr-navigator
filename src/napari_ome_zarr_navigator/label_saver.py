import logging

import napari
import napari.layers
import numpy as np
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    LineEdit,
    PushButton,
)
from napari.qt.threading import thread_worker
from ngio import open_ome_zarr_container
from qtpy.QtCore import QTimer

from napari_ome_zarr_navigator.util import ZarrSelector

logger = logging.getLogger(__name__)

_BACKEND_CHOICES = ["csv", "anndata", "json", "parquet"]

# Write-mode constants (used as ComboBox values and in _do_save_impl branching)
_WM_NEW = "New label (initialise)"
_WM_OVERWRITE = "Overwrite full label"
_WM_RESET = "Reset existing label"
_WM_APPEND = "Append to existing label"
_WRITE_MODES = [_WM_NEW, _WM_OVERWRITE, _WM_RESET, _WM_APPEND]


class LabelSaverImage(Container):
    """Widget for saving a napari Labels layer to an OME-Zarr container.

    Can be used standalone (shows ZarrSelector) or launched from an ROI Loader
    with a pre-populated zarr URL (selector hidden).  After saving, the optional
    ``roi_loader`` reference is called to refresh its label picker.

    Four write modes (auto-selected based on label existence / shape):
    - New label (initialise): create full-size label, fill sub-region if partial
    - Overwrite full label: full-extent validation; replace existing label
    - Reset existing label: destroy & recreate at full size; fill sub-region
    - Append to existing label: patch-write only the loaded sub-region

    Writing to HTTP/remote stores is not supported.
    """

    def __init__(
        self,
        viewer: napari.Viewer,
        zarr_url: str | None = None,
        token: str | None = None,
        source: str = "File",
        roi_loader=None,
    ):
        self._viewer = viewer
        self._roi_loader = roi_loader
        self._ome_zarr_container = None
        self._zarr_url: str | None = zarr_url
        self._source: str = source

        self._zarr_selector = ZarrSelector()

        self._layer_picker = ComboBox(
            label="Label layer",
            choices=self._get_label_layers(),
        )

        self._label_name = LineEdit(label="Label name", value="")
        self._write_mode = ComboBox(
            label="Write mode",
            choices=_WRITE_MODES,
            value=_WM_NEW,
        )
        self._write_mode.tooltip = (
            "New label (initialise): create a new full-size label; fill sub-region if loaded from ROI.\n"
            "Overwrite full label: replace existing label; requires matching spatial extent.\n"
            "Reset existing label: destroy and recreate the label at full size; fill sub-region.\n"
            "Append to existing label: patch-write only the loaded sub-region into the existing label."
        )

        self._save_masking_roi = CheckBox(value=False, text="Save masking ROI table")
        self._masking_roi_table_name = LineEdit(label="ROI table name", value="")
        self._masking_roi_table_name.visible = False
        self._masking_roi_backend = ComboBox(
            label="Backend",
            choices=_BACKEND_CHOICES,
            value="csv",
        )
        self._masking_roi_backend.visible = False

        self._axes_names = LineEdit(label="Axes names", value="")
        self._axes_names.tooltip = (
            "Auto-inferred from the OME-Zarr image axes. "
            "Edit to override (e.g. 'zyx', 'tzyx')."
        )
        self._advanced_toggle = PushButton(text="▶ Advanced settings")
        self._advanced_container = Container(widgets=[self._axes_names])
        self._advanced_container.visible = False
        self._advanced_visible = False

        self._btn_save = PushButton(text="Save label to OME-Zarr", enabled=False)

        super().__init__(
            widgets=[
                self._zarr_selector,
                self._layer_picker,
                self._label_name,
                self._write_mode,
                self._save_masking_roi,
                self._masking_roi_table_name,
                self._masking_roi_backend,
                self._advanced_toggle,
                self._advanced_container,
                self._btn_save,
            ]
        )

        # Debounce timer: fires _update_write_mode_default 400 ms after last keystroke
        self._label_name_timer = QTimer()
        self._label_name_timer.setSingleShot(True)
        self._label_name_timer.timeout.connect(self._update_write_mode_default)

        self._zarr_selector.on_change(self._on_url_changed)
        self._layer_picker.changed.connect(self._update_axes_inference)
        self._layer_picker.changed.connect(self._update_write_mode_default)
        self._label_name.changed.connect(self._on_label_name_changed)
        self._label_name.changed.connect(lambda _: self._label_name_timer.start(400))
        self._save_masking_roi.changed.connect(self._on_save_masking_roi_changed)
        self._advanced_toggle.clicked.connect(self._toggle_advanced)
        self._btn_save.clicked.connect(self._save)

        self._viewer.layers.events.inserted.connect(self._refresh_layer_choices)
        self._viewer.layers.events.removed.connect(self._refresh_layer_choices)

        if zarr_url:
            self._zarr_selector.configure(
                source=source, url=zarr_url, token=token or ""
            )
            self._zarr_selector.hide()
            self._on_url_changed()

    # ------------------------------------------------------------------
    # Layer picker helpers
    # ------------------------------------------------------------------

    def _get_label_layers(self) -> list[str]:
        return [
            layer.name
            for layer in self._viewer.layers
            if isinstance(layer, napari.layers.Labels)
        ]

    def _refresh_layer_choices(self, *_) -> None:
        choices = self._get_label_layers()
        old_value = self._layer_picker.value
        self._layer_picker.choices = choices
        self._layer_picker._default_choices = choices
        if old_value in choices:
            self._layer_picker.value = old_value

    # ------------------------------------------------------------------
    # Container loading and axes inference
    # ------------------------------------------------------------------

    def _on_url_changed(self) -> None:
        url = self._zarr_selector.url
        self._source = self._zarr_selector.source
        self._zarr_url = url if (url and url not in ("", ".")) else None
        if self._zarr_url and self._source == "File":
            self._load_container_async(self._zarr_url)

    def _load_container_async(self, url: str) -> None:
        @thread_worker
        def _load():
            try:
                return open_ome_zarr_container(url, mode="r", cache=True)
            except Exception:  # noqa: BLE001
                return None

        w = _load()  # type: ignore[call-arg]
        w.returned.connect(self._on_container_ready)
        w.start()

    def _on_container_ready(self, container) -> None:
        self._ome_zarr_container = container
        self._update_axes_inference()
        self._update_write_mode_default()

    def _update_axes_inference(self, *_) -> None:
        """Infer axes from the container + selected layer ndim; populate _axes_names."""
        if self._ome_zarr_container is None:
            return
        layer = self._get_selected_layer()
        if layer is None:
            return
        ndim = len(layer.data.shape)
        axes = self._infer_axes(ndim, self._ome_zarr_container)
        self._axes_names.value = axes

    def _infer_axes(self, label_ndim: int, container) -> str:
        """Return axis string (e.g. 'zyx') inferred from the container image axes."""
        try:
            img = container.get_image(path=container.level_paths[0])
            non_channel = [a for a in img.axes if a != "c"]
            if label_ndim <= len(non_channel):
                return "".join(non_channel[-label_ndim:])
        except Exception:  # noqa: BLE001
            pass
        return {2: "yx", 3: "zyx", 4: "tzyx"}.get(label_ndim, "yx")

    # ------------------------------------------------------------------
    # Write mode auto-selection
    # ------------------------------------------------------------------

    def _update_write_mode_default(self, *_) -> None:
        """Auto-select write mode based on label existence and spatial extent match."""
        if self._ome_zarr_container is None:
            return
        label_name = self._label_name.value.strip()
        if not label_name:
            return
        layer = self._get_selected_layer()
        if layer is None:
            return

        container = self._ome_zarr_container
        label_shape = layer.data.shape
        layer_scale = tuple(layer.scale)
        axes_str = self._axes_names.value.strip() or self._infer_axes(
            len(label_shape), container
        )

        @thread_worker
        def _check():
            try:
                existing = container.list_labels()
                if label_name not in existing:
                    return _WM_NEW
                img = container.get_image(path=container.level_paths[0])
                axes_list = list(axes_str)
                img_axes = list(img.axes)
                for a in ("y", "x"):
                    if a not in img_axes or a not in axes_list:
                        continue
                    img_extent = img.shape[img_axes.index(a)] * getattr(
                        img.pixel_size, a
                    )
                    lbl_extent = label_shape[axes_list.index(a)] * abs(
                        float(layer_scale[axes_list.index(a)])
                    )
                    if (
                        img_extent > 0
                        and abs(img_extent - lbl_extent) > 0.01 * img_extent
                    ):
                        return _WM_APPEND
                return _WM_OVERWRITE
            except Exception:  # noqa: BLE001
                return None

        worker = _check()  # type: ignore[call-arg]
        worker.returned.connect(self._on_write_mode_default_ready)
        worker.start()

    def _on_write_mode_default_ready(self, mode: str | None) -> None:
        if mode is not None:
            self._write_mode.value = mode

    # ------------------------------------------------------------------
    # UI event handlers
    # ------------------------------------------------------------------

    def _on_label_name_changed(self, value: str) -> None:
        stripped = value.strip()
        self._btn_save.enabled = bool(stripped)
        if stripped:
            self._masking_roi_table_name.value = f"{stripped}_masking_ROI_table"

    def _on_save_masking_roi_changed(self, value: bool) -> None:
        self._masking_roi_table_name.visible = value
        self._masking_roi_backend.visible = value

    def _toggle_advanced(self) -> None:
        self._advanced_visible = not self._advanced_visible
        self._advanced_container.visible = self._advanced_visible
        self._advanced_toggle.text = (
            "▼ Advanced settings" if self._advanced_visible else "▶ Advanced settings"
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _get_selected_layer(self) -> "napari.layers.Labels | None":
        name = self._layer_picker.value
        if not name:
            return None
        for layer in self._viewer.layers:
            if layer.name == name and isinstance(layer, napari.layers.Labels):
                return layer
        return None

    def _save(self) -> None:
        label_name = self._label_name.value.strip()
        if not label_name:
            logger.warning("Label name cannot be empty.")
            return

        layer = self._get_selected_layer()
        if layer is None:
            logger.warning("No label layer selected.")
            return

        if self._source != "File":
            logger.warning(
                "Writing to remote (HTTP) stores is not supported. "
                "Use a local OME-Zarr file."
            )
            return

        zarr_url = self._zarr_url
        if not zarr_url:
            logger.warning("No Zarr URL set. Select an OME-Zarr store first.")
            return

        write_mode = self._write_mode.value
        axes_str = self._axes_names.value.strip() or None
        save_masking_roi = self._save_masking_roi.value
        masking_roi_table_name = self._masking_roi_table_name.value.strip()
        masking_roi_backend = self._masking_roi_backend.value
        label_array = np.asarray(layer.data)
        layer_scale = tuple(layer.scale)
        layer_translate = tuple(float(v) for v in layer.translate)

        self._btn_save.enabled = False
        self._btn_save.text = "Saving..."

        @thread_worker
        def _do():
            return self._do_save_impl(
                zarr_url=zarr_url,
                label_array=label_array,
                layer_scale=layer_scale,
                layer_translate=layer_translate,
                label_name=label_name,
                axes_str=axes_str,
                write_mode=write_mode,
                save_masking_roi=save_masking_roi,
                masking_roi_table_name=masking_roi_table_name,
                masking_roi_backend=masking_roi_backend,
            )

        worker = _do()  # type: ignore[call-arg]
        worker.returned.connect(self._on_save_complete)
        worker.errored.connect(self._on_save_error)
        worker.start()

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
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

    @staticmethod
    def _validate_full(
        label_array: np.ndarray,
        layer_scale: tuple,
        axes_str: str,
        img,
    ) -> tuple[bool, str]:
        """Check yx spatial extents, z plane count, and timepoint count."""
        axes_list = list(axes_str)
        img_axes = list(img.axes)

        for a in ("z", "y", "x"):
            if a not in img_axes or a not in axes_list:
                continue
            img_extent = img.shape[img_axes.index(a)] * getattr(img.pixel_size, a)
            lbl_idx = axes_list.index(a)
            lbl_extent = label_array.shape[lbl_idx] * abs(float(layer_scale[lbl_idx]))
            if img_extent > 0 and abs(img_extent - lbl_extent) > 0.01 * img_extent:
                return (
                    False,
                    f"Label extent along '{a}' ({lbl_extent:.4g} µm) does not match "
                    f"the OME-Zarr image ({img_extent:.4g} µm). "
                    "The label must cover the same spatial extent as the image.",
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

        y_scale = (
            abs(float(layer_scale[axes_list.index("y")])) if "y" in axes_list else 1.0
        )
        x_scale = (
            abs(float(layer_scale[axes_list.index("x")])) if "x" in axes_list else 1.0
        )

        # Image-relative world offset of the label's top-left corner
        y_world = float(layer_translate[-2]) - float(plate_translation[0])
        x_world = float(layer_translate[-1]) - float(plate_translation[1])
        y0_px = round(y_world / y_scale)
        x0_px = round(x_world / x_scale)

        # Full image extent at label's pixel size
        img_y = img.shape[img_axes.index("y")] if "y" in img_axes else 0
        img_x = img.shape[img_axes.index("x")] if "x" in img_axes else 0
        full_h = round(img_y * img.pixel_size.y / y_scale)
        full_w = round(img_x * img.pixel_size.x / x_scale)

        label_h = (
            label_array.shape[axes_list.index("y")] if "y" in axes_list else full_h
        )
        label_w = (
            label_array.shape[axes_list.index("x")] if "x" in axes_list else full_w
        )

        # Clamp to full image bounds (guards against floating-point edge off-by-one)
        y1_px = min(y0_px + label_h, full_h)
        x1_px = min(x0_px + label_w, full_w)

        if y0_px < 0 or x0_px < 0 or y0_px >= full_h or x0_px >= full_w:
            raise ValueError(
                f"Label offset (y={y0_px}, x={x0_px}) is outside the full image "
                f"({full_h}×{full_w} px). Cannot write label."
            )

        is_partial = y0_px != 0 or x0_px != 0 or y1_px != full_h or x1_px != full_w
        return y0_px, x0_px, y1_px, x1_px, full_h, full_w, is_partial

    # ------------------------------------------------------------------
    # Core save implementation
    # ------------------------------------------------------------------

    def _do_save_impl(
        self,
        zarr_url: str,
        label_array: np.ndarray,
        layer_scale: tuple,
        layer_translate: tuple,
        label_name: str,
        axes_str: str | None,
        write_mode: str,
        save_masking_roi: bool,
        masking_roi_table_name: str,
        masking_roi_backend: str,
    ) -> bool:
        try:
            container = open_ome_zarr_container(zarr_url, mode="a", cache=False)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to open OME-Zarr container: %s", exc)
            return False

        if not axes_str:
            axes_str = self._infer_axes(label_array.ndim, container)

        try:
            img = container.get_image(path=container.level_paths[0])
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to read reference image: %s", exc)
            return False

        existing_labels = container.list_labels()
        label_exists = label_name in existing_labels

        # Mode precondition checks
        if write_mode == _WM_NEW and label_exists:
            logger.warning(
                "Label '%s' already exists. Choose 'Reset existing label' or "
                "'Append to existing label' to modify it.",
                label_name,
            )
            return False
        if write_mode in (_WM_RESET, _WM_APPEND) and not label_exists:
            logger.warning(
                "Label '%s' does not exist. Choose 'New label (initialise)' or "
                "'Overwrite full label' to create it.",
                label_name,
            )
            return False

        # Expand singleton dims (e.g. 2D label for a z=1 image)
        try:
            label_array, layer_scale, axes_str = self._expand_label_dims(
                label_array, layer_scale, axes_str, img
            )
        except ValueError as exc:
            logger.warning("Cannot save label: %s", exc)
            return False

        pixelsize_yx, z_spacing, time_spacing = self._extract_pixel_sizes(
            layer_scale, axes_str, img
        )

        try:
            if write_mode == _WM_OVERWRITE:
                return self._save_full_overwrite(
                    container,
                    img,
                    label_array,
                    layer_scale,
                    axes_str,
                    label_name,
                    label_exists,
                    pixelsize_yx,
                    z_spacing,
                    time_spacing,
                    save_masking_roi,
                    masking_roi_table_name,
                    masking_roi_backend,
                )
            else:
                return self._save_partial(
                    container,
                    img,
                    label_array,
                    layer_scale,
                    layer_translate,
                    axes_str,
                    label_name,
                    write_mode,
                    pixelsize_yx,
                    z_spacing,
                    time_spacing,
                    save_masking_roi,
                    masking_roi_table_name,
                    masking_roi_backend,
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to save label '%s': %s", label_name, exc)
            return False

    def _save_full_overwrite(
        self,
        container,
        img,
        label_array,
        layer_scale,
        axes_str,
        label_name,
        label_exists,
        pixelsize_yx,
        z_spacing,
        time_spacing,
        save_masking_roi,
        masking_roi_table_name,
        masking_roi_backend,
    ) -> bool:
        ok, msg = self._validate_full(label_array, layer_scale, axes_str, img)
        if not ok:
            logger.warning("Cannot save label: %s", msg)
            return False

        label_obj = container.derive_label(
            name=label_name,
            shape=label_array.shape,
            pixelsize=pixelsize_yx,
            z_spacing=z_spacing,
            time_spacing=time_spacing,
            overwrite=label_exists,
        )
        label_obj.set_array(label_array)
        label_obj.consolidate()

        if save_masking_roi:
            self._write_masking_roi_table(
                container,
                label_obj,
                label_name,
                masking_roi_table_name,
                masking_roi_backend,
                overwrite=label_exists,
            )
        return True

    def _save_partial(
        self,
        container,
        img,
        label_array,
        layer_scale,
        layer_translate,
        axes_str,
        label_name,
        write_mode,
        pixelsize_yx,
        z_spacing,
        time_spacing,
        save_masking_roi,
        masking_roi_table_name,
        masking_roi_backend,
    ) -> bool:
        ok, msg = self._validate_tz(label_array, axes_str, img)
        if not ok:
            logger.warning("Cannot save label: %s", msg)
            return False

        plate_translation = getattr(self._roi_loader, "translation", (0.0, 0.0))
        y0, x0, y1, x1, full_h, full_w, is_partial = self._compute_write_region(
            layer_translate, layer_scale, plate_translation, label_array, axes_str, img
        )

        # Build the full shape for derive_label (modes New / Reset)
        axes_list = list(axes_str)
        full_shape = list(label_array.shape)
        if "y" in axes_list:
            full_shape[axes_list.index("y")] = full_h
        if "x" in axes_list:
            full_shape[axes_list.index("x")] = full_w
        full_shape = tuple(full_shape)

        if write_mode == _WM_APPEND:
            label_obj = container.get_label(label_name)
            label_obj.set_array(label_array, y=slice(y0, y1), x=slice(x0, x1))
            logger.info(
                "Appended label '%s' for region y=[%d:%d], x=[%d:%d]. "
                "Pixels outside this region remain unchanged.",
                label_name,
                y0,
                y1,
                x0,
                x1,
            )
        else:
            overwrite = write_mode == _WM_RESET
            label_obj = container.derive_label(
                name=label_name,
                shape=full_shape,
                pixelsize=pixelsize_yx,
                z_spacing=z_spacing,
                time_spacing=time_spacing,
                overwrite=overwrite,
            )
            if is_partial:
                label_obj.set_array(np.zeros(full_shape, dtype=label_array.dtype))
                label_obj.set_array(label_array, y=slice(y0, y1), x=slice(x0, x1))
                logger.warning(
                    "Label '%s' covers only y=[%d:%d], x=[%d:%d] of the full image "
                    "(%d×%d px). The rest is set to 0.",
                    label_name,
                    y0,
                    y1,
                    x0,
                    x1,
                    full_h,
                    full_w,
                )
            else:
                label_obj.set_array(label_array)

        label_obj.consolidate()

        if save_masking_roi:
            self._write_masking_roi_table(
                container,
                label_obj,
                label_name,
                masking_roi_table_name,
                masking_roi_backend,
                overwrite=True,
            )
        return True

    @staticmethod
    def _write_masking_roi_table(
        container,
        label_obj,
        label_name,
        masking_roi_table_name,
        masking_roi_backend,
        overwrite,
    ) -> None:
        if not masking_roi_table_name:
            masking_roi_table_name = f"{label_name}_masking_ROI_table"
        masking_table = label_obj.build_masking_roi_table()
        container.add_table(
            name=masking_roi_table_name,
            table=masking_table,
            backend=masking_roi_backend,
            overwrite=overwrite,
        )

    def _on_save_complete(self, success: bool) -> None:
        self._btn_save.text = "Save label to OME-Zarr"
        self._btn_save.enabled = bool(self._label_name.value.strip())
        if success:
            logger.info(
                "Label '%s' saved successfully.", self._label_name.value.strip()
            )
            if self._roi_loader is not None:
                self._roi_loader.refresh_labels()

    def _on_save_error(self, exc: Exception) -> None:
        self._btn_save.text = "Save label to OME-Zarr"
        self._btn_save.enabled = bool(self._label_name.value.strip())
        logger.error("Unexpected error during label save: %s", exc)
