import logging
from pathlib import Path

import napari
import napari.layers
import numpy as np
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FileEdit,
    Label,
    LineEdit,
    PushButton,
)
from napari.qt.threading import thread_worker
from ngio import open_label, open_ome_zarr_container
from ngio.utils import StoreOrGroup, fractal_fsspec_store

try:
    from ngio.images._label import derive_label as _ngio_derive_label

    _NGIO_REMOTE_LABEL_AVAILABLE = True
except ImportError:
    _ngio_derive_label = None  # type: ignore[assignment]
    _NGIO_REMOTE_LABEL_AVAILABLE = False

from napari_ome_zarr_navigator.util import ZarrSelector

logger = logging.getLogger(__name__)

_BACKEND_CHOICES = ["csv", "anndata", "json", "parquet"]

# Write-mode constants (used as ComboBox values and in _do_save_impl branching)
_WM_NEW = "Save as new label"
_WM_EDIT = "Edit existing label"
_WM_RESET = "Reset existing label"
_WRITE_MODES = [_WM_NEW, _WM_EDIT, _WM_RESET]


class LabelSaverImage(Container):
    """Widget for saving a napari Labels layer to an OME-Zarr container.

    Can be used standalone (shows ZarrSelector) or launched from an ROI Loader
    with a pre-populated zarr URL (selector hidden).  After saving, the optional
    ``roi_loader`` reference is called to refresh its label picker.

    Four write modes (auto-selected based on label existence / shape):
    - New label (initialise): create full-size label, fill sub-region if partial
    - Overwrite full label: full-extent validation; replace existing label
    - Reset existing label: destroy & recreate at full size; fill sub-region
    - Edit existing label: patch-write only the loaded sub-region

    For HTTP/remote stores a local output folder must be chosen; the label is
    written to ``<folder>/labels/<label_name>/`` as a standalone OME-Zarr label.
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
        self._last_saved_label: str | None = None
        self._pending_save_label: str | None = None
        self._pending_save_masking_roi: bool = False
        self._zarr_url: str | None = zarr_url
        self._source: str = source
        self.is_local: bool = True
        self._store: StoreOrGroup | None = None

        self._zarr_selector = ZarrSelector(label="Image")

        self._remote_save_folder = FileEdit(
            label="Local output",
            mode="d",  # type: ignore[arg-type]
            tooltip=(
                "The plugin cannot write back to remote OME-Zarr stores. "
                "Choose a local folder — the label will be saved as "
                "<folder>/labels/<label_name>."
            ),
        )
        self._remote_save_folder.hide()

        self._layer_picker = ComboBox(
            label="Label layer",
            choices=self._get_label_layers(),
        )

        self._write_mode = ComboBox(
            label="Write mode",
            choices=_WRITE_MODES,
            value=_WM_NEW,
        )
        self._write_mode.tooltip = (
            "Save as new label: create a new label; fill sub-region if loaded from ROI.\n"
            "Edit existing label: patch-write only the loaded sub-region into an existing label.\n"
            "Reset existing label: destroy and recreate the label; fill sub-region."
        )

        self._label_name = LineEdit(label="Label name", value="")
        self._existing_label_picker = ComboBox(
            label="Existing label",
            choices=[],
        )
        self._existing_label_picker.visible = False

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
        self._btn_save.native.setStyleSheet("font-weight: bold;")

        _widgets = []
        if zarr_url:
            self._info_label = Label(value=f"Image: {Path(zarr_url).name}")
            self._info_label.tooltip = zarr_url
            _widgets.append(self._info_label)
        _widgets.extend(
            [
                self._zarr_selector,
                self._layer_picker,
                self._write_mode,
                self._label_name,
                self._existing_label_picker,
                self._save_masking_roi,
                self._masking_roi_table_name,
                self._masking_roi_backend,
                self._remote_save_folder,
                self._advanced_toggle,
                self._advanced_container,
                self._btn_save,
            ]
        )

        super().__init__(widgets=_widgets)

        self._zarr_selector.on_change(self._on_url_changed)
        self._remote_save_folder.changed.connect(self._on_remote_folder_changed)
        self._layer_picker.changed.connect(self._update_axes_inference)
        self._write_mode.changed.connect(self._on_write_mode_changed)
        self._label_name.changed.connect(self._on_label_name_changed)
        self._existing_label_picker.changed.connect(self._on_existing_label_changed)
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
        self.is_local = self._source == "File"

        if not self._zarr_url:
            self._store = None
        elif self.is_local:
            self._store = self._zarr_url
            self._remote_save_folder.hide()
        else:
            self._store = fractal_fsspec_store(
                self._zarr_url, fractal_token=self._zarr_selector.token
            )
            self._remote_save_folder.show()

        if self._store is not None:
            self._load_container_async(self._store)

        self._update_save_button_state()

    def _on_remote_folder_changed(self) -> None:
        if not self.is_local and self._write_mode.value in (_WM_EDIT, _WM_RESET):
            self._refresh_existing_label_picker_from_folder()
        self._update_save_button_state()

    def _refresh_existing_label_picker_from_folder(self) -> None:
        folder = str(self._remote_save_folder.value)
        if not folder or folder == ".":
            self._existing_label_picker.choices = []
            return
        labels_path = Path(folder) / "labels"
        if not labels_path.is_dir():
            self._existing_label_picker.choices = []
            return
        candidates = sorted(
            p.name
            for p in labels_path.iterdir()
            if p.is_dir() and (p / ".zattrs").exists()
        )
        self._existing_label_picker.choices = candidates
        self._update_save_button_state()

    def _load_container_async(self, store: StoreOrGroup) -> None:
        @thread_worker
        def _load():
            try:
                return open_ome_zarr_container(store, mode="r", cache=True)
            except Exception:  # noqa: BLE001
                return None

        w = _load()  # type: ignore[call-arg]
        w.returned.connect(self._on_container_ready)
        w.start()

    def _on_container_ready(self, container) -> None:
        self._ome_zarr_container = container
        self._update_axes_inference()
        self._refresh_existing_label_picker()

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
    # Write mode and existing-label picker
    # ------------------------------------------------------------------

    def _on_write_mode_changed(self, mode: str) -> None:
        is_existing = mode in (_WM_EDIT, _WM_RESET)
        self._label_name.visible = not is_existing
        self._existing_label_picker.visible = is_existing
        if is_existing:
            if self.is_local:
                self._refresh_existing_label_picker()
            else:
                self._refresh_existing_label_picker_from_folder()
        self._update_save_button_state()

    def _refresh_existing_label_picker(self) -> None:
        """Populate the existing-label picker from the container; revert to NEW if empty."""
        if self._ome_zarr_container is None:
            return
        try:
            labels = self._ome_zarr_container.list_labels()
        except Exception:  # noqa: BLE001
            labels = []
        if not labels and self._write_mode.value in (_WM_EDIT, _WM_RESET):
            logger.warning(
                "No existing labels found in this OME-Zarr. "
                "Switching to 'Save as new label'."
            )
            self._write_mode.value = _WM_NEW
            return
        self._existing_label_picker.choices = labels
        self._existing_label_picker._default_choices = labels
        preferred = self._last_saved_label
        if preferred and preferred in labels:
            self._existing_label_picker.value = preferred
        elif labels and self._existing_label_picker.value not in labels:
            self._existing_label_picker.value = labels[0]
        self._update_save_button_state()

    def _get_active_label_name(self) -> str:
        """Return the label name from whichever widget is active for the current mode."""
        if self._write_mode.value in (_WM_EDIT, _WM_RESET):
            return self._existing_label_picker.value or ""
        return self._label_name.value.strip()

    # ------------------------------------------------------------------
    # UI event handlers
    # ------------------------------------------------------------------

    def _update_save_button_state(self) -> None:
        if not self.is_local:
            folder = str(self._remote_save_folder.value)
            if not folder or folder == ".":
                self._btn_save.enabled = False
                return
        if self._write_mode.value in (_WM_EDIT, _WM_RESET):
            self._btn_save.enabled = bool(self._existing_label_picker.choices)
        else:
            self._btn_save.enabled = bool(self._label_name.value.strip())

    def _on_label_name_changed(self, value: str) -> None:
        stripped = value.strip()
        self._update_save_button_state()
        if stripped:
            self._masking_roi_table_name.value = f"{stripped}_masking_ROI_table"

    def _on_existing_label_changed(self, value: str) -> None:
        if value:
            self._masking_roi_table_name.value = f"{value}_masking_ROI_table"

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
        label_name = self._get_active_label_name()
        if not label_name:
            logger.warning("Label name cannot be empty.")
            return

        layer = self._get_selected_layer()
        if layer is None:
            logger.warning("No label layer selected.")
            return

        store = self._store
        if store is None:
            logger.warning("No Zarr store set. Select an OME-Zarr store first.")
            return

        output_folder: str | None = None
        if not self.is_local:
            output_folder = str(self._remote_save_folder.value)
            if not output_folder or output_folder == ".":
                logger.warning("Choose a local output folder for remote saves.")
                return

        write_mode = self._write_mode.value
        axes_str = self._axes_names.value.strip() or None
        save_masking_roi = self._save_masking_roi.value
        masking_roi_table_name = self._masking_roi_table_name.value.strip()
        masking_roi_backend = self._masking_roi_backend.value
        label_array = np.asarray(layer.data)
        layer_scale = tuple(layer.scale)
        layer_translate = tuple(float(v) for v in layer.translate)

        self._pending_save_label = label_name
        self._pending_save_masking_roi = save_masking_roi
        self._btn_save.enabled = False
        self._btn_save.text = "Saving..."

        @thread_worker
        def _do():
            return self._do_save_impl(
                store=store,
                label_array=label_array,
                layer_scale=layer_scale,
                layer_translate=layer_translate,
                label_name=label_name,
                axes_str=axes_str,
                write_mode=write_mode,
                save_masking_roi=save_masking_roi,
                masking_roi_table_name=masking_roi_table_name,
                masking_roi_backend=masking_roi_backend,
                output_folder=output_folder,
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
        store: StoreOrGroup,
        label_array: np.ndarray,
        layer_scale: tuple,
        layer_translate: tuple,
        label_name: str,
        axes_str: str | None,
        write_mode: str,
        save_masking_roi: bool,
        masking_roi_table_name: str,
        masking_roi_backend: str,
        output_folder: str | None = None,
    ) -> bool:
        read_mode = "r" if output_folder else "a"
        try:
            container = open_ome_zarr_container(store, mode=read_mode, cache=False)
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

        if output_folder:
            label_exists = (Path(output_folder) / "labels" / label_name).exists()
        else:
            label_exists = label_name in container.list_labels()

        # Mode precondition checks
        if write_mode == _WM_NEW and label_exists:
            logger.warning(
                "Label '%s' already exists. Choose 'Reset existing label' or "
                "'Edit existing label' to modify it.",
                label_name,
            )
            return False
        if write_mode in (_WM_RESET, _WM_EDIT) and not label_exists:
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
            if output_folder:
                return self._save_partial_remote(
                    img=img,
                    labels_path=Path(output_folder) / "labels",
                    output_folder=output_folder,
                    label_array=label_array,
                    layer_scale=layer_scale,
                    layer_translate=layer_translate,
                    axes_str=axes_str,
                    label_name=label_name,
                    write_mode=write_mode,
                    pixelsize_yx=pixelsize_yx,
                    z_spacing=z_spacing,
                    time_spacing=time_spacing,
                    save_masking_roi=save_masking_roi,
                    masking_roi_table_name=masking_roi_table_name,
                    masking_roi_backend=masking_roi_backend,
                    plate_translation=getattr(
                        self._roi_loader, "translation", (0.0, 0.0)
                    ),
                )
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

        if write_mode == _WM_EDIT:
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
                shape=self._insert_c_dim(full_shape, img),
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

    def _save_partial_remote(
        self,
        img,
        labels_path: Path,
        output_folder: str,
        label_array: np.ndarray,
        layer_scale: tuple,
        layer_translate: tuple,
        axes_str: str,
        label_name: str,
        write_mode: str,
        pixelsize_yx,
        z_spacing,
        time_spacing,
        save_masking_roi: bool,
        masking_roi_table_name: str,
        masking_roi_backend: str,
        plate_translation: tuple = (0.0, 0.0),
    ) -> bool:
        if not _NGIO_REMOTE_LABEL_AVAILABLE:
            logger.error(
                "Remote label save requires ngio.images._label.derive_label "
                "(private API). This may have changed in your version of ngio "
                "— please report this issue."
            )
            return False

        ok, msg = self._validate_tz(label_array, axes_str, img)
        if not ok:
            logger.warning("Cannot save label: %s", msg)
            return False

        y0, x0, y1, x1, full_h, full_w, is_partial = self._compute_write_region(
            layer_translate, layer_scale, plate_translation, label_array, axes_str, img
        )

        axes_list = list(axes_str)
        full_shape = list(label_array.shape)
        if "y" in axes_list:
            full_shape[axes_list.index("y")] = full_h
        if "x" in axes_list:
            full_shape[axes_list.index("x")] = full_w
        full_shape = tuple(full_shape)

        if write_mode == _WM_EDIT:
            label_obj = open_label(str(labels_path / label_name))
            label_obj.set_array(label_array, y=slice(y0, y1), x=slice(x0, x1))
        else:
            label_dir = labels_path / label_name
            labels_path.mkdir(parents=True, exist_ok=True)
            _ngio_derive_label(  # type: ignore[operator]
                store=label_dir,
                name=label_name,
                ref_image=img,
                shape=self._insert_c_dim(full_shape, img),
                pixelsize=pixelsize_yx,
                z_spacing=z_spacing,
                time_spacing=time_spacing,
                overwrite=(write_mode == _WM_RESET),
            )
            label_obj = open_label(str(labels_path / label_name))
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
            self._write_masking_roi_table_remote(
                label_obj,
                output_folder,
                masking_roi_table_name,
                masking_roi_backend,
                label_name,
            )
        return True

    @staticmethod
    def _write_masking_roi_table_remote(
        label_obj,
        output_folder: str,
        masking_roi_table_name: str,
        masking_roi_backend: str,
        label_name: str,
    ) -> None:
        from ngio.tables import open_tables_container

        if not masking_roi_table_name:
            masking_roi_table_name = f"{label_name}_masking_ROI_table"
        masking_table = label_obj.build_masking_roi_table()
        dest = str(Path(output_folder) / "tables")
        tc = open_tables_container(dest, mode="a")
        tc.add(
            masking_roi_table_name,
            masking_table,
            backend=masking_roi_backend,
            overwrite=True,
        )
        logger.info("Saved masking ROI table to '%s'.", dest)

    def _on_save_complete(self, success: bool) -> None:
        self._btn_save.text = "Save label to OME-Zarr"
        if success:
            saved_name = self._pending_save_label or ""
            if self._write_mode.value == _WM_NEW:
                self._last_saved_label = saved_name
            logger.info("Label '%s' saved successfully.", saved_name)
            if self.is_local and self._store is not None:
                self._load_container_async(self._store)
            elif not self.is_local and self._write_mode.value in (_WM_NEW, _WM_RESET):
                self._refresh_existing_label_picker_from_folder()
            if self._roi_loader is not None:
                self._roi_loader.refresh_labels()
                if self._pending_save_masking_roi:
                    self._roi_loader.refresh_roi_tables()
        self._update_save_button_state()

    def _on_save_error(self, exc: Exception) -> None:
        self._btn_save.text = "Save label to OME-Zarr"
        self._update_save_button_state()
        logger.error("Unexpected error during label save: %s", exc)
