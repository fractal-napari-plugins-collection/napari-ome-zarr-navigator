#!/usr/bin/env python3
import logging
import string
from enum import Enum, auto
from importlib.resources import files

from magicgui.widgets import (
    Container,
    FileEdit,
    LineEdit,
    PushButton,
    RadioButtons,
)
from napari.utils.notifications import show_info
from ngio import open_ome_zarr_container, open_ome_zarr_plate
from qtpy.QtCore import QTimer  # type: ignore[attr-defined]
from qtpy.QtGui import QIcon  # type: ignore[attr-defined]
from qtpy.QtWidgets import QLineEdit  # type: ignore[attr-defined]


def alpha_to_numeric(alpha: str) -> int:
    """Return the position of a single character in the alphabet

    Args:
        alpha: Single alphabet character

    Returns:
        Integer position in the alphabet
    """
    return ord(alpha.upper()) - 64


def numeric_to_alpha(numeric: int, upper: bool = True) -> str:
    """Return the upper or lowercase character for a given position in the alphabet

    Args:
        numeric: Integer position in the alphabet

    Returns:
        Single alphabet character
    """
    if upper:
        return string.ascii_uppercase[numeric - 1]
    else:
        return string.ascii_lowercase[numeric - 1]


def calculate_well_positions(plate_store, row, col, is_plate=True):
    zarr_plate = open_ome_zarr_plate(plate_store, cache=True, mode="r")
    # Load the first image of the selected well to get shape & pixel sizes
    # Makes the assumption that all images in all wells will have the same shapes
    image_store = zarr_plate.get_image_store(
        row=row,
        column=col,
        image_path=zarr_plate.get_well(row, col).paths()[0],
    )
    ome_zarr_container = open_ome_zarr_container(image_store, cache=True, mode="r")
    level = ome_zarr_container.level_paths[0]
    ome_zarr_image = ome_zarr_container.get_image(path=level)
    axes = ome_zarr_image.axes
    y_idx = axes.index("y")
    x_idx = axes.index("x")
    shape = (ome_zarr_image.shape[y_idx], ome_zarr_image.shape[x_idx])
    scale = (ome_zarr_image.pixel_size.y, ome_zarr_image.pixel_size.x)

    row_i = zarr_plate.rows.index(row)
    col_i = zarr_plate.columns.index(col)

    if is_plate:
        top_left_corner = [
            row_i * scale[0] * shape[0],
            col_i * scale[1] * shape[1],
        ]
    else:
        top_left_corner = [0, 0]
    bottom_right_corner = [
        top_left_corner[0] + scale[0] * shape[0],
        top_left_corner[1] + scale[1] * shape[1],
    ]
    return top_left_corner, bottom_right_corner


class NapariHandler(logging.Handler):
    def emit(self, record):
        from napari.utils.notifications import show_warning

        log_entry = self.format(record)
        if record.levelno >= logging.WARNING:
            show_warning(log_entry)
        else:
            show_info(log_entry)


# Attach once to the package root logger so all sub-module loggers
# (plate_browser, roi_loader, roi_annotator, …) route through it.
# propagate=False prevents double-printing via the root StreamHandler.
_pkg_logger = logging.getLogger("napari_ome_zarr_navigator")
if not any(isinstance(h, NapariHandler) for h in _pkg_logger.handlers):
    _handler = NapariHandler()
    _handler.setLevel(logging.INFO)
    _pkg_logger.addHandler(_handler)
    _pkg_logger.setLevel(logging.INFO)
    _pkg_logger.propagate = False


class LoaderState(Enum):
    INITIALIZING = auto()
    READY = auto()
    LOADING = auto()


class ZarrSelector(Container):
    def __init__(self, label="Image", file_mode="d", debounce_ms=500):
        # Source selector
        self._source_selector = RadioButtons(
            label="Source",
            choices=["File", "HTTP"],
            orientation="horizontal",
            value="File",  # type: ignore[call-arg]
            tooltip="File: select a local OME-Zarr directory.\nHTTP: enter a remote OME-Zarr URL.",
        )

        # internal state
        self._last_file_url = None
        self._last_http_url = None
        self._last_token = None
        self._last_source = "File"

        # Inputs
        self._file_picker = FileEdit(label="Zarr file", mode=file_mode)  # type: ignore[arg-type]
        if label == "Plate":
            self._file_picker.tooltip = "The path to the OME-Zarr plate"
        else:
            self._file_picker.tooltip = (
                "The path to the OME-Zarr image "
                "(if it's in a plate, e.g. /path/to/plate.zarr/A/01/0)"
            )
        self._http_url = LineEdit(label="Zarr URL")
        self._http_token = LineEdit(label="Token")

        # Mask token + add eye action (always white icons)
        le: QLineEdit = self._http_token.native
        le.setEchoMode(QLineEdit.Password)

        pkg = files("napari_ome_zarr_navigator")
        eye_icon = QIcon(str(pkg / "icons" / "eye.svg"))
        eye_off_icon = QIcon(str(pkg / "icons" / "eye-off.svg"))

        self._eye_action = le.addAction(eye_off_icon, QLineEdit.TrailingPosition)
        self._eye_action.setCheckable(True)

        def _toggle(checked: bool) -> None:
            le.setEchoMode(QLineEdit.Normal if checked else QLineEdit.Password)
            self._eye_action.setIcon(eye_icon if checked else eye_off_icon)

        self._eye_action.toggled.connect(_toggle)

        # Stack & layout
        self._stack = Container(
            widgets=[self._file_picker, self._http_url, self._http_token],
            labels=False,
        )
        self._http_url.hide()
        self._http_token.hide()

        self._main = Container(
            widgets=[self._source_selector, self._stack],  # type: ignore[arg-type]
            labels=False,
        )
        super().__init__(widgets=[self._main], labels=False, label=label)

        # Debounce
        self._timer = QTimer()
        self._timer.setInterval(debounce_ms)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._emit_changed)

        # Events
        self._source_selector.changed.connect(self._on_source_changed)
        self._file_picker.changed.connect(self._emit_changed)
        self._http_url.changed.connect(self._restart_timer)
        self._http_token.changed.connect(self._restart_timer)

        self._callbacks = []

    def _on_source_changed(self, value):
        if value == "File":
            self._file_picker.show()
            self._http_url.hide()
            self._http_token.hide()
        else:
            self._file_picker.hide()
            self._http_url.show()
            self._http_token.show()
        if value != self._last_source:
            self._last_source = value
            self._emit_source_changed()

    def _restart_timer(self, *args):
        self._timer.start()

    def _emit_changed(self):
        """
        Only emit if one of the URL/token values changed. Important to avoid
        false-positive re-initializations of the whole plate.
        """
        curr_file = str(self._file_picker.value)
        curr_http = self._http_url.value.strip()
        curr_token = self._http_token.value.strip()

        if (self.source == "File" and curr_file != self._last_file_url) or (
            self.source == "HTTP"
            and (curr_http != self._last_http_url or curr_token != self._last_token)
        ):
            # Update state
            self._last_file_url = curr_file
            self._last_http_url = curr_http
            self._last_token = curr_token

            for cb in self._callbacks:
                cb()
        else:
            pass

    def _emit_source_changed(self):
        """Emit callback if the source type changed (File <-> HTTP)."""
        for cb in self._callbacks:
            cb()

    def on_change(self, callback):
        self._callbacks.append(callback)

    def set_url(self, zarr_url: str, token: str | None = None):
        """Set a zarr_url"""
        self._file_picker.value = zarr_url  # type: ignore[assignment]
        self._http_url.value = zarr_url
        self._http_token.value = token or ""

    def configure(self, source: str, url: str, token: str | None = None) -> None:
        """Set source, URL and token together.

        Use when pre-populating from another widget so the source (File vs HTTP)
        is preserved and credentials are not lost. set_url() alone does not set
        the source selector, so HTTP tokens would be silently ignored.
        """
        self._source_selector.value = source  # triggers _on_source_changed
        self._file_picker.value = url  # type: ignore[assignment]
        self._http_url.value = url
        self._http_token.value = token or ""

    @property
    def source(self):
        return self._source_selector.value

    @property
    def url(self) -> str:
        if self.source == "File":
            return str(self._file_picker.value)
        else:
            return self._http_url.value.strip().rstrip("/")

    @property
    def token(self) -> str | None:
        return self._http_token.value.strip() or None


class LoaderButtonController:
    """Manages a PushButton's text, enabled state, and loading/init animations.

    Owns both QTimers so ROILoader doesn't need to touch them directly.
    Call begin_init(n) before launching n async init steps, then on_step_done()
    from each step's returned callback. set_state() handles LOADING animation.
    """

    def __init__(self, button: PushButton, ready_label: str = "Load ROI") -> None:
        self._button = button
        self._ready_label = ready_label
        self._dots = 0
        self._pending = 0
        self.current_state: LoaderState | None = None

        self._loading_timer = QTimer()  # type: ignore[call-arg]
        self._loading_timer.setInterval(300)
        self._loading_timer.timeout.connect(self._animate_loading)

        self._init_timer = QTimer()  # type: ignore[call-arg]
        self._init_timer.setSingleShot(True)
        self._init_timer.setInterval(150)
        self._init_timer.timeout.connect(self._show_initializing)

    def set_state(self, new: LoaderState) -> None:
        """Set button text/enabled and manage the loading dot animation."""
        self._loading_timer.stop()
        self.current_state = new
        if new is LoaderState.INITIALIZING:
            self._button.enabled = False
            self._button.text = "Initializing"
        elif new is LoaderState.READY:
            self._button.enabled = True
            self._button.text = self._ready_label
        elif new is LoaderState.LOADING:
            self._button.enabled = False
            self._button.text = "Loading"
            self._dots = 0
            self._loading_timer.start()

    def begin_init(self, n_steps: int = 1) -> None:
        """Disable button and restart the debounce timer for n pending steps."""
        self._button.enabled = False
        self._pending = n_steps
        if self._init_timer.isActive():
            self._init_timer.stop()
        self._init_timer.start()

    def on_step_done(self, *_) -> None:
        """Call when one init step completes. Goes READY once all steps are done."""
        self._pending = max(0, self._pending - 1)
        if self._pending == 0:
            if self._init_timer.isActive():
                self._init_timer.stop()
            self.set_state(LoaderState.READY)

    def _show_initializing(self) -> None:
        if self._pending > 0:
            self.set_state(LoaderState.INITIALIZING)

    def _animate_loading(self) -> None:
        self._dots = (self._dots + 1) % 4
        self._button.text = "Loading" + "." * self._dots
