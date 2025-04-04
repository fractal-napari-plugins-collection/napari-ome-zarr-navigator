#!/usr/bin/env python3
import contextlib
import logging
import string
from typing import Optional

from magicgui.widgets import Container, FileEdit, Label, LineEdit, RadioButtons
from napari.utils.notifications import show_info
from ngio import open_ome_zarr_container, open_ome_zarr_plate
from qtpy.QtCore import QTimer


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
        string.ascii_uppercase[numeric - 1]
    else:
        string.ascii_lowercase[numeric - 1]


def calculate_well_positions(plate_store, row, col, is_plate=True):
    zarr_plate = open_ome_zarr_plate(
        plate_store, cache=True, parallel_safe=False, mode="r"
    )
    # Load the first image of the selected well to get shape & pixel sizes
    # Makes the assumption that all images in all wells will have the same shapes
    image_store = zarr_plate.get_image_store(
        row=row,
        column=col,
        image_path=zarr_plate.get_well(row, col).paths()[0],
    )
    ome_zarr_container = open_ome_zarr_container(
        image_store, cache=True, mode="r"
    )
    level = ome_zarr_container.levels_paths[0]
    ome_zarr_image = ome_zarr_container.get_image(path=level)
    shape = ome_zarr_image.shape[-2:]
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
        log_entry = self.format(record)
        show_info(log_entry)


class ZarrSelector(Container):
    def __init__(self, label="Input Source", file_mode="d", debounce_ms=500):
        # Source type selector
        self._source_selector = RadioButtons(
            label="Source",
            choices=["File", "HTTP"],
            orientation="horizontal",
            value="File",
        )

        # Internal state to prevent redundant emits
        self._last_file_url = None
        self._last_http_url = None
        self._last_token = None

        # File path selector
        self._file_picker = FileEdit(label="Zarr file", mode=file_mode)

        # HTTP inputs
        self._http_url = LineEdit(label="Zarr URL")
        self._http_token = LineEdit(label="Token")
        with contextlib.suppress(TypeError):
            self._http_token.native.setEchoMode(2)  # hide token display

        # Stacked input fields
        self._stack = Container(
            widgets=[self._file_picker, self._http_url, self._http_token]
        )
        # Initially hide the HTTP fields
        self._http_url.hide()
        self._http_token.hide()

        # Assemble the container
        self._main = Container(
            widgets=[Label(value=label), self._source_selector, self._stack]
        )
        super().__init__(widgets=[self._main])

        # Debounce timer: Avoids reloading while someone modifies the URL
        self._timer = QTimer()
        self._timer.setInterval(debounce_ms)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._emit_changed)

        # Events
        self._source_selector.changed.connect(self._on_source_changed)
        self._file_picker.changed.connect(self._emit_changed)
        self._http_url.changed.connect(self._restart_timer)
        self._http_token.changed.connect(self._restart_timer)

        # Custom signal-style callback
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
        self._emit_changed()

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
            and (
                curr_http != self._last_http_url
                or curr_token != self._last_token
            )
        ):
            # Update state
            self._last_file_url = curr_file
            self._last_http_url = curr_http
            self._last_token = curr_token

            for cb in self._callbacks:
                cb()
        else:
            pass

    def on_change(self, callback):
        self._callbacks.append(callback)

    def set_url(self, zarr_url: str, token: Optional[str] = None):
        """Set a zarr_url"""
        self._file_picker.value = zarr_url
        self._http_url.value = zarr_url
        self._http_token.value = token or ""

    @property
    def source(self):
        return self._source_selector.value

    @property
    def url(self) -> str:
        if self.source == "File":
            return str(self._file_picker.value)
        else:
            return self._http_url.value.strip()

    @property
    def token(self) -> str:
        return self._http_token.value.strip() or None
