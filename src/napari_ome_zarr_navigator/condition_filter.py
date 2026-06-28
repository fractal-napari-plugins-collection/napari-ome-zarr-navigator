import logging
from collections.abc import Callable

import pandas as pd
from magicgui.widgets import CheckBox, ComboBox, Container
from ngio.utils import NgioValueError
from qtpy.QtCore import QObject, Signal  # type: ignore[attr-defined]

from napari_ome_zarr_navigator.plate_manager import PlateManager
from napari_ome_zarr_navigator.well_utils import parse_condition_table

logger = logging.getLogger(__name__)


class ConditionFilterSignals(QObject):
    wells_changed = Signal(list)

    def __init__(self) -> None:
        super().__init__()


class ConditionTableFilter:
    """Owns all condition-table and filter-widget logic for PlateBrowser.

    Emits `signals.wells_changed` (list[str]) whenever the active filter
    changes so the caller can update the well selector.
    """

    def __init__(self, plate_manager: PlateManager) -> None:
        self._plate_mgr = plate_manager
        self.signals = ConditionFilterSignals()

        self.df: pd.DataFrame | None = None
        self._filter_names = None
        self.condition_tables: list[str] = []

        self.condition_name_selector = ComboBox()
        # Choices come from the zarr plate, not from napari layers. Prevent
        # magicgui's Container.reset_choices walk (triggered on every layer
        # insertion) from resetting this widget to its empty default.
        self.condition_name_selector.reset_choices = lambda *_: None
        self.filter_container = Container(layout="vertical", visible=False)

    def reset(self) -> None:
        """Clear filter state and emit all available wells."""
        self.df = None
        self._filter_names = None
        self.filter_container.clear()
        self.filter_container.visible = False
        if self._plate_mgr.zarr_plate is not None:
            wells_str, _ = self._plate_mgr.get_plate_wells()
            self.signals.wells_changed.emit(wells_str)

    def setup_filters(self, df: pd.DataFrame) -> None:
        """Build filter widgets from df and emit the initial well list."""
        self.df = df
        df_without_pk = df.drop(columns=["row", "col"])
        self._filter_names = df_without_pk.columns

        filter_widgets = [
            Container(
                widgets=[
                    ComboBox(
                        choices=df_without_pk[name]
                        .sort_values()  # type: ignore[call-overload]
                        .unique(),
                        enabled=False,
                    ),
                    CheckBox(label=name, value=False),
                ],
                layout="horizontal",
            )
            for name in self._filter_names
        ]

        self.filter_container.clear()
        self.filter_container.extend(filter_widgets)  # type: ignore[arg-type]
        self.filter_container.visible = True

        for i, fw in enumerate(filter_widgets):
            combo_box, check_box = fw[0], fw[1]
            check_box.changed.connect(self._toggle_filter(i))
            combo_box.changed.connect(self._filter_df)
            check_box.changed.connect(self._filter_df)

        # Emit initial (unfiltered) well list
        wells_str, _ = self._plate_mgr.get_plate_wells()
        self.signals.wells_changed.emit(wells_str)

    def init_condition_tables(self, zarr_plate) -> None:
        """Populate the condition-table selector from the plate."""
        try:
            tables = zarr_plate.list_tables(filter_types="condition_table")
        except NgioValueError:
            tables = []

        if tables:
            self.condition_tables = tables
            self.condition_name_selector.choices = tables
            self.condition_name_selector.visible = True
            self.filter_container.visible = True
        else:
            self.condition_name_selector.visible = False
            self.filter_container.visible = False
            logger.info("No plate condition tables found")

    def clear_condition_tables(self) -> None:
        """Reset the condition-table selector and hide filter UI."""
        self.condition_tables = []
        self.condition_name_selector.choices = []
        self.condition_name_selector.visible = False
        self.filter_container.visible = False

    def load_plate_condition_table(
        self, zarr_plate, table_name: str
    ) -> pd.DataFrame | None:
        """Load and normalise a condition table from the plate."""
        try:
            raw = zarr_plate.get_condition_table(name=table_name).dataframe
        except NgioValueError:
            return None
        return parse_condition_table(raw)

    def init_image_condition_tables(self, zarr_plate) -> None:
        """Populate condition-table selector from image-level tables in the plate."""
        try:
            tables = zarr_plate.list_image_tables(filter_types="condition")
        except NgioValueError:
            tables = []
        if tables:
            self.condition_tables = tables
            self.condition_name_selector.choices = tables
            self.condition_name_selector.visible = True
            self.filter_container.visible = True
        else:
            self.condition_name_selector.visible = False
            self.filter_container.visible = False
            logger.info("No condition tables found in the images")

    def load_image_condition_table(
        self, zarr_plate, table_name: str
    ) -> pd.DataFrame | None:
        """Aggregate a condition table from all images via ngio."""
        try:
            raw = zarr_plate.concatenate_image_tables(
                name=table_name, strict=False
            ).dataframe
        except NgioValueError:
            return None
        df = parse_condition_table(raw)
        return df.drop(columns=["path_in_well"], errors="ignore")

    def _toggle_filter(self, i: int) -> Callable:
        def _toggle() -> None:
            filter_row = self.filter_container[i]
            combo_box, check_box = filter_row[0], filter_row[1]
            combo_box.enabled = check_box.value

        return _toggle

    def _filter_df(self) -> None:
        if self._filter_names is None or self.df is None:
            return

        any_active = False
        filter_conditions = []
        for i, name in enumerate(self._filter_names):
            combo_box, check_box = (
                self.filter_container[i][0],
                self.filter_container[i][1],
            )
            if check_box.value:
                any_active = True
                condition = self.df[name] == combo_box.value
            else:
                condition = pd.Series(True, index=self.df.index)
            filter_conditions.append(condition)

        if not any_active:
            # No active filters — return all plate wells, not just the subset
            # covered by the condition table (which may not include every well).
            wells, _ = self._plate_mgr.get_plate_wells()
            self.signals.wells_changed.emit(wells)
            return

        and_filter = pd.concat(filter_conditions, axis=1).all(axis=1)
        tbl = self.df.loc[and_filter]
        rowcol_set = {(r, int(c)) for r, c in zip(tbl["row"], tbl["col"], strict=True)}
        wells, _ = self._plate_mgr.get_plate_wells(filters=rowcol_set)
        self.signals.wells_changed.emit(wells)
