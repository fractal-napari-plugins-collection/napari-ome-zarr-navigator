"""Integration tests for PlateBrowser using synthetic OME-Zarr plates.

Fixtures are defined in conftest.py and require no network access.
"""

import napari.layers
import numpy as np

from napari_ome_zarr_navigator.plate_browser import PlateBrowser

# ---------------------------------------------------------------------------
# initialize_filters
# ---------------------------------------------------------------------------


class TestInitializeFilters:
    def test_populates_wells(self, make_napari_viewer, qtbot, synthetic_plate_path):
        viewer = make_napari_viewer()
        browser = PlateBrowser(viewer)
        browser._zarr_selector.set_url(synthetic_plate_path)
        qtbot.waitUntil(lambda: browser.btn_load_roi.enabled, timeout=5000)
        assert set(browser.well.choices) == {"B03", "C04"}
        assert browser.select_well.enabled

    def test_wells_sorted(self, make_napari_viewer, qtbot, synthetic_plate_path):
        viewer = make_napari_viewer()
        browser = PlateBrowser(viewer)
        browser._zarr_selector.set_url(synthetic_plate_path)
        qtbot.waitUntil(lambda: browser.btn_load_roi.enabled, timeout=5000)
        assert list(browser.well.choices) == ["B03", "C04"]

    def test_empty_url_disables_buttons(self, make_napari_viewer, qtbot):
        viewer = make_napari_viewer()
        browser = PlateBrowser(viewer)
        browser._zarr_selector.set_url("")
        assert not browser.btn_load_roi.enabled
        assert not browser.select_well.enabled
        assert list(browser.well.choices) == []


# ---------------------------------------------------------------------------
# go_to_well
# ---------------------------------------------------------------------------


class TestGoToWell:
    def test_adds_shape_layer_for_selected_well(
        self, make_napari_viewer, qtbot, synthetic_plate_path
    ):
        viewer = make_napari_viewer()
        browser = PlateBrowser(viewer)
        browser._zarr_selector.set_url(synthetic_plate_path)
        qtbot.waitUntil(lambda: browser.btn_load_roi.enabled, timeout=5000)

        browser.well.value = "B03"
        browser.go_to_well()

        shape_layers = [
            layer for layer in viewer.layers if isinstance(layer, napari.layers.Shapes)
        ]
        assert any("B03" in layer.name for layer in shape_layers)

    def test_old_well_markers_replaced(
        self, make_napari_viewer, qtbot, synthetic_plate_path
    ):
        viewer = make_napari_viewer()
        browser = PlateBrowser(viewer)
        browser._zarr_selector.set_url(synthetic_plate_path)
        qtbot.waitUntil(lambda: browser.btn_load_roi.enabled, timeout=5000)

        # Keep a dummy image layer present so check_empty_layerlist never
        # fires and clears plate_store when go_to_well removes old markers.
        viewer.add_image(np.zeros((64, 64), dtype=np.uint8), name="_dummy")

        browser.well.value = "B03"
        browser.go_to_well()
        browser.well.value = "C04"
        browser.go_to_well()

        shape_names = [
            layer.name
            for layer in viewer.layers
            if isinstance(layer, napari.layers.Shapes)
        ]
        assert not any("B03" in name for name in shape_names)
        assert any("C04" in name for name in shape_names)


# ---------------------------------------------------------------------------
# launch_load_roi
# ---------------------------------------------------------------------------


class TestLaunchLoadRoi:
    def test_creates_roi_loader_widget(
        self, make_napari_viewer, qtbot, synthetic_plate_path
    ):
        viewer = make_napari_viewer()
        browser = PlateBrowser(viewer)
        browser._zarr_selector.set_url(synthetic_plate_path)
        qtbot.waitUntil(lambda: browser.btn_load_roi.enabled, timeout=5000)

        browser.well.value = "B03"
        browser.launch_load_roi()

        roi_loader = browser.roi_loader
        assert roi_loader is not None
        assert browser.roi_widget is not None

        # Wait for ALL async init steps to fully complete before the test ends.
        # Waiting for roi_choices_updated is not sufficient: on slower runners
        # (Linux CI), Qt queues one event per connected slot on the same
        # thread_worker.returned signal.  roi_choices_updated fires from the
        # first slot, but the second slot (_btn_ctrl.on_step_done, which
        # enables _run_button) is still in the event queue.  If teardown begins
        # before that event is processed, Qt deletes the button and the deferred
        # callback crashes.  Waiting for _run_button.enabled guarantees every
        # queued callback has run.
        qtbot.waitUntil(lambda: roi_loader._run_button.enabled, timeout=10000)


# ---------------------------------------------------------------------------
# Condition filter integration
# ---------------------------------------------------------------------------


class TestConditionFilter:
    def test_plate_condition_table_populates_selector(
        self,
        make_napari_viewer,
        qtbot,
        synthetic_plate_with_conditions_path,
    ):
        viewer = make_napari_viewer()
        browser = PlateBrowser(viewer)
        browser._zarr_selector.set_url(synthetic_plate_with_conditions_path)
        qtbot.waitUntil(lambda: browser.btn_load_roi.enabled, timeout=5000)

        browser._condition_table_source.value = "Plate-based condition table"
        qtbot.waitUntil(
            lambda: len(browser._cond_filter.condition_name_selector.choices) > 0,
            timeout=3000,
        )

        assert "differentiation_timepoint" in (
            browser._cond_filter.condition_name_selector.choices
        )

    def test_condition_filter_narrows_well_list(
        self,
        make_napari_viewer,
        qtbot,
        synthetic_plate_with_conditions_path,
    ):
        viewer = make_napari_viewer()
        browser = PlateBrowser(viewer)
        browser._zarr_selector.set_url(synthetic_plate_with_conditions_path)
        qtbot.waitUntil(lambda: browser.btn_load_roi.enabled, timeout=5000)

        # Load plate-based condition table
        browser._condition_table_source.value = "Plate-based condition table"
        qtbot.waitUntil(lambda: browser._cond_filter.df is not None, timeout=3000)

        # Enable the timepoint filter and select "day 0" (only B03)
        filter_widget = browser._cond_filter.filter_container[0]
        check_box, combo_box = filter_widget[1], filter_widget[0]
        combo_box.value = "day 0"
        check_box.value = True

        qtbot.waitUntil(lambda: list(browser.well.choices) == ["B03"], timeout=2000)
        assert list(browser.well.choices) == ["B03"]

    def test_switching_to_no_filter_restores_all_wells(
        self,
        make_napari_viewer,
        qtbot,
        synthetic_plate_with_conditions_path,
    ):
        viewer = make_napari_viewer()
        browser = PlateBrowser(viewer)
        browser._zarr_selector.set_url(synthetic_plate_with_conditions_path)
        qtbot.waitUntil(lambda: browser.btn_load_roi.enabled, timeout=5000)

        browser._condition_table_source.value = "Plate-based condition table"
        qtbot.waitUntil(lambda: browser._cond_filter.df is not None, timeout=3000)

        browser._condition_table_source.value = "No"
        qtbot.waitUntil(
            lambda: set(browser.well.choices) == {"B03", "C04"}, timeout=2000
        )
        assert set(browser.well.choices) == {"B03", "C04"}


# ---------------------------------------------------------------------------
# Regression: deactivating a filter must restore ALL plate wells
# ---------------------------------------------------------------------------


class TestFilterDeactivationRestoresAllWells:
    def test_partial_condition_table_filter_deactivation(
        self, make_napari_viewer, qtbot, synthetic_plate_partial_conditions_path
    ):
        """Disabling a filter must restore ALL plate wells, not just the subset
        covered by the condition table.

        Plate has 3 wells (B03, C04, D05).  The registration_errors condition
        table covers only B03 and C04.  Before the fix, disabling all
        checkboxes caused _filter_df to call
        get_plate_wells(filters={B03,C04}) instead of
        get_plate_wells(filters=None), so D05 never came back.
        """
        viewer = make_napari_viewer()
        browser = PlateBrowser(viewer)
        browser._zarr_selector.set_url(synthetic_plate_partial_conditions_path)
        qtbot.waitUntil(lambda: browser.btn_load_roi.enabled, timeout=5000)

        all_wells = set(browser.well.choices)
        assert all_wells == {"B03", "C04", "D05"}

        # Load the plate-based condition table
        browser._condition_table_source.value = "Plate-based condition table"
        qtbot.waitUntil(
            lambda: len(browser._cond_filter.condition_name_selector.choices) > 0,
            timeout=3000,
        )

        # Select the registration_errors table (covers only B03 and C04)
        browser._cond_filter.condition_name_selector.value = "registration_errors"
        qtbot.waitUntil(lambda: browser._cond_filter.df is not None, timeout=3000)

        # Enable the "reason" filter — first condition column, so index 0.
        reason_fw = browser._cond_filter.filter_container[0]
        check_box = reason_fw[1]
        check_box.value = True
        qtbot.waitUntil(lambda: len(browser.well.choices) < 3, timeout=2000)
        assert set(browser.well.choices) == {"B03", "C04"}  # D05 is absent

        # Deactivate the filter → ALL 3 wells must come back (including D05)
        check_box.value = False
        qtbot.waitUntil(lambda: set(browser.well.choices) == all_wells, timeout=2000)
        assert set(browser.well.choices) == all_wells
