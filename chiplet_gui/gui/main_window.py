"""The application window: tabs for layout, power, setup and results."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core import analysis, traces
from ..core.project import Project, load_any
from ..core.runner import copy_judge_inputs, load_run_result
from ..core.simcore import find_hotspot_binary
from . import style
from .canvas_panel import CanvasPanel
from .controller import RunController
from .panels import LinkPanel, PlacementPanel
from .power_panel import PowerPanel
from .render import power_density_image, temperature_image
from .results_panel import ResultsPanel
from .setup_panel import SetupPanel

PROJECT_FILTER = "Chiplet GUI project (*.json);;All files (*)"
LAYOUT_FILTER = "Layout or project JSON (*.json);;All files (*)"
TRACE_FILTER = "Power traces JSON (*.json);;All files (*)"

#: Overlay refresh delay in ms. Validation is instant; the power preview is not.
OVERLAY_DELAY = 60


class LogPanel(QWidget):
    """Run log, progress and the stop button."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(5000)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        self.status = QLabel("")
        self.status.setProperty("role", "muted")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)

        row = QHBoxLayout()
        row.addWidget(self.progress, 1)
        row.addWidget(self.status, 2)
        row.addWidget(self.stop_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 6)
        layout.addLayout(row)
        layout.addWidget(self.text, 1)

    def log(self, message: str) -> None:
        self.text.appendPlainText(message)


class MainWindow(QMainWindow):
    """Four views over one editable design, plus the run controller."""

    def __init__(self, project: Project | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Chiplet Thermal Studio")
        self.resize(1720, 1000)

        self.project = project or Project.default()
        if not self.project.run.hotspot_path:
            found = find_hotspot_binary()
            if found:
                self.project.run.hotspot_path = str(found)

        self._report: analysis.LayoutReport | None = None
        self._grid: analysis.RouteGrid | None = None
        self._dirty = False
        self._routes_block_legality = False
        self._runs: list = []
        self._current_run = -1

        self.controller = RunController(self)
        self._overlay_timer = QTimer(self)
        self._overlay_timer.setSingleShot(True)
        self._overlay_timer.setInterval(OVERLAY_DELAY)
        self._overlay_timer.timeout.connect(self._refresh_overlay)

        self._build_ui()
        self._build_actions()
        self._connect()

        self.canvas_panel.set_workloads(list(self.project.workloads))
        self.revalidate()
        self.log.log(
            "Chiplet Thermal Studio ready. Drag the chiplets, draw wires on the Wires tab, "
            "then Run simulation."
        )
        if self.project.run.hotspot_path:
            self.log.log(f"HotSpot binary: {self.project.run.hotspot_path}")
        else:
            self.log.log(
                "No HotSpot binary found. Set one in Simulation setup before running."
            )

    # ------------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        self.canvas_panel = CanvasPanel(self.project)
        self.power_panel = PowerPanel(self.project)
        self.setup_panel = SetupPanel(self.project)
        self.results_panel = ResultsPanel()

        self.tabs = QTabWidget()
        self.tabs.addTab(self.canvas_panel, "Layout")
        self.tabs.addTab(self.power_panel, "Power traces")
        self.tabs.addTab(self.setup_panel, "Simulation setup")
        self.tabs.addTab(self.results_panel, "Results")
        self.setCentralWidget(self.tabs)

        self.placement_panel = PlacementPanel(self.project)
        self.link_panel = LinkPanel(self.project)
        side = QTabWidget()
        side.addTab(self.placement_panel, "Placement")
        side.addTab(self.link_panel, "Wires")
        self.side_tabs = side

        dock = QDockWidget("Design", self)
        dock.setObjectName("design_dock")
        dock.setWidget(side)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        dock.setMinimumWidth(360)
        self.design_dock = dock

        self.log = LogPanel()
        log_dock = QDockWidget("Run log", self)
        log_dock.setObjectName("log_dock")
        log_dock.setWidget(self.log)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, log_dock)
        log_dock.setMinimumHeight(150)
        self.log_dock = log_dock

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.legality_label = QLabel("")
        self.coords_label = QLabel("")
        self.coords_label.setProperty("role", "muted")
        self.status.addWidget(self.legality_label, 1)
        self.status.addPermanentWidget(self.coords_label)

    def _build_actions(self) -> None:
        bar = self.addToolBar("Main")
        bar.setObjectName("main_toolbar")
        bar.setMovable(False)

        def action(text: str, handler, shortcut: str = "", tip: str = "") -> QAction:
            item = QAction(text, self)
            item.triggered.connect(handler)
            if shortcut:
                item.setShortcut(QKeySequence(shortcut))
            item.setToolTip(tip or text)
            return item

        self.act_new = action("New", self.new_project, "Ctrl+N", "Start from the reference layout")
        self.act_open = action("Open...", self.open_project, "Ctrl+O", "Open a saved design")
        self.act_save = action("Save", self.save_project, "Ctrl+S", "Save layout, wires, traces and setup")
        self.act_save_as = action("Save as...", self.save_project_as, "Ctrl+Shift+S")
        self.act_import = action(
            "Import layout...",
            self.import_layout,
            "",
            "Read a challenge --layouts file, a bare placement map or another project",
        )
        self.act_export = action(
            "Export for judge...",
            self.export_judge_layout,
            "",
            "Write a --layouts file for run_simulations.py",
        )
        self.act_autoroute = action(
            "Auto-route all",
            self.bake_auto_routes,
            "",
            "Copy the judge's own routes into editable wires",
        )
        self.act_clear_routes = action(
            "Clear wires", self.clear_all_routes, "", "Drop every hand-drawn route"
        )
        self.act_run = action("Run simulation", self.run_simulation, "F5")
        self.act_export_inputs = action(
            "Export HotSpot inputs",
            self.export_inputs_only,
            "",
            "Write the floorplans and power traces without running HotSpot",
        )
        self.act_stop = action("Stop", self.controller.cancel, "Esc")
        self.act_stop.setEnabled(False)
        self.act_fit = action("Fit view", self.canvas_panel.canvas.fit, "Ctrl+0")
        self.act_screenshot = action("Save canvas PNG...", self.save_canvas_png)

        for item in (self.act_new, self.act_open, self.act_save):
            bar.addAction(item)
        bar.addSeparator()
        for item in (self.act_import, self.act_export):
            bar.addAction(item)
        bar.addSeparator()
        for item in (self.act_autoroute, self.act_clear_routes):
            bar.addAction(item)
        bar.addSeparator()
        for item in (self.act_run, self.act_stop, self.act_export_inputs):
            bar.addAction(item)
        bar.addSeparator()
        bar.addAction(self.act_fit)

        file_menu = self.menuBar().addMenu("&File")
        for item in (
            self.act_new,
            self.act_open,
            self.act_save,
            self.act_save_as,
            None,
            self.act_import,
            self.act_export,
            self.act_export_inputs,
            self.act_screenshot,
        ):
            file_menu.addSeparator() if item is None else file_menu.addAction(item)
        file_menu.addSeparator()
        file_menu.addAction(action("Quit", self.close, "Ctrl+Q"))

        design_menu = self.menuBar().addMenu("&Design")
        design_menu.addAction(self.act_autoroute)
        design_menu.addAction(self.act_clear_routes)
        design_menu.addAction(action("Validate now", self.revalidate, "Ctrl+R"))

        run_menu = self.menuBar().addMenu("&Run")
        run_menu.addAction(self.act_run)
        run_menu.addAction(self.act_stop)
        run_menu.addAction(action("Load a previous run...", self.load_previous_run))

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.act_fit)
        view_menu.addAction(self.design_dock.toggleViewAction())
        view_menu.addAction(self.log_dock.toggleViewAction())

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(action("About this tool", self.show_about))

    def _connect(self) -> None:
        canvas = self.canvas_panel.canvas
        canvas.placementsChanged.connect(self._on_placements_changed)
        canvas.placementsCommitted.connect(self._mark_dirty)
        canvas.dieSelected.connect(self.placement_panel.select_dies)
        canvas.routeCommitted.connect(self._on_route_committed)
        canvas.hovered.connect(self.coords_label.setText)
        self.canvas_panel.overlayChanged.connect(self._schedule_overlay)
        self.canvas_panel.requestRouteExit.connect(self._update_hint)

        self.placement_panel.placementEdited.connect(self._on_placement_typed)
        self.placement_panel.dieSelected.connect(canvas.select_dies)
        self.placement_panel.exampleRequested.connect(self.load_example_layout)
        self.placement_panel.randomiseRequested.connect(self.randomise_layout)
        self.placement_panel.snapAllRequested.connect(self.snap_all)

        self.link_panel.drawRequested.connect(self.begin_route)
        self.link_panel.autoRequested.connect(self.drop_route)
        self.link_panel.bakeAllRequested.connect(self.bake_auto_routes)
        self.link_panel.clearAllRequested.connect(self.clear_all_routes)

        self.power_panel.workloadEdited.connect(self._on_power_edited)
        self.power_panel.structureChanged.connect(self._on_workloads_restructured)
        self.power_panel.enabledChanged.connect(self._on_workload_enabled)
        self.power_panel.selectionChanged.connect(self._on_power_selection)
        self.power_panel.dieHighlighted.connect(self._highlight_die)
        self.power_panel.importRequested.connect(self.import_traces)
        self.power_panel.exportRequested.connect(self.export_traces)

        self.setup_panel.settingsChanged.connect(self._on_setup_changed)
        self.setup_panel.geometryChanged.connect(self._on_geometry_changed)
        self.setup_panel.hotspotPathChanged.connect(lambda _: self._mark_dirty())

        self.results_panel.workloadSelected.connect(self._on_result_workload)
        self.results_panel.runSelected.connect(self.show_run)
        self.results_panel.exportResultsRequested.connect(self.export_results)
        self.results_panel.exportInputsRequested.connect(self.export_judge_inputs)
        self.results_panel.openFolderRequested.connect(self.open_run_folder)
        self.results_panel.loadRunRequested.connect(self.load_previous_run)

        self.controller.progress.connect(self._on_progress)
        self.controller.logged.connect(self.log.log)
        self.controller.runFinished.connect(self._on_run_finished)
        self.controller.inputsExported.connect(self._on_inputs_exported)
        self.controller.failed.connect(self._on_run_failed)
        self.controller.cancelled.connect(self._on_run_cancelled)
        self.controller.busyChanged.connect(self._on_busy)
        self.log.stop_button.clicked.connect(self.controller.cancel)

    # ------------------------------------------------------------ validation
    def revalidate(self) -> None:
        """Re-run the judge's checks and refresh every view that depends on them."""

        self._report = analysis.validate(self.project, check_routes=True)
        self._grid = analysis.route_grid(self.project, self._report)
        self._routes_block_legality = False

        if not self._report.legal and self.project.wire_routes_grid:
            probe = self.project.copy()
            probe.wire_routes_grid = {}
            if analysis.validate(probe, check_routes=True).legal:
                self._routes_block_legality = True

        self.canvas_panel.canvas.set_report(self._report, self._grid)
        self.placement_panel.refresh(self._report)
        self.link_panel.refresh(self._report)
        self._update_status()
        self._schedule_overlay()

    def _update_status(self) -> None:
        report = self._report
        if report is None:
            return
        if report.legal:
            used, limit = report.weighted_route, report.route_limit
            manual = len(self.project.wire_routes_grid)
            text = (
                f"Legal layout. Weighted route {used:.0f} of {limit:.0f} GB/s*mm"
                f" ({used / limit * 100:.0f}% of budget)."
            )
            if manual:
                text += f" {manual} hand-drawn wire(s)."
            self.legality_label.setText(text)
            self.legality_label.setStyleSheet(f"color: {style.OK_COLOR.name()};")
        else:
            text = f"Illegal: {report.message}"
            if self._routes_block_legality:
                text += "  The hand-drawn wires no longer fit; use Clear wires."
            self.legality_label.setText(text)
            self.legality_label.setStyleSheet(f"color: {style.ERROR_COLOR.name()};")
        self._update_hint()

    def _update_hint(self) -> None:
        self.canvas_panel.hint_label.setText(
            "Drag a chiplet to move it. Arrow keys nudge by the snap step, shift for ten. "
            "Middle-drag pans, the wheel zooms."
        )

    def _on_placements_changed(self) -> None:
        self.revalidate()

    def _on_placement_typed(self, name: str, x_mm: float, y_mm: float) -> None:
        x_mm, y_mm = analysis.clamp_into_interposer(self.project, name, x_mm, y_mm)
        self.project.placements_mm[name] = analysis.round_origin(x_mm, y_mm)
        self.canvas_panel.canvas.sync_from_project()
        self._mark_dirty()
        self.revalidate()

    # --------------------------------------------------------------- overlays
    def _schedule_overlay(self) -> None:
        self._overlay_timer.start()

    def _refresh_overlay(self) -> None:
        """Rebuild the canvas field and the per-die readouts for the current view."""

        key = self.canvas_panel.overlay_key
        workload = self.canvas_panel.workload
        annotations: dict[str, str] = {}

        if key == "temperature":
            result = self._runs[self._current_run] if 0 <= self._current_run < len(self._runs) else None
            temperatures = result.temperature_map(workload) if result and workload else None
            if temperatures is None:
                self.canvas_panel.set_field(None)
            else:
                low = min(float(entry.temperature_map.min()) for entry in result.workloads.values()
                          if entry.temperature_map is not None)
                high = max(float(entry.temperature_map.max()) for entry in result.workloads.values()
                           if entry.temperature_map is not None)
                self.canvas_panel.set_field(temperature_image(temperatures, low, high))
                peaks = result.workloads[workload].die_peak_c
                annotations = {name: f"{value:.1f} C" for name, value in peaks.items()}
            self.canvas_panel.canvas.set_die_annotations(annotations)
            self.power_panel.set_preview(None, "Showing temperatures from the last run.")
            return

        if key == "none" or not workload:
            self.canvas_panel.set_field(None)
            self.canvas_panel.canvas.set_die_annotations({})
            if key == "none":
                self.power_panel.set_preview(None, "")
            return

        try:
            preview = analysis.power_preview(self.project, workload, self._report)
        except Exception as exc:
            self.canvas_panel.set_field(None)
            self.canvas_panel.canvas.set_die_annotations({})
            self.power_panel.set_preview(None, f"No resolved power: {exc}")
            return

        self.canvas_panel.set_field(power_density_image(preview, self.project, mode=key))
        annotations = {
            name: f"{preview.die_total_w(name):.1f} W" for name in preview.die_maps_w
        }
        self.canvas_panel.canvas.set_die_annotations(annotations)
        if self.power_panel.current == workload:
            self.power_panel.set_preview(preview)
        else:
            self.power_panel.set_preview(None, "")

    # ------------------------------------------------------------------ wires
    def begin_route(self, link_id: str) -> None:
        if self._report is None or not self._report.legal:
            QMessageBox.information(
                self,
                "Draw wire",
                "Make the placement legal first. Ports and track access are only defined "
                "for a legal layout.",
            )
            return
        self.tabs.setCurrentWidget(self.canvas_panel)
        self.canvas_panel.canvas.begin_route(link_id)
        self.canvas_panel.show_route_banner(link_id)
        self.canvas_panel.canvas.setFocus()

    def _on_route_committed(self, link_id: str, path: list) -> None:
        previous = self.project.wire_routes_grid.get(link_id)
        self.project.wire_routes_grid[link_id] = path
        report = analysis.validate(self.project, check_routes=True)
        if not report.legal:
            if previous is None:
                self.project.wire_routes_grid.pop(link_id, None)
            else:
                self.project.wire_routes_grid[link_id] = previous
            QMessageBox.warning(
                self,
                "Wire rejected",
                f"The judge rejected that route for {link_id}:\n\n{report.message}",
            )
        else:
            self.log.log(f"{link_id}: hand-drawn route of {len(path)} track cells accepted.")
            self._mark_dirty()
        self.revalidate()

    def drop_route(self, link_id: str) -> None:
        if self.project.wire_routes_grid.pop(link_id, None) is not None:
            self.log.log(f"{link_id}: back to the automatic route.")
            self._mark_dirty()
            self.revalidate()

    def bake_auto_routes(self) -> None:
        try:
            routes = analysis.auto_routes(self.project)
        except ValueError as exc:
            QMessageBox.warning(self, "Auto-route", str(exc))
            return
        self.project.wire_routes_grid = routes
        self.log.log(f"Copied {len(routes)} automatic route(s) into editable wires.")
        self._mark_dirty()
        self.revalidate()

    def clear_all_routes(self) -> None:
        if not self.project.wire_routes_grid:
            return
        self.project.wire_routes_grid = {}
        self.log.log("Cleared every hand-drawn wire; the judge's router decides again.")
        self._mark_dirty()
        self.revalidate()

    # ----------------------------------------------------------- layout tools
    def load_example_layout(self, name: str) -> None:
        from ..core.simcore import default_spec

        examples = default_spec().get("example_layouts_mm", {})
        if name not in examples:
            QMessageBox.warning(self, "Reference layout", f"'{name}' is not in the challenge spec.")
            return
        self.project.placements_mm = {k: [float(v) for v in xy] for k, xy in examples[name].items()}
        self.project.wire_routes_grid = {}
        self.canvas_panel.canvas.sync_from_project()
        self.log.log(f"Loaded the reference layout '{name}'.")
        self._mark_dirty()
        self.revalidate()

    def randomise_layout(self, seed: int, jitter: float) -> None:
        try:
            placements = analysis.random_legal_layout(self.project, seed=seed, jitter_mm=jitter)
        except RuntimeError as exc:
            QMessageBox.warning(self, "Randomise", str(exc))
            return
        self.project.placements_mm = placements
        self.project.wire_routes_grid = {}
        self.canvas_panel.canvas.sync_from_project()
        self.log.log(f"Randomised with seed {seed} and {jitter:g} mm jitter.")
        self._mark_dirty()
        self.revalidate()

    def snap_all(self) -> None:
        step = self.canvas_panel.canvas.snap_mm or 0.1
        for name, (x_mm, y_mm) in list(self.project.placements_mm.items()):
            x_mm, y_mm = analysis.clamp_into_interposer(
                self.project, name, analysis.snap(x_mm, step), analysis.snap(y_mm, step)
            )
            self.project.placements_mm[name] = analysis.round_origin(x_mm, y_mm)
        self.canvas_panel.canvas.sync_from_project()
        self.log.log(f"Snapped every origin to {step:g} mm.")
        self._mark_dirty()
        self.revalidate()

    # ------------------------------------------------------------------ power
    def _on_power_edited(self) -> None:
        self._mark_dirty()
        self._schedule_overlay()
        self.setup_panel.refresh_deviations()

    def _on_workloads_restructured(self) -> None:
        self.canvas_panel.set_workloads(list(self.project.workloads))
        self.power_panel.refresh_list()
        self.setup_panel.refresh()
        self._mark_dirty()
        self._schedule_overlay()

    def _on_workload_enabled(self) -> None:
        """A tick changed which workloads a run covers; the table itself stands."""

        self.setup_panel.refresh_run_summary()
        self._mark_dirty()

    def _on_power_selection(self, name: str) -> None:
        if name and self.canvas_panel.workload != name:
            index = self.canvas_panel.workload_box.findData(name)
            if index >= 0:
                self.canvas_panel.workload_box.setCurrentIndex(index)
        self._schedule_overlay()

    def _highlight_die(self, name: str) -> None:
        """Clicking a power map selects that chiplet on the layout canvas."""

        self.canvas_panel.canvas.select_dies([name])
        self.placement_panel.select_dies([name])

    def import_traces(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import power traces", self._start_dir(), TRACE_FILTER
        )
        if not path:
            return
        try:
            names = traces.read_traces(self.project, path)
        except Exception as exc:
            QMessageBox.critical(self, "Import power traces", str(exc))
            return
        self.power_panel.set_project(self.project)
        self.canvas_panel.set_workloads(list(self.project.workloads))
        self.log.log(f"Imported {len(names)} workload(s) from {path}.")
        self._mark_dirty()
        self.setup_panel.refresh_deviations()
        self._schedule_overlay()

    def export_traces(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export power traces",
            str(Path(self._start_dir()) / f"{self.project.name}_workloads.json"),
            TRACE_FILTER,
        )
        if not path:
            return
        try:
            traces.write_traces(self.project, path)
        except Exception as exc:
            QMessageBox.critical(self, "Export power traces", str(exc))
            return
        self.log.log(f"Wrote power traces to {path}.")

    # ------------------------------------------------------------------ setup
    def _on_setup_changed(self) -> None:
        self._mark_dirty()
        self._schedule_overlay()

    def _on_geometry_changed(self) -> None:
        """A rule change can move ports and invalidate wires, so rebuild fully."""

        self._mark_dirty()
        self.canvas_panel.canvas.rebuild()
        self.revalidate()

    # -------------------------------------------------------------------- run
    def run_simulation(self) -> None:
        if self.controller.busy:
            QMessageBox.information(self, "Run", "A simulation is already running.")
            return
        if self._report is None or not self._report.legal:
            QMessageBox.warning(
                self,
                "Run",
                "The layout is not legal yet, so it cannot be simulated:\n\n"
                f"{self._report.message if self._report else 'not validated'}",
            )
            return
        workloads = self.project.enabled_workloads()
        if not workloads:
            QMessageBox.warning(
                self, "Run", "Tick at least one workload on the Power traces tab."
            )
            return
        deviations = self.project.deviations()
        if deviations:
            answer = QMessageBox.question(
                self,
                "Setup differs from the spec",
                f"{len(deviations)} setup value(s) differ from challenge_spec.json, so the "
                "score will not be comparable with the leaderboard.\n\nRun anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        if len(workloads) < len(self.project.workloads):
            self.log.log(
                "Not every workload is ticked, so this run's worst case and score are "
                "partial and will not be leaderboard-comparable."
            )
        self.log.log(
            f"Starting {len(workloads)} workload(s) at {self.project.run.grid}x{self.project.run.grid}: "
            + ", ".join(workloads)
        )
        self.controller.start(self.project, workloads)

    def export_inputs_only(self) -> None:
        if self._report is None or not self._report.legal:
            QMessageBox.warning(self, "Export inputs", "The layout must be legal first.")
            return
        directory = QFileDialog.getExistingDirectory(
            self, "Write HotSpot inputs into", self._start_dir()
        )
        if not directory:
            return
        workloads = self.project.enabled_workloads() or list(self.project.workloads)
        self.controller.start(self.project, workloads, Path(directory), inputs_only=True)

    def _on_progress(self, fraction: float, message: str) -> None:
        self.log.progress.setValue(int(fraction * 100))
        self.log.status.setText(message)

    def _on_busy(self, busy: bool) -> None:
        self.log.progress.setVisible(busy)
        self.log.stop_button.setEnabled(busy)
        self.act_run.setEnabled(not busy)
        self.act_stop.setEnabled(busy)
        self.act_export_inputs.setEnabled(not busy)
        if not busy:
            self.log.progress.setValue(0)

    def _on_run_finished(self, result) -> None:
        self._runs.append(result)
        self._current_run = len(self._runs) - 1
        self.log.status.setText(
            f"{result.label}: worst GPU {result.worst_gpu_c:.2f} C, "
            f"worst HBM {result.worst_hbm_c:.2f} C"
        )
        self.results_panel.show_result(result)
        self.results_panel.set_history(
            self._runs, self._current_run, self.project.reference_objectives_c
        )
        self.canvas_panel.set_temperature_available(True)
        self.tabs.setCurrentWidget(self.results_panel)
        index = self.canvas_panel.overlay_box.findData("temperature")
        if index >= 0:
            self.canvas_panel.overlay_box.setCurrentIndex(index)

    def _on_inputs_exported(self, directory) -> None:
        self.log.log(f"HotSpot inputs exported to {directory}.")
        QMessageBox.information(
            self,
            "Export inputs",
            f"Wrote the floorplans, stack, config and power traces to:\n\n{directory}\n\n"
            "layout.json next to them is the --layouts file for run_simulations.py.",
        )

    def _on_run_failed(self, message: str, trace: str) -> None:
        self.log.log(f"Run failed: {message}")
        self.log.log(trace)
        self.log.status.setText("Run failed")
        QMessageBox.critical(self, "Simulation failed", message)

    def _on_run_cancelled(self) -> None:
        self.log.log("Run stopped.")
        self.log.status.setText("Stopped")

    def show_run(self, index: int) -> None:
        if not (0 <= index < len(self._runs)):
            return
        self._current_run = index
        result = self._runs[index]
        self.results_panel.show_result(result)
        self.results_panel.set_history(
            self._runs, index, self.project.reference_objectives_c
        )
        self._schedule_overlay()

    def _on_result_workload(self, name: str) -> None:
        index = self.canvas_panel.workload_box.findData(name)
        if index >= 0:
            self.canvas_panel.workload_box.setCurrentIndex(index)
        self._schedule_overlay()

    def load_previous_run(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Open a run folder (the one holding results.json)", self._start_dir()
        )
        if not directory:
            return
        try:
            result = load_run_result(directory)
        except Exception as exc:
            QMessageBox.critical(self, "Load run", f"Could not read that run:\n\n{exc}")
            return
        self._runs.append(result)
        self._current_run = len(self._runs) - 1
        self.results_panel.show_result(result)
        self.results_panel.set_history(
            self._runs, self._current_run, self.project.reference_objectives_c
        )
        self.canvas_panel.set_temperature_available(True)
        self.tabs.setCurrentWidget(self.results_panel)
        self.log.log(f"Loaded the run in {directory}.")

    # ----------------------------------------------------------------- export
    def export_results(self) -> None:
        if not (0 <= self._current_run < len(self._runs)):
            QMessageBox.information(self, "Export results", "Run a simulation first.")
            return
        result = self._runs[self._current_run]
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export results",
            str(Path(self._start_dir()) / f"{result.label}_results.json"),
            PROJECT_FILTER,
        )
        if not path:
            return
        Path(path).write_text(json.dumps(result.to_dict(), indent=2) + "\n")
        self.log.log(f"Wrote results to {path}.")

    def export_judge_inputs(self) -> None:
        if not (0 <= self._current_run < len(self._runs)):
            QMessageBox.information(self, "Export inputs", "Run a simulation first.")
            return
        directory = QFileDialog.getExistingDirectory(
            self, "Copy the judge's input set into", self._start_dir()
        )
        if not directory:
            return
        target = copy_judge_inputs(self._runs[self._current_run], directory)
        self.log.log(f"Copied the HotSpot input set to {target}.")

    def export_judge_layout(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export layout for the judge",
            str(Path(self._start_dir()) / f"{self.project.run.layout_name}_layout.json"),
            PROJECT_FILTER,
        )
        if not path:
            return
        Path(path).write_text(json.dumps(self.project.to_judge_layouts(), indent=2) + "\n")
        self.log.log(f"Wrote a --layouts file to {path}.")
        QMessageBox.information(
            self,
            "Export for judge",
            "Run it with:\n\npython run_simulations.py --hotspot <hotspot> "
            f"--layouts {Path(path).name} --grid 64 --output my_run",
        )

    def save_canvas_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save the canvas as PNG",
            str(Path(self._start_dir()) / f"{self.project.name}_layout.png"),
            "PNG image (*.png)",
        )
        if path:
            if self.canvas_panel.canvas.grab_png(path):
                self.log.log(f"Saved the canvas to {path}.")
            else:
                self.log.log(f"Could not save the canvas to {path}.")

    def open_run_folder(self) -> None:
        if not (0 <= self._current_run < len(self._runs)):
            QMessageBox.information(self, "Open folder", "Run a simulation first.")
            return
        directory = self._runs[self._current_run].run_dir
        if sys.platform.startswith("win"):
            os.startfile(str(directory))  # noqa: S606 - opening a folder for the user
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(directory)])
        else:
            subprocess.Popen(["xdg-open", str(directory)])

    # ------------------------------------------------------------- project IO
    def _start_dir(self) -> str:
        if self.project.path:
            return str(self.project.path.parent)
        return str(Path.cwd())

    def _mark_dirty(self, dirty: bool = True) -> None:
        self._dirty = dirty
        name = self.project.path.name if self.project.path else f"{self.project.name} (unsaved)"
        self.setWindowTitle(f"Chiplet Thermal Studio - {name}{' *' if dirty else ''}")

    def _confirm_discard(self) -> bool:
        if not self._dirty:
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved changes",
            "This design has unsaved changes. Save before continuing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        if answer == QMessageBox.StandardButton.Save:
            return self.save_project()
        return True

    def _adopt(self, project: Project) -> None:
        """Swap in a different design and rebind every panel to it."""

        self.project = project
        if not self.project.run.hotspot_path:
            found = find_hotspot_binary()
            if found:
                self.project.run.hotspot_path = str(found)
        self.canvas_panel.canvas.set_project(project)
        self.placement_panel.set_project(project)
        self.link_panel.set_project(project)
        self.power_panel.set_project(project)
        self.setup_panel.set_project(project)
        self.canvas_panel.set_workloads(list(project.workloads))
        self._mark_dirty(False)
        self.revalidate()

    def new_project(self) -> None:
        if not self._confirm_discard():
            return
        self._adopt(Project.default())
        self.log.log("Started a new design from the reference ring_center layout.")

    def open_project(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open design", self._start_dir(), PROJECT_FILTER
        )
        if not path:
            return
        try:
            project = Project.load(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open design", f"Could not read that file:\n\n{exc}")
            return
        self._adopt(project)
        self.log.log(f"Opened {path}.")

    def save_project(self) -> bool:
        if self.project.path is None:
            return self.save_project_as()
        try:
            self.project.save(self.project.path)
        except OSError as exc:
            QMessageBox.critical(self, "Save", str(exc))
            return False
        self._mark_dirty(False)
        self.log.log(f"Saved {self.project.path}.")
        return True

    def save_project_as(self) -> bool:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save design",
            str(Path(self._start_dir()) / f"{self.project.name}.json"),
            PROJECT_FILTER,
        )
        if not path:
            return False
        try:
            self.project.save(path)
        except OSError as exc:
            QMessageBox.critical(self, "Save", str(exc))
            return False
        self._mark_dirty(False)
        self.log.log(f"Saved {path}.")
        return True

    def import_layout(self) -> None:
        """Read a placement from any of the challenge's file shapes.

        Only the placement and wires are taken, so an imported layout can be
        compared against the traces and setup already open.
        """

        path, _ = QFileDialog.getOpenFileName(
            self, "Import layout", self._start_dir(), LAYOUT_FILTER
        )
        if not path:
            return
        try:
            imported = load_any(json.loads(Path(path).read_text()))
        except Exception as exc:
            QMessageBox.critical(self, "Import layout", f"Could not read that file:\n\n{exc}")
            return

        answer = QMessageBox.question(
            self,
            "Import layout",
            f"Read '{imported.name}' from {Path(path).name}.\n\n"
            "Take the whole design including power traces and setup?\n"
            "Choosing No imports only the placement and wires.",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            return
        if answer == QMessageBox.StandardButton.Yes:
            imported.path = None
            self._adopt(imported)
        else:
            self.project.placements_mm = imported.placements_mm
            self.project.wire_routes_grid = imported.wire_routes_grid
            self.project.run.layout_name = imported.run.layout_name
            self.canvas_panel.canvas.sync_from_project()
            self.setup_panel.refresh()
            self._mark_dirty()
            self.revalidate()
        self.log.log(f"Imported layout '{imported.name}' from {path}.")

    # ------------------------------------------------------------------ misc
    def show_about(self) -> None:
        from ..core.simcore import load_modules

        modules = load_modules()
        QMessageBox.information(
            self,
            "About Chiplet Thermal Studio",
            "A placement, routing and thermal workbench for the SUNDAI continuous "
            "chiplet-placement challenge.\n\n"
            f"Challenge package: {modules.directory}\n"
            f"HotSpot binary: {self.project.run.hotspot_path or 'not set'}\n\n"
            "Legality, port assignment, link heat, HotSpot inputs and peak extraction all "
            "come from the challenge's own modules, so the numbers match the judge.",
        )

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.controller.busy:
            answer = QMessageBox.question(
                self,
                "Simulation running",
                "A simulation is still running. Stop it and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.controller.cancel()
        if not self._confirm_discard():
            event.ignore()
            return
        event.accept()


__all__ = ["LogPanel", "MainWindow"]
