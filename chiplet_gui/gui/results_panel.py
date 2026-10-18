"""Results: peak temperatures, the power budget and the Pareto view of runs.

The challenge has two objectives, worst GPU and worst HBM temperature, and asks
for both to be kept rather than collapsed into one number. So the panel shows
the pair for every run of the session next to the scalar score.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.runner import RunResult
from . import style
from .widgets import (
    StatusLine,
    compact_table,
    heading,
    muted,
    read_only_item,
    stretch_column,
)


class ParetoPlot(QWidget):
    """Worst GPU against worst HBM temperature, one point per run.

    The reference layout sits at score 1, drawn as crosshairs: anything down
    and to the left of both lines improves both objectives at once.
    """

    pointClicked = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._runs: list[RunResult] = []
        self._current = -1
        self._reference: dict | None = None
        self._points: list[tuple[QPointF, int]] = []
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    def set_runs(self, runs: list[RunResult], current: int, reference: dict | None) -> None:
        self._runs = [run for run in runs if run.worst_gpu_c is not None]
        self._current = current
        self._reference = reference
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(27, 27, 29))
        font = QFont(painter.font())
        font.setPointSizeF(8.0)
        painter.setFont(font)

        if not self._runs:
            painter.setPen(QPen(style.MUTED_TEXT))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "Run a simulation to populate the Pareto view"
            )
            painter.end()
            return

        margin_left, margin_bottom, margin_top, margin_right = 48.0, 34.0, 12.0, 12.0
        plot = QRectF(
            margin_left,
            margin_top,
            max(10.0, self.width() - margin_left - margin_right),
            max(10.0, self.height() - margin_top - margin_bottom),
        )

        gpus = [run.worst_gpu_c for run in self._runs]
        hbms = [run.worst_hbm_c for run in self._runs]
        if self._reference:
            gpus.append(float(self._reference["worst_gpu_c"]))
            hbms.append(float(self._reference["worst_hbm_c"]))
        low_x, high_x = min(gpus), max(gpus)
        low_y, high_y = min(hbms), max(hbms)
        pad_x = max(0.5, (high_x - low_x) * 0.12)
        pad_y = max(0.5, (high_y - low_y) * 0.12)
        low_x, high_x = low_x - pad_x, high_x + pad_x
        low_y, high_y = low_y - pad_y, high_y + pad_y

        def to_pixels(gpu: float, hbm: float) -> QPointF:
            fraction_x = (gpu - low_x) / (high_x - low_x)
            fraction_y = (hbm - low_y) / (high_y - low_y)
            return QPointF(
                plot.left() + fraction_x * plot.width(),
                plot.bottom() - fraction_y * plot.height(),
            )

        painter.setPen(QPen(QColor(90, 90, 96)))
        painter.drawRect(plot)
        painter.setPen(QPen(style.MUTED_TEXT))
        painter.drawText(
            QRectF(plot.left(), plot.bottom() + 4, plot.width(), 14),
            Qt.AlignmentFlag.AlignHCenter,
            "worst GPU (C)",
        )
        painter.save()
        painter.translate(12, plot.center().y())
        painter.rotate(-90)
        painter.drawText(QRectF(-60, -10, 120, 14), Qt.AlignmentFlag.AlignHCenter, "worst HBM (C)")
        painter.restore()
        painter.drawText(
            QRectF(plot.left(), plot.bottom() + 16, 60, 14), Qt.AlignmentFlag.AlignLeft, f"{low_x:.1f}"
        )
        painter.drawText(
            QRectF(plot.right() - 60, plot.bottom() + 16, 60, 14),
            Qt.AlignmentFlag.AlignRight,
            f"{high_x:.1f}",
        )
        painter.drawText(QRectF(2, plot.top() - 2, 44, 14), Qt.AlignmentFlag.AlignRight, f"{high_y:.1f}")
        painter.drawText(
            QRectF(2, plot.bottom() - 12, 44, 14), Qt.AlignmentFlag.AlignRight, f"{low_y:.1f}"
        )

        if self._reference:
            reference = to_pixels(
                float(self._reference["worst_gpu_c"]), float(self._reference["worst_hbm_c"])
            )
            pen = QPen(QColor(150, 150, 160))
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(QPointF(reference.x(), plot.top()), QPointF(reference.x(), plot.bottom()))
            painter.drawLine(QPointF(plot.left(), reference.y()), QPointF(plot.right(), reference.y()))
            painter.setPen(QPen(QColor(200, 200, 210)))
            painter.drawText(
                QRectF(reference.x() + 4, reference.y() - 16, 90, 14),
                Qt.AlignmentFlag.AlignLeft,
                "reference",
            )

        self._points = []
        for index, run in enumerate(self._runs):
            point = to_pixels(float(run.worst_gpu_c), float(run.worst_hbm_c))
            self._points.append((point, index))
            is_current = index == self._current
            colour = QColor(129, 212, 162) if run.comparable else QColor(255, 183, 77)
            painter.setBrush(colour)
            painter.setPen(QPen(QColor(255, 255, 255) if is_current else colour.darker(140), 2))
            radius = 6.0 if is_current else 4.0
            painter.drawEllipse(point, radius, radius)
            painter.setPen(QPen(QColor(215, 215, 220)))
            painter.drawText(QRectF(point.x() + 7, point.y() - 8, 140, 14), Qt.AlignmentFlag.AlignLeft, run.label)
        painter.end()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        for point, index in self._points:
            if (event.position() - point).manhattanLength() < 10:
                run = self._runs[index]
                self.setToolTip(
                    f"{run.label}\nworst GPU {run.worst_gpu_c:.2f} C\n"
                    f"worst HBM {run.worst_hbm_c:.2f} C\n"
                    + (f"score {run.score:.4f}" if run.score is not None else "")
                )
                return
        self.setToolTip("")

    def mousePressEvent(self, event) -> None:  # noqa: N802
        for point, index in self._points:
            if (event.position() - point).manhattanLength() < 12:
                self.pointClicked.emit(index)
                return


class ResultsPanel(QWidget):
    """Objectives, per-workload peaks, per-die peaks and the run history."""

    workloadSelected = Signal(str)
    runSelected = Signal(int)
    exportResultsRequested = Signal()
    exportInputsRequested = Signal()
    openFolderRequested = Signal()
    loadRunRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: RunResult | None = None
        self._runs: list[RunResult] = []

        self.headline = heading("No results yet")
        self.objective_label = QLabel("")
        self.objective_label.setWordWrap(True)
        self.comparability = StatusLine()

        self.workload_table = compact_table(
            ("Workload", "GPU peak", "HBM peak", "Base W", "Total W", "PHY W", "Wire W", "Solve s")
        )
        self.workload_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.workload_table.itemSelectionChanged.connect(self._emit_workload)
        stretch_column(self.workload_table, 0)

        self.die_table = compact_table(("Die", "Kind", "Peak C"))
        self.die_table.setMaximumWidth(260)
        self.die_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )

        self.history_table = compact_table(("Run", "Grid", "GPU", "HBM", "Score", "Wall s"))
        self.history_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.history_table.itemSelectionChanged.connect(self._emit_run)
        stretch_column(self.history_table, 0)

        self.pareto = ParetoPlot()
        self.pareto.pointClicked.connect(self.runSelected.emit)

        export_results = QPushButton("Export results JSON...")
        export_results.clicked.connect(self.exportResultsRequested.emit)
        export_inputs = QPushButton("Export HotSpot inputs...")
        export_inputs.setToolTip("Copy the floorplans, traces and layout the judge would read")
        export_inputs.clicked.connect(self.exportInputsRequested.emit)
        open_folder = QPushButton("Open run folder")
        open_folder.clicked.connect(self.openFolderRequested.emit)
        load_run = QPushButton("Load a previous run...")
        load_run.clicked.connect(self.loadRunRequested.emit)

        buttons = QHBoxLayout()
        buttons.addWidget(export_results)
        buttons.addWidget(export_inputs)
        buttons.addWidget(open_folder)
        buttons.addWidget(load_run)
        buttons.addStretch(1)

        summary = QGroupBox("Objectives")
        summary_layout = QVBoxLayout(summary)
        summary_layout.addWidget(self.headline)
        summary_layout.addWidget(self.objective_label)
        summary_layout.addWidget(self.comparability)

        per_workload = QGroupBox("Per workload")
        per_workload_layout = QHBoxLayout(per_workload)
        per_workload_layout.addWidget(self.workload_table, 1)
        per_workload_layout.addWidget(self.die_table)

        history = QGroupBox("Runs this session")
        history_layout = QHBoxLayout(history)
        history_layout.addWidget(self.history_table, 1)
        history_layout.addWidget(self.pareto, 1)

        splitter = QSplitter(Qt.Orientation.Vertical)
        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(summary)
        top_layout.addWidget(per_workload, 1)
        splitter.addWidget(top)
        splitter.addWidget(history)
        splitter.setSizes([320, 260])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(splitter, 1)
        layout.addLayout(buttons)
        layout.addWidget(
            muted(
                "Peaks are the hottest die-plane grid cell over the die footprint, the same "
                "measure the judge scores.",
                wrap=True,
            )
        )

    # ----------------------------------------------------------------- update
    def show_result(self, result: RunResult | None) -> None:
        self._result = result
        if result is None:
            self.headline.setText("No results yet")
            self.objective_label.setText("")
            self.comparability.set_message("", "muted")
            self.workload_table.setRowCount(0)
            self.die_table.setRowCount(0)
            return

        score = result.score
        self.headline.setText(
            f"{result.label}: worst GPU {result.worst_gpu_c:.2f} C, "
            f"worst HBM {result.worst_hbm_c:.2f} C"
            + (f", score {score:.4f}" if score is not None else "")
        )
        reference = result.project.reference_objectives_c or {}
        parts = [
            f"{result.grid} x {result.grid} steady-state grid",
            f"{len(result.workloads)} workload(s) in {result.seconds:.1f} s",
            f"weighted route {result.objectives.get('weighted_route_gb_s_mm', 0):.1f} GB/s*mm",
        ]
        if reference:
            parts.append(
                f"reference GPU {float(reference['worst_gpu_c']):.2f} C / "
                f"HBM {float(reference['worst_hbm_c']):.2f} C"
            )
        if not result.interconnect_enabled:
            parts.append("link heat disabled (A/B control)")
        self.objective_label.setText("   |   ".join(parts))

        if result.comparable:
            verdict = "Setup matches the spec, so this score is leaderboard-comparable."
            role = "ok"
            if score is not None and score < 1.0:
                verdict = (
                    f"Score {score:.4f} beats the reference on both objectives. "
                    "Setup matches the spec."
                )
        else:
            reasons = []
            if result.deviations:
                reasons.append(f"{len(result.deviations)} setup value(s) differ from the spec")
            if not result.interconnect_enabled:
                reasons.append("link heat was disabled")
            if result.skipped_workloads:
                reasons.append(
                    "only some workloads were run (skipped "
                    + ", ".join(result.skipped_workloads)
                    + ")"
                )
            verdict = "Not leaderboard-comparable: " + "; ".join(reasons) + "."
            role = "warn"
        self.comparability.set_message(verdict, role)

        names = list(result.workloads)
        self.workload_table.setRowCount(len(names))
        for row, name in enumerate(names):
            entry = result.workloads[name]
            self.workload_table.setItem(row, 0, read_only_item(name))
            self.workload_table.setItem(row, 1, read_only_item(f"{entry.gpu_peak_c:.2f}", True))
            self.workload_table.setItem(row, 2, read_only_item(f"{entry.hbm_peak_c:.2f}", True))
            self.workload_table.setItem(row, 3, read_only_item(f"{entry.base_power_w:.1f}", True))
            self.workload_table.setItem(row, 4, read_only_item(f"{entry.total_power_w:.1f}", True))
            self.workload_table.setItem(
                row, 5, read_only_item(f"{entry.interconnect_phy_w:.2f}", True)
            )
            self.workload_table.setItem(
                row, 6, read_only_item(f"{entry.interconnect_wire_w:.2f}", True)
            )
            self.workload_table.setItem(row, 7, read_only_item(f"{entry.seconds:.0f}", True))

        worst = max(names, key=lambda n: result.workloads[n].gpu_peak_c, default=None)
        if worst:
            self._select_workload_row(worst)

    def _select_workload_row(self, name: str) -> None:
        for row in range(self.workload_table.rowCount()):
            item = self.workload_table.item(row, 0)
            if item is not None and item.text() == name:
                self.workload_table.selectRow(row)
                return

    def show_die_peaks(self, workload: str) -> None:
        result = self._result
        if result is None or workload not in result.workloads:
            self.die_table.setRowCount(0)
            return
        peaks = result.workloads[workload].die_peak_c
        names = sorted(peaks)
        self.die_table.setRowCount(len(names))
        hottest = max(peaks.values()) if peaks else 0.0
        for row, name in enumerate(names):
            die = result.project.dies.get(name)
            kind = die.kind if die else ""
            item = read_only_item(name)
            item.setForeground(style.GPU_EDGE if kind == "GPU" else style.HBM_EDGE)
            self.die_table.setItem(row, 0, item)
            self.die_table.setItem(row, 1, read_only_item(kind))
            value = read_only_item(f"{peaks[name]:.2f}", True)
            if abs(peaks[name] - hottest) < 1e-9:
                value.setForeground(style.ERROR_COLOR)
            self.die_table.setItem(row, 2, value)
        self.die_table.resizeColumnsToContents()

    def set_history(self, runs: list[RunResult], current: int, reference: dict | None) -> None:
        self._runs = runs
        self.history_table.blockSignals(True)
        try:
            self.history_table.setRowCount(len(runs))
            for row, run in enumerate(runs):
                self.history_table.setItem(row, 0, read_only_item(run.label))
                self.history_table.setItem(row, 1, read_only_item(str(run.grid), True))
                self.history_table.setItem(
                    row, 2, read_only_item(f"{run.worst_gpu_c:.2f}" if run.worst_gpu_c else "-", True)
                )
                self.history_table.setItem(
                    row, 3, read_only_item(f"{run.worst_hbm_c:.2f}" if run.worst_hbm_c else "-", True)
                )
                score = read_only_item(f"{run.score:.4f}" if run.score is not None else "-", True)
                if run.score is not None:
                    score.setForeground(
                        style.OK_COLOR if run.score < 1.0 else style.WARN_COLOR
                    )
                if not run.comparable:
                    score.setToolTip("Not leaderboard-comparable; see the run's verdict for why.")
                self.history_table.setItem(row, 4, score)
                self.history_table.setItem(row, 5, read_only_item(f"{run.seconds:.0f}", True))
            if 0 <= current < len(runs):
                self.history_table.selectRow(current)
        finally:
            self.history_table.blockSignals(False)
        self.pareto.set_runs(runs, current, reference)

    # ---------------------------------------------------------------- signals
    def _emit_workload(self) -> None:
        rows = {index.row() for index in self.workload_table.selectedIndexes()}
        if not rows:
            return
        item = self.workload_table.item(min(rows), 0)
        if item is None:
            return
        self.show_die_peaks(item.text())
        self.workloadSelected.emit(item.text())

    def _emit_run(self) -> None:
        rows = {index.row() for index in self.history_table.selectedIndexes()}
        if rows:
            self.runSelected.emit(min(rows))

    def selected_workload(self) -> str | None:
        rows = {index.row() for index in self.workload_table.selectedIndexes()}
        if not rows:
            return None
        item = self.workload_table.item(min(rows), 0)
        return item.text() if item else None


__all__ = ["ParetoPlot", "ResultsPanel"]
