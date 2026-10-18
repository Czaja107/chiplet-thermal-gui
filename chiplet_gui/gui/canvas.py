"""The layout canvas: drag the chiplets, draw the wires, see the heat.

Scene units are millimetres with y measured down from the top of the package,
so a die at mm origin ``(x0, y0)`` sits at scene ``(x0, H - y1)``. The two
conversions live in :meth:`LayoutCanvas.scene_y` and are the only place the
flip happens.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsItemGroup,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from ..core import analysis
from ..core.analysis import Cell, LayoutReport, RouteGrid
from ..core.project import DieDef, Project
from ..core.simcore import load_modules
from . import style
from .render import Field

#: Margin of empty scene around the package, in mm.
SCENE_MARGIN = 6.0

MODE_SELECT = "select"
MODE_ROUTE = "route"


class DieItem(QGraphicsRectItem):
    """One draggable chiplet.

    Snapping and the edge-clearance clamp happen in :meth:`itemChange`, so a
    die cannot be dragged off the interposer and always lands on the chosen
    grid step no matter how the drag is performed.
    """

    def __init__(self, canvas: LayoutCanvas, die: DieDef) -> None:
        width, height = die.size_mm
        super().__init__(QRectF(0.0, 0.0, float(width), float(height)))
        self.canvas = canvas
        self.die_name = die.name
        self.kind = die.kind
        self._annotation = ""
        self._invalid = False

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(10.0)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self._apply_colors()

    # ------------------------------------------------------------ appearance
    def _apply_colors(self) -> None:
        fill, edge = style.die_colors(self.kind)
        if self._invalid:
            fill, edge = style.INVALID_FILL, style.INVALID_EDGE
        self.setBrush(QBrush(QColor(fill.red(), fill.green(), fill.blue(), 205)))
        pen = QPen(style.SELECTED_EDGE if self.isSelected() else edge)
        pen.setCosmetic(True)  # constant width on screen, whatever the zoom
        pen.setWidth(3 if self.isSelected() else 2)
        self.setPen(pen)

    def set_invalid(self, invalid: bool) -> None:
        if invalid != self._invalid:
            self._invalid = invalid
            self._apply_colors()

    def set_annotation(self, text: str) -> None:
        """A second label line, used for peak temperature and watts."""

        if text != self._annotation:
            self._annotation = text
            self.update()

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: N802
        super().paint(painter, option, widget)
        label = self.die_name
        if not self._annotation:
            self._draw_text(painter, label, 0.0)
            return
        self._draw_text(painter, label, -0.9)
        self._draw_text(painter, self._annotation, 1.0, small=True)

    def _draw_text(self, painter: QPainter, text: str, offset_mm: float, small: bool = False) -> None:
        """Draw centred text at a constant on-screen size.

        The painter transform is reset so the font does not grow with zoom,
        then the item centre is mapped into device coordinates by hand.
        """

        painter.save()
        transform = painter.transform()
        centre = transform.map(
            QPointF(self.rect().center().x(), self.rect().center().y() + offset_mm)
        )
        painter.resetTransform()
        font = QFont(painter.font())
        font.setPointSizeF(8.0 if small else 10.5)
        font.setBold(not small)
        painter.setFont(font)
        painter.setPen(QPen(style.MUTED_TEXT if small else QColor(248, 248, 250)))
        metrics = painter.fontMetrics()
        painter.drawText(
            QPointF(
                centre.x() - metrics.horizontalAdvance(text) / 2.0,
                centre.y() + metrics.ascent() / 2.0,
            ),
            text,
        )
        painter.restore()

    # -------------------------------------------------------------- behaviour
    def itemChange(self, change, value):  # noqa: N802 - Qt naming
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and not self.canvas.loading:
            return self.canvas.constrain_position(self.die_name, value)
        result = super().itemChange(change, value)
        # ItemSelectedChange fires before isSelected() flips; recolour after it.
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._apply_colors()
        return result

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        super().mouseMoveEvent(event)
        self.canvas.on_die_dragged(self)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().mouseReleaseEvent(event)
        self.canvas.on_drag_finished()


class LayoutCanvas(QGraphicsView):
    """Interactive floorplan: placement, ports, routed wires and field overlays."""

    placementsChanged = Signal()
    placementsCommitted = Signal()
    dieSelected = Signal(list)
    routeCommitted = Signal(str, list)
    routeCancelled = Signal()
    hovered = Signal(str)
    draftChanged = Signal(str)

    def __init__(self, project: Project, parent=None) -> None:
        super().__init__(parent)
        self._project = project
        self._report: LayoutReport | None = None
        self._grid: RouteGrid | None = None
        self._field: Field | None = None
        self._mode = MODE_SELECT

        self.loading = False
        self.snap_mm = 0.1
        self.show_ports = True
        self.show_routes = True
        self.show_track_grid = False
        self.show_blocked = True
        self.show_annotations = True

        self._route_link: str | None = None
        self._draft: list[Cell] = []
        self._hover_cell: Cell | None = None
        self._blocked_image: QImage | None = None
        self._pan_from = None

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QBrush(QColor(18, 18, 20)))
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._dies: dict[str, DieItem] = {}
        self._decor = QGraphicsItemGroup()
        self._decor.setZValue(20.0)
        self._scene.addItem(self._decor)
        self._draft_item = QGraphicsPathItem()
        self._draft_item.setZValue(30.0)
        self._scene.addItem(self._draft_item)

        self._scene.selectionChanged.connect(self._emit_selection)
        self.rebuild()

    # ------------------------------------------------------------- geometry
    @property
    def project(self) -> Project:
        return self._project

    @property
    def width_mm(self) -> float:
        return float(self._project.interposer_mm[0])

    @property
    def height_mm(self) -> float:
        return float(self._project.interposer_mm[1])

    def scene_y(self, y_mm: float) -> float:
        """Convert between mm-up and scene-down y. The conversion is its own inverse."""

        return self.height_mm - float(y_mm)

    def scene_point(self, x_mm: float, y_mm: float) -> QPointF:
        return QPointF(float(x_mm), self.scene_y(y_mm))

    def mm_at(self, scene_position: QPointF) -> tuple[float, float]:
        return float(scene_position.x()), self.scene_y(scene_position.y())

    def die_origin_from_pos(self, name: str, position: QPointF) -> tuple[float, float]:
        """mm origin (lower-left) of a die whose item sits at *position*."""

        _, height = self._project.dies[name].size_mm
        return float(position.x()), self.scene_y(float(position.y()) + float(height))

    def pos_from_die_origin(self, name: str, x_mm: float, y_mm: float) -> QPointF:
        _, height = self._project.dies[name].size_mm
        return QPointF(float(x_mm), self.scene_y(float(y_mm) + float(height)))

    # -------------------------------------------------------------- building
    def set_project(self, project: Project) -> None:
        self._project = project
        self.cancel_route()
        self.rebuild()

    def rebuild(self) -> None:
        """Recreate every item; used when the project or die sizes change."""

        self.loading = True
        try:
            for item in list(self._dies.values()):
                self._scene.removeItem(item)
            self._dies.clear()
            margin = SCENE_MARGIN
            self._scene.setSceneRect(
                QRectF(-margin, -margin, self.width_mm + 2 * margin, self.height_mm + 2 * margin)
            )
            for name in sorted(self._project.placements_mm):
                die = self._project.dies[name]
                item = DieItem(self, die)
                x0, y0, _, _ = self._project.die_box(name)
                item.setPos(self.pos_from_die_origin(name, x0, y0))
                item.setToolTip(self._die_tooltip(name))
                self._scene.addItem(item)
                self._dies[name] = item
        finally:
            self.loading = False
        self.fit()

    def sync_from_project(self) -> None:
        """Move the items to match the project, e.g. after an import or undo."""

        if set(self._dies) != set(self._project.placements_mm):
            self.rebuild()
            return
        self.loading = True
        try:
            for name, item in self._dies.items():
                x0, y0, _, _ = self._project.die_box(name)
                item.setPos(self.pos_from_die_origin(name, x0, y0))
                item.setToolTip(self._die_tooltip(name))
        finally:
            self.loading = False

    def _die_tooltip(self, name: str) -> str:
        die = self._project.dies[name]
        x0, y0, x1, y1 = self._project.die_box(name)
        return (
            f"{name} ({die.kind})\n"
            f"origin  {x0:.3f}, {y0:.3f} mm\n"
            f"extent  {x1:.3f}, {y1:.3f} mm\n"
            f"size    {die.size_mm[0]:g} x {die.size_mm[1]:g} mm"
        )

    # ------------------------------------------------------------ constraints
    def constrain_position(self, name: str, position: QPointF) -> QPointF:
        """Snap to the grid step and clamp inside the edge-clearance window."""

        x_mm, y_mm = self.die_origin_from_pos(name, position)
        x_mm = analysis.snap(x_mm, self.snap_mm)
        y_mm = analysis.snap(y_mm, self.snap_mm)
        x_mm, y_mm = analysis.clamp_into_interposer(self._project, name, x_mm, y_mm)
        return self.pos_from_die_origin(name, x_mm, y_mm)

    def on_die_dragged(self, item: DieItem) -> None:
        """Write every moved die back into the project and report the change."""

        moved = False
        for name, die_item in self._dies.items():
            origin = analysis.round_origin(*self.die_origin_from_pos(name, die_item.pos()))
            if origin != list(self._project.placements_mm[name]):
                self._project.placements_mm[name] = origin
                die_item.setToolTip(self._die_tooltip(name))
                moved = True
        if moved:
            self.placementsChanged.emit()

    def on_drag_finished(self) -> None:
        self.placementsCommitted.emit()

    def nudge_selection(self, dx_mm: float, dy_mm: float) -> None:
        """Move the selected dies by an exact amount, for keyboard fine-tuning."""

        names = self.selected_dies()
        if not names:
            return
        self.loading = True
        try:
            for name in names:
                x0, y0, _, _ = self._project.die_box(name)
                x_mm, y_mm = analysis.clamp_into_interposer(
                    self._project,
                    name,
                    analysis.snap(x0 + dx_mm, self.snap_mm),
                    analysis.snap(y0 + dy_mm, self.snap_mm),
                )
                self._project.placements_mm[name] = analysis.round_origin(x_mm, y_mm)
                self._dies[name].setPos(self.pos_from_die_origin(name, x_mm, y_mm))
                self._dies[name].setToolTip(self._die_tooltip(name))
        finally:
            self.loading = False
        self.placementsChanged.emit()
        self.placementsCommitted.emit()

    # ------------------------------------------------------------- selection
    def selected_dies(self) -> list[str]:
        return sorted(
            item.die_name for item in self._scene.selectedItems() if isinstance(item, DieItem)
        )

    def select_dies(self, names: Iterable[str]) -> None:
        wanted = set(names)
        for name, item in self._dies.items():
            item.setSelected(name in wanted)

    def _emit_selection(self) -> None:
        self.dieSelected.emit(self.selected_dies())

    # ------------------------------------------------------------- overlays
    def set_field(self, field: Field | None) -> None:
        self._field = field
        self.viewport().update()

    def set_report(self, report: LayoutReport | None, grid: RouteGrid | None = None) -> None:
        """Take a fresh validation result and redraw ports, routes and faults."""

        self._report = report
        self._grid = grid
        errors = report.die_errors if report else {}
        for name, item in self._dies.items():
            item.set_invalid(name in errors)
        self._blocked_image = self._build_blocked_image(grid)
        self._rebuild_decorations()
        self._update_draft_item()
        self.viewport().update()

    def set_die_annotations(self, annotations: dict[str, str]) -> None:
        for name, item in self._dies.items():
            item.set_annotation(annotations.get(name, "") if self.show_annotations else "")

    def _build_blocked_image(self, grid: RouteGrid | None) -> QImage | None:
        """A translucent red wash over track cells no wire may use."""

        if grid is None:
            return None
        blocked = np.asarray(grid.blocked, dtype=bool)
        rgba = np.zeros((blocked.shape[0], blocked.shape[1], 4), dtype=np.uint8)
        rgba[blocked] = (235, 90, 95, 46)
        picture = np.ascontiguousarray(np.flipud(rgba))
        image = QImage(
            picture.data,
            picture.shape[1],
            picture.shape[0],
            4 * picture.shape[1],
            QImage.Format.Format_RGBA8888,
        )
        return image.copy()

    # ----------------------------------------------------------- decorations
    def _rebuild_decorations(self) -> None:
        for child in self._decor.childItems():
            self._decor.removeFromGroup(child)
            self._scene.removeItem(child)

        report = self._report
        if report is None or not report.routing:
            return

        pitch = float(self._project.port_rule["route_grid_pitch_mm"])
        sim = load_modules()
        explicit = set(self._project.wire_routes_grid)

        for link_id, route in report.routing["routes"].items():
            color = style.link_color(link_id)
            if self.show_ports:
                for port, normal in (
                    (route["from_port"], route["from_normal"]),
                    (route["to_port"], route["to_normal"]),
                ):
                    self._add_port_marker(port, normal, float(route["required_port_mm"]), color)
            if not self.show_routes or not route["path_grid"]:
                continue
            path = QPainterPath()
            for start, end in sim.interconnect.route_segments(route, pitch):
                path.moveTo(self.scene_point(*start))
                path.lineTo(self.scene_point(*end))
            item = QGraphicsPathItem(path)
            pen = QPen(color)
            pen.setCosmetic(True)
            pen.setWidth(3 if link_id in explicit else 2)
            if link_id not in explicit:
                pen.setStyle(Qt.PenStyle.DashLine)
            item.setPen(pen)
            item.setToolTip(
                f"{link_id}: {route['from']} to {route['to']}\n"
                f"{route['bandwidth_gb_s']:g} GB/s, routed {route['route_length_mm']:.2f} mm "
                f"of {route['limit_mm']:.2f} mm\n"
                + ("hand-drawn route" if link_id in explicit else "automatic shortest route")
            )
            self._decor.addToGroup(item)

    def _add_port_marker(
        self, port: Sequence[float], normal: Sequence[float], length: float, color: QColor
    ) -> None:
        """The reserved edge segment, drawn along the die edge it occupies."""

        x, y = float(port[0]), float(port[1])
        path = QPainterPath()
        if float(normal[0]):
            path.moveTo(self.scene_point(x, y - length / 2))
            path.lineTo(self.scene_point(x, y + length / 2))
        else:
            path.moveTo(self.scene_point(x - length / 2, y))
            path.lineTo(self.scene_point(x + length / 2, y))
        item = QGraphicsPathItem(path)
        pen = QPen(color)
        pen.setCosmetic(True)
        pen.setWidth(5)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        item.setPen(pen)
        item.setOpacity(0.9)
        self._decor.addToGroup(item)

    # ----------------------------------------------------------- wire editing
    @property
    def mode(self) -> str:
        return self._mode

    @property
    def route_link(self) -> str | None:
        return self._route_link

    @property
    def draft(self) -> list[Cell]:
        return list(self._draft)

    def begin_route(self, link_id: str) -> None:
        """Enter wire-drawing mode for one link, seeded with its start cell."""

        self._mode = MODE_ROUTE
        self._route_link = link_id
        existing = self._project.wire_routes_grid.get(link_id)
        if existing:
            self._draft = [(int(i), int(j)) for i, j in existing]
        elif self._grid and link_id in self._grid.endpoints:
            self._draft = [self._grid.endpoints[link_id][0]]
        else:
            self._draft = []
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        for item in self._dies.values():
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self._update_draft_item()
        self.viewport().update()

    def cancel_route(self) -> None:
        if self._mode == MODE_SELECT and not self._draft:
            return
        self._mode = MODE_SELECT
        self._route_link = None
        self._draft = []
        self._hover_cell = None
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        for item in self._dies.values():
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self._update_draft_item()
        self.viewport().update()
        self.routeCancelled.emit()

    def extend_draft_to(self, cell: Cell) -> None:
        """Add a waypoint: clicking far away fills in the Manhattan path to it.

        Clicking a cell already on the draft truncates back to it, which is the
        quickest way to undo a wrong turn.
        """

        grid = self._grid
        if grid is None or self._route_link is None:
            return
        if not grid.is_free(cell):
            self.draftChanged.emit("That track cell is under a die or inside its clearance.")
            return
        if not self._draft:
            self._draft = [cell]
        elif cell in self._draft:
            self._draft = self._draft[: self._draft.index(cell) + 1]
        else:
            bridge = self._bridge(grid, self._draft[-1], cell)
            if bridge is None:
                self.draftChanged.emit("No obstacle-free path to that cell.")
                return
            self._append_cells(bridge[1:])
        self._update_draft_item()
        self._report_draft()

    def _bridge(self, grid: RouteGrid, start: Cell, goal: Cell) -> list[Cell] | None:
        """Join two cells, keeping clear of the draft so a detour is preserved.

        If going around is impossible, fall back to the plain shortest path and
        let :meth:`_append_cells` fold the overlap, so the user is never stuck
        with a wire that cannot be finished.
        """

        avoid = set(self._draft[:-1])
        return analysis.shortest_path(grid, start, goal, avoid=avoid) or analysis.shortest_path(
            grid, start, goal
        )

    def _append_cells(self, cells: Sequence[Cell]) -> None:
        """Append cells, folding back on themselves instead of repeating.

        A bridge computed to a new waypoint can run back over cells the draft
        already holds. Truncating to the first repeat keeps the path simple,
        which the judge requires: it refuses a route that visits a cell twice.
        """

        for step in cells:
            if step in self._draft:
                self._draft = self._draft[: self._draft.index(step) + 1]
            else:
                self._draft.append(step)

    def undo_draft_cell(self) -> None:
        if len(self._draft) > 1:
            self._draft.pop()
            self._update_draft_item()
            self._report_draft()

    def complete_draft(self) -> bool:
        """Join the draft to the far port and commit it if the judge would accept it."""

        grid = self._grid
        link_id = self._route_link
        if grid is None or link_id is None or link_id not in grid.endpoints:
            self.draftChanged.emit("This link has no usable port access at the moment.")
            return False
        start, goal = grid.endpoints[link_id]
        if not self._draft:
            self._draft = [start]
        if self._draft[0] != start:
            self._draft.insert(0, start)
        if self._draft[-1] != goal:
            bridge = self._bridge(grid, self._draft[-1], goal)
            if bridge is None:
                self.draftChanged.emit("No obstacle-free path from the draft to the far port.")
                return False
            self._append_cells(bridge[1:])

        limit = float(self._project.link(link_id)["max_route_mm"])
        problems = analysis.path_problems(grid, link_id, self._draft, limit)
        if not problems:
            problems = self._budget_problems(link_id)
        if problems:
            self.draftChanged.emit("; ".join(problems))
            return False
        path = [[int(i), int(j)] for i, j in self._draft]
        self.cancel_route()
        self.routeCommitted.emit(link_id, path)
        return True

    def _budget_problems(self, link_id: str) -> list[str]:
        """Whether this draft would break the shared weighted-route budget."""

        if self._report is None or self._grid is None:
            return []
        length = self._grid.path_length_mm(link_id, self._draft)
        if length is None:
            return []
        projected, limit = analysis.projected_weighted_route(self._report, link_id, length)
        if limit and projected > limit + 1e-8:
            return [
                f"weighted route would reach {projected:.0f} of the {limit:.0f} GB/s*mm budget "
                f"shared by all links"
            ]
        return []

    def _report_draft(self) -> None:
        grid, link_id = self._grid, self._route_link
        if grid is None or link_id is None:
            return
        length = grid.path_length_mm(link_id, self._draft)
        limit = float(self._project.link(link_id)["max_route_mm"])
        goal = grid.endpoints.get(link_id, (None, None))[1]
        pieces = [f"{link_id}: {len(self._draft)} cells"]
        if length is not None:
            pieces.append(f"{length:.2f} mm of {limit:.2f} mm reach")
            if self._report is not None:
                projected, budget = analysis.projected_weighted_route(
                    self._report, link_id, length
                )
                if budget:
                    over = " -- over budget" if projected > budget + 1e-8 else ""
                    pieces.append(f"weighted {projected:.0f}/{budget:.0f} GB/s*mm{over}")
        if goal is not None and self._draft and self._draft[-1] != goal:
            pieces.append(f"not yet at the far port {list(goal)}")
        self.draftChanged.emit(", ".join(pieces))

    def _update_draft_item(self) -> None:
        if self._mode != MODE_ROUTE or not self._draft or self._grid is None:
            self._draft_item.setPath(QPainterPath())
            return
        grid = self._grid
        path = QPainterPath()
        points = [QPointF(*self._cell_center_scene(cell, grid)) for cell in self._draft]
        path.moveTo(points[0])
        for point in points[1:]:
            path.lineTo(point)
        for point in points:
            path.addEllipse(QRectF(point.x() - 0.14, point.y() - 0.14, 0.28, 0.28))
        self._draft_item.setPath(path)
        color = style.link_color(self._route_link or "P0")
        pen = QPen(QColor(255, 255, 255))
        pen.setCosmetic(True)
        pen.setWidth(4)
        self._draft_item.setPen(pen)
        self._draft_item.setBrush(QBrush(color))

    def _cell_center_scene(self, cell: Cell, grid: RouteGrid) -> tuple[float, float]:
        x_mm, y_mm = grid.center_mm(*cell)
        return x_mm, self.scene_y(y_mm)

    # ---------------------------------------------------------------- events
    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        super().drawBackground(painter, rect)
        package = QRectF(0.0, 0.0, self.width_mm, self.height_mm)

        painter.setBrush(QBrush(style.INTERPOSER_FILL))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(package)

        if self._field is not None:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self._field.smooth)
            painter.drawImage(package, self._field.image)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)

        if self._mode == MODE_ROUTE:
            if self.show_blocked and self._blocked_image is not None:
                painter.drawImage(package, self._blocked_image)
            self._draw_track_grid(painter, rect, force=True)
            self._draw_route_targets(painter)
        elif self.show_track_grid:
            self._draw_track_grid(painter, rect, force=False)

        edge = float(self._project.edge_clearance_mm)
        pen = QPen(style.CLEARANCE_EDGE)
        pen.setCosmetic(True)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(
            QRectF(edge, edge, max(0.0, self.width_mm - 2 * edge), max(0.0, self.height_mm - 2 * edge))
        )

        contact = self._project.thermal_model.get("cooling_contact_rect_mm")
        if contact and len(contact) == 4:
            x, y, width, height = (float(v) for v in contact)
            pen = QPen(style.COOLING_EDGE)
            pen.setCosmetic(True)
            pen.setStyle(Qt.PenStyle.DashDotLine)
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(QRectF(x, self.scene_y(y + height), width, height))

        pen = QPen(style.INTERPOSER_EDGE)
        pen.setCosmetic(True)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(package)

    def _draw_track_grid(self, painter: QPainter, rect: QRectF, force: bool) -> None:
        pitch = float(self._project.port_rule["route_grid_pitch_mm"])
        # One line per pitch is noise below roughly four device pixels per cell.
        if not force and self.transform().m11() * pitch < 4.0:
            return
        pen = QPen(style.TRACK_GRID)
        pen.setCosmetic(True)
        painter.setPen(pen)
        left = max(0.0, rect.left())
        right = min(self.width_mm, rect.right())
        top = max(0.0, rect.top())
        bottom = min(self.height_mm, rect.bottom())
        if right <= left or bottom <= top:
            return
        start = int(left / pitch)
        while start * pitch <= right:
            x = start * pitch
            painter.drawLine(QPointF(x, top), QPointF(x, bottom))
            start += 1
        start = int(top / pitch)
        while start * pitch <= bottom:
            y = start * pitch
            painter.drawLine(QPointF(left, y), QPointF(right, y))
            start += 1

    def _draw_route_targets(self, painter: QPainter) -> None:
        """Mark the two cells the active link's wire must connect."""

        grid, link_id = self._grid, self._route_link
        if grid is None or link_id is None or link_id not in grid.endpoints:
            return
        pitch = grid.pitch
        start, goal = grid.endpoints[link_id]
        for cell, color in ((start, QColor(129, 212, 162)), (goal, QColor(255, 183, 77))):
            x_mm, y_mm = grid.center_mm(*cell)
            pen = QPen(color)
            pen.setCosmetic(True)
            pen.setWidth(3)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(
                QRectF(x_mm - pitch / 2, self.scene_y(y_mm) - pitch / 2, pitch, pitch)
            )
        if self._hover_cell is not None:
            x_mm, y_mm = grid.center_mm(*self._hover_cell)
            free = grid.is_free(self._hover_cell)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(
                QBrush(QColor(255, 255, 255, 70) if free else QColor(239, 108, 112, 90))
            )
            painter.drawRect(
                QRectF(x_mm - pitch / 2, self.scene_y(y_mm) - pitch / 2, pitch, pitch)
            )

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_from = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if self._mode == MODE_ROUTE and self._grid is not None:
            position = self.mapToScene(event.position().toPoint())
            cell = self._grid.cell_at(*self.mm_at(position))
            if event.button() == Qt.MouseButton.LeftButton:
                self.extend_draft_to(cell)
            elif event.button() == Qt.MouseButton.RightButton:
                self.undo_draft_cell()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if self._mode == MODE_ROUTE:
            self.complete_draft()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._pan_from is not None:
            delta = event.position().toPoint() - self._pan_from
            self._pan_from = event.position().toPoint()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        position = self.mapToScene(event.position().toPoint())
        x_mm, y_mm = self.mm_at(position)
        message = f"x {x_mm:8.3f} mm    y {y_mm:8.3f} mm"
        if self._grid is not None:
            cell = self._grid.cell_at(x_mm, y_mm)
            if self._mode == MODE_ROUTE and cell != self._hover_cell:
                self._hover_cell = cell
                self.viewport().update()
            message += f"    track cell [{cell[0]}, {cell[1]}]"
        self.hovered.emit(message)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.MiddleButton and self._pan_from is not None:
            self._pan_from = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        steps = event.angleDelta().y() / 120.0
        if not steps:
            return
        factor = 1.18**steps
        current = self.transform().m11()
        target = current * factor
        # Keep the package between "fits the viewport" and "half a mm per screen".
        if target < 2.0 or target > 400.0:
            return
        self.scale(factor, factor)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        if self._mode == MODE_ROUTE:
            if key == Qt.Key.Key_Escape:
                self.cancel_route()
                return
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.complete_draft()
                return
            if key == Qt.Key.Key_Backspace:
                self.undo_draft_cell()
                return
        step = self.snap_mm if self.snap_mm > 0 else 0.1
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            step *= 10
        moves = {
            Qt.Key.Key_Left: (-step, 0.0),
            Qt.Key.Key_Right: (step, 0.0),
            Qt.Key.Key_Up: (0.0, step),
            Qt.Key.Key_Down: (0.0, -step),
        }
        if key in moves and self.selected_dies():
            self.nudge_selection(*moves[key])
            return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------ view
    def fit(self) -> None:
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def zoom(self, factor: float) -> None:
        self.scale(factor, factor)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.transform().m11() < 2.0:
            self.fit()

    def grab_png(self, path: str, scale: int = 3) -> bool:
        """Save the current view as a PNG, for slides and reports.

        Rendered through the view rather than the scene, so the interposer,
        overlay field, outlines and track grid drawn in ``drawBackground`` are
        included. *scale* supersamples the on-screen size for a sharp image.
        """

        source = QRectF(self.viewport().rect())
        image = QImage(
            int(source.width() * scale), int(source.height() * scale), QImage.Format.Format_ARGB32
        )
        image.setDevicePixelRatio(scale)
        image.fill(QColor(18, 18, 20))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        self.render(painter, source, source.toRect())
        painter.end()
        return image.save(path)


__all__ = ["MODE_ROUTE", "MODE_SELECT", "DieItem", "LayoutCanvas"]
