"""Validation, route-grid geometry and the instant (HotSpot-free) power preview.

Everything here is fast enough to recompute while a die is being dragged. The
authoritative legality verdict always comes from ``geometry.validate_layout``;
the extra helpers add what a GUI needs on top of a single pass/fail, namely
which die is at fault, which track cells are blocked, and where a hand-drawn
wire is allowed to start and end.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .project import Project
from .simcore import load_modules

Cell = tuple[int, int]


# --------------------------------------------------------------------- layout
@dataclass
class LayoutReport:
    """What the GUI needs to know about one candidate placement.

    :attr:`legal` is the judge's verdict. :attr:`routing` carries the judge's
    port assignment and accepted paths when the layout is legal, and
    :attr:`die_errors` localises failures so the canvas can colour the dies
    that are actually at fault even when the layout is rejected.
    """

    legal: bool
    message: str = ""
    routing: dict | None = None
    die_errors: dict[str, list[str]] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    @property
    def weighted_route(self) -> float | None:
        if not self.routing:
            return None
        return float(self.routing["weighted_route_gb_s_mm"])

    @property
    def route_limit(self) -> float | None:
        if not self.routing:
            return None
        return float(self.routing["route_limit_gb_s_mm"])

    def route(self, link_id: str) -> dict | None:
        if not self.routing:
            return None
        return self.routing["routes"].get(link_id)


def placement_issues(project: Project) -> dict[str, list[str]]:
    """Per-die geometric violations: off-interposer, edge clearance, overlap.

    Mirrors the first half of ``geometry.validate_layout``, which stops at the
    first error. Reporting all of them at once is what lets the canvas mark
    every offending die instead of just one.
    """

    width, height = project.interposer_mm
    edge = project.edge_clearance_mm
    gap = project.minimum_die_gap_mm
    errors: dict[str, list[str]] = {name: [] for name in project.placements_mm}

    boxes = {name: project.die_box(name) for name in project.placements_mm}
    for name, (x0, y0, x1, y1) in boxes.items():
        if min(x0, y0) < edge - 1e-8 or x1 > width - edge + 1e-8 or y1 > height - edge + 1e-8:
            errors[name].append(f"closer than {edge:g} mm to the interposer edge")

    names = list(boxes)
    for i, name in enumerate(names):
        a = boxes[name]
        for other in names[i + 1:]:
            b = boxes[other]
            gx = max(b[0] - a[2], a[0] - b[2], 0)
            gy = max(b[1] - a[3], a[1] - b[3], 0)
            if gx < gap - 1e-8 and gy < gap - 1e-8:
                overlapping = gx <= 0 and gy <= 0
                verb = "overlaps" if overlapping else f"is within {gap:g} mm of"
                errors[name].append(f"{verb} {other}")
                errors[other].append(f"{verb} {name}")
    return {name: found for name, found in errors.items() if found}


def validate(project: Project, check_routes: bool = True) -> LayoutReport:
    """Run the judge's own validation and package the result for the GUI."""

    sim = load_modules()
    spec = project.to_spec()
    layout = {name: list(xy) for name, xy in project.placements_mm.items()}
    die_errors = placement_issues(project)

    routes = {k: [list(cell) for cell in v] for k, v in project.wire_routes_grid.items()}
    try:
        routing = sim.geometry.validate_layout(
            spec, layout, check_routes=check_routes, wire_routes_grid=routes or None
        )
    except (ValueError, KeyError) as exc:
        issues = [f"{name}: {', '.join(found)}" for name, found in sorted(die_errors.items())]
        return LayoutReport(
            legal=False, message=str(exc), routing=None, die_errors=die_errors, issues=issues
        )
    return LayoutReport(legal=True, message="", routing=routing, die_errors={}, issues=[])


def auto_routes(project: Project) -> dict[str, list[list[int]]]:
    """The deterministic router's paths for every link, as editable cell lists.

    Baking these into :attr:`Project.wire_routes_grid` is how the GUI turns the
    judge's own routing into a starting point the user can drag around.
    """

    report = validate(project, check_routes=True)
    if not report.legal or not report.routing:
        raise ValueError(report.message or "layout is not legal, so it cannot be routed")
    return {
        link_id: [list(cell) for cell in route["path_grid"]]
        for link_id, route in report.routing["routes"].items()
        if route["path_grid"]
    }


# ----------------------------------------------------------------- route grid
@dataclass
class RouteGrid:
    """The 0.5 mm track lattice a hand-drawn wire must follow.

    ``blocked`` is indexed ``[j, i]`` to match the judge's ``(i, j)`` cell
    convention with j the row. ``endpoints`` gives the required first and last
    cell of each link's path, which is what makes manual routing checkable
    before the judge sees it.
    """

    pitch: float
    nx: int
    ny: int
    blocked: np.ndarray
    endpoints: dict[str, tuple[Cell, Cell]] = field(default_factory=dict)
    access_mm: dict[str, tuple[float, float]] = field(default_factory=dict)
    port_errors: dict[str, str] = field(default_factory=dict)

    def center_mm(self, i: int, j: int) -> tuple[float, float]:
        """Track-cell centre in mm, the judge's ``((i+0.5)*pitch, (j+0.5)*pitch)``."""

        return (i + 0.5) * self.pitch, (j + 0.5) * self.pitch

    def cell_at(self, x_mm: float, y_mm: float) -> Cell:
        """The track cell containing a point in mm, clamped to the lattice."""

        i = min(self.nx - 1, max(0, int(math.floor(x_mm / self.pitch))))
        j = min(self.ny - 1, max(0, int(math.floor(y_mm / self.pitch))))
        return i, j

    def in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.nx and 0 <= cell[1] < self.ny

    def is_blocked(self, cell: Cell) -> bool:
        return not self.in_bounds(cell) or bool(self.blocked[cell[1], cell[0]])

    def is_free(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and not self.blocked[cell[1], cell[0]]

    def path_length_mm(self, link_id: str, path: Sequence[Cell]) -> float | None:
        """Routed length of *path* including both port-to-track access legs.

        Returns ``None`` when the link has no usable port access, matching the
        judge's refusal to measure a route it cannot start.
        """

        if link_id not in self.access_mm or len(path) < 1:
            return None
        start_access, end_access = self.access_mm[link_id]
        return round((len(path) - 1) * self.pitch + start_access + end_access, 3)


def route_grid(project: Project, report: LayoutReport | None = None) -> RouteGrid:
    """Build the track lattice, blocked cells and per-link access cells.

    The blocked test and the port-access cell repeat what
    ``geometry._route_grid`` does internally; that function is the authority
    and every path the GUI produces is still re-checked by
    :func:`validate` before a run.
    """

    rule = project.port_rule
    pitch = float(rule["route_grid_pitch_mm"])
    clearance = float(rule["route_clearance_mm"])
    width, height = project.interposer_mm
    nx, ny = round(width / pitch), round(height / pitch)

    boxes = [project.die_box(name) for name in project.placements_mm]
    xs = (np.arange(nx) + 0.5) * pitch
    ys = (np.arange(ny) + 0.5) * pitch
    gx, gy = np.meshgrid(xs, ys)
    blocked = np.zeros((ny, nx), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        blocked |= (
            (gx > x0 - clearance)
            & (gx < x1 + clearance)
            & (gy > y0 - clearance)
            & (gy < y1 + clearance)
        )

    grid = RouteGrid(pitch=pitch, nx=nx, ny=ny, blocked=blocked)
    report = report or validate(project, check_routes=False)
    if not report.routing:
        return grid

    for link_id, route in report.routing["routes"].items():
        try:
            start, start_access = _access_cell(route["from_port"], route["from_normal"], grid, clearance)
            goal, end_access = _access_cell(route["to_port"], route["to_normal"], grid, clearance)
        except ValueError as exc:
            grid.port_errors[link_id] = str(exc)
            continue
        grid.endpoints[link_id] = (start, goal)
        grid.access_mm[link_id] = (start_access, end_access)
    return grid


def _access_cell(
    port: Sequence[float], normal: Sequence[float], grid: RouteGrid, clearance: float
) -> tuple[Cell, float]:
    """The free track cell a port reaches, and the Manhattan access length to it."""

    x, y = float(port[0]), float(port[1])
    dx, dy = float(normal[0]), float(normal[1])
    px = x + dx * (clearance + grid.pitch)
    py = y + dy * (clearance + grid.pitch)
    cell = (math.floor(px / grid.pitch), math.floor(py / grid.pitch))
    if not grid.in_bounds(cell) or grid.is_blocked(cell):
        raise ValueError("port cannot access a free route track")
    cx, cy = grid.center_mm(*cell)
    return cell, abs(x - cx) + abs(y - cy)


def path_problems(
    grid: RouteGrid, link_id: str, path: Sequence[Cell], limit_mm: float | None = None
) -> list[str]:
    """Everything wrong with a hand-drawn path, in the order a user would fix it.

    Checked while the wire is being drawn, so the canvas can refuse a bad cell
    instead of letting the error surface minutes later inside the judge.
    """

    problems: list[str] = []
    if not path:
        return ["the path is empty"]

    cells = [(int(i), int(j)) for i, j in path]
    for cell in cells:
        if not grid.in_bounds(cell):
            problems.append(f"cell {list(cell)} is outside the interposer")
            break
        if grid.is_blocked(cell):
            problems.append(f"cell {list(cell)} is under a die or inside its clearance")
            break

    for a, b in zip(cells, cells[1:]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            problems.append(f"cells {list(a)} and {list(b)} are not Manhattan-adjacent")
            break

    if len(set(cells)) != len(cells):
        problems.append("the path visits the same cell twice")

    expected = grid.endpoints.get(link_id)
    if expected:
        start, goal = expected
        if cells[0] != start:
            problems.append(f"must start at the port access cell {list(start)}")
        if cells[-1] != goal:
            problems.append(f"must end at the port access cell {list(goal)}")
    elif link_id in grid.port_errors:
        problems.append(grid.port_errors[link_id])

    if limit_mm is not None:
        length = grid.path_length_mm(link_id, cells)
        if length is not None and length > limit_mm + 1e-8:
            problems.append(f"routed {length:.2f} mm exceeds the {limit_mm:.2f} mm reach")
    return problems


def projected_weighted_route(
    report: LayoutReport, link_id: str, new_length_mm: float
) -> tuple[float, float]:
    """``(projected, limit)`` weighted route if *link_id* took *new_length_mm*.

    The judge caps ``sum(bandwidth * routed length)`` across all eight links, so
    lengthening one wire can be rejected even when it is well inside its own
    reach. Projecting the total lets the wire editor say so while drawing.
    """

    routing = report.routing
    if not routing:
        return 0.0, 0.0
    limit = float(routing["route_limit_gb_s_mm"])
    total = 0.0
    for other, route in routing["routes"].items():
        bandwidth = float(route["bandwidth_gb_s"])
        length = float(new_length_mm) if other == link_id else float(route["route_length_mm"])
        total += bandwidth * length
    return round(total, 2), limit


def shortest_path(
    grid: RouteGrid, start: Cell, goal: Cell, avoid: set[Cell] | None = None
) -> list[Cell] | None:
    """Breadth-first Manhattan path between two free cells, or ``None``.

    *avoid* keeps the search off cells a partly drawn wire already occupies, so
    completing a deliberate detour does not simply retrace it. Used by the wire
    editor; the judge's own router is what produces :func:`auto_routes`.
    """

    if grid.is_blocked(start) or grid.is_blocked(goal):
        return None
    if start == goal:
        return [start]
    from collections import deque

    forbidden = (avoid or set()) - {start, goal}
    parent: dict[Cell, Cell] = {start: start}
    queue: deque[Cell] = deque([start])
    while queue:
        cell = queue.popleft()
        for delta in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (cell[0] + delta[0], cell[1] + delta[1])
            if nxt in parent or nxt in forbidden or grid.is_blocked(nxt):
                continue
            parent[nxt] = cell
            if nxt == goal:
                path = [goal]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                path.reverse()
                return path
            queue.append(nxt)
    return None


# -------------------------------------------------------------- power preview
@dataclass
class PowerPreview:
    """Resolved power for one workload at the current placement, without HotSpot.

    This is the input HotSpot would receive: base die maps with the inside-PHY
    rectangles added, and the interposer tile map holding outside-PHY and
    routed-wire heat. It takes milliseconds, so the canvas can show link heat
    move as a die is dragged.
    """

    workload: str
    die_maps_w: dict[str, np.ndarray]
    interposer_tiles_w: np.ndarray
    link_breakdown: dict[str, dict]
    base_w: float
    phy_w: float
    phy_inside_die_w: float
    phy_outside_die_w: float
    wire_w: float
    total_w: float

    def die_total_w(self, name: str) -> float:
        return float(self.die_maps_w[name].sum())

    def peak_density(self, name: str) -> float:
        return float(self.die_maps_w[name].max())


def base_power_maps(project: Project, workload: str) -> dict[str, np.ndarray]:
    """The 16x16 base W/cell maps for one workload, before any link heat.

    Custom maps win when a workload carries them; otherwise the maps are
    generated by the challenge's own ``generate_workloads.map_for``.
    """

    sim = load_modules()
    spec = project.to_spec()
    workload_def = project.workloads[workload]
    maps: dict[str, np.ndarray] = {}
    for name in project.dies:
        custom = workload_def.custom_maps_w.get(name) if workload_def.use_custom_maps else None
        if custom is not None:
            maps[name] = np.array(custom, dtype=float)
        else:
            maps[name] = np.asarray(sim.generate_workloads.map_for(spec, workload, name), dtype=float)
    return maps


def power_preview(
    project: Project, workload: str, report: LayoutReport | None = None
) -> PowerPreview:
    """Resolve base power plus PHY and routed-wire heat for one workload.

    Raises :class:`ValueError` when the layout is not legal, because link heat
    is undefined without an accepted port assignment and route.
    """

    sim = load_modules()
    spec = project.to_spec()
    report = report or validate(project, check_routes=True)
    base = base_power_maps(project, workload)
    layout = {name: list(xy) for name, xy in project.placements_mm.items()}

    if not project.run.interconnect_enabled:
        nx, ny = project.interconnect_model["interposer_heat_tiles"]
        total = float(sum(m.sum() for m in base.values()))
        return PowerPreview(
            workload=workload,
            die_maps_w=base,
            interposer_tiles_w=np.zeros((ny, nx)),
            link_breakdown={},
            base_w=total,
            phy_w=0.0,
            phy_inside_die_w=0.0,
            phy_outside_die_w=0.0,
            wire_w=0.0,
            total_w=total,
        )

    if not report.legal or not report.routing:
        raise ValueError(report.message or "layout is not legal")

    heat = sim.interconnect.combined_heat(
        spec, workload, {k: v.tolist() for k, v in base.items()}, layout, report.routing
    )
    return PowerPreview(
        workload=workload,
        die_maps_w={k: np.asarray(v, dtype=float) for k, v in heat["die_maps_w"].items()},
        interposer_tiles_w=np.asarray(heat["interposer_tiles_w"], dtype=float),
        link_breakdown=heat["link_breakdown"],
        base_w=float(heat["base_w"]),
        phy_w=float(heat["phy_w"]),
        phy_inside_die_w=float(heat["phy_inside_die_w"]),
        phy_outside_die_w=float(heat["phy_outside_die_w"]),
        wire_w=float(heat["wire_w"]),
        total_w=float(heat["total_w"]),
    )


# ------------------------------------------------------------- layout helpers
#: Die origins are kept to this many decimals everywhere: in memory, in saved
#: designs and in the layout handed to the judge, so what is simulated is
#: exactly what is submitted.
PLACEMENT_DECIMALS = 3


def snap(value: float, step: float) -> float:
    """Round *value* to the nearest multiple of *step*; ``step <= 0`` disables."""

    if step <= 0:
        return round(float(value), PLACEMENT_DECIMALS)
    return round(round(float(value) / step) * step, PLACEMENT_DECIMALS)


def round_origin(x: float, y: float) -> list[float]:
    """A die origin at the precision used for saving and judging."""

    return [round(float(x), PLACEMENT_DECIMALS), round(float(y), PLACEMENT_DECIMALS)]


def clamp_into_interposer(project: Project, name: str, x: float, y: float) -> tuple[float, float]:
    """Keep a die origin inside the legal edge-clearance window."""

    width, height = project.interposer_mm
    edge = project.edge_clearance_mm
    w, h = project.dies[name].size_mm
    x = min(max(float(x), edge), max(edge, width - edge - float(w)))
    y = min(max(float(y), edge), max(edge, height - edge - float(h)))
    return x, y


def random_legal_layout(
    project: Project,
    seed: int | None = None,
    jitter_mm: float = 0.65,
    base: dict[str, Sequence[float]] | None = None,
    attempts: int = 10000,
) -> dict[str, list[float]]:
    """Rejection-sample a legal layout around a base placement.

    Same procedure as the challenge's ``generate_random_layout.py``, so a seed
    used there reproduces here. Defaults to jittering the current placement.
    """

    rng = random.Random(seed)
    origin = base or project.placements_mm
    probe = project.copy()
    for _ in range(attempts):
        candidate = {
            die: [
                round(float(xy[0]) + rng.uniform(-jitter_mm, jitter_mm), 3),
                round(float(xy[1]) + rng.uniform(-jitter_mm, jitter_mm), 3),
            ]
            for die, xy in origin.items()
        }
        probe.placements_mm = candidate
        probe.wire_routes_grid = {}
        if validate(probe, check_routes=True).legal:
            return candidate
    raise RuntimeError(
        f"No legal layout in {attempts} attempts. Reduce the jitter or start from a legal layout."
    )


def objectives_from_peaks(
    project: Project, per_workload: dict[str, dict[str, Any]]
) -> dict[str, float]:
    """Worst GPU/HBM temperature and the leaderboard score, as the judge scores it."""

    if not per_workload:
        return {}
    worst_gpu = max(float(v["gpu_peak_c"]) for v in per_workload.values())
    worst_hbm = max(float(v["hbm_peak_c"]) for v in per_workload.values())
    result = {"worst_gpu_c": round(worst_gpu, 2), "worst_hbm_c": round(worst_hbm, 2)}
    reference = project.reference_objectives_c
    if reference:
        ambient = float(project.thermal_model["ambient_c"])
        result["leaderboard_score"] = round(
            max(
                (worst_gpu - ambient) / (float(reference["worst_gpu_c"]) - ambient),
                (worst_hbm - ambient) / (float(reference["worst_hbm_c"]) - ambient),
            ),
            6,
        )
    return result


def die_peak_names(names: Iterable[str], prefix: str) -> list[str]:
    return sorted(name for name in names if name.startswith(prefix))


__all__ = [
    "Cell",
    "LayoutReport",
    "PowerPreview",
    "RouteGrid",
    "auto_routes",
    "base_power_maps",
    "clamp_into_interposer",
    "die_peak_names",
    "objectives_from_peaks",
    "path_problems",
    "projected_weighted_route",
    "placement_issues",
    "power_preview",
    "random_legal_layout",
    "route_grid",
    "shortest_path",
    "snap",
    "validate",
]
