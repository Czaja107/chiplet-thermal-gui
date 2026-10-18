"""Build HotSpot inputs and run the selected workloads, in parallel and cancellably.

The challenge's ``run_simulations.py`` walks the workloads one at a time, which
costs several minutes per layout. The workloads are independent steady-state
solves, so this module fans them out across processes and streams progress
back, while delegating every numerical step -- input generation, grid parsing
and peak extraction -- to that same module.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .analysis import objectives_from_peaks, validate
from .project import Project
from .simcore import is_hotspot_binary, load_modules

#: Signature of the progress callback: fraction in 0..1 and a status line.
ProgressFn = Callable[[float, str], None]
LogFn = Callable[[str], None]


class RunCancelled(RuntimeError):
    """Raised when the user stops a run before it finishes."""


@dataclass
class WorkloadResult:
    """Peak temperatures and the power budget for one simulated workload."""

    workload: str
    gpu_peak_c: float
    hbm_peak_c: float
    die_peak_c: dict[str, float]
    base_power_w: float
    total_power_w: float
    interconnect_phy_w: float
    phy_inside_die_w: float
    phy_outside_die_w: float
    interconnect_wire_w: float
    seconds: float
    temperature_map: np.ndarray | None = None

    def to_dict(self) -> dict:
        """The per-workload record in the judge's ``results.json`` shape."""

        return {
            "gpu_peak_c": self.gpu_peak_c,
            "hbm_peak_c": self.hbm_peak_c,
            "die_peak_c": dict(self.die_peak_c),
            "base_power_w": self.base_power_w,
            "total_power_w": self.total_power_w,
            "interconnect_phy_w": self.interconnect_phy_w,
            "phy_inside_die_w": self.phy_inside_die_w,
            "phy_outside_die_w": self.phy_outside_die_w,
            "interconnect_wire_w": self.interconnect_wire_w,
            "wall_seconds": round(self.seconds, 2),
        }


@dataclass
class RunResult:
    """Everything one completed run produced."""

    label: str
    project: Project
    grid: int
    interconnect_enabled: bool
    run_dir: Path
    workloads: dict[str, WorkloadResult] = field(default_factory=dict)
    objectives: dict[str, float] = field(default_factory=dict)
    routing: dict = field(default_factory=dict)
    deviations: list[str] = field(default_factory=list)
    seconds: float = 0.0
    finished_at: float = field(default_factory=time.time)

    @property
    def worst_gpu_c(self) -> float | None:
        return self.objectives.get("worst_gpu_c")

    @property
    def worst_hbm_c(self) -> float | None:
        return self.objectives.get("worst_hbm_c")

    @property
    def score(self) -> float | None:
        return self.objectives.get("leaderboard_score")

    @property
    def skipped_workloads(self) -> list[str]:
        """Workloads of the design that this run did not simulate."""

        return [name for name in self.project.workloads if name not in self.workloads]

    @property
    def comparable(self) -> bool:
        """True when nothing was changed that would invalidate the score.

        The judge takes the worst case over every workload, so a run that left
        any out can only under-report and is never comparable.
        """

        return not self.deviations and self.interconnect_enabled and not self.skipped_workloads

    def temperature_map(self, workload: str) -> np.ndarray | None:
        entry = self.workloads.get(workload)
        return entry.temperature_map if entry else None

    def to_dict(self) -> dict:
        """A report in the judge's ``results.json`` shape, plus GUI provenance."""

        return {
            "thermal_grid": self.grid,
            "units": "temperatures are Celsius; routes are GB/s*mm",
            "interconnect_enabled": self.interconnect_enabled,
            "interconnect_model": self.project.interconnect_model,
            "reference_objectives_c": self.project.reference_objectives_c,
            "note": (
                "Base powers, link traffic, electrical coefficients and cooling parameters "
                "are synthetic challenge assumptions."
            ),
            "gui": {
                "label": self.label,
                "setup_deviates_from_spec": self.deviations,
                "score_comparable_with_leaderboard": self.comparable,
                "wall_seconds": round(self.seconds, 2),
            },
            "objectives": {self.label: dict(self.objectives)},
            "per_layout_workload": {
                self.label: {
                    **{name: result.to_dict() for name, result in self.workloads.items()},
                    "routing": self.routing,
                }
            },
        }


def build_portable(project: Project, workloads: Sequence[str]) -> dict:
    """The portable workload payload, limited to *workloads*.

    Same structure as ``generate_workloads.generate`` returns, but it honours a
    workload's custom 16x16 maps and skips the ones the run does not need.
    """

    from .analysis import base_power_maps

    spec = project.to_spec()
    sim = load_modules()
    portable = sim.generate_workloads.generate(
        {**spec, "workloads": {name: spec["workloads"][name] for name in workloads}}
    )
    for name in workloads:
        definition = project.workloads[name]
        if not definition.use_custom_maps:
            continue
        maps = {die: array.tolist() for die, array in base_power_maps(project, name).items()}
        entry = portable["workloads"][name]
        entry["power_maps_w"] = maps
        entry["per_die_total_w"] = {
            die: round(sum(sum(row) for row in rows), 6) for die, rows in maps.items()
        }
        entry["total_w"] = round(sum(entry["per_die_total_w"].values()), 6)
    return portable


class SimulationRunner:
    """Runs one layout over the selected workloads.

    One instance serves one run. :meth:`cancel` is safe to call from another
    thread: it stops dispatching further workloads and kills the HotSpot
    processes already started.
    """

    def __init__(self, progress: ProgressFn | None = None, log: LogFn | None = None) -> None:
        self._progress = progress or (lambda fraction, message: None)
        self._log = log or (lambda message: None)
        self._cancel = threading.Event()
        self._processes: set[subprocess.Popen] = set()
        self._lock = threading.Lock()

    # ----------------------------------------------------------------- control
    def cancel(self) -> None:
        """Ask the run to stop and terminate any HotSpot process in flight."""

        self._cancel.set()
        with self._lock:
            processes = list(self._processes)
        for process in processes:
            try:
                process.kill()
            except OSError:
                pass

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def _check_cancelled(self) -> None:
        if self._cancel.is_set():
            raise RunCancelled("the run was stopped")

    # -------------------------------------------------------------------- main
    def prepare(
        self, project: Project, workloads: Sequence[str] | None = None, run_dir: Path | None = None
    ) -> tuple[Path, dict, dict]:
        """Validate the layout and write every HotSpot input file.

        Returns the run directory, the judge's routing report and the per-
        workload power budget. Useful on its own: exporting inputs without
        running HotSpot is exactly this step.
        """

        sim = load_modules()
        names = list(workloads or project.enabled_workloads())
        if not names:
            raise ValueError("Select at least one workload to simulate.")
        missing = [name for name in names if name not in project.workloads]
        if missing:
            raise ValueError(f"Unknown workloads: {', '.join(missing)}")

        self._progress(0.02, "Validating the layout")
        report = validate(project, check_routes=True)
        if not report.legal or not report.routing:
            raise ValueError(f"The layout is not legal, so it cannot be simulated: {report.message}")

        binary = Path(project.run.hotspot_path) if project.run.hotspot_path else None
        if not is_hotspot_binary(binary):
            raise ValueError(
                "Set a valid HotSpot executable in Simulation Setup. It must sit inside its own "
                "repository, because examples/example3/example.config is read from next to it."
            )
        vendor = binary.parent

        target = Path(run_dir or self.default_run_dir(project))
        target.mkdir(parents=True, exist_ok=True)

        self._progress(0.06, "Generating workload power maps")
        portable = build_portable(project, names)

        self._progress(0.12, "Writing HotSpot floorplans and power traces")
        spec = project.to_spec()
        layout = {name: list(xy) for name, xy in project.placements_mm.items()}
        budget = sim.run_simulations.create_hotspot_files(
            spec,
            portable,
            layout,
            report.routing,
            target,
            project.run.grid,
            vendor,
            include_interconnect=project.run.interconnect_enabled,
        )
        (target / "layout.json").write_text(json.dumps(project.to_judge_layouts(), indent=2) + "\n")
        (target / "gui_project.json").write_text(json.dumps(project.to_dict(), indent=2) + "\n")
        self._log(f"HotSpot inputs written to {target}")
        return target, report.routing, budget

    def run(
        self,
        project: Project,
        workloads: Sequence[str] | None = None,
        run_dir: Path | None = None,
        label: str | None = None,
    ) -> RunResult:
        """Prepare inputs, solve every selected workload and collect the results."""

        started = time.time()
        sim = load_modules()
        names = list(workloads or project.enabled_workloads())
        target, routing, budget = self.prepare(project, names, run_dir)
        self._check_cancelled()

        binary = Path(project.run.hotspot_path).resolve()
        vendor = binary.parent
        materials = vendor / "examples/example3/example.materials"
        grid = project.run.grid
        jobs = project.run.effective_jobs(len(names))
        self._log(
            f"Running {len(names)} workload(s) at {grid}x{grid} on {jobs} parallel HotSpot process(es)."
        )

        done = 0
        done_lock = threading.Lock()
        span = 0.84  # progress between input generation and post-processing

        def solve(workload: str) -> tuple[str, float]:
            nonlocal done
            self._check_cancelled()
            elapsed = self._solve_one(workload, target, binary, materials, vendor, grid, project)
            with done_lock:
                done += 1
                fraction = 0.14 + span * done / len(names)
            self._progress(fraction, f"Finished {workload} ({done}/{len(names)})")
            return workload, elapsed

        timings: dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="hotspot") as pool:
            futures = [pool.submit(solve, name) for name in names]
            try:
                for future in futures:
                    workload, elapsed = future.result()
                    timings[workload] = elapsed
            except BaseException:
                self._cancel.set()
                for future in futures:
                    future.cancel()
                raise

        self._check_cancelled()
        self._progress(0.98, "Reading temperature grids")
        spec = project.to_spec()
        layout = {name: list(xy) for name, xy in project.placements_mm.items()}
        results: dict[str, WorkloadResult] = {}
        for workload in names:
            temperatures = sim.run_simulations.parse_grid(target / f"{workload}.grid", grid)
            peaks = sim.run_simulations.die_peaks(spec, layout, temperatures)
            entry = budget[workload]
            results[workload] = WorkloadResult(
                workload=workload,
                gpu_peak_c=max(v for k, v in peaks.items() if project.dies[k].is_gpu),
                hbm_peak_c=max(v for k, v in peaks.items() if not project.dies[k].is_gpu),
                die_peak_c=peaks,
                base_power_w=round(float(entry["base_w"]), 4),
                total_power_w=round(float(entry["total_w"]), 4),
                interconnect_phy_w=round(float(entry["phy_w"]), 4),
                phy_inside_die_w=round(float(entry["phy_inside_die_w"]), 4),
                phy_outside_die_w=round(float(entry["phy_outside_die_w"]), 4),
                interconnect_wire_w=round(float(entry["wire_w"]), 4),
                seconds=timings.get(workload, 0.0),
                temperature_map=temperatures,
            )

        result = RunResult(
            label=label or project.run.layout_name,
            project=project,
            grid=grid,
            interconnect_enabled=project.run.interconnect_enabled,
            run_dir=target,
            workloads=results,
            objectives={
                **objectives_from_peaks(project, {k: v.to_dict() for k, v in results.items()}),
                "weighted_route_gb_s_mm": float(routing["weighted_route_gb_s_mm"]),
            },
            routing=routing,
            deviations=project.deviations(),
            seconds=time.time() - started,
        )
        (target / "results.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n")
        np.savez_compressed(
            target / "temperature_maps.npz",
            **{name: entry.temperature_map for name, entry in results.items()},
        )
        self._progress(1.0, "Done")
        self._log(
            f"Run finished in {result.seconds:.1f} s. "
            f"Worst GPU {result.worst_gpu_c:.2f} C, worst HBM {result.worst_hbm_c:.2f} C."
        )
        return result

    # ----------------------------------------------------------------- helpers
    def _solve_one(
        self,
        workload: str,
        run_dir: Path,
        binary: Path,
        materials: Path,
        vendor: Path,
        grid: int,
        project: Project,
    ) -> float:
        """Run HotSpot for one trace, reusing an identical previous solve."""

        sim = load_modules()
        started = time.time()
        fingerprint = sim.run_simulations.input_fingerprint(run_dir, workload, binary, vendor)
        digest_path = run_dir / f"{workload}.sha256"
        if (
            project.run.reuse_existing
            and (run_dir / f"{workload}.grid").is_file()
            and digest_path.is_file()
            and digest_path.read_text().strip() == fingerprint
        ):
            self._log(f"{workload}: reusing the identical previous solve")
            return 0.0

        command = [
            str(binary),
            "-c", "benchmark.config",
            "-f", "chiplets.flp",
            "-p", f"{workload}.ptrace",
            "-materials_file", str(materials),
            "-model_type", "grid",
            "-detailed_3D", "on",
            "-grid_layer_file", "stack.lcf",
            "-grid_steady_file", f"{workload}.grid",
            "-steady_file", f"{workload}.steady",
        ]
        self._log(f"{workload}: solving {grid}x{grid} steady state")
        process = subprocess.Popen(
            command,
            cwd=run_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        with self._lock:
            self._processes.add(process)
        try:
            stdout, stderr = process.communicate()
        finally:
            with self._lock:
                self._processes.discard(process)

        if self._cancel.is_set():
            raise RunCancelled("the run was stopped")
        if process.returncode:
            raise RuntimeError(
                f"HotSpot failed for {workload} (exit {process.returncode}).\n"
                f"{(stdout or '').strip()}\n{(stderr or '').strip()}".strip()
            )
        digest_path.write_text(fingerprint + "\n")
        return time.time() - started

    @staticmethod
    def default_run_dir(project: Project) -> Path:
        """Where a run writes, unless the project names somewhere else."""

        root = Path(project.run.output_dir) if project.run.output_dir else Path.cwd() / "gui_runs"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return root / f"{project.run.layout_name}-{stamp}"


def load_run_result(directory: str | Path, project: Project | None = None) -> RunResult:
    """Reload a finished run from its directory, maps included.

    Lets the GUI reopen yesterday's results, and makes a run directory
    self-describing rather than a pile of HotSpot files.
    """

    target = Path(directory)
    report = json.loads((target / "results.json").read_text())
    label = next(iter(report["objectives"]))
    per_workload = report["per_layout_workload"][label]

    if project is None:
        saved = target / "gui_project.json"
        if saved.is_file():
            project = Project.from_dict(json.loads(saved.read_text()))
        else:
            project = Project.default()

    maps: dict[str, np.ndarray] = {}
    archive = target / "temperature_maps.npz"
    if archive.is_file():
        with np.load(archive) as handle:
            maps = {name: handle[name] for name in handle.files}

    workloads: dict[str, WorkloadResult] = {}
    for name, entry in per_workload.items():
        if name == "routing" or not isinstance(entry, dict) or "gpu_peak_c" not in entry:
            continue
        workloads[name] = WorkloadResult(
            workload=name,
            gpu_peak_c=float(entry["gpu_peak_c"]),
            hbm_peak_c=float(entry["hbm_peak_c"]),
            die_peak_c={k: float(v) for k, v in entry["die_peak_c"].items()},
            base_power_w=float(entry.get("base_power_w", 0.0)),
            total_power_w=float(entry.get("total_power_w", 0.0)),
            interconnect_phy_w=float(entry.get("interconnect_phy_w", 0.0)),
            phy_inside_die_w=float(entry.get("phy_inside_die_w", 0.0)),
            phy_outside_die_w=float(entry.get("phy_outside_die_w", 0.0)),
            interconnect_wire_w=float(entry.get("interconnect_wire_w", 0.0)),
            seconds=float(entry.get("wall_seconds", 0.0)),
            temperature_map=maps.get(name),
        )

    gui = report.get("gui", {})
    return RunResult(
        label=label,
        project=project,
        grid=int(report.get("thermal_grid", 64)),
        interconnect_enabled=bool(report.get("interconnect_enabled", True)),
        run_dir=target,
        workloads=workloads,
        objectives={k: float(v) for k, v in report["objectives"][label].items()},
        routing=per_workload.get("routing", {}),
        deviations=list(gui.get("setup_deviates_from_spec", [])),
        seconds=float(gui.get("wall_seconds", 0.0)),
        finished_at=(target / "results.json").stat().st_mtime,
    )


def copy_judge_inputs(result: RunResult, destination: str | Path) -> Path:
    """Copy one run's HotSpot input set somewhere the judge can pick it up."""

    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    wanted = ("chiplets.flp", "interposer.flp", "tim.flp", "stack.lcf", "benchmark.config")
    for name in wanted:
        source = result.run_dir / name
        if source.is_file():
            shutil.copy2(source, target / name)
    for workload in result.workloads:
        source = result.run_dir / f"{workload}.ptrace"
        if source.is_file():
            shutil.copy2(source, target / f"{workload}.ptrace")
    for name in ("layout.json", "power_budget.json", "results.json"):
        source = result.run_dir / name
        if source.is_file():
            shutil.copy2(source, target / name)
    return target


__all__ = [
    "LogFn",
    "ProgressFn",
    "RunCancelled",
    "RunResult",
    "SimulationRunner",
    "WorkloadResult",
    "build_portable",
    "copy_judge_inputs",
    "load_run_result",
]
