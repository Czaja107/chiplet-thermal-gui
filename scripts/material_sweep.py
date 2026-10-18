#!/usr/bin/env python3
"""Run one layout across all five workloads for every thermal-stack material variant.

    python scripts/material_sweep.py                 # ring_center, 64x64, all variants
    python scripts/material_sweep.py --jobs 8

Variants: the unchanged silicon stack (``si``), a spreader layer of each other
material between the die tops and the cooler, and, for comparison, the same
materials swapped into the interposer only (see chiplet_gui/core/materials.py).
All HotSpot solves (variants x workloads) share one pool, so every core HotSpot
can use stays busy.

Writes:
    results/material_sweep.csv              one row per variant and workload
    results/material_sweep_si_vs_bas.png    die-plane maps, si vs bas spreader

Sanity checks (exit status 1 if either fails):
    * hbn_worst is hotter than hbn_best, on both objectives, for each target;
    * si matches the baseline from scripts/run_baseline.py exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_baseline import (  # noqa: E402  (also puts the judge and chiplet_gui on sys.path)
    CHALLENGE,
    DEFAULT_HOTSPOT,
    REPO,
    hotspot_command,
    is_elf,
    load_layout,
)

from chiplet_gui.core.materials import DEFAULT_MATERIAL, MATERIALS, apply_material  # noqa: E402
from generate_workloads import generate  # noqa: E402
from geometry import die_box, validate_layout  # noqa: E402
from run_simulations import create_hotspot_files, die_peaks, parse_grid  # noqa: E402

#: (variant name, material, target) in table order.
VARIANTS = [("si", "si", "spreader")]
VARIANTS += [(m, m, "spreader") for m in MATERIALS if m != DEFAULT_MATERIAL]
VARIANTS += [(f"{m}_interposer", m, "interposer") for m in MATERIALS if m != DEFAULT_MATERIAL]


def hotspot_cores(hotspot: Path) -> int:
    """Cores HotSpot can actually use: WSL's share when it runs through WSL."""

    if os.name == "nt" and is_elf(hotspot):
        try:
            out = subprocess.run(["wsl", "nproc"], capture_output=True, text=True, timeout=60)
            return max(1, int(out.stdout.replace("\0", "").strip()))
        except (OSError, ValueError, subprocess.SubprocessError):
            return 1
    return os.cpu_count() or 1


def save_maps(spec: dict, layout: dict, maps: dict, workloads: list[str], left: str, right: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    width, height = spec["interposer_mm"]
    extent = (0, width, 0, height)
    shown = [maps[(v, w)] for v in (left, right) for w in workloads]
    lo, hi = min(float(m.min()) for m in shown), max(float(m.max()) for m in shown)
    diffs = [maps[(right, w)] - maps[(left, w)] for w in workloads]
    span = max(float(abs(d).max()) for d in diffs) or 1.0

    fig, axes = plt.subplots(len(workloads), 3, figsize=(15, 3.3 * len(workloads)),
                             constrained_layout=True, squeeze=False)
    for row, workload in enumerate(workloads):
        for col, (title, data, kwargs) in enumerate((
            (left, maps[(left, workload)], {"cmap": "inferno", "vmin": lo, "vmax": hi}),
            (right, maps[(right, workload)], {"cmap": "inferno", "vmin": lo, "vmax": hi}),
            (f"{right} - {left}", diffs[row], {"cmap": "coolwarm", "vmin": -span, "vmax": span}),
        )):
            ax = axes[row, col]
            # parse_grid's row 0 is the top of the package, as in the judge's own figure.
            image = ax.imshow(data, origin="upper", extent=extent, **kwargs)
            for die in layout:
                x0, y0, x1, y1 = die_box(spec, layout, die)
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="cyan", lw=0.7))
                ax.text((x0 + x1) / 2, (y0 + y1) / 2, die, color="white", fontsize=6, ha="center", va="center")
            peak = f"max {float(data.max()):.2f} C" if col < 2 else f"range {float(data.min()):+.2f}..{float(data.max()):+.2f} C"
            ax.set_title(f"{workload} | {title}\n{peak}", fontsize=9)
            ax.set_xlim(0, width)
            ax.set_ylim(0, height)
            if col in (1, 2):
                fig.colorbar(image, ax=ax, shrink=0.8, label="C" if col == 1 else "delta C")
    fig.suptitle(f"Die-plane temperature: {left} vs {right}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layout", default="ring_center")
    parser.add_argument("--layouts", type=Path, help="JSON of named layouts; default is the spec's example layouts")
    parser.add_argument("--hotspot", type=Path, default=Path(os.environ.get("HOTSPOT_BINARY", DEFAULT_HOTSPOT)))
    parser.add_argument("--grid", type=int, choices=(64, 128, 256), default=64)
    parser.add_argument("--jobs", type=int, help="parallel HotSpot solves (default: cores HotSpot can use)")
    parser.add_argument("--baseline", type=Path,
                        help="run_baseline.py result to check si against "
                             "(default: gui_runs/baseline_<layout>_<grid>/baseline_results.json)")
    parser.add_argument("--work", type=Path, help="run folders (default: gui_runs/sweep_<layout>_<grid>)")
    parser.add_argument("--results", type=Path, default=REPO / "results", help="where the CSV and PNG go")
    args = parser.parse_args()

    hotspot = args.hotspot.resolve()
    vendor = hotspot.parent
    if not hotspot.is_file() or not (vendor / "examples/example3/example.config").is_file():
        parser.error(f"{hotspot} is not a HotSpot build inside its own repository checkout")
    jobs = args.jobs or hotspot_cores(hotspot)

    spec = json.loads((CHALLENGE / "challenge_spec.json").read_text())
    layout, wires = load_layout(spec, args.layout, args.layouts)
    routing = validate_layout(spec, layout, wire_routes_grid=wires)
    portable = generate(spec)
    workloads = list(portable["workloads"])
    work = (args.work or REPO / "gui_runs" / f"sweep_{args.layout}_{args.grid}").resolve()

    run_dirs, stacks = {}, {}
    for variant, material, target in VARIANTS:
        run_dirs[variant] = work / variant
        create_hotspot_files(spec, portable, layout, routing, run_dirs[variant], args.grid, vendor)
        stacks[variant] = apply_material(run_dirs[variant], spec, material, target)

    tasks = [(variant, workload) for variant, _, _ in VARIANTS for workload in workloads]
    print(f"{args.layout} at {args.grid}x{args.grid}: {len(VARIANTS)} variants x {len(workloads)} workloads "
          f"= {len(tasks)} HotSpot solves, {jobs} at a time", flush=True)

    def solve(variant: str, workload: str) -> tuple[str, str, dict, object]:
        command, cwd = hotspot_command(hotspot, vendor, run_dirs[variant], workload)
        proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError(f"HotSpot failed for {variant}/{workload}:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
        grid_c = parse_grid(run_dirs[variant] / f"{workload}.grid", args.grid)
        return variant, workload, die_peaks(spec, layout, grid_c), grid_c

    started = time.perf_counter()
    peaks, maps = {}, {}
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = [pool.submit(solve, variant, workload) for variant, workload in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            variant, workload, die_peak, grid_c = future.result()
            peaks[(variant, workload)] = die_peak
            maps[(variant, workload)] = grid_c
            print(f"  [{done:2d}/{len(tasks)}] {variant:20s} {workload:17s} "
                  f"{time.perf_counter() - started:5.0f} s", flush=True)

    def gpu(v: str, w: str) -> float:
        return max(t for d, t in peaks[(v, w)].items() if d.startswith("G"))

    def hbm(v: str, w: str) -> float:
        return max(t for d, t in peaks[(v, w)].items() if d.startswith("H"))

    dies = sorted(layout)
    args.results.mkdir(parents=True, exist_ok=True)
    csv_path = args.results / "material_sweep.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "material", "target", "k_w_per_mk", "thickness_mm", "layout", "grid",
                         "workload", "gpu_peak_c", "hbm_peak_c", *(f"{d}_peak_c" for d in dies)])
        for variant, material, target in VARIANTS:
            chosen = MATERIALS[material]
            thickness = "" if material == DEFAULT_MATERIAL else (
                chosen.spreader_thickness_mm if target == "spreader"
                else spec["thermal_model"]["interposer_si_thickness_um"] / 1000)
            for workload in workloads:
                writer.writerow([variant, material, "" if material == DEFAULT_MATERIAL else target,
                                 chosen.k_w_per_mk, thickness, args.layout, args.grid, workload,
                                 gpu(variant, workload), hbm(variant, workload),
                                 *(peaks[(variant, workload)][d] for d in dies)])

    worst = {}
    for variant, _, _ in VARIANTS:
        wg = max(workloads, key=lambda w: gpu(variant, w))
        wh = max(workloads, key=lambda w: hbm(variant, w))
        worst[variant] = (gpu(variant, wg), wg, hbm(variant, wh), wh)
    si_gpu, si_hbm = worst["si"][0], worst["si"][2]
    print(f"\n{'variant':20s} {'worst GPU C':>11s} {'vs si':>7s}  {'workload':16s} "
          f"{'worst HBM C':>11s} {'vs si':>7s}  workload")
    for variant, _, _ in VARIANTS:
        g, wg, h, wh = worst[variant]
        print(f"{variant:20s} {g:11.2f} {g - si_gpu:+7.2f}  {wg:16s} {h:11.2f} {h - si_hbm:+7.2f}  {wh}")
        print(f"{'':20s} {stacks[variant]}")

    png_path = args.results / "material_sweep_si_vs_bas.png"
    save_maps(spec, layout, maps, workloads, "si", "bas", png_path)
    print(f"\nwrote {csv_path}\nwrote {png_path}")

    failures = []
    for suffix in ("", "_interposer"):
        best, poor = f"hbn_best{suffix}", f"hbn_worst{suffix}"
        if not (worst[poor][0] > worst[best][0] and worst[poor][2] > worst[best][2]):
            failures.append(f"{poor} is not hotter than {best} on both objectives "
                            f"(GPU {worst[poor][0]:.2f} vs {worst[best][0]:.2f}, "
                            f"HBM {worst[poor][2]:.2f} vs {worst[best][2]:.2f})")

    baseline_path = args.baseline or REPO / "gui_runs" / f"baseline_{args.layout}_{args.grid}" / "baseline_results.json"
    if not baseline_path.is_file():
        failures.append(f"no baseline at {baseline_path}; run scripts/run_baseline.py first")
    else:
        baseline = json.loads(baseline_path.read_text())["workloads"]
        for workload in workloads:
            if baseline[workload]["die_peak_c"] != peaks[("si", workload)]:
                failures.append(f"si differs from the baseline on {workload}: "
                                f"{peaks[('si', workload)]} vs {baseline[workload]['die_peak_c']}")

    print("\nsanity checks:")
    print(f"  hbn_worst hotter than hbn_best (spreader and interposer): "
          f"{'FAIL' if any('hbn' in f for f in failures) else 'ok'}")
    print(f"  si identical to baseline {baseline_path.name if baseline_path.is_file() else ''}: "
          f"{'FAIL' if any('baseline' in f for f in failures) else 'ok'}")
    for failure in failures:
        print(f"  - {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
