#!/usr/bin/env python3
"""Simulate one reference layout across all five workloads with the judge's own code.

    python scripts/run_baseline.py                      # ring_center, 64x64 grid
    python scripts/run_baseline.py --layout ring_spread --grid 128

Every step is the challenge package's: ``generate`` builds the workloads,
``validate_layout`` routes the links, ``create_hotspot_files`` writes the
HotSpot inputs, and ``parse_grid`` / ``die_peaks`` read the result. The script
only adds the command line, parallel solves and a comparison with the
organizers' bundled ``example_outputs/results.json``.

HotSpot is a Linux build on most setups. On Windows, a Linux (ELF) binary is
run through WSL automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHALLENGE = REPO / "chiplet_thermal_continuous"
sys.path.insert(0, str(CHALLENGE))
sys.path.insert(0, str(REPO))

from generate_workloads import generate  # noqa: E402
from geometry import validate_layout  # noqa: E402
from run_simulations import create_hotspot_files, die_peaks, parse_grid  # noqa: E402

from chiplet_gui.core.materials import DEFAULT_MATERIAL, MATERIALS, TARGETS, apply_material  # noqa: E402

#: Where the README tells people to build HotSpot, relative to this repository.
DEFAULT_HOTSPOT = REPO.parent / "vendor" / "HotSpot" / "hotspot"

#: Peaks are rounded to 0.01 C by the judge, so anything larger is a real difference.
TOLERANCE_C = 0.01


def is_elf(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(4) == b"\x7fELF"


def wsl_path(path: Path) -> str:
    """D:\\a\\b -> /mnt/d/a/b, the default WSL mount of a Windows drive."""

    path = path.resolve()
    drive, rest = path.drive.rstrip(":").lower(), path.as_posix()[len(path.drive):]
    return f"/mnt/{drive}{rest}"


def hotspot_command(hotspot: Path, vendor: Path, run_dir: Path, workload: str) -> tuple[list[str], Path | None]:
    """The judge's exact HotSpot command, wrapped in ``wsl`` when needed."""

    via_wsl = os.name == "nt" and is_elf(hotspot)
    convert = wsl_path if via_wsl else str
    command = [
        convert(hotspot), "-c", "benchmark.config", "-f", "chiplets.flp",
        "-p", f"{workload}.ptrace",
        "-materials_file", convert(vendor / "examples/example3/example.materials"),
        "-model_type", "grid", "-detailed_3D", "on", "-grid_layer_file", "stack.lcf",
        "-grid_steady_file", f"{workload}.grid", "-steady_file", f"{workload}.steady",
    ]
    if via_wsl:
        return ["wsl", "--cd", wsl_path(run_dir), "--", *command], None
    return command, run_dir


def load_layout(spec: dict, name: str, layouts_file: Path | None) -> tuple[dict, dict]:
    raw = json.loads(layouts_file.read_text()) if layouts_file else spec["example_layouts_mm"]
    if name not in raw:
        raise SystemExit(f"layout {name!r} not found; choose from {', '.join(raw)}")
    entry = raw[name]
    return entry.get("placements_mm", entry), entry.get("wire_routes_grid", {})


def organizer_reference(name: str, grid: int, interconnect: bool) -> tuple[dict, str] | None:
    """Per-workload peaks for *name* from the organizers' bundled results, if present.

    ``random_example/`` is the current judge's reference run. ``example_outputs/``
    predates the interconnect model: its interposer carries no heat and its
    results.json has no ``interconnect_enabled`` field, so it is only a valid
    comparison for a run with link heat disabled.
    """

    for folder in ("random_example", "example_outputs"):
        path = CHALLENGE / folder / "results.json"
        if not path.is_file():
            continue
        report = json.loads(path.read_text())
        if report.get("interconnect_enabled", False) != interconnect:
            continue
        if report.get("thermal_grid") == grid and name in report.get("per_layout_workload", {}):
            return report["per_layout_workload"][name], folder
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layout", default="ring_center", help="layout name (default: ring_center)")
    parser.add_argument("--layouts", type=Path, help="JSON of named layouts; default is the spec's example layouts")
    parser.add_argument("--hotspot", type=Path, default=Path(os.environ.get("HOTSPOT_BINARY", DEFAULT_HOTSPOT)))
    parser.add_argument("--grid", type=int, choices=(64, 128, 256), default=64)
    parser.add_argument("--output", type=Path, help="run folder (default: gui_runs/baseline_<layout>_<grid>)")
    parser.add_argument("--jobs", type=int, default=5, help="parallel HotSpot solves (default: 5)")
    parser.add_argument("--disable-interconnect", action="store_true",
                        help="zero link heat, as in the pre-interconnect example_outputs/ run")
    parser.add_argument("--material", choices=list(MATERIALS), default=DEFAULT_MATERIAL,
                        help="thermal-stack material variant (default: si, the unchanged challenge stack)")
    parser.add_argument("--material-target", choices=TARGETS, default="spreader",
                        help="spreader: add a layer between die tops and cooler (default); "
                             "interposer: swap only the interposer's material, for comparison")
    args = parser.parse_args()

    hotspot = args.hotspot.resolve()
    vendor = hotspot.parent
    if not hotspot.is_file() or not (vendor / "examples/example3/example.config").is_file():
        parser.error(f"{hotspot} is not a HotSpot build inside its own repository checkout")

    spec = json.loads((CHALLENGE / "challenge_spec.json").read_text())
    layout, wires = load_layout(spec, args.layout, args.layouts)
    routing = validate_layout(spec, layout, wire_routes_grid=wires)
    portable = generate(spec)
    suffix = "_nolinks" if args.disable_interconnect else ""
    if args.material != DEFAULT_MATERIAL:
        suffix += f"_{args.material}" + ("_interposer" if args.material_target == "interposer" else "")
    run_dir = (args.output or REPO / "gui_runs" / f"baseline_{args.layout}_{args.grid}{suffix}").resolve()
    create_hotspot_files(spec, portable, layout, routing, run_dir, args.grid, vendor,
                         include_interconnect=not args.disable_interconnect)
    stack = apply_material(run_dir, spec, args.material, args.material_target)
    workloads = list(portable["workloads"])
    print(f"{args.layout}: {len(workloads)} workloads at {args.grid}x{args.grid}, "
          f"route {routing['weighted_route_gb_s_mm']} GB/s*mm, inputs in {run_dir}", flush=True)
    print(f"stack: {stack}", flush=True)

    def solve(workload: str) -> tuple[str, dict]:
        command, cwd = hotspot_command(hotspot, vendor, run_dir, workload)
        started = time.perf_counter()
        proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError(f"HotSpot failed for {workload}:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
        peaks = die_peaks(spec, layout, parse_grid(run_dir / f"{workload}.grid", args.grid))
        print(f"  solved {workload} in {time.perf_counter() - started:.0f} s", flush=True)
        return workload, {
            "gpu_peak_c": max(t for die, t in peaks.items() if die.startswith("G")),
            "hbm_peak_c": max(t for die, t in peaks.items() if die.startswith("H")),
            "die_peak_c": peaks,
        }

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        results = dict(pool.map(solve, workloads))

    # Organizer results exist only for the unmodified silicon stack.
    found = (organizer_reference(args.layout, args.grid, not args.disable_interconnect)
             if args.material == DEFAULT_MATERIAL else None)
    reference, source = found if found else (None, "")
    print()
    header = f"{'workload':18s} {'GPU C':>8s} {'HBM C':>8s}"
    if reference:
        header += f"   {'ref GPU':>8s} {'ref HBM':>8s}"
    print(header)
    mismatches = []
    for workload in workloads:
        gpu, hbm = results[workload]["gpu_peak_c"], results[workload]["hbm_peak_c"]
        line = f"{workload:18s} {gpu:8.2f} {hbm:8.2f}"
        if reference and workload in reference:
            ref_gpu, ref_hbm = reference[workload]["gpu_peak_c"], reference[workload]["hbm_peak_c"]
            line += f"   {ref_gpu:8.2f} {ref_hbm:8.2f}"
            if abs(gpu - ref_gpu) > TOLERANCE_C or abs(hbm - ref_hbm) > TOLERANCE_C:
                line += "   MISMATCH"
                mismatches.append(workload)
        print(line)

    worst_gpu = max(workloads, key=lambda w: results[w]["gpu_peak_c"])
    worst_hbm = max(workloads, key=lambda w: results[w]["hbm_peak_c"])
    print(f"\nworst GPU {results[worst_gpu]['gpu_peak_c']:.2f} C ({worst_gpu})")
    print(f"worst HBM {results[worst_hbm]['hbm_peak_c']:.2f} C ({worst_hbm})")

    (run_dir / "baseline_results.json").write_text(json.dumps(
        {"layout": args.layout, "grid": args.grid, "hotspot": str(hotspot),
         "interconnect_enabled": not args.disable_interconnect,
         "material": args.material, "material_target": args.material_target, "stack": stack,
         "worst_gpu_c": results[worst_gpu]["gpu_peak_c"], "worst_hbm_c": results[worst_hbm]["hbm_peak_c"],
         "workloads": results}, indent=2) + "\n")

    if reference is None:
        print("(no organizer results bundled for this layout, grid, link-heat and material setting)")
        return 0
    if mismatches:
        print(f"MISMATCH against the organizers' {source}/ results in: {', '.join(mismatches)}")
        return 1
    print(f"Matches the organizers' {source}/ results to within {TOLERANCE_C} C on every workload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
