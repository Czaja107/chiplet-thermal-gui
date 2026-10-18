#!/usr/bin/env python3
"""Evaluate several layouts on several stack materials and compare them.

    python scripts/compare_layouts.py
    python scripts/compare_layouts.py --layout mine=path/to/submission.json --material si bas

Defaults: ring_center plus the optimizer's best-for-si and best-for-bas
layouts (results/optimize_<material>/submission.json), each on the si and bas
stacks. Every evaluation is the full judge path with all five workloads.

Writes, under results/:
    compare_layouts.csv       one row per layout and material
    compare_layouts.md        the worst GPU / worst HBM table plus layout geometry
    compare_layouts_maps.png  die-plane maps, rows = layouts, columns = materials;
                              each panel is the hottest each cell gets over the
                              five workloads, on one shared colour scale
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimize import Evaluator, rounded  # noqa: E402  (puts the judge on sys.path)
from run_baseline import CHALLENGE, DEFAULT_HOTSPOT, REPO  # noqa: E402
from material_sweep import hotspot_cores  # noqa: E402

from chiplet_gui.core.materials import MATERIALS  # noqa: E402
from geometry import die_box, validate_layout  # noqa: E402


def geometry(spec: dict, layout: dict) -> dict:
    """How tightly the dies are packed and how long the wires are."""

    routing = validate_layout(spec, layout)
    boxes = {d: die_box(spec, layout, d) for d in layout}
    x0 = min(b[0] for b in boxes.values())
    y0 = min(b[1] for b in boxes.values())
    x1 = max(b[2] for b in boxes.values())
    y1 = max(b[3] for b in boxes.values())

    def gap(a, b):
        return max(b[0] - a[2], a[0] - b[2], b[1] - a[3], a[1] - b[3], 0.0)

    nearest = [min(gap(boxes[d], boxes[e]) for e in boxes if e != d) for d in boxes]
    width, height = spec["interposer_mm"]
    centres = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes.values()]
    spread = float(np.mean([np.hypot(x - width / 2, y - height / 2) for x, y in centres]))
    cx, cy, cw, ch = spec["thermal_model"]["cooling_contact_rect_mm"]
    inside = sum(max(0.0, min(b[2], cx + cw) - max(b[0], cx)) * max(0.0, min(b[3], cy + ch) - max(b[1], cy))
                 for b in boxes.values())
    die_area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes.values())
    lengths = {k: r["route_length_mm"] for k, r in routing["routes"].items()}
    return {
        "weighted_route_gb_s_mm": routing["weighted_route_gb_s_mm"],
        "total_wire_mm": round(sum(lengths.values()), 3),
        "longest_link": max(lengths, key=lengths.get),
        "longest_link_mm": max(lengths.values()),
        "bounding_box_mm2": round((x1 - x0) * (y1 - y0), 1),
        "mean_nearest_gap_mm": round(float(np.mean(nearest)), 3),
        "min_gap_mm": round(min(nearest), 3),
        "mean_centre_distance_mm": round(spread, 2),
        "die_area_on_cooling_contact": round(inside / die_area, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layout", action="append", metavar="NAME=PATH",
                        help="layout to compare (repeatable); PATH holds placements_mm, or NAME is an example layout")
    parser.add_argument("--material", nargs="+", choices=list(MATERIALS), default=["si", "bas"])
    parser.add_argument("--grid", type=int, choices=(64, 128), default=64)
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--hotspot", type=Path, default=Path(os.environ.get("HOTSPOT_BINARY", DEFAULT_HOTSPOT)))
    parser.add_argument("--results", type=Path, default=REPO / "results")
    args = parser.parse_args()

    spec = json.loads((CHALLENGE / "challenge_spec.json").read_text())
    entries = args.layout or [
        "ring_center",
        f"best-for-si={REPO / 'results' / 'optimize_si' / 'submission.json'}",
        f"best-for-bas={REPO / 'results' / 'optimize_bas' / 'submission.json'}",
    ]
    layouts = {}
    for entry in entries:
        name, _, path = entry.partition("=")
        if path:
            layouts[name] = rounded(json.loads(Path(path).read_text())["placements_mm"])
        else:
            layouts[name] = rounded(spec["example_layouts_mm"][name])

    hotspot = args.hotspot.resolve()
    jobs = args.jobs or hotspot_cores(hotspot)
    pool = ThreadPoolExecutor(max_workers=jobs)
    evaluators = {m: Evaluator(spec, hotspot, args.grid, m, "spreader", jobs, REPO / "gui_runs" / "compare",
                               pool=pool, keep_maps=True)
                  for m in args.material}
    pairs = list(itertools.product(layouts, args.material))
    print(f"{len(pairs)} evaluations ({len(layouts)} layouts x {len(args.material)} materials), "
          f"{jobs} HotSpot solves at a time", flush=True)
    with ThreadPoolExecutor(max_workers=len(pairs)) as outer:
        results = dict(zip(pairs, outer.map(lambda p: evaluators[p[1]].evaluate(layouts[p[0]]), pairs)))

    shapes = {name: geometry(spec, layout) for name, layout in layouts.items()}
    args.results.mkdir(parents=True, exist_ok=True)

    with (args.results / "compare_layouts.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        geo_keys = list(next(iter(shapes.values())))
        writer.writerow(["layout", "material", "worst_gpu_c", "worst_hbm_c", "score", *geo_keys])
        for name, material in pairs:
            r = results[(name, material)]
            writer.writerow([name, material, r["worst_gpu_c"], r["worst_hbm_c"], r["score"],
                             *(shapes[name][k] for k in geo_keys)])

    width = max(len(n) for n in layouts) + 2
    lines = ["Worst GPU / worst HBM over all five workloads (C), grid "
             f"{args.grid}; score uses the hill's si reference.", "",
             "| layout | " + " | ".join(args.material) + " |",
             "|---|" + "---:|" * len(args.material)]
    for name in layouts:
        cells = [f"{results[(name, m)]['worst_gpu_c']:.2f} / {results[(name, m)]['worst_hbm_c']:.2f}"
                 f" (score {results[(name, m)]['score']:.4f})" for m in args.material]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "Geometry (material-independent):", "",
              "| layout | weighted route GB/s*mm | total wire mm | longest link | bounding box mm2 "
              "| mean nearest gap mm | min gap mm | mean centre distance mm | die area on contact |",
              "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for name, g in shapes.items():
        lines.append(f"| {name} | {g['weighted_route_gb_s_mm']} | {g['total_wire_mm']} | "
                     f"{g['longest_link']} {g['longest_link_mm']} | {g['bounding_box_mm2']} | "
                     f"{g['mean_nearest_gap_mm']} | {g['min_gap_mm']} | {g['mean_centre_distance_mm']} | "
                     f"{g['die_area_on_cooling_contact']:.1%} |")
    table = "\n".join(lines) + "\n"
    (args.results / "compare_layouts.md").write_text(table)
    print("\n" + table)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    w_mm, h_mm = spec["interposer_mm"]
    envelopes = {p: np.max(np.stack(list(results[p]["maps"].values())), axis=0) for p in pairs}
    lo = min(float(e.min()) for e in envelopes.values())
    hi = max(float(e.max()) for e in envelopes.values())
    fig, axes = plt.subplots(len(layouts), len(args.material), squeeze=False,
                             figsize=(5.2 * len(args.material) + 1, 3.9 * len(layouts)), constrained_layout=True)
    cx, cy, cw, ch = spec["thermal_model"]["cooling_contact_rect_mm"]
    for row, name in enumerate(layouts):
        for col, material in enumerate(args.material):
            ax = axes[row, col]
            r = results[(name, material)]
            image = ax.imshow(envelopes[(name, material)], origin="upper", extent=(0, w_mm, 0, h_mm),
                              cmap="inferno", vmin=lo, vmax=hi)
            ax.add_patch(Rectangle((cx, cy), cw, ch, fill=False, edgecolor="lime", lw=0.8, ls="--"))
            for die in layouts[name]:
                x0, y0, x1, y1 = die_box(spec, layouts[name], die)
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="cyan", lw=0.7))
                ax.text((x0 + x1) / 2, (y0 + y1) / 2, die, color="white", fontsize=7, ha="center", va="center")
            ax.set_title(f"{name} on {material}\nGPU {r['worst_gpu_c']:.2f} / HBM {r['worst_hbm_c']:.2f} C, "
                         f"route {shapes[name]['weighted_route_gb_s_mm']:.0f}", fontsize=9)
            ax.set_xlim(0, w_mm)
            ax.set_ylim(0, h_mm)
    fig.colorbar(image, ax=axes, shrink=0.6, label="hottest over 5 workloads (C)")
    fig.suptitle("Die-plane temperature envelope; dashed = cooling contact")
    fig.savefig(args.results / "compare_layouts_maps.png", dpi=110)
    plt.close(fig)
    print(f"wrote {args.results / 'compare_layouts.csv'}, compare_layouts.md, compare_layouts_maps.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
