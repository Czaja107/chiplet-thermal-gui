#!/usr/bin/env python3
"""Live demo: ring_center vs our optimized layout, on silicon and boron arsenide.

    python demo.py --prepare     # once, beforehand: runs HotSpot (minutes), fills the cache
    python demo.py               # live, well under 30 s: verify, then show

HotSpot needs minutes per layout, so the temperatures are computed beforehand
by --prepare. The live run does not just replay them: it re-checks each layout
against the judge's legality rules, regenerates every HotSpot input file from
scratch with the judge's own code, and refuses to show a result whose input
fingerprint differs from the one the cached temperatures were solved from.
What is on screen is therefore exactly what HotSpot computes for these
layouts under this code.

Shows three die-plane maps, each the hottest every cell gets over the five
workloads, on one colour scale with the worst GPU and HBM labelled:
    1. ring_center on silicon     (the organizers' reference layout)
    2. optimized layout on silicon
    3. optimized layout on boron arsenide (0.5 mm spreader, k = 1300 W/mK)
and prints the hBN spreader as a best-to-worst range.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

T0 = time.perf_counter()
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "scripts"))

from optimize import Evaluator, rounded  # noqa: E402  (puts the judge on sys.path)
from run_baseline import CHALLENGE, DEFAULT_HOTSPOT  # noqa: E402

import numpy as np  # noqa: E402

from chiplet_gui.core.materials import MATERIALS, apply_material  # noqa: E402
from generate_workloads import generate  # noqa: E402
from geometry import die_box, validate_layout  # noqa: E402
from run_simulations import create_hotspot_files  # noqa: E402

CACHE = REPO / "results" / "demo_cache"
OPTIMIZED = REPO / "results" / "optimize_si" / "submission.json"
INPUT_FILES = ("chiplets.flp", "interposer.flp", "tim.flp", "stack.lcf", "benchmark.config")

#: (cache key, layout name, material, panel title) for the three panels.
PANELS = [
    ("ring_center__si", "ring_center", "si", "ring_center on silicon"),
    ("optimized__si", "optimized", "si", "optimized on silicon"),
    ("optimized__bas", "optimized", "bas", "optimized on boron arsenide"),
]
HBN = [("optimized__hbn_best", "optimized", "hbn_best"), ("optimized__hbn_worst", "optimized", "hbn_worst")]


def load_layouts(spec: dict, optimized: Path) -> dict:
    layouts = {"ring_center": rounded(spec["example_layouts_mm"]["ring_center"])}
    if optimized.is_file():
        layouts["optimized"] = rounded(json.loads(optimized.read_text())["placements_mm"])
    return layouts


def fingerprint(spec: dict, portable: dict, layout: dict, material: str, vendor: Path) -> str:
    """SHA-256 of every HotSpot input file the judge writes for this layout and material."""

    routing = validate_layout(spec, layout)
    run_dir = Path(tempfile.mkdtemp(prefix="demo_"))
    try:
        create_hotspot_files(spec, portable, layout, routing, run_dir, 64, vendor)
        apply_material(run_dir, spec, material)
        digest = hashlib.sha256()
        names = [*INPUT_FILES, *sorted(p.name for p in run_dir.glob("*.ptrace"))]
        if (run_dir / "spreader.flp").is_file():
            names.append("spreader.flp")
        for name in names:
            digest.update(name.encode() + b"\0" + (run_dir / name).read_bytes())
        return digest.hexdigest()
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def prepare(spec: dict, layouts: dict, hotspot: Path) -> None:
    if "optimized" not in layouts:
        raise SystemExit(f"no optimized layout at {OPTIMIZED}; run scripts/optimize.py first")
    portable = generate(spec)
    pairs = [(key, name, material) for key, name, material, _ in PANELS] + HBN
    evaluators = {}
    CACHE.mkdir(parents=True, exist_ok=True)
    for key, name, material in pairs:
        started = time.perf_counter()
        evaluators.setdefault(material, Evaluator(spec, hotspot, 64, material, "spreader", 5,
                                                  REPO / "gui_runs" / "demo", keep_maps=True))
        result = evaluators[material].evaluate(layouts[name])
        np.savez_compressed(CACHE / f"{key}.npz", **result["maps"])
        (CACHE / f"{key}.json").write_text(json.dumps({
            "layout": layouts[name], "material": material,
            "fingerprint": fingerprint(spec, portable, layouts[name], material, hotspot.parent),
            "hotspot_sha256": hashlib.sha256(hotspot.read_bytes()).hexdigest(),
            **{k: result[k] for k in ("score", "worst_gpu_c", "worst_hbm_c", "weighted_route_gb_s_mm",
                                      "per_workload_die_peak_c")}}, indent=2) + "\n")
        print(f"  cached {key}: GPU {result['worst_gpu_c']:.2f}, HBM {result['worst_hbm_c']:.2f} C "
              f"({time.perf_counter() - started:.0f} s)", flush=True)
    print(f"cache ready in {CACHE}")


def worst(meta: dict, kind: str) -> tuple[float, str]:
    best = max(((t, w) for w, dies in meta["per_workload_die_peak_c"].items()
                for d, t in dies.items() if d.startswith(kind)))
    return best


def show(spec: dict, layouts: dict, hotspot: Path, save: Path | None, display: bool) -> int:
    portable = generate(spec)
    vendor = hotspot.parent if hotspot.is_file() else DEFAULT_HOTSPOT.parent
    metas, maps = {}, {}
    print("verifying cached HotSpot results against freshly generated inputs:")
    for key, name, material in [(k, n, m) for k, n, m, _ in PANELS] + HBN:
        meta_path = CACHE / f"{key}.json"
        if not meta_path.is_file() or name not in layouts:
            raise SystemExit(f"  missing {key}; run `python demo.py --prepare` first")
        meta = json.loads(meta_path.read_text())
        if meta["layout"] != layouts[name]:
            raise SystemExit(f"  {key}: the {name} layout changed since --prepare; run it again")
        if fingerprint(spec, portable, layouts[name], material, vendor) != meta["fingerprint"]:
            raise SystemExit(f"  {key}: HotSpot inputs differ from the cached run; run --prepare again")
        metas[key] = meta
        maps[key] = np.load(CACHE / f"{key}.npz")
        print(f"  {key:22s} legal, inputs match (route {meta['weighted_route_gb_s_mm']:.0f} GB/s*mm)  ok",
              flush=True)

    base = metas["ring_center__si"]
    print(f"\n{'':32s} {'worst GPU':>22s} {'worst HBM':>22s}")
    for key, _, _, title in PANELS:
        (g, gw), (h, hw) = worst(metas[key], "G"), worst(metas[key], "H")
        delta = "" if key == "ring_center__si" else \
            f"   ({g - base['worst_gpu_c']:+.2f} / {h - base['worst_hbm_c']:+.2f} C vs ring_center)"
        print(f"{title:32s} {g:7.2f} C {gw:>14s} {h:7.2f} C {hw:>14s}{delta}")
    best, poor = metas["optimized__hbn_best"], metas["optimized__hbn_worst"]
    print(f"\nhBN spreader (0.3 mm), optimized layout: worst GPU {best['worst_gpu_c']:.1f}-{poor['worst_gpu_c']:.1f} C, "
          f"worst HBM {best['worst_hbm_c']:.1f}-{poor['worst_hbm_c']:.1f} C")
    print(f"  caveat: HotSpot layers are isotropic, so hBN is bracketed by its in-plane "
          f"(k={MATERIALS['hbn_best'].k_w_per_mk:g}) and through-plane (k={MATERIALS['hbn_worst'].k_w_per_mk:g} W/mK) "
          "values; the real anisotropic result lies in between.")

    import matplotlib

    if not display:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    w_mm, h_mm = spec["interposer_mm"]
    envelopes = {k: np.max(np.stack([maps[k][w] for w in maps[k].files]), axis=0) for k, *_ in PANELS}
    lo = min(float(e.min()) for e in envelopes.values())
    hi = max(float(e.max()) for e in envelopes.values())
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), constrained_layout=True)
    cx, cy, cw, ch = spec["thermal_model"]["cooling_contact_rect_mm"]
    for ax, (key, name, material, title) in zip(axes, PANELS):
        meta = metas[key]
        image = ax.imshow(envelopes[key], origin="upper", extent=(0, w_mm, 0, h_mm),
                          cmap="inferno", vmin=lo, vmax=hi)
        ax.add_patch(Rectangle((cx, cy), cw, ch, fill=False, edgecolor="lime", lw=0.9, ls="--"))
        (g, gw), (h, hw) = worst(meta, "G"), worst(meta, "H")
        for die in layouts[name]:
            x0, y0, x1, y1 = die_box(spec, layouts[name], die)
            peak = max(meta["per_workload_die_peak_c"][w][die] for w in meta["per_workload_die_peak_c"])
            hottest = peak in (g, h)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor="white" if hottest else "cyan", lw=2.0 if hottest else 0.8))
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, f"{die}\n{peak:.1f}", color="white", fontsize=8,
                    ha="center", va="center", fontweight="bold" if hottest else "normal")
        ax.set_title(f"{title}\nworst GPU {g:.2f} C ({gw})\nworst HBM {h:.2f} C ({hw})", fontsize=11)
        ax.set_xlim(0, w_mm)
        ax.set_ylim(0, h_mm)
        ax.set_xlabel("mm")
    fig.colorbar(image, ax=axes, shrink=0.85, label="hottest over 5 workloads (C)")
    fig.suptitle("Die-plane temperature, HotSpot 64x64 via the judge's own code  "
                 "(dashed = cooling contact, bold = hottest die)", fontsize=12)
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=110)
        print(f"\nsaved {save}")
    print(f"ready in {time.perf_counter() - T0:.1f} s")
    if display:
        plt.show()
    return 0


def main() -> int:
    global CACHE
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepare", action="store_true", help="run HotSpot and fill the cache (slow)")
    parser.add_argument("--optimized", type=Path, default=OPTIMIZED, help="optimized layout (submission JSON)")
    parser.add_argument("--hotspot", type=Path, default=Path(os.environ.get("HOTSPOT_BINARY", DEFAULT_HOTSPOT)))
    parser.add_argument("--save", type=Path, default=REPO / "results" / "demo.png", help="also save the figure")
    parser.add_argument("--no-window", action="store_true", help="do not open a window (for headless checks)")
    parser.add_argument("--cache", type=Path, default=CACHE, help="where --prepare stores HotSpot results")
    args = parser.parse_args()
    CACHE = args.cache

    spec = json.loads((CHALLENGE / "challenge_spec.json").read_text())
    layouts = load_layouts(spec, args.optimized)
    if args.prepare:
        prepare(spec, layouts, args.hotspot.resolve())
        return 0
    return show(spec, layouts, args.hotspot.resolve(), args.save, not args.no_window)


if __name__ == "__main__":
    raise SystemExit(main())
