#!/usr/bin/env python3
"""Simulated annealing over the eight die origins, minimising the hill's score.

    python scripts/optimize.py --time-only              # one evaluation, report its cost
    python scripts/optimize.py --chains 4 --iters 200   # anneal from ring_center
    python scripts/optimize.py --material bas

The score is the hill's own (verify_high_accuracy.judge):

    max((worst_gpu - ambient) / (ref_gpu - ambient),
        (worst_hbm - ambient) / (ref_hbm - ambient))

where worst_* is the peak over all five workloads and the reference is the
64-grid public baseline, 104.59 / 97.68 C. Every evaluation is the full judge
path (generate -> validate_layout -> create_hotspot_files -> HotSpot ->
parse_grid -> die_peaks), so an evaluation is five HotSpot solves.

A move either shifts one die, or swaps two dies of the same type. A proposed
layout is rounded to the submission's 0.001 mm and must pass the judge's
validate_layout (edges, die gaps, port capacity, link reach, track routing,
13,500 GB/s*mm budget) before any HotSpot time is spent; illegal proposals are
redrawn. Chain 0 starts at ring_center and the others at random legal
perturbations of it. Chains run in parallel and share one pool of HotSpot
solves sized to the cores HotSpot can use.

The best layout is saved in the hill's submission format (placements_mm only;
routes are the judge's own) and, if the hill repo is next to this one, checked
with its verify_submission.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_baseline import CHALLENGE, DEFAULT_HOTSPOT, REPO, hotspot_command  # noqa: E402
from material_sweep import hotspot_cores  # noqa: E402

from chiplet_gui.core.materials import DEFAULT_MATERIAL, MATERIALS, TARGETS, apply_material  # noqa: E402
from generate_workloads import generate  # noqa: E402
from geometry import validate_layout  # noqa: E402
from run_simulations import create_hotspot_files, die_peaks, parse_grid  # noqa: E402

#: The hill's public 64-grid reference (chiplet-thermal-simulator/reference_64.json).
REFERENCE_64 = {"worst_gpu_c": 104.59, "worst_hbm_c": 97.68}
HILL_REPO = REPO.parent / "chiplet-thermal-simulator"
DECIMALS = 3  # the submission's precision; also what the judge is fed


class Evaluator:
    """Scores layouts with HotSpot; thread-safe, cached, solves share one pool."""

    def __init__(self, spec, hotspot: Path, grid: int, material: str, target: str, jobs: int, work: Path,
                 pool: ThreadPoolExecutor | None = None, keep_maps: bool = False):
        self.spec, self.hotspot, self.vendor = spec, hotspot, hotspot.parent
        self.grid, self.material, self.target = grid, material, target
        self.portable = generate(spec)
        self.workloads = list(self.portable["workloads"])
        self.pool = pool or ThreadPoolExecutor(max_workers=jobs)
        self.keep_maps = keep_maps  # keep each workload's die-plane map in the result
        self.work = work
        self.cache: dict[tuple, dict] = {}
        self.lock = threading.Lock()
        self.evaluations = 0
        self.seconds = 0.0
        ambient = spec["thermal_model"]["ambient_c"]
        reference = REFERENCE_64 if grid == 64 else None
        reference_file = HILL_REPO / f"reference_{grid}.json"
        if reference_file.is_file():
            reference = json.loads(reference_file.read_text())
        if reference is None:
            raise SystemExit(f"no hill reference for grid {grid}")
        self.ambient, self.reference = ambient, reference

    def score(self, gpu: float, hbm: float) -> float:
        return round(max((gpu - self.ambient) / (self.reference["worst_gpu_c"] - self.ambient),
                         (hbm - self.ambient) / (self.reference["worst_hbm_c"] - self.ambient)), 8)

    def evaluate(self, layout: dict) -> dict:
        key = tuple(sorted((d, tuple(xy)) for d, xy in layout.items()))
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        started = time.perf_counter()
        routing = validate_layout(self.spec, layout)
        self.work.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="eval_", dir=self.work))
        try:
            create_hotspot_files(self.spec, self.portable, layout, routing, run_dir, self.grid, self.vendor)
            apply_material(run_dir, self.spec, self.material, self.target)

            def solve(workload: str) -> tuple[str, dict, object]:
                command, cwd = hotspot_command(self.hotspot, self.vendor, run_dir, workload)
                proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
                if proc.returncode:
                    raise RuntimeError(f"HotSpot failed for {workload}:\n{proc.stderr[-2000:]}")
                grid_c = parse_grid(run_dir / f"{workload}.grid", self.grid)
                return workload, die_peaks(self.spec, layout, grid_c), grid_c

            solved = [f.result() for f in [self.pool.submit(solve, w) for w in self.workloads]]
            peaks = {workload: die_peak for workload, die_peak, _ in solved}
            maps = {workload: grid_c for workload, _, grid_c in solved}
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)
        gpu = max(t for w in peaks.values() for d, t in w.items() if d.startswith("G"))
        hbm = max(t for w in peaks.values() for d, t in w.items() if d.startswith("H"))
        result = {"score": self.score(gpu, hbm), "worst_gpu_c": gpu, "worst_hbm_c": hbm,
                  "weighted_route_gb_s_mm": routing["weighted_route_gb_s_mm"], "per_workload_die_peak_c": peaks}
        if self.keep_maps:
            result["maps"] = maps
        elapsed = time.perf_counter() - started
        with self.lock:
            self.cache[key] = result
            self.evaluations += 1
            self.seconds += elapsed
        return result


def rounded(layout: dict) -> dict:
    return {d: [round(float(x), DECIMALS), round(float(y), DECIMALS)] for d, (x, y) in layout.items()}


def is_legal(spec: dict, layout: dict) -> bool:
    try:
        validate_layout(spec, layout)
        return True
    except ValueError:
        return False


def propose(spec: dict, layout: dict, rng: random.Random, step_mm: float, swap_p: float, tries: int = 400) -> dict | None:
    """A legal neighbour of *layout*: shift one die, or swap two of the same type."""

    kinds: dict[str, list[str]] = {}
    for die, entry in spec["dies"].items():
        kinds.setdefault(entry.get("kind", die[0]), []).append(die)
    for _ in range(tries):
        candidate = {d: list(xy) for d, xy in layout.items()}
        if rng.random() < swap_p:
            a, b = rng.sample(rng.choice(list(kinds.values())), 2)
            candidate[a], candidate[b] = candidate[b], candidate[a]
        else:
            die = rng.choice(list(candidate))
            candidate[die] = [candidate[die][0] + rng.gauss(0, step_mm), candidate[die][1] + rng.gauss(0, step_mm)]
        candidate = rounded(candidate)
        if candidate != layout and is_legal(spec, candidate):
            return candidate
    return None


class Log:
    def __init__(self, path: Path):
        self.path, self.lock, self.started = path, threading.Lock(), time.perf_counter()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8")

    def __call__(self, text: str, record: dict | None = None) -> None:
        stamp = f"[{time.perf_counter() - self.started:7.0f} s] "
        with self.lock:
            print(stamp + text, flush=True)
            if record is not None:
                self.handle.write(json.dumps(record) + "\n")
                self.handle.flush()


def anneal(chain: int, start: dict, evaluator: Evaluator, args, log: Log, best: dict, best_lock: threading.Lock) -> dict:
    rng = random.Random(args.seed * 1000 + chain)
    spec = evaluator.spec
    current = start
    current_result = evaluator.evaluate(current)
    log(f"chain {chain} start score {current_result['score']:.5f} "
        f"(GPU {current_result['worst_gpu_c']:.2f}, HBM {current_result['worst_hbm_c']:.2f})",
        {"chain": chain, "iter": 0, "accepted": True, "layout": current, **_summary(current_result)})
    chain_best, chain_best_result = current, current_result
    with best_lock:
        if current_result["score"] < best["result"]["score"]:
            best.update(layout=current, result=current_result, chain=chain, iter=0)
            save_best(args, evaluator, best)
    for step in range(1, args.iters + 1):
        frac = step / args.iters
        temperature = args.t0 * (args.t_end / args.t0) ** frac
        step_mm = args.step_mm * (args.step_end_mm / args.step_mm) ** frac
        candidate = propose(spec, current, rng, step_mm, args.swap_p)
        if candidate is None:
            log(f"chain {chain} iter {step}: no legal neighbour found, stopping")
            break
        result = evaluator.evaluate(candidate)
        delta = result["score"] - current_result["score"]
        accepted = delta <= 0 or rng.random() < math.exp(-delta / temperature)
        if accepted:
            current, current_result = candidate, result
        if result["score"] < chain_best_result["score"]:
            chain_best, chain_best_result = candidate, result
        with best_lock:
            improved = result["score"] < best["result"]["score"]
            if improved:
                best.update(layout=candidate, result=result, chain=chain, iter=step)
                save_best(args, evaluator, best)
        log(f"chain {chain} iter {step:4d}/{args.iters} T={temperature:.5f} step={step_mm:.2f} mm "
            f"score {result['score']:.5f} ({'acc' if accepted else 'rej'}) "
            f"current {current_result['score']:.5f} best {chain_best_result['score']:.5f}"
            + ("  << new global best" if improved else ""),
            {"chain": chain, "iter": step, "temperature": temperature, "accepted": accepted,
             "layout": candidate, **_summary(result)})
    return {"chain": chain, "layout": chain_best, "result": chain_best_result}


def _summary(result: dict) -> dict:
    return {k: result[k] for k in ("score", "worst_gpu_c", "worst_hbm_c", "weighted_route_gb_s_mm")}


def save_best(args, evaluator: Evaluator, best: dict) -> None:
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"placements_mm": best["layout"]}, indent=2) + "\n")
    out.with_suffix(".meta.json").write_text(json.dumps({
        "material": evaluator.material, "material_target": evaluator.target, "grid": evaluator.grid,
        "reference_c": {k: evaluator.reference[k] for k in ("worst_gpu_c", "worst_hbm_c")},
        "hill_comparable": evaluator.material == DEFAULT_MATERIAL,
        "found_by_chain": best["chain"], "iteration": best["iter"], **best["result"]}, indent=2) + "\n")


def hill_verify(path: Path) -> str:
    """Run the hill's own feasibility gate on the saved submission, if available."""

    script = HILL_REPO / "verify_submission.py"
    if not script.is_file():
        return f"(hill repo not found at {HILL_REPO}; skipped verify_submission)"
    proc = subprocess.run([sys.executable, str(script), str(path)], capture_output=True, text=True, cwd=HILL_REPO)
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return f"verify_submission failed to run: {proc.stderr[-500:]}"
    return "hill verify_submission: " + ("passed" if report.get("passed") else f"FAILED: {report.get('reason')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--material", choices=list(MATERIALS), default=DEFAULT_MATERIAL)
    parser.add_argument("--material-target", choices=TARGETS, default="spreader")
    parser.add_argument("--start", default="ring_center", help="example layout to start from")
    parser.add_argument("--chains", type=int, default=4, help="parallel annealing chains (random starts)")
    parser.add_argument("--iters", type=int, default=150, help="evaluations per chain after its start")
    parser.add_argument("--t0", type=float, default=0.01, help="initial temperature, in score units")
    parser.add_argument("--t-end", type=float, default=0.0005, help="final temperature")
    parser.add_argument("--step-mm", type=float, default=1.5, help="initial shift standard deviation")
    parser.add_argument("--step-end-mm", type=float, default=0.2, help="final shift standard deviation")
    parser.add_argument("--swap-p", type=float, default=0.15, help="probability a move is a same-type swap")
    parser.add_argument("--start-moves", type=int, default=6, help="legal random moves to perturb chains 1..n")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--grid", type=int, choices=(64, 128), default=64)
    parser.add_argument("--jobs", type=int, help="parallel HotSpot solves (default: cores HotSpot can use)")
    parser.add_argument("--hotspot", type=Path, default=Path(os.environ.get("HOTSPOT_BINARY", DEFAULT_HOTSPOT)))
    parser.add_argument("--out", type=Path, help="best layout (default: results/optimize_<material>/submission.json)")
    parser.add_argument("--time-only", action="store_true", help="evaluate the start layout once, report timing, exit")
    args = parser.parse_args()

    hotspot = args.hotspot.resolve()
    if not hotspot.is_file() or not (hotspot.parent / "examples/example3/example.config").is_file():
        parser.error(f"{hotspot} is not a HotSpot build inside its own repository checkout")
    tag = args.material + ("_interposer" if args.material != DEFAULT_MATERIAL and args.material_target == "interposer" else "")
    args.out = (args.out or REPO / "results" / f"optimize_{tag}" / "submission.json").resolve()
    jobs = args.jobs or hotspot_cores(hotspot)

    spec = json.loads((CHALLENGE / "challenge_spec.json").read_text())
    start = rounded(spec["example_layouts_mm"][args.start])
    if not is_legal(spec, start):
        raise SystemExit(f"{args.start} is not legal")
    evaluator = Evaluator(spec, hotspot, args.grid, args.material, args.material_target, jobs,
                          REPO / "gui_runs" / f"optimize_{tag}")
    log = Log(args.out.parent / "progress.jsonl")
    log(f"material {tag}, grid {args.grid}, {jobs} HotSpot solves at a time, reference "
        f"{evaluator.reference['worst_gpu_c']} / {evaluator.reference['worst_hbm_c']} C")

    if args.time_only:
        t = time.perf_counter()
        result = evaluator.evaluate(start)
        seconds = time.perf_counter() - t
        evals = args.chains * (args.iters + 1)
        # An evaluation keeps min(5, jobs) cores busy for `seconds`; a long run
        # keeps all `jobs` busy, so its wall time is total core-time over jobs.
        hours = evals * seconds * min(5, jobs) / jobs / 3600
        log(f"{args.start}: score {result['score']:.5f}, GPU {result['worst_gpu_c']:.2f}, HBM {result['worst_hbm_c']:.2f}")
        log(f"one evaluation (5 HotSpot solves, {jobs} at a time): {seconds:.0f} s wall, "
            f"{seconds * min(5, jobs):.0f} core-seconds")
        log(f"{args.chains} chains x {args.iters + 1} evaluations = {evals} evaluations "
            f"= about {hours:.1f} h on {jobs} cores")
        return 0

    starts = [start]
    rng = random.Random(args.seed)
    while len(starts) < args.chains:
        layout = start
        for _ in range(args.start_moves):
            layout = propose(spec, layout, rng, args.step_mm, args.swap_p) or layout
        starts.append(layout)

    best: dict = {"layout": start, "result": {"score": math.inf}, "chain": None, "iter": None}
    best_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.chains) as chains:
        finals = list(chains.map(lambda c: anneal(c, starts[c], evaluator, args, log, best, best_lock),
                                 range(args.chains)))

    log("chain results: " + ", ".join(f"{f['chain']}: {f['result']['score']:.5f}" for f in finals))
    log(f"best score {best['result']['score']:.5f} (GPU {best['result']['worst_gpu_c']:.2f}, "
        f"HBM {best['result']['worst_hbm_c']:.2f}) from chain {best['chain']} iter {best['iter']}")
    log(f"{evaluator.evaluations} HotSpot evaluations, {evaluator.seconds / max(1, evaluator.evaluations):.0f} s each "
        f"on average; saved {args.out}")
    log(hill_verify(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
