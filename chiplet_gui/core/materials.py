"""Optional thermal-stack material variants, applied on top of the judge's files.

The judge's ``create_hotspot_files`` always writes the challenge stack:

    layer 0  silicon interposer      interposer.flp   (power: link heat)
    layer 1  active die plane        chiplets.flp     (power: dies)
    layer 2  TIM / cooling contact   tim.flp          (no power)

HotSpot attaches its package (spreader, sink, convection) to the *last* layer,
so layer 2 faces the cooler. A variant edits the generated files afterwards,
leaving the challenge package itself untouched:

* ``target="spreader"`` inserts a full-footprint spreader layer between the die
  tops and the TIM, with lateral heat flow on, so heat goes
  dies -> spreader -> TIM contact -> cooler.
* ``target="interposer"`` keeps the three layers and swaps only the interposer's
  material, at the interposer's own thickness, for comparison.

``si`` is the unmodified challenge stack and writes nothing, so results are
byte-for-byte those of the judge. Any other variant is a deviation from the
spec and is not leaderboard-comparable.

HotSpot models every layer as isotropic, so hBN, which conducts very
differently in-plane and through-plane, is bracketed by two runs: one with the
in-plane value (best case) and one with the through-plane value (worst case).
Heat capacity only matters for transient runs; these steady-state solves do not
use it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Material:
    name: str
    label: str
    k_w_per_mk: float
    spreader_thickness_mm: float
    heat_capacity_j_per_m3k: float

    @property
    def resistivity_mk_per_w(self) -> float:
        return 1.0 / self.k_w_per_mk


#: ``si`` is listed for completeness; it is never written into a stack.
MATERIALS: dict[str, Material] = {
    "si": Material("si", "silicon (challenge stack, unchanged)", 130.0, 0.0, 1.75e6),
    "bas": Material("bas", "boron arsenide", 1300.0, 0.5, 2.1e6),
    "hbn_best": Material("hbn_best", "hBN, in-plane k (best case)", 400.0, 0.3, 1.6e6),
    "hbn_worst": Material("hbn_worst", "hBN, through-plane k (worst case)", 3.0, 0.3, 1.6e6),
}
DEFAULT_MATERIAL = "si"
TARGETS = ("spreader", "interposer")

SPREADER_FLP = "spreader.flp"

#: Lines per layer in an LCF file, after comments and blank lines are dropped.
_FIELDS = ("number", "lateral", "power", "heat_capacity", "resistivity", "thickness", "floorplan")


def _read_layers(text: str) -> list[dict[str, str]]:
    values = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if len(values) % len(_FIELDS):
        raise ValueError("stack.lcf does not hold a whole number of layers")
    return [dict(zip(_FIELDS, values[i : i + len(_FIELDS)])) for i in range(0, len(values), len(_FIELDS))]


def _write_layers(layers: list[dict[str, str]], comment: str) -> str:
    lines = [f"# {comment}"]
    for number, layer in enumerate(layers):
        lines += [str(number), *(layer[field] for field in _FIELDS[1:])]
    return "\n".join(lines) + "\n"


def apply_material(
    run_dir: str | Path,
    spec: dict,
    material: str = DEFAULT_MATERIAL,
    target: str = "spreader",
) -> str:
    """Rewrite the HotSpot inputs in *run_dir* for *material*; return a description.

    Call after ``create_hotspot_files``. The die plane stays layer 1 in every
    variant, so ``parse_grid`` and ``die_peaks`` read the right layer unchanged.
    """

    if material not in MATERIALS:
        raise ValueError(f"unknown material {material!r}; choose from {', '.join(MATERIALS)}")
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; choose from {', '.join(TARGETS)}")
    chosen = MATERIALS[material]
    if material == DEFAULT_MATERIAL:
        return "si: challenge stack, unchanged"

    run_dir = Path(run_dir)
    stack = run_dir / "stack.lcf"
    layers = _read_layers(stack.read_text())
    if [layer["floorplan"] for layer in layers] != ["interposer.flp", "chiplets.flp", "tim.flp"]:
        raise ValueError("stack.lcf is not the judge's three-layer stack; apply a variant only once")
    resistivity = f"{chosen.resistivity_mk_per_w:.9g}"
    capacity = f"{chosen.heat_capacity_j_per_m3k:.9g}"

    if target == "interposer":
        layers[0].update(heat_capacity=capacity, resistivity=resistivity)
        stack.write_text(_write_layers(
            layers,
            f"LCF order: {chosen.label} interposer (k={chosen.k_w_per_mk:g} W/mK, "
            "judge thickness), active die plane, TIM to top heat sink",
        ))
        return f"{material}: interposer material swapped to k={chosen.k_w_per_mk:g} W/mK"

    width_mm, height_mm = spec["interposer_mm"]
    # One block over the whole footprint; it takes the layer's own resistivity.
    (run_dir / SPREADER_FLP).write_text(f"spreader\t{width_mm / 1000}\t{height_mm / 1000}\t0.0\t0.0\n")
    spreader = {
        "lateral": "Y",
        "power": "N",
        "heat_capacity": capacity,
        "resistivity": resistivity,
        "thickness": f"{chosen.spreader_thickness_mm * 1e-3:.9g}",
        "floorplan": SPREADER_FLP,
    }
    layers.insert(2, spreader)
    stack.write_text(_write_layers(
        layers,
        f"LCF order: Si interposer, active die plane, {chosen.label} spreader "
        f"({chosen.spreader_thickness_mm:g} mm, k={chosen.k_w_per_mk:g} W/mK), TIM to top heat sink",
    ))
    return (
        f"{material}: {chosen.spreader_thickness_mm:g} mm spreader, k={chosen.k_w_per_mk:g} W/mK, "
        "between die tops and TIM"
    )


__all__ = ["DEFAULT_MATERIAL", "MATERIALS", "Material", "TARGETS", "apply_material"]
