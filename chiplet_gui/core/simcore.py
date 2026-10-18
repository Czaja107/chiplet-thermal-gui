"""Locate the challenge package and import the judge's own modules.

The GUI never reimplements the physics. It puts the challenge directory on
``sys.path`` and imports ``geometry``, ``interconnect``, ``generate_workloads``
and ``run_simulations`` from it, so placement legality, link heat, HotSpot input
files and peak-temperature extraction are computed by the same code the judge
uses. Only presentation lives in :mod:`chiplet_gui.gui`.
"""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

#: Directory names that have held the challenge package in practice.
_CANDIDATE_DIRS = ("chiplet_thermal_continuous", "challenge", ".")

#: Files that must all exist for a directory to be the challenge package.
_REQUIRED = ("challenge_spec.json", "geometry.py", "interconnect.py", "run_simulations.py")


class SimulatorNotFound(RuntimeError):
    """Raised when the challenge package cannot be located."""


def _is_challenge_dir(path: Path) -> bool:
    return all((path / name).is_file() for name in _REQUIRED)


def find_challenge_dir(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Return the directory holding ``challenge_spec.json`` and the judge modules.

    Search order: an explicit path, the ``CHIPLET_CHALLENGE_DIR`` environment
    variable, then the usual subdirectories of this repository and of its
    parent. The parent is included so the GUI also works when it is dropped
    next to an already-extracted challenge archive.
    """

    tried: list[Path] = []
    roots = [Path(__file__).resolve().parents[2]]
    roots.append(roots[0].parent)

    def consider(path: Path) -> Path | None:
        path = path.expanduser().resolve()
        tried.append(path)
        return path if _is_challenge_dir(path) else None

    if explicit:
        found = consider(Path(explicit))
        if found:
            return found
        raise SimulatorNotFound(
            f"{Path(explicit).resolve()} does not contain the challenge package "
            f"(expected {', '.join(_REQUIRED)})."
        )

    from_env = os.environ.get("CHIPLET_CHALLENGE_DIR")
    if from_env:
        found = consider(Path(from_env))
        if found:
            return found

    for root in roots:
        for name in _CANDIDATE_DIRS:
            found = consider(root / name)
            if found:
                return found

    listing = "\n".join(f"  {path}" for path in dict.fromkeys(tried))
    raise SimulatorNotFound(
        "Could not find the chiplet challenge package. It must contain "
        f"{', '.join(_REQUIRED)}.\nLooked in:\n{listing}\n"
        "Set the CHIPLET_CHALLENGE_DIR environment variable to its location."
    )


#: The challenge directory the judge modules were first imported from. Later
#: calls without an explicit directory reuse it, so ``--challenge-dir`` given at
#: start-up governs every spec read, not only the first.
_active_dir: Path | None = None


def load_modules(challenge_dir: str | os.PathLike[str] | None = None) -> SimModules:
    """Import the judge modules once and return them as a bundle."""

    global _active_dir
    if challenge_dir is None and _active_dir is not None:
        directory = _active_dir
    else:
        directory = find_challenge_dir(challenge_dir)
        if _active_dir is None:
            _active_dir = directory
    return _load_from(directory)


@lru_cache(maxsize=4)
def _load_from(directory: Path) -> SimModules:
    text = str(directory)
    if text not in sys.path:
        # Prepend: the judge modules import each other by bare name.
        sys.path.insert(0, text)
    import generate_workloads
    import geometry
    import interconnect
    import run_simulations

    return SimModules(
        directory=directory,
        geometry=geometry,
        interconnect=interconnect,
        generate_workloads=generate_workloads,
        run_simulations=run_simulations,
    )


class SimModules:
    """The judge modules plus the directory they were imported from."""

    def __init__(
        self,
        directory: Path,
        geometry: ModuleType,
        interconnect: ModuleType,
        generate_workloads: ModuleType,
        run_simulations: ModuleType,
    ) -> None:
        self.directory = directory
        self.geometry = geometry
        self.interconnect = interconnect
        self.generate_workloads = generate_workloads
        self.run_simulations = run_simulations

    @property
    def spec_path(self) -> Path:
        return self.directory / "challenge_spec.json"

    def load_spec(self) -> dict:
        """The authoritative, unmodified challenge specification."""

        return json.loads(self.spec_path.read_text())


def default_spec(challenge_dir: str | None = None) -> dict:
    """Convenience wrapper returning a fresh copy of the challenge spec."""

    return load_modules(challenge_dir).load_spec()


#: Paths worth probing for a compiled HotSpot executable, relative to a root.
_HOTSPOT_RELATIVE = (
    "hotspot.exe",
    "hotspot",
    "HotSpot-master/hotspot.exe",
    "HotSpot-master/hotspot",
    "HotSpot/hotspot.exe",
    "HotSpot/hotspot",
    "vendor/HotSpot/hotspot.exe",
    "vendor/HotSpot/hotspot",
)


def is_hotspot_binary(path: str | os.PathLike[str] | None) -> bool:
    """True when *path* looks like a HotSpot build inside its own repository.

    ``run_simulations.create_hotspot_files`` reads
    ``examples/example3/example.config`` next to the binary, so a bare
    executable copied out of its repository is not usable.
    """

    if not path:
        return False
    binary = Path(path)
    return binary.is_file() and (binary.parent / "examples/example3/example.config").is_file()


def find_hotspot_binary(extra_roots: tuple[Path, ...] = ()) -> Path | None:
    """Best-effort search for a usable HotSpot executable.

    Checks ``HOTSPOT_BINARY``, then this repository, its parent and any extra
    roots given. Returns ``None`` rather than raising: the GUI starts fine
    without HotSpot and only needs it to run simulations.
    """

    from_env = os.environ.get("HOTSPOT_BINARY")
    if is_hotspot_binary(from_env):
        return Path(from_env).resolve()

    here = Path(__file__).resolve().parents[2]
    roots = [here, here.parent, *extra_roots]
    for root in roots:
        for relative in _HOTSPOT_RELATIVE:
            candidate = root / relative
            if is_hotspot_binary(candidate):
                return candidate.resolve()
    return None


__all__ = [
    "SimModules",
    "SimulatorNotFound",
    "default_spec",
    "find_challenge_dir",
    "find_hotspot_binary",
    "is_hotspot_binary",
    "load_modules",
]
