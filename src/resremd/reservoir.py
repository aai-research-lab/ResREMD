"""The reservoir: stored structures with known statistical weight.

On disk a reservoir is a directory:

    reservoir.json   what it is: kind, temperature, ensemble, atoms, origin
    positions.npy    (frames, atoms, 3) float32, nm
    box.npy          (frames, 3, 3) float64, nm; absent for a non-periodic system
    weights.npy      (frames,) for a weighted reservoir
    topology.pdb     the atoms, with the first frame's coordinates
    energies/        potential energies cached per System and platform

Positions stay on disk and are memory-mapped, so a reservoir larger than
memory costs only the frames that are drawn.

The energies used in the exchange criterion are computed here, with the
run's own System, on the run's own platform and precision, from exactly the
coordinates that will be injected. A cached set is used only when all of
those match; otherwise it is recomputed, never reused.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .errors import ReservoirError
from .thermo import Ensemble, beta, box_volume_and_area

logger = logging.getLogger("resremd")

FORMAT = 1
KINDS = ("boltzmann", "weighted", "non_boltzmann")

#: Spread (kT_R) of recomputed less recorded reservoir energies above which
#: a run warns, and above which it refuses. 0.4 kT already biases the double
#: well's populations by 0.07 at 300 K (tests/test_exactness.py), so the
#: refusal sits well below that; precision noise is far smaller.
#: Both grow as the square root of the atom count past 10,000 atoms, as
#: precision noise does, and the refusal never exceeds 0.5 kT.
HAMILTONIAN_WARN_KT = 0.03
HAMILTONIAN_REFUSE_KT = 0.1
HAMILTONIAN_REFUSE_MAX_KT = 0.5


def _noise_scale(n_atoms: int) -> float:
    return max(1.0, float(np.sqrt(n_atoms / 1e4)))


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_bytes(Path(path), (json.dumps(payload, indent=2,
                                                sort_keys=False) + "\n")
                        .encode())


def _sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


class ReservoirWriter:
    """Fills a reservoir frame by frame, and can be reopened to continue."""

    def __init__(self, path: str | Path, *, n_frames: int, n_atoms: int,
                 periodic: bool, reopen: bool = False) -> None:
        from numpy.lib.format import open_memmap

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        mode = "r+" if reopen else "w+"
        self.positions = open_memmap(self.path / "positions.npy", mode=mode,
                                     dtype=np.float32,
                                     shape=(n_frames, n_atoms, 3))
        self.box = None
        if periodic:
            self.box = open_memmap(self.path / "box.npy", mode=mode,
                                   dtype=np.float64, shape=(n_frames, 3, 3))

    def write(self, index: int, positions: np.ndarray,
              box: np.ndarray | None) -> None:
        self.positions[index] = positions
        if self.box is not None:
            self.box[index] = box

    def flush(self) -> None:
        self.positions.flush()
        if self.box is not None:
            self.box.flush()


def write_topology(path: Path, topology: Any, positions: np.ndarray,
                   box: np.ndarray | None = None) -> None:
    """The reservoir's atoms as a PDB file, with the box if it has one."""
    from openmm import app

    from .system import subset_topology

    full = subset_topology(topology, np.arange(topology.getNumAtoms()), box)
    with open(path, "w") as fh:
        app.PDBFile.writeFile(full, positions * 10.0, fh, keepIds=True)


class Reservoir:
    """An opened reservoir, checked for internal consistency."""

    def __init__(self, path: Path, meta: dict[str, Any]) -> None:
        self.path = path
        self.meta = meta
        self.positions = np.load(path / "positions.npy", mmap_mode="r")
        box_file = path / "box.npy"
        self.box = np.load(box_file, mmap_mode="r") if box_file.exists() else None
        weights_file = path / "weights.npy"
        self.weights = np.load(weights_file) if weights_file.exists() else None
        self._cumulative = None
        self._digest: str | None = None
        #: Kish effective frames when this run reweighted the reservoir to
        #: its own Hamiltonian, else None.
        self.reweighted: float | None = None
        self._validate()

    # -- opening ----------------------------------------------------------
    @classmethod
    def open(cls, path: str | Path) -> "Reservoir":
        path = Path(path)
        meta_file = path / "reservoir.json"
        if not meta_file.is_file():
            raise ReservoirError(
                f"{path} is not a reservoir: it has no reservoir.json.",
                code="resremd.reservoir.missing")
        meta = json.loads(meta_file.read_text())
        if meta.get("format") != FORMAT:
            raise ReservoirError(
                f"{path} is reservoir format {meta.get('format')!r}; this "
                f"version reads format {FORMAT}.",
                code="resremd.reservoir.format")
        if not meta.get("complete", False):
            raise ReservoirError(
                f"{path} was not finished: its reservoir.json does not say "
                "complete. Resume the build that made it.",
                code="resremd.reservoir.incomplete")
        return cls(path, meta)

    def _validate(self) -> None:
        m = self.meta
        kind = m.get("kind")
        if kind not in KINDS:
            raise ReservoirError(f"Unknown reservoir kind {kind!r}.",
                                 code="resremd.reservoir.kind")
        f, n = int(m["n_frames"]), int(m["n_atoms"])
        if self.positions.shape != (f, n, 3):
            raise ReservoirError(
                f"positions.npy has shape {self.positions.shape}; "
                f"reservoir.json says {f} frames of {n} atoms.",
                code="resremd.reservoir.shape")
        if bool(m["periodic"]) != (self.box is not None):
            raise ReservoirError(
                "reservoir.json and box.npy disagree about whether the "
                "system is periodic.", code="resremd.reservoir.shape")
        if self.box is not None and self.box.shape != (f, 3, 3):
            raise ReservoirError(f"box.npy has shape {self.box.shape}.",
                                 code="resremd.reservoir.shape")
        if f < 1:
            raise ReservoirError("The reservoir holds no frames.",
                                 code="resremd.reservoir.shape")
        if kind == "weighted":
            w = self.weights
            if w is None or w.shape != (f,):
                raise ReservoirError(
                    "A weighted reservoir needs weights.npy with one weight "
                    "per frame.", code="resremd.reservoir.weights")
            if not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
                raise ReservoirError(
                    "Reservoir weights must be finite, non-negative and not "
                    "all zero.", code="resremd.reservoir.weights")
            self._cumulative = np.cumsum(w / w.sum())
        elif self.weights is not None:
            raise ReservoirError(
                f"A {kind} reservoir has weights.npy; only a weighted one "
                "should.", code="resremd.reservoir.weights")
        if kind != "non_boltzmann" and not m.get("temperature_K"):
            raise ReservoirError(f"A {kind} reservoir needs a temperature.",
                                 code="resremd.reservoir.temperature")
        if kind == "non_boltzmann" and self.ensemble.constant_pressure:
            raise ReservoirError(
                "A non-Boltzmann reservoir is defined for constant volume "
                "only: equal weights over structures says nothing about how "
                "volumes should be weighted.",
                code="resremd.reservoir.ensemble")

    # -- what it is ---------------------------------------------------------
    @property
    def kind(self) -> str:
        return self.meta["kind"]

    @property
    def temperature_K(self) -> float | None:
        if self.kind == "non_boltzmann":
            return None
        return float(self.meta["temperature_K"])

    @property
    def beta(self) -> float:
        """1/kT of the simulation the frames are a sample of: 0 for a
        non-Boltzmann reservoir, the infinite-temperature limit, and the real
        temperature for a REST2 reservoir, whose `temperature_K` is the
        solute's effective one."""
        if self.kind == "non_boltzmann":
            return 0.0
        if self.rest2:
            return beta(float(self.rest2["simulation_temperature_K"]))
        return beta(self.temperature_K)

    @property
    def rest2(self) -> dict[str, Any] | None:
        """For a reservoir sampled with REST2 scaling: the scale, the real
        temperature and the solute atoms' digest."""
        return self.meta.get("rest2")

    @property
    def scale(self) -> float:
        """The REST2 scale s its frames were sampled at (1: unscaled)."""
        return float(self.rest2["scale"]) if self.rest2 else 1.0

    @property
    def ensemble(self) -> Ensemble:
        return Ensemble.from_dict(self.meta.get("ensemble"))

    @property
    def n_frames(self) -> int:
        return int(self.meta["n_frames"])

    @property
    def n_atoms(self) -> int:
        return int(self.meta["n_atoms"])

    def describe(self) -> str:
        temp = ("equal weights, exchanged as from infinite temperature"
                if self.kind == "non_boltzmann"
                else f"{self.temperature_K:g} K")
        return (f"{self.kind} reservoir of {self.n_frames} frames, {temp}, "
                f"{self.ensemble.describe()}")

    # -- belonging to a run -------------------------------------------------
    def check_against(self, *, topology_sha256: str, n_atoms: int,
                      periodic: bool, ensemble: Ensemble,
                      box: np.ndarray | None,
                      top_temperature_K: float) -> list[str]:
        """Refuse a reservoir that is not a sample of this run's system.

        Returns warnings for what is valid but unusual.
        """
        if n_atoms != self.n_atoms:
            raise ReservoirError(
                f"The reservoir has {self.n_atoms} atoms and the system "
                f"{n_atoms}.", code="resremd.reservoir.mismatch")
        if topology_sha256 != self.meta.get("topology_sha256"):
            raise ReservoirError(
                "The reservoir's atoms are not the system's atoms in the same "
                "order (their topology digests differ). Injecting its "
                "coordinates would put atoms in each other's places.",
                code="resremd.reservoir.mismatch")
        if periodic != bool(self.meta["periodic"]):
            raise ReservoirError(
                "One of the reservoir and the system is periodic and the "
                "other is not.", code="resremd.reservoir.mismatch")
        if not ensemble.same_as(self.ensemble):
            raise ReservoirError(
                f"The reservoir was sampled at {self.ensemble.describe()} and "
                f"this run is at {ensemble.describe()}. The exchange "
                "criterion assumes one pressure ensemble for both.",
                code="resremd.reservoir.ensemble")
        if periodic and not ensemble.constant_pressure:
            spread = np.abs(np.asarray(self.box) - box[None]).max()
            if spread > 1e-5:
                raise ReservoirError(
                    "At constant volume every reservoir frame must have the "
                    f"run's box; they differ by up to {spread:.3g} nm.",
                    code="resremd.reservoir.box")
        warnings = []
        t = self.temperature_K
        if t is not None and t <= top_temperature_K:
            warnings.append(
                f"The reservoir ({t:g} K) is not hotter than the top replica "
                f"({top_temperature_K:g} K). This is valid, but a reservoir "
                "speeds convergence by supplying structures the ladder "
                "reaches slowly, and a cooler one does that less.")
        return warnings

    # -- drawing ------------------------------------------------------------
    def draw(self, rng: np.random.Generator) -> int:
        """A frame index: uniform, or in proportion to its weight."""
        u = rng.random()
        if self._cumulative is not None:
            return int(min(np.searchsorted(self._cumulative, u, side="right"),
                           self.n_frames - 1))
        return int(min(u * self.n_frames, self.n_frames - 1))

    def frame(self, index: int) -> tuple[np.ndarray, np.ndarray | None]:
        """Positions (nm, float64) and box of one frame."""
        pos = np.array(self.positions[index], dtype=np.float64)
        box = None if self.box is None else np.array(self.box[index],
                                                      dtype=np.float64)
        return pos, box

    # -- energies -----------------------------------------------------------
    def content_digest(self) -> str:
        if self._digest is None:
            h = hashlib.sha256()
            for name in ("positions.npy", "box.npy", "weights.npy"):
                p = self.path / name
                if p.exists():
                    h.update(name.encode() + _sha256_file(p).encode())
            self._digest = h.hexdigest()
        return self._digest

    def energies(self, evaluate: Callable[[np.ndarray, np.ndarray | None],
                                          Any],
                 *, key_fields: dict[str, Any],
                 progress: Callable[[str], None] | None = None,
                 terms: bool = False) -> dict[str, np.ndarray]:
        """Potential energy, volume and area of every frame.

        ``evaluate(positions, box)`` returns the potential energy in kJ/mol
        of one frame. ``key_fields`` names what the energies depend on
        besides the frames (System digest, platform, precision, OpenMM
        version); together with the frames' own digest it keys the cache.
        With ``terms``, ``evaluate`` returns REST2's A, B, C for a frame,
        kept as ``rest2_terms`` with ``potential_kjmol`` their sum (s = 1).
        """
        fields = {"frames_sha256": self.content_digest(), **key_fields}
        if terms:
            fields["terms"] = True
        names = ("potential_kjmol", "volume_nm3", "area_nm2") + \
            (("rest2_terms",) if terms else ())
        key = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()
                             ).hexdigest()
        cache_dir = self.path / "energies"
        cache = cache_dir / f"{key[:24]}.npz"
        if cache.exists():
            try:
                with np.load(cache) as data:
                    if str(data["key"]) == key:
                        logger.info("Reservoir energies read from %s", cache)
                        return {k: np.array(data[k]) for k in names}
            except Exception as exc:  # a damaged cache is recomputed
                logger.warning("Ignoring unreadable energy cache %s: %s",
                               cache, exc)
        n = self.n_frames
        u = np.empty(n)
        v = np.zeros(n)
        a = np.zeros(n)
        abc = np.zeros((n, 3)) if terms else None
        t0 = time.time()
        step = max(1, n // 10)
        for k in range(n):
            pos, box = self.frame(k)
            if terms:
                abc[k] = evaluate(pos, box)
                u[k] = abc[k].sum()
            else:
                u[k] = evaluate(pos, box)
            v[k], a[k] = box_volume_and_area(box)
            if progress and (k + 1) % step == 0:
                progress(f"reservoir energies {k + 1}/{n} "
                         f"({time.time() - t0:.0f} s)")
        if not np.all(np.isfinite(u)):
            bad = np.flatnonzero(~np.isfinite(u))[:5].tolist()
            raise ReservoirError(
                f"Reservoir frames {bad} have non-finite energies under this "
                "System. They cannot come from its ensemble.",
                code="resremd.reservoir.energy")
        result = {"potential_kjmol": u, "volume_nm3": v, "area_nm2": a}
        if terms:
            result["rest2_terms"] = abc
        try:
            cache_dir.mkdir(exist_ok=True)
            # A name of its own, so runs sharing a reservoir do not write
            # into each other's file before it is moved into place.
            tmp = cache.with_name(f"{cache.stem}.{os.getpid()}."
                                  f"{time.time_ns()}.tmp.npz")
            np.savez(tmp, key=np.array(key), fields=np.array(
                json.dumps(fields, sort_keys=True)), **result)
            os.replace(tmp, cache)
        except OSError as exc:
            logger.warning("Could not cache reservoir energies in %s: %s",
                           cache_dir, exc)
        return result


    def use_weights(self, weights: np.ndarray) -> None:
        """Draw frames by these weights from now on (this object only)."""
        w = np.asarray(weights, dtype=float)
        self.weights = w / w.sum()
        self._cumulative = np.cumsum(self.weights)

    def reweight_to(self, potential_kjmol: np.ndarray) -> list[str]:
        """Reweight the frames from the Hamiltonian they were built under to
        the one that gave ``potential_kjmol``.

        w_k is multiplied by exp(-beta_R [U_run(k) - U_built(k)]); a constant
        difference changes nothing. Needs the energies recorded when the
        reservoir was generated here. Returns warnings; refuses when fewer
        than ten effective frames would remain.
        """
        if self.kind == "non_boltzmann":
            return []
        stored = self.path / "build_potential_kjmol.npy"
        if not stored.exists():
            raise ReservoirError(
                "Reweighting a reservoir to another Hamiltonian needs the "
                "energies recorded when it was built, and this one has none "
                "(it was imported from files without them). Generate it with "
                "`resremd reservoir generate`, import an Amber reservoir, "
                "which records them, or import it as `non_boltzmann`.",
                code="resremd.reservoir.reweight")
        built = np.load(stored)
        du = np.asarray(potential_kjmol, dtype=float) - built
        base = np.ones(self.n_frames) if self.weights is None \
            else np.asarray(self.weights, dtype=float)
        x = np.log(np.maximum(base, 1e-300)) - self.beta * (du - du.min())
        w = np.exp(x - x.max())
        w /= w.sum()
        kish = float(1.0 / np.sum(w * w))
        if kish < 10:
            raise ReservoirError(
                f"Reweighted to this Hamiltonian the reservoir keeps {kish:.1f} "
                f"effective frames of {self.n_frames}: the two Hamiltonians "
                "are too different for its frames to stand for this one.",
                code="resremd.reservoir.reweight")
        self.use_weights(w)
        self.reweighted = kish
        warnings = []
        if kish < 0.2 * self.n_frames:
            warnings.append(
                f"Reweighted to this Hamiltonian the reservoir keeps "
                f"{kish:.0f} effective frames of {self.n_frames}; exchanges "
                "will keep returning to the few that dominate.")
        return warnings

    def check_hamiltonian(self, potential_kjmol: np.ndarray,
                          system_sha256: str, *,
                          reweighted: bool = False) -> list[str]:
        """Refuse a reservoir sampled under a different Hamiltonian.

        A reservoir this package generated keeps the potential energy of
        every frame from its own simulation, and an imported Amber
        reservoir the energies Amber recorded. Recomputed under the run's
        System they differ only by precision and platform noise, and by a
        constant if the two Systems differ by one, which does not change the
        distribution. A spread in the difference that is not small next to
        kT means the frames are not a Boltzmann sample of this System, and
        every exchange with them would be biased.
        """
        warnings = []
        recorded = (self.meta.get("source") or {}).get("system_sha256")
        stored = self.path / "build_potential_kjmol.npy"
        if recorded and recorded != system_sha256:
            warnings.append(
                "The reservoir was generated from a System that serialises "
                "differently from this run's (their digests differ).")
        if self.kind == "non_boltzmann" or not stored.exists():
            return warnings
        built = np.load(stored)
        if built.shape != potential_kjmol.shape or reweighted:
            return warnings
        spread = self.hamiltonian_spread(potential_kjmol)
        if spread > self.warn_kt():
            text = (f"Recomputed here, the reservoir's energies differ from "
                    f"those recorded when it was generated by {spread:.2g} kT "
                    "(standard deviation of the difference).")
            if spread <= self.refuse_kt():
                warnings.append(
                    text + " That is more than precision noise; check that "
                    "the reservoir was generated from this prepared system "
                    "with the same nonbonded settings (a different platform "
                    "or PME grid adds a little), or set `reservoir_reweight`.")
                return warnings
            raise ReservoirError(
                "The reservoir's frames were not sampled under this run's "
                "Hamiltonian: recomputed here, their energies differ from "
                f"those recorded when they were generated by {spread:.2g} kT "
                "(standard deviation of the difference). A different force "
                "field, cutoff, restraint or solvent model does this. "
                "Generate the reservoir from the same prepared system, or "
                "set `reservoir_reweight` to reweight its frames to this "
                "Hamiltonian.",
                code="resremd.reservoir.hamiltonian")
        return warnings

    def has_build_energies(self) -> bool:
        return (self.path / "build_potential_kjmol.npy").exists()

    def warn_kt(self) -> float:
        """The warning threshold for this reservoir's size (see
        :data:`HAMILTONIAN_WARN_KT`)."""
        return HAMILTONIAN_WARN_KT * _noise_scale(self.n_atoms)

    def refuse_kt(self) -> float:
        return min(HAMILTONIAN_REFUSE_KT * _noise_scale(self.n_atoms),
                   HAMILTONIAN_REFUSE_MAX_KT)

    def hamiltonian_spread(self, potential_kjmol: np.ndarray) -> float:
        """Spread, in kT_R, of this run's energies less those recorded at
        build; 0 when there is nothing to compare."""
        stored = self.path / "build_potential_kjmol.npy"
        if self.kind == "non_boltzmann" or not stored.exists():
            return 0.0
        built = np.load(stored)
        if built.shape != np.shape(potential_kjmol):
            return 0.0
        return float(np.std(np.asarray(potential_kjmol) - built)) * self.beta


def base_metadata(*, kind: str, temperature_K: float | None,
                  ensemble: Ensemble, n_frames: int, n_atoms: int,
                  periodic: bool, topology_sha256: str,
                  source: dict[str, Any]) -> dict[str, Any]:
    from . import __version__

    return {
        "format": FORMAT,
        "kind": kind,
        "temperature_K": temperature_K,
        "ensemble": ensemble.as_dict(),
        "n_frames": int(n_frames),
        "n_atoms": int(n_atoms),
        "periodic": bool(periodic),
        "topology_sha256": topology_sha256,
        "source": source,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "resremd_version": __version__,
        "complete": False,
    }


def write_reservoir(path: str | Path, *, topology: Any, positions: np.ndarray,
                    kind: str, temperature_K: float | None,
                    ensemble: Ensemble | None = None,
                    box: np.ndarray | None = None,
                    weights: np.ndarray | None = None,
                    source: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write a reservoir from frames already in memory.

    ``positions`` is (frames, atoms, 3) in nm and ``box`` (frames, 3, 3) for
    a periodic system. For frames drawn some other way than by this
    package's own generate and import: exact draws for a model system, or a
    set with known weights.
    """
    from .system import topology_digest

    path = Path(path)
    positions = np.asarray(positions, dtype=float)
    n_frames, n_atoms = positions.shape[:2]
    if kind not in KINDS:
        raise ReservoirError(f"Unknown reservoir kind {kind!r}.",
                             code="resremd.reservoir.kind")
    meta = base_metadata(kind=kind, temperature_K=temperature_K,
                         ensemble=ensemble or Ensemble(), n_frames=n_frames,
                         n_atoms=n_atoms, periodic=box is not None,
                         topology_sha256=topology_digest(topology),
                         source=source or {"method": "written from arrays"})
    writer = ReservoirWriter(path, n_frames=n_frames, n_atoms=n_atoms,
                             periodic=box is not None)
    for k in range(n_frames):
        writer.write(k, positions[k], None if box is None else box[k])
    writer.flush()
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        np.save(path / "weights.npy", w / w.sum())
    write_topology(path / "topology.pdb", topology, positions[0],
                   None if box is None else np.asarray(box[0]))
    meta["complete"] = True
    write_json(path / "reservoir.json", meta)
    Reservoir.open(path)  # everything written is checked on the way out
    return meta
