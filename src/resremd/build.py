"""Making a reservoir: by simulating at one temperature, or from trajectories.

A Boltzmann reservoir is only as good as the simulation behind it: replica
exchange coupled to it converges to the right ensemble exactly as far as the
reservoir is a converged sample at its temperature. The build records how
many effectively independent frames it holds so that can be judged.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .errors import InputError, ReservoirError, ResumeError, require
from .options import GENERATE, IMPORT, resolve as resolve_options
from .reservoir import ReservoirWriter, base_metadata, write_json, \
    write_topology
from .statistics import statistical_inefficiency
from .stopping import StopRequests
from .system import (Prepared, create_context, from_objects, load_prepared,
                     make_integrator, system_digest, topology_digest,
                     warn_if_no_gpu)
from .thermo import Ensemble, simulated_system

logger = logging.getLogger("resremd")


def _prepared(prepared, system, topology, positions, box, options) -> Prepared:
    if isinstance(prepared, (str, Path)):
        return load_prepared(prepared)
    if prepared is not None:
        return prepared
    if system is not None:
        return from_objects(system, topology, positions, box)
    if options.get("prepared"):
        return load_prepared(options["prepared"])
    raise InputError("No system: give `prepared` or the OpenMM objects.",
                     code="resremd.input.prepared")


def generate(prepared: Prepared | str | Path | None = None, *,
             system: Any = None, topology: Any = None, positions: Any = None,
             box: Any = None,
             on_progress: Callable[[dict[str, Any]], None] | None = None,
             **settings: Any) -> dict[str, Any]:
    """Simulate at the reservoir temperature and keep evenly spaced frames.

    Settings are those of :data:`resremd.options.GENERATE`. Returns the
    reservoir's metadata, also written to ``reservoir.json``.
    """
    import openmm
    from openmm import unit

    if isinstance(prepared, (str, Path)):
        settings["prepared"] = str(prepared)
        prepared = None
    o = resolve_options(GENERATE, settings)
    prep = _prepared(prepared, system, topology, positions, box, o)
    t = o["temperature_K"]
    dt = o["timestep_fs"]
    sim_system, ensemble = simulated_system(
        prep.system, ensemble=o["ensemble"], pressure_bar=o["pressure_bar"],
        temperature_K=t, frequency=o["barostat_frequency"])
    biases = o["bias_torsions"] or None
    bias_group = None
    if biases:
        sim_system, bias_group = add_torsion_biases(
            sim_system, biases, prep.n_atoms)
    kind = "weighted" if biases else "boltzmann"
    watch = _convergence_torsions(o, prep.n_atoms)
    interval = o["frame_interval_steps"]
    n_frames = int(round(o["duration_ns"] * 1e6 / dt)) // interval
    if n_frames < 1:
        raise InputError(
            f"{o['duration_ns']:g} ns holds no frame {interval} steps apart.",
            code="resremd.input.length")
    out = Path(o["output"])
    chk_file = out / "build_checkpoint.npz"
    meta_file = out / "reservoir.json"
    if o["resume"]:
        if not (chk_file.exists() and meta_file.exists()):
            raise ResumeError(f"There is no unfinished build in {out}.",
                              code="resremd.resume.missing")
        meta = json.loads(meta_file.read_text())
        if meta["n_frames"] != n_frames \
                or meta["source"].get("frame_interval_steps") != interval \
                or meta["source"].get("timestep_fs") != dt \
                or meta["temperature_K"] != t \
                or meta["source"].get("bias_torsions") != biases \
                or not _same_convergence(meta["source"].get("convergence"),
                                         _convergence_source(o, watch)):
            raise ResumeError(
                "The build in progress was started with a different "
                "temperature, length, frame interval, time step, bias or "
                "convergence test.",
                code="resremd.resume.mismatch")
        with np.load(chk_file) as data:
            saved = json.loads(str(data["meta"]))
            start_pos = np.array(data["positions"])
            start_vel = np.array(data["velocities"])
            start_box = np.array(data["box"]) if "box" in data else None
        seed = saved["seed"]
        done = saved["frames"]
        wall_before = float(saved.get("wall_seconds", 0.0))
        eq_steps = int(saved.get("equilibration_steps", 0))
        eq_left = int(saved.get("equilibration_left", 0))
        history = list(saved.get("convergence_history", []))
        # Converged, but interrupted while finishing: only finish.
        finishing = bool(saved.get("converged", False))
    else:
        history = []
        finishing = False
        if meta_file.exists() and not chk_file.exists() and not json.loads(
                meta_file.read_text()).get("complete"):
            # Killed before its first checkpoint: nothing to resume, and
            # nothing worth keeping. Clear what this build wrote and start.
            _clear_unfinished_build(out)
        if meta_file.exists():
            raise InputError(
                f"{out} already holds a reservoir"
                + (" whose build can be resumed with `resume`."
                   if chk_file.exists() else ".")
                + " Choose another `output` to start a new one.",
                code="resremd.input.output")
        if out.exists() and any(out.iterdir()):
            raise InputError(f"{out} is not empty.", code="resremd.input.output")
        out.mkdir(parents=True, exist_ok=True)
        seed = o["random_seed"] if o["random_seed"] is not None \
            else secrets.randbelow(2**31 - 1)
        done = 0
        wall_before = 0.0
        eq_steps = 0
        meta = base_metadata(
            kind=kind, temperature_K=t, ensemble=ensemble,
            n_frames=n_frames, n_atoms=prep.n_atoms, periodic=prep.periodic,
            topology_sha256=topology_digest(prep.topology),
            source={"method": "simulated at the reservoir temperature",
                    "prepared": prep.source,
                    "system_sha256": system_digest(prep.system),
                    "duration_ns": o["duration_ns"],
                    "frame_interval_steps": interval,
                    "equilibration_ns": o["equilibration_ns"],
                    "integrator": o["integrator"], "timestep_fs": dt,
                    "bias_torsions": biases,
                    "convergence": _convergence_source(o, watch),
                    "friction_per_ps": o["friction_per_ps"],
                    "seed": int(seed)})
        write_json(meta_file, meta)
    session_start = time.time()
    rng = np.random.default_rng([int(seed), 2, done])
    integrator = make_integrator(o["integrator"], t, o["friction_per_ps"], dt,
                                 int(rng.integers(1, 2**31 - 1)))
    context, platform = create_context(
        sim_system, integrator, platform=o["platform"],
        precision=o["precision"], device=o["device_index"],
        cpu_threads=o["cpu_threads"])
    meta["source"]["platform"] = platform
    warn_if_no_gpu(o["platform"], platform)
    if ensemble.temperature_parameter:
        # A barostat that came with the System was made for some other
        # temperature. Left there, it would accept volume moves as if at that
        # temperature, and the frames would not be a sample at this one.
        context.setParameter(ensemble.temperature_parameter, t)
    periodic = prep.periodic

    def set_box(b):
        if periodic and b is not None:
            context.setPeriodicBoxVectors(*(openmm.Vec3(*map(float, row))
                                            for row in b))

    writer = ReservoirWriter(out, n_frames=n_frames, n_atoms=prep.n_atoms,
                             periodic=periodic, reopen=o["resume"])
    from numpy.lib.format import open_memmap

    potential = open_memmap(out / "build_potential_kjmol.npy",
                            mode="r+" if o["resume"] else "w+",
                            dtype=np.float64, shape=(n_frames,))
    bias_energy = None
    if biases:
        bias_energy = open_memmap(out / "build_bias_kjmol.npy",
                                  mode="r+" if o["resume"] else "w+",
                                  dtype=np.float64, shape=(n_frames,))
    angles = None
    if watch:
        angles = open_memmap(out / "build_torsions_deg.npy",
                             mode="r+" if o["resume"] else "w+",
                             dtype=np.float32, shape=(n_frames, len(watch)))
    if o["resume"]:
        set_box(start_box)
        context.setPositions(start_pos)
        context.setVelocities(start_vel)
        logger.info("Resuming the reservoir build at frame %d of %d", done,
                    n_frames)
    else:
        set_box(prep.box)
        context.setPositions(prep.positions)
        if o["minimize"]:
            logger.info("Minimising")
            openmm.LocalEnergyMinimizer.minimize(context)
        context.setVelocitiesToTemperature(t * unit.kelvin,
                                           int(rng.integers(1, 2**31 - 1)))
        eq_left = int(round(o["equilibration_ns"] * 1e6 / dt))
        logger.info("Equilibrating at %g K (%d steps)", t, eq_left)

    per_checkpoint = max(1, int(round(500.0 / (interval * dt / 1000.0))))
    t0 = time.time()
    start = done
    status = "complete"
    with StopRequests() as stop:
        # In pieces, so a stop request during a long equilibration is
        # answered within 10 ps and the rest of it is resumed later.
        piece = max(1, int(round(1e4 / dt)))
        while eq_left > 0 and not stop.requested:
            n = min(piece, eq_left)
            integrator.step(n)
            eq_left -= n
            eq_steps += n
        if stop.requested and done < n_frames:
            _save_build_checkpoint(
                chk_file, context, done, seed, periodic,
                wall_seconds=wall_before + time.time() - session_start,
                equilibration_steps=eq_steps, equilibration_left=eq_left)
            logger.info("Stopped during equilibration. Resume with `resume`.")
            return meta
        for k in range(done, done if finishing else n_frames):
            integrator.step(interval)
            state = context.getState(getPositions=True, getEnergy=True)
            pos = np.asarray(state.getPositions(asNumpy=True)._value)
            b = (np.asarray(state.getPeriodicBoxVectors(asNumpy=True)._value)
                 if periodic else None)
            energy = float(state.getPotentialEnergy()._value)
            if bias_energy is not None:
                v_bias = float(context.getState(
                    getEnergy=True, groups={bias_group})
                    .getPotentialEnergy()._value)
                bias_energy[k] = v_bias
                energy -= v_bias  # recorded unbiased, as the run will see it
            if not np.isfinite(energy):
                raise ReservoirError(
                    f"The simulation at {t:g} K blew up at frame {k}.",
                    code="resremd.simulation.unstable")
            writer.write(k, pos, b)
            potential[k] = energy
            if angles is not None:
                angles[k] = torsion_angles_deg(pos, watch)
            done = k + 1
            converged = False
            scheduled = done % per_checkpoint == 0 or done == n_frames
            if scheduled or stop.requested:
                writer.flush()
                potential.flush()
                if bias_energy is not None:
                    bias_energy.flush()
                # The test runs at scheduled checks only, so two checks in a
                # row are always a checkpoint's worth of frames apart.
                if angles is not None and scheduled:
                    angles.flush()
                    tv = halves_tv(angles[:done], o["convergence_bins"],
                                   None if bias_energy is None else
                                   bias_weights(bias_energy[:done], t))
                    moves = basin_transitions(
                        angles[:done], weights=None if bias_energy is None
                        else bias_weights(bias_energy[:done], t))
                    history.append([done, tv, int(min(moves))])
                    logger.info("halves of the watched torsions differ by "
                                "TV %.3f; fewest transitions %d", tv,
                                min(moves))
                    converged = _converged(history, o["convergence_tv"],
                                           o["convergence_min_transitions"])
                elif angles is not None:
                    angles.flush()
                _save_build_checkpoint(
                    chk_file, context, done, seed, periodic,
                    wall_seconds=wall_before + time.time() - session_start,
                    equilibration_steps=eq_steps, equilibration_left=0,
                    convergence_history=history,
                    converged=converged and done < n_frames)
                rate = (done - start) * interval * dt / 1e6 / \
                    max(time.time() - t0, 1e-9) * 86400
                logger.info("reservoir frame %d/%d  %.1f ns/day", done,
                            n_frames, rate)
                if on_progress:
                    on_progress({"frames": done, "n_frames": n_frames,
                                 "ns_per_day": rate})
            if converged and done < n_frames:
                logger.info("Converged by the watched torsions at frame %d "
                            "of at most %d.", done, n_frames)
                break
            if stop.requested and done < n_frames:
                status = "stopped"
                break
    if status != "complete":
        logger.info("Stopped at frame %d of %d. Resume with `resume`.", done,
                    n_frames)
        return meta

    if done < n_frames:
        # Stopped early by the convergence test: keep only what was made.
        # The checkpoint already says so, so a build killed from here on
        # resumes straight to this point; truncating again is harmless.
        for arr in (writer.positions, writer.box, potential, bias_energy,
                    angles):
            if arr is not None:
                arr.flush()
        del writer, potential, bias_energy, angles, arr
        for f in ("positions.npy", "box.npy", "build_potential_kjmol.npy",
                  "build_bias_kjmol.npy", "build_torsions_deg.npy"):
            if (out / f).exists():
                truncate_npy(out / f, done)
        n_frames = done
        meta["n_frames"] = n_frames
        writer = ReservoirWriter(out, n_frames=n_frames, n_atoms=prep.n_atoms,
                                 periodic=periodic, reopen=True)
        potential = np.load(out / "build_potential_kjmol.npy")
        bias_energy = np.load(out / "build_bias_kjmol.npy") if biases \
            else None
    if watch:
        moves = basin_transitions(
            np.load(out / "build_torsions_deg.npy"),
            weights=None if bias_energy is None
            else bias_weights(np.asarray(bias_energy), t))
        done_ok = _converged(history, o["convergence_tv"],
                             o["convergence_min_transitions"])
        meta["convergence"] = {
            "history": [[int(h[0]), float(h[1])] + [int(x) for x in h[2:]]
                        for h in history],
            "halves_tv": float(history[-1][1]) if history else None,
            "transitions": [int(x) for x in moves],
            "converged": done_ok,
            "frames": int(n_frames),
            "ns": n_frames * interval * dt / 1e6,
        }
        if o["convergence_tv"] is not None and not done_ok and \
                min(moves) < o["convergence_min_transitions"]:
            logger.warning(
                "A watched torsion made only %d transitions between basins "
                "(%d needed), so the build ran to its full length. Either it "
                "stayed in one basin, and the reservoir has not sampled its "
                "other states (a bias on it may help), or its states are not "
                "separated by a barrier of 2 kT at this temperature, and it "
                "is not a slow torsion worth watching.", min(moves),
                o["convergence_min_transitions"])
    g = statistical_inefficiency(np.asarray(potential))
    meta["statistics"] = {
        "potential_mean_kjmol": float(np.mean(potential)),
        "potential_std_kjmol": float(np.std(potential)),
        "statistical_inefficiency_frames": g,
        "effective_independent_frames": n_frames / g,
        "note": "From the potential energy, which decorrelates faster than "
                "slow conformational change; treat as an upper bound.",
    }
    if bias_energy is not None:
        weights = bias_weights(np.asarray(bias_energy), t)
        np.save(out / "weights.npy", weights)
        kish = float(1.0 / np.sum(weights ** 2))
        meta["statistics"]["effective_frames_kish"] = kish
        # Correlation and uneven weights compound.
        meta["statistics"]["effective_independent_frames"] = kish / g
        if kish < 0.05 * n_frames:
            logger.warning(
                "The bias leaves %.0f effective frames of %d: its weights "
                "are dominated by a few frames. A weaker bias, or a longer "
                "run, gives a more useful reservoir.", kish, n_frames)
    first = np.array(writer.positions[0], dtype=float)
    write_topology(out / "topology.pdb", prep.topology, first,
                   None if writer.box is None else np.array(writer.box[0]))
    meta["cost"] = {
        "md_steps": {"equilibration": eq_steps,
                     "production": n_frames * interval},
        "md_steps_total": eq_steps + n_frames * interval,
        "wall_seconds": wall_before + time.time() - session_start,
        "platform": platform,
    }
    meta["complete"] = True
    write_json(meta_file, meta)
    chk_file.unlink(missing_ok=True)
    logger.info("Reservoir complete: %d frames, about %.0f independent by "
                "the potential energy (g = %.1f).", n_frames, n_frames / g, g)
    return meta


_BUILD_FILES = ("reservoir.json", "positions.npy", "box.npy",
                "build_potential_kjmol.npy", "build_bias_kjmol.npy",
                "build_torsions_deg.npy", "weights.npy")


def _convergence_torsions(o: dict[str, Any], n_atoms: int
                          ) -> list[list[int]] | None:
    watch = o["convergence_torsions"]
    if not watch:
        if o["convergence_tv"] is not None:
            raise InputError("`convergence_tv` needs `convergence_torsions` "
                             "to judge by.", code="resremd.input.convergence")
        return None
    out = []
    for i, t in enumerate(watch):
        try:
            atoms = [int(a) for a in t]
        except (TypeError, ValueError):
            atoms = []
        if len(atoms) != 4 or len(set(atoms)) != 4 or min(atoms) < 0 \
                or max(atoms) >= n_atoms:
            raise InputError(f"Convergence torsion {i}: four distinct atom "
                             f"indices below {n_atoms}.",
                             code="resremd.input.convergence")
        out.append(atoms)
    return out


def _convergence_source(o, watch) -> dict[str, Any] | None:
    if not watch:
        return None
    return {"torsions": watch, "tv": o["convergence_tv"],
            "bins": o["convergence_bins"],
            "min_transitions": o["convergence_min_transitions"]}


def _same_convergence(saved: dict | None, now: dict | None) -> bool:
    """The convergence test of a build in progress, against the one asked
    for. Builds started before the transition requirement existed carry no
    `min_transitions`, and take the one asked for."""
    if saved is None or now is None:
        return saved == now
    if "min_transitions" not in saved:
        saved = {**saved, "min_transitions": now["min_transitions"]}
    return saved == now


def _converged(history: list, threshold: float | None,
               min_transitions: int = 0) -> bool:
    """Two checks in a row within the threshold, with every watched torsion
    having made enough transitions by then."""
    def ok(h):
        moves = h[2] if len(h) > 2 else 0
        return h[1] <= threshold and moves >= min_transitions

    return threshold is not None and len(history) >= 2 and all(
        ok(h) for h in history[-2:])


def torsion_angles_deg(positions: np.ndarray, torsions: list[list[int]]
                       ) -> np.ndarray:
    """Dihedral angles (degrees, -180 to 180) of one frame."""
    p = np.asarray(positions, dtype=float)[np.asarray(torsions)]  # (n, 4, 3)
    b0 = p[:, 0] - p[:, 1]
    b1 = p[:, 2] - p[:, 1]
    b2 = p[:, 3] - p[:, 2]
    b1 /= np.linalg.norm(b1, axis=1, keepdims=True)
    v = b0 - np.sum(b0 * b1, axis=1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=1, keepdims=True) * b1
    x = np.sum(v * w, axis=1)
    y = np.sum(np.cross(b1, v) * w, axis=1)
    return np.degrees(np.arctan2(y, x))


def torsion_regions(angles_deg: np.ndarray, bins: int) -> np.ndarray:
    """The region of each angle: ``bins`` equal arcs, region k centred on
    -180 + k 360/bins degrees. With an even number of regions, 0 and 180
    are centres, never boundaries."""
    a = np.asarray(angles_deg, dtype=float)
    width = 360.0 / bins
    return np.floor((a + 180.0 + width / 2) / width).astype(int) % bins


def torsion_basins(angles_deg: np.ndarray, weights: np.ndarray | None = None,
                   *, bins: int = 36, depth: float = 2.0,
                   min_population: float = 0.01) -> tuple[np.ndarray, list]:
    """The basins of one torsion's free energy F = -ln p, in kT of the
    sampled temperature (with the weights, of the unbiased system).

    ``bins`` arcs are grouped by steepest descent on the lightly smoothed F
    into basins; neighbours whose barrier, from the shallower side, is below
    ``depth`` kT are merged, and arcs never visited separate basins
    outright. Each arc's label is its basin's (-1: unvisited, or a basin
    holding less than ``min_population`` of the weight), and the returned
    list holds each basin's core: its arcs within depth/2 of its minimum.
    """
    a = np.asarray(angles_deg, dtype=float)
    w = np.ones(len(a)) if weights is None else np.asarray(weights, float)
    width = 360.0 / bins
    idx = np.floor((a + 180.0) / width).astype(int) % bins
    p = np.bincount(idx, weights=w, minlength=bins)
    visited = p > 0
    p = (np.roll(p, 1) + p + np.roll(p, -1)) / 3.0
    with np.errstate(divide="ignore"):
        f = np.where(visited, -np.log(np.maximum(p, 1e-300) / p.sum()),
                     np.inf)
    label = np.full(bins, -1)
    for b in np.flatnonzero(visited):
        c = b
        while True:
            left, right = (c - 1) % bins, (c + 1) % bins
            nxt = min((f[left], left), (f[c], c), (f[right], right))[1]
            if nxt == c or f[nxt] >= f[c]:
                break
            c = nxt
        label[b] = c
    # Merge neighbouring basins across barriers lower than `depth`.
    while True:
        best = None
        for b in range(bins):
            n = (b + 1) % bins
            la, lb = label[b], label[n]
            if la < 0 or lb < 0 or la == lb:
                continue
            barrier = max(f[b], f[n]) - max(f[la], f[lb])
            if barrier < depth and (best is None or barrier < best[0]):
                best = (barrier, la, lb)
        if best is None:
            break
        _, la, lb = best
        keep, drop = (la, lb) if f[la] <= f[lb] else (lb, la)
        label[label == drop] = keep
    total = p[visited].sum()
    cores = []
    for basin in sorted(set(label[label >= 0].tolist())):
        members = label == basin
        if p[members & visited].sum() < min_population * total:
            label[members] = -1
            continue
        cores.append(np.flatnonzero(members & (f <= f[basin] + depth / 2)))
    return label, cores


def basin_transitions(angles_deg: np.ndarray, bins: int = 36,
                      weights: np.ndarray | None = None) -> list[int]:
    """Per torsion, the moves from one basin's core to another's (basins as
    in :func:`torsion_basins`). Motion within a basin, however wide, never
    counts; only crossing a barrier of at least 2 kT does."""
    a = np.asarray(angles_deg, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    out = []
    width = 360.0 / bins
    for j in range(a.shape[1]):
        _, cores = torsion_basins(a[:, j], weights, bins=bins)
        where = np.full(bins, -1)
        for k, core in enumerate(cores):
            where[core] = k
        idx = np.floor((a[:, j] + 180.0) / width).astype(int) % bins
        seq = where[idx]
        seq = seq[seq >= 0]
        out.append(int(np.count_nonzero(seq[1:] != seq[:-1])))
    return out


def halves_tv(angles_deg: np.ndarray, bins: int,
              weights: np.ndarray | None = None) -> float:
    """The largest, over torsions, total variation distance between the
    region populations of the first and second halves of a series of
    frames (regions as in :func:`torsion_regions`)."""
    a = np.asarray(angles_deg, dtype=float)
    n = len(a)
    if n < 2:
        return 1.0
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    half = n // 2
    worst = 0.0
    for j in range(a.shape[1]):
        r = torsion_regions(a[:, j], bins)
        h = []
        for sl in (slice(0, half), slice(half, n)):
            c = np.bincount(r[sl], weights=w[sl], minlength=bins)
            h.append(c / c.sum() if c.sum() > 0 else c)
        worst = max(worst, 0.5 * float(np.abs(h[0] - h[1]).sum()))
    return worst


def truncate_npy(path: Path, n: int) -> None:
    """Keep the first n rows of a .npy file, in place."""
    from numpy.lib import format as fmt

    with open(path, "r+b") as fh:
        version = fmt.read_magic(fh)
        read = fmt.read_array_header_1_0 if version == (1, 0) \
            else fmt.read_array_header_2_0
        shape, fortran, dtype = read(fh)
        start = fh.tell()
        if fortran or n > shape[0]:
            raise ValueError(f"Cannot truncate {path} to {n} rows.")
        new_shape = (n,) + tuple(shape[1:])
        header = {"descr": fmt.dtype_to_descr(dtype),
                  "fortran_order": False, "shape": new_shape}
        text = repr(header).encode("latin1")
        # Same header length as before, so the data does not move.
        prefix = 10 if version == (1, 0) else 12
        room = start - prefix - 1
        if len(text) > room:
            raise ValueError(f"The header of {path} has no room.")
        text = text + b" " * (room - len(text)) + b"\n"
        fh.seek(prefix)
        fh.write(text)
        fh.truncate(start + n * int(np.prod(shape[1:], dtype=int))
                    * dtype.itemsize)


def add_torsion_biases(system: Any, biases: list[dict[str, Any]],
                       n_atoms: int) -> tuple[Any, int]:
    """A copy of the System with the bias torsions in a force group of
    their own, so their energy can be read apart from the rest."""
    import openmm

    used = {f.getForceGroup() for f in system.getForces()}
    free = [g for g in range(31, -1, -1) if g not in used]
    if not free:
        raise InputError("Every force group is taken; the bias needs one.",
                         code="resremd.input.bias")
    group = free[0]
    system = openmm.XmlSerializer.deserialize(
        openmm.XmlSerializer.serialize(system))
    for i, b in enumerate(biases):
        if not isinstance(b, dict) or set(b) - {"atoms", "energy",
                                                "parameters"}:
            raise InputError(
                f"Bias {i}: a mapping with `atoms`, `energy` and, "
                "optionally, `parameters`.", code="resremd.input.bias")
        atoms = [int(a) for a in b.get("atoms", [])]
        if len(atoms) != 4 or min(atoms) < 0 or max(atoms) >= n_atoms \
                or len(set(atoms)) != 4:
            raise InputError(f"Bias {i}: `atoms` must be four distinct atom "
                             f"indices below {n_atoms}.",
                             code="resremd.input.bias")
        params = dict(b.get("parameters") or {})
        force = openmm.CustomTorsionForce(str(b["energy"]))
        for name in params:
            force.addPerTorsionParameter(str(name))
        force.addTorsion(*atoms, [float(v) for v in params.values()])
        force.setForceGroup(group)
        system.addForce(force)
    try:  # an expression OpenMM cannot parse is reported here, not later
        openmm.Context(system, openmm.VerletIntegrator(0.001),
                       openmm.Platform.getPlatformByName("Reference"))
    except Exception as exc:
        raise InputError(f"The bias expression is not valid: {exc}",
                         code="resremd.input.bias") from exc
    return system, group


def bias_weights(bias_kjmol: np.ndarray, temperature_K: float) -> np.ndarray:
    """exp(beta V_bias), normalised: frames sampled under U + V_bias,
    weighted to be a sample under U."""
    from .thermo import beta

    x = beta(temperature_K) * np.asarray(bias_kjmol, dtype=float)
    w = np.exp(x - x.max())
    return w / w.sum()


def _clear_unfinished_build(out: Path) -> None:
    """Remove what an interrupted build wrote, and only that."""
    for name in _BUILD_FILES:
        (out / name).unlink(missing_ok=True)
    for tmp in out.glob("*.tmp*"):
        tmp.unlink()


def _save_build_checkpoint(path: Path, context: Any, frames: int, seed: int,
                           periodic: bool, *, wall_seconds: float,
                           equilibration_steps: int,
                           equilibration_left: int,
                           convergence_history: list | None = None,
                           converged: bool = False) -> None:
    state = context.getState(getPositions=True, getVelocities=True)
    arrays = {
        "positions": np.asarray(state.getPositions(asNumpy=True)._value),
        "velocities": np.asarray(state.getVelocities(asNumpy=True)._value),
    }
    if periodic:
        arrays["box"] = np.asarray(state.getPeriodicBoxVectors(
            asNumpy=True)._value)
    tmp = path.with_name(path.name + ".tmp.npz")
    with open(tmp, "wb") as fh:
        np.savez(fh, meta=np.array(json.dumps({
            "frames": frames, "seed": int(seed),
            "wall_seconds": wall_seconds,
            "equilibration_steps": int(equilibration_steps),
            "equilibration_left": int(equilibration_left),
            "convergence_history": convergence_history or [],
            "converged": bool(converged)})), **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def import_trajectories(**settings: Any) -> dict[str, Any]:
    """Build a reservoir from existing trajectories.

    Settings are those of :data:`resremd.options.IMPORT`. Needs MDTraj.
    """
    from openmm import app

    o = resolve_options(IMPORT, settings)
    md = require("mdtraj", "Reading trajectories into a reservoir", "import")
    kind = o["kind"]
    if kind == "weighted" and not o["weights"]:
        raise InputError("A weighted reservoir needs `weights`.",
                         code="resremd.input.missing")
    if kind != "weighted" and o["weights"]:
        raise InputError(
            f"`weights` were given for a {kind} reservoir. Say `kind: "
            "weighted` if the frames need them.", code="resremd.input.weights")
    if kind != "non_boltzmann" and o["temperature_K"] is None:
        raise InputError(f"A {kind} reservoir needs `temperature_K`.",
                         code="resremd.input.missing")
    if kind == "non_boltzmann" and o["pressure_bar"] is not None:
        raise InputError("A non-Boltzmann reservoir is for constant volume "
                         "only.", code="resremd.input.ensemble")
    omm_topology = app.PDBFile(o["topology"]).topology
    md_topology = md.load_topology(o["topology"])
    n_atoms = omm_topology.getNumAtoms()
    files = [str(f) for f in o["trajectories"]]
    for f in files:
        if not Path(f).is_file():
            raise InputError(f"No trajectory at {f}.",
                             code="resremd.input.missing")
    lossy = [f for f in files if Path(f).suffix.lower() in (".xtc", ".gro",
                                                            ".pdb")]
    if lossy:
        logger.warning(
            "%s store coordinates rounded to 0.001 nm or coarser. Rounding "
            "stretches every bond a little, which raises the energy by "
            "several kJ/mol for a solvated system, so the frames are no "
            "longer a Boltzmann sample. Prefer DCD, TRR or NetCDF.",
            ", ".join(lossy))

    boxes: list[np.ndarray] = []
    total = 0
    periodic = None
    for f in files:
        for chunk in md.iterload(f, top=md_topology, chunk=1000):
            if chunk.n_atoms != n_atoms:
                raise ReservoirError(
                    f"{f} has {chunk.n_atoms} atoms and the topology "
                    f"{n_atoms}.", code="resremd.reservoir.mismatch")
            has_box = chunk.unitcell_vectors is not None
            if periodic is None:
                periodic = has_box
            elif periodic != has_box:
                raise ReservoirError("Some trajectories have boxes and some "
                                     "do not.", code="resremd.reservoir.box")
            if has_box:
                boxes.append(np.asarray(chunk.unitcell_vectors, dtype=float))
            total += chunk.n_frames
    keep = np.arange(0, total, o["stride"])
    weights = None
    if o["weights"]:
        wf = Path(o["weights"])
        weights = np.load(wf) if wf.suffix == ".npy" else np.loadtxt(wf)
        weights = np.asarray(weights, dtype=float).ravel()
        if weights.size != total:
            raise ReservoirError(
                f"{wf} has {weights.size} weights for {total} frames. Give one "
                "per input frame, before the stride.",
                code="resremd.reservoir.weights")
        weights = weights[keep]
        if not np.all(np.isfinite(weights)) or np.any(weights < 0) \
                or weights.sum() <= 0:
            raise ReservoirError("Weights must be finite, non-negative and "
                                 "not all zero.",
                                 code="resremd.reservoir.weights")
        weights = weights / weights.sum()
    pressure = o["pressure_bar"]
    if periodic:
        box_all = np.concatenate(boxes)[keep]
        varies = float(np.ptp(box_all, axis=0).max()) > 1e-6
        if varies and pressure is None:
            raise ReservoirError(
                "The frames' boxes differ, so they were sampled at constant "
                "pressure. Give `pressure_bar` (and `surface_tension_bar_nm` "
                "for a membrane).", code="resremd.reservoir.ensemble")
        if not varies and pressure is not None:
            raise ReservoirError(
                "Every frame has the same box, which a constant-pressure "
                "simulation does not produce. Leave `pressure_bar` out if they "
                "were sampled at constant volume.",
                code="resremd.reservoir.ensemble")
    elif pressure is not None:
        raise ReservoirError("A system without a box has no pressure.",
                             code="resremd.reservoir.ensemble")
    ensemble = Ensemble(pressure_bar=pressure,
                        surface_tension_bar_nm=o["surface_tension_bar_nm"]
                        if pressure is not None else 0.0)
    out = Path(o["output"])
    if out.exists() and any(out.iterdir()):
        raise InputError(f"{out} is not empty.", code="resremd.input.output")
    out.mkdir(parents=True, exist_ok=True)
    n_frames = int(keep.size)
    meta = base_metadata(
        kind=kind, temperature_K=o["temperature_K"]
        if kind != "non_boltzmann" else None,
        ensemble=ensemble, n_frames=n_frames, n_atoms=n_atoms,
        periodic=bool(periodic), topology_sha256=topology_digest(omm_topology),
        source={"method": "imported from trajectories",
                "trajectories": [str(Path(f).resolve()) for f in files],
                "topology": str(Path(o["topology"]).resolve()),
                "stride": o["stride"],
                "weights": str(Path(o["weights"]).resolve())
                if o["weights"] else None,
                "input_frames": total,
                "lossy_formats": lossy})
    writer = ReservoirWriter(out, n_frames=n_frames, n_atoms=n_atoms,
                             periodic=bool(periodic))
    wanted = set(keep.tolist())
    index = 0
    written = 0
    for f in files:
        for chunk in md.iterload(f, top=md_topology, chunk=1000):
            for i in range(chunk.n_frames):
                if index in wanted:
                    b = (np.asarray(chunk.unitcell_vectors[i], dtype=float)
                         if periodic else None)
                    writer.write(written, chunk.xyz[i], b)
                    written += 1
                index += 1
    writer.flush()
    if weights is not None:
        np.save(out / "weights.npy", weights)
        meta["statistics"] = {
            "effective_frames_kish": float(1.0 / np.sum(weights ** 2))}
    write_topology(out / "topology.pdb", omm_topology,
                   np.array(writer.positions[0], dtype=float),
                   None if writer.box is None else np.array(writer.box[0]))
    meta["complete"] = True
    write_json(out / "reservoir.json", meta)
    logger.info("Reservoir written: %d frames (%s).", n_frames, kind)
    return meta
