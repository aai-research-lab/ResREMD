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
    # With REST2 scaling the simulation runs at the real temperature t_sim
    # and the solute at the effective temperature t, the reservoir's.
    base_system = prep.system
    rest2_meta = None
    t_sim = t
    if o["rest2_run_temperature_K"] is not None:
        from .rest2 import rest2_system, scale_of, solute_digest
        from .system import select_atoms

        t_sim = float(o["rest2_run_temperature_K"])
        solute = select_atoms(prep.topology, o["rest2_selection"],
                              o["rest2_atoms"])
        _, ens = simulated_system(
            prep.system, ensemble=o["ensemble"],
            pressure_bar=o["pressure_bar"], temperature_K=t_sim,
            frequency=o["barostat_frequency"])
        base_system, info = rest2_system(
            prep.system, solute, constant_pressure=ens.constant_pressure)
        rest2_meta = {"scale": scale_of(t_sim, t),
                      "simulation_temperature_K": t_sim,
                      "solute_sha256": solute_digest(solute),
                      "solute_atoms": info["solute_atoms"],
                      "dispersion_correction": info["dispersion_correction"]}
    sim_system, ensemble = simulated_system(
        base_system, ensemble=o["ensemble"], pressure_bar=o["pressure_bar"],
        temperature_K=t_sim, frequency=o["barostat_frequency"])
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
        if meta.get("ensemble") != ensemble.as_dict() or \
                meta["source"].get("system_sha256") != \
                system_digest(prep.system):
            raise ResumeError(
                "The build in progress was started from another System or "
                "in another ensemble (constant volume or pressure, and the "
                "barostat's settings). Resume with the same ones, or start "
                "again in a new directory.", code="resremd.resume.mismatch")
        saved_rest2 = meta.get("rest2")
        if rest2_meta is not None and saved_rest2 is not None and \
                saved_rest2.get("dispersion_correction") != \
                rest2_meta["dispersion_correction"]:
            if "dispersion_correction" not in saved_rest2 and \
                    rest2_meta["dispersion_correction"] in ("none",
                                                            "unchanged"):
                # No correction on the solute: nothing changed.
                saved_rest2 = {**saved_rest2, "dispersion_correction":
                               rest2_meta["dispersion_correction"]}
                meta["rest2"] = saved_rest2
            elif "dispersion_correction" not in saved_rest2:
                raise ResumeError(
                    "This REST2 build was started by an earlier ResREMD, "
                    "whose handling of the dispersion correction cannot be "
                    "told from its records, so the energies recorded for its "
                    "frames might mix two conventions. Start the build again "
                    "in a new directory.", code="resremd.resume.mismatch")
            else:
                raise ResumeError(
                    "The REST2 build in progress handled the dispersion "
                    f"correction as {saved_rest2['dispersion_correction']!r} "
                    f"and would now handle it as "
                    f"{rest2_meta['dispersion_correction']!r}: the ensemble, "
                    "the System's nonbonded settings or the OpenMM version "
                    "changed. Resume with the same ones, or start again.",
                    code="resremd.resume.mismatch")
        if meta["n_frames"] != n_frames \
                or meta["source"].get("frame_interval_steps") != interval \
                or meta["source"].get("timestep_fs") != dt \
                or meta["temperature_K"] != t \
                or meta.get("rest2") != rest2_meta \
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
        if rest2_meta is not None:
            meta["rest2"] = rest2_meta
        write_json(meta_file, meta)
    session_start = time.time()
    rng = np.random.default_rng([int(seed), 2, done])
    integrator = make_integrator(o["integrator"], t_sim, o["friction_per_ps"],
                                 dt,
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
        context.setParameter(ensemble.temperature_parameter, t_sim)
    if rest2_meta is not None:
        from .rest2 import set_scale

        set_scale(context, rest2_meta["scale"])
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
        context.setVelocitiesToTemperature(t_sim * unit.kelvin,
                                           int(rng.integers(1, 2**31 - 1)))
        eq_left = int(round(o["equilibration_ns"] * 1e6 / dt))
        logger.info("Equilibrating at %g K (%d steps)", t_sim, eq_left)

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
                    f"The simulation at {t_sim:g} K blew up at frame {k}.",
                    code="resremd.simulation.unstable")
            writer.write(k, pos, b)
            potential[k] = energy
            if angles is not None:
                angles[k] = torsion_angles_deg(pos, watch)
            done = k + 1
            converged = False
            scheduled = done % per_checkpoint == 0 or done == n_frames
            # Read once: a signal between two reads would stop the build
            # without the checkpoint that makes it resumable.
            stopping = stop.requested
            if scheduled or stopping:
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
                                   bias_weights(bias_energy[:done], t_sim))
                    moves = basin_transitions(
                        angles[:done], weights=None if bias_energy is None
                        else bias_weights(bias_energy[:done], t_sim))
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
            if stopping and done < n_frames:
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
            else bias_weights(np.asarray(bias_energy), t_sim))
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
        weights = bias_weights(np.asarray(bias_energy), t_sim)
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


def _minimiser(prep: Prepared, o: dict[str, Any]):
    """Minimise one frame with the prepared System."""
    import openmm

    ctx, _ = create_context(prep.system, openmm.VerletIntegrator(0.001),
                            platform=o["platform"], precision="mixed",
                            device=None, cpu_threads=None)

    def minimise(pos: np.ndarray, box: np.ndarray | None) -> np.ndarray:
        if box is not None:
            ctx.setPeriodicBoxVectors(*(openmm.Vec3(*map(float, r))
                                        for r in box))
        ctx.setPositions(pos)
        openmm.LocalEnergyMinimizer.minimize(ctx, 10.0, o["minimize_steps"])
        state = ctx.getState(getPositions=True)
        return np.asarray(state.getPositions(asNumpy=True)._value)

    return minimise


#: File types that carry their own atoms. Their frames are matched to the
#: reference topology by name; other formats are read in its order.
_WITH_ATOMS = (".pdb", ".ent", ".cif", ".pdbx", ".mmcif", ".gro", ".mol2",
               ".h5")


def _atom_order(file_top: Any, ref_top: Any, source: str) -> np.ndarray | None:
    """Indices that put a file's atoms in the reference order, matched by
    residue order and atom name; None when they already are."""
    ref_res = list(ref_top.residues)
    file_res = list(file_top.residues)
    if len(ref_res) != len(file_res):
        raise ReservoirError(
            f"{source} has {len(file_res)} residues and the system "
            f"{len(ref_res)}.", code="resremd.reservoir.mismatch")
    order = []
    for k, (rr, fr) in enumerate(zip(ref_res, file_res)):
        if rr.name != fr.name:
            raise ReservoirError(
                f"{source}: residue {k + 1} is {fr.name}, the system's is "
                f"{rr.name}.", code="resremd.reservoir.mismatch")
        names = {}
        for a in fr.atoms:
            if a.name in names:
                raise ReservoirError(
                    f"{source}: residue {k + 1} ({fr.name}) has two atoms "
                    f"named {a.name}.", code="resremd.reservoir.mismatch")
            names[a.name] = a.index
        want = [a.name for a in rr.atoms]
        missing = [n for n in want if n not in names]
        extra = sorted(set(names) - set(want))
        if missing or extra:
            text = f"{source}: residue {k + 1} ({fr.name})"
            if missing:
                text += f" lacks {', '.join(missing[:5])}"
            if extra:
                text += (";" if missing else "") + \
                    f" has {', '.join(extra[:5])}, which the system does not"
            raise ReservoirError(
                text + ". Prepare the structures with the same force field "
                "and hydrogen names as the system.",
                code="resremd.reservoir.mismatch")
        order += [names[n] for n in want]
    order = np.asarray(order)
    return None if np.array_equal(order, np.arange(order.size)) else order


def _read_frames(md, files: list[str], ref_top: Any, chunk: int = 1000):
    """Chunks of (file, xyz, unit cell vectors or None), in the reference
    atom order."""
    for f in files:
        if Path(f).suffix.lower() in _WITH_ATOMS:
            t = md.load(f)
            order = _atom_order(t.topology, ref_top, f)
            xyz = t.xyz if order is None else t.xyz[:, order]
            yield f, xyz, t.unitcell_vectors
            continue
        for part in md.iterload(f, top=ref_top, chunk=chunk):
            if part.n_atoms != ref_top.n_atoms:
                raise ReservoirError(
                    f"{f} has {part.n_atoms} atoms and the topology "
                    f"{ref_top.n_atoms}.", code="resremd.reservoir.mismatch")
            yield f, part.xyz, part.unitcell_vectors


def _amber_reservoirs(files: list[str]) -> dict[str, Any] | None:
    """Temperature, energies (kJ/mol) and cluster bins of Amber reservoirs:
    NetCDF trajectories with the `eptot` and `temp0` variables cpptraj's
    `createreservoir` writes. None when the files are not; refused when
    only some are."""
    from scipy.io import netcdf_file

    found = []
    for f in files:
        if Path(f).suffix.lower() not in (".nc", ".ncdf", ".netcdf") \
                or not Path(f).is_file():
            found.append(None)
            continue
        try:
            # Memory-mapped, so only these small variables are read, not
            # the coordinates; copies outlive the file.
            with netcdf_file(f, "r", mmap=True) as nc:
                v = nc.variables
                entry = None
                if "eptot" in v and "temp0" in v:
                    entry = {
                        "temperature_K": float(np.array(v["temp0"].data)),
                        "energies_kjmol": np.array(v["eptot"].data,
                                                   dtype=float) * 4.184,
                        "bins": np.array(v["bins"].data, dtype=int)
                        if "bins" in v else None}
                del v
            found.append(entry)
        except (OSError, TypeError, ValueError):
            found.append(None)
    if all(x is None for x in found):
        return None
    if any(x is None for x in found):
        raise ReservoirError(
            "Some of the files are Amber reservoirs and some are not; "
            "import them separately.", code="resremd.reservoir.mismatch")
    temps = {x["temperature_K"] for x in found}
    if len(temps) > 1:
        raise ReservoirError(
            f"The Amber reservoirs were made at different temperatures "
            f"({', '.join(f'{t:g}' for t in sorted(temps))} K).",
            code="resremd.reservoir.temperature")
    bins = None if any(x["bins"] is None for x in found) \
        else np.concatenate([x["bins"] for x in found])
    return {"temperature_K": temps.pop(),
            "energies_kjmol": np.concatenate([x["energies_kjmol"]
                                              for x in found]),
            "bins": bins}


def import_trajectories(**settings: Any) -> dict[str, Any]:
    """Build a reservoir from existing trajectories or structures.

    Settings are those of :data:`resremd.options.IMPORT`. Needs MDTraj.
    """
    from openmm import app

    o = resolve_options(IMPORT, settings)
    md = require("mdtraj", "Reading trajectories into a reservoir", "import")
    kind = o["kind"]
    amber = _amber_reservoirs([str(f) for f in o["trajectories"]])
    if amber is not None:
        if o["temperature_K"] is None:
            o["temperature_K"] = amber["temperature_K"]
        elif abs(o["temperature_K"] - amber["temperature_K"]) > 1e-6:
            raise InputError(
                f"The Amber reservoir was made at {amber['temperature_K']:g} "
                f"K, not the {o['temperature_K']:g} K given.",
                code="resremd.input.temperature")
    if o["clusterinfo"]:
        if amber is None or amber["bins"] is None:
            raise InputError(
                "`clusterinfo` weights the frames of Amber reservoirs by "
                "their cluster bins, and these files carry none.",
                code="resremd.input.clusters")
        if kind == "non_boltzmann" or o["weights"]:
            raise InputError(
                "`clusterinfo` makes the weights of a weighted reservoir; "
                "give it without `weights` or a non-Boltzmann kind.",
                code="resremd.input.clusters")
        kind = "weighted"
    if kind == "weighted" and not o["weights"] and not o["clusterinfo"]:
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
    if o["minimize_steps"] and kind != "non_boltzmann":
        raise InputError(
            "Minimised frames are no longer a Boltzmann sample; "
            "`minimize_steps` is for a non_boltzmann reservoir only.",
            code="resremd.input.minimize")
    prep = load_prepared(o["prepared"]) if o["prepared"] else None
    if o["minimize_steps"] and prep is None:
        raise InputError("`minimize_steps` needs `prepared`, whose System "
                         "minimises the frames.", code="resremd.input.missing")
    if prep is None and not o["topology"]:
        raise InputError("Give `topology` or `prepared`: the atoms the "
                         "reservoir is for.", code="resremd.input.missing")
    omm_topology = prep.topology if prep is not None \
        else app.PDBFile(o["topology"]).topology
    ref_top = md.Topology.from_openmm(omm_topology)
    if prep is not None and o["topology"]:
        # Frames in files without atoms are read in this topology's order,
        # which must be the prepared system's.
        read_top = md.load_topology(o["topology"])
        if _atom_order(read_top, ref_top, o["topology"]) is not None:
            raise ReservoirError(
                f"{o['topology']} lists the system's atoms in another order. "
                "Give files that carry their own atoms, or a topology in the "
                "prepared system's order.", code="resremd.reservoir.mismatch")
    n_atoms = omm_topology.getNumAtoms()
    files = [str(f) for f in o["trajectories"]]
    for f in files:
        if not Path(f).is_file():
            raise InputError(f"No trajectory at {f}.",
                             code="resremd.input.missing")
    lossy = [f for f in files if Path(f).suffix.lower() in (".xtc", ".gro",
                                                            ".pdb")]
    if lossy and kind != "non_boltzmann":
        # A non-Boltzmann reservoir does not mind: its exchanges use only
        # the energies of the coordinates as stored.
        logger.warning(
            "%s store coordinates rounded to 0.001 nm or coarser. Rounding "
            "stretches every bond a little, which raises the energy by "
            "several kJ/mol for a solvated system, so the frames are no "
            "longer a Boltzmann sample. Prefer DCD, TRR or NetCDF.",
            ", ".join(lossy))

    boxes: list[np.ndarray] = []
    total = 0
    periodic = None
    for f, xyz, cell in _read_frames(md, files, ref_top):
        has_box = cell is not None
        if periodic is None:
            periodic = has_box
        elif periodic != has_box:
            raise ReservoirError("Some trajectories have boxes and some "
                                 "do not.", code="resremd.reservoir.box")
        if has_box:
            boxes.append(np.asarray(cell, dtype=float))
        total += len(xyz)
    shared_box = None
    if prep is not None and periodic and not prep.periodic:
        # Structures with a box (a CRYST1 record) for a system without one:
        # the box means nothing to it.
        periodic = False
        boxes = []
        logger.info("The frames carry boxes; the prepared system has none, "
                    "so they are dropped.")
    if prep is not None and bool(periodic) != prep.periodic:
        if prep.periodic and prep.box is not None and \
                o["pressure_bar"] is None:
            # Structures without a box, for a periodic system at constant
            # volume: each takes the system's box.
            periodic = True
            shared_box = np.asarray(prep.box, dtype=float)
            boxes = [np.repeat(shared_box[None], total, axis=0)]
            logger.info("The frames carry no box; each is given the "
                        "prepared system's.")
        else:
            raise ReservoirError(
                "The frames and the prepared system disagree about periodic "
                "boundaries.", code="resremd.reservoir.box")
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
    built = None
    if amber is not None:
        if amber["energies_kjmol"].size != total:
            raise ReservoirError(
                f"The Amber reservoirs record {amber['energies_kjmol'].size} "
                f"energies for {total} frames.",
                code="resremd.reservoir.mismatch")
        if kind != "non_boltzmann":
            built = amber["energies_kjmol"][keep]
        if o["clusterinfo"]:
            from .clusters import cluster_weights, read_populations

            weights = cluster_weights(amber["bins"][keep],
                                      read_populations(o["clusterinfo"]))
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
                "topology": str(Path(o["topology"]).resolve())
                if o["topology"] else None,
                "prepared": prep.source if prep is not None else None,
                "minimize_steps": o["minimize_steps"],
                "stride": o["stride"],
                "weights": str(Path(o["weights"]).resolve())
                if o["weights"] else None,
                "amber": None if amber is None else {
                    "temperature_K": amber["temperature_K"],
                    "energies": "eptot, kcal/mol converted to kJ/mol",
                    "cluster_bins": amber["bins"] is not None,
                    "clusterinfo": str(Path(o["clusterinfo"]).resolve())
                    if o["clusterinfo"] else None},
                "input_frames": total,
                "lossy_formats": lossy})
    writer = ReservoirWriter(out, n_frames=n_frames, n_atoms=n_atoms,
                             periodic=bool(periodic))
    wanted = set(keep.tolist())
    minimise = _minimiser(prep, o) if o["minimize_steps"] else None
    index = 0
    written = 0
    for f, xyz, cell in _read_frames(md, files, ref_top):
        for i in range(len(xyz)):
            if index in wanted:
                if not periodic:
                    b = None
                elif shared_box is None and cell is not None:
                    b = np.asarray(cell[i], dtype=float)
                else:
                    b = shared_box
                pos = np.asarray(xyz[i], dtype=float)
                if minimise is not None:
                    pos = minimise(pos, b)
                writer.write(written, pos, b)
                written += 1
            index += 1
    writer.flush()
    if weights is not None:
        np.save(out / "weights.npy", weights)
        meta["statistics"] = {
            "effective_frames_kish": float(1.0 / np.sum(weights ** 2))}
    if built is not None:
        # Amber's own energies: the run's Hamiltonian check compares them
        # with its own, as for a reservoir generated here.
        np.save(out / "build_potential_kjmol.npy", built)
    if amber is not None and amber["bins"] is not None:
        np.save(out / "cluster_labels.npy", amber["bins"][keep])
    write_topology(out / "topology.pdb", omm_topology,
                   np.array(writer.positions[0], dtype=float),
                   None if writer.box is None else np.array(writer.box[0]))
    meta["complete"] = True
    write_json(out / "reservoir.json", meta)
    logger.info("Reservoir written: %d frames (%s).", n_frames, kind)
    return meta
