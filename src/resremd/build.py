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
        if meta["n_frames"] != n_frames or meta["temperature_K"] != t \
                or meta["source"].get("bias_torsions") != biases:
            raise ResumeError(
                "The build in progress was started with a different "
                "temperature, length or bias.",
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
    else:
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
        for k in range(done, n_frames):
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
            done = k + 1
            if done % per_checkpoint == 0 or done == n_frames or stop.requested:
                writer.flush()
                potential.flush()
                if bias_energy is not None:
                    bias_energy.flush()
                _save_build_checkpoint(
                    chk_file, context, done, seed, periodic,
                    wall_seconds=wall_before + time.time() - session_start,
                    equilibration_steps=eq_steps, equilibration_left=0)
                rate = (done - start) * interval * dt / 1e6 / \
                    max(time.time() - t0, 1e-9) * 86400
                logger.info("reservoir frame %d/%d  %.1f ns/day", done,
                            n_frames, rate)
                if on_progress:
                    on_progress({"frames": done, "n_frames": n_frames,
                                 "ns_per_day": rate})
            if stop.requested and done < n_frames:
                status = "stopped"
                break
    if status != "complete":
        logger.info("Stopped at frame %d of %d. Resume with `resume`.", done,
                    n_frames)
        return meta

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
                "weights.npy")


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
                           equilibration_left: int) -> None:
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
            "equilibration_left": int(equilibration_left)})), **arrays)
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
