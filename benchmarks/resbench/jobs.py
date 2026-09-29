"""Turning a spec into jobs, and running them here or on a SLURM cluster.

A benchmark directory holds everything a spec produces:

    spec.yml                 the spec as run
    prepared/<start>/        system.xml, state.xml, topology.pdb per start
    reservoirs/<method>/<start>/seed_<n>/   (or <method>/shared/)
    runs/<method>/<start>/seed_<n>/
    jobs/prepare.txt, reservoirs.txt, runs.txt
    slurm/*.sbatch, slurm/submit.sh
    analysis/

Every job can be run again: a finished one is skipped, and an interrupted
run or reservoir build is resumed. Each job's output records a fingerprint
of the settings that made it, and output made under other settings is
refused rather than reused.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

import resremd
from resremd.system import load_prepared, write_prepared

from . import spec as specs
from . import systems

logger = logging.getLogger("resbench")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def prepared_dir(out: Path, start: str) -> Path:
    return out / "prepared" / start


def reservoir_dir(out: Path, spec: dict[str, Any], method: dict[str, Any],
                  start: str, seed: int) -> Path | None:
    res = method.get("reservoir")
    if res is None:
        return None
    if res.get("shared"):
        return out / "reservoirs" / method["name"] / "shared"
    return out / "reservoirs" / method["name"] / start / f"seed_{seed}"


def reservoir_start(spec: dict[str, Any], method: dict[str, Any],
                    start: str) -> str:
    """The start a reservoir is simulated from."""
    res = method["reservoir"]
    if res.get("shared"):
        return res.get("start") or next(iter(spec["starts"]))
    return start


def run_dir(out: Path, method: str, start: str, seed: int) -> Path:
    return out / "runs" / method / start / f"seed_{seed}"


def job_list(spec: dict[str, Any]) -> dict[str, list[str]]:
    """Job ids in three stages; each stage depends on the one before."""
    reservoirs, runs = [], []
    for m in spec["methods"]:
        res = m.get("reservoir")
        starts = specs.method_starts(spec, m)
        if res is not None:
            if res.get("shared"):
                reservoirs.append(f"reservoir/{m['name']}/shared/0")
            else:
                reservoirs += [f"reservoir/{m['name']}/{st}/{s}"
                               for st in starts for s in spec["seeds"]]
        if not specs.has_runs(m):
            continue
        for start in starts:
            runs += [f"run/{m['name']}/{start}/{s}" for s in spec["seeds"]]
    return {"prepare": ["prepare"], "reservoirs": reservoirs, "runs": runs}


def _record_path(path: Path) -> Path:
    # Beside the job's directory, not in it: resremd refuses to start in a
    # directory holding files it did not write.
    return path.with_name(path.name + ".resbench.json")


def _claim(path: Path, fingerprint: str) -> None:
    """Record the settings a job runs under, or refuse if others made it."""
    record = _record_path(path)
    if record.exists():
        old = json.loads(record.read_text()).get("fingerprint")
        if old != fingerprint:
            raise RuntimeError(
                f"{path} was made under different settings. Remove it (and "
                f"{record.name}), or plan the changed spec into a new "
                "benchmark directory.")
        return
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"{path} holds output with no record of the "
                           "settings that made it; remove it first.")
    path.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"fingerprint": fingerprint}) + "\n")


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def plan(spec_path: Path, out: Path) -> dict[str, list[str]]:
    spec = specs.load(spec_path)
    out.mkdir(parents=True, exist_ok=True)
    target = out / ("spec" + Path(spec_path).suffix)
    if target.resolve() != Path(spec_path).resolve():
        shutil.copyfile(spec_path, target)
    jobs = job_list(spec)
    (out / "jobs").mkdir(exist_ok=True)
    for stage, ids in jobs.items():
        (out / "jobs" / f"{stage}.txt").write_text(
            "".join(i + "\n" for i in ids))
    _write_slurm(spec, out, jobs)
    return jobs


def _write_slurm(spec: dict[str, Any], out: Path,
                 jobs: dict[str, list[str]]) -> None:
    s = spec["slurm"]
    d = out / "slurm"
    d.mkdir(exist_ok=True)
    root = out.resolve()
    for stage, ids in jobs.items():
        if not ids:
            continue
        lines = ["#!/bin/bash",
                 f"#SBATCH --job-name={spec['name']}-{stage}",
                 f"#SBATCH --output={root}/slurm/{stage}-%A_%a.log",
                 f"#SBATCH --array=1-{len(ids)}",
                 f"#SBATCH --cpus-per-task={s['cpus_per_task']}"]
        for key, flag in (("gres", "gres"), ("partition", "partition"),
                          ("time", "time"), ("account", "account")):
            if s.get(key):
                lines.append(f"#SBATCH --{flag}={s[key]}")
        lines += [f"#SBATCH {x}" for x in s.get("extra") or []]
        # Environment set-up first: conda's activation scripts are not
        # written for `set -u`.
        lines += [*s.get("setup", []), "set -euo pipefail"]
        lines += [
            f'JOB=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {root}/jobs/{stage}.txt)',
            f'cd {root}',
            'resbench job . "$JOB"',
        ]
        (d / f"{stage}.sbatch").write_text("\n".join(lines) + "\n")
    submit = ["#!/bin/bash", "set -euo pipefail", f"cd {root}/slurm",
              "dep=\"\""]
    for stage, ids in jobs.items():
        if not ids:
            continue
        submit.append(
            f'id=$(sbatch --parsable $dep {stage}.sbatch); '
            f'echo "{stage}: $id"; dep="--dependency=afterok:$id"')
    (d / "submit.sh").write_text("\n".join(submit) + "\n")
    (d / "submit.sh").chmod(0o755)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def load_out(out: Path) -> dict[str, Any]:
    for name in ("spec.yml", "spec.yaml", "spec.json"):
        if (out / name).exists():
            return specs.load(out / name)
    raise FileNotFoundError(f"{out} has no spec; run `plan` first.")


def run_job(out: Path, job: str) -> None:
    spec = load_out(out)
    kind, *rest = job.split("/")
    if kind == "prepare":
        prepare(spec, out)
    elif kind == "reservoir":
        name, start, seed = rest
        build_reservoir(spec, out, specs.method(spec, name), start, int(seed))
    elif kind == "run":
        name, start, seed = rest
        run_one(spec, out, specs.method(spec, name), start, int(seed))
    else:
        raise ValueError(f"Unknown job {job!r}.")


def run_all(out: Path) -> None:
    spec = load_out(out)
    for stage, ids in job_list(spec).items():
        for job in ids:
            logger.info("== %s", job)
            run_job(out, job)


def _platform(spec: dict[str, Any], system: systems.BenchSystem) -> str:
    return (spec["prepare"].get("platform") or system.platform
            or spec["run"].get("platform", "auto"))


def prepare(spec: dict[str, Any], out: Path) -> None:
    system = systems.get(spec["system"])
    root = out / "prepared"
    done = root / "prepared.json"
    fingerprint = specs.prepare_fingerprint(spec)
    if done.exists():
        if json.loads(done.read_text()).get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"{root} was prepared under different system settings. "
                "Plan the changed spec into a new benchmark directory.")
        logger.info("Prepared systems exist; skipping.")
        return
    platform = _platform(spec, system)
    seed = int(spec["prepare"]["seed"])
    system.prepare(root, temperature_K=spec["temperatures_K"][0],
                   platform=platform, seed=seed)
    record: dict[str, Any] = {"fingerprint": fingerprint, "starts": {}}
    base = None
    for name, how in spec["starts"].items():
        target = prepared_dir(out, name)
        if (target / "state.xml").exists():
            base = base or target
            record["starts"][name] = {"made": "prepared directly"}
            continue
        seek_cfg = how.get("seek")
        if seek_cfg is None:
            raise ValueError(f"Start {name!r} was not prepared by the "
                             "system and has no `seek`.")
        source = base or next(p for p in (root.iterdir())
                              if (p / "state.xml").exists())
        record["starts"][name] = seek(system, source, target, seek_cfg,
                                      platform=platform, seed=seed)
    done.write_text(json.dumps(record, indent=2) + "\n")


def seek(system: systems.BenchSystem, source: Path, target: Path,
         cfg: dict[str, Any], *, platform: str, seed: int) -> dict[str, Any]:
    """Run hot dynamics from a prepared start until a condition holds.

    The condition is a state (``state: alpha_L``) or a feature threshold
    (``feature: rmsd_ca_nm``, ``above`` or ``below``). The box is not
    changed, so the new start is the same system as the old.
    """
    import mdtraj as md
    import openmm
    from openmm import unit

    from resremd.system import create_context, make_integrator

    p = load_prepared(source)
    t = float(cfg.get("temperature_K", 600.0))
    check_steps = max(1, int(round(float(cfg.get("check_ps", 10)) * 500)))
    max_steps = int(round(float(cfg.get("max_ns", 20)) * 5e5))
    integ = make_integrator("langevin_middle", t, 1.0, 2.0, seed)
    seek_system = p.system
    if cfg.get("bias_torsions"):
        # Lower a barrier to get across it; the start is written with the
        # unbiased System.
        from resremd.build import add_torsion_biases

        seek_system, _ = add_torsion_biases(
            p.system, resolve_biases(system, source, cfg["bias_torsions"]),
            p.n_atoms)
    ctx, _ = create_context(seek_system, integ, platform=platform,
                            precision="mixed", device=None, cpu_threads=None)
    if p.box is not None:
        ctx.setPeriodicBoxVectors(*(openmm.Vec3(*r) for r in p.box))
    ctx.setPositions(p.positions)
    ctx.setVelocitiesToTemperature(t * unit.kelvin, seed)
    mdtop = md.Topology.from_openmm(p.topology)

    def satisfied(xyz: np.ndarray) -> bool:
        traj = md.Trajectory(xyz[None], mdtop)
        if "state" in cfg:
            label = system.labels(traj, source)[0]
            return system.states[label] == cfg["state"]
        f = system.features(traj, source)[0]
        value = f[system.feature_names.index(cfg["feature"])]
        if "above" in cfg:
            return value > float(cfg["above"])
        return value < float(cfg["below"])

    steps = 0
    while steps < max_steps:
        integ.step(check_steps)
        steps += check_steps
        state = ctx.getState(getPositions=True, getVelocities=True)
        xyz = np.asarray(state.getPositions(asNumpy=True)._value)
        if satisfied(xyz):
            # Settle the structure without leaving the state it was found in.
            openmm.LocalEnergyMinimizer.minimize(ctx, 10.0, 200)
            xyz = np.asarray(ctx.getState(getPositions=True)
                             .getPositions(asNumpy=True)._value)
            if not satisfied(xyz):
                continue
            write_prepared(target, p.system, p.topology, xyz, p.box)
            logger.info("Start %s found after %.1f ps at %g K", target.name,
                        steps * 0.002, t)
            return {"made": "sought", "from": source.name, "steps": steps,
                    "temperature_K": t, "condition": cfg}
    raise RuntimeError(f"Start {target.name} not reached in "
                       f"{cfg.get('max_ns', 20)} ns at {t:g} K.")


def build_reservoir(spec: dict[str, Any], out: Path, method: dict[str, Any],
                    start: str, seed: int) -> None:
    res = method["reservoir"]
    if res.get("shared"):
        start, seed = reservoir_start(spec, method, start), spec["seeds"][0]
    path = reservoir_dir(out, spec, method, start, seed)
    _claim(path, specs.reservoir_fingerprint(spec, method, start, seed))
    if (path / "reservoir.json").exists() and json.loads(
            (path / "reservoir.json").read_text()).get("complete"):
        logger.info("Reservoir %s exists; skipping.", path)
        return
    job_seed = specs.job_seed(seed, method["name"], start, 99)
    if "exact" in res:
        system = systems.get(spec["system"])
        if path.exists():
            shutil.rmtree(path)
        system.exact_reservoir(path, res["exact"], job_seed)
        return
    resume = (path / "build_checkpoint.npz").exists()
    settings = dict(res["generate"])
    if settings.get("bias_torsions"):
        system = systems.get(spec["system"])
        settings["bias_torsions"] = resolve_biases(
            system, prepared_dir(out, start), settings["bias_torsions"])
    resremd.generate_reservoir(prepared=str(prepared_dir(out, start)),
                               output=str(path), resume=resume,
                               random_seed=job_seed, **settings)


def resolve_biases(system: systems.BenchSystem, prepared: Path,
                   biases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace torsion names (``atoms: omega``) with atom indices."""
    names = None
    out = []
    for b in biases:
        b = dict(b)
        if isinstance(b.get("atoms"), str):
            if names is None:
                names = system.torsions(load_prepared(prepared).topology)
            if b["atoms"] not in names:
                raise ValueError(f"{system.name} names no torsion "
                                 f"{b['atoms']!r}; it names "
                                 f"{', '.join(names) or 'none'}.")
            b["atoms"] = names[b["atoms"]]
        out.append(b)
    return out


def run_one(spec: dict[str, Any], out: Path, method: dict[str, Any],
            start: str, seed: int) -> None:
    path = run_dir(out, method["name"], start, seed)
    _claim(path, specs.run_fingerprint(spec, method, start, seed))
    manifest = path / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) \
            .get("status") == "complete":
        logger.info("Run %s is complete; skipping.", path)
        return
    res = reservoir_dir(out, spec, method, start, seed)
    resremd.run(prepared=str(prepared_dir(out, start)), output=str(path),
                reservoir=None if res is None else str(res),
                temperatures_K=specs.ladder_of(spec, method),
                random_seed=specs.job_seed(seed, method["name"], start, 1),
                resume=(path / "checkpoint.npz").exists(),
                **specs.run_settings(spec, method))


def status(out: Path) -> list[tuple[str, str]]:
    spec = load_out(out)
    rows = []
    for stage, ids in job_list(spec).items():
        for job in ids:
            kind, *rest = job.split("/")
            if kind == "prepare":
                done = (out / "prepared" / "prepared.json").exists()
                rows.append((job, "done" if done else "pending"))
            elif kind == "reservoir":
                m = specs.method(spec, rest[0])
                p = reservoir_dir(out, spec, m, rest[1], int(rest[2])) \
                    / "reservoir.json"
                state = "pending"
                if p.exists():
                    state = "done" if json.loads(p.read_text()).get(
                        "complete") else "partial"
                rows.append((job, state))
            else:
                p = run_dir(out, rest[0], rest[1], int(rest[2])) \
                    / "manifest.json"
                state = "pending"
                if p.exists():
                    m = json.loads(p.read_text())
                    prog = m.get("progress") or {}
                    state = (f"{m['status']} "
                             f"{prog.get('cycles_done', 0)}/"
                             f"{prog.get('cycles_target', '?')}")
                rows.append((job, state))
    return rows
