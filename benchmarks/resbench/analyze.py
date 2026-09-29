"""From a benchmark directory to tables: convergence, cost and agreement.

For every complete run, the samples at the lowest temperature are read in
order and the state populations and feature histogram are estimated from
the first n samples, for n on a logarithmic grid.

**Against a reference.** The error of each estimate is its total variation
(TV) distance from the reference populations, and the Jensen-Shannon
divergence of its histogram from the reference histogram less the
divergence that finite sampling alone would give (the floor: the mean over
multinomial draws of n samples from the reference histogram). The
reference is exact, read from a file of independent long runs, or pooled
from named methods' final stretches. A pooled reference leaves out the
repeat being scored, so no run is compared with itself, and it is flagged
when it is too imprecise for the threshold.

**Convergence** is the first time after which the error stays below the
threshold. A repeat that never gets there is censored. Medians and 95%
bootstrap intervals are over seeds, for each start separately and for
"both starts" (per seed, the later of the two), with every value listed
when there are six seeds or fewer.

**Cost** is MD steps over all replicas, counting equilibration and the
run's reservoir. With equal cost planned, every method's runs have the same
total and are compared at the same budget.

**MBAR** (`analysis: {mbar: true}`): the same populations estimated from
the frames of every temperature, weighted to the lowest by MBAR over the
run's energies, for the same stretches. Columns ending in `_mbar`.

**Agreement** is the TV distance between a seed's runs from opposite
starts. Their reservoirs are separate, so they are independent.

Outputs, in <benchmark>/analysis/: curves.csv, convergence.csv,
agreement.csv, reservoirs.csv, summary.json.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

import resremd
from resremd.convergence import (censored_median, convergence_time,
                                 js_divergence, populations, total_variation)

from . import jobs as jobmod
from . import spec as specs
from . import systems

logger = logging.getLogger("resbench")


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def _read_lowest(run: Path, system: systems.BenchSystem, prepared: Path
                 ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import mdtraj as md

    manifest = json.loads((run / "manifest.json").read_text())
    lowest = manifest["states"][0]
    if not lowest.get("trajectory"):
        raise ValueError(f"{run} saved no trajectory at the lowest "
                         "temperature; set `save_states` to include 0.")
    traj = md.load(str(run / lowest["trajectory"]),
                   top=str(run / "topology.pdb"))
    features = system.features(traj, prepared)
    return features, system.labels_from_features(features), manifest


def _read_state(run: Path, system: systems.BenchSystem, prepared: Path,
                manifest: dict[str, Any], state: int) -> np.ndarray | None:
    """State labels of the frames saved at one temperature, if any."""
    import mdtraj as md

    entry = manifest["states"][state]
    if not entry.get("trajectory"):
        return None
    traj = md.load(str(run / entry["trajectory"]),
                   top=str(run / "topology.pdb"))
    return system.labels_from_features(system.features(traj, prepared))


_NOT_SOLUTE = {"NA", "CL", "K", "MG", "CA2", "Na+", "Cl-", "K+"}


def _solute_frames(res, path: Path):
    """The reservoir's frames as an MDTraj trajectory of the solute atoms,
    read in chunks, so a solvated reservoir never sits in memory whole."""
    import mdtraj as md

    top = md.load_topology(str(path / "topology.pdb"))
    keep = np.array([a.index for a in top.atoms
                     if not a.residue.is_water
                     and a.residue.name not in _NOT_SOLUTE])
    if keep.size == top.n_atoms:
        return md.Trajectory(np.array(res.positions), top)
    pos = res.positions
    xyz = np.concatenate([np.asarray(pos[i:i + 1000][:, keep])
                          for i in range(0, len(pos), 1000)])
    return md.Trajectory(xyz, top.subset(keep))


class ReservoirFrames:
    """Features and labels of each reservoir's frames, read once."""

    def __init__(self, spec, out: Path, system) -> None:
        self.spec, self.out, self.system = spec, out, system
        self._cache: dict[Path, tuple] = {}

    def __call__(self, m, start, seed):

        from resremd.reservoir import Reservoir

        path = jobmod.reservoir_dir(self.out, self.spec, m, start, seed)
        if path not in self._cache:
            res = Reservoir.open(path)
            traj = _solute_frames(res, path)
            made_from = jobmod.reservoir_start(self.spec, m, start)
            f = self.system.features(
                traj, jobmod.prepared_dir(self.out, made_from))
            self._cache[path] = (path, res, f,
                                 self.system.labels_from_features(f))
        return self._cache[path]


def load_runs(spec, out: Path, system) -> tuple[dict, list[str]]:
    """Complete runs, keyed (method, start, seed), and what is missing."""
    loaded, missing = {}, []
    for m in spec["methods"]:
        if not specs.has_runs(m):
            continue
        for start in specs.method_starts(spec, m):
            for seed in spec["seeds"]:
                run = jobmod.run_dir(out, m["name"], start, seed)
                man = run / "manifest.json"
                status = json.loads(man.read_text()).get("status") \
                    if man.exists() else "absent"
                if status != "complete":
                    missing.append(f"{m['name']}/{start}/{seed}: {status}")
                    continue
                f, lab, manifest = _read_lowest(
                    run, system, jobmod.prepared_dir(out, start))
                loaded[(m["name"], start, seed)] = {
                    "features": f, "labels": lab, "manifest": manifest,
                    "summary": resremd.summarize(run), "run": run}
    return loaded, missing


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------

def _tail(d: dict, frac: float) -> slice:
    n = len(d["labels"])
    return slice(int(n * (1 - frac)), n)


def _normalised(h: np.ndarray) -> np.ndarray | None:
    s = h.sum()
    return h / s if s > 0 else None


def pooled(system, runs: list[dict], frac: float) -> dict[str, Any]:
    """Populations and histogram from the final stretch of several runs.

    Each run counts equally. The standard error is over runs.
    """
    k = len(system.states)
    pops = np.array([populations(d["labels"][_tail(d, frac)], k)
                     for d in runs])
    hists = [_normalised(system.histogram(d["features"][_tail(d, frac)]))
             for d in runs]
    hists = [h for h in hists if h is not None]
    return {"populations": pops.mean(axis=0),
            "stderr": (pops.std(axis=0, ddof=1) / np.sqrt(len(pops)))
            if len(pops) > 1 else np.full(k, np.nan),
            "histogram": np.mean(hists, axis=0) if hists else None,
            "runs": len(pops), "last_fraction": frac}


class Reference:
    """What each run is scored against."""

    def __init__(self, spec, system, loaded, out: Path) -> None:
        self.spec, self.system, self.loaded = spec, system, loaded
        ref = spec["reference"]
        self.kind = ref["kind"]
        t = spec["temperatures_K"][0]
        if self.kind == "exact":
            e = system.exact(t)
            self.full = {"populations": e["populations"],
                         "histogram": e["histogram"],
                         "stderr": np.zeros(len(system.states))}
        elif self.kind == "file":
            path = Path(ref["path"])
            if not path.is_absolute():
                path = out / path  # relative to the benchmark directory
            data = json.loads(path.read_text())
            if data.get("system") and data["system"] != spec["system"]:
                raise ValueError(f"{ref['path']} is a reference for another "
                                 "system.")
            if data.get("temperature_K") not in (None, t):
                raise ValueError(f"{ref['path']} is a reference at "
                                 f"{data['temperature_K']} K, not {t} K.")
            self.full = {k: np.asarray(v) if v is not None else None
                         for k, v in data.items()
                         if k in ("populations", "histogram", "stderr")}
        else:
            self.methods = set(ref["methods"])
            self.frac = float(ref["last_fraction"])
            pool = [d for key, d in loaded.items() if key[0] in self.methods]
            if len(pool) < 2:
                raise ValueError("A pooled reference needs at least two "
                                 "complete runs of the named methods.")
            self.full = pooled(system, pool, self.frac)
        self._cache: dict[tuple, dict] = {}

    def for_run(self, key: tuple[str, str, int]) -> dict[str, Any]:
        """The reference a given run is scored against.

        Pooled: without that method's runs of that seed (both starts), so a
        run never contributes to its own reference.
        """
        if self.kind != "pooled" or key[0] not in self.methods:
            return self.full
        leave = (key[0], key[2])
        if leave not in self._cache:
            pool = [d for k2, d in self.loaded.items()
                    if k2[0] in self.methods and (k2[0], k2[2]) != leave]
            self._cache[leave] = pooled(self.system, pool, self.frac)
        return self._cache[leave]

    def precision_warning(self, threshold: float) -> str | None:
        se = np.asarray(self.full.get("stderr"), dtype=float)
        if self.kind == "exact" or not np.isfinite(se).any():
            return None
        # The TV error a reference this uncertain adds by itself: each
        # population off by |N(0, se)|, mean se * sqrt(2/pi), halved.
        tv_se = 0.5 * math.sqrt(2 / math.pi) * float(np.nansum(se))
        if tv_se > threshold / 3:
            return (f"The reference is uncertain by about {tv_se:.3f} in TV "
                    f"distance, more than a third of the threshold "
                    f"({threshold}). Convergence times against it are not "
                    "reliable; use longer or more reference runs.")
        return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _grid(n: int, points: int) -> np.ndarray:
    lo = max(1, n // 200)
    g = np.unique(np.round(np.logspace(np.log10(lo), np.log10(n), points))
                  .astype(int))
    return g[g >= 1]


def _jsd(h: np.ndarray, ref: np.ndarray | None) -> float:
    if ref is None or h.sum() <= 0:
        return float("nan")
    return js_divergence(h, ref)


class Floor:
    """The JSD n independent samples of the reference would show."""

    def __init__(self, ref_hist: np.ndarray | None, draws: int = 8) -> None:
        self.ref = None if ref_hist is None else ref_hist.ravel() \
            / ref_hist.sum()
        self.draws = draws
        self._cache: dict[int, float] = {}
        self.rng = np.random.default_rng(0)

    def __call__(self, n: int) -> float:
        if self.ref is None:
            return float("nan")
        if n not in self._cache:
            self._cache[n] = float(np.mean([
                js_divergence(self.rng.multinomial(n, self.ref), self.ref)
                for _ in range(self.draws)]))
        return self._cache[n]


def _reservoir_cost(spec, out, m, start, seed, uses) -> tuple[float, float]:
    """MD steps and wall seconds of the reservoir a run used."""
    if not m.get("reservoir"):
        return 0.0, 0.0
    path = jobmod.reservoir_dir(out, spec, m, start, seed)
    meta = json.loads((path / "reservoir.json").read_text())
    cost = meta.get("cost")
    if cost is None:
        if "generate" in m["reservoir"]:
            raise ValueError(f"{path} records no cost; it cannot be charged "
                             "fairly. Rebuild it with this version.")
        return 0.0, 0.0
    steps, wall = float(cost["md_steps_total"]), float(cost["wall_seconds"])
    if m["reservoir"].get("shared") and \
            spec["analysis"]["amortize_shared_reservoir"]:
        share = max(1, uses.get(m["name"], 1))
        steps, wall = steps / share, wall / share
    return steps, wall


def _finite(x):
    if x is None:
        return None
    if isinstance(x, (float, np.floating)) and not np.isfinite(x):
        return None
    return x


def analyze(out: Path) -> dict[str, Any]:
    spec = jobmod.load_out(out)
    if spec["reference"]["kind"] == "none":
        raise ValueError(f"{out} has no reference to compare with "
                         "(`reference.kind: none`): it is for making one, "
                         "with `resbench reference`.")
    system = systems.get(spec["system"])
    a = spec["analysis"]
    k = len(system.states)
    loaded, missing = load_runs(spec, out, system)
    for item in missing:
        logger.warning("Left out (not complete): %s", item)
    if not loaded:
        raise ValueError(f"No complete runs under {out}/runs.")
    reference = Reference(spec, system, loaded, out)
    floor = Floor(reference.full.get("histogram"))
    frames = ReservoirFrames(spec, out, system)
    uses: dict[str, int] = {}
    for key in loaded:
        uses[key[0]] = uses.get(key[0], 0) + 1

    curve_rows, conv_rows = [], []
    curves: dict[tuple, dict[str, np.ndarray]] = {}
    for key, d in loaded.items():
        name, start, seed = key
        m = specs.method(spec, name)
        man = d["manifest"]
        ref = reference.for_run(key)
        n_rep = len(man["states"])
        dt_fs = man["settings"]["timestep_fs"]
        interval = man["settings"]["trajectory_interval_steps"]
        eq_steps = man["cost"]["md_steps"]["equilibration"]
        res_steps, res_wall = _reservoir_cost(spec, out, m, start, seed, uses)
        n = len(d["labels"])
        grid = _grid(n, int(a["points"]))
        disc = float(a["discard_fraction"])
        pops = np.array([populations(d["labels"][int(disc * g):g], k)
                         for g in grid])
        tv = np.array([total_variation(p, ref["populations"]) for p in pops])
        jsd = np.array([_jsd(system.histogram(
            d["features"][int(disc * g):g]), ref.get("histogram"))
            for g in grid])
        excess = jsd - np.array([floor(g - int(disc * g)) for g in grid])
        tv_mbar = np.full(len(grid), np.nan)
        if a["mbar"]:
            pm = _mbar_populations(d, system, out, start, grid, disc,
                                   spec["temperatures_K"][0])
            if pm is not None:
                tv_mbar = np.array([total_variation(p, ref["populations"])
                                    for p in pm])
        time_ns = grid * interval * dt_fs / 1e6
        cost = res_steps + eq_steps + grid * interval * n_rep
        curves[key] = {"time_ns": time_ns, "pops": pops}
        for i, g in enumerate(grid):
            curve_rows.append({
                "method": name, "start": start, "seed": seed,
                "frames": int(g), "time_ns": time_ns[i],
                "cost_md_steps": cost[i], "tv_error": tv[i],
                "jsd_bits": jsd[i], "jsd_excess_bits": excess[i],
                "tv_error_mbar": tv_mbar[i],
                **{f"pop_{s}": pops[i, j]
                   for j, s in enumerate(system.states)},
            })
        t_tv = convergence_time(time_ns, tv, a["threshold_tv"])
        t_js = convergence_time(time_ns, excess, a["threshold_jsd"]) \
            if np.isfinite(excess).any() else None
        t_mbar = convergence_time(time_ns, tv_mbar, a["threshold_tv"]) \
            if np.isfinite(tv_mbar).any() else None

        def cost_at(t, res=res_steps, eq=eq_steps, dt=dt_fs, nr=n_rep):
            return None if t is None else float(res + eq + t * 1e6 / dt * nr)

        summ = d["summary"]
        check = summ.get("reservoir_check") or {}
        cover = _coverage(spec, out, system, frames, m, start, seed, d)
        conv_rows.append({
            "method": name, "start": start, "seed": seed,
            "run_ns": time_ns[-1], "run_cost": float(cost[-1]),
            "converged_tv_ns": t_tv, "converged_tv_cost": cost_at(t_tv),
            "converged_jsd_ns": t_js, "converged_jsd_cost": cost_at(t_js),
            "final_tv_error": tv[-1], "final_jsd_excess_bits": excess[-1],
            "converged_tv_mbar_cost": cost_at(t_mbar),
            "final_tv_error_mbar": tv_mbar[-1],
            "reservoir_md_steps": res_steps,
            "equilibration_md_steps": eq_steps,
            "wall_seconds": sum(man["cost"]["wall_seconds"].values())
            + res_wall,
            "mean_neighbour_acceptance": float(np.mean(
                summ["neighbour_acceptance"])),
            "round_trips": summ["round_trips"],
            "reservoir_acceptance": summ.get("reservoir_acceptance"),
            "effective_reservoir_ancestors":
                summ.get("effective_reservoir_ancestors"),
            "lowest_state_fraction_from_reservoir":
                summ.get("lowest_state_fraction_from_reservoir"),
            "reservoir_check_z": check.get("z"),
            "reservoir_implied_K": check.get(
                "reservoir_temperature_implied_K"),
            # Capped, so a certain disagreement (infinite z) survives JSON.
            "coverage_max_abs_z": None if cover.get("max_abs_z") is None
            else min(float(cover["max_abs_z"]), 999.0),
            "coverage_unsupported": " ".join(
                system.states[i] for i in cover.get("unsupported_states", [])),
            **{f"final_pop_{s}": pops[-1, j]
               for j, s in enumerate(system.states)},
        })

    agree_rows = _agreement(spec, curves, a["threshold_tv"])
    res_rows = _reservoir_rows(spec, out, system, frames)
    adir = out / "analysis"
    adir.mkdir(exist_ok=True)
    for name, rows in (("curves", curve_rows), ("convergence", conv_rows),
                       ("agreement", agree_rows), ("reservoirs", res_rows)):
        _write_csv(adir / f"{name}.csv", rows)

    summary = {
        "benchmark": spec["name"], "system": spec["system"],
        "states": list(system.states),
        "temperature_K": spec["temperatures_K"][0],
        "equal_cost": spec["equal_cost"],
        "reference": {
            "kind": reference.kind,
            "populations": np.asarray(reference.full["populations"]).tolist(),
            "stderr": np.asarray(reference.full["stderr"]).tolist()
            if reference.full.get("stderr") is not None else None,
            "left_out_per_run": reference.kind == "pooled",
            "warning": reference.precision_warning(a["threshold_tv"]),
        },
        "thresholds": {"tv": a["threshold_tv"],
                       "jsd_excess": a["threshold_jsd"]},
        "mbar": bool(a["mbar"]),
        "missing_runs": missing,
        "methods": {m["name"]: _method_summary(spec, m, conv_rows, agree_rows,
                                               res_rows, system)
                    for m in spec["methods"]
                    if any(r["method"] == m["name"] for r in conv_rows)},
    }
    text = json.dumps(_clean(summary), indent=2, allow_nan=False)
    (adir / "summary.json").write_text(text + "\n")
    return summary


def _mbar_populations(d, system, out, start, grid, disc, temperature_K
                      ) -> np.ndarray | None:
    """Populations at the lowest temperature from every saved temperature,
    by MBAR, for the same stretches as the lowest-temperature estimates."""
    from resremd.mbar import TemperatureReweighting

    rw = TemperatureReweighting(d["run"])
    if len(rw.saved_states) < 2:
        logger.warning("%s saved one temperature; no MBAR estimate.",
                       d["run"])
        return None
    k = len(system.states)
    labels = {}
    for state in rw.saved_states:
        lab = _read_state(d["run"], system, jobmod.prepared_dir(out, start),
                          d["manifest"], state)
        if len(lab) != rw.n_frames:
            raise ValueError(f"{d['run']}: state {state} has {len(lab)} "
                             f"frames, the tables {rw.n_frames}.")
        labels[state] = lab
    pops = []
    for g in grid:
        first = min(int(disc * g), int(g) - 1)
        try:
            w = rw.weights(temperature_K, frames=(first, int(g)))
        except RuntimeError:            # too little overlap yet
            pops.append(np.full(k, np.nan))
            continue
        p = np.zeros(k)
        for state, ws in w["weights"].items():
            p += np.bincount(labels[state][first:int(g)], weights=ws,
                             minlength=k)[:k]
        pops.append(p)
    return np.array(pops)


def _coverage(spec, out, system, frames, m, start, seed, d
              ) -> dict[str, Any]:
    """The coverage check for one run, or {} when it cannot be made."""
    if not m.get("reservoir"):
        return {}
    top = _read_state(d["run"], system, jobmod.prepared_dir(out, start),
                      d["manifest"], -1)
    if top is None:
        logger.warning("%s saved no top-temperature trajectory; no coverage "
                       "check.", d["run"])
        return {}
    _, _, _, labels = frames(m, start, seed)
    out_ = resremd.reservoir_coverage(d["run"], top, labels,
                                      len(system.states))
    return out_ if out_ and out_.get("status") == "ok" else {}


def _agreement(spec, curves, threshold) -> list[dict[str, Any]]:
    rows = []
    for m in spec["methods"]:
        starts = specs.method_starts(spec, m)
        if len(starts) < 2:
            continue
        a0, a1 = starts[:2]
        for seed in spec["seeds"]:
            c0 = curves.get((m["name"], a0, seed))
            c1 = curves.get((m["name"], a1, seed))
            if c0 is None or c1 is None:
                continue
            common = np.intersect1d(c0["time_ns"], c1["time_ns"])
            if not len(common):
                continue
            i0 = np.searchsorted(c0["time_ns"], common)
            i1 = np.searchsorted(c1["time_ns"], common)
            dist = [total_variation(c0["pops"][x], c1["pops"][y])
                    for x, y in zip(i0, i1)]
            agreed = convergence_time(common, dist, threshold)
            for t, dv in zip(common, dist):
                rows.append({"method": m["name"], "seed": seed,
                             "starts": f"{a0}|{a1}", "time_ns": t,
                             "tv_between_starts": dv, "agreed_at_ns": agreed})
    return rows


def _median_block(values: list, limit: float) -> dict[str, Any]:
    out = censored_median(values, limit)
    if len(values) <= 6:
        out["values"] = [None if v is None else float(v) for v in values]
    return out


def _method_summary(spec, m, conv_rows, agree_rows, res_rows, system):
    name = m["name"]
    rows = [r for r in conv_rows if r["method"] == name]
    starts = specs.method_starts(spec, m)
    plan = spec["_plan"][name]
    per_start = {}
    for start in starts:
        rs = sorted((r for r in rows if r["start"] == start),
                    key=lambda r: r["seed"])
        if not rs:
            continue
        limit_ns = max(r["run_ns"] for r in rs)
        limit_cost = max(r["run_cost"] for r in rs)
        finals = np.array([[r[f"final_pop_{s}"] for s in system.states]
                           for r in rs])
        per_start[start] = {
            "runs": len(rs),
            "converged_tv_cost_md_steps": _median_block(
                [r["converged_tv_cost"] for r in rs], limit_cost),
            "converged_tv_ns": _median_block(
                [r["converged_tv_ns"] for r in rs], limit_ns),
            "converged_jsd_cost_md_steps": _median_block(
                [r["converged_jsd_cost"] for r in rs], limit_cost),
            "final_populations_mean": finals.mean(axis=0).tolist(),
            "final_populations_stderr": (finals.std(axis=0, ddof=1)
                                         / np.sqrt(len(finals))).tolist()
            if len(finals) > 1 else None,
            "final_tv_error_mean": float(np.mean(
                [r["final_tv_error"] for r in rs])),
        }
        if spec["analysis"]["mbar"]:
            per_start[start]["converged_tv_mbar_cost_md_steps"] = \
                _median_block([r["converged_tv_mbar_cost"] for r in rs],
                              limit_cost)
            per_start[start]["final_tv_error_mbar_mean"] = _mean_or_none(
                [r["final_tv_error_mbar"] for r in rs])
    # Both starts: per seed, the later convergence (None if either failed).
    both = []
    for seed in spec["seeds"]:
        vals = [r["converged_tv_cost"] for r in rows if r["seed"] == seed]
        if len(vals) == len(starts):
            both.append(None if any(v is None for v in vals) else max(vals))
    agreed = {r["seed"]: r["agreed_at_ns"] for r in agree_rows
              if r["method"] == name}
    zs = [r["reservoir_check_z"] for r in rows
          if r["reservoir_check_z"] is not None]
    halves = [r["halves_tv"] for r in res_rows
              if r["method"] == name and r["halves_tv"] is not None]
    cover = [r["coverage_max_abs_z"] for r in rows
             if r["coverage_max_abs_z"] is not None]
    unsupported = sorted({s for r in rows
                          for s in r["coverage_unsupported"].split()})
    res = m.get("reservoir")
    return {
        "reservoir": None if res is None else
        ("generate" if "generate" in res else "exact"),
        "note": m.get("note"),
        "runs": len(rows),
        "replicas": plan["replicas"],
        "production_steps": plan["production_steps"],
        "cost_per_run_md_steps": plan["total_steps"],
        "reservoir_md_steps": plan["reservoir_steps"],
        "starts": per_start,
        "both_starts_converged_tv_cost_md_steps": _median_block(
            both, max(r["run_cost"] for r in rows)) if both else None,
        "starts_agree_ns": _median_block(
            list(agreed.values()), max(r["run_ns"] for r in rows))
        if agreed else None,
        "mean_neighbour_acceptance": float(np.mean(
            [r["mean_neighbour_acceptance"] for r in rows])),
        "reservoir_acceptance": _mean_or_none(
            [r["reservoir_acceptance"] for r in rows]),
        "effective_reservoir_ancestors": _mean_or_none(
            [r["effective_reservoir_ancestors"] for r in rows]),
        "reservoir_check_z_mean": float(np.mean(zs)) if zs else None,
        "reservoir_check_z_max_abs": float(np.max(np.abs(zs))) if zs
        else None,
        "reservoir_implied_K_mean": _mean_or_none(
            [r["reservoir_implied_K"] for r in rows]),
        "reservoir_halves_tv_mean": float(np.mean(halves)) if halves
        else None,
        "coverage_z_max_abs": float(np.max(cover)) if cover else None,
        "coverage_unsupported_states": unsupported,
        "coverage_runs_flagged": sum(1 for r in rows
                                     if r["coverage_unsupported"]),
    }


def _reservoir_rows(spec, out: Path, system: systems.BenchSystem,
                    frames: ReservoirFrames) -> list[dict[str, Any]]:
    """The reservoirs' own state populations, whole and by halves.

    Halves in the order the frames were made: a reservoir whose halves
    disagree had not converged at its own temperature, which no amount of
    replica exchange downstream can repair.
    """
    rows = []
    k = len(system.states)
    seen = set()
    for m in spec["methods"]:
        if not m.get("reservoir"):
            continue
        for start in specs.method_starts(spec, m):
            for seed in spec["seeds"]:
                path = jobmod.reservoir_dir(out, spec, m, start, seed)
                if path in seen or not (path / "reservoir.json").exists():
                    continue
                seen.add(path)
                _, res, _, lab = frames(m, start, seed)
                half = len(lab) // 2
                w = res.weights
                pops = (populations(lab, k) if w is None else
                        np.bincount(lab, weights=w, minlength=k)[:k]
                        / w.sum())
                a, b = populations(lab[:half], k), populations(lab[half:], k)
                shared = m["reservoir"].get("shared")
                rows.append({
                    "method": m["name"],
                    "start": "shared" if shared else start,
                    "seed": "shared" if shared else seed,
                    "kind": res.kind, "temperature_K": res.temperature_K,
                    "frames": res.n_frames,
                    "md_steps": (res.meta.get("cost") or {}).get(
                        "md_steps_total", 0),
                    "halves_tv": total_variation(a, b) if w is None
                    else None,
                    **{f"pop_{s}": pops[j]
                       for j, s in enumerate(system.states)},
                })
    return rows


def reservoir_estimates(spec, out: Path, system, names: list[str]
                        ) -> dict[str, Any]:
    """Populations and histogram at the lowest temperature, from reservoirs
    built at that temperature (for example under a bias, and weighted).

    Each reservoir is one independent estimate; the standard error is over
    reservoirs.
    """

    from resremd.reservoir import Reservoir

    t_min = float(spec["temperatures_K"][0])
    k = len(system.states)
    pops, hists, kish, paths = [], [], [], []
    for name in names:
        m = specs.method(spec, name)
        if not m.get("reservoir"):
            raise ValueError(f"Method {name} has no reservoir.")
        for start in specs.method_starts(spec, m):
            for seed in spec["seeds"]:
                path = jobmod.reservoir_dir(out, spec, m, start, seed)
                if path in paths or not (path / "reservoir.json").exists():
                    continue
                res = Reservoir.open(path)
                if res.temperature_K is None or \
                        abs(res.temperature_K - t_min) > 1e-6:
                    raise ValueError(
                        f"{path} is at {res.temperature_K} K; a reference "
                        f"from reservoirs needs them at {t_min} K.")
                paths.append(path)
                traj = _solute_frames(res, path)
                made_from = jobmod.reservoir_start(spec, m, start)
                f = system.features(traj, jobmod.prepared_dir(out, made_from))
                lab = system.labels_from_features(f)
                w = res.weights if res.weights is not None \
                    else np.full(len(lab), 1.0 / len(lab))
                pops.append(np.bincount(lab, weights=w, minlength=k)[:k]
                            / w.sum())
                (a0, a1, n0), (b0, b1, n1) = system.feature_bins
                h, _, _ = np.histogram2d(f[:, 0], f[:, 1], bins=[n0, n1],
                                         range=[[a0, a1], [b0, b1]],
                                         weights=w)
                if h.sum() > 0:
                    hists.append(h / h.sum())
                kish.append(float(1.0 / np.sum((w / w.sum()) ** 2)))
    if len(pops) < 2:
        raise ValueError("A reference from reservoirs needs at least two.")
    pops = np.array(pops)
    return {"populations": pops.mean(axis=0),
            "stderr": pops.std(axis=0, ddof=1) / np.sqrt(len(pops)),
            "histogram": np.mean(hists, axis=0) if hists else None,
            "runs": len(pops), "effective_frames_kish": kish}


def write_reference(out: Path, path: Path, methods: list[str] | None,
                    last_fraction: float, from_reservoirs: bool = False
                    ) -> dict[str, Any]:
    """A reference file for `kind: file`: from a benchmark of long runs, or
    from reservoirs built at the lowest temperature."""
    spec = jobmod.load_out(out)
    system = systems.get(spec["system"])
    missing: list[str] = []
    if from_reservoirs:
        if not methods:
            raise ValueError("Name the reservoir methods to use.")
        ref = reservoir_estimates(spec, out, system, methods)
    else:
        loaded, missing = load_runs(spec, out, system)
        pool = [d for key, d in loaded.items()
                if not methods or key[0] in methods]
        if len(pool) < 2:
            raise ValueError("A reference needs at least two complete runs.")
        ref = pooled(system, pool, last_fraction)
    data = {"system": spec["system"],
            "temperature_K": spec["temperatures_K"][0],
            "states": list(system.states),
            "populations": ref["populations"].tolist(),
            "stderr": ref["stderr"].tolist(),
            "histogram": None if ref["histogram"] is None
            else ref["histogram"].tolist(),
            "feature_names": list(system.feature_names),
            "feature_bins": [list(b) for b in system.feature_bins],
            "runs": ref["runs"],
            "last_fraction": None if from_reservoirs else last_fraction,
            "from_reservoirs": from_reservoirs,
            "from": str(out.resolve()), "methods": methods,
            "missing_runs": missing}
    path.write_text(json.dumps(_clean(data), indent=2, allow_nan=False)
                    + "\n")
    return data


def _mean_or_none(values):
    v = [x for x in values if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None


def _clean(x):
    """Plain JSON: arrays to lists, NaN and infinities to null."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.floating, float)):
        return _finite(float(x))
    if isinstance(x, np.integer):
        return int(x)
    return x


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if _finite(v) is None else v)
                        for k, v in r.items()})


def format_summary(summary: dict[str, Any]) -> str:
    def show(v, scale=1.0):
        return ">run" if v is None else f"{v / scale:.3g}"

    ref = summary["reference"]
    lines = [f"{summary['benchmark']}: {summary['system'].get('name')} at "
             f"{summary['temperature_K']:g} K"
             + (", equal cost" if summary["equal_cost"] else ""),
             "reference ({}): {}".format(ref["kind"], ", ".join(
                 f"{s} {p:.3f}" for s, p in zip(summary["states"],
                                                ref["populations"])))]
    if ref.get("warning"):
        lines.append("WARNING: " + ref["warning"])
    if summary["missing_runs"]:
        lines.append(f"left out, not complete: {len(summary['missing_runs'])}"
                     " runs")
    head = (f"{'method':<20}{'start':<11}{'n':>3}"
            f"{'cost to converge, 1e6 steps (95% CI)':>38}"
            f"{'final TV':>10}{'res acc':>9}{'max|z|':>8}{'halves':>8}"
            f"{'cover':>8}")
    lines += [head, "-" * len(head)]
    flagged = []
    for name, m in summary["methods"].items():
        acc = "" if m["reservoir_acceptance"] is None \
            else f"{m['reservoir_acceptance']:.2f}"
        z = "" if m["reservoir_check_z_max_abs"] is None \
            else f"{m['reservoir_check_z_max_abs']:.1f}"
        halves = "" if m["reservoir_halves_tv_mean"] is None \
            else f"{m['reservoir_halves_tv_mean']:.3f}"
        cover = "" if m.get("coverage_z_max_abs") is None \
            else f"{m['coverage_z_max_abs']:.1f}"
        if m.get("coverage_unsupported_states"):
            cover += "!"
            flagged.append(f"{name}: the top replica visits "
                           + ", ".join(m["coverage_unsupported_states"])
                           + " but the reservoir never holds it, in "
                           f"{m['coverage_runs_flagged']} of {m['runs']} "
                           "runs")
        rows = list(m["starts"].items())
        if m.get("both_starts_converged_tv_cost_md_steps"):
            rows.append(("both", {
                "runs": len(m["both_starts_converged_tv_cost_md_steps"]
                            .get("values", [])) or "",
                "converged_tv_cost_md_steps":
                    m["both_starts_converged_tv_cost_md_steps"],
                "final_tv_error_mean": None}))
        for i, (start, s) in enumerate(rows):
            c = s["converged_tv_cost_md_steps"]
            ci = (f"{show(c['median'], 1e6)} ({show(c['low'], 1e6)}-"
                  f"{show(c['high'], 1e6)})")
            ftv = "" if s["final_tv_error_mean"] is None \
                else f"{s['final_tv_error_mean']:.3f}"
            lines.append(
                f"{name if i == 0 else '':<20}{start:<11}{s['runs']:>3}"
                f"{ci:>38}{ftv:>10}"
                + (f"{acc:>9}{z:>8}{halves:>8}{cover:>8}" if i == 0
                   else ""))
    if summary.get("mbar"):
        lines += ["", "With every temperature, by MBAR:",
                  f"{'method':<20}{'start':<11}"
                  f"{'cost to converge, 1e6 steps (95% CI)':>38}"
                  f"{'final TV':>10}"]
        for name, m in summary["methods"].items():
            for i, (start, s) in enumerate(m["starts"].items()):
                c = s.get("converged_tv_mbar_cost_md_steps")
                if c is None:
                    continue
                ci = (f"{show(c['median'], 1e6)} ({show(c['low'], 1e6)}-"
                      f"{show(c['high'], 1e6)})")
                ftv = "" if s.get("final_tv_error_mbar_mean") is None \
                    else f"{s['final_tv_error_mbar_mean']:.3f}"
                lines.append(f"{name if i == 0 else '':<20}{start:<11}"
                             f"{ci:>38}{ftv:>10}")
    lines += ["", "max|z|: reservoir temperature check; halves: TV between "
              "the reservoir's halves; cover: coverage check, max |z| over "
              "states ('!': a state missing from the reservoir)"]
    lines += ["MISSING STATE: " + f for f in flagged]
    return "\n".join(lines)
