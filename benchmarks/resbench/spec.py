"""A benchmark, described in one file.

    name: alanine_implicit
    system: {name: alanine_dipeptide, solvent: implicit}
    temperatures_K: [300.0, 332.0, 367.42, 406.62]
    seeds: [1, 2, 3, 4, 5]
    run: {duration_ns: 20, exchange_interval_steps: 500, platform: auto}
    methods:
      - name: remd
      - name: resremd
        reservoir:
          generate: {temperature_K: 450, duration_ns: 50,
                     frame_interval_steps: 2500}
    reference: {kind: file, path: reference.json}

Methods share the run settings, the seeds and, unless a method gives its
own `temperatures_K`, the ladder. Every ladder starts at the same lowest
temperature, the one the benchmark measures.

**Equal cost.** With `equal_cost: true` (the default) the production length
of each method is set so that every run costs the same number of MD steps
over all its replicas, counting equilibration and the run's reservoir. The
most expensive method keeps the length the spec gives; the others run
longer. Methods are then compared at the same budget, not the same length.

**Reservoirs** are made for every run separately (one per method, start and
seed), from that run's own start, so runs from different starts are
independent and no run inherits frames from another start. `shared: true`
makes one reservoir per method instead, from `start`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from resremd.options import GENERATE, RUN, resolve

from . import systems

#: Run settings the harness sets itself for every run.
_HARNESS_RUN = {"prepared", "output", "reservoir", "resume", "temperatures_K",
                "random_seed"}
_HARNESS_GENERATE = {"prepared", "output", "resume", "random_seed"}
_EXACT_KEYS = {"kind", "temperature_K", "sampled_temperature_K", "n_frames",
               "drop_state"}

DEFAULTS: dict[str, Any] = {
    "starts": None,
    "equal_cost": True,
    "reference": {"kind": None, "methods": None, "last_fraction": 0.5,
                  "path": None},
    "analysis": {"threshold_tv": 0.05, "threshold_jsd": 0.02,
                 "points": 40, "discard_fraction": 0.0,
                 "amortize_shared_reservoir": False},
    "slurm": {"partition": None, "gres": "gpu:1", "time": None,
              "cpus_per_task": 4, "account": None, "setup": [],
              "extra": []},
    "prepare": {"platform": None, "seed": 1},
}

_KEYS = set(DEFAULTS) | {"name", "description", "system", "temperatures_K",
                         "seeds", "run", "methods"}


class SpecError(ValueError):
    pass


def load(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text()
    if str(path).endswith((".yml", ".yaml")):
        import yaml

        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    return check(raw)


def check(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise SpecError("A benchmark spec is a mapping.")
    raw = {k: v for k, v in raw.items() if k != "_plan"}
    unknown = set(raw) - _KEYS
    if unknown:
        raise SpecError(f"Unknown spec keys: {', '.join(sorted(unknown))}.")
    for key in ("name", "system", "temperatures_K", "seeds", "run", "methods"):
        if key not in raw:
            raise SpecError(f"The spec needs `{key}`.")
    spec = copy.deepcopy(raw)
    for key, default in DEFAULTS.items():
        if isinstance(default, dict):
            spec[key] = {**default, **(spec.get(key) or {})}
        elif key not in spec:
            spec[key] = default
    system = systems.get(spec["system"])
    if spec["starts"] is None:
        spec["starts"] = system.default_starts()
    if not isinstance(spec["starts"], dict) or not spec["starts"]:
        raise SpecError("`starts` maps start names to how each is made.")
    seeds = spec["seeds"]
    if not seeds or any(not isinstance(s, int) or isinstance(s, bool)
                        for s in seeds) or len(set(seeds)) != len(seeds):
        raise SpecError("`seeds` are distinct integers.")
    _check_run(spec["run"], "run")
    t_min = float(spec["temperatures_K"][0])
    names = []
    for m in spec["methods"]:
        if "name" not in m:
            raise SpecError("Every method needs a `name`.")
        if m["name"] in names:
            raise SpecError(f"Method name {m['name']!r} is used twice.")
        names.append(m["name"])
        extra = set(m) - {"name", "reservoir", "run", "starts", "note",
                          "temperatures_K"}
        if extra:
            raise SpecError(f"Method {m['name']}: unknown keys "
                            f"{', '.join(sorted(extra))}.")
        ladder = ladder_of(spec, m)
        if float(ladder[0]) != t_min:
            raise SpecError(f"Method {m['name']}: its ladder starts at "
                            f"{ladder[0]} K, not the benchmark's {t_min} K.")
        _check_run({**spec["run"], **(m.get("run") or {}),
                    "temperatures_K": ladder}, f"method {m['name']}",
                   allow_ladder=True)
        _check_reservoir(spec, system, m)
        starts = m.get("starts")
        if starts is not None and not set(starts) <= set(spec["starts"]):
            raise SpecError(f"Method {m['name']}: unknown starts.")
    _check_reference(spec, system, names)
    spec["_plan"] = plan_costs(spec)
    return spec


def _check_run(run: dict[str, Any], where: str,
               allow_ladder: bool = False) -> None:
    bad = set(run) & _HARNESS_RUN - ({"temperatures_K"} if allow_ladder
                                     else set())
    if bad:
        raise SpecError(f"{where}: the harness sets {', '.join(sorted(bad))} "
                        "itself.")
    try:
        resolve(RUN, {**run, "prepared": "x"})
    except Exception as exc:
        raise SpecError(f"{where}: {exc}") from exc


def _check_reservoir(spec, system, m) -> None:
    res = m.get("reservoir")
    if res is None:
        return
    name = m["name"]
    extra = set(res) - {"generate", "exact", "shared", "start"}
    if extra:
        raise SpecError(f"Method {name}: unknown reservoir keys "
                        f"{', '.join(sorted(extra))}.")
    if res.get("start") is not None:
        if not res.get("shared"):
            raise SpecError(f"Method {name}: `start` is for a shared "
                            "reservoir; others are made from each run's "
                            "own start.")
        if res["start"] not in spec["starts"]:
            raise SpecError(f"Method {name}: reservoir start "
                            f"{res['start']!r} is not a start.")
    kinds = [k for k in ("generate", "exact") if k in res]
    if len(kinds) != 1:
        raise SpecError(f"Method {name}: a reservoir is either `generate` "
                        "or `exact`.")
    if "generate" in res:
        bad = set(res["generate"]) & _HARNESS_GENERATE
        if bad:
            raise SpecError(f"Method {name}: the harness sets "
                            f"{', '.join(sorted(bad))} itself.")
        try:
            resolve(GENERATE, {**res["generate"], "prepared": "x"})
        except Exception as exc:
            raise SpecError(f"Method {name}: {exc}") from exc
    else:
        if not hasattr(system, "exact_reservoir"):
            raise SpecError(f"{system.name} has no exact reservoirs.")
        ex = res["exact"]
        extra = set(ex) - _EXACT_KEYS
        if extra:
            raise SpecError(f"Method {name}: unknown exact reservoir keys "
                            f"{', '.join(sorted(extra))}.")
        if "n_frames" not in ex:
            raise SpecError(f"Method {name}: an exact reservoir needs "
                            "`n_frames`.")
        if ex.get("kind", "boltzmann") == "boltzmann" \
                and "temperature_K" not in ex:
            raise SpecError(f"Method {name}: a Boltzmann reservoir needs "
                            "`temperature_K`.")
        if ex.get("drop_state") is not None \
                and ex["drop_state"] not in system.states:
            raise SpecError(f"Method {name}: no state {ex['drop_state']!r}.")


def _check_reference(spec, system, names) -> None:
    ref = spec["reference"]
    if ref["kind"] is None:
        ref["kind"] = "exact" if system.exact(spec["temperatures_K"][0]) \
            is not None else None
    if ref["kind"] not in ("exact", "pooled", "file"):
        raise SpecError("Say what the runs are compared with: "
                        "`reference.kind` is exact, file (long independent "
                        "runs, see `resbench reference`) or pooled.")
    if ref["kind"] == "exact" and system.exact(spec["temperatures_K"][0]) \
            is None:
        raise SpecError(f"{system.name} has no exact answer.")
    if ref["kind"] == "file" and not ref.get("path"):
        raise SpecError("A file reference needs `reference.path`.")
    if ref["kind"] == "pooled":
        if not ref.get("methods"):
            raise SpecError("A pooled reference names the methods it pools "
                            "(`reference.methods`); pooling every method "
                            "would score each against itself.")
        unknown = set(ref["methods"]) - set(names)
        if unknown:
            raise SpecError(f"Reference methods not in the spec: {unknown}.")


# ---------------------------------------------------------------------------
# Derived: ladders, run settings, costs, seeds, fingerprints
# ---------------------------------------------------------------------------

def ladder_of(spec: dict[str, Any], method: dict[str, Any]) -> list[float]:
    return [float(t) for t in method.get("temperatures_K")
            or spec["temperatures_K"]]


def method(spec: dict[str, Any], name: str) -> dict[str, Any]:
    for m in spec["methods"]:
        if m["name"] == name:
            return m
    raise SpecError(f"No method {name!r}.")


def method_starts(spec: dict[str, Any], m: dict[str, Any]) -> list[str]:
    return list(m.get("starts") or spec["starts"])


def reservoir_steps(m: dict[str, Any]) -> int:
    """MD steps one reservoir of this method costs (0 for an exact one)."""
    res = m.get("reservoir") or {}
    if "generate" not in res:
        return 0
    g = resolve(GENERATE, {**res["generate"], "prepared": "x"})
    dt = g["timestep_fs"]
    interval = g["frame_interval_steps"]
    frames = int(round(g["duration_ns"] * 1e6 / dt)) // interval
    return int(round(g["equilibration_ns"] * 1e6 / dt)) + frames * interval


def plan_costs(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per method: replicas, steps per phase, and the production length.

    With equal cost, production is lengthened for the cheaper methods so
    every run costs what the most expensive one does (rounded up to a whole
    exchange interval).
    """
    plan = {}
    for m in spec["methods"]:
        o = resolve(RUN, {**spec["run"], **(m.get("run") or {}),
                          "prepared": "x"})
        n = len(ladder_of(spec, m))
        dt = o["timestep_fs"]
        interval = o["exchange_interval_steps"]
        prod = o["production_steps"] or int(round(o["duration_ns"] * 1e6
                                                   / dt))
        eq = int(round(o["equilibration_ns"] * 1e6 / dt))
        res = reservoir_steps(m)
        plan[m["name"]] = {"replicas": n, "timestep_fs": dt,
                           "exchange_interval_steps": interval,
                           "equilibration_steps": eq,
                           "reservoir_steps": res, "production_steps": prod}
    if spec["equal_cost"]:
        def total(p):
            return (p["reservoir_steps"] + p["replicas"]
                    * (p["equilibration_steps"] + p["production_steps"]))

        budget = max(total(p) for p in plan.values())
        for p in plan.values():
            need = (budget - p["reservoir_steps"]) / p["replicas"] \
                - p["equilibration_steps"]
            p["production_steps"] = max(p["production_steps"], int(
                math.ceil(need / p["exchange_interval_steps"] - 1e-9))
                * p["exchange_interval_steps"])
    for p in plan.values():
        p["total_steps"] = (p["reservoir_steps"] + p["replicas"]
                            * (p["equilibration_steps"]
                               + p["production_steps"]))
    return plan


def run_settings(spec: dict[str, Any], m: dict[str, Any]) -> dict[str, Any]:
    """What resremd.run is given for this method, production length fixed."""
    s = {**spec["run"], **(m.get("run") or {})}
    s.pop("duration_ns", None)
    s["production_steps"] = spec["_plan"][m["name"]]["production_steps"]
    return s


def job_seed(seed: int, method_name: str, start: str, salt: int) -> int:
    """An independent, reproducible seed for one job of one repeat."""
    import numpy as np

    key = int(hashlib.sha256(f"{method_name}|{start}".encode())
              .hexdigest()[:8], 16)
    return int(np.random.default_rng([seed, key, salt])
               .integers(1, 2**31 - 1))


def fingerprint(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str)
                          .encode()).hexdigest()[:16]


def prepare_fingerprint(spec: dict[str, Any]) -> str:
    return fingerprint(spec["system"], spec["starts"],
                       spec["temperatures_K"][0], spec["prepare"])


def reservoir_fingerprint(spec, m, start, seed) -> str:
    return fingerprint(prepare_fingerprint(spec), m["reservoir"], start, seed)


def run_fingerprint(spec, m, start, seed) -> str:
    return fingerprint(prepare_fingerprint(spec), m.get("reservoir"),
                       ladder_of(spec, m), run_settings(spec, m), start, seed)
