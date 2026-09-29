"""What a run did: exchange statistics, round trips, and reservoir lineage.

These say whether the replicas mixed, not whether the ensemble converged.
Convergence is judged on the observables of interest, per temperature and
over time, with the trajectories this run wrote.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _table(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return data


def round_trips(states: np.ndarray, top: int) -> tuple[int, list[float]]:
    """Completed lowest -> highest -> lowest trips, and their lengths in cycles.

    ``states`` is (cycles, replicas) of state indices.
    """
    trips = 0
    lengths: list[float] = []
    for r in range(states.shape[1]):
        path = states[:, r]
        start = None
        reached_top = False
        for c, s in enumerate(path):
            if s == 0:
                if start is not None and reached_top:
                    trips += 1
                    lengths.append(float(c - start))
                start = c
                reached_top = False
            elif s == top and start is not None:
                reached_top = True
    return trips, lengths


def effective_ancestors(origins) -> float:
    """How many independent reservoir structures a state's samples rest on.

    ``origins`` is, for each sample, the reservoir frame its coordinates
    descend from (-1 for the starting structure, which counts as one more
    ancestor). The inverse Simpson index of the ancestor frequencies,
    1 / sum p_k^2, is the number of equally represented ancestors that would
    give the same concentration. Samples that share an ancestor share its
    history, so this bounds the number of independent configurations the
    ensemble was annealed from.
    """
    origins = np.asarray(origins).ravel()
    if origins.size == 0:
        return 0.0
    _, counts = np.unique(origins, return_counts=True)
    p = counts / counts.sum()
    return float(1.0 / np.sum(p * p))


def _slope_fit(top_h, res_h, w, edges, min_count, g_t=1.0, g_r=1.0):
    """Weighted straight-line fit of ln(P_top / P_R) against h.

    Returns (slope, analytic standard error, bins used), or None when fewer
    than three bins hold enough samples on both sides.
    """
    n_t = np.histogram(top_h, edges)[0].astype(float)
    n_r = np.histogram(res_h, edges, weights=w)[0]
    # Effective counts for a weighted histogram (Kish), so the error of a
    # bin reflects how many frames really support it.
    n_r_eff = n_r ** 2 / np.maximum(
        np.histogram(res_h, edges, weights=w * w)[0], 1e-300)
    keep = (n_t >= min_count) & (n_r_eff >= min_count)
    if keep.sum() < 3:
        return None
    centres = 0.5 * (edges[1:] + edges[:-1])[keep]
    y = np.log(n_t[keep] / n_t.sum()) - np.log(n_r[keep] / n_r.sum())
    var = g_t * (1 / n_t[keep] - 1 / n_t.sum()) + \
        g_r * (1 / n_r_eff[keep] - 1 / n_r_eff.sum())
    wt = 1.0 / var
    x0 = np.sum(wt * centres) / np.sum(wt)
    sxx = np.sum(wt * (centres - x0) ** 2)
    slope = np.sum(wt * (centres - x0) * y) / sxx
    return float(slope), float(np.sqrt(1.0 / sxx)), int(keep.sum())


def _blocks(n: int, length: int, rng) -> np.ndarray:
    """Indices of a moving-block bootstrap resample of a series of n."""
    length = max(1, min(length, n))
    starts = rng.integers(0, n - length + 1, size=int(np.ceil(n / length)))
    return (starts[:, None] + np.arange(length)[None]).ravel()[:n]


def ensemble_check(top_h, reservoir_h, beta_top: float, beta_reservoir: float,
                   *, reservoir_weights=None, bins: int = 20,
                   min_count: int = 10, bootstrap: int = 200,
                   seed: int = 0) -> dict[str, Any]:
    """Is the reservoir a sample at the temperature it claims?

    For two samples of one system at inverse temperatures beta_t and
    beta_R, the ratio of their enthalpy distributions is exactly

        ln[P_t(h) / P_R(h)] = const - (beta_t - beta_R) h

    whatever the density of states (Shirts, J. Chem. Theory Comput. 2013,
    9, 909). A straight-line fit over the energies both visit gives a slope;
    its departure from -(beta_t - beta_R), in standard errors, is ``z``, and
    the temperature the slope implies for the reservoir is
    ``reservoir_temperature_implied_K``. A non-Boltzmann reservoir is
    checked as beta_R = 0.

    What it catches: frames drawn at another temperature than their label.
    What it cannot catch: a reservoir missing part of its ensemble, or one
    whose simulation never converged. The relation above holds just as well
    between two ensembles restricted to the same region, and the top
    replica, fed by the reservoir, takes on its restriction.

    The standard error is the larger of the fit's own (with each bin's
    variance inflated by the series' statistical inefficiency) and a
    moving-block bootstrap over both series, which also covers bins that
    rise and fall together with a slow mode.

    ``status`` is ``ok``, ``too_few_frames`` or ``no_overlap``; the
    numbers are present only when it is ``ok``.
    """
    from .statistics import statistical_inefficiency
    from .thermo import BOLTZ

    top_h = np.asarray(top_h, dtype=float)
    res_h = np.asarray(reservoir_h, dtype=float)
    w = np.ones_like(res_h) if reservoir_weights is None \
        else np.asarray(reservoir_weights, dtype=float)
    if top_h.size < 3 * min_count or res_h.size < 3 * min_count:
        return {"status": "too_few_frames"}
    lo = max(np.quantile(top_h, 0.005), np.quantile(res_h, 0.005))
    hi = min(np.quantile(top_h, 0.995), np.quantile(res_h, 0.995))
    if not hi > lo:
        return {"status": "no_overlap"}
    edges = np.linspace(lo, hi, bins + 1)
    g_t = statistical_inefficiency(top_h)
    g_r = statistical_inefficiency(res_h)
    fit = _slope_fit(top_h, res_h, w, edges, min_count, g_t, g_r)
    if fit is None:
        return {"status": "no_overlap" if res_h.size >= 10 * min_count
                else "too_few_frames"}
    slope, se_fit, used = fit
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(bootstrap):
        it = _blocks(top_h.size, int(np.ceil(2 * g_t)), rng)
        ir = _blocks(res_h.size, int(np.ceil(2 * g_r)), rng)
        b = _slope_fit(top_h[it], res_h[ir], w[ir], edges, min_count)
        if b is not None:
            boots.append(b[0])
    se_boot = float(np.std(boots, ddof=1)) if len(boots) > 10 else 0.0
    se = max(se_fit, se_boot)
    expected = -(beta_top - beta_reservoir)
    implied_beta = beta_top + slope
    return {
        "status": "ok",
        "slope": slope, "slope_stderr": se,
        "slope_stderr_fit": se_fit, "slope_stderr_bootstrap": se_boot,
        "expected_slope": float(expected),
        "z": float((slope - expected) / se),
        "reservoir_temperature_implied_K":
            float(1.0 / (BOLTZ * implied_beta)) if implied_beta > 0 else None,
        "bins_used": used,
        "statistical_inefficiency": {"top": g_t, "reservoir": g_r},
    }


def coverage_check(top_labels, reservoir_labels, reservoir_h,
                   beta_top: float, beta_reservoir: float, n_states: int,
                   *, reservoir_weights=None, min_visits: int = 5
                   ) -> dict[str, Any]:
    """Does the reservoir hold every state the top replica reaches?

    The top replica's state populations are compared with the reservoir's,
    reweighted from its temperature to the top one by
    exp(-(beta_top - beta_R) h). A correct reservoir gives the same numbers
    within error. A state the top replica visits and the reservoir never
    holds is reported as ``unsupported`` whatever the error bars say: that
    is the flaw that biases every temperature and that the temperature
    check cannot see.

    Labels are state indices from any classification the caller trusts.
    The check can only see states the top replica reaches on its own; a
    state reachable neither by the ladder nor by the reservoir stays
    invisible to it, as to every method.
    """
    from .statistics import statistical_inefficiency

    top = np.asarray(top_labels, dtype=int)
    lab = np.asarray(reservoir_labels, dtype=int)
    h = np.asarray(reservoir_h, dtype=float)
    w0 = np.ones_like(h) if reservoir_weights is None \
        else np.asarray(reservoir_weights, dtype=float)
    if top.size == 0 or lab.size == 0:
        return {"status": "too_few_frames"}
    x = np.log(np.maximum(w0, 1e-300)) - (beta_top - beta_reservoir) * h
    w = np.exp(x - x.max())
    w /= w.sum()
    n_eff = float(1.0 / np.sum(w * w))
    p_top = np.bincount(top, minlength=n_states)[:n_states] / top.size
    p_res = np.bincount(lab, weights=w, minlength=n_states)[:n_states]
    visits = np.bincount(top, minlength=n_states)[:n_states]
    # A frame of zero weight (underflowed, or weighted out) holds nothing.
    held = np.bincount(lab[(w0 > 0) & (w > 0)],
                       minlength=n_states)[:n_states]
    z = np.zeros(n_states)
    for k in range(n_states):
        hit = (top == k).astype(float)
        g_t = statistical_inefficiency(hit)
        # The reservoir's is a ratio estimate with weights that depend on
        # the state; its variance is sum w_i^2 (1_i - p)^2 (delta method),
        # times the inefficiency of the terms themselves.
        terms = w * ((lab == k) - p_res[k])
        g_r = statistical_inefficiency(terms * lab.size)
        var = (p_top[k] * (1 - p_top[k]) * g_t / top.size
               + g_r * float(np.sum(terms ** 2)))
        diff = p_top[k] - p_res[k]
        if abs(diff) < 1e-9:
            # Equal, up to rounding in the weights; also covers a state
            # both sides hold always or never.
            z[k] = 0.0
        elif var > 0:
            z[k] = diff / np.sqrt(var)
        else:
            # Both sides certain and different: as far apart as can be.
            z[k] = np.copysign(np.inf, diff)
    unsupported = [int(k) for k in range(n_states)
                   if held[k] == 0 and visits[k] >= min_visits]
    return {
        "status": "ok",
        "top_populations": p_top.tolist(),
        "reservoir_populations_at_top": p_res.tolist(),
        "z": z.tolist(),
        "max_abs_z": float(np.max(np.abs(z))),
        "unsupported_states": unsupported,
        "reservoir_effective_frames": n_eff,
    }


def _reservoir_enthalpy(run_dir: Path, manifest: dict[str, Any]
                        ) -> np.ndarray | None:
    saved = run_dir / "reservoir_enthalpy_kjmol.npy"
    if saved.exists():
        return np.load(saved)
    # Runs from before the enthalpies were saved: find the reservoir's cache
    # for this System and platform.
    res = manifest.get("reservoir") or {}
    cache = Path(res.get("path", "")) / "energies"
    if not cache.is_dir():
        return None
    from .thermo import Ensemble

    ensemble = Ensemble.from_dict(manifest["system"]["ensemble"])
    for f in sorted(cache.glob("*.npz")):
        with np.load(f) as data:
            fields = json.loads(str(data["fields"]))
            if fields.get("system_sha256") == \
                    manifest["system"]["system_sha256"] and \
                    fields.get("platform") == manifest["engine"]["platform"]:
                return ensemble.enthalpy(data["potential_kjmol"],
                                         data["volume_nm3"],
                                         data["area_nm2"])
    return None


def reservoir_check(run_dir: str | Path) -> dict[str, Any] | None:
    """:func:`ensemble_check` between a run's top replica and its reservoir."""
    from .reservoir import Reservoir
    from .thermo import Ensemble, beta

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("reservoir") is None:
        return None
    res_h = _reservoir_enthalpy(run_dir, manifest)
    if res_h is None:
        return {"status": "no_energies"}
    states = _table(run_dir / "states.csv")[:, 2:].astype(int)
    energies = _table(run_dir / "energies.csv")[:, 2:]
    top = states.shape[1] - 1
    ensemble = Ensemble.from_dict(manifest["system"]["ensemble"])
    h = energies
    if ensemble.constant_pressure:
        vol = _table(run_dir / "volumes.csv")[:, 2:]
        area = _table(run_dir / "areas.csv")[:, 2:] \
            if (run_dir / "areas.csv").exists() else np.zeros_like(vol)
        h = ensemble.enthalpy(energies, vol, area)
    top_h = h[states == top]
    reservoir = Reservoir.open(manifest["reservoir"]["path"])
    t_top = manifest["states"][top]["temperature_K"]
    return ensemble_check(top_h, res_h, beta(t_top), reservoir.beta,
                          reservoir_weights=reservoir.weights)


def reservoir_coverage(run_dir: str | Path, top_labels, reservoir_labels,
                       n_states: int, *, min_visits: int = 5
                       ) -> dict[str, Any] | None:
    """:func:`coverage_check` between a run's top replica and its reservoir.

    ``top_labels`` are the states of the frames saved at the top
    temperature, in order; ``reservoir_labels`` those of the reservoir's
    frames, in the reservoir's order. The classification is the caller's.
    """
    from .reservoir import Reservoir
    from .thermo import beta

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("reservoir") is None:
        return None
    reservoir = Reservoir.open(manifest["reservoir"]["path"])
    if len(reservoir_labels) != reservoir.n_frames:
        raise ValueError(f"{len(reservoir_labels)} reservoir labels for "
                         f"{reservoir.n_frames} frames.")
    res_h = _reservoir_enthalpy(run_dir, manifest)
    if res_h is None:
        return {"status": "no_energies"}
    t_top = manifest["states"][-1]["temperature_K"]
    return coverage_check(top_labels, reservoir_labels, res_h, beta(t_top),
                          reservoir.beta, n_states,
                          reservoir_weights=reservoir.weights,
                          min_visits=min_visits)


def summarize(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    states_table = _table(run_dir / "states.csv")
    cycles = states_table.shape[0]
    states = states_table[:, 2:].astype(int)
    n = states.shape[1]
    trips, lengths = round_trips(states, n - 1)
    out: dict[str, Any] = {
        "status": manifest["status"],
        "method": manifest["method"],
        "cycles": cycles,
        "time_ns_per_replica": manifest["progress"]["time_ns_per_replica"],
        "temperatures_K": [s["temperature_K"] for s in manifest["states"]],
        "neighbour_acceptance": [p["acceptance"] for p in
                                 manifest["exchanges"]["neighbour_pairs"]],
        "round_trips": trips,
        "mean_round_trip_cycles": float(np.mean(lengths)) if lengths else None,
        "cost": manifest.get("cost"),
    }
    visited = [len(set(states[:, r].tolist())) for r in range(n)]
    out["replicas_that_visited_every_state"] = int(sum(v == n for v in visited))
    res = manifest["exchanges"].get("reservoir")
    if res is not None:
        out["reservoir_acceptance"] = res["acceptance"]
        out["reservoir_distinct_frames_accepted"] = \
            res["distinct_frames_accepted"]
        origins = _table(run_dir / "origins.csv")[:, 2:].astype(int)
        lowest = origins[np.arange(cycles), np.argmin(states, axis=1)]
        from_reservoir = lowest >= 0
        out["lowest_state_fraction_from_reservoir"] = \
            float(from_reservoir.mean())
        out["reservoir_frames_that_reached_lowest_state"] = \
            int(np.unique(lowest[from_reservoir]).size)
        out["effective_reservoir_ancestors"] = effective_ancestors(lowest)
        try:
            out["reservoir_check"] = reservoir_check(run_dir)
        except Exception as exc:  # a summary must not fail over a diagnostic
            out["reservoir_check"] = {"error": str(exc)}
        exchanges = _table(run_dir / "reservoir_exchanges.csv")
        if exchanges.size:
            accepted = exchanges[:, -1]
            quarters = np.array_split(accepted, 4)
            out["reservoir_acceptance_by_quarter"] = [
                float(q.mean()) if q.size else None for q in quarters]
    return out


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"{summary['method']} ({summary['status']})",
        f"  {summary['cycles']} cycles, "
        f"{summary['time_ns_per_replica']:.3f} ns per replica",
        "  temperatures (K): " + ", ".join(f"{t:.1f}" for t in
                                          summary["temperatures_K"]),
        "  neighbour swap acceptance: " + ", ".join(
            f"{a:.2f}" for a in summary["neighbour_acceptance"]),
        f"  round trips lowest-highest-lowest: {summary['round_trips']}"
        + (f" (mean {summary['mean_round_trip_cycles']:.0f} cycles)"
           if summary["mean_round_trip_cycles"] else ""),
        f"  replicas that visited every temperature: "
        f"{summary['replicas_that_visited_every_state']}",
    ]
    if "reservoir_acceptance" in summary:
        lines += [
            f"  reservoir acceptance: {summary['reservoir_acceptance']:.3f}"
            + (" (by quarter: " + ", ".join(
                f"{q:.3f}" for q in summary["reservoir_acceptance_by_quarter"]
                if q is not None) + ")"
               if summary.get("reservoir_acceptance_by_quarter") else ""),
            f"  distinct reservoir frames accepted: "
            f"{summary['reservoir_distinct_frames_accepted']}",
            f"  lowest-temperature samples descended from the reservoir: "
            f"{100 * summary['lowest_state_fraction_from_reservoir']:.1f}%",
            f"  distinct reservoir frames that reached the lowest temperature: "
            f"{summary['reservoir_frames_that_reached_lowest_state']}",
            f"  effective number of reservoir ancestors at the lowest "
            f"temperature: {summary['effective_reservoir_ancestors']:.1f}",
        ]
        check = summary.get("reservoir_check")
        if check and check.get("status") == "ok":
            verdict = "consistent" if abs(check["z"]) < 3 else "INCONSISTENT"
            implied = check["reservoir_temperature_implied_K"]
            like = (f"{implied:.0f} K" if implied is not None
                    else "no finite temperature")
            lines.append(
                f"  reservoir temperature check: {verdict} (z = "
                f"{check['z']:+.1f}; its energies look like {like})")
        elif check is not None:
            why = {"too_few_frames": "too few frames to test",
                   "no_overlap": "the top replica and the reservoir share "
                                 "too few energies",
                   "no_energies": "the reservoir energies were not found"}.get(
                check.get("status"), check.get("error", "not available"))
            lines.append(f"  reservoir temperature check: {why}")
    return "\n".join(lines)
