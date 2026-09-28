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
        ]
    return "\n".join(lines)
