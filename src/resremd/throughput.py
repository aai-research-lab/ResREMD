"""How fast the replicas of a prepared system run, for choosing hardware
settings.

A small solute leaves a GPU idle between short kernels, and every cycle
costs host work besides the dynamics: temperatures and velocities set,
energies read back, replicas that share a context swapped in and out, and
with REST2 two more energy evaluations per replica. This times cycles of
the run's engine for several numbers of contexts per device, reporting
ns/day per replica and for all replicas together, and the share of each
cycle that is not dynamics (timed from cycles of zero steps). Aggregate
throughput that still grows at the largest count means the device has room
for more concurrent contexts; a large overhead share means longer exchange
intervals, or fewer evaluations per cycle, would pay more than hardware.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger("resremd")


def measure(prepared: Any, *, n_replicas: int = 8,
            contexts_per_device: list[int] | None = None,
            steps: int = 500, cycles: int = 10, timestep_fs: float = 2.0,
            temperature_K: float = 300.0, platform: str = "auto",
            precision: str = "mixed", devices: list[int] | None = None,
            cpu_threads: int | None = None, rest2: bool = False,
            rest2_selection: str = "solute",
            random_seed: int = 1) -> list[dict[str, Any]]:
    """Throughput of ``n_replicas`` replicas for each number of contexts per
    device, after one warm-up cycle each. Returns one row per count."""
    from .engine import Engine, Replica
    from .ladder import geometric
    from .system import from_objects, load_prepared, select_atoms
    from .thermo import simulated_system

    if not hasattr(prepared, "system"):
        prepared = load_prepared(prepared) if isinstance(prepared, str) \
            else from_objects(*prepared)
    system = prepared.system
    scales = None
    if rest2:
        from .rest2 import rest2_system, scale_of

        solute = select_atoms(prepared.topology, rest2_selection)
        system, _ = rest2_system(system, solute)
        temps = geometric(temperature_K, 3 * temperature_K, n_replicas)
        scales = [scale_of(temperature_K, t) for t in temps]
        temps = [temperature_K] * n_replicas
    else:
        temps = geometric(temperature_K, 1.5 * temperature_K, n_replicas)
    system, ensemble = simulated_system(system, ensemble=None,
                                        pressure_bar=None,
                                        temperature_K=temperature_K,
                                        frequency=25)
    rows = []
    for count in contexts_per_device or [1, 2, 4, 8]:
        engine = Engine(system=system, ensemble=ensemble,
                        n_replicas=n_replicas, integrator="langevin_middle",
                        timestep_fs=timestep_fs, friction_per_ps=1.0,
                        temperature_K=temperature_K, platform=platform,
                        precision=precision, devices=devices,
                        contexts_per_device=count, cpu_threads=cpu_threads,
                        seeds=np.random.default_rng(random_seed),
                        rest2=rest2)
        try:
            replicas = [Replica(r, reset=(prepared.positions.copy(),
                                          None if prepared.box is None
                                          else prepared.box.copy(), r + 1))
                        for r in range(n_replicas)]

            def timed(n_steps: int, n_cycles: int) -> float:
                t0 = time.perf_counter()
                for _ in range(n_cycles):
                    # A rotation, as exchanges move replicas between
                    # temperatures and so between contexts' settings.
                    order = temps[1:] + temps[:1]
                    temps[:] = order
                    if scales is not None:
                        scales[:] = scales[1:] + scales[:1]
                    engine.run(replicas, temps, n_steps, set(), scales)
                return (time.perf_counter() - t0) / n_cycles

            timed(steps, 1)                  # warm-up: kernels, first loads
            per_cycle = timed(steps, cycles)
            overhead = timed(0, cycles)
            ns = steps * timestep_fs * 1e-6
            description = engine.describe()
        finally:
            engine.close()
        per_replica = ns / per_cycle * 86400.0
        rows.append({
            "contexts_per_device": count,
            "contexts": description["contexts"],
            "platform": description["platform"],
            "replicas_resident": description["replicas_resident"],
            "seconds_per_cycle": per_cycle,
            "ns_per_day_per_replica": per_replica,
            "ns_per_day_total": per_replica * n_replicas,
            "overhead_fraction": min(1.0, overhead / per_cycle),
        })
        logger.info("%d contexts per device: %.1f ns/day per replica, %.1f "
                    "total, %.0f%% overhead", count, per_replica,
                    per_replica * n_replicas, 100 * rows[-1][
                        "overhead_fraction"])
    return rows


def format_rows(rows: list[dict[str, Any]], n_replicas: int) -> str:
    lines = [f"{n_replicas} replicas on {rows[0]['platform']}",
             "contexts/device  contexts  ns/day/replica  ns/day total  "
             "overhead"]
    for r in rows:
        lines.append(f"{r['contexts_per_device']:>15d}  {r['contexts']:>8d}  "
                     f"{r['ns_per_day_per_replica']:>14.1f}  "
                     f"{r['ns_per_day_total']:>12.1f}  "
                     f"{100 * r['overhead_fraction']:>7.0f}%")
    best = max(rows, key=lambda r: r["ns_per_day_total"])
    lines.append(f"fastest: contexts_per_device: "
                 f"{best['contexts_per_device']}")
    return "\n".join(lines)
