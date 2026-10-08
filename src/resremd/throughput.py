"""How fast the replicas of a prepared system run, for choosing hardware
settings.

A small solute leaves a GPU idle between short kernels, and every cycle
costs host work besides the dynamics: temperatures and velocities set,
energies read back, replicas that share a context swapped in and out, and
with REST2 two more energy evaluations per replica. This times cycles of
the run's engine for several numbers of contexts per device, reporting
ns/day per replica and for all replicas together, and the share of each
cycle that is not dynamics (timed from cycles of zero steps). The command
times each count in a fresh process. Aggregate throughput that still grows
at the largest count means the device has room for more concurrent
contexts; a large overhead share means longer exchange intervals, or fewer
evaluations per cycle, would pay more than hardware.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("resremd")


def measure(prepared: Any, *, contexts_per_device: list[int] | None = None,
            isolate: bool = False, **settings: Any) -> list[dict[str, Any]]:
    """Throughput of ``n_replicas`` replicas for each number of contexts per
    device, after one warm-up cycle each. Returns one row per count.

    ``prepared`` is a prepared directory, a Prepared system, or (system,
    topology, positions, box). With ``isolate`` (as the command line does)
    each count is timed in a fresh process: none inherits another's device
    state, a count that fails is recorded and the others still run, and a
    process that fails as it exits, after its row is in, is reported rather
    than taking the table with it. Like any use of multiprocessing, a script
    that sets ``isolate`` needs an ``if __name__ == "__main__":`` guard and
    must be read from a file. A SIGTERM that would end this process first
    stops the timing process; a SIGTERM handler of the caller's own is left
    to decide. Other settings are those of :func:`_measure`.
    """
    import inspect
    import numbers

    from .errors import InputError
    from .system import from_objects, load_prepared, select_atoms

    if isolate:
        _check_isolatable()

    def whole(name: str, value: Any, least: int) -> int:
        if not isinstance(value, numbers.Integral) or isinstance(value, bool):
            raise InputError(f"{name} takes whole numbers, not {value!r}.",
                             code="resremd.input.type")
        if value < least:
            raise InputError(f"{name} takes counts of at least {least}, not "
                             f"{value}.", code="resremd.input.range")
        return int(value)

    counts = [1, 2, 4, 8] if contexts_per_device is None \
        else contexts_per_device
    counts = list(counts) if hasattr(counts, "__iter__") else [counts]
    if not counts:
        raise InputError("contexts_per_device names no count to time.",
                         code="resremd.input.range")
    counts = [whole("contexts_per_device", c, 1) for c in counts]
    # The names of the settings, their counts and the solute are checked
    # here once rather than failing in every process.
    bound = inspect.signature(_measure).bind(None, counts, **settings)
    bound.apply_defaults()
    for name, least in (("n_replicas", 2), ("steps", 1), ("cycles", 1)):
        settings[name] = whole(name, bound.arguments[name], least)
    source = prepared
    if isinstance(prepared, (str, Path)):
        prepared = load_prepared(prepared)
    elif not hasattr(prepared, "system"):
        prepared = source = from_objects(*prepared)
    if settings.get("rest2"):
        select_atoms(prepared.topology,
                     settings.get("rest2_selection", "solute"), option="rest2")
    if not isolate:
        return _measure(prepared, counts, **settings)
    return _stoppable(source, counts, settings)


def _check_isolatable() -> None:
    """Refuse at once where the timing processes could not run."""
    import multiprocessing
    import sys

    from .errors import InputError

    process = multiprocessing.current_process()
    # The flag multiprocessing itself checks: a script without a main guard
    # is being run again, as the main module of a new process.
    if getattr(process, "_inheriting", False):
        raise InputError(
            "measure(isolate=True) was called again as a timing process "
            "started: the script calling it needs an "
            "`if __name__ == \"__main__\":` guard.",
            code="resremd.input.isolate")
    if process.daemon:
        raise InputError(
            "measure(isolate=True) cannot start timing processes from a "
            "daemonic process (a multiprocessing.Pool worker, say). Call it "
            "with isolate=False there, or from a process of your own.",
            code="resremd.input.isolate")
    main = sys.modules.get("__main__")
    if getattr(main, "__file__", None) == "<stdin>":
        raise InputError(
            "measure(isolate=True) cannot run from standard input: each "
            "timing process imports the calling script again, so it must "
            "be a file. Run the script as a file, or set isolate=False.",
            code="resremd.input.isolate")


class _Stopped(BaseException):
    """A SIGTERM that ended the timing."""


def _stoppable(source: Any, counts: list[int],
               settings: dict[str, Any]) -> list[dict[str, Any]]:
    """The timings in fresh processes, with a SIGTERM that would end this
    process (kill, timeout, a job manager) stopping the timing process
    first rather than leaving it to run on."""
    import multiprocessing
    import os
    import signal
    import threading

    spawn = multiprocessing.get_context("spawn")
    if threading.current_thread() is not threading.main_thread() or \
            signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL:
        # A caller's own handler decides; one that raises or exits still
        # stops the timing process on the way out.
        return _isolated(spawn, source, counts, settings, {})
    state = {"starting": False, "signal": None, "raised": False}

    def stop(number, _frame):
        # A second signal, or one while a process is being started, is
        # only noted: the cleanup it would cut short must finish.
        state["signal"] = number
        if not state["starting"] and not state["raised"]:
            state["raised"] = True
            raise _Stopped

    signal.signal(signal.SIGTERM, stop)
    try:
        return _isolated(spawn, source, counts, settings, state)
    except _Stopped:
        pass
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    logger.warning("Stopped by SIGTERM; the timing process was stopped "
                   "first.")
    # The signal then does what it would have done.
    os.kill(os.getpid(), signal.SIGTERM)
    raise SystemExit(128 + signal.SIGTERM)


def _isolated(spawn: Any, source: Any, counts: list[int],
              settings: dict[str, Any],
              state: dict[str, Any]) -> list[dict[str, Any]]:
    import queue as queues

    rows = []
    for count in counts:
        channel = spawn.Queue()
        # A directory is read again by the process; a System travels as XML.
        worker = spawn.Process(target=_worker,
                               args=(channel, source, count, settings),
                               daemon=True)
        result = None
        hung = False
        try:
            state["starting"] = True
            try:
                worker.start()
            except Exception as exc:  # a System that cannot be sent
                rows.append({"contexts_per_device": count,
                             "error": f"could not start: {exc}"})
                logger.warning("Timing %d contexts per device could not "
                               "start: %s", count, exc)
                continue
            finally:
                state["starting"] = False
            if state.get("signal") is not None and not state["raised"]:
                state["raised"] = True
                raise _Stopped
            while result is None:
                try:
                    result = channel.get(timeout=1.0)
                except queues.Empty:
                    if not worker.is_alive():
                        try:
                            result = channel.get(timeout=1.0)
                        except queues.Empty:
                            break
            # A process stuck in teardown after reporting is not waited on.
            worker.join(timeout=120)
            hung = worker.is_alive()
        finally:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=10)
                if worker.is_alive():
                    worker.kill()
                    worker.join()
        if result is None:
            result = ("error", f"the process ended (exit code "
                               f"{worker.exitcode}) before it reported; "
                               "possible causes are a crash in the platform "
                               "or, from a script, a missing "
                               "`if __name__ == \"__main__\":` guard")
        status, value = result
        if status == "error":
            logger.warning("Timing %d contexts per device failed: %s", count,
                           value)
            rows.append({"contexts_per_device": count, "error": value})
            continue
        value["exit_code"] = worker.exitcode
        value["hung_on_exit"] = hung
        if hung:
            logger.warning(
                "The process timing %d contexts per device measured, then "
                "hung as it exited and was stopped.", count)
        elif worker.exitcode:
            logger.warning(
                "The process timing %d contexts per device measured, then "
                "failed as it exited (exit code %s).", count, worker.exitcode)
        logger.info("%d contexts per device: %.1f ns/day per replica, %.1f "
                    "total, %.0f%% overhead", count,
                    value["ns_per_day_per_replica"],
                    value["ns_per_day_total"],
                    100 * value["overhead_fraction"])
        rows.append(value)
    return rows


def _worker(channel: Any, prepared: Any, count: int,
            settings: dict[str, Any]) -> None:
    import multiprocessing
    import os
    import signal
    import threading
    from multiprocessing.connection import wait

    # The calling script, run again here, may have set a SIGTERM handler;
    # terminate() must still stop this process.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    parent = multiprocessing.parent_process()

    def orphaned() -> None:
        wait([parent.sentinel])
        os._exit(1)

    if parent is not None:
        # Nor may it outlive a parent that was killed outright.
        threading.Thread(target=orphaned, daemon=True).start()
    logging.basicConfig(level=logging.WARNING)
    try:
        row = _measure(prepared, [count], **settings)[0]
        row["pid"] = os.getpid()
        channel.put(("ok", row))
    except BaseException as exc:
        channel.put(("error", f"{type(exc).__name__}: {exc}"))


def _measure(prepared: Any, counts: list[int], *, n_replicas: int = 8,
             steps: int = 500, cycles: int = 10, timestep_fs: float = 2.0,
             temperature_K: float = 300.0, platform: str = "auto",
             precision: str = "mixed", devices: list[int] | None = None,
             cpu_threads: int | None = None, rest2: bool = False,
             rest2_selection: str = "solute", ensemble: str | None = None,
             random_seed: int = 1) -> list[dict[str, Any]]:
    """The timings, in this process. ``ensemble`` is as for a run: left
    out, the prepared System decides (its barostat, if it has one)."""
    from .engine import Engine, Replica
    from .ladder import geometric
    from .system import from_objects, load_prepared, select_atoms
    from .thermo import simulated_system

    if isinstance(prepared, (str, Path)):
        prepared = load_prepared(prepared)
    elif not hasattr(prepared, "system"):
        prepared = from_objects(*prepared)
    system = prepared.system
    scales = None
    if rest2:
        from .rest2 import rest2_system, scale_of

        solute = select_atoms(prepared.topology, rest2_selection,
                              option="rest2")
        _, known = simulated_system(system, ensemble=ensemble,
                                    pressure_bar=None,
                                    temperature_K=temperature_K,
                                    frequency=25)
        system, _ = rest2_system(
            system, solute, constant_pressure=known.constant_pressure)
        temps = geometric(temperature_K, 3 * temperature_K, n_replicas)
        scales = [scale_of(temperature_K, t) for t in temps]
        temps = [temperature_K] * n_replicas
    else:
        temps = geometric(temperature_K, 1.5 * temperature_K, n_replicas)
    system, ensemble = simulated_system(system, ensemble=ensemble,
                                        pressure_bar=None,
                                        temperature_K=temperature_K,
                                        frequency=25)
    rows = []
    for count in counts:
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
    platform = next((r["platform"] for r in rows if "platform" in r), "?")
    lines = [f"{n_replicas} replicas on {platform}",
             "contexts/device  contexts  ns/day/replica  ns/day total  "
             "overhead"]
    done = [r for r in rows if "error" not in r]
    for r in rows:
        if "error" in r:
            lines.append(f"{r['contexts_per_device']:>15d}  failed: "
                         f"{r['error']}")
            continue
        lines.append(f"{r['contexts_per_device']:>15d}  {r['contexts']:>8d}  "
                     f"{r['ns_per_day_per_replica']:>14.1f}  "
                     f"{r['ns_per_day_total']:>12.1f}  "
                     f"{100 * r['overhead_fraction']:>7.0f}%"
                     + ("  (process hung on exit)" if r.get("hung_on_exit")
                        else "  (process failed on exit)"
                        if r.get("exit_code") else ""))
    if not done:
        return "\n".join(lines)
    best = max(done, key=lambda r: r["ns_per_day_total"])
    lines.append(f"fastest: contexts_per_device: "
                 f"{best['contexts_per_device']}")
    return "\n".join(lines)
