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
    than taking the table with it. As each process is spawned, and imports
    the calling script again, a script that sets ``isolate`` needs an
    ``if __name__ == "__main__":`` guard and cannot be read from standard
    input. Called from the main thread, a SIGTERM that would end this
    process first stops the timing process (from another thread, the timing
    process ends soon after this one); a SIGTERM handler of the caller's own
    is left to decide. The other settings, with their defaults, are
    ``n_replicas=8``, ``steps=500`` (per cycle), ``cycles=10``,
    ``timestep_fs=2.0``, ``temperature_K=300.0``, ``platform="auto"``,
    ``precision="mixed"``, ``devices``, ``cpu_threads``, ``rest2=False``,
    ``rest2_selection="solute"``, ``ensemble`` (as for a run) and
    ``random_seed=1``.
    """
    import inspect
    import math
    import numbers

    from .errors import InputError
    from .options import (CPU_THREADS, DEVICES, ENSEMBLE, PLATFORM,
                          PRECISION, RANDOM_SEED, REST2, REST2_SELECTION,
                          _check_one)
    from .system import from_objects, load_prepared, select_atoms

    if isolate:
        _check_isolatable()

    def whole(name: str, value: Any, least: int,
              most: int = 2**31 - 1) -> int:
        if not isinstance(value, numbers.Integral) or isinstance(value, bool):
            raise InputError(f"{name} takes whole numbers, not {value!r}.",
                             code="resremd.input.type")
        if value < least:
            raise InputError(f"{name} takes counts of at least {least}, not "
                             f"{value}.", code="resremd.input.range")
        if value > most:  # what OpenMM takes as a number of steps
            raise InputError(f"{name} takes counts of at most {most}, not "
                             f"{value}.", code="resremd.input.range")
        return int(value)

    counts = [1, 2, 4, 8] if contexts_per_device is None \
        else contexts_per_device
    try:
        counts = list(counts)
    except TypeError:  # one count
        counts = [counts]
    if not counts:
        raise InputError("contexts_per_device names no count to time.",
                         code="resremd.input.range")
    counts = [whole("contexts_per_device", c, 1) for c in counts]
    # The settings and the solute are checked here once rather than
    # failing in every process (whether the platform is there is found as
    # each process starts on it), the words as for a run.
    names = set(inspect.signature(_measure).parameters) - {"prepared",
                                                            "counts"}
    unknown = sorted(set(settings) - names)
    if unknown:
        raise TypeError(f"measure() got unexpected settings: "
                        f"{', '.join(unknown)}")
    bound = inspect.signature(_measure).bind(None, counts, **settings)
    bound.apply_defaults()
    for name, least in (("n_replicas", 2), ("steps", 1), ("cycles", 1)):
        settings[name] = whole(name, bound.arguments[name], least)
    for name in ("timestep_fs", "temperature_K"):
        value = bound.arguments[name]
        if not isinstance(value, numbers.Real) or isinstance(value, bool) \
                or not math.isfinite(value) or value <= 0:
            raise InputError(f"{name} takes a number above 0, not "
                             f"{value!r}.", code="resremd.input.range")
        settings[name] = float(value)
    for option in (ENSEMBLE, PLATFORM, PRECISION, CPU_THREADS, DEVICES,
                   REST2, REST2_SELECTION, RANDOM_SEED):
        value = bound.arguments[option.name]
        if isinstance(value, numbers.Integral) and \
                not isinstance(value, bool):
            value = int(value)  # a numpy integer, say
        settings[option.name] = _check_one(option, value)
    source = prepared
    if isinstance(prepared, (str, Path)):
        prepared = load_prepared(prepared)
    elif not hasattr(prepared, "system"):
        try:
            prepared = source = from_objects(*prepared)
        except TypeError:
            raise InputError(
                "measure() times a prepared directory, a Prepared system or "
                f"(system, topology, positions, box), not {prepared!r:.80}.",
                code="resremd.input.type") from None
    if bound.arguments["rest2"]:
        select_atoms(prepared.topology, bound.arguments["rest2_selection"],
                     option="rest2")
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
        # Said once by this process, without a traceback, which ends it.
        raise SystemExit(
            "measure(isolate=True) ran again inside a timing process as it "
            "started: the script calling it needs an "
            "`if __name__ == \"__main__\":` guard.")
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


# Seconds a process that has reported may take to exit before it is
# stopped and reported as hung.
HUNG_AFTER_S = 120.0


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
        # Handlers can be set only from the main thread, and a caller's own
        # handler decides: one that raises or exits still stops the timing
        # process on the way out (within 10 s, should the handler reach a
        # process still importing the calling script).
        return _isolated(spawn, source, counts, settings, {})
    state = {"signal": None}

    def stop(number, _frame):
        # Only noted, and acted on where the timing loop looks (four times
        # a second): raised here it could land anywhere, in a process being
        # started or a finalizer that would swallow it.
        state["signal"] = number

    try:
        signal.signal(signal.SIGTERM, stop)
        return _isolated(spawn, source, counts, settings, state)
    except _Stopped:
        pass
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        if state["signal"] is not None:
            logger.warning("Stopped by SIGTERM.")
            # The signal then does what it would have done, whatever else
            # (a Ctrl-C, say) was under way.
            os.kill(os.getpid(), signal.SIGTERM)
    # Reached only if that did not end the process (SIGTERM blocked).
    raise SystemExit(128 + signal.SIGTERM)


def _isolated(spawn: Any, source: Any, counts: list[int],
              settings: dict[str, Any],
              state: dict[str, Any]) -> list[dict[str, Any]]:
    import contextlib
    import pickle
    import tempfile

    def check() -> None:
        if state.get("signal") is not None:
            raise _Stopped

    if isinstance(source, (str, Path)):  # each process reads it again
        return [_one(spawn, str(source), count, settings, check)
                for count in counts]
    def not_written(where: str | None, exc: Exception) -> list[dict]:
        problem = OSError(
            "the System could not be written to a temporary file"
            + (f" in {where}" if where else "")
            + f" (TMPDIR sets the directory; or no file descriptor was "
            f"free): {exc}")
        return [_not_started(count, problem) for count in counts]

    # An in-memory System goes to each process, pickled as XML, in a
    # temporary file unlinked as it is made (usually never named at all),
    # so that nothing is left behind however this process ends. Sent
    # through a pipe instead, a large System would wait on the process
    # importing the calling script.
    where = None
    try:
        where = tempfile.gettempdir()
        handoff = tempfile.TemporaryFile(dir=where)
    except OSError as exc:
        return not_written(where, exc)
    try:
        pickle.dump(source, handoff)
        handoff.flush()
    except Exception as exc:
        # Closing tries a failed write again.
        with contextlib.suppress(OSError):
            handoff.close()
        if isinstance(exc, OSError):
            return not_written(where, exc)
        return [_not_started(count, exc)  # a System that cannot be sent
                for count in counts]
    with handoff:
        return [_one(spawn, _Handoff(handoff.fileno()), count, settings,
                     check) for count in counts]


class _Handoff:
    """The temporary file, as its descriptor passed to a process."""

    def __init__(self, descriptor: int) -> None:
        self.descriptor = descriptor

    def __reduce__(self) -> tuple:
        from multiprocessing.reduction import DupFd

        return (_handed_over, (DupFd(self.descriptor),))


def _handed_over(duplicate: Any) -> Any:
    import os

    return os.fdopen(duplicate.detach(), "rb")


def _not_started(count: int, exc: Exception) -> dict[str, Any]:
    logger.warning("Timing %d contexts per device could not start: %s",
                   count, exc)
    return {"contexts_per_device": count, "error": f"could not start: {exc}"}


def _one(spawn: Any, handoff: Any, count: int, settings: dict[str, Any],
         check: Any) -> dict[str, Any]:
    """One count's row, from a process of its own."""
    import signal

    poll = 0.25  # seconds between looks at a noted signal
    try:
        results, reporter = spawn.Pipe(duplex=False)
    except OSError as exc:  # out of descriptors, say
        return _not_started(count, exc)
    worker = spawn.Process(target=_worker,
                           args=(reporter, handoff, count, settings),
                           daemon=True)
    result = None
    hung = False
    try:
        try:
            # Here, not earlier: setting up this count let the last one's
            # finalizers run, and a signal may have come during one.
            check()
            worker.start()
        except Exception as exc:
            return _not_started(count, exc)
        finally:
            # The process has its own end now; with this one closed, its
            # end shows here as the end of the pipe.
            reporter.close()
        while result is None:
            check()
            if results.poll(poll):
                try:
                    result = results.recv()
                except (EOFError, OSError):  # ended without reporting
                    break
        # A process stuck in teardown after reporting is waited on for
        # HUNG_AFTER_S at most, then stopped.
        for _ in range(int(HUNG_AFTER_S / poll)):
            check()
            worker.join(timeout=poll)
            if not worker.is_alive():
                break
        hung = worker.is_alive()
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=10)
            if worker.is_alive():
                worker.kill()
                worker.join()
        results.close()
    if result is None and (worker.exitcode or 0) < 0:
        try:
            name = signal.Signals(-worker.exitcode).name
        except ValueError:
            name = f"signal {-worker.exitcode}"
        result = ("error", f"the process was ended by {name} before it "
                           "reported; possible causes are a crash in the "
                           "platform or a signal from outside")
    elif result is None:
        result = ("error", f"the process ended (exit code "
                           f"{worker.exitcode}) before it reported; "
                           "possible causes are a crash in the platform "
                           "or, from a script, a missing "
                           "`if __name__ == \"__main__\":` guard")
    status, value = result
    if status == "error":
        logger.warning("Timing %d contexts per device failed: %s", count,
                       value)
        return {"contexts_per_device": count, "error": value}
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
    return value


def _worker(channel: Any, handoff: Any, count: int,
            settings: dict[str, Any]) -> None:
    import multiprocessing
    import os
    import pickle
    import signal
    import threading
    from multiprocessing.connection import wait

    # The calling script, run again here, may have set a SIGTERM handler;
    # from here on terminate() stops this process whatever it was.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    parent = multiprocessing.parent_process()

    def orphaned() -> None:
        wait([parent.sentinel])
        os._exit(1)

    if parent is not None:
        # Nor may it outlive a parent that was killed outright (from here
        # on: while it still imports the calling script, it cannot tell).
        threading.Thread(target=orphaned, daemon=True).start()
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        if isinstance(handoff, str):  # a prepared directory
            prepared = handoff
        else:
            with handoff:
                handoff.seek(0)  # shared with the processes before this one
                prepared = pickle.load(handoff)
        row = _measure(prepared, [count], **settings)[0]
        row["pid"] = os.getpid()
        channel.send(("ok", row))
    except BaseException as exc:
        try:
            channel.send(("error", f"{type(exc).__name__}: {exc}"))
        except OSError:  # nobody left to tell: the parent has gone
            os._exit(1)


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
    from .system import load_prepared, select_atoms
    from .thermo import simulated_system

    if isinstance(prepared, (str, Path)):  # a process reading a directory
        prepared = load_prepared(prepared)
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


def format_rows(rows: list[dict[str, Any]], n_replicas: int,
                platform: str = "?") -> str:
    """The rows as a table; ``platform`` names the one asked for, for a
    table in which no count ran."""
    platform = next((r["platform"] for r in rows if "platform" in r),
                    platform)
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
    # Counts that came to the same number of contexts (more than the
    # replicas need) ran alike: the smallest of them stands for them.
    alike: dict[int, dict[str, Any]] = {}
    for r in sorted(done, key=lambda r: r["contexts_per_device"]):
        alike.setdefault(r["contexts"], r)
    best = max(alike.values(), key=lambda r: r["ns_per_day_total"])
    lines.append(f"fastest: contexts_per_device: "
                 f"{best['contexts_per_device']}")
    return "\n".join(lines)
