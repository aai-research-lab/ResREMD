import json

import pytest

from resremd import testsystems
from resremd.cli import main
from resremd.system import write_prepared
from resremd.throughput import measure


def test_throughput_reports_each_context_count():
    rows = measure(testsystems.lj_box(), n_replicas=4,
                   contexts_per_device=[1, 2], steps=20, cycles=2,
                   platform="Reference", isolate=False)
    assert [r["contexts"] for r in rows] == [1, 2]
    for r in rows:
        assert r["ns_per_day_total"] == \
            r["ns_per_day_per_replica"] * 4
        assert 0.0 < r["overhead_fraction"] <= 1.0
    assert not rows[0]["replicas_resident"]


def test_throughput_command_with_rest2(tmp_path, capsys):
    p = testsystems.lj_box()
    write_prepared(tmp_path / "setup", p.system, p.topology, p.positions,
                   p.box)
    assert main(["-q", "throughput", "--prepared", str(tmp_path / "setup"),
                 "--n-replicas", "2", "--contexts-per-device", "1",
                 "--steps", "10", "--cycles", "1", "--platform", "Reference",
                 "--rest2", "--rest2-selection", "all", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["contexts_per_device"] == 1


def test_the_cpu_runs_one_single_threaded_context_per_core():
    import numpy as np

    from resremd.engine import Engine, available_cores
    from resremd.thermo import ensemble_of

    p = testsystems.lj_box()
    ensemble, _ = ensemble_of(p.system)
    engine = Engine(system=p.system, ensemble=ensemble, n_replicas=16,
                    integrator="langevin_middle", timestep_fs=2.0,
                    friction_per_ps=1.0, temperature_K=100.0, platform="CPU",
                    precision="mixed", devices=None, contexts_per_device=None,
                    cpu_threads=None, seeds=np.random.default_rng(0))
    try:
        cores = available_cores()
        assert engine.describe()["contexts"] == min(16, cores)
        if cores > 1:
            threads = engine.slots[0].context.getPlatform().getPropertyValue(
                engine.slots[0].context, "Threads")
            assert int(threads) == 1
    finally:
        engine.close()


def test_each_count_is_timed_in_its_own_process():
    import os
    import signal

    import numpy as np

    before = signal.getsignal(signal.SIGTERM)
    p = testsystems.lj_box()
    rows = measure((p.system, p.topology, p.positions, p.box), n_replicas=2,
                   contexts_per_device=np.array([1, 2]), steps=10, cycles=1,
                   platform="Reference", isolate=True)
    assert signal.getsignal(signal.SIGTERM) == before
    assert [r["contexts"] for r in rows] == [1, 2]
    json.dumps(rows)
    assert all(r["exit_code"] == 0 for r in rows)
    pids = {r["pid"] for r in rows}
    assert len(pids) == 2 and os.getpid() not in pids


def test_a_failing_count_is_recorded_and_the_others_reported(tmp_path,
                                                             capsys):
    """A count that fails in its process becomes a row saying why; a table
    in which every count failed makes the command fail."""
    from resremd.cli import main
    from resremd.system import write_prepared
    from resremd.throughput import format_rows

    rows = measure(testsystems.lj_box(), n_replicas=2,
                   contexts_per_device=[1, 2], steps=10, cycles=1,
                   platform="NoSuchPlatform", isolate=True)
    assert len(rows) == 2 and all("NoSuchPlatform" in r["error"]
                                  for r in rows)
    assert "failed" in format_rows(rows, 2)
    p = testsystems.lj_box()
    write_prepared(tmp_path / "setup", p.system, p.topology, p.positions,
                   p.box)
    assert main(["-q", "throughput", "--prepared", str(tmp_path / "setup"),
                 "--n-replicas", "2", "--contexts-per-device", "1",
                 "--steps", "10", "--cycles", "1",
                 "--platform", "NoSuchPlatform"]) == 2


def test_settings_are_checked_before_any_process_starts():
    import pytest

    from resremd.errors import InputError

    with pytest.raises(TypeError):
        measure(testsystems.lj_box(), n_replica=2, isolate=True)
    with pytest.raises(InputError, match="at least 1"):
        measure(testsystems.lj_box(), contexts_per_device=[0], isolate=True)
    with pytest.raises(InputError, match="rest2_selection: solute"):
        measure(testsystems.lj_box(), rest2=True, isolate=True)


def test_counts_must_be_whole_numbers(capsys):
    import numpy as np
    import pytest

    from resremd.errors import InputError

    for counts, match in (([1.5], "takes whole numbers, not 1.5"),
                          ([True], "takes whole numbers"),
                          ([], "no count"),
                          (0, "at least 1, not 0")):
        with pytest.raises(InputError, match=match):
            measure(testsystems.lj_box(), contexts_per_device=counts)
    for name, value, match in (("n_replicas", 1, "at least 2, not 1"),
                               ("steps", 0, "at least 1, not 0"),
                               ("steps", -5, "at least 1, not -5"),
                               ("cycles", 2.0, "whole numbers, not 2.0")):
        with pytest.raises(InputError, match=f"{name} takes .*{match}"):
            measure(testsystems.lj_box(), **{name: value})
    rows = measure(testsystems.lj_box(), n_replicas=np.int64(2),
                   contexts_per_device=np.int64(1), steps=10, cycles=1,
                   platform="Reference")
    json.dumps(rows)
    for option, value, least in (("--contexts-per-device", "x", 1),
                                 ("--n-replicas", "1", 2),
                                 ("--steps", "-5", 1), ("--cycles", "0", 1)):
        with pytest.raises(SystemExit):
            main(["throughput", "--prepared", "x", option, value])
        assert f"{value} is not a count of at least {least}" in \
            capsys.readouterr().err


def _script(tmp_path, text, *, stdin=False, wait=True):
    """Run a script in a new interpreter, as a user would: from a file, or
    read from standard input."""
    import os
    import subprocess
    import sys

    path = tmp_path / "script.py"
    path.write_text(text)
    env = {**os.environ,
           "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    if not wait:
        return subprocess.Popen([sys.executable, str(path)], cwd=tmp_path,
                                env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True,
                                start_new_session=True)
    if stdin:
        return subprocess.run([sys.executable, "-"], input=text,
                              cwd=tmp_path, env=env, capture_output=True,
                              text=True, timeout=300)
    return subprocess.run([sys.executable, str(path)], cwd=tmp_path,
                          env=env, capture_output=True, text=True,
                          timeout=300)


TIME_IT = """
from resremd import testsystems
from resremd.throughput import measure

def time_it(steps=10):
    rows = measure(testsystems.lj_box(), n_replicas=2,
                   contexts_per_device=[1], steps=steps, cycles=1,
                   platform="Reference", isolate=True)
    return [r.get("error", "ok") for r in rows]
"""


def test_a_script_without_a_main_guard_is_told_so(tmp_path):
    """Its timing process runs the script again; there measure refuses at
    once rather than timing, and the script's own rows name the cause."""
    done = _script(tmp_path, TIME_IT + "print('rows', time_it())\n")
    assert done.returncode == 0, done.stderr
    assert done.stdout.count("rows") == 1
    assert "called again as a timing process started" in done.stderr
    assert "__main__" in done.stdout.split("rows", 1)[1]


def test_where_timing_processes_cannot_start_is_said_at_once(tmp_path):
    done = _script(tmp_path, TIME_IT + "print('rows', time_it())\n",
                   stdin=True)
    assert done.returncode == 1
    assert "cannot run from standard input" in done.stderr
    done = _script(tmp_path, TIME_IT + """
import multiprocessing

if __name__ == "__main__":
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        pool.apply(time_it)
""")
    assert done.returncode == 1
    assert "from a daemonic process" in done.stderr


def test_a_process_of_the_users_own_may_time(tmp_path):
    done = _script(tmp_path, TIME_IT + """
import multiprocessing

def report(channel):
    channel.put(time_it())

if __name__ == "__main__":
    spawn = multiprocessing.get_context("spawn")
    channel = spawn.Queue()
    worker = spawn.Process(target=report, args=(channel,))
    worker.start()
    print("rows", channel.get())
    worker.join()
""")
    assert done.returncode == 0, done.stderr
    assert "rows ['ok']" in done.stdout


def test_a_count_that_cannot_start_is_recorded(monkeypatch):
    import multiprocessing

    context = multiprocessing.get_context("spawn")

    def refuse(self):
        raise RuntimeError("cannot send this System")

    monkeypatch.setattr(type(context.Process(target=print)), "start", refuse)
    rows = measure(testsystems.lj_box(), n_replicas=2,
                   contexts_per_device=[1, 2], steps=10, cycles=1,
                   platform="Reference", isolate=True)
    assert [r["error"] for r in rows] == ["could not start: cannot send "
                                          "this System"] * 2


@pytest.mark.parametrize("handler", [False, True])
def test_sigterm_stops_the_timing_process_too(tmp_path, handler):
    """Two SIGTERMs, once the timing process runs: without a handler the
    script dies of the signal; with its own, the handler decides. Either
    way no timing process is left, even one that inherited the handler."""
    import os
    import signal
    import time

    script = _script(tmp_path, TIME_IT + ("""
import signal

def leave(number, frame):
    raise SystemExit(7)

signal.signal(signal.SIGTERM, leave)
""" if handler else "") + """
import multiprocessing
import threading
import time

def announce():
    while not multiprocessing.active_children():
        time.sleep(0.05)
    time.sleep(0.5)
    print("worker", multiprocessing.active_children()[0].pid, flush=True)

if __name__ == "__main__":
    threading.Thread(target=announce, daemon=True).start()
    time_it(steps=10**7)
""", wait=False)
    worker = None
    try:
        line = script.stdout.readline()
        assert line.startswith("worker"), script.stderr.read()
        worker = int(line.split()[1])
        os.kill(script.pid, signal.SIGTERM)
        os.kill(script.pid, signal.SIGTERM)
        status = script.wait(timeout=60)
        assert status == (7 if handler else -signal.SIGTERM)
        if not handler:
            assert "the timing process was stopped" in script.stderr.read()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                os.kill(worker, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            pytest.fail("the timing process was left running")
    finally:
        for pid, kill in ((script.pid, os.killpg), (worker, os.kill)):
            try:
                if pid is not None:
                    kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        script.wait()
