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
    for option, value, said in (
            ("--contexts-per-device", "x", "x is not a whole number"),
            ("--n-replicas", "1", "1 is not a count of at least 2"),
            ("--steps", "-5", "-5 is not a count of at least 1"),
            ("--cycles", "0", "0 is not a count of at least 1")):
        with pytest.raises(SystemExit):
            main(["throughput", "--prepared", "x", option, value])
        assert said in capsys.readouterr().err


def _script(tmp_path, text, *, stdin=False):
    """Run a script in a new interpreter, as a user would: from a file, or
    read from standard input."""
    import os
    import subprocess
    import sys

    path = tmp_path / "script.py"
    path.write_text(text)
    env = {**os.environ,
           "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
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

def time_it(steps=10, counts=(1,)):
    rows = measure(testsystems.lj_box(), n_replicas=2,
                   contexts_per_device=list(counts), steps=steps, cycles=1,
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


def _alive(pid):
    """Whether a process runs; a zombie no one reaps here has ended."""
    import os

    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        if os.path.isdir("/proc/self"):
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


# The scripts below time one count for hours unless they are stopped.
LONG = """
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
"""
DEFAULT = """
import signal

# As pytest itself may have been started with SIGTERM ignored.
signal.signal(signal.SIGTERM, signal.SIG_DFL)
"""
SIGNAL_ITSELF = DEFAULT + """
import os
from multiprocessing import popen_spawn_posix, process, util

def then_signal(function, note=None):
    def wrapped(*args):
        result = function(*args)
        os.kill(os.getpid(), signal.SIGTERM)
        if note:
            print(note, flush=True)
        return result
    return wrapped
"""
# Each way a timing script can be stopped: the script, the signal the test
# sends once the timing process runs (None: the script signals itself),
# the status the script ends with, and whether it says it was stopped.
STOPS = {
    # No handler: the script dies of the signal.
    "sigterm": (DEFAULT + LONG, "SIGTERM", -15, True),
    # Its own handler, which the timing process inherits as the script is
    # imported again there, decides; the timing process still stops at once.
    "own handler": ("""
import multiprocessing
import signal

def leave(number, frame):
    if multiprocessing.parent_process() is None:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise SystemExit(7)

signal.signal(signal.SIGTERM, leave)
""" + LONG, "SIGTERM", 7, False),
    # Killed outright: the timing process sees it go.
    "sigkill": (DEFAULT + LONG, "SIGKILL", -9, False),
    # A second signal while the timing process is stopped does not cut
    # the cleanup short.
    "again in cleanup": (SIGNAL_ITSELF + """
process.BaseProcess.terminate = then_signal(process.BaseProcess.terminate,
                                            "cleanup done")
""" + LONG, "SIGTERM", -15, True),
    # A signal as a timing process is being started.
    "while starting": (SIGNAL_ITSELF + """
def launch(self, obj):
    launched(self, obj)
    print("worker", self.pid, flush=True)
    os.kill(os.getpid(), signal.SIGTERM)

launched = popen_spawn_posix.Popen._launch
popen_spawn_posix.Popen._launch = launch

if __name__ == "__main__":
    time_it(steps=10**7)
""", None, -15, True),
    # A signal in a finalizer, as the second count is set up: no process
    # is started after it.
    "in a finalizer": (SIGNAL_ITSELF + """
util.close_fds = then_signal(util.close_fds)

def launch(self, obj):
    print("launch", flush=True)
    launched(self, obj)

launched = popen_spawn_posix.Popen._launch
popen_spawn_posix.Popen._launch = launch

if __name__ == "__main__":
    time_it(counts=(1, 1, 1))
""", None, -15, True),
}


@pytest.mark.parametrize("how", list(STOPS))
def test_a_stopped_script_leaves_no_timing_process(tmp_path, how):
    import os
    import signal
    import subprocess
    import sys
    import time

    text, send, expected, says = STOPS[how]
    (tmp_path / "script.py").write_text(TIME_IT + text)
    env = {**os.environ,
           "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    out, err = tmp_path / "out.txt", tmp_path / "err.txt"
    with open(out, "w") as o, open(err, "w") as e:
        script = subprocess.Popen([sys.executable, "script.py"],
                                  cwd=tmp_path, env=env, stdout=o, stderr=e,
                                  start_new_session=True)
    worker = None
    try:
        deadline = time.monotonic() + 120
        while "\n" not in out.read_text() and script.poll() is None and \
                time.monotonic() < deadline:
            time.sleep(0.1)
        line = out.read_text().split("\n")[0]
        if line.startswith("worker"):
            worker = int(line.split()[1])
        sent = time.monotonic()
        if send:
            assert worker is not None, err.read_text()
            os.kill(script.pid, getattr(signal, send))
        status = script.wait(timeout=60)
        assert status == expected, err.read_text()
        assert ("Stopped by SIGTERM." in err.read_text()) == says
        # Nothing it started fails noisily on the way down.
        assert "Traceback" not in err.read_text(), err.read_text()
        if how == "own handler":
            assert time.monotonic() - sent < 5
        if how == "again in cleanup":
            assert "cleanup done" in out.read_text()
        if how == "in a finalizer":
            assert out.read_text().count("launch") == 1
        if worker is not None and how != "sigkill":
            # Stopped and reaped before the script ended.
            assert not _alive(worker), "the timing process was left running"
        elif worker is not None:
            while _alive(worker) and time.monotonic() < sent + 30:
                time.sleep(0.1)
            assert not _alive(worker), "the timing process was left running"
    finally:
        if script.poll() is None:
            os.killpg(script.pid, signal.SIGKILL)
            script.wait()
        if worker is not None and _alive(worker):
            os.kill(worker, signal.SIGKILL)
