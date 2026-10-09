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


def test_each_count_is_timed_in_its_own_process(tmp_path, monkeypatch):
    import os
    import signal
    import tempfile

    import numpy as np

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
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
    assert not os.listdir(tmp_path)  # nothing left behind


def test_a_failing_count_is_recorded_and_the_others_reported(tmp_path,
                                                             capsys):
    """A count that fails in its process becomes a row saying why; a table
    in which every count failed makes the command fail."""
    import openmm

    from resremd.throughput import format_rows

    # A platform this OpenMM may have but does not here.
    here = {openmm.Platform.getPlatform(i).getName()
            for i in range(openmm.Platform.getNumPlatforms())}
    missing = next((name for name in ("HIP", "OpenCL", "CUDA")
                    if name not in here), None)
    if missing is None:
        pytest.skip("every GPU platform is here")
    rows = measure(testsystems.lj_box(), n_replicas=2,
                   contexts_per_device=[1, 2], steps=10, cycles=1,
                   platform=missing, isolate=True)
    assert len(rows) == 2 and all(f"no {missing} platform" in r["error"]
                                  for r in rows)
    assert "failed" in format_rows(rows, 2)
    assert f"on {missing}" in format_rows(rows, 2, missing)
    p = testsystems.lj_box()
    write_prepared(tmp_path / "setup", p.system, p.topology, p.positions,
                   p.box)
    assert main(["-q", "throughput", "--prepared", str(tmp_path / "setup"),
                 "--n-replicas", "2", "--contexts-per-device", "1",
                 "--steps", "10", "--cycles", "1",
                 "--platform", missing]) == 2


def test_settings_are_checked_before_any_process_starts():
    from resremd.errors import InputError

    with pytest.raises(TypeError):
        measure(testsystems.lj_box(), n_replica=2, isolate=True)
    with pytest.raises(InputError, match="at least 1"):
        measure(testsystems.lj_box(), contexts_per_device=[0], isolate=True)
    with pytest.raises(InputError, match="rest2_selection: solute"):
        measure(testsystems.lj_box(), rest2=True, isolate=True)


def test_counts_must_be_whole_numbers(capsys):
    import numpy as np

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
                               ("cycles", 2.0, "whole numbers, not 2.0"),
                               ("timestep_fs", 0, "above 0, not 0"),
                               ("timestep_fs", float("nan"), "not nan"),
                               ("temperature_K", -1.0, "not -1.0")):
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
            ("--cycles", "0", "0 is not a count of at least 1"),
            ("--timestep-fs", "nan", "nan is not a number above 0"),
            ("--timestep-fs", "0", "0 is not a number above 0")):
        with pytest.raises(SystemExit):
            main(["throughput", "--prepared", "x", option, value])
        assert said in capsys.readouterr().err


def test_settings_are_checked_as_for_a_run():
    import numpy as np
    import openmm

    from resremd.errors import InputError

    box = testsystems.lj_box()
    for setting, match in ((dict(ensemble="NVT"), "ensemble"),
                           (dict(rest2_selection="bogus"), "rest2_selection"),
                           (dict(rest2="no"), "rest2"),
                           (dict(precision="quad"), "precision"),
                           (dict(cpu_threads=-3), "cpu_threads"),
                           (dict(random_seed=-1), "random_seed"),
                           (dict(platform="reference"), "platform"),
                           (dict(steps=2**31), "at most")):
        with pytest.raises(InputError, match=match):
            measure(box, **setting)
    with pytest.raises(TypeError, match="unexpected settings: counts"):
        measure(box, counts=[1])
    with pytest.raises(InputError, match="a Prepared system"):
        measure(openmm.System())
    # Numbers of numpy's own kinds come back as plain ones.
    rows = measure(box, n_replicas=2, contexts_per_device=[1], steps=10,
                   cycles=1, timestep_fs=np.float32(2.0),
                   platform="Reference")
    json.dumps(rows)


def test_devices_are_checked_as_for_a_run(tmp_path):
    import numpy as np

    import resremd
    from resremd.errors import InputError

    small = dict(n_replicas=2, contexts_per_device=[1], steps=10, cycles=1,
                 platform="Reference")
    for devices in (["a"], [1.7], [-1], [True], ["0"]):
        code = "resremd.input.range" if devices == [-1] \
            else "resremd.input.type"
        with pytest.raises(InputError, match="GPU indices") as error:
            measure(testsystems.lj_box(), devices=devices, **small)
        assert error.value.code == code
        with pytest.raises(InputError, match="GPU indices") as error:
            resremd.run(testsystems.lj_box(), output=str(tmp_path / "run"),
                        temperatures_K=[100.0, 120.0], production_steps=10,
                        exchange_interval_steps=10, platform="Reference",
                        devices=devices)
        assert error.value.code == code
    rows = measure(testsystems.lj_box(), n_replicas=2,
                   contexts_per_device=[1], steps=10, cycles=1,
                   platform="Reference", devices=[np.int64(0)])
    json.dumps(rows)


def test_exchanges_are_timed_with_their_velocity_rescaling(monkeypatch):
    """As in a run, a replica that moves to another temperature has its
    velocities rescaled, which costs a GPU a round trip."""
    import numpy as np

    from resremd.engine import Engine

    seen = []
    run = Engine.run

    def watched(self, replicas, temperatures, *args, **kwargs):
        seen.append((list(temperatures),
                     [r.velocity_scale for r in replicas]))
        return run(self, replicas, temperatures, *args, **kwargs)

    monkeypatch.setattr(Engine, "run", watched)
    measure(testsystems.lj_box(), n_replicas=4, contexts_per_device=[1],
            steps=10, cycles=2, platform="Reference")
    # Each cycle after the first: from the last temperature to this one.
    for (before, _), (after, scales) in zip(seen, seen[1:]):
        assert np.allclose(scales, np.sqrt(np.array(after) / before))


def test_timing_processes_log_through_their_caller(tmp_path):
    """What a timing process logs is handled by the caller's logging: once,
    where the caller logs, and only as much as the caller logs."""
    done = _script(tmp_path, """
import logging

from resremd import testsystems
from resremd.throughput import measure

def time_it(count):
    # devices on Reference: a warning from the timing process.
    measure(testsystems.lj_box(), n_replicas=2, contexts_per_device=[count],
            steps=10, cycles=1, platform="Reference", devices=[0],
            isolate=True)

if __name__ == "__main__":
    logging.basicConfig(filename="log.txt", level=logging.INFO)
    time_it(1)
    logging.getLogger().setLevel(logging.WARNING)
    time_it(2)
    logging.disable(logging.CRITICAL)
    time_it(3)
""")
    assert done.returncode == 0, done.stderr
    assert "contexts per device" not in done.stderr, done.stderr
    assert "ignored" not in done.stderr, done.stderr
    log = (tmp_path / "log.txt").read_text()
    assert log.count("1 contexts per device:") == 1, log
    assert "2 contexts per device:" not in log, log
    assert log.count("`devices` is ignored") == 2, log


def test_a_callers_unusual_logging_neither_loses_nor_stops_anything(
        tmp_path):
    """Logging everything (NOTSET), a record factory of its own, a filter
    that fails: the timing goes on, and its lines arrive."""
    done = _script(tmp_path, """
import logging

from resremd import testsystems
from resremd.throughput import measure

def time_it():
    rows = measure(testsystems.lj_box(), n_replicas=2,
                   contexts_per_device=[1], steps=10, cycles=1,
                   platform="Reference", isolate=True)
    print("rows", [r.get("error", "ok") for r in rows])

class Broken(logging.Filter):
    def filter(self, record):
        raise RuntimeError("a filter that fails")

if __name__ == "__main__":
    logging.basicConfig(filename="log.txt", level=logging.NOTSET,
                        format="%(levelname)s %(relativeCreated)d "
                               "%(message)s")
    # The caller's name for a level is the one shown.
    logging.addLevelName(logging.INFO, "NOTE")
    logging.getLogger("before").warning("mark")
    time_it()
    logging.getLogger("after").warning("mark")
    made = logging.getLogRecordFactory()

    def factory(name, *args, **kwargs):
        record = made(name, *args, **kwargs)
        record.package = name.split(".")[0]
        return record

    logging.setLogRecordFactory(factory)
    logging.getLogger().handlers[0].setFormatter(
        logging.Formatter("%(package)s %(message)s"))
    time_it()
    # One that takes its arguments only as Logger.makeRecord gives them.
    logging.setLogRecordFactory(lambda *fields: factory(*fields))
    time_it()
    logging.getLogger("resremd").addFilter(Broken())
    time_it()
    # Said only as logging would say it.
    logging.raiseExceptions = False
    time_it()
""")
    assert done.returncode == 0, done.stderr
    assert done.stdout.count("rows ['ok']") == 5, done.stdout + done.stderr
    # The failing filter is said once, as logging says such things.
    assert done.stderr.count("--- Logging error") == 1, done.stderr
    assert "a filter that fails" in done.stderr, done.stderr
    log = (tmp_path / "log.txt").read_text()
    assert log.count("1 contexts per device:") == 3, log
    first = log.split("\n")[:3]
    assert first[1].startswith("NOTE "), log
    # Its time since logging began falls between the caller's lines.
    times = [int(line.split()[1]) for line in first]
    assert times[0] <= times[1] <= times[2], log
    assert "resremd 1 contexts per device:" in log, log


def test_a_system_in_memory_is_given_whole():
    from resremd.errors import InputError

    box = testsystems.lj_box()
    small = dict(n_replicas=2, contexts_per_device=[1], steps=10, cycles=1,
                 platform="Reference")
    assert measure((box.system, box.topology, box.positions), **small)
    for wrong, match in (((box.system, box.topology), "a Prepared"),
                         ((box.system, box.topology, box.positions, None,
                           None), "a Prepared"),
                         ((box.system, box.topology,
                           box.positions.ravel()), "rows of three")):
        with pytest.raises(InputError, match=match):
            measure(wrong, **small)


def test_counts_that_ran_alike_are_not_told_apart():
    from resremd.throughput import format_rows

    row = dict(platform="CPU", replicas_resident=True, overhead_fraction=0.1,
               ns_per_day_per_replica=1.0)
    rows = [dict(row, contexts_per_device=c, contexts=n, ns_per_day_total=t)
            for c, n, t in ((1, 1, 10.0), (2, 2, 20.0), (4, 2, 21.0))]
    assert format_rows(rows, 2).endswith("fastest: contexts_per_device: 2")


def _script(tmp_path, text, *, stdin=False):
    """Run a script in a new interpreter, as a user would: from a file, or
    read from standard input. Whatever it started goes with it if it
    overruns."""
    import os
    import signal
    import subprocess
    import sys

    path = tmp_path / "script.py"
    path.write_text(text)
    temporary = tmp_path / "tmp"
    temporary.mkdir(exist_ok=True)
    env = {**os.environ, "TMPDIR": str(temporary),
           "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    script = subprocess.Popen(
        [sys.executable, "-" if stdin else str(path)], cwd=tmp_path, env=env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True)
    try:
        out, err = script.communicate(text if stdin else "", timeout=120)
    finally:
        try:
            os.killpg(script.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        script.wait()
    assert not os.listdir(temporary), "the script left temporary files"
    return subprocess.CompletedProcess(script.args, script.returncode, out,
                                       err)


# As pytest itself may have been started with SIGTERM ignored, which the
# scripts would inherit.
DEFAULT = """
import signal

signal.signal(signal.SIGTERM, signal.SIG_DFL)
"""


TIME_IT = """
from resremd import testsystems
from resremd.throughput import measure

def time_it(steps=10, counts=(1,), n_side=5):
    rows = measure(testsystems.lj_box(n_side=n_side), n_replicas=2,
                   contexts_per_device=list(counts), steps=steps, cycles=1,
                   platform="Reference", isolate=True)
    return [r.get("error", "ok") for r in rows]
"""


def test_a_script_without_a_main_guard_is_told_so(tmp_path):
    """Its timing process runs the script again; there measure refuses at
    once rather than timing, and the script's own rows name the cause."""
    # A large System the process never reads, from a script that lets a
    # broken pipe end it.
    done = _script(tmp_path, TIME_IT + """
import signal

signal.signal(signal.SIGPIPE, signal.SIG_DFL)
print('rows', time_it(n_side=10))
""")
    assert done.returncode == 0, done.stderr
    assert done.stdout.count("rows") == 1
    assert "ran again inside a timing process" in done.stderr
    assert "Traceback" not in done.stderr, done.stderr
    assert "__main__" in done.stdout.split("rows", 1)[1]


def test_where_timing_processes_cannot_start_is_said_at_once(tmp_path):
    done = _script(tmp_path, TIME_IT + "print('rows', time_it())\n",
                   stdin=True)
    assert done.returncode == 1
    assert "cannot run from standard input" in done.stderr
    done = _script(tmp_path, DEFAULT + TIME_IT + """
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


def test_a_process_that_reported_is_judged_by_its_exit(tmp_path):
    """One that then fails as it exits, or hangs there, keeps its row and
    says so."""
    for exit, said in (("os._exit(3)", "(process failed on exit)"),
                       ("time.sleep(600)", "(process hung on exit)")):
        done = _script(tmp_path, f"""
import atexit
import os
import time

from resremd import testsystems, throughput

if __name__ == "__mp_main__":
    atexit.register(lambda: {exit})
if __name__ == "__main__":
    throughput.HUNG_AFTER_S = 2.0
    rows = throughput.measure(testsystems.lj_box(), n_replicas=2,
                              contexts_per_device=[1], steps=10, cycles=1,
                              platform="Reference", isolate=True)
    print(throughput.format_rows(rows, 2))
""")
        assert done.returncode == 0, done.stderr
        assert said in done.stdout, done.stdout


def test_a_report_with_nowhere_to_go_ends_the_process_quietly(tmp_path):
    """As when its parent has gone: it just ends."""
    done = _script(tmp_path, """
import multiprocessing

from resremd.throughput import _worker

if __name__ == "__main__":
    results, reporter = multiprocessing.Pipe(duplex=False)
    results.close()
    _worker(reporter, "no such file", 1, {})
""")
    assert done.returncode == 1
    assert done.stderr == ""


def test_a_full_temporary_directory_gives_rows_saying_so(tmp_path):
    """With nowhere to write an in-memory System, each count says so and
    the script goes on; a directory needs no temporary file at all."""
    done = _script(tmp_path, """
import resource

from resremd import testsystems
from resremd.system import write_prepared
from resremd.throughput import measure

if __name__ == "__main__":
    box = testsystems.lj_box()
    write_prepared("setup", box.system, box.topology, box.positions,
                   box.box)
    hard = resource.getrlimit(resource.RLIMIT_FSIZE)[1]
    # Files written from here on stop at none, so that no temporary
    # directory is usable at all, and then at 10 bytes, as on a nearly
    # full disk.
    for limit, source in ((0, box), (10, "setup"), (10, box)):
        resource.setrlimit(resource.RLIMIT_FSIZE, (limit, hard))
        rows = measure(source, n_replicas=2, contexts_per_device=[1, 2],
                       steps=10, cycles=1, platform="Reference",
                       isolate=True)
        print("rows", [r.get("error", "ok") for r in rows])
""")
    assert done.returncode == 0, done.stderr
    lines = [line for line in done.stdout.split("\n")
             if line.startswith("rows")]
    assert lines[1] == "rows ['ok', 'ok']", done.stderr
    assert len(lines) == 3 and all(
        line.count("could not be written to a temporary file") == 2
        for line in (lines[0], lines[2])), done.stdout


def test_a_count_that_cannot_start_is_recorded(monkeypatch):
    import multiprocessing

    context = multiprocessing.get_context("spawn")

    def refuse(self):
        raise RuntimeError("no more processes")

    settings = dict(n_replicas=2, contexts_per_device=[1, 2], steps=10,
                    cycles=1, platform="Reference", isolate=True)
    with monkeypatch.context() as patch:
        patch.setattr(context.Process, "start", refuse)
        rows = measure(testsystems.lj_box(), **settings)
    assert [r["error"] for r in rows] == \
        ["could not start: no more processes"] * 2
    # A System that cannot be sent.
    box = testsystems.lj_box()
    box.unsendable = lambda: None
    rows = measure(box, **settings)
    assert len(rows) == 2 and all("could not start" in r["error"]
                                  for r in rows)


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


def test_a_process_ended_by_a_signal_is_named(tmp_path):
    done = _script(tmp_path, TIME_IT + """
import os
import signal
import threading

if __name__ == "__mp_main__":
    threading.Timer(1.0, os.kill, (os.getpid(), signal.SIGKILL)).start()
if __name__ == "__main__":
    print("rows", time_it(steps=10**7))
""")
    assert done.returncode == 0, done.stderr
    assert "ended by SIGKILL" in done.stdout


# Every script says which processes it started, so that the test can see
# that none is left.
LAUNCHES = """
from multiprocessing import popen_spawn_posix as _spawning

def _launch(self, obj, _launch=_spawning.Popen._launch):
    _launch(self, obj)
    print("launched", self.pid, flush=True)

_spawning.Popen._launch = _launch
"""
# Says when the timing process runs, for a signal sent then.
ANNOUNCE = """
import multiprocessing
import threading
import time

def announce():
    while not multiprocessing.active_children():
        time.sleep(0.05)
    time.sleep(0.5)
    print("worker", multiprocessing.active_children()[0].pid, flush=True)
"""
# Times one count for hours unless it is stopped.
LONG = ANNOUNCE + """
if __name__ == "__main__":
    threading.Thread(target=announce, daemon=True).start()
    time_it(steps=10**7)
"""
SIGNAL_ITSELF = DEFAULT + """
import os
from multiprocessing import popen_spawn_posix, process, util

def then_signal(function, note=None, number=signal.SIGTERM):
    def wrapped(*args, **kwargs):
        result = function(*args, **kwargs)
        os.kill(os.getpid(), number)
        if note:
            print(note, flush=True)
        return result
    return wrapped
"""
# Each way a timing script can be stopped: the script; the signal the test
# sends once the timing process runs (none: the script signals itself);
# the status the script ends with; whether it says it was stopped; what
# else its output shows; how many processes it starts; and the seconds it
# may take to end.
STOPS = {
    # No handler: the script dies of the signal.
    "sigterm": dict(script=DEFAULT + LONG, send="SIGTERM", status=-15),
    # Its own handler, which the timing process inherits as the script is
    # imported again there, decides; the timing process still stops at once.
    "own handler": dict(script="""
import multiprocessing
import signal

def leave(number, frame):
    if multiprocessing.parent_process() is None:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise SystemExit(7)

signal.signal(signal.SIGTERM, leave)
""" + LONG, send="SIGTERM", status=7, says=False, within=5),
    # Killed outright: the timing process sees it go.
    "sigkill": dict(script=DEFAULT + LONG, send="SIGKILL", status=-9,
                    says=False),
    # A second signal while the timing process is stopped does not cut
    # the cleanup short.
    "again in cleanup": dict(script=SIGNAL_ITSELF + """
process.BaseProcess.terminate = then_signal(process.BaseProcess.terminate,
                                            "cleanup done")
""" + LONG, send="SIGTERM", status=-15, shows="cleanup done"),
    # A SIGTERM while a Ctrl-C is being handled still ends the script.
    "after a ctrl-c": dict(script=SIGNAL_ITSELF + """
# A shell starts a background job with SIGINT ignored.
signal.signal(signal.SIGINT, signal.default_int_handler)
process.BaseProcess.terminate = then_signal(process.BaseProcess.terminate)
""" + ANNOUNCE + """
if __name__ == "__main__":
    threading.Thread(target=announce, daemon=True).start()
    try:
        time_it(steps=10**7)
    except KeyboardInterrupt:
        print("still running", flush=True)
        time.sleep(5)
""", send="SIGINT", status=-15),
    # A signal as a timing process is being started.
    "while starting": dict(script=SIGNAL_ITSELF + """
def launch(self, obj):
    launched(self, obj)
    print("worker", self.pid, flush=True)
    os.kill(os.getpid(), signal.SIGTERM)

launched = popen_spawn_posix.Popen._launch
popen_spawn_posix.Popen._launch = launch

if __name__ == "__main__":
    time_it(steps=10**7)
""", status=-15),
    # A signal while a process imports the script slowly, with a large
    # System to read: the stop does not wait for the import, even in a
    # script that lets a broken pipe end it.
    "during a slow import": dict(script=DEFAULT + """
import os
import threading
import time

signal.signal(signal.SIGPIPE, signal.SIG_DFL)

if __name__ == "__mp_main__":
    time.sleep(60)
if __name__ == "__main__":
    threading.Timer(2.0, os.kill, (os.getpid(), signal.SIGTERM)).start()
    time_it(steps=10**7, n_side=10)
""", status=-15, within=20),
    # Killed outright while a process still imports the script: that
    # process ends quietly once it finds its parent gone.
    "sigkill during a slow import": dict(script=DEFAULT + """
import time

if __name__ == "__mp_main__":
    time.sleep(3)
""" + LONG, send="SIGKILL", status=-9, says=False),
    # A signal in a finalizer, as the second count is set up: no process
    # is started after it.
    "in a finalizer": dict(script=SIGNAL_ITSELF + """
util.close_fds = then_signal(util.close_fds)

if __name__ == "__main__":
    time_it(counts=(1, 1, 1))
""", status=-15, launches=1),
    # A signal while a process that reported is stuck on its way out.
    "hung on exit": dict(script=SIGNAL_ITSELF + """
import atexit
import time

if __name__ == "__mp_main__":
    atexit.register(time.sleep, 600)
process.BaseProcess.join = then_signal(process.BaseProcess.join)

if __name__ == "__main__":
    time_it()
""", status=-15, within=20),
}


@pytest.mark.parametrize("how", list(STOPS))
def test_a_stopped_script_leaves_no_timing_process(tmp_path, how):
    import os
    import signal
    import subprocess
    import sys
    import time

    stop = {"send": None, "says": True, "shows": None, "launches": None,
            "within": 60, **STOPS[how]}
    (tmp_path / "script.py").write_text(LAUNCHES + TIME_IT + stop["script"])
    temporary = tmp_path / "tmp"
    temporary.mkdir()
    env = {**os.environ, "TMPDIR": str(temporary),
           "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    out, err = tmp_path / "out.txt", tmp_path / "err.txt"
    with open(out, "w") as o, open(err, "w") as e:
        script = subprocess.Popen([sys.executable, "script.py"],
                                  cwd=tmp_path, env=env, stdout=o, stderr=e,
                                  start_new_session=True)

    def launched():
        return [int(line.split()[1]) for line in out.read_text().split("\n")
                if line.startswith("launched ")]

    started = time.monotonic()
    try:
        if stop["send"]:
            while "worker" not in out.read_text() and \
                    script.poll() is None and \
                    time.monotonic() < started + 120:
                time.sleep(0.1)
            assert "worker" in out.read_text(), err.read_text()
            started = time.monotonic()
            os.kill(script.pid, getattr(signal, stop["send"]))
        status = script.wait(timeout=120)
        assert time.monotonic() - started < stop["within"]
        assert status == stop["status"], err.read_text()
        assert launched()
        if stop["launches"]:
            assert len(launched()) == stop["launches"]
        if status != -signal.SIGKILL:
            # Stopped and reaped before the script ended.
            assert not any(_alive(pid) for pid in launched()), \
                "a timing process was left running"
        else:
            # Killed outright, the script leaves its processes to see that.
            while any(_alive(pid) for pid in launched()) and \
                    time.monotonic() < started + 30:
                time.sleep(0.1)
            assert not any(_alive(pid) for pid in launched()), \
                "a timing process was left running"
        said = err.read_text()
        assert ("Stopped by SIGTERM." in said) == stop["says"]
        # Nothing it started fails noisily on the way down, and nothing is
        # left for the resource tracker to clean up after it.
        assert "Traceback" not in said, said
        assert "leaked" not in said, said
        if stop["shows"]:
            assert stop["shows"] in out.read_text()
        # However it ended, it left no file behind.
        assert not os.listdir(temporary)
    finally:
        if script.poll() is None:
            os.killpg(script.pid, signal.SIGKILL)
            script.wait()
        # What is left of the script's own processes (its process group,
        # so that a number used again by another process is not touched).
        try:
            os.killpg(script.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
