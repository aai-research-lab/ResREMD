import json
import os
import signal

import numpy as np
import pytest

from resremd import testsystems
import resremd
from resremd.errors import InputError, ResumeError

md = pytest.importorskip("mdtraj")


def settings(tmp_path, fast, cycles, **extra):
    reservoir = tmp_path / "reservoir"
    if not reservoir.exists():
        testsystems.write_double_well_reservoir(reservoir, kind="boltzmann",
                                           n_frames=500, temperature_K=520.0)
    return {**fast,
            **dict(output=str(tmp_path / "run"), reservoir=str(reservoir),
                   temperatures_K=[300.0, 360.0, 432.0],
                   production_steps=50 * cycles,
                   trajectory_interval_steps=100,
                   checkpoint_interval_steps=500),
            **extra}


def rows(path):
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


def frames(run, s, t):
    return md.load(f"{run}/trajectories/state_{s:03d}_{t:.2f}K.dcd",
                   top=f"{run}/topology.pdb").n_frames


def test_outputs_and_manifest(tmp_path, fast):
    m = resremd.run(testsystems.double_well(), **settings(tmp_path, fast, 40))
    run = tmp_path / "run"
    assert m["status"] == "complete"
    assert m["progress"]["cycles_done"] == 40
    assert rows(run / "states.csv").shape == (40, 5)
    assert rows(run / "reservoir_exchanges.csv").shape[0] == 40
    assert frames(run, 0, 300.0) == 20
    on_disk = json.loads((run / "manifest.json").read_text())
    assert on_disk["reservoir"]["kind"] == "boltzmann"
    assert (run / "run.log").read_text().count("cycle") > 0
    # Every state is held by exactly one replica in every cycle.
    states = rows(run / "states.csv")[:, 2:]
    assert all(sorted(r) == [0, 1, 2] for r in states.tolist())


def test_a_fresh_run_will_not_overwrite(tmp_path, fast):
    resremd.run(testsystems.double_well(), **settings(tmp_path, fast, 10))
    with pytest.raises(InputError, match="holds a run"):
        resremd.run(testsystems.double_well(), **settings(tmp_path, fast, 10))


def test_resume_cuts_back_what_was_written_after_the_checkpoint(tmp_path, fast):
    resremd.run(testsystems.double_well(), **settings(tmp_path, fast, 20))
    run = tmp_path / "run"
    # Pretend the run went on past its last checkpoint and then died.
    with open(run / "states.csv", "a") as fh:
        fh.write("21,1.05,0,1,2\n22,1.1,0,1,2\n")
    with open(run / "trajectories/state_000_300.00K.dcd", "ab") as fh:
        fh.write(b"\0" * 64)
    m = resremd.run(testsystems.double_well(),
                    **settings(tmp_path, fast, 60, resume=True))
    assert m["status"] == "complete"
    cycles = rows(run / "states.csv")[:, 0]
    assert cycles.tolist() == list(range(1, 61))
    assert frames(run, 0, 300.0) == 30
    assert rows(run / "energies.csv").shape[0] == 60


def test_resume_refuses_different_science(tmp_path, fast):
    resremd.run(testsystems.double_well(), **settings(tmp_path, fast, 20))
    with pytest.raises(ResumeError, match="temperatures_K"):
        resremd.run(testsystems.double_well(), **settings(
            tmp_path, fast, 40, resume=True,
            temperatures_K=[300.0, 350.0, 432.0]))
    with pytest.raises(ResumeError, match="shortened"):
        resremd.run(testsystems.double_well(),
                    **settings(tmp_path, fast, 10, resume=True))


def test_a_signal_stops_at_a_cycle_with_a_checkpoint(tmp_path, fast):
    def stop_early(info):
        if info["cycle"] >= 20:
            os.kill(os.getpid(), signal.SIGTERM)

    m = resremd.run(testsystems.double_well(), on_progress=stop_early,
                    **settings(tmp_path, fast, 200))
    assert m["status"] == "stopped"
    done = m["progress"]["cycles_done"]
    assert 20 <= done < 200
    m = resremd.run(testsystems.double_well(),
                    **settings(tmp_path, fast, 200, resume=True))
    assert m["status"] == "complete"
    assert rows(tmp_path / "run/states.csv")[:, 0].tolist() == \
        list(range(1, 201))


def test_plain_replica_exchange_without_a_reservoir(tmp_path, fast):
    s = settings(tmp_path, fast, 20)
    s.pop("reservoir")
    m = resremd.run(testsystems.double_well(), **s)
    assert m["reservoir"] is None
    assert m["method"].startswith("temperature replica exchange")
    assert not (tmp_path / "run/reservoir_exchanges.csv").exists()


def test_lengths_that_do_not_divide_are_refused(tmp_path, fast):
    with pytest.raises(InputError, match="whole number"):
        resremd.run(testsystems.double_well(),
                    **{**settings(tmp_path, fast, 20), "production_steps": 1025})
    with pytest.raises(InputError, match="multiple"):
        resremd.run(testsystems.double_well(),
                    **{**settings(tmp_path, fast, 20),
                       "trajectory_interval_steps": 75})


def test_cost_is_counted_across_sessions(tmp_path, fast):
    m = resremd.run(testsystems.double_well(),
                    **settings(tmp_path, fast, 20, equilibration_ns=0.001))
    cost = m["cost"]
    assert cost["md_steps"]["equilibration"] == 500 * 3
    assert cost["md_steps"]["production"] == 20 * 50 * 3
    assert cost["md_steps_total"] == 1500 + 3000
    first = cost["wall_seconds"]["production"]
    assert first > 0
    m = resremd.run(testsystems.double_well(), **settings(
        tmp_path, fast, 40, equilibration_ns=0.001, resume=True))
    assert m["cost"]["md_steps"]["production"] == 40 * 50 * 3
    assert m["cost"]["md_steps"]["equilibration"] == 1500
    assert m["cost"]["wall_seconds"]["production"] > first
