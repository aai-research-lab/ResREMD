import os
import signal

import numpy as np
import pytest

from resremd import testsystems
import resremd
from resremd.errors import InputError, ReservoirError
from resremd.output import DcdTrajectory
from resremd.reservoir import Reservoir
from resremd.system import subset_topology

md = pytest.importorskip("mdtraj")


def test_generate_records_how_independent_the_frames_are(tmp_path):
    meta = resremd.generate_reservoir(
        testsystems.double_well(), output=str(tmp_path / "r"), temperature_K=520.0,
        duration_ns=0.2, frame_interval_steps=200, equilibration_ns=0.01,
        platform="Reference", random_seed=2, friction_per_ps=5.0)
    assert meta["complete"] and meta["n_frames"] == 500
    stats = meta["statistics"]
    assert 0 < stats["effective_independent_frames"] <= 500
    res = Reservoir.open(tmp_path / "r")
    assert res.temperature_K == 520.0
    assert not (tmp_path / "r/build_checkpoint.npz").exists()


def test_generate_stops_and_resumes(tmp_path):
    def stop(info):
        os.kill(os.getpid(), signal.SIGTERM)

    common = dict(output=str(tmp_path / "r"), temperature_K=520.0,
                  duration_ns=1.2, frame_interval_steps=200,
                  equilibration_ns=0.0, platform="Reference", random_seed=2,
                  minimize=False)
    # A frame every 0.4 ps and a checkpoint every 0.5 ns: 1250 frames. The
    # signal arrives at the first checkpoint, frame 1250 of 3000.
    meta = resremd.generate_reservoir(testsystems.double_well(), on_progress=stop,
                                      **common)
    assert not meta["complete"]
    with pytest.raises(ReservoirError, match="not finished"):
        Reservoir.open(tmp_path / "r")
    meta = resremd.generate_reservoir(testsystems.double_well(), resume=True,
                                      **common)
    assert meta["complete"]
    assert Reservoir.open(tmp_path / "r").n_frames == 3000


def _write_dcd(path, prepared, frames, boxes=None):
    top = subset_topology(prepared.topology, np.arange(prepared.n_atoms))
    traj = DcdTrajectory(path, top, timestep_ps=0.002, interval_steps=10)
    for k, f in enumerate(frames):
        traj.write(f, None if boxes is None else boxes[k])
    traj.close()


def test_import_with_weights_and_stride(tmp_path):
    prepared = testsystems.double_well()
    frames = [np.array([[x, 0.0, 0.0]]) for x in np.linspace(-1, 1, 10)]
    _write_dcd(tmp_path / "a.dcd", prepared, frames[:6])
    _write_dcd(tmp_path / "b.dcd", prepared, frames[6:])
    from openmm import app

    with open(tmp_path / "top.pdb", "w") as fh:
        app.PDBFile.writeFile(prepared.topology, prepared.positions * 10, fh)
    np.save(tmp_path / "w.npy", np.arange(1.0, 11.0))
    meta = resremd.import_reservoir(
        trajectories=[str(tmp_path / "a.dcd"), str(tmp_path / "b.dcd")],
        topology=str(tmp_path / "top.pdb"), output=str(tmp_path / "r"),
        temperature_K=400.0, kind="weighted", weights=str(tmp_path / "w.npy"),
        stride=2)
    assert meta["n_frames"] == 5
    res = Reservoir.open(tmp_path / "r")
    assert np.allclose(res.weights, np.array([1, 3, 5, 7, 9]) / 25.0)
    assert np.allclose(res.positions[:, 0, 0], np.linspace(-1, 1, 10)[::2],
                       atol=1e-6)


def test_import_refuses_what_it_cannot_know(tmp_path):
    prepared = testsystems.lj_box()
    boxes = [prepared.box * s for s in (1.0, 1.01, 0.99)]
    _write_dcd(tmp_path / "npt.dcd", prepared, [prepared.positions] * 3, boxes)
    from openmm import app

    with open(tmp_path / "top.pdb", "w") as fh:
        app.PDBFile.writeFile(prepared.topology, prepared.positions * 10, fh)
    common = dict(trajectories=[str(tmp_path / "npt.dcd")],
                  topology=str(tmp_path / "top.pdb"), temperature_K=150.0)
    with pytest.raises(ReservoirError, match="constant pressure"):
        resremd.import_reservoir(output=str(tmp_path / "r1"), **common)
    with pytest.raises(InputError, match="weights"):
        resremd.import_reservoir(output=str(tmp_path / "r2"),
                                 kind="weighted", **common)
    meta = resremd.import_reservoir(output=str(tmp_path / "r3"),
                                    pressure_bar=200.0, **common)
    assert meta["ensemble"]["pressure_bar"] == 200.0


def test_generate_records_its_cost(tmp_path):
    meta = resremd.generate_reservoir(
        testsystems.double_well(), output=str(tmp_path / "r"),
        temperature_K=520.0, duration_ns=0.02, frame_interval_steps=100,
        equilibration_ns=0.002, platform="Reference", random_seed=2)
    assert meta["cost"]["md_steps"] == {"equilibration": 1000,
                                        "production": 10000}
    assert meta["cost"]["wall_seconds"] > 0
