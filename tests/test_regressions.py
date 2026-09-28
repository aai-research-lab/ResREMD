"""Regression tests: each guards a way this package could go wrong quietly."""

import numpy as np
import openmm
import pytest
from openmm import unit

import resremd
from resremd import testsystems
from resremd.cli import main
from resremd.engine import Engine
from resremd.errors import InputError, ReservoirError
from resremd.output import DcdTrajectory
from resremd.system import from_objects, subset_topology
from resremd.thermo import Ensemble

md = pytest.importorskip("mdtraj")


def test_generate_sets_the_barostat_to_the_reservoir_temperature(
        tmp_path, monkeypatch):
    import resremd.build as build

    made = {}
    real = build.create_context

    def spy(*args, **kwargs):
        ctx, name = real(*args, **kwargs)
        made["context"] = ctx
        return ctx, name

    monkeypatch.setattr(build, "create_context", spy)
    prepared = testsystems.lj_box(pressure=True)  # barostat made at 100 K
    resremd.generate_reservoir(prepared, output=str(tmp_path / "r"),
                               temperature_K=150.0, duration_ns=0.0004,
                               frame_interval_steps=100, timestep_fs=4.0,
                               equilibration_ns=0.0, platform="Reference",
                               random_seed=1)
    assert made["context"].getParameter("MonteCarloTemperature") == 150.0


def test_a_dcd_past_two_billion_steps_resumes(tmp_path):
    top = subset_topology(testsystems.double_well().topology, [0])
    path = tmp_path / "long.dcd"
    traj = DcdTrajectory(path, top, timestep_ps=0.002,
                         interval_steps=1_000_000)
    for k in range(2150):  # passes 2^31 steps at frame 2148
        traj.write(np.array([[k * 1e-3, 0, 0]]), None)
    size, frames = traj.flush(), traj.frames
    traj.write(np.array([[9.0, 0, 0]]), None)
    traj.close()
    traj = DcdTrajectory(path, top, timestep_ps=0.002,
                         interval_steps=1_000_000, resume=(size, frames))
    for k in range(2150, 2155):
        traj.write(np.array([[k * 1e-3, 0, 0]]), None)
    traj.close()
    x = md.load(str(path), top=md.Topology.from_openmm(top)).xyz[:, 0, 0]
    assert x.size == 2155
    assert np.allclose(x, np.arange(2155) * 1e-3, atol=1e-5)


def test_trajectories_keep_the_box_when_the_topology_had_none(tmp_path):
    box_model = testsystems.lj_box(pressure=True)
    box_model.topology.setPeriodicBoxVectors(None)
    prepared = from_objects(box_model.system, box_model.topology,
                            box_model.positions, box_model.box)
    resremd.run(prepared, output=str(tmp_path / "run"),
                temperatures_K=[100.0, 110.0], production_steps=400,
                exchange_interval_steps=100, trajectory_interval_steps=100,
                timestep_fs=4.0, platform="Reference", equilibration_ns=0.0,
                random_seed=1, save_selection="all")
    t = md.load(str(tmp_path / "run/trajectories/state_000_100.00K.dcd"),
                top=str(tmp_path / "run/topology.pdb"))
    assert t.unitcell_vectors is not None
    assert "CRYST1" in (tmp_path / "run/topology.pdb").read_text()


def test_a_reservoir_from_another_hamiltonian_is_refused(tmp_path):
    resremd.generate_reservoir(testsystems.double_well(),
                               output=str(tmp_path / "r"), temperature_K=520.0,
                               duration_ns=0.02, frame_interval_steps=100,
                               equilibration_ns=0.0, platform="Reference",
                               random_seed=1)
    common = dict(reservoir=str(tmp_path / "r"), temperatures_K=[300, 400],
                  production_steps=100, exchange_interval_steps=50,
                  platform="Reference", equilibration_ns=0.0,
                  save_selection="all", minimize=False)
    m = resremd.run(testsystems.double_well(), output=str(tmp_path / "a"),
                    **common)
    assert m["status"] == "complete" and not m["warnings"]
    with pytest.raises(ReservoirError) as err:
        resremd.run(testsystems.double_well(barrier=40.0),
                    output=str(tmp_path / "b"), **common)
    assert err.value.code == "resremd.reservoir.hamiltonian"


def test_membrane_areas_are_logged(tmp_path):
    prepared = testsystems.lj_box()
    system = openmm.XmlSerializer.deserialize(
        openmm.XmlSerializer.serialize(prepared.system))
    system.addForce(openmm.MonteCarloMembraneBarostat(
        200 * unit.bar, 5.0 * unit.bar * unit.nanometer, 100 * unit.kelvin,
        openmm.MonteCarloMembraneBarostat.XYIsotropic,
        openmm.MonteCarloMembraneBarostat.ZFree, 10))
    prepared = from_objects(system, prepared.topology, prepared.positions,
                            prepared.box)
    m = resremd.run(prepared, output=str(tmp_path / "run"),
                    temperatures_K=[100.0, 110.0], production_steps=400,
                    exchange_interval_steps=100, timestep_fs=4.0,
                    platform="Reference", equilibration_ns=0.0, random_seed=1,
                    save_selection="all")
    assert m["system"]["ensemble"]["barostat"] == "MonteCarloMembraneBarostat"
    areas = np.loadtxt(tmp_path / "run/areas.csv", delimiter=",", skiprows=1)
    assert areas.shape == (4, 4)


def test_nvt_removes_the_systems_barostat(tmp_path):
    m = resremd.run(testsystems.lj_box(pressure=True),
                    output=str(tmp_path / "run"), ensemble="nvt",
                    temperatures_K=[100.0, 110.0], production_steps=200,
                    exchange_interval_steps=100, timestep_fs=4.0,
                    platform="Reference", equilibration_ns=0.0, random_seed=1,
                    save_selection="all")
    assert m["system"]["ensemble"]["pressure_bar"] is None
    assert not (tmp_path / "run/volumes.csv").exists()


def test_no_more_contexts_than_replicas():
    p = testsystems.double_well()
    engine = Engine(system=p.system, ensemble=Ensemble(), n_replicas=3,
                    integrator="langevin_middle", timestep_fs=2.0,
                    friction_per_ps=1.0, temperature_K=300.0,
                    platform="Reference", precision="double", devices=None,
                    contexts_per_device=8, cpu_threads=None,
                    seeds=np.random.default_rng(0))
    assert len(engine.slots) == 3
    assert all(s.exclusive for s in engine.slots)
    engine.close()


def test_leftovers_of_a_run_that_never_checkpointed_are_cleared(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "run.log").write_text("stopped early\n")
    common = dict(temperatures_K=[300, 400], production_steps=100,
                  exchange_interval_steps=50, platform="Reference",
                  equilibration_ns=0.0, save_selection="all", minimize=False)
    resremd.run(testsystems.double_well(), output=str(out), **common)
    assert "stopped early" not in (out / "run.log").read_text()
    other = tmp_path / "mine"
    other.mkdir()
    (other / "notes.txt").write_text("keep me")
    with pytest.raises(InputError, match="did not write"):
        resremd.run(testsystems.double_well(), output=str(other), **common)


def test_mixed_save_states_on_the_command_line(tmp_path):
    assert main(["-q", "run", "--output", str(tmp_path / "x"),
                 "--save-states", "lowest", "1"]) == 2
