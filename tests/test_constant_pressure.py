import numpy as np
import pytest

from resremd import testsystems
import resremd
from resremd.engine import Engine, Replica
from resremd.thermo import ensemble_of


def test_each_replica_barostat_runs_at_its_own_temperature():
    prepared = testsystems.lj_box(pressure=True)
    ensemble, _ = ensemble_of(prepared.system)
    engine = Engine(system=prepared.system, ensemble=ensemble, n_replicas=2,
                    integrator="langevin_middle", timestep_fs=4.0,
                    friction_per_ps=1.0, temperature_K=100.0,
                    platform="Reference", precision="double", devices=None,
                    contexts_per_device=2, cpu_threads=None,
                    seeds=np.random.default_rng(0))
    replicas = [Replica(r, reset=(prepared.positions, prepared.box, r + 1))
                for r in range(2)]
    engine.run(replicas, [100.0, 130.0], 20, set())
    for slot, t in zip(engine.slots, [100.0, 130.0]):
        assert slot.context.getParameter(ensemble.temperature_parameter) == t
        assert slot.integrator.getTemperature()._value == pytest.approx(t)
    engine.close()


def test_volumes_are_logged_and_the_run_is_npt(tmp_path):
    prepared = testsystems.lj_box(pressure=True)
    m = resremd.run(prepared, output=str(tmp_path / "run"),
                    temperatures_K=[100.0, 115.0], production_steps=2000,
                    exchange_interval_steps=100, timestep_fs=4.0,
                    platform="Reference", random_seed=3,
                    equilibration_ns=0.004, save_selection="all")
    assert m["system"]["ensemble"]["pressure_bar"] == pytest.approx(200.0)
    volumes = np.loadtxt(tmp_path / "run/volumes.csv", delimiter=",",
                         skiprows=1)[:, 2:]
    assert volumes.shape == (20, 2)
    assert np.ptp(volumes) > 0, "the barostat moved the box"
