"""Reservoir energies with virtual sites (TIP4P-Ew) and constraints."""

import numpy as np
import openmm
import pytest
from openmm import app, unit

import resremd
from resremd.reservoir import Reservoir
from resremd.system import from_objects


@pytest.fixture(scope="module")
def water_box():
    ff = app.ForceField("tip4pew.xml")
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(ff, model="tip4pew",
                        boxSize=openmm.Vec3(1.7, 1.7, 1.7) * unit.nanometer)
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=0.7 * unit.nanometer,
                             constraints=app.HBonds, rigidWater=True)
    positions = np.asarray(modeller.positions.value_in_unit(unit.nanometer))
    box = np.eye(3) * 1.7
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001),
                         openmm.Platform.getPlatformByName("CPU"))
    ctx.setPeriodicBoxVectors(*box)
    ctx.setPositions(positions)
    openmm.LocalEnergyMinimizer.minimize(ctx, 10.0, 200)
    positions = np.asarray(ctx.getState(getPositions=True)
                           .getPositions(asNumpy=True)._value)
    return from_objects(system, modeller.topology, positions, box)


def test_energies_do_not_trust_stored_virtual_site_positions(tmp_path,
                                                             water_box):
    resremd.generate_reservoir(water_box, output=str(tmp_path / "r"),
                               temperature_K=330.0, duration_ns=0.00016,
                               frame_interval_steps=20, equilibration_ns=0.0,
                               minimize=False, platform="CPU", random_seed=1)
    res = Reservoir.open(tmp_path / "r")
    assert res.n_frames == 4
    system = water_box.system
    sites = [i for i in range(system.getNumParticles())
             if system.isVirtualSite(i)]
    assert sites
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001),
                         openmm.Platform.getPlatformByName("CPU"))

    def truth(pos, box):
        ctx.setPeriodicBoxVectors(*box)
        ctx.setPositions(pos)
        ctx.computeVirtualSites()
        return ctx.getState(getEnergy=True).getPotentialEnergy()._value

    expected = [truth(*res.frame(k)) for k in range(res.n_frames)]
    # Corrupt the stored virtual-site coordinates: the energies must not
    # depend on them, because OpenMM places the sites from the atoms.
    positions = np.load(tmp_path / "r/positions.npy", mmap_mode="r+")
    positions[:, sites, :] += 0.05
    positions.flush()
    del positions
    res = Reservoir.open(tmp_path / "r")

    from resremd.engine import Engine
    from resremd.thermo import Ensemble

    engine = Engine(system=system, ensemble=Ensemble(), n_replicas=1,
                    integrator="langevin_middle", timestep_fs=2.0,
                    friction_per_ps=1.0, temperature_K=300.0, platform="CPU",
                    precision="mixed", devices=None, contexts_per_device=1,
                    cpu_threads=None, seeds=np.random.default_rng(0))
    got = res.energies(engine.evaluator(), key_fields={"t": 1})
    engine.close()
    assert np.allclose(got["potential_kjmol"], expected, rtol=1e-5, atol=1e-3)


def test_a_run_with_virtual_sites(tmp_path, water_box):
    resremd.generate_reservoir(water_box, output=str(tmp_path / "r"),
                               temperature_K=330.0, duration_ns=0.00016,
                               frame_interval_steps=20, equilibration_ns=0.0,
                               minimize=False, platform="CPU", random_seed=1)
    m = resremd.run(water_box, output=str(tmp_path / "run"),
                    reservoir=str(tmp_path / "r"), temperatures_K=[300, 305],
                    production_steps=60, exchange_interval_steps=20,
                    trajectory_interval_steps=20, equilibration_ns=0.0,
                    minimize=False, platform="CPU", random_seed=1,
                    save_selection="all")
    assert m["status"] == "complete"
    energies = np.loadtxt(tmp_path / "run/energies.csv", delimiter=",",
                          skiprows=1)[:, 2:]
    assert np.all(np.isfinite(energies))
