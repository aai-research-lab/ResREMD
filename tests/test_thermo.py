import openmm
import pytest
from openmm import unit

from resremd.errors import InputError, UnsupportedSystem
from resremd.thermo import box_volume_and_area, ensemble_of, simulated_system

import numpy as np


def periodic_system():
    s = openmm.System()
    s.addParticle(1.0)
    s.setDefaultPeriodicBoxVectors(openmm.Vec3(3, 0, 0), openmm.Vec3(0, 3, 0),
                                   openmm.Vec3(0, 0, 3))
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    nb.addParticle(0, 0.3, 0)
    s.addForce(nb)
    return s


def test_constant_volume():
    e, index = ensemble_of(periodic_system())
    assert not e.constant_pressure and index is None


@pytest.mark.parametrize("make,parameter", [
    (lambda: openmm.MonteCarloBarostat(2.0 * unit.bar, 300 * unit.kelvin),
     "MonteCarloTemperature"),
    (lambda: openmm.MonteCarloMembraneBarostat(
        2.0 * unit.bar, 10.0 * unit.bar * unit.nanometer, 300 * unit.kelvin,
        openmm.MonteCarloMembraneBarostat.XYIsotropic,
        openmm.MonteCarloMembraneBarostat.ZFree),
     "MembraneMonteCarloTemperature"),
    (lambda: openmm.MonteCarloAnisotropicBarostat(
        openmm.Vec3(2, 2, 2) * unit.bar, 300 * unit.kelvin),
     "AnisotropicMonteCarloTemperature"),
])
def test_barostat_temperature_parameter_comes_from_its_class(make, parameter):
    s = periodic_system()
    s.addForce(make())
    e, index = ensemble_of(s)
    assert e.pressure_bar == pytest.approx(2.0)
    assert e.temperature_parameter == parameter
    assert index == 1


def test_membrane_tension_is_read():
    s = periodic_system()
    s.addForce(openmm.MonteCarloMembraneBarostat(
        1.0 * unit.bar, 10.0 * unit.bar * unit.nanometer, 300 * unit.kelvin,
        openmm.MonteCarloMembraneBarostat.XYIsotropic,
        openmm.MonteCarloMembraneBarostat.ZFree))
    assert ensemble_of(s)[0].surface_tension_bar_nm == pytest.approx(10.0)


def test_unequal_anisotropic_pressure_is_refused():
    s = periodic_system()
    s.addForce(openmm.MonteCarloAnisotropicBarostat(
        openmm.Vec3(1, 1, 3) * unit.bar, 300 * unit.kelvin))
    with pytest.raises(UnsupportedSystem):
        ensemble_of(s)


def test_andersen_thermostat_is_refused():
    s = periodic_system()
    s.addForce(openmm.AndersenThermostat(300, 1.0))
    with pytest.raises(UnsupportedSystem, match="AndersenThermostat"):
        ensemble_of(s)


def test_the_ensemble_choice():
    s = periodic_system()
    kw = dict(temperature_K=300, frequency=25)
    run_system, e = simulated_system(s, ensemble=None, pressure_bar=1.0, **kw)
    assert e.pressure_bar == pytest.approx(1.0)
    assert s.getNumForces() == 1 and run_system.getNumForces() == 2
    with pytest.raises(InputError):
        simulated_system(run_system, ensemble=None, pressure_bar=2.0, **kw)
    assert not simulated_system(s, ensemble=None, pressure_bar=None,
                                **kw)[1].constant_pressure
    assert simulated_system(s, ensemble="npt", pressure_bar=None,
                            **kw)[1].pressure_bar == pytest.approx(1.0)
    nvt_system, e = simulated_system(run_system, ensemble="nvt",
                                     pressure_bar=None, **kw)
    assert not e.constant_pressure and nvt_system.getNumForces() == 1
    assert run_system.getNumForces() == 2, "the input is not changed"
    with pytest.raises(InputError, match="contradict"):
        simulated_system(s, ensemble="nvt", pressure_bar=1.0, **kw)


def test_barostats_that_move_the_box_differently_are_different_ensembles():
    from resremd.thermo import Ensemble

    a = Ensemble(1.0, 0.0, "MonteCarloMembraneBarostat", "xy mode 0, z mode 0")
    b = Ensemble(1.0, 0.0, "MonteCarloMembraneBarostat", "xy mode 0, z mode 2")
    c = Ensemble(1.0, 0.0, None, None)
    assert not a.same_as(b)
    assert a.same_as(c) and c.same_as(b), "unrecorded is not held against"


def test_box_volume_and_area():
    box = np.array([[3.0, 0, 0], [1.0, 2.0, 0], [0.5, 0.2, 4.0]])
    v, a = box_volume_and_area(box)
    assert v == pytest.approx(24.0) and a == pytest.approx(6.0)
