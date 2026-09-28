"""Small systems for validation, with equilibrium distributions known exactly.

The double well is the check the test suite and the example use: a particle
whose well populations and temperature can be computed to any precision, so
a replica exchange run can be compared against the truth rather than
against another simulation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .reservoir import (ReservoirWriter, base_metadata, write_json,
                        write_topology)
from .system import from_objects, topology_digest
from .thermo import BOLTZ, Ensemble

#: An asymmetric double well along x, harmonic in y and z (kJ/mol, nm).
BARRIER = 20.0
TILT = 3.0
SPRING = 1000.0


def double_well_energy(x):
    return BARRIER * (x * x - 1.0) ** 2 + TILT * x


def double_well(start_x: float = 1.0, *, barrier: float = BARRIER,
                tilt: float = TILT):
    """One particle in the double well, as a prepared system.

    ``barrier`` and ``tilt`` change the Hamiltonian, for checks that a
    reservoir from one is refused by the other; the exact results in this
    module are for the defaults.
    """
    import openmm
    from openmm import app

    system = openmm.System()
    system.addParticle(12.0)
    force = openmm.CustomExternalForce(
        f"{barrier}*(x^2-1)^2 + {tilt}*x + 0.5*{SPRING}*(y^2+z^2)")
    force.addParticle(0, [])
    system.addForce(force)
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("DW", chain)
    topology.addAtom("X", app.element.carbon, residue)
    positions = np.array([[start_x, 0.0, 0.0]])
    return from_objects(system, topology, positions)


def grid():
    return np.linspace(-2.2, 2.2, 20001)


def left_fraction(temperature_K: float) -> float:
    """Exact P(x < 0) at a temperature."""
    x = grid()
    p = np.exp(-(double_well_energy(x) - double_well_energy(x).min())
               / (BOLTZ * temperature_K))
    return float(p[x < 0].sum() / p.sum())


def exact_x_samples(temperature_K: float, n: int, rng) -> np.ndarray:
    """Independent draws of x from the Boltzmann marginal, by inverse CDF."""
    x = grid()
    p = np.exp(-(double_well_energy(x) - double_well_energy(x).min())
               / (BOLTZ * temperature_K))
    cdf = np.cumsum(p)
    cdf /= cdf[-1]
    return np.interp(rng.random(n), cdf, x)


def write_double_well_reservoir(path: Path, *, kind: str, n_frames: int,
                                temperature_K: float | None, seed: int = 11):
    """A reservoir drawn exactly: Boltzmann at a temperature, or uniform."""
    prepared = double_well()
    rng = np.random.default_rng(seed)
    if kind == "boltzmann":
        x = exact_x_samples(temperature_K, n_frames, rng)
        sigma = np.sqrt(BOLTZ * temperature_K / SPRING)
        yz = rng.normal(0.0, sigma, size=(n_frames, 2))
    elif kind == "non_boltzmann":
        # Uniform over a region that holds all but a negligible part of the
        # density at the temperatures the tests use.
        x = rng.uniform(-2.0, 2.0, n_frames)
        yz = rng.uniform(-0.35, 0.35, size=(n_frames, 2))
    else:
        raise ValueError(kind)
    positions = np.column_stack([x, yz]).reshape(n_frames, 1, 3)
    meta = base_metadata(kind=kind, temperature_K=temperature_K,
                         ensemble=Ensemble(), n_frames=n_frames, n_atoms=1,
                         periodic=False,
                         topology_sha256=topology_digest(prepared.topology),
                         source={"method": "exact draws for testing"})
    writer = ReservoirWriter(path, n_frames=n_frames, n_atoms=1,
                             periodic=False)
    for k in range(n_frames):
        writer.write(k, positions[k], None)
    writer.flush()
    write_topology(Path(path) / "topology.pdb", prepared.topology,
                   positions[0].astype(float))
    meta["complete"] = True
    write_json(Path(path) / "reservoir.json", meta)
    return prepared


def lj_box(n_side: int = 5, spacing: float = 0.5, pressure: bool = False):
    """A small periodic Lennard-Jones fluid, for constant-pressure checks."""
    import openmm
    from openmm import app, unit

    n = n_side ** 3
    system = openmm.System()
    length = n_side * spacing
    system.setDefaultPeriodicBoxVectors(openmm.Vec3(length, 0, 0),
                                        openmm.Vec3(0, length, 0),
                                        openmm.Vec3(0, 0, length))
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(0.7)
    topology = app.Topology()
    chain = topology.addChain()
    positions = []
    for i in range(n):
        system.addParticle(39.9)
        nb.addParticle(0.0, 0.34, 0.99)
        residue = topology.addResidue("AR", chain)
        topology.addAtom("AR", app.element.argon, residue)
        positions.append([(i % n_side + 0.5) * spacing,
                          ((i // n_side) % n_side + 0.5) * spacing,
                          (i // n_side ** 2 + 0.5) * spacing])
    system.addForce(nb)
    topology.setPeriodicBoxVectors(
        [openmm.Vec3(length, 0, 0), openmm.Vec3(0, length, 0),
         openmm.Vec3(0, 0, length)] * unit.nanometer)
    if pressure:
        system.addForce(openmm.MonteCarloBarostat(200.0 * unit.bar,
                                                  100 * unit.kelvin, 10))
    box = np.eye(3) * length
    return from_objects(system, topology, np.array(positions), box)
