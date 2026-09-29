"""Small systems for validation, with equilibrium distributions known exactly.

The double well is the check the test suite and the example use: a particle
whose well populations and temperature can be computed to any precision, so
a replica exchange run can be compared against the truth rather than
against another simulation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .reservoir import write_reservoir
from .system import from_objects
from .thermo import BOLTZ

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
    elif kind == "non_boltzmann":
        # Uniform over a region that holds all but a negligible part of the
        # density at the temperatures the tests use.
        x = rng.uniform(-2.0, 2.0, n_frames)
    else:
        raise ValueError(kind)
    positions = double_well_frames(x, temperature_K, rng,
                                   uniform_yz=kind == "non_boltzmann")
    write_reservoir(path, topology=prepared.topology, positions=positions,
                    kind=kind, temperature_K=temperature_K,
                    source={"method": "exact draws for testing",
                            "seed": seed})
    return prepared


def double_well_frames(x, temperature_K: float | None, rng, *,
                       uniform_yz: bool = False) -> np.ndarray:
    """Frames (n, 1, 3) with the given x and y, z drawn to match.

    y and z are harmonic, so at a temperature they are exactly Gaussian;
    ``uniform_yz`` instead spreads them evenly, for a uniform reservoir.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if uniform_yz:
        yz = rng.uniform(-0.35, 0.35, size=(n, 2))
    else:
        sigma = np.sqrt(BOLTZ * temperature_K / SPRING)
        yz = rng.normal(0.0, sigma, size=(n, 2))
    return np.column_stack([x, yz]).reshape(n, 1, 3)


def _place(a, b, c, bond, angle, dihedral):
    """Position of d with |cd| = bond, angle bcd and dihedral abcd (deg)."""
    angle, dihedral = np.radians(angle), np.radians(dihedral)
    bc = (c - b) / np.linalg.norm(c - b)
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n)
    m = np.cross(n, bc)
    d = np.array([-bond * np.cos(angle),
                  bond * np.sin(angle) * np.cos(dihedral),
                  bond * np.sin(angle) * np.sin(dihedral)])
    return c + d[0] * bc + d[1] * m + d[2] * n


def alanine_dipeptide(phi: float = -80.0, psi: float = 150.0):
    """Alanine dipeptide (ACE-ALA-NME) from ideal geometry.

    Heavy atoms are placed from standard bond lengths and angles at the
    given backbone dihedrals (degrees), hydrogens are added by OpenMM with
    the Amber14 templates, and the structure is energy-minimised in vacuum.
    Returns the topology and positions in nm.
    """
    return _capped("ALA", phi=phi, psi=psi, omega=180.0)


def proline_dipeptide(omega: float = 180.0, psi: float = 150.0):
    """Ac-Pro-NMe (ACE-PRO-NME), the model for prolyl cis/trans isomerism.

    ``omega`` is the ACE-PRO peptide bond (CH3-C-N-CA): 180 trans, 0 cis.
    Its barrier, about 80 kJ/mol, is crossed on the order of minutes at room
    temperature, so heating alone does not sample it.
    """
    return _capped("PRO", phi=-65.0, psi=psi, omega=omega)


def _capped(resname: str, *, phi: float, psi: float, omega: float):
    import openmm
    from openmm import app, unit

    r = resname
    x: dict[str, np.ndarray] = {}
    x["ACE:CH3"] = np.zeros(3)
    x["ACE:C"] = np.array([0.152, 0.0, 0.0])
    x[f"{r}:N"] = _place(np.array([0.0, 0.1, 0.0]), x["ACE:CH3"],
                         x["ACE:C"], 0.133, 116.0, 180.0)
    x[f"{r}:CA"] = _place(x["ACE:CH3"], x["ACE:C"], x[f"{r}:N"],
                          0.146, 122.0, omega)
    x[f"{r}:C"] = _place(x["ACE:C"], x[f"{r}:N"], x[f"{r}:CA"],
                         0.152, 111.0, phi)
    x["NME:N"] = _place(x[f"{r}:N"], x[f"{r}:CA"], x[f"{r}:C"],
                        0.133, 116.0, psi)
    x["NME:C"] = _place(x[f"{r}:CA"], x[f"{r}:C"], x["NME:N"],
                        0.146, 122.0, 180.0)
    # Carbonyl oxygens lie opposite the atom before them across the C=O
    # carbon, so they sit at omega + 180 from the next C-alpha.
    x["ACE:O"] = _place(x[f"{r}:CA"], x[f"{r}:N"], x["ACE:C"], 0.123, 123.0,
                        omega + 180.0)
    x[f"{r}:O"] = _place(x["NME:C"], x["NME:N"], x[f"{r}:C"], 0.123, 123.0,
                         0.0)
    # The CB position that makes CA an L centre: seen with the hydrogen
    # towards the viewer, CO -> R -> N runs clockwise.
    ca = x[f"{r}:CA"]
    for sign in (1.0, -1.0):
        cb = _place(x[f"{r}:C"], x[f"{r}:N"], ca, 0.153, 110.0, sign * 122.0)
        units = [(p - ca) / np.linalg.norm(p - ca)
                 for p in (x[f"{r}:N"], x[f"{r}:C"], cb)]
        if np.dot(np.cross(x[f"{r}:C"] - ca, cb - ca), -sum(units)) < 0:
            x[f"{r}:CB"] = cb
            break
    side = ("CB",)
    if r == "PRO":
        # Close the pyrrolidine ring back onto N; minimisation settles it.
        x["PRO:CG"] = _place(x["PRO:N"], ca, x["PRO:CB"], 0.150, 104.5, 30.0)
        x["PRO:CD"] = _place(ca, x["PRO:CB"], x["PRO:CG"], 0.150, 105.5,
                             -35.0)
        side = ("CB", "CG", "CD")
    top = app.Topology()
    chain = top.addChain()
    positions = []
    for res, names in (("ACE", ("CH3", "C", "O")),
                       (r, ("N", "CA", "C", "O") + side),
                       ("NME", ("N", "C"))):
        residue = top.addResidue(res, chain)
        for name in names:
            top.addAtom(name, app.element.Element.getBySymbol(name[0]),
                        residue)
            positions.append(x[f"{res}:{name}"])
    top.createStandardBonds()
    forcefield = app.ForceField("amber14-all.xml")
    modeller = app.Modeller(top, np.array(positions) * unit.nanometer)
    modeller.addHydrogens(forcefield)
    system = forcefield.createSystem(modeller.topology,
                                     nonbondedMethod=app.NoCutoff)
    context = openmm.Context(system, openmm.VerletIntegrator(0.001),
                             openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(modeller.positions)
    openmm.LocalEnergyMinimizer.minimize(context)
    pos = context.getState(getPositions=True).getPositions(asNumpy=True)
    return modeller.topology, np.asarray(pos.value_in_unit(unit.nanometer))


def omega_atoms(topology) -> list[int]:
    """Atom indices of the first peptide bond's omega: CH3/CA, C, N, CA."""
    res = list(topology.residues())
    first, second = res[0], res[1]

    def idx(residue, name):
        for a in residue.atoms():
            if a.name == name:
                return a.index
        raise ValueError(f"No atom {name} in {residue.name}.")

    lead = "CH3" if first.name == "ACE" else "CA"
    return [idx(first, lead), idx(first, "C"), idx(second, "N"),
            idx(second, "CA")]


def prepare_peptide(topology, positions, *, solvent: str = "implicit",
                    padding_nm: float = 1.2, npt_ns: float = 0.5,
                    temperature_K: float = 300.0, platform: str = "auto",
                    seed: int | None = None):
    """Amber14 parameters and solvent for a peptide, ready to simulate.

    ``implicit`` is GBn2 with no cutoff. ``explicit`` is TIP3P-FB with PME
    (0.9 nm cutoff), neutralised, and the box then equilibrated for
    ``npt_ns`` at ``temperature_K`` and 1 bar so that constant-volume runs
    start at the right density. Bonds to hydrogen are constrained. Returns
    (system, topology, positions, box) with no barostat in the System, and
    box None for implicit solvent.
    """
    import openmm
    from openmm import app, unit

    from .system import create_context

    if solvent == "implicit":
        forcefield = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    elif solvent == "explicit":
        forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    else:
        raise ValueError(f"solvent is implicit or explicit, not {solvent!r}")
    modeller = app.Modeller(topology, np.asarray(positions) * unit.nanometer)
    modeller.addHydrogens(forcefield)
    if solvent == "implicit":
        system = forcefield.createSystem(modeller.topology,
                                         nonbondedMethod=app.NoCutoff,
                                         constraints=app.HBonds)
    else:
        modeller.addSolvent(forcefield, padding=padding_nm * unit.nanometer,
                            neutralize=True)
        system = forcefield.createSystem(modeller.topology,
                                         nonbondedMethod=app.PME,
                                         nonbondedCutoff=0.9 * unit.nanometer,
                                         constraints=app.HBonds,
                                         rigidWater=True)
    context, _ = create_context(system, openmm.VerletIntegrator(0.001),
                                platform=platform, precision="mixed",
                                device=None, cpu_threads=None)
    context.setPositions(modeller.positions)
    openmm.LocalEnergyMinimizer.minimize(context)
    state = context.getState(getPositions=True)
    box = None
    if solvent == "explicit" and npt_ns > 0:
        npt = openmm.XmlSerializer.deserialize(
            openmm.XmlSerializer.serialize(system))
        npt.addForce(openmm.MonteCarloBarostat(1.0 * unit.bar,
                                               temperature_K * unit.kelvin))
        dynamics = openmm.LangevinMiddleIntegrator(
            temperature_K * unit.kelvin, 1.0 / unit.picosecond,
            2.0 * unit.femtosecond)
        if seed is not None:
            dynamics.setRandomNumberSeed(int(seed))
        ctx, _ = create_context(npt, dynamics, platform=platform,
                                precision="mixed", device=None,
                                cpu_threads=None)
        ctx.setState(state)
        ctx.setVelocitiesToTemperature(temperature_K * unit.kelvin,
                                       int(seed or 1))
        dynamics.step(int(round(npt_ns * 1e6 / 2.0)))
        state = ctx.getState(getPositions=True)
    if solvent == "explicit":
        box = np.asarray(state.getPeriodicBoxVectors(asNumpy=True)
                         .value_in_unit(unit.nanometer))
    positions = np.asarray(state.getPositions(asNumpy=True)
                           .value_in_unit(unit.nanometer))
    return system, modeller.topology, positions, box


#: A four-atom torsion with a cis/trans barrier far too high to cross by
#: heating: V(phi) = (A/2)(1 - cos 2 phi) + (B/2)(1 + cos phi), with cis at
#: phi = 0 raised by B above trans at phi = pi.
TORSION_BARRIER = 80.0
TORSION_TILT = 6.0
_BOND_NM, _BOND_K = 0.15, 5.0e4
_ANGLE_RAD, _ANGLE_K = 1.9, 400.0


def torsion_energy(phi):
    return (TORSION_BARRIER / 2) * (1 - np.cos(2 * phi)) \
        + (TORSION_TILT / 2) * (1 + np.cos(phi))


def torsion_model(cis: bool = False):
    """Four atoms whose dihedral has a prolyl-like 80 kJ/mol barrier.

    The dihedral's equilibrium distribution is exactly proportional to
    exp(-V(phi)/kT): in internal coordinates its Jacobian is constant, and
    nothing else in the model depends on it. Bond and angle terms are
    harmonic, with no nonbonded interactions.
    """
    import openmm
    from openmm import app

    system = openmm.System()
    for _ in range(4):
        system.addParticle(12.0)
    bonds = openmm.HarmonicBondForce()
    for i in range(3):
        bonds.addBond(i, i + 1, _BOND_NM, _BOND_K)
    angles = openmm.HarmonicAngleForce()
    for i in range(2):
        angles.addAngle(i, i + 1, i + 2, _ANGLE_RAD, _ANGLE_K)
    torsion = openmm.CustomTorsionForce(
        f"{TORSION_BARRIER / 2}*(1-cos(2*theta))"
        f"+{TORSION_TILT / 2}*(1+cos(theta))")
    torsion.addTorsion(0, 1, 2, 3, [])
    for f in (bonds, angles, torsion):
        system.addForce(f)
    topology = app.Topology()
    residue = topology.addResidue("TOR", topology.addChain())
    atoms = [topology.addAtom(f"C{i}", app.element.carbon, residue)
             for i in range(4)]
    for i in range(3):
        topology.addBond(atoms[i], atoms[i + 1])
    positions = np.zeros((4, 3))
    positions[1] = [0.0, 0.0, 0.0]
    positions[0] = [_BOND_NM * np.sin(_ANGLE_RAD), _BOND_NM *
                    np.cos(_ANGLE_RAD), 0.0]
    positions[2] = [0.0, _BOND_NM, 0.0]
    positions[3] = _place(positions[0], positions[1], positions[2], _BOND_NM,
                          np.degrees(_ANGLE_RAD), 0.0 if cis else 180.0)
    return from_objects(system, topology, positions)


def torsion_grid() -> np.ndarray:
    return np.linspace(-np.pi, np.pi, 20001)


def cis_fraction(temperature_K: float) -> float:
    """Exact probability that |phi| < pi/2."""
    phi = torsion_grid()
    p = np.exp(-(torsion_energy(phi) - torsion_energy(phi).min())
               / (BOLTZ * temperature_K))
    return float(p[np.abs(phi) < np.pi / 2].sum() / p.sum())


def torsion_bias(k: float = 70.0) -> list[dict]:
    """A bias that lowers the model's barrier by k: -k sin(phi)^2."""
    return [{"atoms": [0, 1, 2, 3], "energy": "-k*sin(theta)^2",
             "parameters": {"k": float(k)}}]


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
