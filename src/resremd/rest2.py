"""Replica exchange with solute tempering, REST2 (Wang, Friesner and Berne,
J. Phys. Chem. B 2011, 115, 9431).

Every replica runs at the same temperature T0. State k scales the solute's
interactions instead: with s = sqrt(T0 / T_k), where T_k is the state's
effective temperature,

    solute-solute nonbonded and solute torsions    by s^2 (= T0 / T_k)
    solute-solvent nonbonded, and torsions only
    partly in the solute                            by s
    everything else                                 not at all

so the solute feels temperature T_k while the solvent stays at T0. Charges
of solute atoms are scaled by s and their Lennard-Jones well depths by s^2,
through OpenMM parameter offsets on global parameters; 1-4 pairs within the
solute by s^2, and those reaching outside it by s. Bonds and angles are not
scaled.

OpenMM does not apply parameter offsets to the long-range dispersion
correction. At constant volume that does not matter: the correction is a
constant of each state, the same for every configuration, so it cancels
from every exchange and every reweighting, and it is left as OpenMM
computes it. At constant pressure it acts on the barostat, and it is moved
out of the NonbondedForce into a CustomVolumeForce that scales it as the
pairs it stands for: the solute's own pairs by s^2, its pairs with the
solvent by s. Its coefficients come from OpenMM's own correction for the
solute's well depths scaled to s = 1, 0.5 and 0, so at s = 1 it is the
original term. OpenMM evaluates a CustomVolumeForce on the host, which
costs a GPU a round trip every step; equilibrating at constant pressure and
running REST2 at constant volume avoids it. With an OpenMM older than 8.3,
which has no CustomVolumeForce, the correction stays unscaled at constant
pressure too (exact for that Hamiltonian, a small departure from Wang et
al.'s).

The potential energy of any configuration is then exactly quadratic in s,

    U(s) = A s^2 + B s + C,

and three evaluations give A, B and C. With them a configuration's energy
at every state is known, for exchanges between neighbours, for a reservoir
at any scale, and for MBAR across all states.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import numpy as np

from .errors import InputError

logger = logging.getLogger("resremd")

#: Global parameters, all set from s by :func:`parameters`.
Q, L, LAMBDA, S = "rest2_q", "rest2_l", "rest2_lambda", "rest2_s"

#: Forces that act on the solute and cannot be scaled here. Bonded terms
#: that are meant to stay unscaled are fine.
_REFUSED = ("CustomNonbondedForce", "GBSAOBCForce", "CustomGBForce",
            "CMAPTorsionForce", "RBTorsionForce", "AmoebaMultipoleForce",
            "AmoebaVdwForce", "DrudeForce")


def parameters(s: float) -> dict[str, float]:
    """The global parameter values for scale s."""
    return {Q: s - 1.0, L: s * s - 1.0, LAMBDA: s * s, S: s}


def scale_of(temperature_K: float, effective_K: float) -> float:
    """s for a state of effective temperature ``effective_K`` run at
    ``temperature_K``."""
    return float(np.sqrt(temperature_K / effective_K))


def solute_digest(atoms: np.ndarray) -> str:
    return hashlib.sha256(json.dumps(
        [int(a) for a in atoms]).encode()).hexdigest()


def rest2_system(system: Any, solute: np.ndarray, *,
                 constant_pressure: bool = False) -> tuple[Any, dict]:
    """A copy of ``system`` whose solute interactions scale with the REST2
    global parameters (at s = 1 it is the original System).

    With ``constant_pressure`` the dispersion correction is scaled too (see
    the module's description). Refuses forces that act on the solute and
    cannot be scaled.
    """
    import openmm

    solute = np.asarray(sorted({int(a) for a in solute}), dtype=int)
    if solute.size == 0:
        raise InputError("REST2 needs at least one solute atom.",
                         code="resremd.input.rest2")
    hot = np.zeros(system.getNumParticles(), dtype=bool)
    hot[solute] = True
    system = openmm.XmlSerializer.deserialize(
        openmm.XmlSerializer.serialize(system))
    info = {"solute_atoms": int(solute.size), "nonbonded": False,
            "torsions_scaled": 0, "torsions_partly_scaled": 0,
            "exceptions_scaled": 0, "dispersion_correction": "none"}
    forces = list(system.getForces())
    for f in forces:
        name = f.__class__.__name__
        if name in _REFUSED and _touches(f, hot):
            raise InputError(
                f"REST2 cannot scale the solute's terms in {name}. It needs "
                "a NonbondedForce for nonbonded interactions (explicit "
                "solvent or vacuum; implicit-solvent GB forces are not "
                "supported) and periodic or custom torsions.",
                code="resremd.input.rest2")
    for f in forces:
        if isinstance(f, openmm.NonbondedForce):
            _scale_dispersion_correction(system, f, hot, info,
                                         constant_pressure)
            _scale_nonbonded(f, hot, info)
    for index in reversed(range(system.getNumForces())):
        f = system.getForce(index)
        if isinstance(f, openmm.PeriodicTorsionForce):
            _scale_periodic_torsions(system, f, hot, info)
        elif isinstance(f, openmm.CustomTorsionForce) and \
                not f.getName().startswith("REST2"):
            _scale_custom_torsions(system, index, f, hot, info)
    return system, info


def _touches(force: Any, hot: np.ndarray) -> bool:
    """Whether a force acts on any solute atom (conservatively: yes when
    its atoms cannot be listed)."""
    if hasattr(force, "getNumParticles") and \
            force.getNumParticles() == len(hot):
        return bool(hot.any())
    if force.__class__.__name__ == "CMAPTorsionForce":
        return any(any(hot[int(a)] for a in force.getTorsionParameters(i)[1:9])
                   for i in range(force.getNumTorsions()))
    if hasattr(force, "getNumTorsions"):
        return any(any(hot[int(a)] for a in force.getTorsionParameters(i)[:4])
                   for i in range(force.getNumTorsions()))
    return True


def _scale_nonbonded(f: Any, hot: np.ndarray, info: dict) -> None:
    for name, default in ((Q, 0.0), (L, 0.0)):
        f.addGlobalParameter(name, default)
    for i in np.flatnonzero(hot):
        q, _sigma, eps = f.getParticleParameters(int(i))
        q, eps = q._value, eps._value
        if q != 0.0:
            f.addParticleParameterOffset(Q, int(i), q, 0.0, 0.0)
        if eps != 0.0:
            f.addParticleParameterOffset(L, int(i), 0.0, 0.0, eps)
    for k in range(f.getNumExceptions()):
        i, j, qq, _sigma, eps = f.getExceptionParameters(k)
        qq, eps = qq._value, eps._value
        if qq == 0.0 and eps == 0.0:
            continue
        both, one = hot[i] and hot[j], hot[i] != hot[j]
        if both:
            f.addExceptionParameterOffset(L, k, qq, 0.0, eps)
        elif one:
            f.addExceptionParameterOffset(Q, k, qq, 0.0, eps)
        else:
            continue
        info["exceptions_scaled"] += 1
    info["nonbonded"] = True


#: Nonbonded methods with a dispersion correction (LJPME has none).
_CORRECTED = ("CutoffPeriodic", "Ewald", "PME")


def dispersion_coefficient(force: Any, epsilon_scale: np.ndarray) -> float:
    """OpenMM's dispersion correction of ``force`` times the box volume
    (kJ/mol nm^3), with each particle's well depth multiplied by
    ``epsilon_scale``.

    Found from a copy of the particles spaced wider than the cutoff with no
    charges, so no pair is within the cutoff and the energy is the
    correction alone.
    """
    import openmm

    n = force.getNumParticles()
    cutoff = force.getCutoffDistance()._value
    probe = openmm.NonbondedForce()
    probe.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    probe.setCutoffDistance(cutoff)
    probe.setUseSwitchingFunction(force.getUseSwitchingFunction())
    probe.setSwitchingDistance(force.getSwitchingDistance())
    probe.setUseDispersionCorrection(True)
    system = openmm.System()
    for i in range(n):
        _q, sigma, eps = force.getParticleParameters(i)
        system.addParticle(1.0)
        probe.addParticle(0.0, sigma, eps * float(epsilon_scale[i]))
    system.addForce(probe)
    side = int(np.ceil(n ** (1.0 / 3.0)))
    spacing = 1.1 * cutoff
    length = max(side * spacing, 2.2 * cutoff)
    system.setDefaultPeriodicBoxVectors(openmm.Vec3(length, 0, 0),
                                        openmm.Vec3(0, length, 0),
                                        openmm.Vec3(0, 0, length))
    grid = np.indices((side, side, side)).reshape(3, -1).T[:n] * spacing
    context = openmm.Context(system, openmm.VerletIntegrator(0.001),
                             openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(grid)
    energy = context.getState(getEnergy=True).getPotentialEnergy()._value
    return energy * length ** 3


def _scale_dispersion_correction(system: Any, f: Any, hot: np.ndarray,
                                 info: dict, constant_pressure: bool) -> None:
    import openmm

    method = f.getNonbondedMethod()
    if not f.getUseDispersionCorrection() or not any(
            method == getattr(openmm.NonbondedForce, m) for m in _CORRECTED):
        return
    if not any(f.getParticleParameters(int(i))[2]._value != 0.0
               for i in np.flatnonzero(hot)):
        info["dispersion_correction"] = "unchanged"
        return
    if not constant_pressure:
        # A constant of each state at fixed volume: it cancels everywhere.
        info["dispersion_correction"] = "constant"
        return
    if not hasattr(openmm, "CustomVolumeForce"):
        info["dispersion_correction"] = "unscaled"
        return
    # E(s) = (c2 s^2 + c1 s + c0) / V, through three scales.
    values = [dispersion_coefficient(f, np.where(hot, s * s, 1.0))
              for s in (1.0, 0.5, 0.0)]
    c2, c1, c0 = fit([1.0, 0.5, 0.0], values)
    volume = openmm.CustomVolumeForce(
        f"({c2!r}*{LAMBDA} + {c1!r}*{S} + {c0!r})/v")
    volume.setName("REST2 dispersion correction")
    volume.addGlobalParameter(LAMBDA, 1.0)
    volume.addGlobalParameter(S, 1.0)
    volume.setForceGroup(f.getForceGroup())
    system.addForce(volume)
    f.setUseDispersionCorrection(False)
    info["dispersion_correction"] = "scaled"


def _new_torsion_force(expression: str, parameter: str, names: list[str],
                       globals_: list[tuple[str, float]], group: int,
                       periodic: bool) -> Any:
    import openmm

    f = openmm.CustomTorsionForce(f"{parameter}*({expression})")
    f.setName(f"REST2 {parameter}")
    f.addGlobalParameter(parameter, 1.0)
    for g, v in globals_:
        f.addGlobalParameter(g, v)
    for n in names:
        f.addPerTorsionParameter(n)
    f.setForceGroup(group)
    f.setUsesPeriodicBoundaryConditions(periodic)
    return f


def _scale_periodic_torsions(system: Any, f: Any, hot: np.ndarray,
                             info: dict) -> None:
    full, part = [], []
    for k in range(f.getNumTorsions()):
        a, b, c, d, n, phase, kk = f.getTorsionParameters(k)
        inside = sum(bool(hot[x]) for x in (a, b, c, d))
        if inside == 0:
            continue
        (full if inside == 4 else part).append(
            (a, b, c, d, [float(n), phase._value, kk._value]))
        f.setTorsionParameters(k, a, b, c, d, n, phase, 0.0)
    expr = "k*(1+cos(n*theta-theta0))"
    for parameter, items in ((LAMBDA, full), (S, part)):
        if not items:
            continue
        new = _new_torsion_force(expr, parameter, ["n", "theta0", "k"], [],
                                 f.getForceGroup(),
                                 f.usesPeriodicBoundaryConditions())
        for a, b, c, d, p in items:
            new.addTorsion(a, b, c, d, p)
        system.addForce(new)
    info["torsions_scaled"] += len(full)
    info["torsions_partly_scaled"] += len(part)


def _scale_custom_torsions(system: Any, index: int, f: Any,
                           hot: np.ndarray, info: dict) -> None:
    if f.getNumEnergyParameterDerivatives():
        if any(any(hot[x] for x in f.getTorsionParameters(k)[:4])
               for k in range(f.getNumTorsions())):
            raise InputError(
                "REST2 cannot scale a custom torsion force with energy "
                "parameter derivatives on the solute.",
                code="resremd.input.rest2")
        return
    names = [f.getPerTorsionParameterName(i)
             for i in range(f.getNumPerTorsionParameters())]
    globals_ = [(f.getGlobalParameterName(i),
                 f.getGlobalParameterDefaultValue(i))
                for i in range(f.getNumGlobalParameters())]
    full, part, keep = [], [], []
    for k in range(f.getNumTorsions()):
        a, b, c, d, p = f.getTorsionParameters(k)
        inside = sum(bool(hot[x]) for x in (a, b, c, d))
        target = keep if inside == 0 else full if inside == 4 else part
        target.append((a, b, c, d, list(p)))
    if not full and not part:
        return
    expr = f.getEnergyFunction()
    for parameter, items in ((LAMBDA, full), (S, part), (None, keep)):
        if not items:
            continue
        if parameter is None:
            import openmm

            new = openmm.CustomTorsionForce(expr)
            new.setName(f.getName())
            for g, v in globals_:
                new.addGlobalParameter(g, v)
            for n in names:
                new.addPerTorsionParameter(n)
            new.setForceGroup(f.getForceGroup())
            new.setUsesPeriodicBoundaryConditions(
                f.usesPeriodicBoundaryConditions())
        else:
            new = _new_torsion_force(expr, parameter, names, globals_,
                                     f.getForceGroup(),
                                     f.usesPeriodicBoundaryConditions())
        for a, b, c, d, p in items:
            new.addTorsion(a, b, c, d, p)
        system.addForce(new)
    # The replacements were appended after it, so its index still holds.
    system.removeForce(index)
    info["torsions_scaled"] += len(full)
    info["torsions_partly_scaled"] += len(part)


def set_scale(context: Any, s: float) -> None:
    for name, value in parameters(s).items():
        try:
            context.setParameter(name, value)
        except Exception:  # a parameter no force in this System uses
            pass


#: The three scales at which a configuration's energy is evaluated to
#: find A, B and C: its own and two of these.
_PROBES = (1.0, 0.0, 0.5)


def probes(own: float) -> list[float]:
    """Two scales besides ``own`` to evaluate at, well separated from it."""
    return [p for p in _PROBES if abs(p - own) > 0.1][:2]


def fit(scales: list[float], energies: list[float]) -> tuple[float, float,
                                                             float]:
    """A, B, C with U(s) = A s^2 + B s + C through three points."""
    m = np.array([[s * s, s, 1.0] for s in scales])
    a, b, c = np.linalg.solve(m, np.asarray(energies, dtype=float))
    return float(a), float(b), float(c)


def energy(terms: tuple[float, float, float] | np.ndarray, s: float
           ) -> float | np.ndarray:
    t = np.asarray(terms, dtype=float)
    return t[..., 0] * s * s + t[..., 1] * s + t[..., 2]
