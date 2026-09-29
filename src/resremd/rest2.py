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


def rest2_system(system: Any, solute: np.ndarray) -> tuple[Any, dict]:
    """A copy of ``system`` whose solute interactions scale with the REST2
    global parameters (at s = 1 it is the original System).

    Refuses forces that act on the solute and cannot be scaled.
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
            "exceptions_scaled": 0}
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
