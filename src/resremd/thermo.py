"""Thermodynamic states: temperatures, the pressure ensemble, and enthalpy.

Replica exchange compares reduced potentials, u = beta * h, where h is the
potential energy plus, at constant pressure, the P V work, minus the
gamma A work for a membrane held at a surface tension. Getting h wrong
does not stop a run; it changes the ensemble every replica samples. So the
ensemble is read from the System itself rather than from a setting that
might disagree with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import InputError, UnsupportedSystem

#: Boltzmann's constant in kJ/mol/K, the value OpenMM uses internally.
BOLTZ = 0.008314462618
AVOGADRO = 6.02214076e23
#: kJ/mol in one bar nm^3. Pressure (bar) times volume (nm^3) times this is
#: an energy on the same scale as OpenMM's potential energy.
BAR_NM3_TO_KJMOL = 1e5 * 1e-27 * AVOGADRO / 1e3


def beta(temperature_K: float) -> float:
    """1/kT in mol/kJ."""
    return 1.0 / (BOLTZ * float(temperature_K))


def _value(quantity: Any, unit_name: str | None = None) -> Any:
    """A plain number from an OpenMM Quantity, or the number itself."""
    from openmm import unit

    if unit.is_quantity(quantity):
        if unit_name is None:
            return quantity._value
        return quantity.value_in_unit(getattr(unit, unit_name))
    return quantity


@dataclass(frozen=True)
class Ensemble:
    """How volume is treated, and what that adds to the enthalpy.

    ``pressure_bar`` is None at constant volume. ``temperature_parameter`` is
    the Context parameter the barostat reads its temperature from. Its name
    depends on the barostat class (``MonteCarloTemperature``,
    ``MembraneMonteCarloTemperature``, ...), so it is taken from the class,
    never assumed. A replica whose barostat kept the old temperature would
    accept volume moves at the wrong temperature and still finish.
    """

    pressure_bar: float | None = None
    surface_tension_bar_nm: float = 0.0
    barostat: str | None = None
    #: How the barostat moves the box, where the class leaves a choice
    #: (a membrane barostat's xy and z modes, an anisotropic one's axes).
    #: Barostats that move the box differently sample different ensembles.
    detail: str | None = None
    temperature_parameter: str | None = None

    @property
    def constant_pressure(self) -> bool:
        return self.pressure_bar is not None

    def enthalpy(self, potential_kjmol, volume_nm3, area_nm2):
        """h = U + P V - gamma A in kJ/mol. Works on scalars and arrays."""
        if self.pressure_bar is None:
            return potential_kjmol
        work = (self.pressure_bar * volume_nm3
                - self.surface_tension_bar_nm * area_nm2)
        return potential_kjmol + BAR_NM3_TO_KJMOL * work

    def same_as(self, other: "Ensemble", *, rtol: float = 1e-9) -> bool:
        """Whether two ensembles weight volume identically.

        A barostat that one side did not record (frames imported from
        elsewhere) is not held against the other.
        """
        if self.constant_pressure != other.constant_pressure:
            return False
        if not self.constant_pressure:
            return True
        if self.barostat and other.barostat and (
                self.barostat != other.barostat or self.detail != other.detail):
            return False
        return (np.isclose(self.pressure_bar, other.pressure_bar, rtol=rtol)
                and np.isclose(self.surface_tension_bar_nm,
                               other.surface_tension_bar_nm, rtol=rtol,
                               atol=1e-12))

    def describe(self) -> str:
        if not self.constant_pressure:
            return "constant volume"
        how = self.barostat or "barostat not recorded"
        if self.detail:
            how += f", {self.detail}"
        text = f"constant pressure, {self.pressure_bar:g} bar ({how})"
        if self.surface_tension_bar_nm:
            text += f", surface tension {self.surface_tension_bar_nm:g} bar nm"
        return text

    def as_dict(self) -> dict[str, Any]:
        return {"pressure_bar": self.pressure_bar,
                "surface_tension_bar_nm": self.surface_tension_bar_nm,
                "barostat": self.barostat, "detail": self.detail}

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "Ensemble":
        d = d or {}
        return cls(pressure_bar=d.get("pressure_bar"),
                   surface_tension_bar_nm=float(
                       d.get("surface_tension_bar_nm") or 0.0),
                   barostat=d.get("barostat"), detail=d.get("detail"))


def box_volume_and_area(box: np.ndarray | None) -> tuple[float, float]:
    """Volume (nm^3) and xy area (nm^2) of a box given as three row vectors.

    The area is the one a membrane barostat couples to the surface tension:
    OpenMM keeps boxes in reduced form, with the first vector along x and the
    second in the xy plane, so it is a_x * b_y.
    """
    if box is None:
        return 0.0, 0.0
    box = np.asarray(box, dtype=float)
    return float(abs(np.linalg.det(box))), float(abs(box[0, 0] * box[1, 1]))


def _barostat_classes() -> dict[str, type]:
    import openmm

    names = ("MonteCarloBarostat", "MonteCarloMembraneBarostat",
             "MonteCarloAnisotropicBarostat", "MonteCarloFlexibleBarostat")
    return {n: getattr(openmm, n) for n in names if hasattr(openmm, n)}


def _refuse_unsupported_forces(system: Any) -> None:
    import openmm

    refused = {
        "AndersenThermostat": "it thermostats at one fixed temperature, and "
                              "each replica needs its own. Remove it; the "
                              "Langevin integrator is the thermostat here.",
        "DrudeForce": "Drude polarisable models need their own dual "
                      "thermostat, which temperature exchange does not "
                      "handle yet.",
    }
    for force in system.getForces():
        name = type(force).__name__
        if name in refused and hasattr(openmm, name):
            raise UnsupportedSystem(
                f"The System has a {name}, and {refused[name]}",
                code="resremd.system.force")


def ensemble_of(system: Any) -> tuple[Ensemble, int | None]:
    """The ensemble a System's barostat defines, and that force's index."""
    _refuse_unsupported_forces(system)
    classes = _barostat_classes()
    found: list[tuple[int, Any, str]] = []
    for index, force in enumerate(system.getForces()):
        for name, cls in classes.items():
            if isinstance(force, cls):
                found.append((index, force, name))
    if not found:
        return Ensemble(), None
    if len(found) > 1:
        raise UnsupportedSystem(
            "The System has more than one barostat: "
            + ", ".join(n for _, _, n in found) + ". Keep one.",
            code="resremd.system.barostat")
    index, force, name = found[0]
    if not system.usesPeriodicBoundaryConditions():
        raise UnsupportedSystem(
            f"The System has a {name} but is not periodic; a barostat needs "
            "a periodic box.", code="resremd.system.barostat")
    parameter = type(force).Temperature()
    tension = 0.0
    detail = None
    if name == "MonteCarloAnisotropicBarostat":
        pressures = [float(p) for p in _value(force.getDefaultPressure(), "bar")]
        scaled = [force.getScaleX(), force.getScaleY(), force.getScaleZ()]
        acting = [p for p, s in zip(pressures, scaled) if s]
        if not acting:
            raise UnsupportedSystem(
                "The anisotropic barostat scales no axis, so the volume never "
                "changes. Remove it and run at constant volume.",
                code="resremd.system.barostat")
        if max(acting) - min(acting) > 1e-9 * max(1.0, abs(max(acting))):
            raise UnsupportedSystem(
                "The anisotropic barostat applies different pressures to "
                f"different axes ({pressures} bar). The work term is then not "
                "P V, and exchanges would sample the wrong ensemble.",
                code="resremd.system.barostat")
        pressure = acting[0]
        detail = "scales " + "".join(a for a, on in zip("xyz", scaled) if on)
    else:
        pressure = float(_value(force.getDefaultPressure(), "bar"))
        if name == "MonteCarloMembraneBarostat":
            from openmm import unit

            t = force.getDefaultSurfaceTension()
            tension = float(t.value_in_unit(unit.bar * unit.nanometer)
                            if unit.is_quantity(t) else t)
            detail = f"xy mode {force.getXYMode()}, z mode {force.getZMode()}"
    return Ensemble(pressure_bar=pressure, surface_tension_bar_nm=tension,
                    barostat=name, detail=detail,
                    temperature_parameter=parameter), index


def simulated_system(system: Any, *, ensemble: str | None,
                     pressure_bar: float | None, temperature_K: float,
                     frequency: int) -> tuple[Any, Ensemble]:
    """The System to simulate, and its ensemble, from what was asked.

    ``ensemble`` None lets the System decide: its barostat is used if it has
    one, a requested pressure adds one if it does not, and otherwise the
    volume is constant. ``nvt`` removes any barostat. ``npt`` needs one: the
    System's, or an isotropic one added at ``pressure_bar`` (1 bar if not
    given). A System whose barostat holds a different pressure from the one
    requested is refused rather than silently overridden. The System passed
    in is never changed; a copy is made where anything is.
    """
    import math

    import openmm

    def copy(s):
        return openmm.XmlSerializer.deserialize(
            openmm.XmlSerializer.serialize(s))

    current, index = ensemble_of(system)
    if ensemble == "nvt":
        if pressure_bar is not None:
            raise InputError(
                "`ensemble: nvt` and `pressure_bar` contradict each other.",
                code="resremd.input.pressure")
        if index is None:
            return system, current
        system = copy(system)
        system.removeForce(index)
        return system, Ensemble()
    if index is not None:
        if pressure_bar is not None and not math.isclose(
                current.pressure_bar, pressure_bar, rel_tol=1e-9):
            raise InputError(
                f"`pressure_bar` is {pressure_bar:g} but the System's "
                f"{current.barostat} holds {current.pressure_bar:g} bar. Leave "
                "`pressure_bar` out to use the System's.",
                code="resremd.input.pressure")
        return system, current
    if pressure_bar is None and ensemble != "npt":
        return system, current
    system = copy(system)
    add_barostat(system, pressure_bar=1.0 if pressure_bar is None
                 else pressure_bar, temperature_K=temperature_K,
                 frequency=frequency)
    return system, ensemble_of(system)[0]


def add_barostat(system: Any, *, pressure_bar: float, temperature_K: float,
                 frequency: int) -> None:
    """Add an isotropic Monte Carlo barostat to a periodic System."""
    import openmm
    from openmm import unit

    if not system.usesPeriodicBoundaryConditions():
        raise InputError(
            "`pressure_bar` was given, but the System is not periodic, so "
            "there is no volume to hold at a pressure.",
            code="resremd.input.pressure")
    system.addForce(openmm.MonteCarloBarostat(
        pressure_bar * unit.bar, temperature_K * unit.kelvin, int(frequency)))
