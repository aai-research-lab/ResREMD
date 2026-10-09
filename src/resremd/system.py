"""The molecular system: loading it, identifying it, and running it.

A prepared system is three files, the layout FastMDXplora's setup phase
writes: ``system.xml`` (an OpenMM System), ``state.xml`` (a State with the
positions and box) and ``topology.pdb``.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import InputError, ResRemdError

logger = logging.getLogger("resremd")

#: Residue names read as water. The list MDTraj and the common force-field
#: libraries use; heavy water is included.
WATER_RESIDUES = frozenset({
    "HOH", "WAT", "SOL", "H2O", "DOD", "TIP", "TIP3", "TIP3P", "TIP4",
    "TIP4P", "TIP5", "TIP5P", "TP3", "T3P", "T4P", "T4E", "T5P", "SPC",
    "SPCE", "OPC", "OPC3",
})


@dataclass
class Prepared:
    """A System with its topology and a starting configuration (nm)."""

    system: Any
    topology: Any
    positions: np.ndarray
    box: np.ndarray | None
    source: str = "objects"

    @property
    def n_atoms(self) -> int:
        return self.system.getNumParticles()

    @property
    def periodic(self) -> bool:
        return bool(self.system.usesPeriodicBoundaryConditions())


def load_prepared(directory: str | Path) -> Prepared:
    """Read system.xml, state.xml and topology.pdb from one directory."""
    import openmm
    from openmm import app, unit

    directory = Path(directory)
    files = {n: directory / n for n in ("system.xml", "state.xml",
                                        "topology.pdb")}
    missing = [n for n, p in files.items() if not p.is_file()]
    if missing:
        raise InputError(
            f"{directory} is not a prepared system: {', '.join(missing)} "
            "missing. It needs system.xml, state.xml and topology.pdb.",
            code="resremd.input.prepared")
    def read(name: str, how: Any) -> Any:
        try:
            return how(files[name])
        except Exception as exc:  # OpenMM's errors are no subclass of one
            raise InputError(
                f"{files[name]} could not be read; prepare the system "
                f"again. {type(exc).__name__}: {exc}",
                code="resremd.input.prepared") from exc

    system = read("system.xml", lambda f: openmm.XmlSerializer.deserialize(
        f.read_text()))
    state = read("state.xml", lambda f: openmm.XmlSerializer.deserialize(
        f.read_text()))
    topology = read("topology.pdb", lambda f: app.PDBFile(str(f)).topology)
    for name, value, kind in (("system.xml", system, openmm.System),
                              ("state.xml", state, openmm.State)):
        if not isinstance(value, kind):
            raise InputError(
                f"{files[name]} holds a {type(value).__name__}, not a "
                f"{kind.__name__}. Prepare the system again.",
                code="resremd.input.prepared")
    positions = read("state.xml", lambda _: np.asarray(
        state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
        dtype=float))
    box = None
    if system.usesPeriodicBoundaryConditions():
        box = read("state.xml", lambda _: np.asarray(
            state.getPeriodicBoxVectors(asNumpy=True)
            .value_in_unit(unit.nanometer), dtype=float))
    return checked(Prepared(system, topology, positions, box,
                            source=str(directory.resolve())))


def write_prepared(directory: str | Path, system: Any, topology: Any,
                   positions: np.ndarray, box: np.ndarray | None = None,
                   velocities: np.ndarray | None = None) -> Path:
    """Write system.xml, state.xml and topology.pdb: the inverse of
    :func:`load_prepared`, in the layout FastMDXplora uses."""
    import openmm
    from openmm import Vec3, app

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(system, integrator,
                             openmm.Platform.getPlatformByName("Reference"))
    if box is not None:
        context.setPeriodicBoxVectors(*(Vec3(*map(float, r)) for r in box))
    context.setPositions(np.asarray(positions, dtype=float))
    if velocities is not None:
        context.setVelocities(np.asarray(velocities, dtype=float))
    state = context.getState(getPositions=True, getVelocities=True)
    (directory / "system.xml").write_text(
        openmm.XmlSerializer.serialize(system))
    (directory / "state.xml").write_text(openmm.XmlSerializer.serialize(state))
    top = subset_topology(topology, np.arange(topology.getNumAtoms()), box)
    with open(directory / "topology.pdb", "w") as fh:
        # PDBFile takes plain numbers as angstroms.
        app.PDBFile.writeFile(top, np.asarray(positions) * 10.0, fh,
                              keepIds=True)
    return directory


def from_objects(system: Any, topology: Any, positions: Any,
                 box: Any = None) -> Prepared:
    """Wrap objects already in memory. Positions and box in nm or Quantity."""
    from openmm import unit

    def number(c):
        return float(c.value_in_unit(unit.nanometer)
                     if unit.is_quantity(c) else c)

    def nm(value, what):
        if value is None:
            return None
        try:
            if unit.is_quantity(value):
                value = value.value_in_unit(unit.nanometer)
            # Rows, or their numbers, may be quantities themselves: OpenMM
            # gives box vectors as a list of three.
            return np.array([[number(c) for c in (
                row.value_in_unit(unit.nanometer) if unit.is_quantity(row)
                else row)] for row in value])
        except (TypeError, ValueError) as exc:
            raise InputError(
                f"The {what} are rows of three numbers, in nm or as a "
                f"Quantity of length ({type(exc).__name__}: {exc}).",
                code="resremd.input.type") from None

    if positions is None:
        raise InputError("Positions are needed with a System.",
                         code="resremd.input.type")
    box = nm(box, "box vectors")
    if box is None and system.usesPeriodicBoundaryConditions():
        box = nm(system.getDefaultPeriodicBoxVectors(), "box vectors")
    return checked(Prepared(system, topology, nm(positions, "positions"),
                            box))


def checked(prepared: Prepared) -> Prepared:
    n = prepared.system.getNumParticles()
    if prepared.positions.shape != (n, 3):
        raise InputError(
            f"The System has {n} particles and the positions have shape "
            f"{prepared.positions.shape}.", code="resremd.input.prepared")
    if prepared.topology.getNumAtoms() != n:
        raise InputError(
            f"The System has {n} particles and the topology "
            f"{prepared.topology.getNumAtoms()} atoms.",
            code="resremd.input.prepared")
    if not np.all(np.isfinite(prepared.positions)):
        raise InputError("The starting positions are not all finite.",
                         code="resremd.input.prepared")
    return prepared


def topology_digest(topology: Any) -> str:
    """A hash of which atoms there are, in which order.

    Two topologies with the same digest name the same atoms in the same
    order, which is what matters for injecting one system's coordinates into
    another. Positions and bonds do not enter it.
    """
    h = hashlib.sha256()
    for atom in topology.atoms():
        residue = atom.residue
        element = atom.element.symbol if atom.element is not None else "-"
        h.update(f"{residue.chain.index}|{residue.name}|{residue.index}|"
                 f"{atom.name}|{element}\n".encode())
    return h.hexdigest()


def system_digest(system: Any) -> str:
    """A hash of the serialised System: its forces and every parameter.

    The OpenMM version the XML records is left out, so the same System
    serialised by another OpenMM build hashes the same.
    """
    import re

    import openmm

    xml = openmm.XmlSerializer.serialize(system)
    xml = re.sub(r'\sopenmmVersion="[^"]*"', "", xml, count=1)
    return hashlib.sha256(xml.encode()).hexdigest()


def make_integrator(name: str, temperature_K: float, friction_per_ps: float,
                    timestep_fs: float, seed: int | None):
    import openmm
    from openmm import unit

    cls = {"langevin_middle": openmm.LangevinMiddleIntegrator,
           "langevin": openmm.LangevinIntegrator}.get(name)
    if cls is None:
        raise InputError(f"Unknown integrator {name!r}.",
                         code="resremd.input.choice")
    integrator = cls(temperature_K * unit.kelvin,
                     friction_per_ps / unit.picosecond,
                     timestep_fs * unit.femtosecond)
    if seed is not None:
        integrator.setRandomNumberSeed(int(seed))
    return integrator


_AUTO_ORDER = ("CUDA", "HIP", "OpenCL", "CPU")


def _properties(platform: str, precision: str, device: int | None,
                cpu_threads: int | None) -> dict[str, str]:
    if platform in ("CUDA", "HIP", "OpenCL"):
        props = {"Precision": precision}
        if device is not None:
            props["DeviceIndex"] = str(int(device))
        return props
    if platform == "CPU" and cpu_threads:
        return {"Threads": str(int(cpu_threads))}
    return {}


def create_context(system: Any, integrator: Any, *, platform: str,
                   precision: str, device: int | None,
                   cpu_threads: int | None) -> tuple[Any, str]:
    """A Context on the platform asked for, or the first that works.

    Returns the context and the platform name it ended up on. With
    ``platform="auto"`` each candidate is tried in turn and the reason each
    one failed is logged, so a machine whose GPU is unusable says why before
    the run falls back to the CPU.
    """
    import openmm

    available = {openmm.Platform.getPlatform(i).getName()
                 for i in range(openmm.Platform.getNumPlatforms())}
    if platform != "auto":
        if platform not in available:
            raise InputError(
                f"OpenMM here has no {platform} platform. It has: "
                f"{', '.join(sorted(available))}.",
                code="resremd.input.platform")
        candidates = [platform]
    else:
        candidates = [p for p in _AUTO_ORDER if p in available]
    reasons = []
    for name in candidates:
        try:
            ctx = openmm.Context(
                system, integrator, openmm.Platform.getPlatformByName(name),
                _properties(name, precision, device, cpu_threads))
            return ctx, name
        except Exception as exc:  # OpenMMException has no finer types
            if platform != "auto":
                raise ResRemdError(
                    f"Could not create a {name} context: {exc}",
                    code="resremd.environment.platform") from exc
            reasons.append(f"{name}: {exc}")
            logger.info("Platform %s not usable here: %s", name, exc)
    raise ResRemdError("No OpenMM platform could create a context. "
                       + "; ".join(reasons),
                       code="resremd.environment.platform")


def warn_if_no_gpu(platform_asked: str, platform_used: str) -> None:
    """Say so, loudly, when `auto` ended up on a CPU.

    The usual cause on a cluster is a GPU plugin that failed to load (a
    driver older than the CUDA build), which leaves the platform absent
    rather than failing, so a job would otherwise spend its allocation on
    the CPU without a word.
    """
    import openmm

    if platform_asked != "auto" or platform_used in ("CUDA", "HIP", "OpenCL"):
        return
    failures = list(openmm.Platform.getPluginLoadFailures())
    detail = (" Plugins that failed to load: " + "; ".join(failures)
              if failures else "")
    logger.warning("No GPU platform is usable here; running on %s.%s",
                   platform_used, detail)


def select_atoms(topology: Any, selection: str,
                 atoms: list[int] | None = None, *,
                 option: str = "save") -> np.ndarray:
    """Indices of the selected atoms: to save, or with ``option="rest2"``
    REST2's solute. ``option`` names the settings in messages
    (`<option>_selection`, `<option>_atoms`)."""
    n = topology.getNumAtoms()
    if atoms is not None:
        try:
            idx = np.array(sorted({int(a) for a in atoms}), dtype=int)
        except (TypeError, ValueError):
            raise InputError(f"`{option}_atoms` must be atom indices.",
                             code="resremd.input.type")
        if idx.size == 0 or idx[0] < 0 or idx[-1] >= n:
            raise InputError(
                f"`{option}_atoms` must be indices between 0 and {n - 1}.",
                code="resremd.input.range")
        return idx
    if selection == "all":
        return np.arange(n)
    keep = []
    for residue in topology.residues():
        water = residue.name.upper() in WATER_RESIDUES
        atom_list = list(residue.atoms())
        ion = len(atom_list) == 1 and not water
        if water or (selection == "solute" and ion):
            continue
        keep.extend(a.index for a in atom_list)
    if not keep:
        raise InputError(
            f"`{option}_selection: {selection}` leaves no atoms"
            + (" to save." if option == "save" else "."),
            code="resremd.input.selection")
    return np.array(sorted(keep), dtype=int)


def subset_topology(topology: Any, indices: np.ndarray,
                    box: np.ndarray | None = None):
    """A Topology holding only the given atoms, with bonds between them.

    ``box`` (nm) sets the periodic box, for a topology that was built
    without one; otherwise the topology's own box is kept.
    """
    from openmm import Vec3, app, unit

    wanted = set(int(i) for i in indices)
    sub = app.Topology()
    vectors = topology.getPeriodicBoxVectors()
    if box is not None:
        sub.setPeriodicBoxVectors(
            [Vec3(*map(float, row)) for row in box] * unit.nanometer)
    elif vectors is not None:
        # Topologies carry their box in more than one form (a Quantity of
        # Vec3s, or Vec3s of Quantities); write it back in one.
        def nm(x):
            return x.value_in_unit(unit.nanometer) if unit.is_quantity(x) else x

        rows = [Vec3(*(float(nm(c)) for c in nm(v))) for v in vectors]
        sub.setPeriodicBoxVectors(rows * unit.nanometer)
    mapping = {}
    for chain in topology.chains():
        new_chain = None
        for residue in chain.residues():
            new_residue = None
            for atom in residue.atoms():
                if atom.index not in wanted:
                    continue
                if new_chain is None:
                    new_chain = sub.addChain(chain.id)
                if new_residue is None:
                    new_residue = sub.addResidue(residue.name, new_chain,
                                                 residue.id,
                                                 residue.insertionCode)
                mapping[atom] = sub.addAtom(atom.name, atom.element,
                                            new_residue, atom.id)
    for a, b in topology.bonds():
        if a in mapping and b in mapping:
            sub.addBond(mapping[a], mapping[b])
    return sub
