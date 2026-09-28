"""Build alanine dipeptide (ACE-ALA-NME) from ideal geometry.

Heavy atoms are placed from standard bond lengths and angles, hydrogens are
added by OpenMM, and the result is written as alanine-dipeptide.pdb. Any
peptide PDB can be used in its place.

    python build_alanine_dipeptide.py
"""

import numpy as np
import openmm
from openmm import app, unit


def place(a, b, c, bond, angle, dihedral):
    """Position of d with |cd| = bond, angle bcd and dihedral abcd (degrees)."""
    angle, dihedral = np.radians(angle), np.radians(dihedral)
    bc = (c - b) / np.linalg.norm(c - b)
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n)
    m = np.cross(n, bc)
    d = np.array([-bond * np.cos(angle),
                  bond * np.sin(angle) * np.cos(dihedral),
                  bond * np.sin(angle) * np.sin(dihedral)])
    return c + d[0] * bc + d[1] * m + d[2] * n


def heavy_atoms(phi=-80.0, psi=150.0):
    """Backbone in nm, in the extended (beta) region by default."""
    x = {}
    x["ACE:CH3"] = np.array([0.0, 0.0, 0.0])
    x["ACE:C"] = np.array([0.152, 0.0, 0.0])
    x["ALA:N"] = place(np.array([0.0, 0.1, 0.0]), x["ACE:CH3"], x["ACE:C"],
                       0.133, 116.0, 180.0)
    x["ALA:CA"] = place(x["ACE:CH3"], x["ACE:C"], x["ALA:N"],
                        0.146, 122.0, 180.0)
    x["ALA:C"] = place(x["ACE:C"], x["ALA:N"], x["ALA:CA"],
                       0.152, 111.0, phi)
    x["NME:N"] = place(x["ALA:N"], x["ALA:CA"], x["ALA:C"],
                       0.133, 116.0, psi)
    x["NME:C"] = place(x["ALA:CA"], x["ALA:C"], x["NME:N"],
                       0.146, 122.0, 180.0)
    x["ACE:O"] = place(x["ALA:CA"], x["ALA:N"], x["ACE:C"], 0.123, 123.0, 0.0)
    x["ALA:O"] = place(x["NME:C"], x["NME:N"], x["ALA:C"], 0.123, 123.0, 0.0)
    # Choose the CB position that makes CA an L centre: seen with the
    # hydrogen towards the viewer, CO -> R -> N runs clockwise.
    for sign in (1.0, -1.0):
        cb = place(x["ALA:C"], x["ALA:N"], x["ALA:CA"], 0.153, 110.0,
                   sign * 122.0)
        ca = x["ALA:CA"]
        units = [(p - ca) / np.linalg.norm(p - ca)
                 for p in (x["ALA:N"], x["ALA:C"], cb)]
        h = -sum(units)
        if np.dot(np.cross(x["ALA:C"] - ca, cb - ca), h) < 0:
            x["ALA:CB"] = cb
            break
    return x


def main(output="alanine-dipeptide.pdb"):
    x = heavy_atoms()
    top = app.Topology()
    chain = top.addChain()
    positions = []
    for resname, names in (("ACE", ("CH3", "C", "O")),
                           ("ALA", ("N", "CA", "C", "O", "CB")),
                           ("NME", ("N", "C"))):
        residue = top.addResidue(resname, chain)
        for name in names:
            element = app.element.Element.getBySymbol(name[0])
            top.addAtom(name, element, residue)
            positions.append(x[f"{resname}:{name}"])
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
    state = context.getState(getPositions=True)
    with open(output, "w") as fh:
        app.PDBFile.writeFile(modeller.topology, state.getPositions(), fh)
    print(f"Wrote {output}: {modeller.topology.getNumAtoms()} atoms")


if __name__ == "__main__":
    main()
