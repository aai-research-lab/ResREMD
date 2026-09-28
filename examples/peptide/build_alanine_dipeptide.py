"""Build alanine dipeptide (ACE-ALA-NME) and write alanine-dipeptide.pdb.

The structure comes from ideal geometry in the extended region, with
hydrogens added by OpenMM (``resremd.testsystems.alanine_dipeptide``). Any
peptide PDB can be used in its place.

    python build_alanine_dipeptide.py
"""

from openmm import app

from resremd.testsystems import alanine_dipeptide


def main(output: str = "alanine-dipeptide.pdb") -> None:
    topology, positions = alanine_dipeptide()
    with open(output, "w") as fh:
        app.PDBFile.writeFile(topology, positions * 10.0, fh)
    print(f"Wrote {output}: {topology.getNumAtoms()} atoms")


if __name__ == "__main__":
    main()
