"""Prepare a peptide for replica exchange, in the layout FastMDXplora writes.

Writes system.xml, state.xml and topology.pdb to the output directory: the
same three files FastMDXplora's setup phase produces, so a system prepared
there can be used instead.

    python prepare.py alanine-dipeptide.pdb --solvent implicit --output setup

Implicit solvent (GBn2) is the fast choice for a demonstration and for small
peptides. With explicit solvent (TIP3P-FB, PME) the box is first
equilibrated at constant pressure at the lowest replica temperature, so that
constant-volume replica exchange and the reservoir both run at the right
density; the System written has no barostat.
"""

import argparse

from openmm import app, unit

from resremd.system import write_prepared
from resremd.testsystems import prepare_peptide


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pdb")
    parser.add_argument("--solvent", choices=("implicit", "explicit"),
                        default="implicit")
    parser.add_argument("--padding-nm", type=float, default=1.2,
                        help="Solvent padding around the peptide (explicit).")
    parser.add_argument("--temperature-K", type=float, default=300.0,
                        help="Lowest replica temperature (explicit: the "
                             "density is equilibrated at it).")
    parser.add_argument("--npt-ns", type=float, default=0.5,
                        help="Constant-pressure equilibration at 1 bar "
                             "(explicit).")
    parser.add_argument("--output", default="setup")
    args = parser.parse_args()

    pdb = app.PDBFile(args.pdb)
    positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    system, topology, positions, box = prepare_peptide(
        pdb.topology, positions, solvent=args.solvent,
        padding_nm=args.padding_nm, npt_ns=args.npt_ns,
        temperature_K=args.temperature_K)
    write_prepared(args.output, system, topology, positions, box)
    print(f"Prepared {topology.getNumAtoms()} atoms ({args.solvent} solvent) "
          f"in {args.output}/")


if __name__ == "__main__":
    main()
