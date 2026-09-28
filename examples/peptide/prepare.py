"""Prepare a peptide for replica exchange, in the layout FastMDXplora writes.

Writes system.xml, state.xml and topology.pdb to the output directory: the
same three files FastMDXplora's setup phase produces, so a system prepared
there can be used instead.

    python prepare.py alanine-dipeptide.pdb --solvent implicit --output setup

Implicit solvent (GBn2) is the fast choice for a demonstration and for small
peptides. Explicit solvent (TIP3P-FB, PME) is what the JPCB 2022 benchmarks
used. With it, the box is first equilibrated at constant pressure at the
lowest replica temperature, so that the constant-volume replica exchange
and the reservoir both run at the right density; the System written has no
barostat.
"""

import argparse
from pathlib import Path

import openmm
from openmm import app, unit


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
    if args.solvent == "implicit":
        forcefield = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(forcefield)
        system = forcefield.createSystem(modeller.topology,
                                         nonbondedMethod=app.NoCutoff,
                                         constraints=app.HBonds)
    else:
        forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(forcefield)
        modeller.addSolvent(forcefield, padding=args.padding_nm * unit.nanometer,
                            neutralize=True)
        system = forcefield.createSystem(modeller.topology,
                                         nonbondedMethod=app.PME,
                                         nonbondedCutoff=0.9 * unit.nanometer,
                                         constraints=app.HBonds,
                                         rigidWater=True)

    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(system, integrator)
    context.setPositions(modeller.positions)
    openmm.LocalEnergyMinimizer.minimize(context)
    state = context.getState(getPositions=True, enforcePeriodicBox=False)

    if args.solvent == "explicit" and args.npt_ns > 0:
        npt = openmm.XmlSerializer.deserialize(
            openmm.XmlSerializer.serialize(system))
        npt.addForce(openmm.MonteCarloBarostat(
            1.0 * unit.bar, args.temperature_K * unit.kelvin))
        dynamics = openmm.LangevinMiddleIntegrator(
            args.temperature_K * unit.kelvin, 1.0 / unit.picosecond,
            2.0 * unit.femtosecond)
        equilibrate = openmm.Context(npt, dynamics)
        equilibrate.setState(state)
        equilibrate.setVelocitiesToTemperature(args.temperature_K * unit.kelvin)
        steps = int(round(args.npt_ns * 1e6 / 2.0))
        print(f"Equilibrating the density: {args.npt_ns:g} ns at "
              f"{args.temperature_K:g} K and 1 bar")
        dynamics.step(steps)
        state = equilibrate.getState(getPositions=True,
                                     enforcePeriodicBox=False)
        modeller.topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "system.xml").write_text(openmm.XmlSerializer.serialize(system))
    (out / "state.xml").write_text(openmm.XmlSerializer.serialize(state))
    with open(out / "topology.pdb", "w") as fh:
        app.PDBFile.writeFile(modeller.topology, state.getPositions(), fh)
    print(f"Prepared {modeller.topology.getNumAtoms()} atoms "
          f"({args.solvent} solvent) in {out}/")


if __name__ == "__main__":
    main()
