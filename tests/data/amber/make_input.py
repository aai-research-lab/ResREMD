"""The input cpptraj turned into res_bins.nc: 60 frames of alanine
dipeptide in vacuum (Amber14, no cutoff) at 500 K, as ala.dcd, and each
frame's OpenMM energy in kcal/mol, as ene.dat. See README.md."""

import numpy as np
import openmm
from openmm import app, unit

from resremd import testsystems

top, pos = testsystems.alanine_dipeptide()
system = app.ForceField("amber14-all.xml").createSystem(
    top, nonbondedMethod=app.NoCutoff, constraints=None)
reference = openmm.Platform.getPlatformByName("Reference")
integrator = openmm.LangevinMiddleIntegrator(
    500 * unit.kelvin, 5 / unit.picosecond, 1 * unit.femtosecond)
context = openmm.Context(system, integrator, reference)
context.setPositions(pos * unit.nanometer)
openmm.LocalEnergyMinimizer.minimize(context)
context.setVelocitiesToTemperature(500 * unit.kelvin, 1)
integrator.step(2000)
with open("ala.pdb", "w") as fh:
    app.PDBFile.writeFile(
        top, context.getState(getPositions=True).getPositions(), fh)
dcd = app.DCDFile(open("ala.dcd", "wb"), top, 0.001)
evaluate = openmm.Context(system, openmm.VerletIntegrator(0.001), reference)
energies = []
for _ in range(60):
    integrator.step(500)
    x = context.getState(getPositions=True).getPositions(asNumpy=True)
    # The energy of the coordinates as the DCD stores them (float32).
    x32 = np.asarray(x.value_in_unit(unit.nanometer), dtype=np.float32)
    dcd.writeModel(x32.astype(float) * 10 * unit.angstrom)
    evaluate.setPositions(x32.astype(float))
    energies.append(evaluate.getState(getEnergy=True)
                    .getPotentialEnergy()._value)
with open("ene.dat", "w") as fh:
    fh.write("#Frame E\n")
    for k, e in enumerate(energies):
        fh.write(f"{k + 1} {e / 4.184:.6f}\n")
