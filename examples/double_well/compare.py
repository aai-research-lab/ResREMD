"""Reservoir REMD against plain REMD, on a system with an exact answer.

A particle in an asymmetric double well starts in the less favoured well.
Both runs use the same three temperatures and the same length; one is also
coupled to a reservoir drawn exactly at 520 K. The table shows the
population of the favoured well at 300 K in each quarter of each run,
against the exact value. Runs on a CPU in about a minute.

    python compare.py
"""

import logging
import shutil
from pathlib import Path

import numpy as np

import resremd
from resremd import testsystems

TEMPERATURES = [300.0, 360.0, 432.0]
CYCLES = 40000


def left_well_by_quarter(run: Path) -> list[float]:
    import mdtraj as md

    x = md.load(run / "trajectories/state_000_300.00K.dcd",
                top=run / "topology.pdb").xyz[:, 0, 0]
    return [float((q < 0).mean()) for q in np.array_split(x, 4)]


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    work = Path("double_well_runs")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    testsystems.write_double_well_reservoir(work / "reservoir",
                                            kind="boltzmann", n_frames=20000,
                                            temperature_K=520.0)
    common = dict(temperatures_K=TEMPERATURES,
                  production_steps=250 * CYCLES, exchange_interval_steps=250,
                  trajectory_interval_steps=250, friction_per_ps=5.0,
                  platform="Reference", random_seed=1, save_selection="all",
                  equilibration_ns=0.0, minimize=False)
    for name, extra in (("remd", {}),
                        ("reservoir_remd",
                         {"reservoir": str(work / "reservoir")})):
        resremd.run(testsystems.double_well(start_x=1.0),
                    output=str(work / name), **common, **extra)

    exact = testsystems.left_fraction(300.0)
    print(f"Favoured-well population at 300 K (exact {exact:.3f})")
    print(f"{'quarter':>8} {'REMD':>8} {'Res-REMD':>9}")
    plain = left_well_by_quarter(work / "remd")
    reservoir = left_well_by_quarter(work / "reservoir_remd")
    for i, (a, b) in enumerate(zip(plain, reservoir), start=1):
        print(f"{i:>8} {a:>8.3f} {b:>9.3f}")


if __name__ == "__main__":
    main()
