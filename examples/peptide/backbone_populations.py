"""Backbone populations over time, for one or more runs at their lowest
temperature.

Alanine dipeptide's (phi, psi) map is divided into three regions: right-handed
helix (alpha_R), extended and polyproline II (beta/PPII), and everything with
phi > 0 (alpha_L and C7ax). Populations are printed for each quarter of each
run. A converged run gives the same numbers in its later quarters, and a
reservoir run should reach them sooner than plain REMD.

    python backbone_populations.py remd reservoir_remd
"""

import json
import sys
from pathlib import Path

import mdtraj as md
import numpy as np


def regions(phi: np.ndarray, psi: np.ndarray) -> dict[str, np.ndarray]:
    left = phi > 0
    alpha_r = ~left & (psi > -120) & (psi < 50)
    return {"alpha_R": alpha_r, "beta/PPII": ~left & ~alpha_r,
            "phi>0": left}


def main(runs: list[str]) -> None:
    for run in map(Path, runs):
        manifest = json.loads((run / "manifest.json").read_text())
        lowest = manifest["states"][0]
        traj = md.load(run / lowest["trajectory"], top=run / "topology.pdb")
        _, phi = md.compute_phi(traj)
        _, psi = md.compute_psi(traj)
        phi, psi = np.degrees(phi[:, 0]), np.degrees(psi[:, 0])
        print(f"{run}: {manifest['method']}, "
              f"{lowest['temperature_K']:.1f} K, {traj.n_frames} frames")
        names = list(regions(phi, psi))
        print("  quarter " + " ".join(f"{n:>10}" for n in names))
        parts = np.array_split(np.arange(traj.n_frames),
                               min(4, traj.n_frames))
        for i, idx in enumerate(parts, start=1):
            r = regions(phi[idx], psi[idx])
            print(f"  {i:>7} " + " ".join(f"{r[n].mean():>10.3f}"
                                          for n in names))


if __name__ == "__main__":
    main(sys.argv[1:] or ["remd"])
