"""The sampled distributions match exact ones.

A particle in an asymmetric double well starts in the less favoured well.
Its well populations at each temperature are known exactly, and the
harmonic y and z directions fix the temperature itself (mean energy kT).
The reservoir is drawn exactly, so any deviation beyond statistical error
is an error in the method.
"""

import numpy as np
import pytest

from resremd import testsystems
import resremd
from resremd.statistics import statistical_inefficiency
from resremd.thermo import BOLTZ

md = pytest.importorskip("mdtraj")

TEMPERATURES = [300.0, 360.0, 432.0]


def check_against_exact(run_dir, tolerance_sigma=4.0):
    for s, t in enumerate(TEMPERATURES):
        xyz = md.load(f"{run_dir}/trajectories/state_{s:03d}_{t:.2f}K.dcd",
                      top=f"{run_dir}/topology.pdb").xyz[:, 0, :].astype(float)
        left = (xyz[:, 0] < 0).astype(float)
        p = testsystems.left_fraction(t)
        se = np.sqrt(p * (1 - p) * statistical_inefficiency(left) / left.size)
        assert abs(left.mean() - p) < tolerance_sigma * se, (t, left.mean(), p)
        # 0.5 k (y^2 + z^2) has mean kT at temperature T.
        e = 0.5 * testsystems.SPRING * (xyz[:, 1] ** 2 + xyz[:, 2] ** 2)
        kt = BOLTZ * t
        se_e = e.std() * np.sqrt(statistical_inefficiency(e) / e.size)
        assert abs(e.mean() - kt) < tolerance_sigma * se_e, (t, e.mean(), kt)


def run_double_well(tmp_path, kind, cycles, **extra):
    reservoir = tmp_path / "reservoir"
    prepared = testsystems.write_double_well_reservoir(
        reservoir, kind=kind, n_frames=20000,
        temperature_K=520.0 if kind == "boltzmann" else None)
    resremd.run(prepared, output=str(tmp_path / "run"),
                reservoir=str(reservoir), temperatures_K=TEMPERATURES,
                production_steps=250 * cycles, exchange_interval_steps=250,
                trajectory_interval_steps=250, friction_per_ps=5.0,
                platform="Reference", random_seed=5, save_selection="all",
                equilibration_ns=0.0, minimize=False, **extra)
    return tmp_path / "run"


def test_boltzmann_reservoir_shared_context(tmp_path):
    run = run_double_well(tmp_path, "boltzmann", 30000)
    check_against_exact(run)


def test_non_boltzmann_reservoir_resident_contexts(tmp_path):
    run = run_double_well(tmp_path, "non_boltzmann", 30000,
                          contexts_per_device=3)
    check_against_exact(run)


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["boltzmann", "non_boltzmann"])
def test_long(tmp_path, kind):
    run = run_double_well(tmp_path, kind, 200000)
    check_against_exact(run)
