import numpy as np
import pytest

from resremd import mbar
from resremd.thermo import BOLTZ, beta


def _gamma_ladder(rng, temps, n, a=30):
    """Energies with density of states h^(a-1): f(beta) = a ln(beta)."""
    h = np.concatenate([rng.gamma(a, BOLTZ * t, n) for t in temps])
    b = np.array([beta(t) for t in temps])
    return h, b, np.full(len(temps), n)


def test_solve_recovers_exact_free_energies():
    rng = np.random.default_rng(1)
    temps = [300.0, 330.0, 363.0, 400.0]
    h, b, n = _gamma_ladder(rng, temps, 5000)
    f = mbar.solve(b[:, None] * h[None, :], n)
    exact = 30 * np.log(b / b[0])
    assert f == pytest.approx(exact, abs=0.05)


def test_weights_reach_a_temperature_between_the_rungs():
    rng = np.random.default_rng(2)
    temps = [300.0, 330.0, 363.0, 400.0]
    h, b, n = _gamma_ladder(rng, temps, 5000)
    u = b[:, None] * h[None, :]
    f = mbar.solve(u, n)
    w = mbar.weights(u, n, f, beta(345.0) * h)
    assert np.sum(w * h) == pytest.approx(30 * BOLTZ * 345.0, rel=0.01)


def test_solve_refuses_bad_counts():
    with pytest.raises(ValueError, match="n_k"):
        mbar.solve(np.zeros((2, 5)), [2, 2])
    with pytest.raises(ValueError, match="needs samples"):
        mbar.solve(np.zeros((2, 5)), [5, 0])


def test_temperature_weights_give_exact_populations(tmp_path):
    """Double well: every temperature's frames, weighted to 300 K, give the
    exact well populations."""
    md = pytest.importorskip("mdtraj")
    import resremd
    from resremd import testsystems

    # An exact reservoir keeps every rung mixed, so the check is of MBAR.
    testsystems.write_double_well_reservoir(
        tmp_path / "r", kind="boltzmann", n_frames=4000, temperature_K=520.0)
    resremd.run(testsystems.double_well(), output=str(tmp_path / "run"),
                reservoir=str(tmp_path / "r"),
                temperatures_K=[300.0, 360.0, 432.0],
                production_steps=250 * 8000, exchange_interval_steps=250,
                trajectory_interval_steps=250 * 4, friction_per_ps=5.0,
                platform="Reference", random_seed=5, save_selection="all",
                equilibration_ns=0.0, minimize=False)
    out = mbar.temperature_weights(tmp_path / "run", 300.0,
                                   discard_fraction=0.1)
    assert out["states"] == [0, 1, 2]
    left = 0.0
    for s, w in out["weights"].items():
        traj = md.load(str(tmp_path / "run" / resremd_traj(tmp_path, s)),
                       top=str(tmp_path / "run/topology.pdb"))
        x = traj.xyz[out["first_frame"]:, 0, 0]
        assert len(x) == len(w)
        left += np.sum(w * (x < 0))
    exact = testsystems.left_fraction(300.0)
    assert left == pytest.approx(exact, abs=0.03)
    assert out["effective_samples"] > 1000


def resremd_traj(tmp_path, s):
    import json

    m = json.loads((tmp_path / "run/manifest.json").read_text())
    return m["states"][s]["trajectory"]
