import pytest

from resremd.analysis import effective_ancestors, round_trips

import numpy as np


def test_effective_ancestors():
    assert effective_ancestors([3, 3, 3, 3]) == pytest.approx(1.0)
    assert effective_ancestors([1, 2, 3, 4]) == pytest.approx(4.0)
    assert effective_ancestors([1, 1, 2, 2]) == pytest.approx(2.0)
    assert effective_ancestors([]) == 0.0


def test_round_trips():
    # one replica: 0 -> 2 -> 0 -> 2 -> 0 is two trips of 2 cycles each
    states = np.array([[0], [2], [0], [2], [0]])
    trips, lengths = round_trips(states, top=2)
    assert trips == 2 and lengths == [2.0, 2.0]


def _gamma(rng, temperature, n, k=30):
    from resremd.thermo import BOLTZ

    return rng.gamma(k, BOLTZ * temperature, n)


def test_ensemble_check_passes_a_true_reservoir_and_catches_a_mislabel():
    from resremd.analysis import ensemble_check
    from resremd.thermo import beta

    rng = np.random.default_rng(3)
    top = _gamma(rng, 432, 20000)
    good = ensemble_check(top, _gamma(rng, 520, 20000), beta(432), beta(520))
    assert abs(good["z"]) < 4
    assert good["reservoir_temperature_implied_K"] == pytest.approx(520, rel=0.05)
    bad = ensemble_check(top, _gamma(rng, 700, 20000), beta(432), beta(520))
    assert bad["z"] < -10
    assert bad["reservoir_temperature_implied_K"] == pytest.approx(700, rel=0.05)


def test_ensemble_check_uses_reservoir_weights():
    from resremd.analysis import ensemble_check
    from resremd.thermo import beta

    rng = np.random.default_rng(4)
    top = _gamma(rng, 432, 20000)
    hot = _gamma(rng, 600, 40000)
    # Weights that turn a 600 K sample into a 520 K one.
    w = np.exp(-(beta(520) - beta(600)) * (hot - hot.mean()))
    r = ensemble_check(top, hot, beta(432), beta(520), reservoir_weights=w)
    assert abs(r["z"]) < 4


def test_summary_flags_a_mislabelled_reservoir(tmp_path):
    import resremd
    from resremd import testsystems
    from resremd.analysis import summarize
    from resremd.reservoir import write_reservoir

    rng = np.random.default_rng(5)
    x = testsystems.exact_x_samples(900.0, 5000, rng)
    frames = testsystems.double_well_frames(x, 900.0, rng)
    write_reservoir(tmp_path / "r", topology=testsystems.double_well().topology,
                    positions=frames, kind="boltzmann", temperature_K=520.0)
    resremd.run(testsystems.double_well(), output=str(tmp_path / "run"),
                reservoir=str(tmp_path / "r"), temperatures_K=[300, 432],
                production_steps=250 * 8000, exchange_interval_steps=250,
                trajectory_interval_steps=250 * 100, friction_per_ps=5.0,
                platform="Reference", random_seed=3, save_selection="all",
                equilibration_ns=0.0, minimize=False)
    check = summarize(tmp_path / "run")["reservoir_check"]
    assert check["z"] < -5
    assert check["reservoir_temperature_implied_K"] > 700
