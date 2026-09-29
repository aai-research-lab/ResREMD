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
    assert good["status"] == "ok" and abs(good["z"]) < 4
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


def test_ensemble_check_says_why_it_cannot_test():
    from resremd.analysis import ensemble_check
    from resremd.thermo import beta

    rng = np.random.default_rng(6)
    top = _gamma(rng, 300, 5000)
    assert ensemble_check(top, _gamma(rng, 520, 20), beta(300), beta(520)
                          )["status"] == "too_few_frames"
    assert ensemble_check(top, top + 1e4, beta(300), beta(520)
                          )["status"] == "no_overlap"


def test_ensemble_check_is_calibrated_with_correlated_samples():
    """z stays near unit spread when both series are strongly correlated."""
    from resremd.analysis import ensemble_check
    from resremd.thermo import beta

    zs = []
    for k in range(20):
        rng = np.random.default_rng(100 + k)
        # 400 independent draws each repeated 25 times: g about 25.
        top = np.repeat(_gamma(rng, 432, 400), 25)
        res = np.repeat(_gamma(rng, 520, 400), 25)
        r = ensemble_check(top, res, beta(432), beta(520), bootstrap=100,
                           seed=k)
        zs.append(r["z"])
    assert np.std(zs) < 1.6 and abs(np.mean(zs)) < 1.0


def _two_state(rng, n, p_right, stay=0.0):
    """Labels 0/1 with right-state probability p_right, optionally sticky."""
    x = (rng.random(n) < p_right).astype(int)
    if stay:
        for i in range(1, n):
            if rng.random() < stay:
                x[i] = x[i - 1]
    return x


def test_coverage_check_passes_a_true_reservoir():
    from resremd.analysis import coverage_check
    from resremd.thermo import beta

    rng = np.random.default_rng(11)
    # States of equal energy spread, so reweighting 520 K -> 432 K changes
    # nothing and both sides should show the same populations.
    top = _two_state(rng, 4000, 0.3, stay=0.9)
    res = _two_state(rng, 3000, 0.3)
    h = rng.normal(0.0, 5.0, res.size)
    r = coverage_check(top, res, h, beta(432), beta(520), 2)
    assert r["status"] == "ok" and r["max_abs_z"] < 4
    assert r["unsupported_states"] == []


def test_coverage_check_reweights_to_the_top_temperature():
    from resremd.analysis import coverage_check
    from resremd.thermo import BOLTZ, beta

    rng = np.random.default_rng(12)
    # State 1 lies 10 kJ/mol above state 0. At 520 K it holds p520; the
    # top replica at 432 K sees less, and reweighting must account for it.
    def p(t):
        return 1.0 / (1.0 + np.exp(10.0 / (BOLTZ * t)))

    res = _two_state(rng, 20000, p(520))
    h = 10.0 * res + rng.normal(0.0, 1.0, res.size)
    top = _two_state(rng, 20000, p(432))
    r = coverage_check(top, res, h, beta(432), beta(520), 2)
    assert r["max_abs_z"] < 4
    assert r["reservoir_populations_at_top"][1] == pytest.approx(p(432),
                                                                 abs=0.01)


def test_coverage_check_flags_a_state_the_reservoir_lacks():
    from resremd.analysis import coverage_check
    from resremd.thermo import beta

    rng = np.random.default_rng(13)
    top = np.zeros(2000, dtype=int)
    top[rng.choice(2000, 10, replace=False)] = 2   # a rare third state
    res = np.zeros(1000, dtype=int)
    h = rng.normal(0.0, 5.0, res.size)
    r = coverage_check(top, res, h, beta(432), beta(520), 3)
    assert r["unsupported_states"] == [2]
    # Visited fewer times than min_visits: noise, not a missing state.
    assert coverage_check(top, res, h, beta(432), beta(520), 3,
                          min_visits=20)["unsupported_states"] == []


def test_reservoir_coverage_reads_a_run(tmp_path):
    import resremd
    from resremd import testsystems
    from resremd.reservoir import write_reservoir

    rng = np.random.default_rng(14)
    x = testsystems.exact_x_samples(520.0, 3000, rng)
    x = x[x > 0]                     # the left well removed
    frames = testsystems.double_well_frames(x, 520.0, rng)
    write_reservoir(tmp_path / "r", topology=testsystems.double_well().topology,
                    positions=frames, kind="boltzmann", temperature_K=520.0)
    resremd.run(testsystems.double_well(), output=str(tmp_path / "run"),
                reservoir=str(tmp_path / "r"), temperatures_K=[300, 432],
                production_steps=250 * 400, exchange_interval_steps=250,
                trajectory_interval_steps=250, friction_per_ps=5.0,
                platform="Reference", random_seed=3, save_selection="all",
                equilibration_ns=0.0, minimize=False)
    # The caller's classification of the top-temperature frames: here
    # made up, with the left well visited 50 times.
    top = np.ones(400, dtype=int)
    top[:50] = 0
    labels = np.ones(len(x), dtype=int)
    r = resremd.reservoir_coverage(tmp_path / "run", top, labels, 2)
    assert r["status"] == "ok" and r["unsupported_states"] == [0]
    with pytest.raises(ValueError, match="reservoir labels"):
        resremd.reservoir_coverage(tmp_path / "run", top, labels[:-1], 2)


def test_coverage_check_is_calibrated_with_state_dependent_weights():
    """A biased reservoir oversamples a rare state and its weights undo it;
    z must still have unit spread when nothing is wrong."""
    from resremd.analysis import coverage_check
    from resremd.thermo import beta

    zs = []
    for k in range(200):
        rng = np.random.default_rng(1000 + k)
        top = (rng.random(2000) < 0.02).astype(int)
        lab = (rng.random(2000) < 0.5).astype(int)      # sampled at 0.5
        w = np.where(lab == 1, 0.02 / 0.5, 0.98 / 0.5)   # weighted to 0.02
        r = coverage_check(top, lab, np.zeros(lab.size), beta(432),
                           beta(432), 2, reservoir_weights=w)
        zs.append(r["z"][1])
    assert 0.8 < np.std(zs) < 1.2


def test_coverage_check_flags_certain_disagreement():
    from resremd.analysis import coverage_check
    from resremd.thermo import beta

    top = np.zeros(100, dtype=int)
    lab = np.array([0] * 10 + [1] * 90)
    w = np.where(lab == 0, 0.0, 1.0)      # state 0 weighted out entirely
    r = coverage_check(top, lab, np.zeros(lab.size), beta(432), beta(432), 2,
                       reservoir_weights=w)
    assert r["max_abs_z"] == np.inf and r["unsupported_states"] == [0]
