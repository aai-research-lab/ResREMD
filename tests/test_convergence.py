import numpy as np
import pytest

from resremd.convergence import (censored_median, convergence_time,
                                 js_divergence, populations,
                                 running_populations, standard_error,
                                 total_variation)


def test_populations_skip_unassigned():
    assert populations([0, 0, 1, -1], 3).tolist() == [2 / 3, 1 / 3, 0.0]
    assert np.isnan(populations([-1, -1], 2)).all()


def test_running_populations_with_discard():
    labels = [1, 1, 0, 0]
    out = running_populations(labels, 2, [2, 4], discard_fraction=0.5)
    assert out[0].tolist() == [0.0, 1.0]      # frames 1..1
    assert out[1].tolist() == [1.0, 0.0]      # frames 2..3


def test_divergences():
    assert js_divergence([1, 0], [1, 0]) == 0.0
    assert js_divergence([1, 0], [0, 1]) == pytest.approx(1.0)
    assert total_variation([0.5, 0.5], [0.8, 0.2]) == pytest.approx(0.3)
    assert js_divergence([2, 2], [1, 1]) == 0.0, "counts are normalised"


def test_convergence_time_needs_the_error_to_stay_down():
    t = [1, 2, 3, 4, 5]
    assert convergence_time(t, [0.3, 0.01, 0.2, 0.04, 0.03], 0.05) == 4.0
    assert convergence_time(t, [0.01] * 5, 0.05) == 1.0
    assert convergence_time(t, [0.01, 0.01, 0.01, 0.01, 0.2], 0.05) is None
    assert convergence_time(t, [np.nan, 0.01, 0.01, 0.01, 0.01], 0.05) == 2.0


def test_censored_median():
    r = censored_median([10, 20, 30], limit=100)
    assert r["median"] == 20 and r["converged_fraction"] == 1.0
    r = censored_median([10, None, None], limit=100)
    assert r["median"] is None and r["converged_fraction"] == pytest.approx(1 / 3)


def test_standard_error_accounts_for_correlation():
    rng = np.random.default_rng(0)
    white = rng.normal(size=10000)
    correlated = np.repeat(white[:1000], 10)
    assert standard_error(white) == pytest.approx(0.01, rel=0.15)
    assert standard_error(correlated) > 2.5 * standard_error(white)


def test_censored_median_with_even_repeats():
    r = censored_median([10, 30, None, 20], limit=100)
    assert r["median"] == 25.0
    r = censored_median([10, None, None, 20], limit=100)
    assert r["median"] is None
