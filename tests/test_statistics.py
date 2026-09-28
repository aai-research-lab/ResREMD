import numpy as np
import pytest

from resremd.statistics import statistical_inefficiency


def test_white_noise():
    x = np.random.default_rng(0).normal(size=20000)
    assert statistical_inefficiency(x) == pytest.approx(1.0, abs=0.1)


def test_ar1_matches_theory():
    phi = 0.9
    rng = np.random.default_rng(1)
    x = np.empty(200000)
    x[0] = 0.0
    for i in range(1, x.size):
        x[i] = phi * x[i - 1] + rng.normal()
    expected = (1 + phi) / (1 - phi)
    assert statistical_inefficiency(x) == pytest.approx(expected, rel=0.1)


def test_constant():
    assert statistical_inefficiency(np.ones(100)) == 1.0
