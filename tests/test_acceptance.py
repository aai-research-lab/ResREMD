import math

import numpy as np
import pytest
from openmm import unit

from resremd.acceptance import accept, log_acceptance, log_acceptance_reservoir
from resremd.thermo import BAR_NM3_TO_KJMOL, BOLTZ, Ensemble, beta


def test_constants_match_openmm():
    assert BOLTZ == pytest.approx(
        (unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA)
        .value_in_unit(unit.kilojoule_per_mole / unit.kelvin), rel=1e-9)
    pv = (1 * unit.bar * unit.nanometer ** 3 * unit.AVOGADRO_CONSTANT_NA)
    assert BAR_NM3_TO_KJMOL == pytest.approx(
        pv.value_in_unit(unit.kilojoule_per_mole), rel=1e-9)


def test_swap_matches_the_general_reduced_potential_form():
    ba, bb, hi, hj = beta(300), beta(330), -1000.0, -950.0
    general = -(ba * hj + bb * hi - ba * hi - bb * hj)
    assert log_acceptance(ba, bb, hi, hj) == pytest.approx(general)


def test_a_hotter_state_passing_down_a_lower_enthalpy_is_always_accepted():
    # The replica at the hotter state (b) holds the lower enthalpy.
    assert log_acceptance(beta(300), beta(330), -900.0, -1000.0) > 0


def test_non_boltzmann_is_the_infinite_temperature_limit():
    bt = beta(400)
    assert log_acceptance_reservoir(bt, -10.0, 0.0, -12.0) == \
        pytest.approx(bt * 2.0)
    assert log_acceptance_reservoir(bt, -10.0, 1e-12, -12.0) == \
        pytest.approx(bt * 2.0, rel=1e-9)


def test_enthalpy_includes_pv_and_tension():
    e = Ensemble(pressure_bar=1.0, surface_tension_bar_nm=5.0)
    h = e.enthalpy(-100.0, 30.0, 9.0)
    assert h == pytest.approx(-100.0 + BAR_NM3_TO_KJMOL * (30.0 - 45.0))
    assert Ensemble().enthalpy(-100.0, 30.0, 9.0) == -100.0


def test_metropolis_rate():
    rng = np.random.default_rng(0)
    la = math.log(0.3)
    rate = np.mean([accept(la, rng) for _ in range(40000)])
    assert rate == pytest.approx(0.3, abs=0.01)
    assert accept(5.0, rng) and not accept(-math.inf, rng)


def test_detailed_balance_on_two_levels():
    """Swaps alone leave the product distribution of two states stationary."""
    rng = np.random.default_rng(1)
    levels = np.array([0.0, 4.0])
    ba, bb = beta(300), beta(600)
    # Two replicas, each a two-level system; exchanges only. Start from
    # independent Boltzmann draws and check the joint distribution stays put.
    pa = np.exp(-ba * levels)
    pa /= pa.sum()
    pb = np.exp(-bb * levels)
    pb /= pb.sum()
    counts = np.zeros((2, 2))
    for _ in range(40000):
        i, j = rng.choice(2, p=pa), rng.choice(2, p=pb)
        if accept(log_acceptance(ba, bb, levels[i], levels[j]), rng):
            i, j = j, i
        counts[i, j] += 1
    counts /= counts.sum()
    assert np.allclose(counts, np.outer(pa, pb), atol=0.01)
