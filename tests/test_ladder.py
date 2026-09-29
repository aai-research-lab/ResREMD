import numpy as np
import pytest

from resremd.errors import InputError
from resremd.ladder import below_reservoir, check, geometric, resolve
from resremd.options import RUN, resolve as resolve_options


def test_geometric_has_constant_ratio():
    t = geometric(300, 450, 5)
    ratios = [b / a for a, b in zip(t, t[1:])]
    assert t[0] == 300 and t[-1] == 450
    assert max(ratios) - min(ratios) < 1e-12


def test_reservoir_takes_the_next_rung():
    t = below_reservoir(300, 500, 7)
    full = geometric(300, 500, 8)
    assert t == full[:-1] and len(t) == 7


def test_resolve_prefers_an_explicit_ladder():
    o = resolve_options(RUN, {"temperatures_K": [300, 320], "n_replicas": 9})
    assert resolve(o, 500.0) == [300.0, 320.0]


def test_resolve_without_a_top():
    o = resolve_options(RUN, {"n_replicas": 4})
    assert resolve(o, 500.0)[-1] < 500.0
    with pytest.raises(InputError, match="temperature_max_K"):
        resolve(o, None)


@pytest.mark.parametrize("bad", [[300], [300, 300], [320, 300], [0, 300]])
def test_bad_ladders(bad):
    with pytest.raises(InputError):
        check(bad)


@pytest.fixture(scope="module")
def pilot(tmp_path_factory):
    import resremd
    from resremd import testsystems

    out = tmp_path_factory.mktemp("pilot") / "p"
    resremd.run(testsystems.double_well(), output=str(out),
                temperatures_K=[300, 340, 390, 450, 520],
                production_steps=250 * 4000, exchange_interval_steps=250,
                trajectory_interval_steps=250 * 100, friction_per_ps=5.0,
                platform="Reference", random_seed=3, save_selection="all",
                equilibration_ns=0.0, minimize=False)
    return out


def test_pilot_predicts_its_own_acceptance(pilot):
    from resremd.ladder import from_pilot

    r = from_pilot(pilot, target_acceptance=0.9)
    assert np.allclose(r["pilot_predicted_acceptance"],
                       r["pilot_observed_acceptance"], atol=0.03)
    temps = r["temperatures_K"]
    assert temps[0] == 300 and temps[-1] == 520 and len(temps) > 2
    acc = r["predicted_acceptance"]
    assert min(acc) >= 0.9 - 1e-6 and max(acc) - min(acc) < 0.01


def test_pilot_cannot_predict_beyond_its_ladder(pilot):
    from resremd.errors import ResRemdError
    from resremd.ladder import from_pilot

    with pytest.raises(ResRemdError, match="outside the pilot"):
        from_pilot(pilot, temperature_max_K=600.0)


def test_ladder_cli_from_pilot(pilot, capsys):
    from resremd.cli import main

    assert main(["ladder", "--from-pilot", str(pilot),
                 "--target-acceptance", "0.8"]) == 0
    assert "temperatures_K: [300.00" in capsys.readouterr().out


def test_pilot_refuses_a_target_needing_too_many_rungs(pilot):
    from resremd.errors import ResRemdError
    from resremd.ladder import PilotAcceptance

    with pytest.raises(ResRemdError, match="lower the target"):
        PilotAcceptance(pilot).ladder(300.0, 520.0, 0.99,
                                      max_temperatures=5)


@pytest.fixture(scope="module")
def rest2_pilot(tmp_path_factory):
    import resremd
    from resremd import testsystems

    out = tmp_path_factory.mktemp("rest2_pilot") / "p"
    resremd.run(testsystems.torsion_model(), output=str(out), rest2=True,
                rest2_selection="all", temperatures_K=geometric(300, 3000, 5),
                production_steps=250 * 4000, exchange_interval_steps=250,
                trajectory_interval_steps=250 * 100, friction_per_ps=5.0,
                platform="Reference", random_seed=4, save_selection="all",
                equilibration_ns=0.0, minimize=False)
    return out


def test_rest2_pilot_predicts_a_new_ladder(rest2_pilot, tmp_path):
    """The prediction, checked against a REST2 run on the tuned ladder."""
    import json

    import resremd
    from resremd import testsystems
    from resremd.ladder import from_pilot

    r = from_pilot(rest2_pilot, target_acceptance=0.6)
    assert r["rest2"]
    temps = r["temperatures_K"]
    assert temps[0] == 300 and temps[-1] == 3000 and 2 < len(temps) < 5
    resremd.run(testsystems.torsion_model(), output=str(tmp_path / "run"),
                rest2=True, rest2_selection="all", temperatures_K=temps,
                production_steps=250 * 4000, exchange_interval_steps=250,
                trajectory_interval_steps=250 * 100, friction_per_ps=5.0,
                platform="Reference", random_seed=9, save_selection="all",
                equilibration_ns=0.0, minimize=False)
    man = json.loads((tmp_path / "run/manifest.json").read_text())
    for pair, predicted in zip(man["exchanges"]["neighbour_pairs"],
                               r["predicted_acceptance"]):
        # The pilot is as long as this run, so the prediction's error is
        # about that of the observed fraction: sqrt(2) standard errors.
        p = pair["acceptance"]
        se = np.sqrt(2 * p * (1 - p) / pair["attempts"])
        assert abs(p - predicted) < 5 * se, (p, predicted, se)


def test_a_rest2_ladder_starts_at_the_pilots_temperature(rest2_pilot):
    from resremd.errors import ResRemdError
    from resremd.ladder import from_pilot

    with pytest.raises(ResRemdError, match="REST2 ladder starts"):
        from_pilot(rest2_pilot, temperature_min_K=350.0)


def test_ladder_cli_names_a_rest2_ladder(rest2_pilot, capsys):
    from resremd.cli import main

    assert main(["ladder", "--from-pilot", str(rest2_pilot)]) == 0
    assert "REST2: effective temperatures" in capsys.readouterr().out
