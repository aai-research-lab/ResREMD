import pytest

from resremd.errors import InputError
from resremd.options import GENERATE, RUN, SCHEMAS, resolve


def test_defaults_are_declared_once():
    o = resolve(RUN, {})
    assert o["exchange_interval_steps"] == 500
    assert o["integrator"] == "langevin_middle"
    assert o["pressure_bar"] is None


def test_unknown_setting_is_refused_with_a_suggestion():
    with pytest.raises(InputError, match="temperature_min_K") as err:
        resolve(RUN, {"temperature_minK": 300})
    assert err.value.code == "resremd.input.unknown"


@pytest.mark.parametrize("key,value,code", [
    ("temperature_min_K", 0.0, "resremd.input.range"),
    ("temperature_min_K", -5, "resremd.input.range"),
    ("n_replicas", 1, "resremd.input.range"),
    ("exchange_interval_steps", 2.5, "resremd.input.type"),
    ("minimize", 1, "resremd.input.type"),
    ("n_replicas", True, "resremd.input.type"),
    ("platform", "Metal", "resremd.input.choice"),
])
def test_values_are_checked(key, value, code):
    with pytest.raises(InputError) as err:
        resolve(RUN, {key: value})
    assert err.value.code == code


def test_integers_are_accepted_as_floats():
    assert resolve(RUN, {"timestep_fs": 2})["timestep_fs"] == 2.0


def test_required_settings():
    with pytest.raises(InputError, match="temperature_K"):
        resolve(GENERATE, {"duration_ns": 1.0})


def test_no_em_dashes_in_help():
    for schema in SCHEMAS.values():
        for option in schema.options:
            assert "\u2014" not in option.help, option.name
