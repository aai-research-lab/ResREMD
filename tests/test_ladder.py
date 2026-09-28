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
