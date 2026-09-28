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
