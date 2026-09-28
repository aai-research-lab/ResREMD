"""How fast an estimate settles, measured the same way for every method.

A run's estimate at time t uses every sample up to t (optionally after
discarding an initial fraction). Its error is a distance from a reference
distribution, or, without one, from a run of the same method started on the
other side of the barrier: two runs that began in different states and now
agree have both forgotten where they began.

Everything here is plain NumPy on arrays of state labels or histogram
counts, so it applies to any observable and any method's output.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def populations(labels, n_states: int) -> np.ndarray:
    """Fraction of samples in each state. Labels below 0 are left out."""
    labels = np.asarray(labels, dtype=int).ravel()
    labels = labels[labels >= 0]
    if labels.size == 0:
        return np.full(n_states, np.nan)
    counts = np.bincount(labels, minlength=n_states)[:n_states]
    return counts / counts.sum()


def running_populations(labels, n_states: int, points: Sequence[int], *,
                        discard_fraction: float = 0.0) -> np.ndarray:
    """State populations estimated from the first n samples, for each n.

    ``discard_fraction`` drops that fraction of each prefix from its start,
    the usual allowance for equilibration that shrinks as a run gets longer.
    Returns an array of shape (len(points), n_states).
    """
    labels = np.asarray(labels, dtype=int).ravel()
    out = np.empty((len(points), n_states))
    for i, n in enumerate(points):
        start = int(math.floor(discard_fraction * n))
        out[i] = populations(labels[start:n], n_states)
    return out


def js_divergence(p, q) -> float:
    """Jensen-Shannon divergence in bits: 0 for identical, 1 at most.

    Symmetric and finite when a state is empty in one distribution, which
    is exactly the case an unconverged run produces.
    """
    p = _normalised(p)
    q = _normalised(q)
    m = 0.5 * (p + q)
    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def total_variation(p, q) -> float:
    """Half the L1 distance: the largest population error over any set of
    states. 0.05 means no state, or group of states, is off by more than
    5 percentage points."""
    return float(0.5 * np.abs(_normalised(p) - _normalised(q)).sum())


def _normalised(p) -> np.ndarray:
    p = np.asarray(p, dtype=float).ravel()
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError("A distribution needs a positive, finite total.")
    return p / s


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def convergence_time(times, errors, threshold: float) -> float | None:
    """The first time after which the error stays below the threshold.

    Not the first time it dips below: an estimate that passes through the
    right answer on its way elsewhere has not converged. None if the run
    ends above the threshold.
    """
    times = np.asarray(times, dtype=float)
    errors = np.asarray(errors, dtype=float)
    above = ~(errors < threshold)  # NaN counts as not converged
    if above[-1]:
        return None
    last_above = np.flatnonzero(above)
    if last_above.size == 0:
        return float(times[0])
    return float(times[last_above[-1] + 1])


def censored_median(values: Sequence[float | None], limit: float,
                    *, n_boot: int = 2000, level: float = 0.95,
                    seed: int = 0) -> dict[str, float | None]:
    """Median convergence time over repeats, with a bootstrap interval.

    A repeat that never converged is censored: all that is known is that it
    would take longer than ``limit``. It is counted as infinite, which keeps
    the median honest (it is infinite when most repeats failed) and makes
    the interval open-ended where censoring reaches it.
    """
    x = np.array([np.inf if v is None else float(v) for v in values])
    if x.size == 0:
        return {"median": None, "low": None, "high": None,
                "converged_fraction": None, "repeats": 0, "limit": limit}
    rng = np.random.default_rng(seed)
    boot = _median(rng.choice(x, size=(n_boot, x.size), replace=True))
    alpha = (1.0 - level) / 2.0

    def finite(v: float) -> float | None:
        return None if not np.isfinite(v) else float(v)

    return {
        "median": finite(float(_median(x[None])[0])),
        "low": finite(np.quantile(boot, alpha, method="inverted_cdf")),
        "high": finite(np.quantile(boot, 1.0 - alpha,
                                   method="inverted_cdf")),
        "converged_fraction": float(np.isfinite(x).mean()),
        "repeats": int(x.size),
        "limit": float(limit),
    }


def _median(rows: np.ndarray) -> np.ndarray:
    """Row medians that accept infinities (censored repeats).

    NumPy interpolates between the two middle values, which is undefined
    when both are infinite; here the mean of two finite middles, and
    infinity if either is infinite.
    """
    rows = np.sort(rows, axis=1)
    n = rows.shape[1]
    if n % 2:
        return rows[:, n // 2]
    lo, hi = rows[:, n // 2 - 1], rows[:, n // 2]
    out = np.full(rows.shape[0], np.inf)
    both = np.isfinite(lo) & np.isfinite(hi)
    out[both] = 0.5 * (lo[both] + hi[both])
    return out


def standard_error(series) -> float:
    """Standard error of a correlated series' mean, from its statistical
    inefficiency."""
    from .statistics import statistical_inefficiency

    x = np.asarray(series, dtype=float).ravel()
    if x.size < 2:
        return float("nan")
    return float(x.std(ddof=1) * math.sqrt(statistical_inefficiency(x)
                                           / x.size))
