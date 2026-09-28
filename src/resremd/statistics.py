"""How many independent samples a correlated series holds."""

from __future__ import annotations

import numpy as np


def statistical_inefficiency(series, *, minimum_lag: int = 3) -> float:
    """g = 1 + 2 sum_t (1 - t/N) C(t), stopping where C first falls to zero.

    N / g is the effective number of independent samples. The truncation at
    the first non-positive autocorrelation after ``minimum_lag`` follows
    Chodera et al., J. Chem. Theory Comput. 2007, 3, 26. A constant series
    has no fluctuations to correlate and returns 1.
    """
    x = np.asarray(series, dtype=float).ravel()
    n = x.size
    if n < 2:
        return 1.0
    dx = x - x.mean()
    variance = float(dx @ dx) / n
    if variance <= 0.0:
        return 1.0
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(dx, size)
    acov = np.fft.irfft(spectrum * np.conj(spectrum), size)[:n]
    acov /= np.arange(n, 0, -1)
    correlation = acov / variance
    g = 1.0
    for t in range(1, n):
        c = correlation[t]
        if c <= 0.0 and t > minimum_lag:
            break
        g += 2.0 * c * (1.0 - t / n)
    return max(1.0, float(g))


def effective_samples(series) -> float:
    """N / g."""
    x = np.asarray(series).ravel()
    return x.size / statistical_inefficiency(x)
