"""Metropolis criteria for the two kinds of exchange.

Both are the same rule. Two thermodynamic states a and b hold
configurations x_i and x_j; swapping them is accepted with probability

    min(1, exp(-[u_a(x_j) + u_b(x_i) - u_a(x_i) - u_b(x_j)]))

with u_k(x) = beta_k * h(x) and h the enthalpy of :mod:`resremd.thermo`.
When every state shares one Hamiltonian and one pressure this is

    log alpha = (beta_a - beta_b) * (h_i - h_j)

A reservoir is one more state. Its configuration is a fresh draw each time,
so what the replica gives up is discarded instead of stored: the reservoir
is an independent sample of its own distribution, which an exchange cannot
change. For a Boltzmann reservoir beta_R is 1/kT_R. A non-Boltzmann
reservoir, whose structures carry equal weight, is the limit beta_R = 0
(Roitberg, Okur and Simmerling, J. Phys. Chem. B 2007, 111, 2415).
"""

from __future__ import annotations

import math

import numpy as np


def log_acceptance(beta_a: float, beta_b: float, h_i: float, h_j: float) -> float:
    """log alpha for swapping x_i (held by state a) with x_j (held by b)."""
    return (beta_a - beta_b) * (h_i - h_j)


def log_acceptance_reservoir(beta_top: float, h_replica: float,
                             beta_reservoir: float, h_frame: float) -> float:
    """log alpha for replacing the hottest replica with a reservoir frame.

    ``beta_reservoir`` is 0 for a non-Boltzmann reservoir.
    """
    return (beta_top - beta_reservoir) * (h_replica - h_frame)


def accept(log_alpha: float, rng: np.random.Generator) -> bool:
    """Metropolis: accept with probability min(1, exp(log_alpha)).

    One uniform number is drawn whatever the outcome, so the random stream,
    and with it a seeded run, does not depend on the energies.
    """
    u = rng.random()
    if not math.isfinite(log_alpha):
        return log_alpha > 0
    return u < math.exp(min(0.0, log_alpha))
