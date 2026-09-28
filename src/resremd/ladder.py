"""Temperature ladders."""

from __future__ import annotations

from typing import Any

import numpy as np

from .errors import InputError


def geometric(t_min: float, t_max: float, n: int) -> list[float]:
    """n temperatures from t_min to t_max with a constant ratio.

    Geometric spacing gives roughly uniform acceptance when the heat
    capacity changes little across the ladder, which is the usual starting
    point before a ladder is tuned from a pilot run.
    """
    if n < 2:
        raise InputError("A ladder needs at least two temperatures.",
                         code="resremd.input.ladder")
    if not 0 < t_min < t_max:
        raise InputError(
            f"A ladder runs upward from a positive temperature; got "
            f"{t_min:g} K to {t_max:g} K.", code="resremd.input.ladder")
    ratio = (t_max / t_min) ** (1.0 / (n - 1))
    values = [t_min * ratio ** i for i in range(n)]
    values[-1] = float(t_max)
    return values


def below_reservoir(t_min: float, t_reservoir: float, n: int) -> list[float]:
    """n replica temperatures, with the reservoir as the next rung up.

    This reproduces the GROMACS implementation's layout, where the hottest
    of n + 1 replicas became the reservoir.
    """
    return geometric(t_min, t_reservoir, n + 1)[:-1]


def check(temperatures: list[Any]) -> list[float]:
    """Positive and strictly increasing, or refused."""
    try:
        values = [float(t) for t in temperatures]
    except (TypeError, ValueError):
        raise InputError(f"`temperatures_K` must be numbers; got "
                         f"{temperatures!r}.", code="resremd.input.ladder")
    if len(values) < 2:
        raise InputError("Replica exchange needs at least two temperatures.",
                         code="resremd.input.ladder")
    if any(not np.isfinite(t) or t <= 0 for t in values):
        raise InputError(f"Temperatures must be positive; got {values}.",
                         code="resremd.input.ladder")
    if any(b <= a for a, b in zip(values, values[1:])):
        raise InputError(
            f"Temperatures must increase from the first to the last; got "
            f"{values}. State 0 is the lowest.", code="resremd.input.ladder")
    return values


def resolve(options: dict[str, Any], reservoir_temperature_K: float | None
            ) -> list[float]:
    """The replica temperatures a run's settings describe."""
    if options.get("temperatures_K"):
        return check(options["temperatures_K"])
    n = options.get("n_replicas")
    if n is None:
        raise InputError(
            "Give the temperatures: `temperatures_K`, or `n_replicas` with "
            "`temperature_min_K` and either `temperature_max_K` or a "
            "reservoir to space the ladder up to.",
            code="resremd.input.ladder")
    t_min = options["temperature_min_K"]
    t_max = options.get("temperature_max_K")
    if t_max is not None:
        return check(geometric(t_min, t_max, n))
    if reservoir_temperature_K is None:
        raise InputError(
            "`temperature_max_K` is needed: there is no reservoir whose "
            "temperature could top the ladder (a non-Boltzmann reservoir has "
            "none).", code="resremd.input.ladder")
    return check(below_reservoir(t_min, reservoir_temperature_K, n))
