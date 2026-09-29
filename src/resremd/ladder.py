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


class PilotAcceptance:
    """Neighbour acceptance between any two temperatures, predicted from a
    pilot run's energies.

    The distribution of states at a temperature T is the pilot's pooled
    samples weighted to T by MBAR, so T must lie within the pilot's ladder.
    A swap of x (from state a) with y (from state b) is accepted with
    min(1, exp(g(y) - g(x))), where g = u_b - u_a is a sample's reduced
    energy difference between the two states: (beta_b - beta_a) h in
    temperature REMD, beta0 [U(s_b) - U(s_a)] in REST2. The expected
    acceptance over both distributions is computed exactly over the
    samples, after sorting them by g.

    For a REST2 pilot, temperatures are the solute's effective ones, and
    the ladder starts at the pilot's run temperature (scale 1).
    """

    def __init__(self, run_dir, *, discard_fraction: float = 0.1,
                 max_samples: int = 20000) -> None:
        from .mbar import TemperatureReweighting, _logsumexp

        rw = TemperatureReweighting(run_dir)
        self.rest2 = bool(rw.rest2)
        n = rw.state_of.shape[0]
        # Every state's energy at a cycle is one sample. Consecutive cycles
        # are correlated, so thinning to max_samples loses little.
        stride = max(1, int(np.ceil(n * (1 - discard_fraction)
                                    * rw.state_of.shape[1] / max_samples)))
        rows = slice(int(n * discard_fraction), None, stride)
        self.run_dir = run_dir
        self.pilot_temperatures = rw.temperatures
        self.observed = [p["acceptance"] for p in
                         rw.manifest["exchanges"]["neighbour_pairs"]]
        x = rw.h[rows]
        self.x = x.reshape(-1, 3) if self.rest2 else x.ravel()
        self.k = rw.state_of[rows].ravel()
        self.n_k = np.bincount(self.k, minlength=len(rw.betas))
        self._reduced = rw._reduced
        self.f = rw.free_energies(rows.start)
        self._cache: dict[float, np.ndarray] = {}
        self._log_denominator = _logsumexp(
            (self.f + np.log(self.n_k))[:, None] - self._reduced(self.x), 0)

    def _check_range(self, temperature_K: float) -> None:
        lo, hi = self.pilot_temperatures[0], self.pilot_temperatures[-1]
        if not lo - 1e-9 <= temperature_K <= hi + 1e-9:
            raise InputError(
                f"{temperature_K:g} K is outside the pilot's ladder "
                f"({lo:g} to {hi:g} K); a pilot cannot predict beyond it.",
                code="resremd.input.ladder")

    def _log_weights(self, temperature_K: float) -> np.ndarray:
        self._check_range(temperature_K)
        if temperature_K not in self._cache:
            if len(self._cache) > 8:          # bisection visits many
                self._cache.clear()
            log_w = -self._reduced(self.x, np.array([temperature_K]))[0] \
                - self._log_denominator
            self._cache[temperature_K] = log_w - np.logaddexp.reduce(log_w)
        return self._cache[temperature_K]

    def __call__(self, t_a: float, t_b: float) -> float:
        la, lb = self._log_weights(t_a), self._log_weights(t_b)
        u = self._reduced(self.x, np.array([t_a, t_b], dtype=float))
        g = u[1] - u[0]
        order = np.argsort(g, kind="stable")
        g, la, lb = g[order], la[order], lb[order]
        # x = sample i from a, y = sample j from b. Every j with
        # g_j >= g_i is accepted (ties count; they are measure zero for
        # continuous energies): the suffix sums of b's weights from i on.
        at_or_above = np.exp(np.logaddexp.accumulate(lb[::-1])[::-1])
        # The others with exp(g_j - g_i): the prefix sums strictly below i.
        below = np.concatenate([[-np.inf],
                                np.logaddexp.accumulate(lb + g)[:-1]])
        rest = np.exp(np.minimum(below - g, 0.0))
        return float(np.sum(np.exp(la) * (at_or_above + rest)))

    def _next(self, t: float, t_max: float, target: float) -> float:
        if self(t, t_max) >= target:
            return t_max
        lo, hi = t, t_max
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if self(t, mid) >= target:
                lo = mid
            else:
                hi = mid
        return lo

    def ladder(self, t_min: float, t_max: float, target: float = 0.3, *,
               max_temperatures: int = 1000) -> list[float]:
        """The fewest temperatures from t_min to t_max with every
        neighbour pair predicted at ``target`` acceptance or above, spaced
        so the pairs are predicted equal."""
        if not 0.0 < target < 1.0:
            raise InputError("The target acceptance is between 0 and 1.",
                             code="resremd.input.ladder")

        def greedy(a, limit=max_temperatures - 1):
            temps = [t_min]
            while temps[-1] < t_max:
                if len(temps) > limit:
                    return None
                temps.append(self._next(temps[-1], t_max, a))
            return temps

        temps = greedy(target)
        if temps is None:
            raise InputError(
                f"More than {max_temperatures} temperatures would be needed "
                f"for {target:g} acceptance; lower the target.",
                code="resremd.input.ladder")
        n = len(temps)
        if n <= 2:
            return [float(t_min), float(t_max)]
        # Raise the target while the same number of rungs still reaches
        # t_max: the last pair then stops being the odd one out.
        lo, hi = target, 0.999
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            if greedy(mid, limit=n - 1) is not None:
                lo = mid
            else:
                hi = mid
        best = greedy(lo)
        best[-1] = float(t_max)
        return [float(t) for t in best]


def from_pilot(run_dir, *, temperature_min_K: float | None = None,
               temperature_max_K: float | None = None,
               target_acceptance: float = 0.3,
               discard_fraction: float = 0.1) -> dict[str, Any]:
    """A ladder tuned on a pilot run's energies.

    The range defaults to the pilot's own and cannot exceed it. For a REST2
    pilot the temperatures are effective ones and the ladder starts at the
    pilot's run temperature. Returns the temperatures, the acceptance
    predicted for each neighbour pair, and for the pilot's own ladder the
    predicted against the observed acceptance, as a check of the
    prediction.
    """
    pilot = PilotAcceptance(run_dir, discard_fraction=discard_fraction)
    pt = [float(t) for t in pilot.pilot_temperatures]
    t_min = pt[0] if temperature_min_K is None else float(temperature_min_K)
    if pilot.rest2 and abs(t_min - pt[0]) > 1e-6:
        raise InputError(
            f"A REST2 ladder starts at the temperature every replica runs "
            f"at, the pilot's {pt[0]:g} K; a pilot at another temperature "
            f"is needed to tune from {t_min:g} K.",
            code="resremd.input.ladder")
    t_max = pt[-1] if temperature_max_K is None else float(temperature_max_K)
    check([t_min, t_max])
    temps = pilot.ladder(t_min, t_max, target_acceptance)
    return {
        "temperatures_K": temps,
        "predicted_acceptance": [pilot(a, b) for a, b in zip(temps,
                                                             temps[1:])],
        "pilot_temperatures_K": pt,
        "pilot_predicted_acceptance": [pilot(a, b) for a, b in zip(pt,
                                                                   pt[1:])],
        "pilot_observed_acceptance": pilot.observed,
        "target_acceptance": target_acceptance,
        "rest2": pilot.rest2,
    }
