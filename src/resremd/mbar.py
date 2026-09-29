"""MBAR across the temperatures of a run (Shirts and Chodera 2008).

In temperature replica exchange every state has the same Hamiltonian at a
different beta, so a sample's reduced energy at state k is beta_k h, with h
the potential energy (U + PV, minus gamma A, at constant pressure). The
free energies f_k follow from all states' energies at once, and any sample
from any temperature can then be weighted to any temperature, including
ones between or beside the ladder's rungs while they overlap.

MBAR is consistent: its bias falls as 1/N. Replica exchange samples are
correlated, which does not change that but makes the effective number of
samples smaller than the count; judge uncertainty from independent runs.

pymbar is not needed. The solver is Newton's method on MBAR's convex
objective, with a backtracking line search.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .thermo import beta


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(a - m), axis=axis))


def solve(u_kn, n_k, *, initial=None, tolerance: float = 1e-10,
          max_iterations: int = 200) -> np.ndarray:
    """Dimensionless free energies f_k (f_0 = 0) from reduced energies.

    ``u_kn[k, n]`` is sample n's reduced energy at state k, for all samples
    pooled; ``n_k[k]`` the number of samples drawn from state k (at least
    one each). ``initial`` is a first guess.
    """
    u = np.asarray(u_kn, dtype=float)
    n = np.asarray(n_k, dtype=float)
    k, total = u.shape
    if n.shape != (k,) or n.sum() != total:
        raise ValueError("n_k must count the samples of each state, "
                         f"summing to {total}.")
    if np.any(n <= 0):
        raise ValueError("Every state needs samples.")
    log_n = np.log(n)

    def objective(f):
        return float(np.sum(_logsumexp((f + log_n)[:, None] - u, 0))
                     - np.dot(n, f))

    f = np.zeros(k) if initial is None else np.asarray(initial, float).copy()
    free = np.arange(1, k)
    for _ in range(max_iterations):
        a = (f + log_n)[:, None] - u
        w = np.exp(a - _logsumexp(a, 0)[None, :])          # (k, total)
        grad = w.sum(axis=1) - n
        hess = np.diag(w.sum(axis=1)) - w @ w.T
        try:
            step = np.linalg.solve(hess[np.ix_(free, free)], -grad[free])
        except np.linalg.LinAlgError:
            step = -grad[free] / np.maximum(np.diag(hess)[free], 1e-12)
        # Half the Newton decrement: how far the objective is from its
        # minimum, near it.
        decrement = -0.5 * float(np.dot(grad[free], step))
        if decrement < tolerance * total:
            break
        full = np.zeros(k)
        full[free] = step
        f0 = objective(f)
        t = 1.0
        while objective(f + t * full) > f0 + 1e-4 * t * np.dot(grad, full):
            t *= 0.5
            if t < 1e-10:
                # No decrease is representable: as close as float allows.
                return f - f[0]
        f = f + t * full
    else:
        raise RuntimeError("MBAR did not converge. The states may not "
                           "overlap.")
    return f - f[0]


def weights(u_kn, n_k, f_k, u_target) -> np.ndarray:
    """Normalised weights of the pooled samples at a target state."""
    u = np.asarray(u_kn, dtype=float)
    log_n = np.log(np.maximum(np.asarray(n_k, dtype=float), 1e-300))
    log_w = -np.asarray(u_target, dtype=float) - _logsumexp(
        (np.asarray(f_k) + log_n)[:, None] - u, 0)
    w = np.exp(log_w - log_w.max())
    return w / w.sum()


class TemperatureReweighting:
    """A finished run's energies, read once, reweighted on request.

        rw = TemperatureReweighting("remd")
        out = rw.weights(300.0)
        # out["weights"][k][i] is the weight of frame i of state k's
        # trajectory (counting from out["first_frame"]).

    The free energies come from every replica's energy at every cycle in
    the chosen stretch. The weights are for the frames saved at ``states``
    (by default every state with a trajectory) in that stretch; those
    frames are a mixture of the states, and the weights are normalised over
    that mixture.

    A REST2 run is reweighted the same way, with the reduced energy of a
    sample at state k being beta0 U(s_k), known at every scale from the
    run's `rest2_terms.csv`; temperatures are then the solute's effective
    ones. The PV term is the same at every REST2 state and drops out.
    """

    def __init__(self, run_dir: str | Path) -> None:
        from .analysis import _table
        from .thermo import Ensemble

        self.run_dir = Path(run_dir)
        manifest = json.loads((self.run_dir / "manifest.json").read_text())
        self.manifest = manifest
        settings = manifest["settings"]
        self.temperatures = np.array([s["temperature_K"]
                                      for s in manifest["states"]])
        self.betas = np.array([beta(t) for t in self.temperatures])
        table = _table(self.run_dir / "states.csv")
        self.state_of = table[:, 2:].astype(int)
        self.rest2 = manifest.get("rest2")
        if self.rest2:
            n = self.state_of.shape[1]
            terms = _table(self.run_dir / "rest2_terms.csv")[:, 2:]
            #: Per cycle and replica, A, B, C of U(s).
            self.h = terms.reshape(len(terms), n, 3)
            self.beta0 = beta(self.rest2["run_temperature_K"])
            self.scales = np.asarray(self.rest2["scales"], dtype=float)
        h = _table(self.run_dir / "energies.csv")[:, 2:]
        ensemble = Ensemble.from_dict(manifest["system"]["ensemble"])
        if ensemble.constant_pressure:
            vol = _table(self.run_dir / "volumes.csv")[:, 2:]
            area = _table(self.run_dir / "areas.csv")[:, 2:] \
                if (self.run_dir / "areas.csv").exists() \
                else np.zeros_like(vol)
            h = ensemble.enthalpy(h, vol, area)
        if not self.rest2:
            self.h = h
        every = settings["trajectory_interval_steps"] // \
            settings["exchange_interval_steps"]
        self.every = every
        #: Rows of the tables at which frames were saved.
        self.saved_rows = np.flatnonzero(table[:, 0].astype(int) % every
                                         == 0)
        self.saved_states = [i for i, s in enumerate(manifest["states"])
                             if s.get("trajectory")]

    @property
    def n_frames(self) -> int:
        """Frames in each state's trajectory."""
        return len(self.saved_rows)

    def _reduced(self, x: np.ndarray, targets: np.ndarray | None = None
                 ) -> np.ndarray:
        """Reduced energies (states or targets, samples) of samples ``x``:
        enthalpies, or REST2 terms. ``targets`` are temperatures (K)."""
        if self.rest2:
            from .rest2 import energy

            s = self.scales if targets is None else \
                np.sqrt(self.rest2["run_temperature_K"] / np.asarray(targets))
            return self.beta0 * energy(x[None, :, :], s[:, None])
        b = self.betas if targets is None else \
            np.array([beta(t) for t in targets])
        return b[:, None] * x[None, :]

    def free_energies(self, first_row: int = 0, last_row: int | None = None
                      ) -> np.ndarray:
        rows = slice(first_row, last_row)
        pooled_k = self.state_of[rows].ravel()
        n_k = np.bincount(pooled_k, minlength=len(self.betas))
        if self.rest2:
            x = self.h[rows].reshape(-1, 3)
            s = self.scales
            # First guess by the trapezoidal rule on d f / d s
            # = beta0 <2 A s + B>.
            g = np.array([self.beta0 * np.mean(2 * x[pooled_k == k, 0] * s[k]
                                               + x[pooled_k == k, 1])
                          for k in range(len(s))])
            guess = np.concatenate([[0.0], np.cumsum(
                np.diff(s) * 0.5 * (g[1:] + g[:-1]))])
            return solve(self._reduced(x), n_k, initial=guess)
        pooled_h = self.h[rows].ravel()
        b = self.betas
        # First guess by the trapezoidal rule on d f / d beta = <h>.
        mean_h = np.array([pooled_h[pooled_k == k].mean()
                           for k in range(len(b))])
        guess = np.concatenate([[0.0], np.cumsum(
            np.diff(b) * 0.5 * (mean_h[1:] + mean_h[:-1]))])
        return solve(b[:, None] * pooled_h[None, :], n_k, initial=guess)

    def weights(self, temperature_K: float, *,
                states: list[int] | None = None,
                frames: tuple[int, int] | None = None,
                discard_fraction: float = 0.0) -> dict[str, Any]:
        """Weights at ``temperature_K``.

        ``frames`` is the stretch (first, last) in saved frames; by default
        everything after ``discard_fraction`` of them.
        """
        n = self.n_frames
        first, last = frames if frames is not None \
            else (int(n * discard_fraction), n)
        if not 0 <= first < last <= n:
            raise ValueError(f"No frames in {first}..{last} of {n}.")
        states = list(self.saved_states if states is None else states)
        if not states:
            raise ValueError(f"{self.run_dir} saved no trajectories.")
        rows = self.saved_rows[first:last]
        f_k = self.free_energies(max(0, rows[0] - self.every + 1),
                                 rows[-1] + 1)
        fh = np.concatenate([
            self.h[rows, np.argmax(self.state_of[rows] == k, axis=1)]
            for k in states])
        counts = np.zeros(len(self.betas))
        counts[states] = len(rows)
        w = weights(self._reduced(fh), counts, f_k,
                    self._reduced(fh, np.array([temperature_K]))[0])
        per_state = np.split(w, len(states))
        return {
            "temperature_K": float(temperature_K),
            "states": states,
            "first_frame": first,
            "weights": {k: per_state[i] for i, k in enumerate(states)},
            "free_energies": f_k.tolist(),
            "effective_samples": float(1.0 / np.sum(w * w)),
        }


def temperature_weights(run_dir: str | Path, temperature_K: float, *,
                        states: list[int] | None = None,
                        discard_fraction: float = 0.0) -> dict[str, Any]:
    """Weights at ``temperature_K`` for every saved frame of a run; see
    :class:`TemperatureReweighting`."""
    return TemperatureReweighting(run_dir).weights(
        temperature_K, states=states, discard_fraction=discard_fraction)
