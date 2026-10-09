"""Runs of OpenMM's own replica exchange, read for ResREMD's analysis.

OpenMM 8.6 added `openmm.app.ReplicaExchangeSampler`, whose
`ReplicaExchangeReporter` writes a directory of plain files:

- ``log.csv``: the state each replica held at each reported iteration;
- ``energy.csv`` (with ``energy=True``): every replica's reduced energy in
  every state, u = U / kT of that state;
- ``volume.csv`` (with ``volume=True``): every replica's box volume (nm^3);
- ``state_<k>.dcd`` or ``.xtc`` (with ``trajectoryPerState=True``): one
  frame per reported iteration of whichever replica held state k;
- ``checkpoint_<i>.xml`` (with ``checkpoints=True``): each replica's last
  State: its box, its time, and among its parameters a barostat if there
  is one.

:class:`OpenMMRun` reads them into ResREMD's exchange summary and MBAR
weights, so a run of either sampler is analysed the same way:

    run = resremd.OpenMMRun("remd_openmm", temperatures_K=temps)
    print(resremd.format_openmm_summary(run.summary()))
    out = run.weights(temperature_K=300.0, discard_fraction=0.1)
    # out["weights"][k][i]: frame out["first_frame"] + i of state_<k>

States are those the sampler was given, in its order. Frames can be
weighted to any of them, whatever they vary; for a run whose states differ
in temperature only, with ``temperatures_K``, to any temperature they
overlap. MBAR is for runs in one fixed box: a run at constant pressure, or
with replicas in boxes of their own, is summarised, but not weighted.
"""

from __future__ import annotations

import math
import operator
import re
import struct
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .errors import InputError
from .mbar import solve
from .mbar import weights as mbar_weights

#: The gas constant in kJ/(mol K) as OpenMM's Python layer has it
#: (`openmm.unit.MOLAR_GAS_CONSTANT_R`): the reporter divided by it.
GAS_CONSTANT = 0.00831446261815324

#: How far (relative) a replica's reduced energies may stray from one
#: energy at several temperatures and still be one: round-off.
SAME_ENERGY = 1e-9

#: DCDFile's time unit (AKMA) in ps.
_AKMA_PS = 0.04888821

#: The context parameters by which OpenMM's barostats hold their pressure
#: or surface tension.
_BAROSTAT = {"MonteCarloPressure", "MonteCarloPressureX",
             "MonteCarloPressureY", "MonteCarloPressureZ",
             "MembraneMonteCarloPressure", "MembraneMonteCarloSurfaceTension"}

_CODE = "resremd.input.openmm"

_STILL_GOING = ("If the run is still going, read it again in a moment; if "
                "it stopped there and was not resumed, delete that line.")


def is_openmm_run(path: str | Path) -> bool:
    """Whether a directory holds a log.csv, as OpenMM's
    ReplicaExchangeReporter writes, and no ResREMD run."""
    path = Path(path)
    return (path / "log.csv").is_file() and \
        not (path / "manifest.json").exists()


def _lines(path: Path) -> list[str]:
    """A file's complete lines, without blank ones at its end."""
    try:
        with open(path, encoding="utf-8-sig", newline=None) as fh:
            text = fh.read()
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(f"{path} could not be read: {exc}.",
                         code=_CODE) from None
    lines = text.split("\n")
    # What follows the last newline: nothing, or a line still being
    # written (OpenMM ends every line it writes).
    if lines[-1].strip():
        raise InputError(
            f"{path}, line {len(lines)}, is only partly written. "
            + _STILL_GOING, code=_CODE)
    lines.pop()
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def _rows(path: Path, width: int, kind: type, skip: int = 0) -> np.ndarray:
    """A CSV of ``width`` fields per line, each line checked."""
    lines = _lines(path)[skip:]
    if kind is float and lines and all(
            line.count(",") == width - 1 for line in lines):
        # All at once; line by line only to find a line that is wrong.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = np.loadtxt(lines, delimiter=",", ndmin=2,
                                  comments=None)
            if data.shape == (len(lines), width):
                return data
        except ValueError:
            pass
    out = []
    for number, line in enumerate(lines, start=skip + 1):
        fields = line.split(",")
        try:
            if len(fields) != width:
                raise ValueError
            out.append(np.array(fields, dtype=kind))
        except (ValueError, OverflowError):
            raise InputError(
                f"{path}, line {number}, is not {width} numbers; the file "
                "was changed after OpenMM wrote it, and only a copy from "
                "before can be read.", code=_CODE) from None
    if not out:
        return np.zeros((0, width), dtype=kind)
    return np.stack(out)


def _dcd(path: Path) -> tuple[int, int, float, np.ndarray | None]:
    """The whole frames, steps between frames and timestep (ps, or nan) of
    a DCD file written by OpenMM's DCDFile, and each frame's box (its three
    lengths and three cosines) if it has a box."""
    not_dcd = InputError(f"{path} is not a DCD file written by OpenMM; move "
                         "it aside to read the rest of the run.", code=_CODE)
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            head = fh.read(100)
            if len(head) < 100 or head[:8] != b"T\0\0\0CORD":
                raise not_dcd
            frames, _, interval = struct.unpack("<iii", head[8:20])
            dt_akma = struct.unpack("<f", head[44:48])[0]
            has_box = struct.unpack("<i", head[48:52])[0] == 1
            title = struct.unpack("<i", head[92:96])[0]
            if frames < 0 or not 0 <= title <= size:
                raise not_dcd
            fh.seek(100 + title)
            block = fh.read(12)
            if len(block) < 12:
                raise not_dcd
            n_atoms = struct.unpack("<iii", block)[1]
            if n_atoms < 1:
                raise not_dcd
            start = 112 + title
            frame = 3 * (8 + 4 * n_atoms) + (56 if has_box else 0)
            # A run stopped while writing a frame leaves the header ahead
            # of the file: only whole frames count.
            frames = min(frames, max(0, size - start) // frame)
            boxes = None
            if has_box:
                boxes = np.empty((frames, 6))
                for f in range(frames):
                    fh.seek(start + f * frame + 4)
                    box = struct.unpack("<6d", fh.read(48))
                    a, cg, b, cb, ca, c = box
                    with np.errstate(all="ignore"):
                        ok = all(math.isfinite(x) for x in box) and \
                            min(a, b, c) > 0 and 1 - ca * ca - cb * cb \
                            - cg * cg + 2 * ca * cb * cg > 0
                    if not ok:
                        raise InputError(
                            f"{path}, frame {f + 1}, has a box that is not "
                            "a box: the file was changed, or the run stopped "
                            "while writing a frame and was resumed. Move it "
                            "aside to read the rest of the run.", code=_CODE)
                    boxes[f] = box
    except OSError as exc:
        raise InputError(f"{path} could not be read: {exc}.",
                         code=_CODE) from None
    # The header holds the timestep as a 32-bit float.
    dt_ps = float(f"{dt_akma * _AKMA_PS:.7g}") \
        if math.isfinite(dt_akma) and dt_akma > 0 else math.nan
    return frames, interval, dt_ps, boxes


def _checkpoint(path: Path) -> dict[str, Any]:
    """A checkpoint State's step count, time (ps), box vectors (nm) and
    context parameters."""
    try:
        with open(path, encoding="utf-8") as fh:
            head = ""
            # The parameters come before the positions; read up to them.
            while "<Positions" not in head:
                chunk = fh.read(1 << 16)
                if not chunk:
                    break
                head += chunk
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(f"{path} could not be read: {exc}.",
                         code=_CODE) from None
    if "<State" not in head:
        raise InputError(f"{path} is not a State checkpoint; move the "
                         "checkpoints aside to read the rest of the run.",
                         code=_CODE)
    end = head.find("<Positions")
    found = re.search(r"<Parameters\b([^>]*)>", head[:max(end, 0)])
    if end < 0 or found is None:
        # The sampler saves every State with its parameters, then the
        # positions.
        raise InputError(
            f"{path} is not a checkpoint of OpenMM's sampler, or is only "
            "partly written; move the checkpoints aside to read the rest of "
            "the run.", code=_CODE)
    state = re.search(r"<State\b([^>]*)>", head)
    box = re.search(r"<PeriodicBoxVectors>(.*?)</PeriodicBoxVectors>",
                    head[:end], re.S)
    if state is None:
        raise InputError(f"{path} is not a State checkpoint; move the "
                         "checkpoints aside to read the rest of the run.",
                         code=_CODE)
    try:
        attributes = dict(re.findall(r'([\w.]+)="([^"]*)"', state.group(1)))
        vectors = tuple(float(x) for x in re.findall(
            r'[xyz]="([^"]*)"', box.group(1))) if box else ()
        if len(vectors) != 9 or not all(map(math.isfinite, vectors)) or \
                min(vectors[0], vectors[4], vectors[8]) <= 0:
            raise ValueError
        time_ps = float(attributes["time"])
        if not math.isfinite(time_ps) or time_ps < 0:
            raise ValueError
        return {
            "steps": int(attributes["stepCount"]),
            "time_ps": time_ps,
            "box": vectors,
            "parameters": {k: float(v) for k, v in re.findall(
                r'([\w.]+)="([^"]*)"', found.group(1))},
        }
    except (KeyError, ValueError):
        raise InputError(f"{path} has a step count, time, box or parameter "
                         "that is not a number; move the checkpoints aside "
                         "to read the rest of the run.", code=_CODE) from None


def _number(value: Any) -> float:
    """A float, or nan for what is not a number (refused by the caller)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _whole(value: Any, name: str) -> int:
    """A whole number, or refused."""
    try:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError
        return operator.index(value)
    except TypeError:
        raise InputError(f"`{name}` must be a whole number; got {value!r}.",
                         code=_CODE) from None


class OpenMMRun:
    """A directory written by OpenMM's ReplicaExchangeReporter.

    ``temperatures_K`` are the states' temperatures, in the sampler's
    order. For states that differ in temperature only, their ratios are
    checked against the energies, and frames can be weighted to any
    temperature; the first must be right, as the energies cannot tell. A
    resume that gave the states other temperatures is found when their
    ratios changed (each part is then weighted to its own states), not
    when all were scaled by one factor.

    MBAR here is for runs in one fixed box, shared by every replica.
    Whether the run was is read from the volumes (``volume.csv``), the DCD
    trajectories' boxes, or the checkpoints (their boxes, and a barostat
    among their parameters), when they are of the log's last row; when none
    of them tells, ``fixed_box=True`` says it was. A run at constant
    pressure, or with replicas in boxes of their own, is summarised, but
    not weighted.

    ``timestep_fs`` gives the time simulated, as that many fs per step of
    the sampler. DCD trajectories give it too, as the timestep they were
    begun with, unless the checkpoints are of another moment than the log's
    last row or their time is not their steps at that timestep (both noted
    in the summary); a timestep given must match a DCD's that is used.
    """

    def __init__(self, path: str | Path, *,
                 temperatures_K: list[float] | None = None,
                 fixed_box: bool = False,
                 timestep_fs: float | None = None) -> None:
        self.path = Path(path)
        log = self.path / "log.csv"
        if (self.path / "manifest.json").exists():
            raise InputError(
                f"{self.path} is a ResREMD run; `resremd.summarize` and "
                "`resremd.TemperatureReweighting` read it.", code=_CODE)
        if not is_openmm_run(self.path):
            raise InputError(
                f"{self.path} holds no log.csv of OpenMM's "
                "ReplicaExchangeReporter.", code=_CODE)
        lines = _lines(log)
        header = lines[0].split(",") if lines else []
        n = len(header) - 2
        if n < 1 or header != ["Iteration", "Step"] + [
                f"Replica_{i}_State" for i in range(n)]:
            raise InputError(
                f"{log} does not start with the header OpenMM's "
                "ReplicaExchangeReporter writes.", code=_CODE)
        table = _rows(log, n + 2, int, skip=1)
        if len(table) == 0:
            raise InputError(f"{log} has no iterations yet.", code=_CODE)
        #: Reported iteration numbers, one per row of every file.
        self.iterations = table[:, 0]
        self.steps = table[:, 1]
        #: The state of each replica, per row: state_of[row, replica].
        self.state_of = table[:, 2:]
        self.n_states = n
        wrong = np.any(np.sort(self.state_of, axis=1) != np.arange(n), axis=1)
        if np.any(wrong):
            raise InputError(
                f"{log}, line {int(np.flatnonzero(wrong)[0]) + 2}, does not "
                "give each replica its own state.", code=_CODE)
        if np.any(np.diff(self.iterations) <= 0):
            raise InputError(f"{log} does not count iterations upward; it "
                             "holds more than one run.", code=_CODE)
        #: The replica holding each state, per row: holder[row, state].
        self.holder = np.argsort(self.state_of, axis=1)
        rows = len(table)
        self.energies = None
        if (self.path / "energy.csv").exists():
            u = _rows(self.path / "energy.csv", n * n, float)
            self._same_rows(u, "energy.csv")
            #: Reduced energies: u[row, replica, state].
            self.energies = u.reshape(rows, n, n)
        #: Volumes (nm^3) from volume.csv: volumes[row, replica], or None.
        self.volumes = None
        if (self.path / "volume.csv").exists():
            v = _rows(self.path / "volume.csv", n, float)
            self._same_rows(v, "volume.csv")
            bad = ~np.all(np.isfinite(v) & (v > 0), axis=1)
            if np.any(bad):
                raise InputError(
                    f"{self.path / 'volume.csv'}, line "
                    f"{int(np.flatnonzero(bad)[0]) + 1}, has a volume that "
                    "is not a number above 0.", code=_CODE)
            self.volumes = v
        self._read_trajectories()
        self._read_checkpoints()
        if not isinstance(fixed_box, (bool, np.bool_)):
            raise InputError(f"`fixed_box` must be True or False; got "
                             f"{fixed_box!r}.", code=_CODE)
        self._ensemble(bool(fixed_box))
        #: Whether the states share one energy at different temperatures,
        #: u_0 = r_k u_k (None without finite energies).
        self.temperature_only = None
        #: Rows at which temperature states took other temperatures.
        self.state_changes: list[int] = []
        self._ratios = None
        if self.energies is not None:
            self._shares_one_energy()
        self.temperatures = None
        if temperatures_K is not None:
            self.temperatures = self._temperatures(temperatures_K)
        self.timestep_ps = self._dcd_timestep_ps
        #: Why the DCD header's timestep was not used, if it was not.
        self._timestep_doubt = None
        if self.timestep_ps is not None and \
                self._stale_checkpoints is not None:
            self._timestep_doubt = (
                "the checkpoints are of another moment than log.csv's last "
                "row (a resume, perhaps with another timestep, which a DCD "
                "header does not record; a report still being written; or "
                "replicas begun at different step counts)")
        elif self.timestep_ps is not None and self._checkpoints is not None:
            s = self._checkpoints[0]
            if s["steps"] > 0 and not abs(s["time_ps"] / s["steps"]
                                          - self.timestep_ps) \
                    <= 1e-4 * self.timestep_ps:
                self._timestep_doubt = (
                    "the checkpoints' time is not their steps at that "
                    "timestep (steps at another timestep before the sampler, "
                    "a step count reset without the time, or a timestep "
                    "changed on a resume)")
        if self._timestep_doubt is not None:
            self.timestep_ps = None
        if timestep_fs is not None:
            dt = _number(timestep_fs)
            if not math.isfinite(dt) or dt <= 0:
                raise InputError(f"`timestep_fs` must be a number above 0; "
                                 f"got {timestep_fs!r}.", code=_CODE)
            if self.timestep_ps is not None and \
                    abs(dt / 1000 - self.timestep_ps) > 1e-4 * dt / 1000:
                written = 1000 * self.timestep_ps
                raise InputError(
                    f"The trajectories were written with a {written:.6g} fs "
                    f"timestep, not {dt:g} fs.", code=_CODE)
            self.timestep_ps = dt / 1000

    def _same_rows(self, data: np.ndarray, name: str) -> None:
        if len(data) != len(self.iterations):
            raise InputError(
                f"{self.path / name} has {len(data)} rows and log.csv "
                f"{len(self.iterations)}: the run was read, or stopped, "
                "between writing them. If it is still going, read it again "
                "in a moment; if it stopped there and was not resumed, "
                "delete the last line of the longer file. If it was resumed "
                "since, its rows no longer line up.", code=_CODE)

    def _read_trajectories(self) -> None:
        """Frames, timestep and box changes from DCD trajectories."""
        self.trajectory_frames: dict[int, int] = {}
        #: Each state file's first box, and whether any file's boxes vary.
        self._dcd_boxes: set[tuple[float, ...]] = set()
        self._dcd_boxes_vary = False
        self._dcd_box_frames = 0
        self._dcd_with_box = 0
        timesteps = set()
        between = np.unique(np.diff(self.steps))
        spacing = int(between[0]) if len(between) == 1 and between[0] > 0 \
            else None
        for k in range(self.n_states):
            path = self.path / f"state_{k}.dcd"
            if not path.exists():
                continue
            frames, interval, dt_ps, boxes = _dcd(path)
            self.trajectory_frames[k] = frames
            # The header's timestep holds if its interval is the log's
            # steps between rows; DCDFile folds the interval into the
            # timestep once a file passes 2^31 steps, leaving 1.
            if not math.isfinite(dt_ps):
                timesteps.add(None)
            elif spacing is not None and interval == spacing:
                timesteps.add(dt_ps)
            elif spacing is not None and interval == 1:
                timesteps.add(dt_ps / spacing)
            elif spacing is None and interval > 1:
                timesteps.add(dt_ps)
            else:
                timesteps.add(None)
            if boxes is not None and len(boxes):
                self._dcd_boxes.add(tuple(boxes[0].tolist()))
                self._dcd_boxes_vary |= bool(np.any(boxes != boxes[0]))
                self._dcd_box_frames += len(boxes)
                self._dcd_with_box += 1
        # Trajectories that disagree on the timestep give none.
        self._dcd_timestep_ps = None
        if len(timesteps) == 1:
            self._dcd_timestep_ps = next(iter(timesteps))
        elif timesteps - {None}:
            values = sorted(timesteps - {None})
            if None not in timesteps and \
                    values[-1] - values[0] <= 1e-6 * values[-1]:
                self._dcd_timestep_ps = values[0]

    def _read_checkpoints(self) -> None:
        """The replicas' checkpoints, if every one is of the log's last
        row: their boxes, the barostat they name, and the time."""
        paths = [self.path / f"checkpoint_{i}.xml"
                 for i in range(self.n_states)]
        self._checkpoints = None
        self._stale_checkpoints = None
        self._barostat = set()
        present = [p for p in paths if p.exists()]
        states = [_checkpoint(p) for p in present]
        for s in states:
            self._barostat.update(_BAROSTAT & set(s["parameters"]))
        # Checkpoints of another moment (a report behind, or left by an
        # earlier part of the run) can show the box was not fixed, but not
        # that it was.
        if len(states) == self.n_states and all(
                s["steps"] == int(self.steps[-1]) for s in states):
            self._checkpoints = states
        elif states:
            self._stale_checkpoints = states

    def _ensemble(self, fixed_box: bool) -> None:
        """Whether every replica was in one fixed box: True, False, or None
        when nothing tells."""
        seen = []        # (source, whether its boxes are all one)
        if self.volumes is not None:
            seen.append(("volume.csv", np.ptp(self.volumes) == 0,
                         self.volumes.size))
        if self._dcd_box_frames:
            # For one fixed box, every state's trajectory; against it, any.
            every = self._dcd_with_box == self.n_states
            seen.append(("the DCD boxes", len(self._dcd_boxes) == 1
                         and not self._dcd_boxes_vary,
                         self._dcd_box_frames if every else 0))
        for states in (self._checkpoints, self._stale_checkpoints):
            if states is not None:
                same = len({s["box"] for s in states}) == 1
                # Stale ones count only against.
                seen.append(("the checkpoints' boxes", same,
                             len(states) if states is self._checkpoints
                             else 0))
        differ = list(dict.fromkeys(name for name, same, _ in seen
                                    if not same))
        self._boxes_differ = bool(differ)
        # One box over the run's rows, whatever parameters the context
        # holds: volume.csv, or every state's trajectory, over more than one
        # row. Every sample then has one volume, and P V, were there any,
        # would cancel from MBAR's weights.
        held = (self.volumes is not None and len(self.volumes) > 1) or (
            self._dcd_with_box == self.n_states
            and self._dcd_box_frames >= 2 * self.n_states)
        #: Barostat parameters in a context whose box never changed (a
        #: barostat disabled or removed), or that `fixed_box` overrode.
        self._idle_barostat = False
        if differ:
            if fixed_box:
                raise InputError(
                    f"`fixed_box` was given, but it is contradicted by "
                    f"{differ[0]}.",
                    code=_CODE)
            self.fixed_box = False
        elif self._barostat and not (held or fixed_box):
            # The parameters say a barostat may have moved the box, and
            # nothing over the run's rows says it did not.
            self.fixed_box = False
        elif fixed_box or self._barostat or any(
                count > 1 for _, _, count in seen):
            self._idle_barostat = bool(self._barostat)
            self.fixed_box = True
        else:
            self.fixed_box = None

    def _shares_one_energy(self) -> None:
        """Whether every replica's reduced energies are one energy over
        different kT: u_0 = r_k u_k with one r_k = T_k / T_0 per state."""
        u = self.energies.reshape(-1, self.n_states)
        u = u[np.all(np.isfinite(u), axis=1)]
        if len(u) == 0:
            return
        # Scaled to the largest, so that no product overflows; a reduced
        # energy of 1 becomes `floor`.
        scale = max(float(np.max(np.abs(u))), 1e-300)
        u = u / scale
        floor = 1.0 / scale
        ratios = [1.0]
        for k in range(1, self.n_states):
            norm = float(np.dot(u[:, k], u[:, k]))
            r = float(np.dot(u[:, 0], u[:, k]) / norm) if norm > 0 else 1.0
            if not r > 0 or np.any(np.abs(u[:, 0] - r * u[:, k])
                                   > SAME_ENERGY
                                   * np.maximum(np.abs(u[:, 0]), floor)):
                self.temperature_only = False
                self._find_changes(scale, floor)
                return
            ratios.append(r)
        self.temperature_only = True
        self._ratios = np.array(ratios)

    def _find_changes(self, scale: float, floor: float) -> None:
        """Rows at which temperature states took other temperatures: each
        row one energy over different kT, but not the same kT throughout
        (a run resumed with other states)."""
        u = self.energies / scale
        finite = np.all(np.isfinite(u), axis=(1, 2))
        ratios = np.ones((len(u), self.n_states))
        with np.errstate(all="ignore"):
            for k in range(1, self.n_states):
                a, b = u[:, :, 0], u[:, :, k]
                r = np.sum(a * b, axis=1) / np.sum(b * b, axis=1)
                fits = np.all(np.abs(a - r[:, None] * b) <= SAME_ENERGY
                              * np.maximum(np.abs(a), floor), axis=1)
                if np.any(finite & ~(fits & (r > 0))):
                    return
                ratios[:, k] = r
        rows = np.flatnonzero(finite)
        moved = np.any(np.abs(np.diff(ratios[rows], axis=0))
                       > 1e-6 * ratios[rows][1:], axis=1)
        self.state_changes = [int(rows[i + 1]) for i in np.flatnonzero(moved)]

    def _temperatures(self, values: list[float]) -> np.ndarray:
        try:
            t = np.array([_number(x) for x in values])
        except TypeError:
            raise InputError(f"`temperatures_K` must be numbers; got "
                             f"{values!r}.", code=_CODE) from None
        if self.state_changes:
            raise InputError(
                f"The states' temperatures changed at row "
                f"{self.state_changes[0] + 1} (the run was resumed with "
                "other states), so no one list fits them.", code=_CODE)
        if t.shape != (self.n_states,):
            raise InputError(
                f"{len(t)} temperatures for the run's {self.n_states} "
                "states.", code=_CODE)
        if np.any(~np.isfinite(t) | (t <= 0)):
            raise InputError(f"`temperatures_K` must be numbers above 0; "
                             f"got {values!r}.", code=_CODE)
        if self.temperature_only:
            implied = t[0] * self._ratios
            if np.any(np.abs(t - implied) > 1e-6 * t):
                raise InputError(
                    f"The states' energies are one energy at temperatures "
                    f"in the ratios of {np.round(implied, 6).tolist()} K "
                    f"(with state 0 at {t[0]:g} K), not at {t.tolist()} K. "
                    "Give the sampler's temperatures exactly, in its order; "
                    "if the states' Hamiltonians are scaled copies of one "
                    "another, leave them out and weight to a state.",
                    code=_CODE)
        return t

    # ----------------------------------------------------------- summary

    @property
    def report_interval(self) -> int | None:
        """Iterations between rows, if constant (None for one row)."""
        steps = np.unique(np.diff(self.iterations))
        return int(steps[0]) if len(steps) == 1 else None

    @property
    def steps_per_iteration(self) -> int | None:
        """MD steps per iteration, if the log's steps say so plainly."""
        if len(self.iterations) < 2:
            return None
        per = np.diff(self.steps) / np.diff(self.iterations)
        if np.all(per == per[0]) and per[0] > 0 and per[0] == int(per[0]):
            return int(per[0])
        return None

    def neighbour_acceptance(self) -> list[float | None] | None:
        """The chance that a swap between states k and k + 1 is accepted,
        averaged over the rows: the mean of min(1, exp(delta)) for the two
        replicas holding them, by the sampler's own rule (which leaves P V
        out at constant pressure). None without energy.csv, and for a pair
        without finite energies."""
        if self.energies is None:
            return None
        rows = np.arange(len(self.iterations))
        u = self.energies
        out: list[float | None] = []
        for k in range(self.n_states - 1):
            a, b = self.holder[:, k], self.holder[:, k + 1]
            with np.errstate(over="ignore", invalid="ignore"):
                delta = (u[rows, a, k] + u[rows, b, k + 1]
                         - u[rows, a, k + 1] - u[rows, b, k])
                p = np.exp(np.minimum(delta, 0.0))
            p = p[np.isfinite(p)]
            out.append(float(p.mean()) if p.size else None)
        return out

    def round_trips(self) -> tuple[int, list[int]]:
        """Trips first state -> last state -> first state, counted in the
        logged rows (a trip within one report interval is not seen), and
        their lengths in iterations."""
        trips, lengths = 0, []
        top = self.n_states - 1
        for r in range(self.n_states):
            start, reached = None, False
            for row, s in enumerate(self.state_of[:, r]):
                if s == 0:
                    if start is not None and reached:
                        trips += 1
                        lengths.append(int(self.iterations[row]
                                           - self.iterations[start]))
                    start, reached = row, False
                elif s == top and start is not None:
                    reached = True
        return trips, lengths

    def summary(self) -> dict[str, Any]:
        trips, lengths = self.round_trips()
        steps = self.steps_per_iteration
        last = int(self.iterations[-1])
        time_ns = None
        if steps is not None and self.timestep_ps is not None:
            time_ns = last * steps * self.timestep_ps / 1000
        visited = [len(set(self.state_of[:, r].tolist()))
                   for r in range(self.n_states)]
        notes = []
        if self.report_interval != 1:
            notes.append("Round trips are counted in the logged rows only; "
                         "one within a report interval is not seen.")
        moves = self.volumes is not None and \
            bool(np.any(np.ptp(self.volumes, axis=0) > 0))
        if self._idle_barostat:
            notes.append(
                "The checkpoints name barostat parameters, but the box did "
                "not change (a barostat disabled or removed).")
        elif self._barostat or moves:
            notes.append(
                "The System has a barostat; at constant pressure OpenMM's "
                "sampler (8.6) leaves P V out of its exchanges, so unless "
                "its states share one temperature and one pressure they are "
                "not sampled exactly.")
        if self._timestep_doubt is not None:
            notes.append(
                f"The DCD header's timestep is not used: "
                f"{self._timestep_doubt}. "
                + ("Give the timestep for the time simulated."
                   if self.timestep_ps is None else
                   f"The time simulated takes every step to be "
                   f"{1000 * self.timestep_ps:g} fs, as given."))
        if self.state_changes:
            notes.append(
                f"The states' temperatures changed at row "
                f"{self.state_changes[0] + 1} (the run was resumed with "
                "other states).")
        if self.fixed_box is False:
            notes.append("The replicas were not all in one fixed box; MBAR "
                         "here is for runs that were.")
        elif self.fixed_box is None:
            notes.append(
                "Whether the replicas were all in one fixed box cannot be "
                "told (no volume.csv, current checkpoints or DCD boxes).")
        return {
            "source": "openmm",
            "path": str(self.path),
            "n_states": self.n_states,
            "iterations": last,
            "rows": len(self.iterations),
            "report_interval_iterations": self.report_interval,
            "steps_per_iteration": steps,
            "time_ns_per_replica": time_ns,
            "temperatures_K": None if self.temperatures is None
            else self.temperatures.tolist(),
            "fixed_box": self.fixed_box,
            "neighbour_acceptance": self.neighbour_acceptance(),
            "round_trips": trips,
            "mean_round_trip_iterations": float(np.mean(lengths))
            if lengths else None,
            "replicas_that_visited_every_state":
                int(sum(v == self.n_states for v in visited)),
            "notes": notes,
        }

    # -------------------------------------------------------------- MBAR

    def reduced_energies(self, rows: slice = slice(None)) -> np.ndarray:
        """u[row, replica, state], as MBAR takes them for a run in one
        fixed box."""
        if self.energies is None:
            raise InputError(
                f"{self.path} has no energy.csv; MBAR needs the reporter's "
                "energy=True.", code=_CODE)
        if self.fixed_box is False:
            raise InputError(
                f"The replicas of {self.path} were not all in one fixed box "
                + ("(their boxes differ)" if self._boxes_differ else
                   "(the checkpoints name a barostat; give `fixed_box=True` "
                   "if it was disabled or removed)")
                + "; MBAR here is for runs that were.", code=_CODE)
        if self.fixed_box is None:
            raise InputError(
                f"Whether the replicas of {self.path} were all in one fixed "
                "box cannot be told: it has no volume.csv, current "
                "checkpoints or DCD boxes. Give `fixed_box=True` if they "
                "were.", code=_CODE)
        return self.energies[rows]

    def free_energies(self, first_row: int = 0,
                      last_row: int | None = None) -> np.ndarray:
        """Dimensionless free energies f_k (f_0 = 0) from every replica's
        energy in every state, over rows first_row..last_row."""
        u = self.reduced_energies(slice(first_row, last_row))
        if len(u) == 0:
            raise InputError("No rows to weight.", code=_CODE)
        inside = [c + 1 for c in self.state_changes
                  if first_row < c < first_row + len(u)]
        if inside:
            raise InputError(
                f"The states changed at row {inside[0]} (the run was "
                "resumed with other states); weight one part at a time, "
                "with `frames`.", code=_CODE)
        if not np.all(np.isfinite(u)):
            r = first_row + int(np.flatnonzero(
                ~np.all(np.isfinite(u), axis=(1, 2)))[0])
            raise InputError(
                f"energy.csv, row {r + 1}, has an energy that is not a "
                "finite number; leave that stretch out with `frames`.",
                code=_CODE)
        n = self.n_states
        u_kn = u.reshape(-1, n).T
        n_k = np.full(n, len(u))
        # First guess: each step k -> k + 1 by exponential averaging over
        # the samples in state k.
        holder = self.holder[first_row:last_row]
        rows = np.arange(len(u))
        guess = [0.0]
        with np.errstate(all="ignore"):
            for k in range(n - 1):
                d = u[rows, holder[:, k], k + 1] - u[rows, holder[:, k], k]
                m = np.max(-d)
                step = m + math.log(np.mean(np.exp(-d - m)))
                guess.append(guess[-1] - step if math.isfinite(step)
                             else guess[-1])
            try:
                return solve(u_kn, n_k, initial=np.array(guess))
            except RuntimeError:
                raise InputError(
                    f"MBAR did not converge over rows {first_row + 1} to "
                    f"{first_row + len(u)}: the states may not overlap.",
                    code=_CODE) from None

    def weights(self, *, temperature_K: float | None = None,
                state: int | None = None, states: list[int] | None = None,
                frames: tuple[int, int] | None = None,
                discard_fraction: float = 0.0) -> dict[str, Any]:
        """MBAR weights of the frames of ``states`` (by default all) at a
        target: one of the run's states, or for states that differ in
        temperature only, with ``temperatures_K``, any temperature they
        overlap.

        ``frames`` is the stretch (first, last) in rows of the log, one
        frame of each state_<k> trajectory per row; by default everything
        after ``discard_fraction`` of them. ``out["weights"][k][i]`` is the
        weight of frame ``out["first_frame"] + i`` of state_<k>. A DCD
        trajectory is checked to hold a frame for every row; an XTC one is
        not.
        """
        if (temperature_K is None) == (state is None):
            raise InputError("Give either `temperature_K` or `state`.",
                             code=_CODE)
        self.reduced_energies()  # says what is missing
        if self.temperature_only is None:
            raise InputError(
                f"{self.path / 'energy.csv'} has no row of finite "
                "energies.", code=_CODE)
        if state is not None:
            state = _whole(state, "state")
            if not 0 <= state < self.n_states:
                raise InputError(f"`state` must be among 0.."
                                 f"{self.n_states - 1}; got {state}.",
                                 code=_CODE)
        if temperature_K is not None:
            t = _number(temperature_K)
            if not math.isfinite(t) or t <= 0:
                raise InputError(f"`temperature_K` must be a number above "
                                 f"0; got {temperature_K!r}.", code=_CODE)
            if self.temperatures is None:
                raise InputError(
                    "Weighting to a temperature needs `temperatures_K`.",
                    code=_CODE)
            if not self.temperature_only:
                raise InputError(
                    "The states differ in more than temperature, so frames "
                    "are weighted to one of them, with `state`.", code=_CODE)
        n = len(self.iterations)
        if frames is not None and discard_fraction:
            raise InputError("Give `frames` or `discard_fraction`, not "
                             "both.", code=_CODE)
        if frames is not None:
            try:
                first, last = (_whole(x, "frames") for x in frames)
            except (TypeError, ValueError):
                raise InputError(f"`frames` must be (first, last); got "
                                 f"{frames!r}.", code=_CODE) from None
        else:
            d = _number(discard_fraction)
            if not 0 <= d < 1:
                raise InputError(f"`discard_fraction` must be at least 0 "
                                 f"and below 1; got {discard_fraction!r}.",
                                 code=_CODE)
            first, last = int(n * d), n
        if not 0 <= first < last <= n:
            raise InputError(f"No rows in {first}..{last} of {n}.",
                             code=_CODE)
        if states is None:
            states = list(range(self.n_states))
        else:
            try:
                states = [_whole(k, "states") for k in states]
            except TypeError:
                raise InputError(f"`states` must be a list of states; got "
                                 f"{states!r}.", code=_CODE) from None
        if not states or len(set(states)) != len(states) or any(
                not 0 <= k < self.n_states for k in states):
            raise InputError(f"`states` must be among 0..{self.n_states - 1}"
                             f", each once; got {states}.", code=_CODE)
        for k in states:
            got = self.trajectory_frames.get(k)
            if got is not None and got != n:
                raise InputError(
                    f"state_{k}.dcd has {got} frames for the {n} rows of "
                    "log.csv, so its frames cannot be matched to the rows. "
                    "If the run is still going, read it again in a moment; "
                    "else leave that state out with `states`.", code=_CODE)
        f_k = self.free_energies(first, last)
        u = self.reduced_energies(slice(first, last))
        holder = self.holder[first:last]
        rows = np.arange(last - first)
        # The samples: each chosen state's frames in turn.
        samples = np.concatenate([u[rows, holder[:, k], :] for k in states])
        if state is not None:
            target = samples[:, state]
        else:
            # Every state gives the same U; state 0's will do.
            energy = samples[:, 0] * GAS_CONSTANT * self.temperatures[0]
            target = energy / (GAS_CONSTANT * float(temperature_K))
        counts = np.zeros(self.n_states)
        counts[states] = last - first
        w = mbar_weights(samples.T, counts, f_k, target)
        per_state = np.split(w, len(states))
        out: dict[str, Any] = {
            "states": states,
            "first_frame": first,
            "weights": {k: per_state[i] for i, k in enumerate(states)},
            "free_energies": f_k.tolist(),
            "effective_samples": float(1.0 / np.sum(w * w)),
        }
        if state is not None:
            out["state"] = state
        else:
            out["temperature_K"] = float(temperature_K)
        return out


def format_openmm_summary(summary: dict[str, Any]) -> str:
    """:meth:`OpenMMRun.summary` as lines to read."""
    every = summary["report_interval_iterations"]
    steps = summary["steps_per_iteration"]
    n = summary["n_states"]
    its = summary["iterations"]
    head = (f"  {n} state{'s' if n != 1 else ''}, {its} "
            f"iteration{'s' if its != 1 else ''} ({summary['rows']} logged"
            + (f", every {every}" if every else "") + ")"
            + (f", {steps} steps each" if steps else ""))
    if summary["time_ns_per_replica"] is not None:
        head += f", {summary['time_ns_per_replica']:.3g} ns per replica"
    lines = [f"OpenMM ReplicaExchangeSampler run ({summary['path']})", head]
    if summary["temperatures_K"] is not None:
        lines.append("  temperatures (K): " + ", ".join(
            f"{t:.6g}" for t in summary["temperatures_K"]))
    acc = summary["neighbour_acceptance"]
    if acc is None:
        said = "not available (no energy.csv)"
    elif not acc:
        said = "none (one state)"
    else:
        said = ", ".join("n/a" if a is None else f"{a:.2f}" for a in acc)
    lines.append("  neighbour swap acceptance, from the energies: " + said)
    lines.append(
        f"  round trips first-last-first state: {summary['round_trips']}"
        + (f" (mean {summary['mean_round_trip_iterations']:.0f} iterations)"
           if summary["mean_round_trip_iterations"] else ""))
    lines.append(f"  replicas that visited every state: "
                 f"{summary['replicas_that_visited_every_state']}")
    lines += [f"  note: {n}" for n in summary["notes"]]
    return "\n".join(lines)
