"""Files a run writes, and cutting them back to a checkpoint on resume.

Every file is appended to as the run goes. A checkpoint records how long
each one was at that moment. A run that stops between checkpoints has
written past that point; resuming cuts every file back to the recorded
length first, so nothing sampled after the checkpoint appears twice.
"""

from __future__ import annotations

import csv
import io
import os
import struct
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .errors import ResumeError


class CsvLog:
    """An append-only CSV file with a header."""

    def __init__(self, path: Path, header: list[str], *,
                 truncate_to: int | None = None) -> None:
        self.path = Path(path)
        if truncate_to is None:
            self._fh = open(self.path, "w", newline="")
            self._fh.write(",".join(header) + "\n")
        else:
            _truncate(self.path, truncate_to)
            self._fh = open(self.path, "a", newline="")
        self._writer = csv.writer(self._fh, lineterminator="\n")

    def write(self, row: Iterable[Any]) -> None:
        self._writer.writerow([_fmt(x) for x in row])

    def flush(self) -> int:
        self._fh.flush()
        os.fsync(self._fh.fileno())
        return self._fh.tell()

    def close(self) -> None:
        self._fh.close()


def _fmt(x: Any) -> Any:
    if isinstance(x, (float, np.floating)):
        return repr(float(x))
    if isinstance(x, (bool, np.bool_)):
        return int(x)
    return x


def _truncate(path: Path, size: int) -> None:
    if not path.exists():
        raise ResumeError(f"{path} is missing; the run cannot be resumed "
                          "without it.", code="resremd.resume.files")
    if path.stat().st_size < size:
        raise ResumeError(
            f"{path} is shorter ({path.stat().st_size} bytes) than its "
            f"checkpoint recorded ({size} bytes); it was changed after the "
            "run stopped.", code="resremd.resume.files")
    with open(path, "r+b") as fh:
        fh.truncate(size)


class DcdTrajectory:
    """One DCD file written through OpenMM's DCDFile."""

    def __init__(self, path: Path, topology: Any, *, timestep_ps: float,
                 interval_steps: int, resume: tuple[int, int] | None = None
                 ) -> None:
        from openmm.app import DCDFile

        self.path = Path(path)
        self.frames = 0
        first, interval, dt_ps = interval_steps, interval_steps, timestep_ps
        if resume is None:
            self._fh = open(self.path, "w+b")
            append = False
        else:
            size, frames = resume
            _truncate(self.path, size)
            # DCDFile rescales its header once a trajectory passes 2^31
            # steps, so the header, not the settings, says how to go on.
            first, interval, dt_ps = _set_dcd_frame_count(self.path, frames)
            self._fh = open(self.path, "r+b")
            self.frames = frames
            append = True
        self._dcd = DCDFile(self._fh, topology, dt_ps, firstStep=first,
                            interval=interval, append=append)
        self._periodic = topology.getPeriodicBoxVectors() is not None

    def write(self, positions_nm: np.ndarray, box_nm: np.ndarray | None
              ) -> None:
        from openmm import Vec3, unit

        vectors = None
        if self._periodic and box_nm is not None:
            vectors = tuple(Vec3(*map(float, row)) for row in box_nm) \
                * unit.nanometer
        self._dcd.writeModel(positions_nm, periodicBoxVectors=vectors)
        self.frames += 1

    def flush(self) -> int:
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.seek(0, os.SEEK_END)
        return self._fh.tell()

    def close(self) -> None:
        self._fh.close()


#: DCDFile stores its timestep in AKMA time units of this many ps.
_AKMA_PS = 0.04888821


def _set_dcd_frame_count(path: Path, frames: int) -> tuple[int, int, float]:
    """Rewrite the header fields DCDFile keeps current as it appends.

    Offset 8 holds the number of frames and offset 20 the step of the last
    one; both ran past the checkpoint before the run stopped. The first
    step (offset 12), the interval (16) and the timestep (44) are read back
    and returned, to continue the file with.
    """
    with open(path, "r+b") as fh:
        head = fh.read(48)
        if len(head) < 48 or head[4:8] != b"CORD":
            raise ResumeError(f"{path} is not a DCD file.",
                              code="resremd.resume.files")
        first, interval = struct.unpack("<ii", head[12:20])
        dt_akma = struct.unpack("<f", head[44:48])[0]
        fh.seek(8)
        fh.write(struct.pack("<i", frames))
        fh.seek(20)
        fh.write(struct.pack("<i", first + max(frames - 1, 0) * interval))
    return first, interval, dt_akma * _AKMA_PS


def write_pdb(path: Path, topology: Any, positions_nm: np.ndarray) -> None:
    from openmm import app

    buffer = io.StringIO()
    app.PDBFile.writeFile(topology, positions_nm * 10.0, buffer, keepIds=True)
    Path(path).write_text(buffer.getvalue())
