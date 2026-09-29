"""Replicas, and the contexts that simulate them.

A slot is one OpenMM Context with its integrator, on one device. Each
replica is assigned to one slot for the whole run. When a slot holds one
replica, that replica never leaves the device: an exchange changes its
temperature, not its coordinates. When a slot holds several, they take
turns and their state is kept on the host in between.

Slots run in parallel threads. OpenMM releases the Python interpreter lock
while it integrates, so one thread per device is enough to keep several
GPUs busy without MPI.
"""

from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import ResRemdError
from .system import create_context, make_integrator, warn_if_no_gpu
from .thermo import Ensemble

logger = logging.getLogger("resremd")


@dataclass
class Replica:
    """One replica's identity and, when it is off the device, its state."""

    index: int
    positions: np.ndarray | None = None
    velocities: np.ndarray | None = None
    box: np.ndarray | None = None
    #: Factor to apply to the velocities before the next segment, set when
    #: an exchange moved the replica to a new temperature.
    velocity_scale: float = 1.0
    #: Coordinates (and box) to start the next segment from, with a seed for
    #: fresh velocities: a reservoir frame that was accepted, or the start.
    reset: tuple[np.ndarray, np.ndarray | None, int] | None = None
    #: -1 for the starting structure, else the reservoir frame this
    #: replica's coordinates descend from.
    origin: int = -1


@dataclass
class Segment:
    """What one stretch of dynamics produced."""

    potential_kjmol: float
    box: np.ndarray | None
    frame: np.ndarray | None = None
    #: REST2 only: A, B, C of U(s) = A s^2 + B s + C for the configuration.
    terms: tuple[float, float, float] | None = None


@dataclass
class Slot:
    context: Any
    integrator: Any
    platform: str
    device: int | None
    ensemble: Ensemble
    periodic: bool
    replicas: list[int] = field(default_factory=list)
    resident: int | None = None
    temperature: float | None = None
    #: REST2 scale the context's parameters are set to (None: no REST2).
    scale: float | None = None
    rest2: bool = False

    @property
    def exclusive(self) -> bool:
        return len(self.replicas) == 1

    # -- state transfer -----------------------------------------------------
    def set_temperature(self, temperature_K: float) -> None:
        if self.temperature == temperature_K:
            return
        from openmm import unit

        self.integrator.setTemperature(temperature_K * unit.kelvin)
        if self.ensemble.temperature_parameter:
            self.context.setParameter(self.ensemble.temperature_parameter,
                                      temperature_K)
        self.temperature = temperature_K

    def set_scale(self, s: float | None) -> None:
        if not self.rest2 or s is None or self.scale == s:
            return
        from .rest2 import set_scale

        set_scale(self.context, s)
        self.scale = s

    def _energy_now(self) -> float:
        return _kj(self.context.getState(getEnergy=True).getPotentialEnergy())

    def terms(self, own: float, own_energy: float) -> tuple[float, float,
                                                            float]:
        """A, B, C of the configuration in the context, from its energy at
        its own scale and at two others; the scale is restored."""
        from .rest2 import fit, probes

        scales, energies = [own], [own_energy]
        for p in probes(own):
            self.set_scale(p)
            scales.append(p)
            energies.append(self._energy_now())
        self.set_scale(own)
        return fit(scales, energies)

    def _set_box(self, box: np.ndarray | None) -> None:
        if self.periodic and box is not None:
            from openmm import Vec3

            self.context.setPeriodicBoxVectors(
                *(Vec3(*map(float, row)) for row in box))

    def load(self, replica: Replica) -> None:
        self._set_box(replica.box)
        self.context.setPositions(replica.positions)
        self.context.setVelocities(replica.velocities)
        self.resident = replica.index

    def store(self, replica: Replica) -> None:
        state = self.context.getState(getPositions=True, getVelocities=True)
        replica.positions = _nm(state.getPositions(asNumpy=True))
        replica.velocities = _nm(state.getVelocities(asNumpy=True))
        replica.box = _box(state) if self.periodic else None

    def inject(self, positions: np.ndarray, box: np.ndarray | None) -> None:
        """Place coordinates exactly as a reservoir frame is placed.

        Constraints are applied and virtual sites put where the atoms say.
        The same steps run when a frame's energy is computed and when a
        replica continues from it, so the energy in the exchange criterion
        is that of the state the replica actually starts from.
        """
        self._set_box(box)
        self.context.setPositions(positions)
        self.context.applyConstraints(self.integrator.getConstraintTolerance())
        self.context.computeVirtualSites()

    def energy(self, positions: np.ndarray, box: np.ndarray | None) -> float:
        """Potential energy of a frame as injected (at s = 1 with REST2).
        Leaves the slot empty."""
        self.inject(positions, box)
        self.resident = None
        self.set_scale(1.0)
        return self._energy_now()

    def frame_terms(self, positions: np.ndarray, box: np.ndarray | None
                    ) -> tuple[float, float, float]:
        """REST2 A, B, C of a frame as injected. Leaves the slot empty."""
        e1 = self.energy(positions, box)
        return self.terms(1.0, e1)

    # -- dynamics -----------------------------------------------------------
    def run(self, replica: Replica, temperature_K: float, steps: int, *,
            want_frame: bool, scale: float | None = None) -> Segment:
        ctx = self.context
        self.set_scale(scale)
        if replica.reset is not None:
            positions, box, seed = replica.reset
            self.inject(positions, box)
            self.set_temperature(temperature_K)
            ctx.setVelocitiesToTemperature(temperature_K, int(seed))
            replica.reset = None
            replica.velocity_scale = 1.0
            self.resident = replica.index
        elif self.resident != replica.index:
            if replica.velocity_scale != 1.0:
                replica.velocities = replica.velocities * replica.velocity_scale
                replica.velocity_scale = 1.0
            self.load(replica)
        elif replica.velocity_scale != 1.0:
            v = ctx.getState(getVelocities=True).getVelocities(asNumpy=True)
            ctx.setVelocities(v * replica.velocity_scale)
            replica.velocity_scale = 1.0
        self.set_temperature(temperature_K)
        if steps > 0:
            self.integrator.step(int(steps))
        if self.exclusive:
            state = ctx.getState(getEnergy=True)
        else:
            state = ctx.getState(getEnergy=True, getPositions=True,
                                 getVelocities=True)
            replica.positions = _nm(state.getPositions(asNumpy=True))
            replica.velocities = _nm(state.getVelocities(asNumpy=True))
            replica.box = _box(state) if self.periodic else None
        energy = _kj(state.getPotentialEnergy())
        if not math.isfinite(energy):
            raise ResRemdError(
                f"Replica {replica.index} at {temperature_K:g} K reached a "
                f"non-finite energy ({energy}). The integration blew up: a "
                "shorter timestep, a better-equilibrated start or a lower top "
                "temperature is usually the fix.",
                code="resremd.simulation.unstable")
        box = _box(state) if self.periodic else None
        frame = None
        if want_frame:
            framed = ctx.getState(getPositions=True,
                                  enforcePeriodicBox=self.periodic)
            frame = _nm(framed.getPositions(asNumpy=True))
        terms = self.terms(scale, energy) if self.rest2 else None
        return Segment(energy, box, frame, terms)


def _nm(quantity: Any) -> np.ndarray:
    return np.asarray(quantity._value, dtype=float)


def _kj(quantity: Any) -> float:
    return float(quantity._value)


def _box(state: Any) -> np.ndarray:
    return np.asarray(state.getPeriodicBoxVectors(asNumpy=True)._value,
                      dtype=float)


class Engine:
    """All slots, and the replicas assigned to them."""

    def __init__(self, *, system: Any, ensemble: Ensemble, n_replicas: int,
                 integrator: str, timestep_fs: float, friction_per_ps: float,
                 temperature_K: float, platform: str, precision: str,
                 devices: list[int] | None, contexts_per_device: int | None,
                 cpu_threads: int | None, seeds: np.random.Generator,
                 rest2: bool = False) -> None:
        self.periodic = bool(system.usesPeriodicBoundaryConditions())
        self.rest2 = rest2
        self.slots: list[Slot] = []
        device_list = list(devices) if devices else [None]

        def new_slot(platform_name: str, device: int | None,
                     threads: int | None) -> Slot:
            integ = make_integrator(integrator, temperature_K,
                                    friction_per_ps, timestep_fs,
                                    int(seeds.integers(1, 2**31 - 1)))
            ctx, used = create_context(system, integ, platform=platform_name,
                                       precision=precision, device=device,
                                       cpu_threads=threads)
            return Slot(ctx, integ, used, device, ensemble, self.periodic,
                        rest2=rest2)

        first = new_slot(platform, device_list[0], cpu_threads)
        self.platform = first.platform
        warn_if_no_gpu(platform, self.platform)
        gpu = self.platform in ("CUDA", "HIP", "OpenCL")
        if not gpu and devices:
            logger.warning("`devices` is ignored on the %s platform.",
                           self.platform)
            device_list = [None]
        per_device_replicas = math.ceil(n_replicas / len(device_list))
        wanted = contexts_per_device or (4 if gpu else 1)
        per_device = max(1, min(wanted, per_device_replicas))
        threads = cpu_threads
        if self.platform == "CPU" and per_device > 1 and not cpu_threads:
            threads = max(1, (os.cpu_count() or 1) // per_device)
            if threads != (os.cpu_count() or 1):
                # The first context was made with every core; remake it so
                # the contexts do not compete for the same ones.
                del first
                first = new_slot(self.platform, None, threads)
        # Contexts are spread over the devices in turn, and never more of
        # them than there are replicas: an empty one only holds memory.
        plan = [d for _ in range(per_device) for d in device_list][:n_replicas]
        self.slots.append(first)
        for device in plan[1:]:
            self.slots.append(new_slot(self.platform, device, threads))
        for r in range(n_replicas):
            self.slots[r % len(self.slots)].replicas.append(r)
        self.slot_of = {r: self.slots[r % len(self.slots)]
                        for r in range(n_replicas)}
        self._pool = ThreadPoolExecutor(max_workers=len(self.slots)) \
            if len(self.slots) > 1 else None

    def describe(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "contexts": len(self.slots),
            "devices": sorted({s.device for s in self.slots},
                              key=lambda d: -1 if d is None else d),
            "replicas_resident": all(s.exclusive for s in self.slots),
        }

    def run(self, replicas: list[Replica], temperatures: list[float],
            steps: int, want_frame: set[int],
            scales: list[float] | None = None) -> dict[int, Segment]:
        """Advance every replica by ``steps`` at its temperature (and, with
        REST2, its scale)."""
        def work(slot: Slot) -> dict[int, Segment]:
            return {r: slot.run(replicas[r], temperatures[r], steps,
                                want_frame=r in want_frame,
                                scale=None if scales is None else scales[r])
                    for r in slot.replicas}

        results: dict[int, Segment] = {}
        if self._pool is None:
            for slot in self.slots:
                results.update(work(slot))
        else:
            for part in self._pool.map(work, self.slots):
                results.update(part)
        return results

    def gather(self, replicas: list[Replica]) -> None:
        """Bring every resident replica's state to the host, for a checkpoint."""
        for slot in self.slots:
            if slot.exclusive and slot.resident is not None:
                slot.store(replicas[slot.replicas[0]])

    def evaluator(self):
        """A function giving the potential energy of coordinates, on slot 0."""
        return self.slots[0].energy

    def terms_evaluator(self):
        """A function giving the REST2 A, B, C of coordinates, on slot 0."""
        return self.slots[0].frame_terms

    def minimize(self, positions: np.ndarray, box: np.ndarray | None, *,
                 tolerance_kjmol_nm: float = 10.0, max_iterations: int = 0
                 ) -> np.ndarray:
        import openmm

        slot = self.slots[0]
        slot._set_box(box)
        slot.set_scale(1.0)
        slot.context.setPositions(positions)
        openmm.LocalEnergyMinimizer.minimize(slot.context, tolerance_kjmol_nm,
                                             max_iterations)
        slot.resident = None
        state = slot.context.getState(getPositions=True)
        return _nm(state.getPositions(asNumpy=True))

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
        for slot in self.slots:
            del slot.context
        self.slots = []
