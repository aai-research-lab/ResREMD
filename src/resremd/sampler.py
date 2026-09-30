"""Temperature replica exchange, or REST2, coupled to a reservoir when one
is given.

One exchange cycle:

1. Every replica runs ``exchange_interval_steps`` of Langevin dynamics at
   its current temperature.
2. Energies, states and the frames due are written. A frame belongs to the
   temperature its replica held while it was sampled.
3. Neighbouring temperatures attempt to swap, on the even or the odd pairs
   (chosen at random each cycle, which keeps detailed balance). A replica
   that moves has its velocities rescaled by sqrt(T_new / T_old).
4. Every ``reservoir_interval`` cycles, the replica at the top temperature
   attempts to exchange with a frame drawn from the reservoir. On
   acceptance it continues from that frame with velocities drawn at the top
   temperature; the reservoir is unchanged.

The criteria are in :mod:`resremd.acceptance`.

With REST2 (:mod:`resremd.rest2`) every replica runs at the lowest
temperature T0 and the states differ by the scale of the solute's
interactions instead. Each replica's energy is then known at every scale
from three evaluations, and exchanges use it:

    log alpha = -beta0 [U_a(x_j) + U_b(x_i) - U_a(x_i) - U_b(x_j)]

with no velocity rescaling. A reservoir exchange is an independence move
with the reservoir's own distribution as the proposal,

    log alpha = [u_top(x) - u_R(x)] - [u_top(y) - u_R(y)],

u the reduced potential of the top state and of the reservoir (its real
temperature and REST2 scale; zero for a non-Boltzmann reservoir). For a
temperature ladder with an unscaled reservoir this is the criterion above.
"""

from __future__ import annotations

import json
import logging
import math
import os
import secrets
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import __version__
from .acceptance import accept, log_acceptance, log_acceptance_reservoir
from .engine import Engine, Replica
from .errors import InputError, ResumeError
from .ladder import resolve as resolve_ladder
from .options import RUN, resolve as resolve_options
from .output import CsvLog, DcdTrajectory, write_pdb
from .reservoir import Reservoir, write_json
from .stopping import StopRequests
from .system import (Prepared, from_objects, load_prepared, select_atoms,
                     subset_topology, system_digest, topology_digest)
from .thermo import beta, box_volume_and_area, simulated_system

logger = logging.getLogger("resremd")

CITATIONS = [
    "Okur A, Roe DR, Cui G, Hornak V, Simmerling C. Improving convergence of "
    "replica-exchange simulations through coupling to a high-temperature "
    "structure reservoir. J. Chem. Theory Comput. 2007, 3, 557-568. "
    "doi:10.1021/ct600263e",
    "Roitberg AE, Okur A, Simmerling C. Coupling of replica exchange "
    "simulations to a non-Boltzmann structure reservoir. J. Phys. Chem. B "
    "2007, 111, 2415-2418. doi:10.1021/jp068335b",
    "Hsueh SCC, Aina A, Plotkin SS. Ensemble generation for linear and cyclic "
    "peptides using a reservoir replica exchange molecular dynamics "
    "implementation in GROMACS. J. Phys. Chem. B 2022, 126, 10384-10399. "
    "doi:10.1021/acs.jpcb.2c05470",
    "Sugita Y, Okamoto Y. Replica-exchange molecular dynamics method for "
    "protein folding. Chem. Phys. Lett. 1999, 314, 141-151. "
    "doi:10.1016/S0009-2614(99)01123-9",
    "Eastman P, et al. OpenMM 8: Molecular dynamics simulation with machine "
    "learning potentials. J. Phys. Chem. B 2024, 128, 109-116. "
    "doi:10.1021/acs.jpcb.3c06662",
]

REST2_CITATION = (
    "Wang L, Friesner RA, Berne BJ. Replica exchange with solute scaling: a "
    "more efficient version of replica exchange with solute tempering "
    "(REST2). J. Phys. Chem. B 2011, 115, 9431-9438. doi:10.1021/jp204407d")

#: Settings that decide what is sampled. A resumed run must match them.
_SCIENTIFIC = ("temperatures_K", "rest2", "exchange_interval_steps",
               "reservoir_interval", "reservoir_reweight",
               "reservoir_update", "integrator",
               "timestep_fs",
               "friction_per_ps", "trajectory_interval_steps", "save_states",
               "save_atoms", "save_replica_trajectories")


def _steps(value_ns: float | None, timestep_fs: float) -> int:
    return int(round(value_ns * 1e6 / timestep_fs)) if value_ns else 0


def plan(options: dict[str, Any], reservoir_temperature_K: float | None
         ) -> dict[str, Any]:
    """Resolve the settings into temperatures and step counts, or refuse."""
    temperatures = resolve_ladder(options, reservoir_temperature_K)
    interval = options["exchange_interval_steps"]
    dt = options["timestep_fs"]
    if (options["duration_ns"] is None) == (options["production_steps"] is None):
        raise InputError(
            "Give the production length as `duration_ns` or as "
            "`production_steps`, one of them.", code="resremd.input.length")
    steps = options["production_steps"] or _steps(options["duration_ns"], dt)
    if steps % interval:
        low = steps // interval * interval
        raise InputError(
            f"The production length ({steps} steps) is not a whole number of "
            f"exchange intervals ({interval} steps). {low} or "
            f"{low + interval} steps would be.", code="resremd.input.length")
    trajectory = options["trajectory_interval_steps"] or 10 * interval
    if trajectory % interval:
        raise InputError(
            f"`trajectory_interval_steps` ({trajectory}) must be a multiple of "
            f"the exchange interval ({interval}): a frame is assigned to the "
            "temperature its replica held while it was sampled, and that is "
            "only fixed between exchanges.", code="resremd.input.interval")
    checkpoint = options["checkpoint_interval_steps"]
    if checkpoint is None:
        per_cycle_ps = interval * dt / 1000.0
        checkpoint = max(1, round(500.0 / per_cycle_ps)) * interval
    if checkpoint % interval:
        raise InputError(
            f"`checkpoint_interval_steps` ({checkpoint}) must be a multiple "
            f"of the exchange interval ({interval}).",
            code="resremd.input.interval")
    n = len(temperatures)
    save = options["save_states"]
    if save == "all":
        save_states = list(range(n))
    elif save == "lowest":
        save_states = [0]
    elif isinstance(save, list):
        try:
            save_states = sorted({int(s) for s in save})
        except (TypeError, ValueError):
            raise InputError("`save_states` must be `all`, `lowest` or state "
                             "indices.", code="resremd.input.type")
        if not save_states or save_states[0] < 0 or save_states[-1] >= n:
            raise InputError(f"`save_states` indices run from 0 to {n - 1}.",
                             code="resremd.input.range")
    else:
        raise InputError("`save_states` must be `all`, `lowest` or a list of "
                         "state indices.", code="resremd.input.choice")
    if options["devices"] is not None:
        try:
            options["devices"] = [int(d) for d in options["devices"]]
        except (TypeError, ValueError):
            raise InputError("`devices` must be GPU indices.",
                             code="resremd.input.type")
    return {
        "temperatures_K": temperatures,
        "production_steps": steps,
        "cycles": steps // interval,
        "equilibration_steps": _steps(options["equilibration_ns"], dt),
        "trajectory_interval_steps": trajectory,
        "checkpoint_interval_steps": checkpoint,
        "save_states": save_states,
    }


def run(prepared: Prepared | str | Path | None = None, *,
        system: Any = None, topology: Any = None, positions: Any = None,
        box: Any = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        **settings: Any) -> dict[str, Any]:
    """Run temperature replica exchange, with a reservoir if one is given.

    The system is either a prepared directory (``prepared`` or the
    ``prepared`` setting) or OpenMM objects (``system``, ``topology``,
    ``positions`` and optionally ``box``). Every other keyword is a setting
    of :data:`resremd.options.RUN`. Returns the manifest, which is also
    written to ``manifest.json`` in the output directory.

    ``on_progress`` receives a dictionary after every logged cycle, for a
    caller that shows progress its own way.
    """
    if isinstance(prepared, (str, Path)):
        settings["prepared"] = str(prepared)
        prepared = None
    options = resolve_options(RUN, settings)
    if prepared is None:
        if system is not None:
            prepared = from_objects(system, topology, positions, box)
        elif options["prepared"]:
            prepared = load_prepared(options["prepared"])
        else:
            raise InputError(
                "No system: give `prepared` (a directory with system.xml, "
                "state.xml and topology.pdb) or the OpenMM objects.",
                code="resremd.input.prepared")
    return _Run(prepared, options, on_progress).execute()


class _Run:
    def __init__(self, prepared: Prepared, options: dict[str, Any],
                 on_progress: Callable[[dict[str, Any]], None] | None) -> None:
        self.prepared = prepared
        self.options = options
        self.on_progress = on_progress
        self.out = Path(options["output"])

    # -- setting up ---------------------------------------------------------
    def _prepare(self) -> None:
        o = self.options
        self.warnings: list[str] = []
        system = self.prepared.system
        # The digest is of the System as given, before any barostat is added
        # here, so the same inputs give the same cache key and fingerprint.
        self.system_sha256 = system_digest(system)
        self.topology_sha256 = topology_digest(self.prepared.topology)
        self.reservoir = Reservoir.open(o["reservoir"]) if o["reservoir"] \
            else None
        self.plan = plan(o, self.reservoir.temperature_K
                         if self.reservoir else None)
        self.temperatures = self.plan["temperatures_K"]
        self.betas = [beta(t) for t in self.temperatures]
        self.rest2 = None
        self.scales = None
        if o["rest2"]:
            from .rest2 import rest2_system, scale_of, solute_digest

            solute = select_atoms(self.prepared.topology,
                                  o["rest2_selection"], o["rest2_atoms"])
            system, info = rest2_system(system, solute)
            # What the replicas simulate, for keys of energies computed with
            # it: the scaled System changes with this package's version.
            info["system_sha256"] = system_digest(system)
            self.T0 = self.temperatures[0]
            self.beta0 = beta(self.T0)
            self.scales = [scale_of(self.T0, t) for t in self.temperatures]
            self.rest2 = {**info, "solute_sha256": solute_digest(solute),
                          "run_temperature_K": self.T0,
                          "scales": self.scales}
            logger.info("REST2: %d solute atoms at %g K, scales %s",
                        solute.size, self.T0,
                        ", ".join(f"{x:.3f}" for x in self.scales))
        if self.rest2 is not None and self.reservoir is not None and \
                not self.reservoir.rest2 and \
                solute.size < self.prepared.n_atoms and \
                self.reservoir.kind != "non_boltzmann":
            self.warnings.append(
                "The reservoir was sampled without REST2 scaling, so its "
                "exchanges with the REST2 top state weigh the whole "
                "system's energy, solvent included; with many solvent "
                "atoms almost none will be accepted. A reservoir generated "
                "with REST2 scaling (`rest2_run_temperature_K`) exchanges "
                "through the solute alone.")
        if self.reservoir is not None and self.reservoir.rest2:
            if self.rest2 is None:
                raise InputError(
                    "The reservoir was sampled with REST2 scaling, which "
                    "only a REST2 run (`rest2: true`) can use.",
                    code="resremd.reservoir.rest2")
            if self.reservoir.rest2.get("solute_sha256") != \
                    self.rest2["solute_sha256"]:
                raise InputError(
                    "The reservoir's REST2 solute is not this run's: the two "
                    "scale different atoms.", code="resremd.reservoir.rest2")
        system, ensemble = simulated_system(
            system, ensemble=o["ensemble"], pressure_bar=o["pressure_bar"],
            temperature_K=self.temperatures[0],
            frequency=o["barostat_frequency"])
        self.system = system
        self.ensemble = ensemble
        if o["reservoir_update"] != "none":
            if self.reservoir is None or self.rest2 is not None or \
                    self.reservoir.kind != "boltzmann" or \
                    ensemble.constant_pressure or o["reservoir_reweight"]:
                raise InputError(
                    "`reservoir_update` is for temperature REMD at constant "
                    "volume with a Boltzmann reservoir, not reweighted.",
                    code="resremd.input.reservoir_update")
            if o["resume"]:
                raise InputError(
                    "A run with `reservoir_update` cannot be resumed: the "
                    "updated reservoir is not saved.",
                    code="resremd.input.reservoir_update")
            self.warnings.append(
                f"The reservoir is updated during the run "
                f"(`reservoir_update: {o['reservoir_update']}`), for "
                "studying the method. The reservoir checks judge it as it "
                "was at the start, and frame numbers in the logs and origins "
                "are slots whose contents change.")
        if self.rest2 is not None and ensemble.constant_pressure and \
                self.rest2["dispersion_correction"] == "unscaled":
            self.warnings.append(
                "This OpenMM has no CustomVolumeForce (8.3 and later), so "
                "the dispersion correction is not scaled with the solute: "
                "the barostat of every REST2 state feels the unscaled one. "
                "Update OpenMM to scale it.")
        if self.reservoir is not None:
            self.warnings += self.reservoir.check_against(
                topology_sha256=self.topology_sha256,
                n_atoms=self.prepared.n_atoms,
                periodic=self.prepared.periodic, ensemble=ensemble,
                box=self.prepared.box,
                top_temperature_K=self.temperatures[-1])
        for w in self.warnings:
            logger.warning(w)
        self.save_atoms = select_atoms(self.prepared.topology,
                                       o["save_selection"], o["save_atoms"])
        self.save_topology = subset_topology(
            self.prepared.topology, self.save_atoms,
            self.prepared.box if self.prepared.periodic else None)
        self.fingerprint = {
            "system_sha256": self.system_sha256,
            "topology_sha256": self.topology_sha256,
            "ensemble": ensemble.as_dict(),
            "reservoir": (None if self.reservoir is None else {
                "kind": self.reservoir.kind,
                "temperature_K": self.reservoir.temperature_K,
                "frames_sha256": self.reservoir.content_digest()}),
            "rest2_solute": None if self.rest2 is None
            else self.rest2["solute_sha256"],
            # How the dispersion correction scales: it changes sampling
            # through the barostat, and at any volume the energies MBAR and
            # pilots read from rest2_terms.csv.
            "rest2_dispersion": self.rest2["dispersion_correction"]
            if self.rest2 is not None and
            self.rest2["dispersion_correction"] in ("scaled", "unscaled")
            else None,
            **{k: (self.temperatures if k == "temperatures_K" else
                   self.plan["trajectory_interval_steps"]
                   if k == "trajectory_interval_steps" else
                   self.plan["save_states"] if k == "save_states" else
                   self.save_atoms.tolist() if k == "save_atoms" else o[k])
               for k in _SCIENTIFIC},
        }

    # -- the run ------------------------------------------------------------
    def execute(self) -> dict[str, Any]:
        self._prepare()
        o = self.options
        n = len(self.temperatures)
        checkpoint_file = self.out / "checkpoint.npz"
        if o["resume"]:
            if not checkpoint_file.exists():
                raise ResumeError(
                    f"There is no checkpoint in {self.out} to resume from. "
                    "Start the run again without `resume`.",
                    code="resremd.resume.missing")
            saved = _load_checkpoint(checkpoint_file)
            self._check_resumable(saved["meta"])
            seed = saved["meta"]["seed"]
        else:
            _claim_output(self.out)
            saved = None
            seed = o["random_seed"]
            if seed is None:
                seed = secrets.randbelow(2**31 - 1)
        self.seed = int(seed)
        self.out.mkdir(parents=True, exist_ok=True)
        log_handler = _log_to(self.out / "run.log")
        self.exchange_rng = np.random.default_rng([self.seed, 0])
        self.seed_rng = np.random.default_rng([self.seed, 1])
        if saved is not None:
            self.exchange_rng.bit_generator.state = saved["meta"]["exchange_rng"]
            self.seed_rng.bit_generator.state = saved["meta"]["seed_rng"]

        self.engine = Engine(
            system=self.system, ensemble=self.ensemble, n_replicas=n,
            integrator=o["integrator"], timestep_fs=o["timestep_fs"],
            friction_per_ps=o["friction_per_ps"],
            temperature_K=self.temperatures[0], platform=o["platform"],
            precision=o["precision"], devices=o["devices"],
            contexts_per_device=o["contexts_per_device"],
            cpu_threads=o["cpu_threads"], seeds=self.seed_rng,
            rest2=self.rest2 is not None)
        logger.info("%d replicas, %s", n, ", ".join(
            f"{t:.2f}" for t in self.temperatures) + " K")
        logger.info("Engine: %s", self.engine.describe())
        # Signals are taken from here on, so a stop requested while reservoir
        # energies are computed or replicas equilibrate is honoured too.
        with StopRequests() as self.stop:
            try:
                self.cost = (saved["meta"].get("cost") if saved else None) \
                    or _new_cost()
                clock = time.time()
                self._reservoir_energies()
                self.cost["wall_seconds"]["reservoir_energies"] += \
                    time.time() - clock
                if saved is None:
                    try:
                        if self.stop.requested:
                            raise _StoppedEarly
                        self._start_fresh()
                    except _StoppedEarly:
                        logger.warning("Stopped before equilibration "
                                       "finished; there is nothing to "
                                       "resume. Start the run again.")
                        self.manifest = {"status": "stopped",
                                         "progress": None}
                        return self.manifest
                else:
                    self._start_from(saved)
                self._loop()
            finally:
                self._close_files()
                self.engine.close()
                log_handler()
        return self.manifest

    def _reservoir_energies(self) -> None:
        self.res_h = None
        self.res_store = None
        if self.reservoir is None:
            return
        import openmm

        logger.info("Reservoir: %s", self.reservoir.describe())
        rest2 = self.rest2 is not None
        energies = self.reservoir.energies(
            self.engine.terms_evaluator() if rest2
            else self.engine.evaluator(),
            key_fields={"system_sha256": self.system_sha256,
                        "platform": self.engine.platform,
                        "precision": self.options["precision"]
                        if self.engine.platform in ("CUDA", "HIP", "OpenCL")
                        else None,
                        "openmm": openmm.__version__,
                        **({"rest2": self.rest2["solute_sha256"],
                            "rest2_system": self.rest2["system_sha256"]}
                           if rest2 else {})},
            progress=logger.info, terms=rest2)
        pot = energies["potential_kjmol"]
        if rest2:
            from .rest2 import energy

            # The reservoir's own Hamiltonian is the one at its scale.
            pot = energy(energies["rest2_terms"], self.reservoir.scale)
        found = []
        # Reweight only when the Hamiltonians differ beyond precision noise;
        # otherwise the weights would only carry that noise.
        # A reservoir without build energies cannot be reweighted, and
        # reweight_to says so rather than letting the option do nothing.
        if self.options["reservoir_reweight"] and \
                self.reservoir.kind != "non_boltzmann" and (
                    not self.reservoir.has_build_energies()
                    or self.reservoir.hamiltonian_spread(pot)
                    > self.reservoir.warn_kt()):
            found += self.reservoir.reweight_to(pot)
            logger.info("Reservoir reweighted to this Hamiltonian: %.0f "
                        "effective frames of %d.", self.reservoir.reweighted,
                        self.reservoir.n_frames)
            np.save(self.out / "reservoir_weights.npy",
                    self.reservoir.weights)
        found += self.reservoir.check_hamiltonian(
            pot, self.system_sha256,
            reweighted=self.reservoir.reweighted is not None)
        for w in found:
            logger.warning(w)
            self.warnings.append(w)
        if rest2:
            from .rest2 import energy

            v, a = energies["volume_nm3"], energies["area_nm2"]
            u_top = self.beta0 * self.ensemble.enthalpy(
                energy(energies["rest2_terms"], self.scales[-1]), v, a)
            u_res = self.reservoir.beta * self.ensemble.enthalpy(pot, v, a)
            #: u_top(y) - u_R(y) for every frame; the exchange needs only it.
            self.res_delta = u_top - u_res
            self.res_h = None
            np.save(self.out / "reservoir_delta.npy", self.res_delta)
            return
        self.res_h = self.ensemble.enthalpy(energies["potential_kjmol"],
                                            energies["volume_nm3"],
                                            energies["area_nm2"])
        # Kept with the run: the reservoir check in `resremd summary` compares
        # these with the top replica's own energies.
        np.save(self.out / "reservoir_enthalpy_kjmol.npy", self.res_h)
        if self.options["reservoir_update"] != "none":
            # In memory, to be changed: positions, boxes and energies.
            self.res_store = {
                "positions": [np.array(self.reservoir.positions[k],
                                       dtype=float)
                              for k in range(self.reservoir.n_frames)],
                "box": [self.reservoir.frame(k)[1]
                        for k in range(self.reservoir.n_frames)],
                "h": list(map(float, self.res_h)),
                "replaced": 0, "added": 0}

    def _start_fresh(self) -> None:
        o = self.options
        n = len(self.temperatures)
        positions = self.prepared.positions
        if o["minimize"]:
            logger.info("Minimising the starting structure")
            positions = self.engine.minimize(positions, self.prepared.box)
        self.replicas = [Replica(r) for r in range(n)]
        for r in self.replicas:
            r.reset = (positions.copy(), None if self.prepared.box is None
                       else self.prepared.box.copy(),
                       int(self.seed_rng.integers(1, 2**31 - 1)))
        self.state_of = list(range(n))
        self.replica_at = list(range(n))
        self.cycle = 0
        self.pair_attempts = np.zeros(n - 1, dtype=int)
        self.pair_accepts = np.zeros(n - 1, dtype=int)
        self.res_attempts = 0
        self.res_accepts = 0
        self.res_frames: set[int] = set()
        eq = self.plan["equilibration_steps"]
        logger.info("Equilibrating each replica at its own temperature "
                    "(%d steps)", eq)
        t0 = time.time()
        # In pieces of 10 ps, so a stop request is answered promptly.
        piece = max(1, int(round(1e4 / o["timestep_fs"])))
        left = eq
        while True:  # at least once: zero steps still places every replica
            if self.stop.requested:
                raise _StoppedEarly
            step = min(piece, left)
            self.engine.run(self.replicas, self._temps(), step, set(),
                            self._scales())
            left -= step
            if left <= 0:
                break
        self.cost["wall_seconds"]["equilibration"] += time.time() - t0
        self.cost["md_steps"]["equilibration"] += eq * n
        logger.info("Equilibrated in %.0f s", time.time() - t0)
        write_pdb(self.out / "topology.pdb", self.save_topology,
                  positions[self.save_atoms])
        self._open_files(None)
        self.started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._checkpoint(status="running")

    def _start_from(self, saved: dict[str, Any]) -> None:
        meta = saved["meta"]
        n = len(self.temperatures)
        self.replicas = []
        for r in range(n):
            rep = Replica(r, positions=saved["positions"][r],
                          velocities=saved["velocities"][r],
                          box=None if saved["box"] is None else saved["box"][r],
                          origin=int(meta["origins"][r]))
            self.replicas.append(rep)
        for r in meta.get("pending_reset", []):
            rep = self.replicas[r]
            rep.reset = (rep.positions, rep.box,
                         int(self.seed_rng.integers(1, 2**31 - 1)))
        self.state_of = list(meta["state_of"])
        self.replica_at = [0] * n
        for r, s in enumerate(self.state_of):
            self.replica_at[s] = r
        self.cycle = int(meta["cycle"])
        self.pair_attempts = np.array(meta["pair_attempts"], dtype=int)
        self.pair_accepts = np.array(meta["pair_accepts"], dtype=int)
        self.res_attempts = int(meta["reservoir_attempts"])
        self.res_accepts = int(meta["reservoir_accepts"])
        self.res_frames = set(meta["reservoir_frames"])
        self.started = meta["started"]
        self._open_files(meta["files"])
        logger.info("Resumed at cycle %d of %d", self.cycle,
                    self.plan["cycles"])

    def _check_resumable(self, meta: dict[str, Any]) -> None:
        before = meta["fingerprint"]
        # Settings added since a run was started read as their defaults.
        added = {"rest2": False, "reservoir_reweight": False,
                 "rest2_solute": None, "rest2_dispersion": None,
                 "reservoir_update": "none"}
        differ = [k for k in self.fingerprint
                  if json.dumps(before.get(k, added.get(k)), sort_keys=True)
                  != json.dumps(self.fingerprint[k], sort_keys=True)]
        if differ:
            raise ResumeError(
                "The run cannot be resumed with different "
                + ", ".join(differ) + ". These decide what is sampled; "
                "start a new run in another directory to change them.",
                code="resremd.resume.mismatch")
        if self.plan["cycles"] < int(meta["cycle"]):
            raise ResumeError(
                f"The run has already done {meta['cycle']} cycles, more than "
                f"the {self.plan['cycles']} now asked for. A run can be "
                "extended, not shortened.", code="resremd.resume.shorter")

    def _temps(self) -> list[float]:
        if self.rest2 is not None:
            return [self.T0] * len(self.temperatures)
        return [self.temperatures[self.state_of[r]]
                for r in range(len(self.temperatures))]

    def _scales(self) -> list[float] | None:
        if self.rest2 is None:
            return None
        return [self.scales[self.state_of[r]]
                for r in range(len(self.temperatures))]

    # -- files --------------------------------------------------------------
    def _open_files(self, sizes: dict[str, Any] | None) -> None:
        n = len(self.temperatures)
        replicas = [f"r{r}" for r in range(n)]
        head = ["cycle", "time_ps"]
        self.files: dict[str, Any] = {}

        def csv(name: str, header: list[str]) -> CsvLog:
            log = CsvLog(self.out / name, header,
                         truncate_to=None if sizes is None else sizes[name])
            self.files[name] = log
            return log

        self.log_states = csv("states.csv", head + replicas)
        self.log_energy = csv("energies.csv", head + replicas)
        self.log_volume = csv("volumes.csv", head + replicas) \
            if self.ensemble.constant_pressure else None
        # With a surface tension the reduced potential needs the area too.
        self.log_area = csv("areas.csv", head + replicas) \
            if self.ensemble.surface_tension_bar_nm else None
        self.log_origin = csv("origins.csv", head + replicas) \
            if self.reservoir is not None else None
        self.log_reservoir = csv(
            "reservoir_exchanges.csv",
            head + ["replica", "frame"]
            + (["delta_replica", "delta_frame"] if self.rest2 is not None
               else ["h_replica_kjmol", "h_frame_kjmol"])
            + ["log_acceptance", "accepted"]) \
            if self.reservoir is not None else None
        # REST2: each replica's A, B, C every cycle, its energy at any scale.
        self.log_terms = csv(
            "rest2_terms.csv",
            head + [f"r{r}_{x}" for r in range(n) for x in "abc"]) \
            if self.rest2 is not None else None
        dt_ps = self.options["timestep_fs"] / 1000.0
        interval = self.plan["trajectory_interval_steps"]
        (self.out / "trajectories").mkdir(exist_ok=True)
        self.dcd_states: dict[int, DcdTrajectory] = {}
        for s in self.plan["save_states"]:
            name = f"trajectories/state_{s:03d}_{self.temperatures[s]:.2f}K.dcd"
            self.dcd_states[s] = self._dcd(name, sizes, dt_ps, interval)
        self.dcd_replicas: dict[int, DcdTrajectory] = {}
        if self.options["save_replica_trajectories"]:
            (self.out / "replicas").mkdir(exist_ok=True)
            for r in range(n):
                self.dcd_replicas[r] = self._dcd(
                    f"replicas/replica_{r:03d}.dcd", sizes, dt_ps, interval)

    def _dcd(self, name: str, sizes: dict[str, Any] | None, dt_ps: float,
             interval: int) -> DcdTrajectory:
        resume = None if sizes is None else tuple(sizes[name])
        traj = DcdTrajectory(self.out / name, self.save_topology,
                             timestep_ps=dt_ps, interval_steps=interval,
                             resume=resume)
        self.files[name] = traj
        return traj

    def _flush_files(self) -> dict[str, Any]:
        sizes: dict[str, Any] = {}
        for name, f in self.files.items():
            size = f.flush()
            sizes[name] = [size, f.frames] if isinstance(f, DcdTrajectory) \
                else size
        return sizes

    def _close_files(self) -> None:
        for f in getattr(self, "files", {}).values():
            try:
                f.close()
            except Exception:  # closing must not hide the error that got here
                pass

    # -- checkpoints and the record -----------------------------------------
    def _checkpoint(self, status: str) -> None:
        now = time.time()
        if getattr(self, "_production_clock", None) is not None:
            self.cost["wall_seconds"]["production"] += \
                now - self._production_clock
            self._production_clock = now
        self.cost["md_steps"]["production"] = (
            self.cycle * self.options["exchange_interval_steps"]
            * len(self.replicas))
        self.engine.gather(self.replicas)
        sizes = self._flush_files()
        meta = {
            "cycle": self.cycle,
            "state_of": self.state_of,
            "origins": [r.origin for r in self.replicas],
            "pair_attempts": self.pair_attempts.tolist(),
            "pair_accepts": self.pair_accepts.tolist(),
            "reservoir_attempts": self.res_attempts,
            "reservoir_accepts": self.res_accepts,
            "reservoir_frames": sorted(self.res_frames),
            "exchange_rng": self.exchange_rng.bit_generator.state,
            "seed_rng": self.seed_rng.bit_generator.state,
            "seed": self.seed,
            "files": sizes,
            "fingerprint": self.fingerprint,
            "started": self.started,
            "cost": self.cost,
        }
        arrays = {
            "positions": np.stack([r.positions for r in self.replicas]),
            "velocities": np.stack([r.velocities * r.velocity_scale
                                    for r in self.replicas]),
        }
        if self.engine.periodic:
            arrays["box"] = np.stack([r.box for r in self.replicas])
        # A replica still waiting to start from a reservoir frame is saved as
        # that frame, with velocities to be drawn on resume.
        pending = [r.index for r in self.replicas if r.reset is not None]
        for r in pending:
            pos, box, _ = self.replicas[r].reset
            arrays["positions"][r] = pos
            if box is not None:
                arrays["box"][r] = box
        meta["pending_reset"] = pending
        tmp = self.out / "checkpoint.tmp.npz"
        with open(tmp, "wb") as fh:
            np.savez(fh, meta=np.array(json.dumps(meta)), **arrays)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.out / "checkpoint.npz")
        self._write_manifest(status)

    def _write_manifest(self, status: str) -> None:
        o = self.options
        dt = o["timestep_fs"]
        interval = o["exchange_interval_steps"]
        rate = np.divide(self.pair_accepts, self.pair_attempts,
                         out=np.zeros(len(self.pair_attempts)),
                         where=self.pair_attempts > 0)
        files = {name: name for name in self.files}
        self.manifest = {
            "tool": "resremd",
            "version": __version__,
            "status": status,
            "method": ("REST2 " if self.rest2 is not None else "")
            + ("reservoir replica exchange molecular dynamics"
               if self.reservoir is not None else
               "replica exchange molecular dynamics"
               if self.rest2 is not None else
               "temperature replica exchange molecular dynamics"),
            "rest2": self.rest2,
            "started": self.started,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "system": {
                "source": self.prepared.source,
                "n_atoms": self.prepared.n_atoms,
                "periodic": self.prepared.periodic,
                "system_sha256": self.system_sha256,
                "topology_sha256": self.topology_sha256,
                "ensemble": self.ensemble.as_dict(),
                "saved_atoms": int(self.save_atoms.size),
            },
            "states": [
                {"index": s, "temperature_K": t,
                 "rest2_scale": None if self.scales is None
                 else self.scales[s],
                 "trajectory": (f"trajectories/state_{s:03d}_{t:.2f}K.dcd"
                                if s in self.dcd_states else None)}
                for s, t in enumerate(self.temperatures)],
            "reservoir": None if self.reservoir is None else {
                "path": str(self.reservoir.path.resolve()),
                "kind": self.reservoir.kind,
                "temperature_K": self.reservoir.temperature_K,
                "n_frames": self.reservoir.n_frames,
                "frames_sha256": self.reservoir.content_digest(),
                "interval_cycles": o["reservoir_interval"],
                "reweighted_effective_frames": self.reservoir.reweighted,
                "rest2": self.reservoir.rest2,
            },
            "progress": {
                "cycles_done": self.cycle,
                "cycles_target": self.plan["cycles"],
                "steps_per_cycle": interval,
                "time_ns_per_replica": self.cycle * interval * dt / 1e6,
            },
            "exchanges": {
                "neighbour_pairs": [
                    {"states": [s, s + 1],
                     "temperatures_K": self.temperatures[s:s + 2],
                     "attempts": int(self.pair_attempts[s]),
                     "accepted": int(self.pair_accepts[s]),
                     "acceptance": float(rate[s])}
                    for s in range(len(self.pair_attempts))],
                "reservoir": None if self.reservoir is None else {
                    "attempts": self.res_attempts,
                    "accepted": self.res_accepts,
                    "acceptance": (self.res_accepts / self.res_attempts
                                   if self.res_attempts else 0.0),
                    "distinct_frames_accepted": len(self.res_frames),
                    **({"update": {
                        "rule": self.options["reservoir_update"],
                        "frames_replaced": self.res_store["replaced"],
                        "frames_added": self.res_store["added"],
                        "frames_at_end": len(self.res_store["h"])}}
                       if getattr(self, "res_store", None) else {}),
                },
            },
            "engine": {**self.engine.describe(), "seed": self.seed},
            "cost": {
                **self.cost,
                "md_steps_total": sum(self.cost["md_steps"].values()),
                "note": "MD steps summed over replicas. Wall time is summed "
                        "over every session of the run, on the hardware in "
                        "`engine`. A reservoir's own cost is in its "
                        "reservoir.json.",
            },
            "settings": {**o, **self.plan},
            "warnings": self.warnings,
            "files": {"topology": "topology.pdb", "log": "run.log", **files},
            "citations": CITATIONS + ([REST2_CITATION]
                                      if self.rest2 is not None else []),
        }
        write_json(self.out / "manifest.json", self.manifest)

    # -- the cycle loop -----------------------------------------------------
    def _loop(self) -> None:
        o = self.options
        n = len(self.temperatures)
        interval = o["exchange_interval_steps"]
        dt = o["timestep_fs"]
        traj = self.plan["trajectory_interval_steps"]
        chk = self.plan["checkpoint_interval_steps"]
        target = self.plan["cycles"]
        saved_states = set(self.plan["save_states"])
        log_every = max(1, min(target // 20 or 1, round(100.0 / (interval * dt
                                                                  / 1000.0))))
        t_start = time.time()
        self._production_clock = t_start
        c_start = self.cycle
        status = "complete"
        stop = self.stop
        while self.cycle < target:
            if stop.requested:
                status = "stopped"
                break
            step = (self.cycle + 1) * interval
            time_ps = step * dt / 1000.0
            save = step % traj == 0
            want = set()
            if save:
                want = {self.replica_at[s] for s in saved_states}
                if self.dcd_replicas:
                    want = set(range(n))
            if o["reservoir_update"] != "none" and \
                    (self.cycle + 1) % o["reservoir_interval"] == 0:
                # The top replica after this cycle's neighbour swaps: the
                # one at the top now, or the one just below it.
                want |= {self.replica_at[n - 1], self.replica_at[max(0, n - 2)]}
            seg = self.engine.run(self.replicas, self._temps(), interval,
                                  want, self._scales())
            h = np.array([self.ensemble.enthalpy(
                seg[r].potential_kjmol,
                *_vol_area(seg[r].box)) for r in range(n)])
            terms = np.array([seg[r].terms for r in range(n)]) \
                if self.rest2 is not None else None
            head = [self.cycle + 1, time_ps]
            if terms is not None:
                self.log_terms.write(head + terms.ravel().tolist())
            self.log_states.write(head + self.state_of)
            self.log_energy.write(head + [seg[r].potential_kjmol
                                          for r in range(n)])
            if self.log_volume is not None:
                self.log_volume.write(head + [_vol_area(seg[r].box)[0]
                                              for r in range(n)])
            if self.log_area is not None:
                self.log_area.write(head + [_vol_area(seg[r].box)[1]
                                            for r in range(n)])
            if self.log_origin is not None:
                self.log_origin.write(head + [r.origin
                                              for r in self.replicas])
            if save:
                atoms = self.save_atoms
                for s in saved_states:
                    r = self.replica_at[s]
                    self.dcd_states[s].write(seg[r].frame[atoms],
                                             seg[r].box)
                for r, d in self.dcd_replicas.items():
                    d.write(seg[r].frame[atoms], seg[r].box)
            if terms is None:
                self._swap(h)
            else:
                self._swap_rest2(terms)
            self.cycle += 1
            if self.reservoir is not None and \
                    self.cycle % o["reservoir_interval"] == 0:
                if o["reservoir_update"] != "none":
                    self._reservoir_exchange_updated(h, seg, head)
                elif terms is None:
                    self._reservoir_exchange(h, head)
                else:
                    self._reservoir_exchange_rest2(terms, seg, head)
            if self.cycle % log_every == 0 or self.cycle == target:
                self._report(t_start, c_start, time_ps)
            if step % chk == 0 and self.cycle < target:
                self._checkpoint(status="running")
        self._checkpoint(status=status)
        logger.info("Run %s at cycle %d of %d. Output in %s", status,
                    self.cycle, target, self.out)

    def _swap(self, h: np.ndarray) -> None:
        n = len(self.temperatures)
        parity = int(self.exchange_rng.integers(2))
        for s in range(parity, n - 1, 2):
            i, j = self.replica_at[s], self.replica_at[s + 1]
            la = log_acceptance(self.betas[s], self.betas[s + 1], h[i], h[j])
            self.pair_attempts[s] += 1
            if accept(la, self.exchange_rng):
                self.pair_accepts[s] += 1
                self.replica_at[s], self.replica_at[s + 1] = j, i
                self.state_of[i], self.state_of[j] = s + 1, s
                t_lo, t_hi = self.temperatures[s], self.temperatures[s + 1]
                self.replicas[i].velocity_scale *= math.sqrt(t_hi / t_lo)
                self.replicas[j].velocity_scale *= math.sqrt(t_lo / t_hi)

    def _swap_rest2(self, terms: np.ndarray) -> None:
        from .rest2 import energy

        n = len(self.temperatures)
        parity = int(self.exchange_rng.integers(2))
        for s in range(parity, n - 1, 2):
            i, j = self.replica_at[s], self.replica_at[s + 1]
            sa, sb = self.scales[s], self.scales[s + 1]
            la = -self.beta0 * (energy(terms[j], sa) + energy(terms[i], sb)
                                - energy(terms[i], sa) - energy(terms[j], sb))
            self.pair_attempts[s] += 1
            if accept(la, self.exchange_rng):
                self.pair_accepts[s] += 1
                self.replica_at[s], self.replica_at[s + 1] = j, i
                self.state_of[i], self.state_of[j] = s + 1, s

    def _reservoir_exchange_rest2(self, terms: np.ndarray, seg: dict,
                                  head: list[Any]) -> None:
        from .rest2 import energy

        top = len(self.temperatures) - 1
        r = self.replica_at[top]
        k = self.reservoir.draw(self.exchange_rng)
        v, a = _vol_area(seg[r].box)
        u_top = self.beta0 * self.ensemble.enthalpy(
            energy(terms[r], self.scales[top]), v, a)
        u_res = self.reservoir.beta * self.ensemble.enthalpy(
            energy(terms[r], self.reservoir.scale), v, a)
        delta = float(u_top - u_res)
        la = delta - float(self.res_delta[k])
        ok = accept(la, self.exchange_rng)
        self.res_attempts += 1
        self.log_reservoir.write(head + [r, k, delta,
                                         float(self.res_delta[k]), la, ok])
        if ok:
            self.res_accepts += 1
            self.res_frames.add(k)
            pos, box = self.reservoir.frame(k)
            self.replicas[r].reset = (pos, box,
                                      int(self.seed_rng.integers(1, 2**31 - 1)))
            self.replicas[r].origin = k

    def _reservoir_exchange(self, h: np.ndarray, head: list[Any]) -> None:
        top = len(self.temperatures) - 1
        r = self.replica_at[top]
        k = self.reservoir.draw(self.exchange_rng)
        la = log_acceptance_reservoir(self.betas[top], float(h[r]),
                                      self.reservoir.beta,
                                      float(self.res_h[k]))
        ok = accept(la, self.exchange_rng)
        self.res_attempts += 1
        self.log_reservoir.write(head + [r, k, float(h[r]),
                                         float(self.res_h[k]), la, ok])
        if ok:
            self.res_accepts += 1
            self.res_frames.add(k)
            pos, box = self.reservoir.frame(k)
            self.replicas[r].reset = (pos, box,
                                      int(self.seed_rng.integers(1, 2**31 - 1)))
            self.replicas[r].origin = k

    def _reservoir_exchange_updated(self, h: np.ndarray, seg: dict,
                                    head: list[Any]) -> None:
        """A reservoir exchange whose outgoing configuration then changes
        the reservoir (`reservoir_update`)."""
        store = self.res_store
        top = len(self.temperatures) - 1
        r = self.replica_at[top]
        n = len(store["h"])
        k = int(min(self.exchange_rng.random() * n, n - 1))
        la = log_acceptance_reservoir(self.betas[top], float(h[r]),
                                      self.reservoir.beta, store["h"][k])
        ok = accept(la, self.exchange_rng)
        self.res_attempts += 1
        self.log_reservoir.write(head + [r, k, float(h[r]), store["h"][k],
                                         la, ok])
        if not ok:
            return
        self.res_accepts += 1
        self.res_frames.add(k)
        pos, box = store["positions"][k].copy(), store["box"][k]
        out = (np.array(seg[r].frame, dtype=float),
               None if seg[r].box is None else np.array(seg[r].box),
               float(h[r]))
        if self.options["reservoir_update"] == "swap":
            store["positions"][k], store["box"][k], store["h"][k] = out
            store["replaced"] += 1
        else:
            store["positions"].append(out[0])
            store["box"].append(out[1])
            store["h"].append(out[2])
            store["added"] += 1
        self.replicas[r].reset = (pos, None if box is None else box.copy(),
                                  int(self.seed_rng.integers(1, 2**31 - 1)))
        self.replicas[r].origin = k

    def _report(self, t_start: float, c_start: int, time_ps: float) -> None:
        o = self.options
        elapsed = max(time.time() - t_start, 1e-9)
        done = self.cycle - c_start
        ns = done * o["exchange_interval_steps"] * o["timestep_fs"] / 1e6
        speed = ns / elapsed * 86400.0
        rate = np.divide(self.pair_accepts, self.pair_attempts,
                         out=np.zeros(len(self.pair_attempts)),
                         where=self.pair_attempts > 0)
        info = {
            "cycle": self.cycle, "cycles_target": self.plan["cycles"],
            "time_ps": time_ps, "ns_per_day_per_replica": speed,
            "neighbour_acceptance": rate.tolist(),
            "reservoir_acceptance": (self.res_accepts / self.res_attempts
                                     if self.res_attempts else None),
        }
        text = (f"cycle {self.cycle}/{self.plan['cycles']}  "
                f"{time_ps / 1000:.3f} ns  {speed:.1f} ns/day/replica  "
                f"swaps {rate.min():.2f}-{rate.max():.2f}")
        if self.reservoir is not None:
            text += f"  reservoir {info['reservoir_acceptance'] or 0:.3f}"
        logger.info(text)
        if self.on_progress:
            self.on_progress(info)


class _StoppedEarly(Exception):
    """A stop was requested before the first checkpoint existed."""


def _new_cost() -> dict[str, Any]:
    """What a run has spent: MD steps over all replicas, and wall time."""
    return {"md_steps": {"equilibration": 0, "production": 0},
            "wall_seconds": {"reservoir_energies": 0.0, "equilibration": 0.0,
                             "production": 0.0}}


#: What a run writes before its first checkpoint. A directory holding only
#: these is what is left of a run that stopped before it had anything to
#: resume, and a fresh run may clear it.
_RUN_FILES = frozenset({
    "run.log", "topology.pdb", "states.csv", "energies.csv", "volumes.csv",
    "areas.csv", "origins.csv", "reservoir_exchanges.csv", "trajectories",
    "replicas", "checkpoint.tmp.npz", "reservoir_enthalpy_kjmol.npy",
    "reservoir_weights.npy", "reservoir_delta.npy", "rest2_terms.csv"})


def _claim_output(out: Path) -> None:
    """Make sure a fresh run can write here without destroying anything."""
    import shutil

    if not out.exists():
        return
    names = {p.name for p in out.iterdir()}
    if names & {"checkpoint.npz", "manifest.json"}:
        raise InputError(
            f"{out} holds a run. Resume it with `resume`, or choose another "
            "`output`.", code="resremd.input.output")
    other = sorted(names - _RUN_FILES)
    if other:
        raise InputError(
            f"{out} holds files this package did not write "
            f"({', '.join(other[:5])}). Choose another `output`.",
            code="resremd.input.output")
    for name in names:
        path = out / name
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _log_to(path: Path) -> Callable[[], None]:
    """Also log to a file in the run directory; returns the undo."""
    handler = logging.FileHandler(path)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"))
    previous_level, previous_propagate = logger.level, logger.propagate
    effective = logger.getEffectiveLevel()
    forward = None
    if effective > logging.INFO:
        # The file wants progress lines that the caller's logging does not.
        # Lower this logger's level for the file, and pass on to the
        # caller's handlers only what they asked for.
        class _Forward(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                if logger.parent is not None:
                    logger.parent.handle(record)

        forward = _Forward(level=effective)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.addHandler(forward)
    logger.addHandler(handler)

    def undo() -> None:
        logger.removeHandler(handler)
        handler.close()
        if forward is not None:
            logger.removeHandler(forward)
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate

    return undo


def _vol_area(box: np.ndarray | None) -> tuple[float, float]:
    return box_volume_and_area(box)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        with np.load(path) as data:
            meta = json.loads(str(data["meta"]))
            out = {"meta": meta,
                   "positions": np.array(data["positions"]),
                   "velocities": np.array(data["velocities"]),
                   "box": np.array(data["box"]) if "box" in data else None}
    except Exception as exc:
        raise ResumeError(f"The checkpoint {path} could not be read: {exc}",
                          code="resremd.resume.unreadable") from exc
    return out
