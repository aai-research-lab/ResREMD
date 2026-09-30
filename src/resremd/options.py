"""Every setting, declared once.

The Python API, the command line, the run-file template and any program that
embeds this package (FastMDXplora wraps these as a block of its simulation
phase) all read the declarations below. A default, a list of choices or a
bound is written here and nowhere else.

``Option`` has the shape of FastMDXplora's ``Field`` on purpose: name, type,
default, help, example, choices, minimum, maximum. A caller can map one onto
the other field by field.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, replace
from typing import Any

from .errors import InputError


@dataclass(frozen=True)
class Option:
    """One setting.

    ``minimum`` and ``maximum`` are inclusive unless ``minimum_exclusive``
    says otherwise. They are only given where the limit is a fact about the
    quantity (a temperature is positive), never a preference.
    """

    name: str
    type: type | tuple[type, ...]
    default: Any
    help: str
    group: str
    example: Any = None
    choices: tuple[str, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    minimum_exclusive: bool = False
    required: bool = False
    #: Element type of a list setting, for the command line.
    items: type | None = None


@dataclass(frozen=True)
class Schema:
    """The settings one command accepts."""

    name: str
    description: str
    options: tuple[Option, ...]

    def get(self, name: str) -> Option | None:
        for option in self.options:
            if option.name == name:
                return option
        return None

    def names(self) -> tuple[str, ...]:
        return tuple(o.name for o in self.options)

    def defaults(self) -> dict[str, Any]:
        return {o.name: o.default for o in self.options}

    def groups(self) -> tuple[tuple[str, tuple[Option, ...]], ...]:
        """Options in their groups, in declaration order."""
        order: list[str] = []
        for option in self.options:
            if option.group not in order:
                order.append(option.group)
        return tuple((g, tuple(o for o in self.options if o.group == g))
                     for g in order)


# ---------------------------------------------------------------------------
# Declarations. Options shared by more than one command are declared once.
# ---------------------------------------------------------------------------

PREPARED = Option(
    "prepared", str, None,
    "Directory holding system.xml, state.xml and topology.pdb: an OpenMM "
    "System, a State with the starting positions and box, and the topology. "
    "This is what FastMDXplora's setup phase writes.",
    group="Input", example="setup/")

OUTPUT_RUN = Option(
    "output", str, "resremd_run",
    "Directory the run writes to. Created if absent.",
    group="Input")

OUTPUT_RESERVOIR = Option(
    "output", str, "reservoir",
    "Directory the reservoir is written to. Created if absent.",
    group="Input")

RESUME = Option(
    "resume", bool, False,
    "Continue from the last checkpoint in `output`. Asking for a longer run "
    "than the one recorded extends it; everything else must match.",
    group="Input")

TEMPERATURES = Option(
    "temperatures_K", list, None,
    "The replica temperatures in kelvin, lowest first. Given this, the "
    "ladder settings below are not used.",
    group="Temperatures", example=[300.0, 318.0, 337.0, 357.0], items=float)

TEMPERATURE_MIN = Option(
    "temperature_min_K", float, 300.0,
    "Lowest replica temperature: the one whose ensemble is usually wanted.",
    group="Temperatures", minimum=0.0, minimum_exclusive=True)

TEMPERATURE_MAX = Option(
    "temperature_max_K", float, None,
    "Highest replica temperature. Left out with a reservoir, the ladder is "
    "spaced geometrically from the lowest temperature to the reservoir's, "
    "and the reservoir takes the top rung: the arrangement of the GROMACS "
    "implementation, where the hottest replica became the reservoir.",
    group="Temperatures", minimum=0.0, minimum_exclusive=True, example=450.0)

N_REPLICAS = Option(
    "n_replicas", int, None,
    "Number of replicas on a geometric ladder. Not counting the reservoir.",
    group="Temperatures", minimum=2, example=8)

RESERVOIR = Option(
    "reservoir", str, None,
    "Reservoir directory, from `resremd reservoir generate` or `resremd "
    "reservoir import`. Without one this is ordinary temperature replica "
    "exchange, which is the control a reservoir run should be compared with.",
    group="Reservoir", example="reservoir/")

RESERVOIR_INTERVAL = Option(
    "reservoir_interval", int, 1,
    "Attempt an exchange between the hottest replica and the reservoir every "
    "this many exchange cycles.",
    group="Reservoir", minimum=1)

RESERVOIR_REWEIGHT = Option(
    "reservoir_reweight", bool, False,
    "Use a reservoir generated under another Hamiltonian (another force "
    "field, solvent model or mutation-free change of parameters, with the "
    "same atoms) by reweighting its frames to this one: each frame's weight "
    "is multiplied by exp(-[U_run - U_built] / kT_R), from its energy under "
    "this run's System and the one recorded when it was built. The run "
    "reports the effective number of frames left and refuses when too few "
    "remain. A non-Boltzmann reservoir needs no reweighting: its exchanges "
    "use only this run's energies, so it serves any Hamiltonian as it is.",
    group="Reservoir")

EXCHANGE_INTERVAL = Option(
    "exchange_interval_steps", int, 500,
    "MD steps between exchange attempts. 500 steps at 2 fs is 1 ps, as in "
    "`-replex 500` in GROMACS.",
    group="Exchanges", minimum=1)

DURATION = Option(
    "duration_ns", float, None,
    "Production length per replica, in nanoseconds. Give this or "
    "`production_steps`.",
    group="Run length", minimum=0.0, minimum_exclusive=True, example=100.0)

PRODUCTION_STEPS = Option(
    "production_steps", int, None,
    "Production length per replica, in MD steps. Give this or "
    "`duration_ns`.",
    group="Run length", minimum=1)

EQUILIBRATION = Option(
    "equilibration_ns", float, 0.1,
    "Dynamics each replica runs at its own temperature before exchanges "
    "begin. Nothing from it is written or analysed.",
    group="Run length", minimum=0.0)

MINIMIZE = Option(
    "minimize", bool, True,
    "Energy-minimise the starting structure once before anything runs.",
    group="Run length")

INTEGRATOR = Option(
    "integrator", str, "langevin_middle",
    "Langevin dynamics, which every replica needs: exchanges change a "
    "replica's temperature, and only a thermostatted integrator makes the "
    "new temperature hold.",
    group="Dynamics", choices=("langevin_middle", "langevin"))

TIMESTEP = Option(
    "timestep_fs", float, 2.0,
    "Integration timestep in femtoseconds.",
    group="Dynamics", minimum=0.0, minimum_exclusive=True)

FRICTION = Option(
    "friction_per_ps", float, 1.0,
    "Langevin friction coefficient in inverse picoseconds.",
    group="Dynamics", minimum=0.0, minimum_exclusive=True)

ENSEMBLE = Option(
    "ensemble", str, None,
    "`nvt` at constant volume, `npt` at constant pressure. Left out, the "
    "System decides: a barostat in it is used, `pressure_bar` adds one, and "
    "otherwise the volume is constant. `nvt` removes a barostat the System "
    "has. Constant volume is the usual choice for explicit-solvent "
    "temperature replica exchange, because water at the top of a ladder "
    "expands and can boil at 1 bar; start it from a box whose density was "
    "equilibrated at constant pressure at the lowest temperature.",
    group="Dynamics", choices=("nvt", "npt"))

PRESSURE = Option(
    "pressure_bar", float, None,
    "Pressure for a barostat added here, in bar, when the System has none. "
    "With `ensemble: npt` and no pressure given, 1 bar.",
    group="Dynamics", minimum=0.0, minimum_exclusive=True, example=1.0)

BAROSTAT_FREQUENCY = Option(
    "barostat_frequency", int, 25,
    "Steps between Monte Carlo volume moves, for a barostat added here.",
    group="Dynamics", minimum=1)

RANDOM_SEED = Option(
    "random_seed", int, None,
    "Seed for exchanges, reservoir draws, velocities and the integrators. "
    "Left out, one is chosen and recorded so the run can be repeated.",
    group="Dynamics", minimum=0)

PLATFORM = Option(
    "platform", str, "auto",
    "OpenMM platform. `auto` tries CUDA, HIP, OpenCL and then CPU, and uses "
    "the first that works.",
    group="Hardware", choices=("auto", "CUDA", "HIP", "OpenCL", "CPU",
                               "Reference"))

PRECISION = Option(
    "precision", str, "mixed",
    "Floating-point precision on a GPU platform.",
    group="Hardware", choices=("mixed", "single", "double"))

DEVICES = Option(
    "devices", list, None,
    "GPU indices to spread the replicas over, one thread per device, for "
    "example [0, 1]. Left out, one device.",
    group="Hardware", example=[0, 1], items=int)

DEVICE_INDEX = Option(
    "device_index", int, None,
    "GPU index to run on. Left out, the platform's default.",
    group="Hardware", minimum=0)

CONTEXTS_PER_DEVICE = Option(
    "contexts_per_device", int, None,
    "Simulations kept on each device at once. Replicas beyond this share a "
    "context and are swapped in and out of it. More contexts keep a small "
    "system's GPU busy and cost memory. Left out: up to 4 on a GPU, 1 on the "
    "CPU, whose one context already uses every core.",
    group="Hardware", minimum=1)

CPU_THREADS = Option(
    "cpu_threads", int, None,
    "Threads for the CPU platform. Left out, OpenMM's default.",
    group="Hardware", minimum=1)

TRAJECTORY_INTERVAL = Option(
    "trajectory_interval_steps", int, None,
    "MD steps between saved frames. A multiple of the exchange interval, "
    "since a frame belongs to the temperature its replica held while it was "
    "sampled. Left out, every 10 exchange cycles.",
    group="Output", minimum=1)

REST2 = Option(
    "rest2", bool, False,
    "Replica exchange with solute tempering (REST2) instead of temperature "
    "replica exchange. Every replica runs at the lowest temperature of the "
    "ladder; the others become the solute's effective temperatures, reached "
    "by scaling the solute's interactions (Wang, Friesner and Berne 2011). "
    "Far fewer replicas span a range in explicit solvent, since only the "
    "solute's energy grows with the system. Needs a NonbondedForce "
    "(explicit solvent or vacuum, not implicit-solvent GB).",
    group="Temperatures")

REST2_SELECTION = Option(
    "rest2_selection", str, "solute",
    "The atoms REST2 heats: `solute` (neither water nor monatomic ions), "
    "`not water`, or `all`.",
    group="Temperatures", choices=("all", "not water", "solute"))

REST2_ATOMS = Option(
    "rest2_atoms", list, None,
    "Atom indices REST2 heats, overriding `rest2_selection`: part of a "
    "protein, a ligand, a loop.",
    group="Temperatures", items=int)

GENERATE_REST2_T0 = Option(
    "rest2_run_temperature_K", float, None,
    "Generate the reservoir with REST2 scaling: simulate at this real "
    "temperature with the solute (`rest2_selection` or `rest2_atoms`) at the "
    "effective temperature `temperature_K`. The reservoir then serves a "
    "REST2 run whose lowest temperature this is.",
    group="Reservoir", minimum=0.0, minimum_exclusive=True)

SAVE_SELECTION = Option(
    "save_selection", str, "not water",
    "Atoms written to the trajectories: `all`, `not water`, or `solute` "
    "(neither water nor monatomic ions). Water is recognised by residue name.",
    group="Output", choices=("all", "not water", "solute"))

SAVE_ATOMS = Option(
    "save_atoms", list, None,
    "Atom indices to write, overriding `save_selection`. For a program that "
    "resolves its own selections.",
    group="Output", items=int)

SAVE_STATES = Option(
    "save_states", (str, list), "all",
    "Which temperatures get a trajectory: `all`, `lowest`, or a list of "
    "state indices where 0 is the lowest.",
    group="Output", example="lowest", items=int)

SAVE_REPLICAS = Option(
    "save_replica_trajectories", bool, False,
    "Also write one continuous trajectory per replica, following it through "
    "the temperatures. For kinetics and for checking how replicas move.",
    group="Output")

CHECKPOINT_INTERVAL = Option(
    "checkpoint_interval_steps", int, None,
    "MD steps between checkpoints. A multiple of the exchange interval. Left "
    "out, about every 0.5 ns of simulated time.",
    group="Output", minimum=1)

RESERVOIR_TEMPERATURE = Option(
    "temperature_K", float, None,
    "Temperature the reservoir is sampled at, in kelvin. Usually a rung "
    "above the hottest replica you intend to run.",
    group="Reservoir", minimum=0.0, minimum_exclusive=True, required=True,
    example=500.0)

FRAME_INTERVAL = Option(
    "frame_interval_steps", int, 5000,
    "MD steps between reservoir frames. Frames closer than the correlation "
    "time cost storage without adding independent structures; the build "
    "reports how many effectively independent frames it holds.",
    group="Reservoir", minimum=1)

RESERVOIR_DURATION = Option(
    "duration_ns", float, None,
    "Length of the reservoir simulation, in nanoseconds. The reservoir can "
    "only hold what this run visits, so it is the quantity that decides "
    "whether the replica exchange converges to the right answer.",
    group="Run length", minimum=0.0, minimum_exclusive=True, required=True,
    example=200.0)

RESERVOIR_EQUILIBRATION = Option(
    "equilibration_ns", float, 1.0,
    "Dynamics at the reservoir temperature before frames are kept.",
    group="Run length", minimum=0.0)

BIAS_TORSIONS = Option(
    "bias_torsions", list, None,
    "Static biases on torsions, to cross barriers that temperature alone "
    "does not. Each is a mapping: `atoms`, four atom indices; `energy`, an "
    "OpenMM expression in `theta` (radians) in kJ/mol; `parameters`, values "
    "for the other names in it. The frames are then reweighted by "
    "exp(V_bias / kT), which makes them a Boltzmann sample of the unbiased "
    "system at `temperature_K`, and the reservoir is written as weighted. "
    "To flatten a peptide bond's cis/trans barrier: energy `-k*sin(theta)^2` "
    "with k a little below the barrier height.",
    group="Reservoir", items=dict,
    example=[{"atoms": [4, 6, 8, 10], "energy": "-k*sin(theta)^2",
              "parameters": {"k": 60.0}}])

CONVERGENCE_TORSIONS = Option(
    "convergence_torsions", list, None,
    "Torsions (four atom indices each) whose distribution judges whether "
    "the reservoir has converged: the build compares the histograms of its "
    "first and second halves, with the bias weights if any, and reports the "
    "largest total variation distance. With `convergence_tv` it also stops "
    "there. Pick the torsions of the slow change: a proline omega, the "
    "backbone phi and psi of a turn.",
    group="Run length", items=list, example=[[4, 6, 8, 10]])

CONVERGENCE_TV = Option(
    "convergence_tv", float, None,
    "Stop the build once the halves of every `convergence_torsions` "
    "histogram agree within this total variation distance, at two checks "
    "in a row (checks come with each checkpoint, about every 500 ps), and "
    "every watched torsion has made `convergence_min_transitions` "
    "transitions; `duration_ns` is the most it may take. Agreeing halves "
    "are necessary, not sufficient: the transitions rule out a build stuck "
    "in its first state, but not one that never found a state at all. "
    "Building from two different starts and comparing them covers that.",
    group="Run length", minimum=0.0, maximum=1.0, minimum_exclusive=True,
    example=0.02)

CONVERGENCE_BINS = Option(
    "convergence_bins", int, 6,
    "Regions per torsion for the halves test: equal arcs, centred on 0 and "
    "180 degrees when their number is even (60 degrees each by default). "
    "Fewer regions need fewer frames for the same distance.",
    group="Run length", minimum=2)

CONVERGENCE_MIN_TRANSITIONS = Option(
    "convergence_min_transitions", int, 10,
    "Transitions that every watched torsion must have made before the build "
    "may stop. Basins are found from the torsion's own sampled free energy "
    "(reweighted, under a bias), as wells separated by barriers of at least "
    "2 kT, and a transition is a move from one basin's core to another's; "
    "motion within a basin, however wide, never counts. Halves that agree "
    "mean little for a torsion that never crossed a barrier: without this, "
    "a build stuck in its first basin would stop at once. A torsion with no "
    "such barrier (a fast one, or one stuck) keeps the build going to "
    "`duration_ns`. 0 turns the requirement off.",
    group="Run length", minimum=0)

TRAJECTORIES = Option(
    "trajectories", list, None,
    "Trajectory files to read frames from (any format MDTraj reads), in "
    "order. Amber reservoirs (NetCDF files from cpptraj `createreservoir`) "
    "bring their temperature and each frame's energy, which the run's "
    "Hamiltonian check then compares with its own.",
    group="Input", required=True, example=["hot_run.dcd"], items=str)

TOPOLOGY = Option(
    "topology", str, None,
    "PDB file with the same atoms, in the same order, as the system the "
    "reservoir will be used with. Needed for trajectory formats that carry "
    "no atoms (DCD, XTC, TRR, NetCDF) unless `prepared` is given.",
    group="Input", example="topology.pdb")

IMPORT_PREPARED = Option(
    "prepared", str, None,
    "The prepared system the reservoir is for. Its topology is then the "
    "reference: frames from files that carry their own atoms (PDB, mmCIF, "
    "GRO, MOL2, HDF5), for example docking poses or predicted models, are "
    "matched to it atom by atom by residue order and atom name, whatever "
    "order the file lists them in, and refused with the first atom that "
    "does not match. Needed for `minimize_steps`.",
    group="Input", example="setup/")

MINIMIZE_STEPS = Option(
    "minimize_steps", int, 0,
    "Energy-minimise every frame for at most this many steps with the "
    "prepared System before it is stored, to relax clashes in structures "
    "that did not come from this force field. Only for a `non_boltzmann` "
    "reservoir: a minimised set is no Boltzmann sample, but a non-Boltzmann "
    "reservoir needs none.",
    group="Reservoir", minimum=0)

KIND = Option(
    "kind", str, "boltzmann",
    "What the frames are a sample of. `boltzmann`: the equilibrium ensemble "
    "at `temperature_K`. `weighted`: any sampling, with `weights` that make it "
    "that ensemble (for example from umbrella sampling and MBAR). "
    "`non_boltzmann`: structures of equal weight covering configuration space "
    "uniformly, exchanged as if from infinite temperature (Roitberg et al. "
    "2007); constant volume only.",
    group="Reservoir", choices=("boltzmann", "weighted", "non_boltzmann"))

WEIGHTS = Option(
    "weights", str, None,
    "File with one non-negative weight per input frame, before `stride` is "
    "applied: .npy, or text with one number per line. Required for "
    "`weighted`.",
    group="Reservoir", example="weights.npy")

STRIDE = Option(
    "stride", int, 1,
    "Keep every this many input frames.",
    group="Reservoir", minimum=1)

CLUSTERINFO = Option(
    "clusterinfo", str, None,
    "For Amber reservoirs with cluster bins (cpptraj `createreservoir ... "
    "bin`): the `clusterinfo` file of the `clusterdihedral` run that made "
    "them. Each frame is then weighted so that its cluster carries the "
    "population listed there, giving a weighted reservoir. The frames must "
    "be a Boltzmann sample at the reservoir temperature within each "
    "cluster, as from a simulation there.",
    group="Reservoir", example="clusterinfo.dat")

SURFACE_TENSION = Option(
    "surface_tension_bar_nm", float, 0.0,
    "Surface tension the frames were sampled at, for a membrane system run "
    "with a membrane barostat. In bar nm, as OpenMM takes it.",
    group="Reservoir")


RUN = Schema(
    name="run",
    description="Temperature replica exchange, or REST2, coupled to a "
                "reservoir when one is given.",
    options=(
        PREPARED, OUTPUT_RUN, RESUME,
        TEMPERATURES, TEMPERATURE_MIN, TEMPERATURE_MAX, N_REPLICAS,
        REST2, REST2_SELECTION, REST2_ATOMS,
        RESERVOIR, RESERVOIR_INTERVAL, RESERVOIR_REWEIGHT,
        EXCHANGE_INTERVAL,
        DURATION, PRODUCTION_STEPS, EQUILIBRATION, MINIMIZE,
        INTEGRATOR, TIMESTEP, FRICTION, ENSEMBLE, PRESSURE,
        BAROSTAT_FREQUENCY, RANDOM_SEED,
        PLATFORM, PRECISION, DEVICES, CONTEXTS_PER_DEVICE, CPU_THREADS,
        TRAJECTORY_INTERVAL, SAVE_SELECTION, SAVE_ATOMS, SAVE_STATES,
        SAVE_REPLICAS, CHECKPOINT_INTERVAL,
    ),
)

GENERATE = Schema(
    name="reservoir generate",
    description="Build a reservoir by simulating at one high temperature, "
                "optionally under a known bias that is then reweighted away.",
    options=(
        PREPARED, OUTPUT_RESERVOIR, RESUME,
        RESERVOIR_TEMPERATURE, FRAME_INTERVAL, BIAS_TORSIONS,
        GENERATE_REST2_T0, REST2_SELECTION, REST2_ATOMS,
        RESERVOIR_DURATION, RESERVOIR_EQUILIBRATION, CONVERGENCE_TORSIONS,
        CONVERGENCE_TV, CONVERGENCE_BINS, CONVERGENCE_MIN_TRANSITIONS,
        MINIMIZE,
        INTEGRATOR, TIMESTEP, FRICTION, ENSEMBLE, PRESSURE,
        BAROSTAT_FREQUENCY, RANDOM_SEED,
        PLATFORM, PRECISION, DEVICE_INDEX, CPU_THREADS,
    ),
)

IMPORT = Schema(
    name="reservoir import",
    description="Build a reservoir from trajectories or structures that "
                "already exist.",
    options=(
        TRAJECTORIES, TOPOLOGY, IMPORT_PREPARED, OUTPUT_RESERVOIR,
        replace(RESERVOIR_TEMPERATURE, required=False,
                help=RESERVOIR_TEMPERATURE.help
                + " Required unless the kind is `non_boltzmann`, or the "
                  "files are Amber reservoirs, which record it."),
        KIND, WEIGHTS, CLUSTERINFO, STRIDE,
        replace(PRESSURE, group="Reservoir",
                help="Pressure the frames were sampled at, in bar. Required "
                     "when their boxes differ, and absent when they were "
                     "sampled at constant volume."),
        SURFACE_TENSION, MINIMIZE_STEPS,
        replace(PLATFORM, help="OpenMM platform for `minimize_steps`."),
    ),
)

CLUSTER_SOURCE = Option(
    "source", str, None,
    "The reservoir whose clusters are to be given populations: generated "
    "or imported, Boltzmann or weighted.",
    group="Input", required=True, example="reservoir/")

CLUSTER_LABELS = Option(
    "labels", (str, list), None,
    "A cluster label (an integer) for each of the source's frames: a list, "
    "a .npy file, or a text file with one per line. For Amber clusters, the "
    "cluster numbers `clusterdihedral` wrote per frame. Give this or "
    "`torsions`.",
    group="Clusters", example="labels.npy")

CLUSTER_TORSIONS = Option(
    "torsions", list, None,
    "Torsions (four atom indices each): each cell of a grid of `bins` arcs "
    "per torsion is a cluster, labelled by its index from 0. Give this or "
    "`labels`. For Amber `clusterdihedral` clusters, give its per-frame "
    "cluster numbers as `labels` instead.",
    group="Clusters", items=list, example=[[4, 6, 8, 10]])

CLUSTER_BINS = Option(
    "bins", int, 10,
    "Arcs per torsion for `torsions`. Cell k of a torsion is centred on "
    "-180 + k 360/bins degrees.",
    group="Clusters", minimum=2)

POPULATIONS = Option(
    "populations", (str, dict), None,
    "Each cluster's population at the reservoir temperature, by label: a "
    "mapping, a JSON file of one, or an Amber `clusterinfo` file. Left out, "
    "clusters keep their populations in the source, which only "
    "`representatives` can use.",
    group="Clusters", example="populations.json")

REPRESENTATIVES = Option(
    "representatives", bool, False,
    "Keep one frame per cluster, weighted by the cluster's population: a "
    "smaller reservoir, but an approximation, close only for narrow "
    "clusters. Without it every frame is kept and the clusters are "
    "reweighted, which is exact when the frames are a Boltzmann sample "
    "within each cluster.",
    group="Clusters")

CLUSTER = Schema(
    name="reservoir cluster",
    description="Give a reservoir's clusters populations: a weighted "
                "reservoir with every frame reweighted, or one "
                "representative per cluster.",
    options=(
        CLUSTER_SOURCE,
        replace(OUTPUT_RESERVOIR, default=None, required=True,
                example="clustered/"),
        CLUSTER_LABELS, CLUSTER_TORSIONS, CLUSTER_BINS, POPULATIONS,
        REPRESENTATIVES,
        replace(RANDOM_SEED, group="Clusters",
                help="Seed for picking representatives when clusters are "
                     "given as labels."),
    ),
)

SCHEMAS = {s.name: s for s in (RUN, GENERATE, IMPORT, CLUSTER)}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _type_name(t: type | tuple[type, ...]) -> str:
    if isinstance(t, tuple):
        return " or ".join(x.__name__ for x in t)
    return t.__name__


def _check_one(option: Option, value: Any) -> Any:
    """Check a value against its declaration and return it normalised."""
    if value is None:
        return None
    types = option.type if isinstance(option.type, tuple) else (option.type,)
    # bool is an int to Python and never a number to a person.
    if isinstance(value, bool) and bool not in types:
        raise InputError(
            f"`{option.name}` takes {_type_name(option.type)}, not "
            f"{value!r}.", code="resremd.input.type")
    if float in types and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if list in types and isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, types):
        raise InputError(
            f"`{option.name}` takes {_type_name(option.type)}, not "
            f"{type(value).__name__} ({value!r}).", code="resremd.input.type")
    if option.choices is not None and isinstance(value, str) \
            and value not in option.choices:
        raise InputError(
            f"`{option.name}` is one of {', '.join(option.choices)}; "
            f"{value!r} is not.", code="resremd.input.choice")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if option.minimum is not None:
            low = option.minimum
            if value < low or (option.minimum_exclusive and value == low):
                word = "above" if option.minimum_exclusive else "at least"
                raise InputError(
                    f"`{option.name}` must be {word} {low:g}; got {value!r}.",
                    code="resremd.input.range")
        if option.maximum is not None and value > option.maximum:
            raise InputError(
                f"`{option.name}` must be at most {option.maximum:g}; got "
                f"{value!r}.", code="resremd.input.range")
    return value


def resolve(schema: Schema, given: dict[str, Any] | None) -> dict[str, Any]:
    """Defaults overlaid with what was given, each value checked.

    An unknown name is refused, with the nearest known one suggested: a
    misspelt setting that is silently ignored runs the study the person did
    not ask for.
    """
    given = dict(given or {})
    names = schema.names()
    for key in given:
        if key not in names:
            close = difflib.get_close_matches(key, names, n=1)
            hint = f" Did you mean `{close[0]}`?" if close else ""
            raise InputError(
                f"`{key}` is not a setting of `{schema.name}`.{hint}",
                code="resremd.input.unknown")
    out = schema.defaults()
    for key, value in given.items():
        out[key] = _check_one(schema.get(key), value)
    for option in schema.options:
        if option.required and out[option.name] is None:
            raise InputError(f"`{option.name}` is required: {option.help}",
                             code="resremd.input.missing")
    return out
