# ResREMD

Reservoir replica exchange molecular dynamics (Res-REMD) for OpenMM, on CPUs
and GPUs.

Temperature replica exchange (REMD) cannot converge at the temperature you
care about until its hottest replica has found every relevant state, and
that search is repeated by every replica for the length of the run.
Res-REMD separates the two: the search is done once, by a simulation at a
single high temperature, and its structures are stored in a reservoir. The
replica exchange then only has to anneal those structures down the
temperature ladder, which converges much faster. The hottest replica
periodically attempts to exchange with a structure drawn at random from the
reservoir, with a Metropolis criterion that keeps every temperature's
ensemble exact.

This package implements the method for OpenMM. Its GROMACS 4.6.7
counterpart is
[PlotkinLab/Reservoir-REMD](https://github.com/PlotkinLab/Reservoir-REMD)
(Hsueh, Aina and Plotkin, *J. Phys. Chem. B* 2022).

## Install

Python 3.10 or later. The only dependencies are OpenMM and NumPy.

```
conda create -n resremd -c conda-forge python=3.12 openmm numpy mdtraj
conda activate resremd
pip install git+https://github.com/aai-research-lab/ResREMD
```

MDTraj is only needed to import existing trajectories into a reservoir, and
PyYAML only to read YAML settings files.

## Quick start

The input is a prepared system: a directory holding `system.xml`,
`state.xml` and `topology.pdb`, as
[FastMDXplora](https://github.com/aai-research-lab/FastMDXplora)'s setup
phase writes them. `examples/peptide/prepare.py` makes one from a PDB file.

```
# 1. The reservoir: 200 ns at 450 K, a frame every 10 ps
resremd reservoir generate --prepared setup --temperature-K 450 \
    --duration-ns 200 --output reservoir

# 2. Replica exchange from 300 K, with the reservoir as the rung above the
#    hottest of 8 replicas
resremd run --prepared setup --reservoir reservoir \
    --temperature-min-K 300 --n-replicas 8 --duration-ns 50 --output remd

# 3. How the exchanges went
resremd summary remd
```

The trajectory at 300 K is `remd/trajectories/state_000_300.00K.dcd`, with
`remd/topology.pdb`.

From Python:

```python
import resremd

resremd.generate_reservoir("setup", output="reservoir", temperature_K=450,
                           duration_ns=200)
resremd.run("setup", output="remd", reservoir="reservoir",
            temperature_min_K=300, n_replicas=8, duration_ns=50)
print(resremd.format_summary(resremd.summarize("remd")))
```

`resremd options run` prints a settings file listing every option with its
default and meaning. Settings can be given as flags, in a `--config` file,
or as keyword arguments, and are the same in all three.

## The method

Each exchange cycle:

1. Every replica runs `exchange_interval_steps` of Langevin dynamics at its
   temperature.
2. Neighbouring temperatures attempt to swap, on the even or the odd pairs
   at random, with log α = (β_a − β_b)(h_i − h_j). A replica that changes
   temperature has its velocities scaled by √(T_new / T_old).
3. Every `reservoir_interval` cycles, the replica at the top temperature
   attempts to exchange with a reservoir frame drawn at random, with
   log α = (β_top − β_R)(h_replica − h_frame). On acceptance it continues from
   that frame with velocities drawn at the top temperature. The reservoir
   itself never changes: each draw is an independent sample of its
   distribution.

h is the potential energy at constant volume, and U + PV (minus γA for a
membrane under surface tension) at constant pressure.

### Reservoir kinds

| kind | frames are | β_R | notes |
|---|---|---|---|
| `boltzmann` | an equilibrium sample at `temperature_K` | 1/kT_R | the usual choice |
| `weighted` | any sampling, with weights making it Boltzmann at `temperature_K` | 1/kT_R | from `generate` with `bias_torsions`, or imported, e.g. umbrella sampling reweighted with MBAR |
| `non_boltzmann` | structures of equal weight covering configuration space | 0 | Roitberg et al. 2007; constant volume only |

Some barriers are too high for temperature to cross in any affordable
time: prolyl cis/trans isomerisation is the common one in peptides. A
reservoir can still carry both sides. `resremd reservoir generate` with
`bias_torsions` simulates under a known bias on chosen torsions (for example
`-k*sin(theta)^2`, which lowers a peptide bond's barrier by k) and
reweights every frame by exp(V_bias/kT_R), so the reservoir is a weighted
sample of the unbiased system. The build reports the Kish effective number
of frames, which falls as the bias grows.

A replica exchange run coupled to a reservoir is exactly as correct as the
reservoir. The build reports how many effectively independent frames the
reservoir holds, judged from the potential energy. Slow conformational
change decorrelates more slowly than the energy, so treat that number as an
upper bound, and not as proof that the reservoir is converged.

### REST2, with or without a reservoir

`rest2: true` replaces the temperature ladder with replica exchange with
solute tempering (REST2; Wang, Friesner and Berne 2011). Every replica runs
at the lowest temperature, and the rest of the ladder becomes the solute's
effective temperatures, reached by scaling the solute's interactions:
solute-solute nonbonded terms and torsions by T0/T_k, solute-solvent terms
by its square root. Only the solute's energy has to overlap between
neighbours, so far fewer replicas span a range in explicit solvent. The
solute is `rest2_selection` (by default everything but water and ions) or
`rest2_atoms`.

```
resremd run --prepared setup --rest2 --temperature-min-K 300 \
    --temperature-max-K 600 --n-replicas 6 --duration-ns 50 --output rest2
```

Each replica's energy is exactly quadratic in the scale, so three energy
evaluations per cycle give it at every state; exchanges, MBAR across the
states (`TemperatureReweighting`) and reservoir exchanges all use that. A
reservoir can be coupled to the top state as before. It can be one sampled
at a real temperature, as for temperature REMD, or one generated with the
same REST2 scaling (`resremd reservoir generate --rest2-run-temperature-K
300 --temperature-K 600`, the solute at 600 K and everything else at 300 K).
The exchange is the general independence move

    log alpha = [u_top(x) - u_R(x)] - [u_top(y) - u_R(y)],

with u the reduced potential of the top state and of the reservoir, which
for temperature REMD is the criterion above. REST2 needs a NonbondedForce:
explicit solvent or vacuum, not implicit-solvent GB.

### What is checked before a run starts

- The reservoir holds the system's atoms in the same order (by a digest of
  the topology), and was sampled in the same pressure ensemble, with the
  same kind of barostat.
- For a reservoir generated here, its frames' energies are recomputed under
  the run's System and compared with those recorded when it was built. A
  spread of more than 0.1 kT means another Hamiltonian (force field, cutoff,
  solvent model) and the run is refused; above 0.03 kT it is warned about.
  With `reservoir_reweight` the frames are reweighted to the run's
  Hamiltonian instead, and the run reports how many effective frames are
  left. A non-Boltzmann reservoir needs neither: its exchanges use only the
  run's own energies, so one reservoir serves any Hamiltonian of the same
  atoms, as Kasavajhala and Simmerling showed (JCTC 2023).
- At constant volume every reservoir frame has the run's box.
- Reservoir energies are computed with the run's own System, platform and
  precision, from exactly the coordinates that will be injected: constraints
  applied and virtual sites placed by OpenMM, the same steps a replica takes
  when it continues from the frame. They are cached, and a cache that does not
  match all of these is recomputed, never reused.
- The barostat, if any, is found by type, and each replica's copy is set to
  that replica's temperature through the parameter its class defines
  (`MonteCarloTemperature`, `MembraneMonteCarloTemperature`, ...).
- Forces that cannot be run at several temperatures (an Andersen thermostat,
  Drude polarisation) are refused.

## Choosing settings

- **Ladder.** Geometric spacing is the default. Aim for 20 to 40 percent
  acceptance between neighbours; `resremd summary` reports it per pair. Give
  `temperatures_K` to set the ladder by hand. To tune one, run a short pilot
  over the range and let its energies place the rungs:
  ```
  resremd ladder --from-pilot pilot --target-acceptance 0.3
  ```
  This prints the fewest temperatures that give every neighbour pair the
  target, with the acceptance it predicts for each, and for the pilot's own
  ladder the predicted against the observed acceptance as a check. It
  cannot place rungs outside the pilot's range. A REST2 pilot gives a REST2
  ladder, in effective temperatures from the pilot's run temperature; on
  the torsion model its predictions match the acceptance then observed on
  the tuned ladder within 0.03.
- **Reservoir temperature.** Hot enough that barriers are crossed readily in
  the reservoir simulation, and close enough to the top replica that frames
  are accepted. Leaving `temperature_max_K` out places the reservoir one
  geometric rung above the hottest replica, as in the GROMACS
  implementation.
- **Reservoir length.** The reservoir's own simulation is usually most of
  the cost, and it only needs to be long enough to have converged at its
  temperature. Name the torsions of the slow change and a tolerance, and
  the build stops once the populations of its two halves agree and every
  watched torsion has crossed between basins (wells separated by at least
  2 kT in its own sampled free energy) at least
  `convergence_min_transitions` times (10 by default), with `duration_ns`
  as the most it may take:
  ```
  resremd reservoir generate --prepared setup --temperature-K 450 \
      --duration-ns 200 --convergence-torsions "[4, 6, 8, 10]" \
      --convergence-tv 0.02 --output reservoir
  ```
  The history of the test, and each torsion's transitions, are in
  `reservoir.json`. The transitions stop a build stuck in its first basin
  from passing as converged; such a build runs to its full length and
  says so. Watch torsions with real barriers (a proline omega, a backbone
  phi): one without a 2 kT barrier never counts a transition and keeps
  the build going. A state the build never found at all is beyond any test of the
  build itself: build two reservoirs from different starts and compare
  them.
- **Exchange interval.** 500 steps (1 ps at 2 fs) by default, as
  `-replex 500` in GROMACS.
- **Ensemble.** Constant volume is usual for explicit-solvent temperature
  REMD: at 1 bar, water at the top of a ladder expands and can boil. Start
  from a box equilibrated at constant pressure at the lowest temperature, and
  generate the reservoir from the same prepared system so it has the same
  box. `ensemble: nvt` removes a barostat the System carries; `ensemble: npt`
  or `pressure_bar` runs at constant pressure.
- **Imported frames.** XTC, GRO and PDB files round coordinates to 0.001 nm,
  which stretches bonds enough to take frames out of the Boltzmann
  distribution. Import a Boltzmann or weighted reservoir from DCD, TRR or
  NetCDF.
- **Reservoirs of any structures.** A non-Boltzmann reservoir can hold
  structures from anywhere: docking poses, predicted models, a
  coarse-grained search. Its exchanges use only the energies of the
  coordinates as stored, so rounding does no harm, and it serves any
  Hamiltonian of the same atoms. Give the prepared system, and structures
  in PDB, mmCIF, GRO or MOL2 are matched to it atom by atom by residue
  order and name, whatever order they list atoms in:
  ```
  resremd reservoir import --kind non_boltzmann --prepared setup \
      --trajectories poses/*.pdb --minimize-steps 500 --output poses_reservoir
  ```
  `--minimize-steps` relaxes clashes with the run's force field first.
  Structures of a solvated system need the same solvent atoms; in practice
  this is for implicit solvent or vacuum.

## Output

| file | contents |
|---|---|
| `trajectories/state_NNN_TTT.TTK.dcd` | frames at each temperature |
| `topology.pdb` | topology of the saved atoms |
| `states.csv` | the temperature index each replica held, every cycle |
| `energies.csv` | each replica's potential energy (kJ/mol), every cycle |
| `volumes.csv` | each replica's volume (nm³), at constant pressure |
| `areas.csv` | each replica's xy area (nm²), with a surface tension |
| `reservoir_exchanges.csv` | every reservoir attempt: frame, enthalpies, log α, outcome |
| `origins.csv` | the reservoir frame each replica's coordinates descend from (−1: the start) |
| `manifest.json` | settings, system digests, acceptance, reservoir provenance, citations |
| `run.log` | the log |
| `checkpoint.npz` | the last checkpoint |

The energies, states, and at constant pressure the volumes (and areas), are
what MBAR needs to combine all temperatures. `resremd.TemperatureReweighting`
does it (Shirts and Chodera 2008, no pymbar needed): it weights every saved
frame, from every temperature, to any temperature the ladder overlaps.

```python
rw = resremd.TemperatureReweighting("remd")
out = rw.weights(300.0, discard_fraction=0.1)
# out["weights"][k][i]: frame out["first_frame"] + i of state k's trajectory
```

## Stopping and resuming

`SIGINT` or `SIGTERM` (what a batch scheduler sends before its time limit)
stops the run at the end of the current cycle, with a checkpoint.
`--resume` continues it. A longer duration extends a finished run.
Everything that decides what is sampled must match, and anything written
after the last checkpoint is cut away before the run continues, so nothing
is counted twice. Reservoir builds stop and resume the same way.

## Hardware

- `platform`: `auto` tries CUDA, HIP, OpenCL and then CPU, and warns, with
  OpenMM's plugin load failures, if it ends up on a CPU.
- `devices`: GPU indices, for example `--devices 0 1`. Replicas are spread
  over them, one thread per device. No MPI is needed.
- `contexts_per_device`: simulations kept on each device at once. With as
  many contexts as replicas, replicas never leave the GPU and an exchange
  changes only their temperature. With fewer, replicas share contexts and
  are swapped in and out. The default is up to 4 on a GPU and 1 on the CPU.

## Validation

The test suite compares runs against distributions known exactly: a
particle in an asymmetric double well, with a reservoir drawn exactly from
the Boltzmann distribution or uniformly. Well populations and the
temperature of every state must match the exact values within statistical
error, for both reservoir kinds and for both ways of assigning replicas to
contexts. `examples/double_well/compare.py` shows the same system converging
with a reservoir while plain REMD of the same length does not.

```
pip install -e ".[test]"
pytest                 # about a minute
pytest -m slow         # longer statistical runs
```

## Diagnostics

`resremd summary` reports, besides acceptance and round trips:

- **Reservoir temperature check.** The top replica's enthalpies and the
  reservoir's must satisfy ln[P_top(h)/P_R(h)] = const - (beta_top - beta_R) h,
  whatever the density of states (Shirts, *J. Chem. Theory Comput.* 2013).
  The check gives a z-score (block-bootstrap errors) and the temperature
  the reservoir's frames actually behave like, so frames drawn at another
  temperature than their label are caught without a reference. It cannot
  see a reservoir that is missing part of its ensemble: the relation holds
  between two equally restricted ensembles, and the top replica, fed by the
  reservoir, inherits the restriction.
- **Lineage.** How many of the lowest-temperature samples descend from
  reservoir frames, how many distinct frames reached the bottom, and the
  effective number of independent reservoir ancestors behind the ensemble.
- **Cost.** MD steps and wall time per phase, summed over sessions, in the
  manifest (and a reservoir's own cost in its `reservoir.json`).

Two flaws pass the temperature check. A reservoir that never converged at
its own temperature is caught by comparing its two halves, as the benchmark
harness does. A reservoir with a state missing is caught by the coverage
check, `resremd.reservoir_coverage(run, top_labels, reservoir_labels,
n_states)`, given state labels from any classification you trust: the top
replica's populations are compared with the reservoir's reweighted to the
top temperature, and a state the top replica visits but the reservoir never
holds is reported as unsupported. It sees only states the top replica reaches
by its own dynamics. A state that neither the ladder nor the reservoir
reaches stays invisible, which is why a reservoir must come from a
simulation long enough, or biased enough, to have visited everything that
matters at its temperature.

## Benchmarks

`benchmarks/` holds a harness that compares reservoir REMD with plain REMD
at equal cost, on exact models, alanine dipeptide and chignolin. It covers
flawed reservoirs, runs from opposite starts, and SLURM job arrays. See
`benchmarks/README.md`.

## Use from FastMDXplora

Settings are declared once, in `resremd.options`, in the shape of
FastMDXplora's `Field` (name, type, default, help, choices, bounds) and with
its unit-suffixed names (`temperature_K`, `timestep_fs`, `friction_per_ps`).
A prepared FastMDXplora `setup/` directory is a valid `--prepared` input,
the per-temperature trajectories are ordinary DCD files with a
`topology.pdb`, and errors carry stable codes (`resremd.reservoir.mismatch`,
...) for a caller to map onto its own.

## Citing

Please cite the method papers listed in `CITATION.cff`, which every run also
records in its `manifest.json`.

## License

MIT. See `LICENSE`.
