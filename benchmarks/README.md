# Benchmarks

A reproducible comparison of reservoir replica exchange with plain
temperature replica exchange, built to answer two questions:

1. **At equal total cost, with the reservoir's own simulation counted, when
   does a reservoir make replica exchange converge faster?**
2. **What does an inadequate reservoir do to the answer, and which
   diagnostics notice?**

Everything is driven by a spec file, runs here or on a SLURM cluster, and
ends in tables (CSV/JSON) and a plotting script you run yourself.

## Systems

| tier | system | states | features (2D histogram) | starts |
|---|---|---|---|---|
| 1 | double well (`double_well`), exact | left, right well | x, y | right (disfavoured), left |
| 2 | alanine dipeptide, GBn2 or TIP3P-FB | alpha_R, beta, PPII, alpha_L | phi, psi | extended; alpha_L (sought at 600 K) |
| 3 | chignolin CLN025 (PDB 5AWL), GBn2 or TIP3P-FB | folded, unfolded | C-alpha RMSD to 5AWL, C-alpha Rg | folded; unfolded (sought at 600 K) |

- Force field: Amber14 (ff14SB), HBonds constrained, 2 fs.
- Alanine dipeptide states are our own division of the Ramachandran map:
  - alpha_R: phi < 0, -120 <= psi < 50
  - beta: phi < -100 otherwise
  - PPII: -100 <= phi < 0 otherwise
  - alpha_L: phi >= 0, which also holds C7ax
- Chignolin counts as folded at a C-alpha RMSD of 0.2 nm or less from 5AWL.
- Second starts are *sought*: dynamics at 600 K from the first start until
  the condition holds, then a short minimisation. The box is not touched,
  so every start is the same system and can share a reservoir.

## How runs are compared

- **At equal cost.** With `equal_cost: true` (the default) every run
  spends the same number of MD steps over all its replicas, counting
  equilibration and the run's own reservoir. Methods without a reservoir,
  or with a cheaper one, run longer. `resbench cost` shows the plan.
- **One reservoir per run.** Each reservoir is made from its own run's
  start, so runs from opposite starts are independent, and no run inherits
  frames from the other side. (`shared: true` makes one per method.)
- **Fair baselines.** A method may carry its own `temperatures_K`, for
  example plain REMD with the reservoir's temperature as an extra rung
  (`remd_plus`). Every ladder starts at the same lowest temperature.
- **Seeds** are derived from the repeat, the method and the start, so no
  two jobs share a random stream.
- **Stale output is refused.** Every job records a fingerprint of the
  settings that made it, and output made under other settings is refused
  rather than reused.

## What is measured

For every complete run, the samples at the lowest temperature are read in
order, and populations and the feature histogram are estimated from the
first n samples, for n on a logarithmic grid.

**Error against a reference**
- **Total variation (TV) distance of the populations.** This is the
  headline. 0.02 means no state, or set of states, is off by more than 2
  percentage points.
- **Jensen-Shannon divergence of the 2D histogram, less its floor.** The
  floor is what n independent samples of the reference would show by
  chance, so perfect sampling reads zero rather than a sample-size effect.

**The reference**, in order of preference:
- **exact:** tier 1.
- **file:** independent long runs (`*_reference.yml` specs, written with
  `resbench reference`).
- **pooled:** the final halves of named methods.
  - A pooled reference leaves out the repeat being scored, so no run is
    compared with itself.
  - The analysis warns when a reference is too uncertain for the threshold.

**Convergence time.** The first time after which the error stays below the
threshold for the rest of the run.
- It is reported per start and for **both starts**: per seed, the later of
  the two, which is the cost to be right whichever side you start from.
- A repeat that never converges is censored.
- Medians and 95 percent intervals are bootstrapped over seeds, and every
  seed's value is listed when there are six or fewer.

**Agreement between starts.** The TV distance between the same seed's runs
from opposite starts. It needs no reference.
- Caveat: agreement shows the starts were forgotten, not that the answer is
  right. A flawed reservoir can bias both runs the same way.

**Reservoir diagnostics**, from `resremd summary`:
- **Temperature check.** The enthalpy distributions of the top replica and
  the reservoir must satisfy ln[P_top(h)/P_R(h)] = const - (beta_top - beta_R) h
  (Shirts 2013), whatever the density of states.
  - It reports a block-bootstrapped z and the temperature the frames behave
    like.
  - It catches a reservoir labelled with the wrong temperature.
  - It cannot catch a reservoir missing part of its ensemble.
- **Halves.** The reservoir's own populations in its first and second
  halves. They differ when its simulation had not converged.
- **Lineage.** Acceptance, the fraction of lowest-temperature samples
  descended from the reservoir, and the effective number of ancestors.

A reservoir with a state entirely missing passes all of these. Only a
reference exposes it, which is itself one of the findings tier 1 is
designed to show.

## Specs

| spec | what it asks | hardware |
|---|---|---|
| `smoke.yml` | does the harness work (CI runs it) | CPU, 15 s |
| `exact_defects.yml` | tier 1: seven reservoirs, each flawed in one way, against plain REMD | CPU, about an hour |
| `alanine_implicit_reference.yml` | long plain REMD for tier 2's reference | one GPU, a day |
| `alanine_implicit.yml` | tier 2: reservoir length and temperature | one GPU, hours |
| `alanine_explicit.yml` | tier 2 in water, 14 replicas | GPU-days |
| `chignolin_implicit_reference.yml` | long plain REMD for tier 3's reference | one GPU, a week |
| `chignolin_implicit.yml` | tier 3: folding from both sides | one GPU, days |
| `chignolin_explicit.yml` | tier 3 in water, 20 replicas | GPU-weeks; pilot first |

The explicit-solvent ladders are geometric first guesses. Before committing
the full budget, run one seed briefly and check `resremd summary`: 20 to 40
percent neighbour acceptance.

## Running

```
pip install -e ./benchmarks          # adds the `resbench` command

resbench plan benchmarks/specs/smoke.yml bench_smoke
resbench cost bench_smoke --ns-per-day 500      # size the plan first
resbench run bench_smoke                        # every job, in order, here
resbench status bench_smoke
resbench analyze bench_smoke
python benchmarks/plots/plot_benchmark.py bench_smoke
```

A spec whose reference is a file needs that file first. Run its reference
spec, then write the file into the benchmark directory (paths are relative
to it):

```
resbench plan benchmarks/specs/alanine_implicit_reference.yml bench_ala_ref
resbench run bench_ala_ref
resbench plan benchmarks/specs/alanine_implicit.yml bench_ala
resbench reference bench_ala_ref bench_ala/alanine_implicit_reference.json
resbench run bench_ala && resbench analyze bench_ala
```

Every job can be run again: finished ones are skipped, and an interrupted
run or reservoir build resumes from its checkpoint.

### On the lab HPC (SLURM, no internet)

1. On a machine with internet, fetch the structure the spec names and
   prepare the systems. Preparation runs a short constant-pressure
   equilibration and the seeking of second starts; use a GPU for explicit
   solvent.
   ```
   curl -o 5AWL.pdb https://files.rcsb.org/download/5AWL.pdb
   resbench plan benchmarks/specs/chignolin_implicit.yml bench_chig
   resbench job bench_chig prepare
   ```
2. Copy `bench_chig/` to the cluster, then plan again there so the SLURM
   scripts carry the cluster's paths. The `setup` lines in the spec's
   `slurm` block activate the environment.
   ```
   resbench plan bench_chig/spec.yml bench_chig
   bench_chig/slurm/submit.sh
   ```
   `submit.sh` queues three job arrays, each waiting for the one before:
   prepare (already done, so it returns at once), reservoirs, then runs.
   Each task takes one GPU (`--gres=gpu:1`).
3. Copy back and run `resbench analyze`, or analyse on the cluster.

## Adding a system

Subclass `BenchSystem` in `resbench/systems.py` and register it in
`SYSTEMS`. It needs to say:
- how it is prepared;
- its states;
- two features computed from an MDTraj trajectory;
- optionally, the exact answer.

## Outputs

`<benchmark>/analysis/`:

| file | contents |
|---|---|
| `curves.csv` | per run and grid point: time, cost, populations, TV error, JSD |
| `convergence.csv` | per run: convergence time and cost, final error, exchange and reservoir diagnostics |
| `agreement.csv` | per method, seed and time: distance between starts |
| `reservoirs.csv` | per reservoir: populations, halves distance, cost |
| `figures/` | from `plots/plot_benchmark.py`: convergence, cost to converge, populations, agreement |
| `summary.json` | per method: medians with intervals, final populations, the reference |
