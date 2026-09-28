# Examples

## double_well

An exactly solvable check that runs on a CPU in about a minute. A particle
in an asymmetric double well starts in the less favoured well; plain REMD
and reservoir REMD of the same length are compared against the exact
population of the favoured well at 300 K.

```
cd examples/double_well
python compare.py
```

## peptide

Alanine dipeptide in implicit solvent, the first benchmark system of the
JPCB 2022 paper, from structure to backbone populations. Any peptide PDB
can take its place, and a `setup/` directory from FastMDXplora can replace
step 2.

```
cd examples/peptide

# 1. Structure (writes alanine-dipeptide.pdb)
python build_alanine_dipeptide.py

# 2. system.xml, state.xml and topology.pdb in setup/
python prepare.py alanine-dipeptide.pdb --solvent implicit --output setup

# 3. Reservoir at 450 K
resremd reservoir generate --prepared setup --temperature-K 450 \
    --duration-ns 100 --frame-interval-steps 2500 --output reservoir

# 4. Reservoir REMD and, as the control, plain REMD on the same ladder:
#    four rungs of a geometric ladder whose fifth rung is the reservoir
T=$(resremd ladder --temperature-min-K 300 --temperature-max-K 450 \
    --n-replicas 5 | head -4)
resremd run --prepared setup --reservoir reservoir --temperatures-K $T \
    --duration-ns 20 --save-selection solute --output res_remd
resremd run --prepared setup --temperatures-K $T \
    --duration-ns 20 --save-selection solute --output remd

# 5. Exchange statistics, and populations over time at 300 K
resremd summary res_remd
python backbone_populations.py remd res_remd
```

In zsh, write `${=T}` in place of `$T` so the list splits into words.
