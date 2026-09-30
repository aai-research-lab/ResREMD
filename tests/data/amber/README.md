# A reservoir written by cpptraj

`res_bins.nc` and `clusterinfo.dat` were written by cpptraj V7.11.2
(Amber-MD/cpptraj, built from source with NetCDF 4.9.3), so the tests read
the files Amber users actually have: AMBER-convention NetCDF 3 with an
unlimited frame dimension, `eptot` per frame in kcal/mol, the cluster
number of each frame in `bins`, the scalar `temp0` and the `iseed`
attribute.

To make them again:

```
python make_input.py              # ala.pdb, ala.dcd, ene.dat
cpptraj -i cpptraj.in             # res_bins.nc, clusterinfo.dat
```

`ala.dcd` is not kept: every frame is in `res_bins.nc`, and `ene.dat` holds
the OpenMM energy of each, rounded to 1e-6 kcal/mol by the text file.
