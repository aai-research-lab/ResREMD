"""Amber reservoirs: NetCDF files as cpptraj's `createreservoir` writes
them (AMBER trajectory conventions, 64-bit offset NetCDF 3, with `eptot`
per frame in kcal/mol, optional `bins`, and the scalar `temp0`)."""

import numpy as np
import pytest

import resremd
from resremd import testsystems
from resremd.errors import InputError
from resremd.reservoir import Reservoir
from resremd.thermo import BOLTZ

from test_exactness import TEMPERATURES, check_against_exact

pytest.importorskip("mdtraj")
T_R = 520.0


def write_amber_reservoir(path, positions_nm, *, temperature_K,
                          energies_kjmol, bins=None, iseed=1234):
    from scipy.io import netcdf_file

    n, atoms = positions_nm.shape[:2]
    with netcdf_file(path, "w", version=2) as nc:
        nc.Conventions = b"AMBER"
        nc.ConventionVersion = b"1.0"
        nc.program = b"cpptraj"
        nc.title = b"Cpptraj Generated structure reservoir"
        nc.iseed = np.int32(iseed)
        # cpptraj makes `frame` unlimited; SciPy's writer garbles small
        # record variables, so here it is fixed, which readers treat alike.
        nc.createDimension("frame", n)
        nc.createDimension("spatial", 3)
        nc.createDimension("atom", atoms)
        spatial = nc.createVariable("spatial", "c", ("spatial",))
        spatial[:] = np.array(list("xyz"), dtype="S1")
        coords = nc.createVariable("coordinates", "f", ("frame", "atom",
                                                        "spatial"))
        coords.units = b"angstrom"
        coords[:] = (positions_nm * 10.0).astype(np.float32)
        time = nc.createVariable("time", "f", ("frame",))
        time.units = b"picosecond"
        time[:] = np.arange(n, dtype=np.float32)
        eptot = nc.createVariable("eptot", "d", ("frame",))
        eptot[:] = np.asarray(energies_kjmol) / 4.184
        if bins is not None:
            b = nc.createVariable("bins", "i", ("frame",))
            b[:] = np.asarray(bins, dtype=np.int32)
        temp0 = nc.createVariable("temp0", "d", ())
        temp0.units = b"kelvin"
        temp0.data[()] = temperature_K


def _energy(frames):
    f = frames.astype(np.float32).astype(float)
    return testsystems.double_well_energy(f[:, 0, 0]) + 0.5 * \
        testsystems.SPRING * (f[:, 0, 1] ** 2 + f[:, 0, 2] ** 2)


def _topology(tmp_path):
    from openmm import app

    prep = testsystems.double_well()
    with open(tmp_path / "top.pdb", "w") as fh:
        app.PDBFile.writeFile(prep.topology, prep.positions * 10.0, fh)
    return str(tmp_path / "top.pdb")


def test_an_amber_reservoir_brings_its_temperature_and_energies(tmp_path):
    rng = np.random.default_rng(2)
    frames = testsystems.double_well_frames(
        testsystems.exact_x_samples(T_R, 200, rng), T_R, rng)
    write_amber_reservoir(tmp_path / "res.nc", frames, temperature_K=T_R,
                          energies_kjmol=_energy(frames))
    meta = resremd.import_reservoir(trajectories=[str(tmp_path / "res.nc")],
                                    topology=_topology(tmp_path),
                                    output=str(tmp_path / "r"))
    assert meta["kind"] == "boltzmann" and meta["temperature_K"] == T_R
    r = Reservoir.open(tmp_path / "r")
    assert r.has_build_energies()
    assert np.allclose(r.positions, frames, atol=1e-6)
    # Recomputed under the run's System, the energies agree.
    recomputed = _energy(np.asarray(r.positions))
    assert r.hamiltonian_spread(recomputed) < 1e-6
    # A differing temperature is refused.
    with pytest.raises(InputError, match="made at 520"):
        resremd.import_reservoir(trajectories=[str(tmp_path / "res.nc")],
                                 topology=_topology(tmp_path),
                                 output=str(tmp_path / "r2"),
                                 temperature_K=500.0)


def test_amber_cluster_bins_and_clusterinfo_make_it_exact(tmp_path):
    """Frames exact within each well but half in each, and the
    clusterinfo populations restore the wells' true weights."""
    rng = np.random.default_rng(4)
    x = testsystems.exact_x_samples(T_R, 80000, rng)
    x = np.concatenate([x[x < 0][:10000], x[x >= 0][:10000]])
    frames = testsystems.double_well_frames(x, T_R, rng)
    bins = np.where(x < 0, 1, 2)
    write_amber_reservoir(tmp_path / "res.nc", frames, temperature_K=T_R,
                          energies_kjmol=_energy(frames), bins=bins)
    left = testsystems.left_fraction(T_R)
    (tmp_path / "clusterinfo").write_text(
        f"1\n 1 1 1 1 2 -180.000\n2\n 1 {left * 1e6:.0f} 0\n"
        f" 2 {(1 - left) * 1e6:.0f} 1\n")
    meta = resremd.import_reservoir(
        trajectories=[str(tmp_path / "res.nc")], topology=_topology(tmp_path),
        output=str(tmp_path / "r"), clusterinfo=str(tmp_path / "clusterinfo"))
    assert meta["kind"] == "weighted"
    r = Reservoir.open(tmp_path / "r")
    assert np.array_equal(np.load(tmp_path / "r/cluster_labels.npy"), bins)
    assert r.weights[bins == 1].sum() == pytest.approx(left, abs=1e-6)
    resremd.run(testsystems.double_well(), output=str(tmp_path / "run"),
                reservoir=str(tmp_path / "r"), temperatures_K=TEMPERATURES,
                production_steps=250 * 30000, exchange_interval_steps=250,
                trajectory_interval_steps=250, friction_per_ps=5.0,
                platform="Reference", random_seed=5, save_selection="all",
                equilibration_ns=0.0, minimize=False)
    check_against_exact(tmp_path / "run")


def test_amber_energies_from_another_hamiltonian_are_refused(tmp_path):
    rng = np.random.default_rng(3)
    frames = testsystems.double_well_frames(
        testsystems.exact_x_samples(T_R, 400, rng), T_R, rng)
    noisy = _energy(frames) + rng.normal(0, BOLTZ * T_R, len(frames))
    write_amber_reservoir(tmp_path / "res.nc", frames, temperature_K=T_R,
                          energies_kjmol=noisy)
    resremd.import_reservoir(trajectories=[str(tmp_path / "res.nc")],
                             topology=_topology(tmp_path),
                             output=str(tmp_path / "r"))
    from resremd.errors import ReservoirError

    with pytest.raises(ReservoirError, match="reservoir_reweight"):
        resremd.run(testsystems.double_well(), output=str(tmp_path / "run"),
                    reservoir=str(tmp_path / "r"), temperatures_K=TEMPERATURES,
                    production_steps=250 * 4, exchange_interval_steps=250,
                    platform="Reference", equilibration_ns=0.0,
                    minimize=False)


def test_clusterinfo_needs_amber_bins(tmp_path):
    rng = np.random.default_rng(3)
    frames = testsystems.double_well_frames(
        testsystems.exact_x_samples(T_R, 20, rng), T_R, rng)
    write_amber_reservoir(tmp_path / "res.nc", frames, temperature_K=T_R,
                          energies_kjmol=_energy(frames))
    (tmp_path / "ci").write_text("0\n1\n 1 10\n")
    with pytest.raises(InputError, match="cluster bins"):
        resremd.import_reservoir(trajectories=[str(tmp_path / "res.nc")],
                                 topology=_topology(tmp_path),
                                 output=str(tmp_path / "r"),
                                 clusterinfo=str(tmp_path / "ci"))


DATA = __import__("pathlib").Path(__file__).parent / "data" / "amber"


def test_a_reservoir_written_by_cpptraj(tmp_path):
    """The real file: unlimited frame dimension, eptot, bins and temp0 as
    cpptraj V7.11.2 writes them (see tests/data/amber)."""
    import openmm
    from openmm import app

    from resremd.system import write_prepared

    pdb = app.PDBFile(str(DATA / "ala.pdb"))
    system = app.ForceField("amber14-all.xml").createSystem(
        pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None)
    write_prepared(tmp_path / "setup", system, pdb.topology,
                   np.array(pdb.getPositions(asNumpy=True)._value), None)
    meta = resremd.import_reservoir(
        trajectories=[str(DATA / "res_bins.nc")],
        prepared=str(tmp_path / "setup"), output=str(tmp_path / "r"),
        clusterinfo=str(DATA / "clusterinfo.dat"))
    assert meta["temperature_K"] == 500.0 and meta["n_frames"] == 60
    assert meta["kind"] == "weighted"
    labels = np.load(tmp_path / "r/cluster_labels.npy")
    assert np.bincount(labels).tolist() == [0, 29, 17, 8, 3, 2, 1]
    # Populations from clustering these same frames weight them equally.
    r = Reservoir.open(tmp_path / "r")
    assert np.allclose(r.weights, 1 / 60)
    # Amber's energies are OpenMM's here, to the file's rounding: far
    # inside what the run's Hamiltonian check lets through.
    context = openmm.Context(system, openmm.VerletIntegrator(0.001),
                             openmm.Platform.getPlatformByName("Reference"))
    ours = []
    for k in range(r.n_frames):
        context.setPositions(r.frame(k)[0])
        ours.append(context.getState(getEnergy=True)
                    .getPotentialEnergy()._value)
    assert r.hamiltonian_spread(np.array(ours)) < 0.01 * r.warn_kt()
