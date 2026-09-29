import os
import signal

import numpy as np
import pytest

from resremd import testsystems
import resremd
from resremd.errors import InputError, ReservoirError
from resremd.output import DcdTrajectory
from resremd.reservoir import Reservoir
from resremd.system import subset_topology

md = pytest.importorskip("mdtraj")


def test_generate_records_how_independent_the_frames_are(tmp_path):
    meta = resremd.generate_reservoir(
        testsystems.double_well(), output=str(tmp_path / "r"), temperature_K=520.0,
        duration_ns=0.2, frame_interval_steps=200, equilibration_ns=0.01,
        platform="Reference", random_seed=2, friction_per_ps=5.0)
    assert meta["complete"] and meta["n_frames"] == 500
    stats = meta["statistics"]
    assert 0 < stats["effective_independent_frames"] <= 500
    res = Reservoir.open(tmp_path / "r")
    assert res.temperature_K == 520.0
    assert not (tmp_path / "r/build_checkpoint.npz").exists()


def test_generate_stops_and_resumes(tmp_path):
    def stop(info):
        os.kill(os.getpid(), signal.SIGTERM)

    common = dict(output=str(tmp_path / "r"), temperature_K=520.0,
                  duration_ns=1.2, frame_interval_steps=200,
                  equilibration_ns=0.0, platform="Reference", random_seed=2,
                  minimize=False)
    # A frame every 0.4 ps and a checkpoint every 0.5 ns: 1250 frames. The
    # signal arrives at the first checkpoint, frame 1250 of 3000.
    meta = resremd.generate_reservoir(testsystems.double_well(), on_progress=stop,
                                      **common)
    assert not meta["complete"]
    with pytest.raises(ReservoirError, match="not finished"):
        Reservoir.open(tmp_path / "r")
    meta = resremd.generate_reservoir(testsystems.double_well(), resume=True,
                                      **common)
    assert meta["complete"]
    assert Reservoir.open(tmp_path / "r").n_frames == 3000


def _write_dcd(path, prepared, frames, boxes=None):
    top = subset_topology(prepared.topology, np.arange(prepared.n_atoms))
    traj = DcdTrajectory(path, top, timestep_ps=0.002, interval_steps=10)
    for k, f in enumerate(frames):
        traj.write(f, None if boxes is None else boxes[k])
    traj.close()


def test_import_with_weights_and_stride(tmp_path):
    prepared = testsystems.double_well()
    frames = [np.array([[x, 0.0, 0.0]]) for x in np.linspace(-1, 1, 10)]
    _write_dcd(tmp_path / "a.dcd", prepared, frames[:6])
    _write_dcd(tmp_path / "b.dcd", prepared, frames[6:])
    from openmm import app

    with open(tmp_path / "top.pdb", "w") as fh:
        app.PDBFile.writeFile(prepared.topology, prepared.positions * 10, fh)
    np.save(tmp_path / "w.npy", np.arange(1.0, 11.0))
    meta = resremd.import_reservoir(
        trajectories=[str(tmp_path / "a.dcd"), str(tmp_path / "b.dcd")],
        topology=str(tmp_path / "top.pdb"), output=str(tmp_path / "r"),
        temperature_K=400.0, kind="weighted", weights=str(tmp_path / "w.npy"),
        stride=2)
    assert meta["n_frames"] == 5
    res = Reservoir.open(tmp_path / "r")
    assert np.allclose(res.weights, np.array([1, 3, 5, 7, 9]) / 25.0)
    assert np.allclose(res.positions[:, 0, 0], np.linspace(-1, 1, 10)[::2],
                       atol=1e-6)


def test_import_refuses_what_it_cannot_know(tmp_path):
    prepared = testsystems.lj_box()
    boxes = [prepared.box * s for s in (1.0, 1.01, 0.99)]
    _write_dcd(tmp_path / "npt.dcd", prepared, [prepared.positions] * 3, boxes)
    from openmm import app

    with open(tmp_path / "top.pdb", "w") as fh:
        app.PDBFile.writeFile(prepared.topology, prepared.positions * 10, fh)
    common = dict(trajectories=[str(tmp_path / "npt.dcd")],
                  topology=str(tmp_path / "top.pdb"), temperature_K=150.0)
    with pytest.raises(ReservoirError, match="constant pressure"):
        resremd.import_reservoir(output=str(tmp_path / "r1"), **common)
    with pytest.raises(InputError, match="weights"):
        resremd.import_reservoir(output=str(tmp_path / "r2"),
                                 kind="weighted", **common)
    meta = resremd.import_reservoir(output=str(tmp_path / "r3"),
                                    pressure_bar=200.0, **common)
    assert meta["ensemble"]["pressure_bar"] == 200.0


def test_generate_records_its_cost(tmp_path):
    meta = resremd.generate_reservoir(
        testsystems.double_well(), output=str(tmp_path / "r"),
        temperature_K=520.0, duration_ns=0.02, frame_interval_steps=100,
        equilibration_ns=0.002, platform="Reference", random_seed=2)
    assert meta["cost"]["md_steps"] == {"equilibration": 1000,
                                        "production": 10000}
    assert meta["cost"]["wall_seconds"] > 0


def test_a_build_killed_before_its_first_checkpoint_starts_over(tmp_path):
    out = tmp_path / "r"
    out.mkdir()
    (out / "reservoir.json").write_text('{"complete": false}')
    (out / "positions.npy").write_bytes(b"partial")
    meta = resremd.generate_reservoir(
        testsystems.double_well(), output=str(out), temperature_K=520.0,
        duration_ns=0.004, frame_interval_steps=100, equilibration_ns=0.0,
        platform="Reference", random_seed=2)
    assert meta["complete"] and meta["n_frames"] == 20
    with pytest.raises(InputError, match="already holds"):
        resremd.generate_reservoir(
            testsystems.double_well(), output=str(out), temperature_K=520.0,
            duration_ns=0.004, frame_interval_steps=100,
            equilibration_ns=0.0, platform="Reference", random_seed=2)


def _cis_estimate(path):
    r = Reservoir.open(path)
    top = md.load_topology(str(path / "topology.pdb"))
    phi = md.compute_dihedrals(md.Trajectory(np.array(r.positions), top),
                               [[0, 1, 2, 3]])[:, 0]
    cis = (np.abs(phi) < np.pi / 2).astype(float)
    w = r.weights if r.weights is not None else np.full(len(cis),
                                                         1 / len(cis))
    blocks = np.array_split(np.arange(len(cis)), 10)
    means = [(w[b] * cis[b]).sum() / w[b].sum() for b in blocks]
    return float((w * cis).sum()), float(np.std(means, ddof=1) / np.sqrt(10))


def test_a_bias_crosses_the_barrier_and_the_weights_remove_it(tmp_path):
    common = dict(temperature_K=520.0, duration_ns=6.0,
                  frame_interval_steps=100, equilibration_ns=0.01,
                  friction_per_ps=5.0, platform="Reference", random_seed=1)
    meta = resremd.generate_reservoir(
        testsystems.torsion_model(), output=str(tmp_path / "biased"),
        bias_torsions=testsystems.torsion_bias(70.0), **common)
    assert meta["kind"] == "weighted"
    assert meta["source"]["bias_torsions"][0]["parameters"] == {"k": 70.0}
    assert meta["statistics"]["effective_frames_kish"] > 1000
    est, se = _cis_estimate(tmp_path / "biased")
    exact = testsystems.cis_fraction(520.0)
    assert abs(est - exact) < max(4 * se, 0.01), (est, se, exact)
    # Without the bias the barrier is never crossed at all.
    resremd.generate_reservoir(testsystems.torsion_model(),
                               output=str(tmp_path / "plain"), **common)
    assert _cis_estimate(tmp_path / "plain")[0] == 0.0


def test_biased_build_energies_are_unbiased(tmp_path):
    import openmm

    resremd.generate_reservoir(
        testsystems.torsion_model(), output=str(tmp_path / "r"),
        temperature_K=520.0, duration_ns=0.02, frame_interval_steps=100,
        equilibration_ns=0.0, platform="Reference", random_seed=1,
        bias_torsions=testsystems.torsion_bias(70.0))
    built = np.load(tmp_path / "r/build_potential_kjmol.npy")
    bias = np.load(tmp_path / "r/build_bias_kjmol.npy")
    assert np.all(bias <= 0)
    p = testsystems.torsion_model()
    ctx = openmm.Context(p.system, openmm.VerletIntegrator(0.001),
                         openmm.Platform.getPlatformByName("Reference"))
    res = Reservoir.open(tmp_path / "r")
    for k in (0, res.n_frames - 1):
        ctx.setPositions(res.frame(k)[0])
        u = ctx.getState(getEnergy=True).getPotentialEnergy()._value
        assert u == pytest.approx(built[k], abs=1e-3)


@pytest.mark.parametrize("bias,match", [
    ([{"atoms": [0, 1, 2], "energy": "theta"}], "four distinct"),
    ([{"atoms": [0, 1, 2, 3], "energy": "k*"}], "not valid"),
    ([{"atoms": [0, 1, 2, 3], "energy": "theta", "extra": 1}], "mapping"),
])
def test_bad_biases_are_refused(tmp_path, bias, match):
    with pytest.raises(InputError, match=match):
        resremd.generate_reservoir(
            testsystems.torsion_model(), output=str(tmp_path / "r"),
            temperature_K=520.0, duration_ns=0.02, frame_interval_steps=100,
            platform="Reference", bias_torsions=bias)


def test_truncate_npy_keeps_the_first_rows(tmp_path):
    from resremd.build import truncate_npy

    for shape, dtype in (((50, 3, 3), np.float64), ((50,), np.float32)):
        a = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
        np.save(tmp_path / "a.npy", a)
        truncate_npy(tmp_path / "a.npy", 17)
        assert np.array_equal(np.load(tmp_path / "a.npy"), a[:17])


def test_halves_tv():
    from resremd.build import halves_tv

    same = np.tile([[-170.0], [10.0]], (50, 1))
    assert halves_tv(same, 6) == 0.0
    split = np.concatenate([np.full((50, 1), -170.0), np.full((50, 1), 10.0)])
    assert halves_tv(split, 6) == pytest.approx(1.0)
    # Weights count: the second half's lone left frame outweighs the rest.
    w = np.ones(100)
    mixed = split.copy()
    mixed[99] = -170.0
    w[99] = 1e6
    assert halves_tv(mixed, 6, w) < 0.01


CONVERGE = dict(temperature_K=520.0, duration_ns=20.0,
                frame_interval_steps=500, equilibration_ns=0.05,
                friction_per_ps=5.0, platform="Reference", random_seed=3,
                minimize=False, bias_torsions=testsystems.torsion_bias(70.0),
                convergence_torsions=[[0, 1, 2, 3]], convergence_tv=0.03)


def test_a_build_stops_once_its_halves_agree(tmp_path):
    meta = resremd.generate_reservoir(testsystems.torsion_model(),
                                      output=str(tmp_path / "r"), **CONVERGE)
    conv = meta["convergence"]
    assert conv["converged"] and meta["n_frames"] < 40000
    assert all(h[1] <= 0.03 and h[2] >= 10 for h in conv["history"][-2:])
    assert min(conv["transitions"]) >= 10
    assert meta["cost"]["md_steps"]["production"] == meta["n_frames"] * 500
    res = Reservoir.open(tmp_path / "r")
    assert res.positions.shape[0] == len(res.weights) == meta["n_frames"]
    assert np.load(tmp_path / "r/build_torsions_deg.npy").shape == (
        meta["n_frames"], 1)
    est, se = _cis_estimate(tmp_path / "r")
    assert abs(est - testsystems.cis_fraction(520.0)) < max(4 * se, 0.02)


def test_a_stopped_convergence_build_resumes(tmp_path):
    def stop(info):
        os.kill(os.getpid(), signal.SIGTERM)

    first = resremd.generate_reservoir(testsystems.torsion_model(),
                                       output=str(tmp_path / "r"),
                                       on_progress=stop, **CONVERGE)
    assert not first["complete"]
    with pytest.raises(Exception, match="convergence"):
        resremd.generate_reservoir(testsystems.torsion_model(),
                                   output=str(tmp_path / "r"), resume=True,
                                   **{**CONVERGE, "convergence_tv": 0.05})
    meta = resremd.generate_reservoir(testsystems.torsion_model(),
                                      output=str(tmp_path / "r"), resume=True,
                                      **CONVERGE)
    # The check made before the stop (at frame 500) is kept.
    frames = [h[0] for h in meta["convergence"]["history"]]
    assert meta["complete"] and frames[0] == 500
    assert frames == sorted(set(frames))


def test_convergence_settings_are_checked(tmp_path):
    args = dict(temperature_K=520.0, duration_ns=0.02,
                frame_interval_steps=100, platform="Reference")
    with pytest.raises(InputError, match="needs `convergence_torsions`"):
        resremd.generate_reservoir(testsystems.torsion_model(),
                                   output=str(tmp_path / "a"),
                                   convergence_tv=0.02, **args)
    with pytest.raises(InputError, match="four distinct"):
        resremd.generate_reservoir(testsystems.torsion_model(),
                                   output=str(tmp_path / "b"),
                                   convergence_torsions=[[0, 1, 2, 2]], **args)


def test_a_build_that_fails_while_finishing_early_resumes_to_finish(
        tmp_path, monkeypatch):
    import resremd.build as build

    real = build.write_topology

    def full_disk(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(build, "write_topology", full_disk)
    with pytest.raises(OSError):
        resremd.generate_reservoir(testsystems.torsion_model(),
                                   output=str(tmp_path / "r"), **CONVERGE)
    monkeypatch.setattr(build, "write_topology", real)
    meta = resremd.generate_reservoir(testsystems.torsion_model(),
                                      output=str(tmp_path / "r"), resume=True,
                                      **CONVERGE)
    assert meta["complete"] and meta["convergence"]["converged"]
    assert Reservoir.open(tmp_path / "r").n_frames == meta["n_frames"] < 40000


def test_a_resume_with_another_frame_interval_is_refused(tmp_path):
    from resremd.errors import ResumeError

    def stop(info):
        os.kill(os.getpid(), signal.SIGTERM)

    common = dict(output=str(tmp_path / "r"), temperature_K=520.0,
                  duration_ns=1.2, equilibration_ns=0.0, platform="Reference",
                  random_seed=2, minimize=False)
    resremd.generate_reservoir(testsystems.double_well(), on_progress=stop,
                               frame_interval_steps=200, **common)
    with pytest.raises(ResumeError, match="frame interval"):
        resremd.generate_reservoir(testsystems.double_well(), resume=True,
                                   frame_interval_steps=400, **common)


def test_a_stop_request_is_not_a_convergence_check(tmp_path, monkeypatch):
    import resremd.build as build

    real = build.torsion_angles_deg
    frames = []

    def angles_then_stop(*args):
        frames.append(1)
        if len(frames) == 501:      # one frame after the first check
            os.kill(os.getpid(), signal.SIGTERM)
        return real(*args)

    monkeypatch.setattr(build, "torsion_angles_deg", angles_then_stop)
    # Any two checks would pass this tolerance.
    meta = resremd.generate_reservoir(testsystems.torsion_model(),
                                      output=str(tmp_path / "r"),
                                      **{**CONVERGE, "convergence_tv": 1.0})
    assert not meta["complete"]
    with np.load(tmp_path / "r/build_checkpoint.npz") as data:
        import json

        saved = json.loads(str(data["meta"]))
    assert saved["frames"] == 501
    assert [h[0] for h in saved["convergence_history"]] == [500]


def test_regions_are_centred_on_cis_and_trans():
    from resremd.build import torsion_regions

    # With an even number of regions, 0 and 180 are centres.
    r = torsion_regions(np.array([179.0, -179.0, 1.0, -1.0]), 6)
    assert r[0] == r[1] and r[2] == r[3] and r[0] != r[2]


def _ar(rng, centre, sd, n=4000, rho=0.9):
    x = np.zeros(n)
    e = rng.normal(0.0, sd * np.sqrt(1 - rho ** 2), n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return (centre + x + 180.0) % 360.0 - 180.0


def test_motion_within_one_basin_is_not_a_transition():
    """A broad well, wherever it sits (on a region boundary too), counts no
    transitions, however much it wanders."""
    from resremd.build import basin_transitions

    rng = np.random.default_rng(2)
    for centre, sd in ((150.0, 15.0), (-150.0, 15.0), (-40.0, 15.0),
                       (180.0, 20.0), (-100.0, 30.0)):
        assert basin_transitions(_ar(rng, centre, sd)) == [0], centre


def test_hops_between_basins_are_counted():
    from resremd.build import basin_transitions

    rng = np.random.default_rng(3)
    parts, state = [], 180.0
    for _ in range(12):
        parts.append(_ar(rng, state, 10.0, n=100))
        state = 0.0 if state == 180.0 else 180.0
    assert basin_transitions(np.concatenate(parts)) == [11]


def test_basins_from_biased_frames_use_the_weights():
    """Under a bias that flattens the barrier the frames spread evenly; the
    weights restore the barrier, and with it the two basins."""
    from resremd.build import basin_transitions, bias_weights

    phi = np.linspace(-180.0, 180.0, 20000, endpoint=False)
    rng = np.random.default_rng(4)
    rng.shuffle(phi)
    phi = np.sort(phi.reshape(200, 100), axis=1).ravel()  # sweeps
    bias = -80.0 * np.sin(np.radians(phi)) ** 2
    assert basin_transitions(phi) == [0]
    assert basin_transitions(phi, weights=bias_weights(bias, 520.0))[0] > 10


def test_a_build_stuck_in_its_start_does_not_stop(tmp_path):
    """Without a bias the torsion never crosses at 300 K: its halves agree
    at once, but with no transitions the build runs to its full length."""
    meta = resremd.generate_reservoir(
        testsystems.torsion_model(), output=str(tmp_path / "r"),
        temperature_K=300.0, duration_ns=3.0, frame_interval_steps=500,
        equilibration_ns=0.0, friction_per_ps=5.0, platform="Reference",
        random_seed=4, minimize=False, convergence_torsions=[[0, 1, 2, 3]],
        convergence_tv=0.05)
    conv = meta["convergence"]
    assert not conv["converged"] and conv["transitions"] == [0]
    assert meta["n_frames"] == 3000
    assert all(h[1] <= 0.05 for h in conv["history"])   # halves agree


def test_a_build_started_before_the_transition_rule_resumes():
    from resremd.build import _same_convergence

    now = {"torsions": [[0, 1, 2, 3]], "tv": 0.02, "bins": 6,
           "min_transitions": 10}
    old = {k: v for k, v in now.items() if k != "min_transitions"}
    assert _same_convergence(old, now)
    assert not _same_convergence({**old, "tv": 0.05}, now)
    assert not _same_convergence(None, now) and _same_convergence(None, None)


def _alanine_prepared(tmp_path):
    from openmm import app

    from resremd.system import write_prepared

    top, pos = testsystems.alanine_dipeptide()
    system = app.ForceField("amber14-all.xml").createSystem(
        top, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    write_prepared(tmp_path / "setup", system, top, pos)
    return top, pos


def _shuffled_pdb(path, top, pos, drop=None):
    """A PDB file of the structure with every residue's atoms in reverse
    order, as a file from another program might list them."""
    import io

    from openmm import app

    buf = io.StringIO()
    app.PDBFile.writeFile(top, pos * 10.0, buf)
    lines = buf.getvalue().splitlines()
    atoms = [ln for ln in lines if ln.startswith(("ATOM", "HETATM"))]
    by_res: dict[str, list[str]] = {}
    for ln in atoms:
        by_res.setdefault(ln[17:27], []).append(ln)
    out = []
    for group in by_res.values():
        for ln in reversed(group):
            if drop is None or ln[12:16].strip() != drop:
                out.append(ln)
    path.write_text("\n".join(out) + "\nEND\n")


def test_structures_in_another_atom_order_are_matched_by_name(tmp_path):
    top, pos = _alanine_prepared(tmp_path)
    rng = np.random.default_rng(1)
    files = []
    for k in range(3):
        p = pos + rng.normal(0.0, 0.01, pos.shape)
        _shuffled_pdb(tmp_path / f"pose{k}.pdb", top, p)
        files.append((tmp_path / f"pose{k}.pdb", p))
    meta = resremd.import_reservoir(
        trajectories=[str(f) for f, _ in files],
        prepared=str(tmp_path / "setup"), output=str(tmp_path / "r"),
        kind="non_boltzmann")
    res = Reservoir.open(tmp_path / "r")
    assert meta["n_frames"] == 3
    for k, (_, p) in enumerate(files):
        # PDB keeps 0.001 nm; the atoms are back in the system's order.
        assert np.abs(res.frame(k)[0] - p).max() < 1e-3
    # It serves a run of the prepared system.
    resremd.run(str(tmp_path / "setup"), output=str(tmp_path / "run"),
                reservoir=str(tmp_path / "r"), temperatures_K=[300, 400],
                production_steps=100, exchange_interval_steps=50,
                platform="Reference", equilibration_ns=0.0, minimize=False)


def test_structures_missing_atoms_are_refused_by_name(tmp_path):
    top, pos = _alanine_prepared(tmp_path)
    _shuffled_pdb(tmp_path / "pose.pdb", top, pos, drop="HA")
    with pytest.raises(ReservoirError, match="lacks HA"):
        resremd.import_reservoir(trajectories=[str(tmp_path / "pose.pdb")],
                                 prepared=str(tmp_path / "setup"),
                                 output=str(tmp_path / "r"),
                                 kind="non_boltzmann")


def test_imported_structures_can_be_minimised(tmp_path):
    import openmm

    from resremd.system import load_prepared

    top, pos = _alanine_prepared(tmp_path)
    rng = np.random.default_rng(2)
    strained = pos + rng.normal(0.0, 0.02, pos.shape)
    _shuffled_pdb(tmp_path / "pose.pdb", top, strained)
    resremd.import_reservoir(trajectories=[str(tmp_path / "pose.pdb")],
                             prepared=str(tmp_path / "setup"),
                             output=str(tmp_path / "r"), kind="non_boltzmann",
                             minimize_steps=200, platform="Reference")
    prep = load_prepared(tmp_path / "setup")
    ctx = openmm.Context(prep.system, openmm.VerletIntegrator(0.001),
                         openmm.Platform.getPlatformByName("Reference"))

    def energy(x):
        ctx.setPositions(x)
        return ctx.getState(getEnergy=True).getPotentialEnergy()._value

    after = energy(Reservoir.open(tmp_path / "r").frame(0)[0])
    assert after < energy(strained) - 100.0
    with pytest.raises(InputError, match="non_boltzmann"):
        resremd.import_reservoir(trajectories=[str(tmp_path / "pose.pdb")],
                                 prepared=str(tmp_path / "setup"),
                                 output=str(tmp_path / "r2"),
                                 kind="boltzmann", temperature_K=400.0,
                                 minimize_steps=200)
