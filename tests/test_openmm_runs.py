"""Runs of OpenMM's own ReplicaExchangeSampler, read by ResREMD."""

import random
import re
import shutil
import struct

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import app, unit  # noqa: E402

if not hasattr(app, "ReplicaExchangeSampler"):
    pytest.skip("OpenMM's sampler is new in 8.6", allow_module_level=True)

import resremd  # noqa: E402
from resremd import testsystems  # noqa: E402
from resremd.errors import InputError  # noqa: E402
from resremd.mbar import solve  # noqa: E402
from resremd.openmm_runs import GAS_CONSTANT, OpenMMRun  # noqa: E402
from resremd.statistics import statistical_inefficiency  # noqa: E402
from resremd.system import from_objects  # noqa: E402

LADDER = [float(t) for t in np.geomspace(300.0, 1500.0, 6)]
#: MD steps per iteration.
STEPS = 1000


def _sampler(prepared, states, *, seed=3, temperature=300.0, steps=STEPS):
    random.seed(seed)  # the sampler draws its exchanges from `random`
    integrator = openmm.LangevinMiddleIntegrator(
        temperature * unit.kelvin, 5 / unit.picosecond,
        0.002 * unit.picoseconds)
    integrator.setRandomNumberSeed(seed)
    simulation = app.Simulation(
        prepared.topology, prepared.system, integrator,
        openmm.Platform.getPlatformByName("Reference"))
    simulation.context.setPositions(prepared.positions)
    simulation.context.setVelocitiesToTemperature(
        temperature * unit.kelvin, seed)
    return app.ReplicaExchangeSampler(states, simulation, steps)


def _run(path, states, iterations, *, prepared=None, every=2, fmt="dcd",
         volume=False, resume=False, seed=3, temperature=300.0,
         steps=STEPS, checkpoints=True):
    # The favoured well, where the replicas would be most of the time.
    sampler = _sampler(prepared or testsystems.double_well(-1.0), states,
                       seed=seed, temperature=temperature, steps=steps)
    sampler.reporters.append(app.ReplicaExchangeReporter(
        str(path), every, sampler, trajectoryPerState=True,
        trajectoryFormat=fmt, energy=True, volume=volume,
        checkpoints=checkpoints, resume=resume))
    sampler.simulate(iterations)
    return path


def _temperature_states(temperatures):
    return [{"temperature": t * unit.kelvin} for t in temperatures]


def _left(path, out):
    """The weighted fraction of frames with x < 0, and its standard error
    (delta method, rows' contributions correlated in time)."""
    md = pytest.importorskip("mdtraj")
    top = md.Topology.from_openmm(testsystems.double_well().topology)
    first = out["first_frame"]
    x = {k: md.load(str(path / f"state_{k}.dcd"), top=top).xyz[first:, 0, 0]
         for k in out["states"]}
    estimate = sum(float(np.sum(w * (x[k] < 0)))
                   for k, w in out["weights"].items())
    rows = sum(w * ((x[k] < 0) - estimate)
               for k, w in out["weights"].items())
    se = np.sqrt(statistical_inefficiency(rows) * np.sum(rows ** 2))
    return estimate, se


@pytest.fixture(scope="module")
def temperature_run(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("omm") / "run",
                _temperature_states(LADDER), 2000)


def test_the_gas_constant_is_openmms():
    assert GAS_CONSTANT == unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
        unit.kilojoule_per_mole / unit.kelvin)


def test_an_openmm_run_is_summarised(temperature_run):
    run = OpenMMRun(temperature_run, temperatures_K=LADDER)
    s = run.summary()
    assert s["source"] == "openmm" and s["n_states"] == 6
    assert (s["iterations"], s["rows"]) == (2000, 1000)
    assert s["report_interval_iterations"] == 2
    assert s["steps_per_iteration"] == STEPS
    assert s["time_ns_per_replica"] == pytest.approx(4.0, rel=1e-6)
    assert s["temperatures_K"] == pytest.approx(LADDER)
    assert all(0.3 < a < 1 for a in s["neighbour_acceptance"])
    assert s["round_trips"] > 10
    assert s["replicas_that_visited_every_state"] == 6
    assert "logged rows only" in s["notes"][0]
    text = resremd.format_openmm_summary(s)
    assert "6 states, 2000 iterations (1000 logged, every 2), 1000 steps " \
           "each, 4 ns per replica" in text


def test_an_openmm_run_weighted_gives_the_exact_populations(temperature_run):
    """Every state's frames, weighted to a rung (300 K) and between rungs
    (330 K), give the double well's exact populations."""
    run = OpenMMRun(temperature_run, temperatures_K=LADDER)
    for t in (300.0, 330.0):
        out = run.weights(temperature_K=t, discard_fraction=0.05)
        assert out["first_frame"] == 50
        assert all(len(w) == 950 for w in out["weights"].values())
        left, se = _left(temperature_run, out)
        assert se < 0.025
        assert abs(left - testsystems.left_fraction(t)) < 4.5 * se
    # A rung's temperature and the rung itself are the same target.
    for rung, t in enumerate(LADDER):
        by_state = run.weights(state=rung, discard_fraction=0.05)
        by_t = run.weights(temperature_K=t, discard_fraction=0.05)
        for k in range(6):
            assert np.allclose(by_state["weights"][k], by_t["weights"][k],
                               rtol=1e-8, atol=1e-14)
    by_state = run.weights(state=0, discard_fraction=0.05)
    # The free energies are MBAR's on the energies as written.
    u = run.energies[50:]
    f = solve(u.reshape(-1, 6).T, np.full(6, len(u)))
    assert np.allclose(by_state["free_energies"], f, atol=1e-4)


def test_an_openmm_run_of_hamiltonian_states(tmp_path):
    """States that differ in a parameter, not in temperature: weighted to
    one of them, exact; temperatures are refused."""
    system = openmm.System()
    system.addParticle(12.0)
    force = openmm.CustomExternalForce(
        f"lam*({testsystems.BARRIER}*(x^2-1)^2 + {testsystems.TILT}*x)"
        f" + 0.5*{testsystems.SPRING}*(y^2+z^2)")
    force.addGlobalParameter("lam", 1.0)
    force.addParticle(0, [])
    system.addForce(force)
    well = testsystems.double_well(-1.0)
    prepared = from_objects(system, well.topology, well.positions)
    lams = np.geomspace(1.0, 0.2, 5)  # like temperatures 300 to 1500 K
    path = _run(tmp_path / "run", [{"lam": float(x)} for x in lams],
                2000, prepared=prepared)
    run = OpenMMRun(path)
    out = run.weights(state=0, discard_fraction=0.05)
    left, se = _left(path, out)
    assert se < 0.08
    assert abs(left - testsystems.left_fraction(300.0)) < 4.5 * se
    assert run.temperature_only is False
    with pytest.raises(InputError, match="needs `temperatures_K`"):
        run.weights(temperature_K=300.0)
    # Temperatures are taken as given, but not to weight to.
    at = OpenMMRun(path, temperatures_K=[300] * 5)
    with pytest.raises(InputError, match="differ in more than temperature"):
        at.weights(temperature_K=300.0)
    assert run.summary()["temperatures_K"] is None


def test_a_resumed_openmm_run_reads_as_one(tmp_path):
    """A run OpenMM resumed reads as one; XTC trajectories too."""
    states = _temperature_states(LADDER[:3])
    _run(tmp_path / "run", states, 40, fmt="xtc")
    _run(tmp_path / "run", states, 40, fmt="xtc", resume=True, seed=4)
    run = OpenMMRun(tmp_path / "run", temperatures_K=LADDER[:3],
                    timestep_fs=2.0)
    assert run.iterations.tolist() == list(range(2, 82, 2))
    assert run.summary()["time_ns_per_replica"] == pytest.approx(0.16)
    assert len(run.weights(temperature_K=300.0)["weights"][0]) == 40


def test_frames_of_an_openmm_run_make_a_reservoir(temperature_run, tmp_path):
    pytest.importorskip("mdtraj")
    well = testsystems.double_well()
    app.PDBFile.writeFile(well.topology, well.positions * 10,
                          str(tmp_path / "top.pdb"))
    resremd.import_reservoir(
        trajectories=[str(temperature_run / "state_5.dcd")],
        topology=str(tmp_path / "top.pdb"), temperature_K=LADDER[-1],
        output=str(tmp_path / "reservoir"))
    reservoir = resremd.Reservoir.open(tmp_path / "reservoir")
    assert reservoir.n_frames == 1000
    assert reservoir.temperature_K == pytest.approx(LADDER[-1])


@pytest.fixture
def small(tmp_path_factory):
    path = _run(tmp_path_factory.mktemp("omm") / "small",
                _temperature_states(LADDER[:3]), 10, every=1, volume=True,
                steps=250)
    return path


def _copy(small, tmp_path):
    shutil.copytree(small, tmp_path / "copy")
    return tmp_path / "copy"


def _write(path, states, u=None, v=None, *, every=1, steps=250,
           iterations=None):
    """A run's files as OpenMM's reporter writes them."""
    path.mkdir(exist_ok=True)
    states = np.asarray(states)
    n = states.shape[1]
    if iterations is None:
        iterations = [every * (i + 1) for i in range(len(states))]
    (path / "log.csv").write_text(
        ",".join(["Iteration", "Step"] + [f"Replica_{i}_State"
                                          for i in range(n)]) + "\n"
        + "".join(f"{it},{steps * it}," + ",".join(map(str, row)) + "\n"
                  for it, row in zip(iterations, states)))
    for name, data in (("energy.csv", u), ("volume.csv", v)):
        if data is not None:
            (path / name).write_text("".join(
                ",".join(repr(float(x)) for x in np.ravel(row)) + "\n"
                for row in data))
    return path


#: Six rows of three replicas. Replica 0 goes 0, 1, 2, 0 (a trip of 3
#: iterations), then 1, 0 (no trip); replica 1 goes 1, 0, 0, 2, 0 (a trip
#: of 2, from the later 0); replica 2 never reaches state 0.
STATES = [[0, 1, 2], [1, 0, 2], [2, 0, 1], [0, 2, 1], [1, 0, 2], [0, 1, 2]]


def test_exchange_statistics_by_hand(tmp_path):
    rng = np.random.default_rng(4)
    u = rng.normal(0.0, 2.0, size=(6, 3, 3))
    u[2, :, 1] = np.nan          # one row's pair is left out
    run = OpenMMRun(_write(tmp_path / "a", STATES, u), fixed_box=True)
    s = run.summary()
    assert (s["round_trips"], s["mean_round_trip_iterations"]) == (2, 2.5)
    assert s["replicas_that_visited_every_state"] == 2
    text = resremd.format_openmm_summary(s)
    assert "round trips first-last-first state: 2 (mean 2 iterations)" in text
    assert "replicas that visited every state: 2" in text
    assert s["report_interval_iterations"] == 1 and s["notes"] == []
    assert s["steps_per_iteration"] == 250
    # The sampler's own rule: replica i in s_i, j in s_j swap with
    # min(1, exp((E_i(s_i) - E_j(s_i)) / kT_si + (E_j(s_j) - E_i(s_j))
    # / kT_sj)), here in reduced energies.
    for k in (0, 1):
        p = []
        for row, held in zip(u, STATES):
            i, j = held.index(k), held.index(k + 1)
            x = row[i, k] - row[j, k] + row[j, k + 1] - row[i, k + 1]
            if np.isfinite(x):
                p.append(min(1.0, np.exp(x)))
        assert s["neighbour_acceptance"][k] == pytest.approx(np.mean(p))
    # A pair with no finite energies has no estimate, and JSON stays valid.
    u[:, :, 2] = np.nan
    s = OpenMMRun(_write(tmp_path / "b", STATES, u)).summary()
    assert s["neighbour_acceptance"][1] is None
    import json

    json.dumps(s, allow_nan=False)
    assert "n/a" in resremd.format_openmm_summary(s)
    # One row: no interval, no steps per iteration; one state.
    one = OpenMMRun(_write(tmp_path / "c", [[0]], np.zeros((1, 1, 1))))
    s = one.summary()
    assert s["report_interval_iterations"] is None
    assert s["steps_per_iteration"] is None
    text = resremd.format_openmm_summary(s)
    assert "1 state, 1 iteration (" in text and "none (one state)" in text
    # Every second iteration logged: trips in iterations, and a note.
    s = OpenMMRun(_write(tmp_path / "d", STATES, every=2)).summary()
    assert s["mean_round_trip_iterations"] == 5.0
    assert s["report_interval_iterations"] == 2
    assert "logged rows only" in s["notes"][0]
    assert "cannot be told" in s["notes"][1]
    # Uneven rows: no report interval, but steps per iteration still.
    s = OpenMMRun(_write(tmp_path / "e", STATES,
                         iterations=[1, 2, 4, 5, 9, 10])).summary()
    assert s["report_interval_iterations"] is None
    assert s["steps_per_iteration"] == 250
    assert s["mean_round_trip_iterations"] == 4.5  # (5 - 1 + 9 - 4) / 2


def test_weights_of_some_states_by_hand(tmp_path):
    """MBAR weights of chosen states' frames, against the formula."""
    rng = np.random.default_rng(5)
    states = [rng.permutation(3).tolist() for _ in range(300)]
    u = rng.normal(0.0, 1.0, size=(300, 3, 3)) + np.array([0.0, 0.4, 0.9])
    with pytest.raises(InputError, match="cannot be told"):
        OpenMMRun(_write(tmp_path / "a", states, u)).weights(state=1)
    run = OpenMMRun(tmp_path / "a", fixed_box=True)
    out = run.weights(state=1, states=[2], frames=(100, 300))
    f = np.array(out["free_energies"])
    assert np.allclose(f, solve(u[100:].reshape(-1, 3).T, np.full(3, 200)),
                       atol=1e-4)
    held = np.argsort(np.array(states), axis=1)[100:, 2]
    x = u[100:][np.arange(200), held]          # state 2's frames
    log_w = -x[:, 1] - np.log(200 * np.exp(f[2] - x[:, 2]))
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    assert out["states"] == [2] and out["first_frame"] == 100
    assert np.allclose(out["weights"][2], w, rtol=1e-8)
    assert out["effective_samples"] == pytest.approx(1 / np.sum(w * w))
    # A row without a finite energy is named.
    u[150, 0, 0] = np.inf
    bad = OpenMMRun(_write(tmp_path / "b", states, u), fixed_box=True)
    for frames in (None, (100, 300)):
        with pytest.raises(InputError, match="row 151, has an energy"):
            bad.weights(state=0, frames=frames)
    assert len(bad.weights(state=0, frames=(151, 300))["weights"][0]) == 149
    u[:] = np.nan
    none = OpenMMRun(_write(tmp_path / "c", states, u), fixed_box=True)
    with pytest.raises(InputError, match="no row of finite energies"):
        none.weights(state=0)


def test_damaged_openmm_files_are_named(small, tmp_path):
    temps = LADDER[:3]
    run = _copy(small, tmp_path)
    energy = (run / "energy.csv").read_text()
    log = (run / "log.csv").read_text()
    lines = log.splitlines()
    # A line still being written (no newline yet), in any file.
    for name, text in (("energy.csv", energy), ("log.csv", log),
                       ("volume.csv", (run / "volume.csv").read_text())):
        (run / name).write_text(text[:-4])
        with pytest.raises(InputError, match=f"{name}, line 1[01], is only "
                                             "partly written. If the run"):
            OpenMMRun(run)
        (run / name).write_text(text)
    # Blank lines at the end, Windows line ends and a byte-order mark are
    # all OpenMM's lines still.
    (run / "log.csv").write_text("\ufeff" + log.replace("\n", "\r\n")
                                 + "\n\n")
    assert len(OpenMMRun(run).iterations) == 10
    (run / "log.csv").write_text(log)
    cut = energy.splitlines()
    cut[-1] = ",".join(cut[-1].split(",")[:5])
    (run / "energy.csv").write_text("\n".join(cut) + "\n")
    with pytest.raises(InputError, match=r"energy.csv, line 10, is not 9"):
        OpenMMRun(run)
    final = energy.splitlines()[-1]
    for last in (final + ",1.0", re.sub(r"[^,]+$", "x", final),
                 final + "#x"):
        (run / "energy.csv").write_text(
            "\n".join(energy.splitlines()[:-1] + [last]) + "\n")
        with pytest.raises(InputError, match=r"energy.csv, line 10, is not "
                                             "9"):
            OpenMMRun(run)
    (run / "energy.csv").write_text(energy + energy.splitlines()[-1] + "\n")
    with pytest.raises(InputError, match="11 rows and log.csv 10.*read it "
                                         "again"):
        OpenMMRun(run)
    (run / "energy.csv").write_text(energy)
    volume = (run / "volume.csv").read_text()
    (run / "volume.csv").write_text("".join(volume.splitlines(True)[:-1]))
    with pytest.raises(InputError, match="volume.csv has 9 rows"):
        OpenMMRun(run)
    for bad in ("nan", "-8.0", "0.0"):
        (run / "volume.csv").write_text(volume.replace("8.0", bad, 1))
        with pytest.raises(InputError, match="volume.csv, line 1, has a "
                                             "volume that is not"):
            OpenMMRun(run)
    (run / "volume.csv").write_text(volume)
    (run / "log.csv").write_text("\n".join(lines[:3] + [lines[2]]
                                           + lines[4:]) + "\n")
    with pytest.raises(InputError, match="does not count iterations upward"):
        OpenMMRun(run)
    head, *rest = lines
    (run / "log.csv").write_text("\n".join([head] + rest[:-1]
                                           + ["10,2500,0,1"]) + "\n")
    with pytest.raises(InputError, match="line 11, is not 5"):
        OpenMMRun(run)
    (run / "log.csv").write_text("\n".join([head] + rest[:-1] + [
        ",".join(rest[-1].split(",")[:2] + ["0", "0", "1"])]) + "\n")
    with pytest.raises(InputError, match="line 11, does not give each"):
        OpenMMRun(run)
    (run / "log.csv").write_text("\n".join([head] + rest[:-1] + [
        ",".join(rest[-1].split(",")[:2] + ["0", "1", "9" * 30])]) + "\n")
    with pytest.raises(InputError, match="line 11, is not 5"):
        OpenMMRun(run)
    for text in (log.replace("Replica_2_State", "Replica_2"), "",
                 "Iteration,Step\n"):
        (run / "log.csv").write_text(text)
        with pytest.raises(InputError, match="does not start with the "
                                             "header"):
            OpenMMRun(run)
    (run / "log.csv").write_text(lines[0] + "\n")
    with pytest.raises(InputError, match="no iterations yet"):
        OpenMMRun(run)
    (run / "log.csv").write_bytes(log.encode() + b"\xff\xfe\n")
    with pytest.raises(InputError, match="log.csv could not be read"):
        OpenMMRun(run)
    (run / "log.csv").write_text(log)
    (run / "energy.csv").unlink()
    (run / "energy.csv").mkdir()
    with pytest.raises(InputError, match="energy.csv could not be read"):
        OpenMMRun(run)
    (run / "energy.csv").rmdir()
    (run / "energy.csv").write_text(energy)
    # The sampler's temperatures, exactly.
    with pytest.raises(InputError, match=r"in the ratios of .* not at .* "
                                         "exactly"):
        OpenMMRun(run, temperatures_K=[temps[0], temps[1] * 1.01,
                                       temps[2]])
    with pytest.raises(InputError, match="2 temperatures for the run's 3"):
        OpenMMRun(run, temperatures_K=temps[:2])
    with pytest.raises(InputError, match="written with a 2 fs timestep"):
        OpenMMRun(run, timestep_fs=1.0)
    # A row more, or a frame more, than the trajectories hold.
    dcd = (run / "state_0.dcd").read_bytes()
    (run / "log.csv").write_text(log + "11,2750,0,1,2\n")
    (run / "energy.csv").write_text(energy + energy.splitlines()[-1] + "\n")
    (run / "volume.csv").write_text(volume + volume.splitlines()[-1] + "\n")
    extra = OpenMMRun(run, temperatures_K=temps)
    with pytest.raises(InputError, match="state_0.dcd has 10 frames for the "
                                         "11 rows"):
        extra.weights(state=0)
    (run / "log.csv").write_text("".join(log.splitlines(True)[:-1]))
    (run / "energy.csv").write_text("".join(energy.splitlines(True)[:-1]))
    (run / "volume.csv").write_text("".join(volume.splitlines(True)[:-1]))
    fewer = OpenMMRun(run, temperatures_K=temps)
    with pytest.raises(InputError, match="state_0.dcd has 10 frames for the "
                                         "9 rows"):
        fewer.weights(state=0)
    for size in (50, 150):
        (run / "state_0.dcd").write_bytes(dcd[:size])
        with pytest.raises(InputError, match="state_0.dcd is not a DCD"):
            OpenMMRun(run)
    # A frame cut short (the run stopped while writing it) is left out.
    (run / "state_0.dcd").write_bytes(dcd[:-5])
    assert OpenMMRun(run).trajectory_frames[0] == 9
    # Without energies: a summary, but no weights.
    run = _copy(small, tmp_path / "b")
    (run / "energy.csv").unlink()
    bare = OpenMMRun(run, temperatures_K=temps)
    assert bare.summary()["neighbour_acceptance"] is None
    assert "not available" in resremd.format_openmm_summary(bare.summary())
    for target in ({"state": 0}, {"temperature_K": 300.0}):
        with pytest.raises(InputError, match="no energy.csv"):
            bare.weights(**target)
    with pytest.raises(InputError, match="no log.csv"):
        OpenMMRun(tmp_path)
    from resremd.openmm_runs import is_openmm_run

    (run / "manifest.json").write_text("{}")
    assert not is_openmm_run(run)
    with pytest.raises(InputError, match="Give either"):
        OpenMMRun(small).weights()


def test_wrong_settings_for_an_openmm_run_are_refused(small):
    temps = LADDER[:3]
    with pytest.raises(InputError, match="numbers above 0"):
        OpenMMRun(small, temperatures_K=["hot", 1, 2])
    with pytest.raises(InputError, match="numbers above 0"):
        OpenMMRun(small, temperatures_K=[0, 1, 2])
    with pytest.raises(InputError, match="`timestep_fs` must be a number"):
        OpenMMRun(small, timestep_fs=-2)
    run = OpenMMRun(small, temperatures_K=temps)
    with pytest.raises(InputError, match="a number above 0"):
        run.weights(temperature_K=-1)
    with pytest.raises(InputError, match="`state` must be among 0..2"):
        run.weights(state=3)
    with pytest.raises(InputError, match="`states` must be among"):
        run.weights(state=0, states=[5])
    with pytest.raises(InputError, match="No rows in 5..3"):
        run.weights(state=0, frames=(5, 3))


def test_a_resremd_run_is_not_read_as_an_openmm_run(tmp_path, fast):
    resremd.run(testsystems.double_well(), output=str(tmp_path / "r"),
                temperatures_K=[300.0, 360.0], production_steps=200, **fast)
    with pytest.raises(InputError, match="is a ResREMD run"):
        OpenMMRun(tmp_path / "r")
    from resremd.openmm_runs import is_openmm_run

    assert not is_openmm_run(tmp_path / "r")


def test_a_run_that_does_not_say_its_box(tmp_path):
    """XTC frames, no volume.csv and no checkpoints: whether the box
    changed is not known, so weights across temperatures wait to be told."""
    temps = LADDER[:3]
    path = _run(tmp_path / "x", _temperature_states(temps), 10, every=1,
                fmt="xtc", checkpoints=False, steps=250)
    run = OpenMMRun(path, temperatures_K=temps)
    assert run.fixed_box is None
    assert "cannot be told" in run.summary()["notes"][0]
    for target in ({"state": 0}, {"temperature_K": 300.0}):
        with pytest.raises(InputError, match="cannot be told"):
            run.weights(**target)
    fixed = OpenMMRun(path, temperatures_K=temps, fixed_box=True)
    assert fixed.fixed_box is True
    assert len(fixed.weights(temperature_K=300.0)["weights"][0]) == 10
    # An XTC file is no DCD file.
    (path / "state_0.xtc").rename(path / "state_0.dcd")
    with pytest.raises(InputError, match="not a DCD file written by OpenMM"):
        OpenMMRun(path)


def test_a_hamiltonian_scaled_copy_at_one_temperature(tmp_path):
    """States whose energies are one another's scaled copies read as
    temperature states; at one temperature they are weighted to a state."""
    rng = np.random.default_rng(6)
    e = rng.normal(0.0, 3.0, size=(50, 2))
    u = np.stack([e, 0.6 * e], axis=2) / (GAS_CONSTANT * 300.0)
    states = [rng.permutation(2).tolist() for _ in range(50)]
    _write(tmp_path, states, u)
    with pytest.raises(InputError, match="scaled copies of one another, "
                                         "leave them out"):
        OpenMMRun(tmp_path, temperatures_K=[300, 300])
    run = OpenMMRun(tmp_path, fixed_box=True)
    assert run.temperature_only
    assert len(run.weights(state=1)["weights"][1]) == 50
    # Read the same however large the energies, without a warning.
    import warnings

    for scale in (1e-6, 1e200):
        _write(tmp_path / str(scale), states, u * scale)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert OpenMMRun(tmp_path / str(scale)).temperature_only
    # The first temperature is taken as given: a ladder whose ratios fit.
    assert OpenMMRun(tmp_path, temperatures_K=[300, 500]).temperatures[1] \
        == 500


def test_the_timestep_of_a_long_dcd(tmp_path):
    """DCDFile folds its interval into the timestep past 2^31 steps; the
    timestep read stays the integrator's."""
    well = testsystems.double_well()
    _write(tmp_path, [[0, 1], [1, 0]], steps=1000)
    with open(tmp_path / "state_0.dcd", "wb") as fh:
        dcd = app.DCDFile(fh, well.topology, 0.002 * unit.picoseconds,
                          firstStep=2 ** 31 - 1500, interval=1000)
        for _ in range(2):
            dcd.writeModel(well.positions * unit.nanometer)
    run = OpenMMRun(tmp_path)
    assert run.timestep_ps == pytest.approx(0.002, rel=1e-6)
    _write(tmp_path, [[0, 1], [1, 0], [0, 1]], iterations=[1, 2, 4],
           steps=1000)
    assert OpenMMRun(tmp_path).timestep_ps is None


def test_summary_of_an_openmm_run_on_the_command_line(small, capsys, caplog,
                                                      tmp_path, fast):
    import json

    from resremd.cli import main

    assert main(["summary", str(small), "--temperatures-K",
                 *map(str, LADDER[:3])]) == 0
    out = capsys.readouterr().out
    assert out.startswith("OpenMM ReplicaExchangeSampler run")
    assert "temperatures (K): 300, 413.919, 571.096" in out
    assert main(["summary", str(small), "--timestep-fs", "2"]) == 0
    assert "ns per replica" in capsys.readouterr().out
    assert main(["summary", str(small), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["n_states"] == 3
    assert main(["summary", str(small), "--temperatures-K", "1", "2",
                 "3"]) == 2
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "log.csv").write_text("")
    assert main(["summary", str(tmp_path / "x")]) == 2
    assert "does not start with the header" in caplog.text
    assert main(["summary", str(tmp_path / "none")]) == 2
    assert "holds neither a ResREMD run" in caplog.text
    resremd.run(testsystems.double_well(), output=str(tmp_path / "r"),
                temperatures_K=[300.0, 360.0], production_steps=200, **fast)
    for flag in (["--temperatures-K", "300"], ["--timestep-fs", "2"]):
        with pytest.raises(SystemExit):
            main(["summary", str(tmp_path / "r"), *flag])
    assert "are for a run of OpenMM's sampler" in capsys.readouterr().err


def test_an_openmm_run_at_constant_pressure_is_summarised_only(tmp_path):
    """Constant pressure is seen in volume.csv, the DCD boxes and the
    checkpoints, each alone; such a run is summarised, and MBAR refused."""
    temps = [100.0, 110.0, 121.0]
    path = _run(tmp_path / "npt", _temperature_states(temps), 6, every=1,
                volume=True, prepared=testsystems.lj_box(pressure=True),
                temperature=100.0, steps=20)
    run = OpenMMRun(path, temperatures_K=temps)
    assert run.fixed_box is False
    s = run.summary()
    assert s["fixed_box"] is False
    assert "constant pressure" in s["notes"][0]
    assert "not all in one fixed box" in s["notes"][1]
    for target in ({"state": 0}, {"temperature_K": 105.0}):
        with pytest.raises(InputError, match="their boxes differ"):
            run.weights(**target)
    with pytest.raises(InputError, match="`fixed_box` was given, but it is "
                                         "contradicted by volume.csv"):
        OpenMMRun(path, fixed_box=True)
    kept = tmp_path / "kept"
    shutil.copytree(path, kept)
    for i in range(3):
        (path / f"checkpoint_{i}.xml").unlink()
    with pytest.raises(InputError, match="their boxes differ"):
        OpenMMRun(path).weights(state=0)                  # volume.csv
    assert "has a barostat" in OpenMMRun(path).summary()["notes"][0]
    with pytest.raises(InputError, match="contradicted by volume.csv"):
        OpenMMRun(path, fixed_box=True)
    (path / "volume.csv").unlink()
    assert OpenMMRun(path).fixed_box is False             # the DCD boxes
    with pytest.raises(InputError, match="contradicted by the DCD boxes"):
        OpenMMRun(path, fixed_box=True)
    for k in range(3):
        (path / f"state_{k}.dcd").unlink()
    assert OpenMMRun(path).fixed_box is None
    for i in range(3):
        shutil.copy(kept / f"checkpoint_{i}.xml", path)
    assert OpenMMRun(path).fixed_box is False             # the checkpoints
    with pytest.raises(InputError, match="contradicted by the checkpoints' "
                                         "boxes"):
        OpenMMRun(path, fixed_box=True)
    # A checkpoint of another moment still names the barostat.
    stale = (path / "checkpoint_2.xml").read_text()
    (path / "checkpoint_2.xml").write_text(
        re.sub(r'stepCount="\d+"', 'stepCount="1"', stale))
    assert OpenMMRun(path).fixed_box is False


def test_a_barostat_that_never_moved_the_box(tmp_path):
    """Barostat parameters in a context whose box never changed (here a
    barostat disabled with frequency 0): one fixed box, with a note."""
    box = testsystems.lj_box(pressure=True)
    for force in box.system.getForces():
        if isinstance(force, openmm.MonteCarloBarostat):
            force.setFrequency(0)
    temps = [100.0, 110.0]
    path = _run(tmp_path / "off", _temperature_states(temps), 4, every=1,
                volume=True, prepared=box, temperature=100.0, steps=20)
    run = OpenMMRun(path, temperatures_K=temps)
    assert run.fixed_box is True
    assert "did not change" in run.summary()["notes"][0]
    assert len(run.weights(temperature_K=105.0)["weights"][0]) == 4


def test_one_fixed_box_is_read_from_what_does_not_change(tmp_path):
    temps = LADDER[:3]
    path = _run(tmp_path / "nvt", _temperature_states(temps), 4, every=1,
                volume=True, prepared=testsystems.lj_box(), steps=20,
                checkpoints=False)
    assert OpenMMRun(path).fixed_box is True              # volume.csv
    (path / "volume.csv").unlink()
    assert OpenMMRun(path).fixed_box is True              # the DCD boxes
    for k in range(3):
        (path / f"state_{k}.dcd").unlink()
    assert OpenMMRun(path).fixed_box is None
    run = _run(tmp_path / "cp", _temperature_states(temps), 2, every=1,
               prepared=testsystems.lj_box(), steps=20, fmt="xtc")
    assert OpenMMRun(run).fixed_box is True               # the checkpoints
    # ... but not those of another moment, which only count against.
    stale = (run / "checkpoint_1.xml").read_text()
    (run / "checkpoint_1.xml").write_text(
        re.sub(r'stepCount="\d+"', 'stepCount="1"', stale))
    assert OpenMMRun(run).fixed_box is None
    # Replicas in boxes of their own, each fixed, are not one fixed box.
    _write(tmp_path / "own", [[0, 1], [1, 0]], np.zeros((2, 2, 2)),
           [[8.0, 9.0], [8.0, 9.0]])
    own = OpenMMRun(tmp_path / "own")
    assert own.fixed_box is False
    assert not any("barostat" in n for n in own.summary()["notes"])


def _checkpoint_dir(tmp_path, texts):
    path = _write(tmp_path, [[0, 1], [1, 0]], np.zeros((2, 2, 2)))
    for i, text in enumerate(texts):
        (path / f"checkpoint_{i}.xml").write_text(text)
    return path


STATE = ('<?xml version="1.0" ?>\n<State openmmVersion="8.6.1" '
         'stepCount="500" time="1">\n<PeriodicBoxVectors>\n<A x="{}" y="0" '
         'z="0"/>\n<B x="0" y="2" z="0"/>\n<C x="0" y="0" z="2"/>\n'
         '</PeriodicBoxVectors>\n<Parameters{}/>\n<Positions>\n'
         '</Positions>\n</State>\n')


@pytest.mark.parametrize("parameters, pressure", [
    ("", False),
    (' MonteCarloPressure="1" MonteCarloTemperature="300"', True),
    (' MonteCarloPressureX="1" AnisotropicMonteCarloTemperature="300"', True),
    (' MembraneMonteCarloSurfaceTension="0"', True),
])
def test_a_barostat_in_the_checkpoints(tmp_path, parameters, pressure):
    path = _checkpoint_dir(tmp_path, [STATE.format(2, ""),
                                      STATE.format(2, parameters)])
    assert OpenMMRun(path).fixed_box is (not pressure)


def test_barostat_parameters_alone(tmp_path):
    """Checkpoints that name a barostat, in one box, and nothing else: not
    one fixed box, unless told it was."""
    path = _checkpoint_dir(tmp_path, [
        STATE.format(2, ' MonteCarloPressure="1"')] * 2)
    run = OpenMMRun(path)
    assert run.fixed_box is False
    with pytest.raises(InputError, match="the checkpoints name a barostat; "
                                         "give `fixed_box=True` if it was "
                                         "disabled"):
        run.weights(state=0)
    told = OpenMMRun(path, fixed_box=True)
    assert told.fixed_box is True
    assert "did not change" in told.summary()["notes"][-1]


def test_replicas_in_boxes_of_their_own_by_checkpoint(tmp_path):
    path = _checkpoint_dir(tmp_path, [STATE.format(2, ""),
                                      STATE.format(2.5, "")])
    assert OpenMMRun(path).fixed_box is False


def test_damaged_checkpoints_are_named(tmp_path):
    good = STATE.format(2, "")
    for text, said in (
            (good[:60], "is not a checkpoint of OpenMM's sampler, or is "
                        "only partly written"),
            (good.replace("<Parameters/>", ""), "is not a checkpoint of "),
            ("<Thing/>\n", "is not a State checkpoint"),
            (STATE.format(2, ' MonteCarloPressure="high"'),
             "has a step count, time, box or parameter that is not"),
            (good.replace('stepCount="500" ', ""), "has a step count"),
            (good.replace('x="2"', 'x="two"'), "has a step count"),
            (good.replace('x="2"', 'x="nan"'), "has a step count"),
            (good.replace('time="1"', 'time="nan"'), "has a step count"),
            (good.replace('time="1"', 'time="-1"'), "has a step count"),
            (good.replace('x="2"', 'x="0"'), "has a step count"),
            (good.replace('<C x="0" y="0" z="2"/>', ""), "has a step count"),
            (good.replace("<State ", "<Statex "), "is not a State")):
        path = _checkpoint_dir(tmp_path, [good, text])
        with pytest.raises(InputError, match="checkpoint_1.xml " + said):
            OpenMMRun(path)


def _dcd_dir(tmp_path, *, n=2, box=None, steps=1000, interval=1000,
             first=1000, frames=2, dt=0.002):
    """A log of n states and a DCD of each, written by DCDFile."""
    well = testsystems.lj_box() if box is not None \
        else testsystems.double_well()
    path = _write(tmp_path, [list(range(n))] * frames, np.zeros(
        (frames, n, n)), steps=steps)
    for k in range(n):
        with open(path / f"state_{k}.dcd", "wb") as fh:
            dcd = app.DCDFile(fh, well.topology, dt * unit.picoseconds,
                              firstStep=first, interval=interval)
            for f in range(frames):
                vectors = None if box is None else \
                    np.eye(3) * box[f] * unit.nanometer
                dcd.writeModel(well.positions * unit.nanometer,
                               periodicBoxVectors=vectors)
    return path


def test_damaged_dcd_headers_are_named(tmp_path):
    path = _dcd_dir(tmp_path)
    good = (path / "state_0.dcd").read_bytes()
    for offset, value, said in ((8, -1, "is not a DCD file"),
                                (92, -4, "is not a DCD file"),
                                (268, 0, "is not a DCD file")):
        bad = bytearray(good)
        bad[offset:offset + 4] = struct.pack("<i", value)
        (path / "state_0.dcd").write_bytes(bytes(bad))
        with pytest.raises(InputError, match="state_0.dcd " + said):
            OpenMMRun(path)
    # A header ahead of the file: only whole frames count.
    bad = bytearray(good)
    bad[8:12] = struct.pack("<i", 3)
    (path / "state_0.dcd").write_bytes(bytes(bad))
    assert OpenMMRun(path).trajectory_frames == {0: 2, 1: 2}
    # A timestep that is not a number above 0 is not used.
    bad = bytearray(good)
    bad[44:48] = struct.pack("<f", float("nan"))
    (path / "state_0.dcd").write_bytes(bytes(bad))
    (path / "state_1.dcd").write_bytes(bytes(bad))
    assert OpenMMRun(path).timestep_ps is None
    # Boxes that change within each file, every file starting alike.
    assert OpenMMRun(_dcd_dir(tmp_path / "v", box=[2.5, 2.6])).fixed_box \
        is False
    # Files that each hold one box, but not the same one.
    a = _dcd_dir(tmp_path / "a1", box=[2.5, 2.5])
    b = _dcd_dir(tmp_path / "b1", box=[2.6, 2.6])
    (a / "state_1.dcd").write_bytes((b / "state_1.dcd").read_bytes())
    assert OpenMMRun(a).fixed_box is False
    # A DCD with a box and no frames yet.
    empty = _dcd_dir(tmp_path / "e", box=[2.5], frames=1)
    (empty / "state_0.dcd").write_bytes(
        (empty / "state_0.dcd").read_bytes()[:280])
    assert OpenMMRun(empty).trajectory_frames[0] == 0
    # One state's boxes alone do not show one fixed box.
    fixed = _dcd_dir(tmp_path / "f", box=[2.5, 2.5])
    assert OpenMMRun(fixed).fixed_box is True
    (fixed / "state_1.dcd").unlink()
    assert OpenMMRun(fixed).fixed_box is None
    # A box that is not a box: frame 2's first length made negative.
    path = _dcd_dir(tmp_path / "box", box=[2.5, 2.6])
    bad = bytearray((path / "state_0.dcd").read_bytes())
    title = struct.unpack("<i", bad[92:96])[0]
    atoms = struct.unpack("<i", bad[104 + title:108 + title])[0]
    at = 112 + title + 56 + 3 * (8 + 4 * atoms) + 4
    assert struct.unpack("<d", bad[at:at + 8])[0] == pytest.approx(26.0)
    bad[at:at + 8] = struct.pack("<d", -26.0)
    (path / "state_0.dcd").write_bytes(bytes(bad))
    with pytest.raises(InputError, match="frame 2, has a box that is not"):
        OpenMMRun(path)


def test_the_timestep_from_dcd_headers(tmp_path):
    """Taken when every header agrees with the log's steps between rows,
    and with the checkpoints' time."""
    path = _dcd_dir(tmp_path / "a")
    assert OpenMMRun(path).timestep_ps == 0.002
    # Checkpoints that agree; that do not (the timestep changed on a
    # resume); and of another moment (there was a resume).
    for time, steps, timestep in (("4", "2000", 0.002), ("6", "2000", None),
                                  ("4", "1000", None)):
        for i in range(2):
            text = STATE.format(2, "").replace(
                'stepCount="500"', f'stepCount="{steps}"')
            (path / f"checkpoint_{i}.xml").write_text(
                text.replace('time="1"', f'time="{time}"'))
        assert OpenMMRun(path).timestep_ps == timestep
    assert OpenMMRun(path, timestep_fs=3).timestep_ps == 0.003
    # Current checkpoints whose time agrees but for round-off, or
    # disagrees by 0.1%; and a timestep then given.
    for time, kept in (("4.0000001", True), ("4.004", False)):
        for i in range(2):
            text = STATE.format(2, "").replace('stepCount="500"',
                                               'stepCount="2000"')
            (path / f"checkpoint_{i}.xml").write_text(
                text.replace('time="1"', f'time="{time}"'))
        run = OpenMMRun(path)
        assert (run.timestep_ps == 0.002) is kept
        assert any("header's timestep is not used" in n
                   for n in run.summary()["notes"]) is not kept
    given = OpenMMRun(path, timestep_fs=2)
    assert given.timestep_ps == 0.002
    assert "takes every step to be 2 fs" in given.summary()["notes"][-1]
    # A timestep given that is 1% off the header's.
    for i in range(2):
        (path / f"checkpoint_{i}.xml").unlink()
    with pytest.raises(InputError, match="written with a 2 fs timestep"):
        OpenMMRun(path, timestep_fs=2.02)
    # An interval that is not the log's: unknown.
    run = OpenMMRun(_dcd_dir(tmp_path / "b", interval=500))
    assert run.timestep_ps is None
    # Headers that disagree: unknown.
    path = _dcd_dir(tmp_path / "c")
    other = _dcd_dir(tmp_path / "d", dt=0.004)
    (path / "state_1.dcd").write_bytes((other / "state_1.dcd").read_bytes())
    assert OpenMMRun(path).timestep_ps is None


def test_wrong_arguments_to_weights_are_refused(tmp_path):
    rng = np.random.default_rng(7)
    states = [rng.permutation(3).tolist() for _ in range(20)]
    run = OpenMMRun(_write(tmp_path, states, rng.normal(
        size=(20, 3, 3))), fixed_box=True)
    for kwargs, said in (({"state": "x"}, "`state` must be a whole number"),
                         ({"state": 1.7}, "`state` must be a whole number"),
                         ({"state": 0, "states": ["a"]}, "`states` must"),
                         ({"state": 0, "frames": (0.5, 3)}, "`frames` must"),
                         ({"state": 0, "frames": (0, 1, 2)}, "`frames` must"),
                         ({"state": 0, "frames": 5}, "`frames` must"),
                         ({"state": 0, "discard_fraction": "a"},
                          "`discard_fraction` must"),
                         ({"state": 0, "discard_fraction": 1.0},
                          "`discard_fraction` must"),
                         ({"state": True}, "`state` must be a whole number"),
                         ({"state": -1}, "`state` must be among 0..2"),
                         ({"state": 0, "states": [0, 0]}, "each once"),
                         ({"state": 0, "states": [-1]}, "each once"),
                         ({"state": 0, "states": 1}, "a list of states"),
                         ({"state": 0, "frames": (-1, 5)}, "No rows in"),
                         ({"state": 0, "frames": (0, 5),
                           "discard_fraction": 0.1}, "not both"),
                         ({"state": 0, "temperature_K": 300.0},
                          "Give either")):
        with pytest.raises(InputError, match=said):
            run.weights(**kwargs)
    out = run.weights(state=np.int64(1), frames=(np.int64(2), 20))
    assert out["first_frame"] == 2 and out["state"] == 1
    with pytest.raises(InputError, match="4 temperatures for the run's 3"):
        OpenMMRun(tmp_path, temperatures_K=[300, 310, 320, 330])
    with pytest.raises(InputError, match="`timestep_fs` must be a number"):
        OpenMMRun(tmp_path, timestep_fs=float("inf"))
    with pytest.raises(InputError, match="`fixed_box` must be True or"):
        OpenMMRun(tmp_path, fixed_box="no")
    # States that do not overlap at all: MBAR says so.
    u = rng.normal(size=(20, 3, 3))
    u[:, :, 2] = 1e308
    far = OpenMMRun(_write(tmp_path / "far", states, u), fixed_box=True)
    with pytest.raises(InputError):
        far.weights(state=0)


def test_weights_to_a_temperature_by_hand(tmp_path):
    rng = np.random.default_rng(8)
    temps = np.array([300.0, 340.0])
    states = np.array([rng.permutation(2) for _ in range(200)])
    e = rng.normal(-50 + 0.1 * temps[states], 2.0)
    u = e[:, :, None] / (GAS_CONSTANT * temps)
    run = OpenMMRun(_write(tmp_path, states, u), temperatures_K=temps,
                    fixed_box=True)
    out = run.weights(temperature_K=320.0)
    f = out["free_energies"]
    held = np.argsort(states, axis=1)
    x = np.concatenate([e[np.arange(200), held[:, k]] for k in (0, 1)])
    log_w = -x / (GAS_CONSTANT * 320.0) - np.logaddexp(
        np.log(200) + f[0] - x / (GAS_CONSTANT * temps[0]),
        np.log(200) + f[1] - x / (GAS_CONSTANT * temps[1]))
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    assert np.allclose(np.concatenate([out["weights"][0],
                                       out["weights"][1]]), w, rtol=1e-8)


def test_a_run_resumed_with_other_states(tmp_path):
    """Temperature states that took other temperatures partway through:
    one part at a time."""
    rng = np.random.default_rng(9)
    states = np.array([rng.permutation(2) for _ in range(40)])
    e = rng.normal(-50.0, 2.0, size=(40, 2))
    temps = np.where(np.arange(40)[:, None] < 25, [300.0, 330.0],
                     [300.0, 360.0])
    u = e[:, :, None] / (GAS_CONSTANT * temps[:, None, :])
    run = OpenMMRun(_write(tmp_path, states, u), fixed_box=True)
    assert run.state_changes == [25]
    # A change of half a percent is found too; a term of a millionth that
    # is no scaling makes the states no temperature states at all.
    small = e[:, :, None] / (GAS_CONSTANT * np.where(
        np.arange(40)[:, None] < 25, [300.0, 330.0], [300.0, 331.65])[
            :, None, :])
    assert OpenMMRun(_write(tmp_path / "s", states, small),
                     fixed_box=True).state_changes == [25]
    other = e[:, :, None] / (GAS_CONSTANT * np.array([300.0, 330.0]))
    other[:, :, 1] *= 1 + 1e-6 * rng.choice([-1, 1], size=(40, 2))
    mixed = OpenMMRun(_write(tmp_path / "m", states, other), fixed_box=True)
    assert mixed.temperature_only is False and mixed.state_changes == []
    assert "changed at row 26" in run.summary()["notes"][-1]
    with pytest.raises(InputError, match="changed at row 26"):
        run.weights(state=0)
    with pytest.raises(InputError, match="no one list fits"):
        OpenMMRun(tmp_path, temperatures_K=[300, 330])
    assert len(run.weights(state=0, frames=(0, 25))["weights"][0]) == 25
    assert len(run.weights(state=1, frames=(25, 40))["weights"][1]) == 15


def test_steps_before_the_sampler(tmp_path):
    """Equilibration at another timestep before the sampler: the header's
    timestep is set aside with a note, and one given is taken."""
    sampler = _sampler(testsystems.double_well(-1.0),
                       _temperature_states(LADDER[:2]), steps=20)
    integrator = sampler.simulation.integrator
    integrator.setStepSize(0.001 * unit.picoseconds)
    sampler.simulation.step(100)
    integrator.setStepSize(0.002 * unit.picoseconds)
    for i in range(2):
        sampler.replicaConformation[i] = \
            sampler.simulation.context.getState(
                positions=True, velocities=True, parameters=True,
                integratorParameters=True)
    sampler.reporters.append(app.ReplicaExchangeReporter(
        str(tmp_path / "run"), 1, sampler, trajectoryPerState=True,
        trajectoryFormat="dcd", energy=True, checkpoints=True))
    sampler.simulate(5)
    run = OpenMMRun(tmp_path / "run")
    assert run.timestep_ps is None
    assert "steps at another timestep" in run.summary()["notes"][-1]
    s = OpenMMRun(tmp_path / "run", timestep_fs=2).summary()
    assert s["time_ns_per_replica"] == pytest.approx(5 * 20 * 2e-6)
