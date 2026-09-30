"""REST2: the scaled System, and replica exchange on it against exact
answers.

The torsion model's only coupling to the dihedral is its torsion, which
REST2 scales by T0 / T_k, so at state k its cis fraction is exactly that of
the unscaled model at the effective temperature T_k.
"""

import json

import numpy as np
import openmm
import pytest

import resremd
from resremd import rest2, testsystems
from resremd.errors import InputError
from resremd.ladder import geometric
from resremd.statistics import statistical_inefficiency

md = pytest.importorskip("mdtraj")

LADDER = geometric(300.0, 3000.0, 5)


def _context(system):
    return openmm.Context(system, openmm.VerletIntegrator(0.001),
                          openmm.Platform.getPlatformByName("Reference"))


@pytest.mark.parametrize("constant_pressure", [False, True])
def test_energy_is_quadratic_in_the_scale_and_exact_at_one(
        constant_pressure):
    """At constant volume the dispersion correction is left to OpenMM, a
    constant of each state; at constant pressure it is scaled."""
    from openmm import app, unit

    top, pos = testsystems.alanine_dipeptide()
    mod = app.Modeller(top, pos * unit.nanometer)
    ff = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    mod.addSolvent(ff, padding=1.0 * unit.nanometer)
    system = ff.createSystem(mod.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=0.9 * unit.nanometer,
                             constraints=app.HBonds)
    solute = [a.index for a in mod.topology.atoms()
              if a.residue.name != "HOH"]
    scaled, info = rest2.rest2_system(system, np.array(solute),
                                      constant_pressure=constant_pressure)
    assert info["nonbonded"] and info["torsions_scaled"] > 0
    volume_forces = [f for f in scaled.getForces()
                     if f.getName() == "REST2 dispersion correction"]
    x = np.array(mod.positions.value_in_unit(unit.nanometer))
    plain = _context(system)
    plain.setPositions(x)
    e_plain = plain.getState(getEnergy=True).getPotentialEnergy()._value
    ctx = _context(scaled)
    ctx.setPositions(x)

    def u(s):
        rest2.set_scale(ctx, s)
        return ctx.getState(getEnergy=True).getPotentialEnergy()._value

    assert u(1.0) == pytest.approx(e_plain, abs=1e-3)
    terms = rest2.fit([1.0, 0.0, 0.5], [u(1.0), u(0.0), u(0.5)])
    for s in (0.3, 0.7, 0.9):
        assert rest2.energy(terms, s) == pytest.approx(u(s), abs=1e-2)
    if not constant_pressure:
        assert info["dispersion_correction"] == "constant"
        assert not volume_forces
    elif hasattr(openmm, "CustomVolumeForce"):
        assert info["dispersion_correction"] == "scaled"
        _check_dispersion_correction(system, scaled, solute, x)


def _check_dispersion_correction(system, scaled, solute, x):
    """The scaled correction is OpenMM's own for the solute's well depths
    scaled by s^2."""
    def energy(sys_, s=None, groups=-1):
        ctx = _context(sys_)
        ctx.setPositions(x)
        if s is not None:
            rest2.set_scale(ctx, s)
        return ctx.getState(getEnergy=True, groups=groups
                            ).getPotentialEnergy()._value

    volume = [f for f in scaled.getForces()
              if f.getName() == "REST2 dispersion correction"]
    assert len(volume) == 1
    volume[0].setForceGroup(31)
    for s in (0.8, 0.35):
        ref = openmm.XmlSerializer.deserialize(
            openmm.XmlSerializer.serialize(system))
        nb = [f for f in ref.getForces()
              if isinstance(f, openmm.NonbondedForce)][0]
        for i in solute:
            q, sigma, eps = nb.getParticleParameters(int(i))
            nb.setParticleParameters(int(i), q, sigma, eps * s * s)
        on = energy(ref)
        nb.setUseDispersionCorrection(False)
        assert energy(scaled, s, groups={31}) == pytest.approx(
            on - energy(ref), abs=1e-6)


def test_unscalable_solute_forces_are_refused():
    from openmm import app

    top, pos = testsystems.alanine_dipeptide()
    ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    system = ff.createSystem(top, nonbondedMethod=app.NoCutoff)
    with pytest.raises(InputError, match="GB"):
        rest2.rest2_system(system, np.arange(top.getNumAtoms()))


def _cis_by_state(run):
    man = json.loads((run / "manifest.json").read_text())
    out = []
    for st in man["states"]:
        t = md.load(str(run / st["trajectory"]), top=str(run / "topology.pdb"))
        phi = md.compute_dihedrals(t, [[0, 1, 2, 3]])[:, 0]
        cis = (np.abs(phi) < np.pi / 2).astype(float)
        se = np.sqrt(cis.var() * statistical_inefficiency(cis) / cis.size)
        out.append((st["temperature_K"], cis.mean(), se))
    return out


def _run(tmp_path, name, cycles, **extra):
    resremd.run(testsystems.torsion_model(), output=str(tmp_path / name),
                rest2=True, rest2_selection="all", temperatures_K=LADDER,
                production_steps=250 * cycles, exchange_interval_steps=250,
                trajectory_interval_steps=250, friction_per_ps=5.0,
                platform="Reference", random_seed=11, save_selection="all",
                equilibration_ns=0.0, minimize=False, **extra)
    return tmp_path / name


def _check(run, sigma=4.5):
    for t, cis, se in _cis_by_state(run):
        exact = testsystems.cis_fraction(t)
        assert abs(cis - exact) < max(sigma * se, 0.005), (t, cis, exact, se)


def test_rest2_samples_every_state_exactly(tmp_path):
    run = _run(tmp_path, "run", 12000)
    _check(run)
    man = json.loads((run / "manifest.json").read_text())
    assert man["rest2"]["scales"][0] == 1.0
    assert man["states"][-1]["rest2_scale"] == pytest.approx(
        np.sqrt(300.0 / 3000.0))
    assert (run / "rest2_terms.csv").exists()
    # MBAR over the Hamiltonian states, to the bottom and between rungs.
    from resremd.mbar import TemperatureReweighting

    rw = TemperatureReweighting(run)
    phi = {}
    for st in man["states"]:
        t = md.load(str(run / st["trajectory"]), top=str(run / "topology.pdb"))
        phi[st["index"]] = md.compute_dihedrals(t, [[0, 1, 2, 3]])[:, 0]
    for target in (300.0, 700.0):
        w = rw.weights(target, discard_fraction=0.1)
        cis, se = _weighted_mean(
            {k: (ws, (np.abs(phi[k][w["first_frame"]:]) < np.pi / 2))
             for k, ws in w["weights"].items()})
        exact = testsystems.cis_fraction(target)
        assert abs(cis - exact) < max(4.5 * se, 0.005), (target, cis, exact,
                                                          se)


def _weighted_mean(parts):
    """An MBAR estimate sum(w f) and its standard error, from each state's
    correlated series w (f - estimate), with the free energies taken as
    known."""
    est = sum(float(np.sum(w * f)) for w, f in parts.values())
    var = 0.0
    for w, f in parts.values():
        y = w * (f - est)
        if y.var() > 0:
            var += y.size * y.var() * statistical_inefficiency(y)
    return est, float(np.sqrt(var))


def test_a_reservoir_of_the_top_state_is_always_accepted(tmp_path):
    """Frames sampled with REST2 at the top state's own scale are a sample
    of the top state: exchanging with them is a Gibbs move."""
    resremd.generate_reservoir(
        testsystems.torsion_model(), output=str(tmp_path / "r"),
        temperature_K=LADDER[-1], rest2_run_temperature_K=300.0,
        rest2_selection="all", duration_ns=4.0, frame_interval_steps=250,
        equilibration_ns=0.05, friction_per_ps=5.0, platform="Reference",
        random_seed=2, minimize=False)
    meta = json.loads((tmp_path / "r/reservoir.json").read_text())
    assert meta["rest2"]["scale"] == pytest.approx(np.sqrt(0.1))
    run = _run(tmp_path, "run", 8000, reservoir=str(tmp_path / "r"))
    man = json.loads((run / "manifest.json").read_text())
    assert man["exchanges"]["reservoir"]["acceptance"] == 1.0
    _check(run)


def test_an_unscaled_reservoir_serves_a_rest2_run(tmp_path):
    """A reservoir at a real temperature, under a bias and weighted, feeds
    the top of a REST2 ladder with the general exchange criterion."""
    resremd.generate_reservoir(
        testsystems.torsion_model(), output=str(tmp_path / "r"),
        temperature_K=520.0, duration_ns=6.0, frame_interval_steps=250,
        equilibration_ns=0.05, friction_per_ps=5.0, platform="Reference",
        random_seed=3, minimize=False,
        bias_torsions=testsystems.torsion_bias(70.0))
    run = _run(tmp_path, "run", 8000, reservoir=str(tmp_path / "r"))
    man = json.loads((run / "manifest.json").read_text())
    assert 0.0 < man["exchanges"]["reservoir"]["acceptance"] < 1.0
    _check(run)
    check = resremd.summarize(run)["reservoir_check"]
    assert check["status"] == "ok" and abs(check["z"]) < 4


def test_a_rest2_reservoir_needs_a_rest2_run(tmp_path):
    resremd.generate_reservoir(
        testsystems.torsion_model(), output=str(tmp_path / "r"),
        temperature_K=3000.0, rest2_run_temperature_K=300.0,
        rest2_selection="all", duration_ns=0.01, frame_interval_steps=100,
        equilibration_ns=0.0, platform="Reference", minimize=False)
    with pytest.raises(InputError, match="REST2"):
        resremd.run(testsystems.torsion_model(), output=str(tmp_path / "run"),
                    reservoir=str(tmp_path / "r"), temperatures_K=LADDER,
                    production_steps=100, exchange_interval_steps=50,
                    platform="Reference", equilibration_ns=0.0,
                    minimize=False)


def test_constant_pressure_rest2_scales_the_dispersion_correction(tmp_path):
    """At constant pressure the scaled correction acts on the barostat, so a
    run begun with it unscaled is not resumed with it scaled."""
    if not hasattr(openmm, "CustomVolumeForce"):
        pytest.skip("needs OpenMM 8.3 or later")
    common = dict(rest2=True, rest2_atoms=list(range(20)),
                  temperatures_K=[100.0, 150.0], exchange_interval_steps=50,
                  timestep_fs=4.0, platform="Reference", random_seed=1,
                  equilibration_ns=0.0, save_selection="all")
    run = tmp_path / "run"
    m = resremd.run(testsystems.lj_box(pressure=True), output=str(run),
                    production_steps=100, **common)
    assert m["rest2"]["dispersion_correction"] == "scaled"
    chk = run / "checkpoint.npz"
    with np.load(chk) as data:
        arrays = {k: np.array(data[k]) for k in data.files}
    meta = json.loads(str(arrays["meta"]))
    assert meta["fingerprint"].pop("rest2_dispersion") == "scaled"
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez(chk, **arrays)
    with pytest.raises(resremd.errors.ResumeError, match="rest2_dispersion"):
        resremd.run(testsystems.lj_box(pressure=True), output=str(run),
                    production_steps=200, resume=True, **common)


def test_reservoir_energies_are_not_reused_across_rest2_systems(
        tmp_path, monkeypatch):
    """The cache of a reservoir's REST2 energies is keyed by the scaled
    System itself: one built another way is never reused."""
    rng = np.random.default_rng(0)
    base = testsystems.lj_box()
    frames = base.positions[None] + rng.normal(0, 0.01, (20, 125, 3))
    from resremd.reservoir import write_reservoir

    write_reservoir(tmp_path / "r", topology=base.topology, positions=frames,
                    kind="boltzmann", temperature_K=200.0,
                    box=np.repeat(base.box[None], 20, axis=0))
    common = dict(rest2=True, rest2_atoms=list(range(20)),
                  temperatures_K=[100.0, 150.0], exchange_interval_steps=50,
                  production_steps=50, timestep_fs=4.0, platform="Reference",
                  random_seed=1, equilibration_ns=0.0, save_selection="all",
                  reservoir=str(tmp_path / "r"))
    real = rest2.rest2_system

    def another(system, solute, **kw):
        # The same energies from a System that serialises differently.
        scaled, info = real(system, solute, **kw)
        scaled.addForce(openmm.CustomExternalForce("0"))
        return scaled, info

    def caches():
        return len(list((tmp_path / "r/energies").glob("*.npz")))

    monkeypatch.setattr(rest2, "rest2_system", another)
    resremd.run(testsystems.lj_box(), output=str(tmp_path / "a"), **common)
    assert caches() == 1
    monkeypatch.setattr(rest2, "rest2_system", real)
    resremd.run(testsystems.lj_box(), output=str(tmp_path / "b"), **common)
    assert caches() == 2
    resremd.run(testsystems.lj_box(), output=str(tmp_path / "c"), **common)
    assert caches() == 2


def _nvt_rest2_run(tmp_path, name, **extra):
    common = dict(rest2=True, rest2_atoms=list(range(20)),
                  temperatures_K=[100.0, 130.0, 170.0],
                  exchange_interval_steps=50, timestep_fs=4.0,
                  platform="Reference", random_seed=3, equilibration_ns=0.0,
                  save_selection="all")
    common.update(extra)
    return resremd.run(testsystems.lj_box(), output=str(tmp_path / name),
                       **common)


def test_at_constant_volume_scaling_the_correction_changes_nothing(
        tmp_path, monkeypatch):
    """The correction is a constant of each state at fixed volume: scaling
    it or not gives the same exchanges, the same reservoir decisions, and
    energies that differ by a constant per state."""
    resremd.generate_reservoir(
        testsystems.lj_box(), output=str(tmp_path / "r"), temperature_K=170.0,
        duration_ns=0.024, frame_interval_steps=100, timestep_fs=4.0,
        equilibration_ns=0.02, platform="Reference", random_seed=4,
        minimize=False)
    run = dict(production_steps=50 * 60, reservoir=str(tmp_path / "r"))
    m = _nvt_rest2_run(tmp_path, "new", **run)
    assert m["rest2"]["dispersion_correction"] == "constant"
    real = rest2.rest2_system
    monkeypatch.setattr(rest2, "rest2_system",
                        lambda s, solute, **kw: real(s, solute,
                                                     constant_pressure=True))
    m = _nvt_rest2_run(tmp_path, "old", **run)
    assert m["rest2"]["dispersion_correction"] == "scaled"
    def table(name, columns):
        return [np.loadtxt(tmp_path / which / name, delimiter=",",
                           skiprows=1, usecols=columns)
                for which in ("new", "old")]

    new, old = table("states.csv", (2, 3, 4))
    assert np.array_equal(new, old)
    # replica, frame, log acceptance, accepted
    new, old = table("reservoir_exchanges.csv", (2, 3, 6, 7))
    assert 0 < new[:, 3].sum() < len(new)
    assert np.array_equal(new[:, [0, 1, 3]], old[:, [0, 1, 3]])
    assert np.allclose(new[:, 2], old[:, 2], atol=1e-8)
    dn = np.load(tmp_path / "new/reservoir_delta.npy")
    do = np.load(tmp_path / "old/reservoir_delta.npy")
    assert np.ptp(dn - do) < 1e-8 and abs(np.mean(dn - do)) > 1e-3


def test_a_run_from_before_the_constant_volume_change_is_not_resumed(
        tmp_path):
    """Runs that scaled the correction at constant volume would mix two
    conventions in rest2_terms.csv; ones that left it unscaled match."""
    from resremd.errors import ResumeError

    _nvt_rest2_run(tmp_path, "run", production_steps=100)
    chk = tmp_path / "run/checkpoint.npz"
    with np.load(chk) as data:
        arrays = {k: np.array(data[k]) for k in data.files}
    meta = json.loads(str(arrays["meta"]))
    assert meta["fingerprint"]["rest2_dispersion"] is None
    for old, refused in (("scaled", True), ("unscaled", False)):
        meta["fingerprint"]["rest2_dispersion"] = old
        arrays["meta"] = np.array(json.dumps(meta))
        np.savez(chk, **arrays)
        if refused:
            with pytest.raises(ResumeError, match="earlier ResREMD"):
                _nvt_rest2_run(tmp_path, "run", production_steps=200,
                               resume=True)
        else:
            m = _nvt_rest2_run(tmp_path, "run", production_steps=200,
                               resume=True)
            assert m["status"] == "complete"


def test_a_changed_ensemble_is_reported_as_one(tmp_path):
    """A constant-pressure REST2 run resumed at constant volume is refused
    for its ensemble, not as a run from an earlier version."""
    from resremd.errors import ResumeError

    common = dict(rest2=True, rest2_atoms=list(range(20)),
                  temperatures_K=[100.0, 150.0], exchange_interval_steps=50,
                  timestep_fs=4.0, platform="Reference", random_seed=1,
                  equilibration_ns=0.0, save_selection="all")
    resremd.run(testsystems.lj_box(pressure=True),
                output=str(tmp_path / "run"), production_steps=100, **common)
    with pytest.raises(ResumeError) as error:
        resremd.run(testsystems.lj_box(pressure=True),
                    output=str(tmp_path / "run"), production_steps=200,
                    resume=True, ensemble="nvt", **common)
    assert "ensemble" in str(error.value)
    assert "earlier ResREMD" not in str(error.value)
