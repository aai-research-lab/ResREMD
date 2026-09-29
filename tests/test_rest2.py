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


def test_energy_is_quadratic_in_the_scale_and_exact_at_one():
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
    scaled, info = rest2.rest2_system(system, np.array(solute))
    assert info["nonbonded"] and info["torsions_scaled"] > 0
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
