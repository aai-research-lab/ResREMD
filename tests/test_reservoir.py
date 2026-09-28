import json

import numpy as np
import pytest

from resremd import testsystems
from resremd.errors import ReservoirError
from resremd.reservoir import Reservoir, write_reservoir
from resremd.system import create_context, make_integrator, topology_digest
from resremd.thermo import Ensemble


def test_open_and_draw_uniformly(tmp_path):
    testsystems.write_double_well_reservoir(tmp_path / "r", kind="boltzmann",
                                       n_frames=10, temperature_K=500)
    res = Reservoir.open(tmp_path / "r")
    assert res.n_frames == 10 and res.kind == "boltzmann"
    rng = np.random.default_rng(0)
    counts = np.bincount([res.draw(rng) for _ in range(20000)], minlength=10)
    assert counts.min() > 1800 and counts.max() < 2200


def test_weighted_draws_follow_the_weights(tmp_path):
    testsystems.write_double_well_reservoir(tmp_path / "r", kind="boltzmann",
                                       n_frames=4, temperature_K=500)
    meta = json.loads((tmp_path / "r/reservoir.json").read_text())
    meta["kind"] = "weighted"
    (tmp_path / "r/reservoir.json").write_text(json.dumps(meta))
    np.save(tmp_path / "r/weights.npy", np.array([0.1, 0.2, 0.0, 0.7]))
    res = Reservoir.open(tmp_path / "r")
    rng = np.random.default_rng(0)
    counts = np.bincount([res.draw(rng) for _ in range(40000)], minlength=4)
    assert np.allclose(counts / counts.sum(), [0.1, 0.2, 0.0, 0.7], atol=0.01)


def test_unfinished_reservoir_is_refused(tmp_path):
    testsystems.write_double_well_reservoir(tmp_path / "r", kind="boltzmann",
                                       n_frames=4, temperature_K=500)
    meta = json.loads((tmp_path / "r/reservoir.json").read_text())
    meta["complete"] = False
    (tmp_path / "r/reservoir.json").write_text(json.dumps(meta))
    with pytest.raises(ReservoirError) as err:
        Reservoir.open(tmp_path / "r")
    assert err.value.code == "resremd.reservoir.incomplete"


def test_a_reservoir_of_other_atoms_is_refused(tmp_path):
    prepared = testsystems.write_double_well_reservoir(
        tmp_path / "r", kind="boltzmann", n_frames=4, temperature_K=500)
    res = Reservoir.open(tmp_path / "r")
    common = dict(n_atoms=1, periodic=False, ensemble=Ensemble(), box=None,
                  top_temperature_K=400)
    assert res.check_against(
        topology_sha256=topology_digest(prepared.topology), **common) == []
    with pytest.raises(ReservoirError, match="order"):
        res.check_against(topology_sha256="0" * 64, **common)
    with pytest.raises(ReservoirError, match="pressure|ensemble"):
        res.check_against(topology_sha256=topology_digest(prepared.topology),
                          **{**common, "ensemble": Ensemble(pressure_bar=1.0)})
    warnings = res.check_against(
        topology_sha256=topology_digest(prepared.topology),
        **{**common, "top_temperature_K": 600})
    assert warnings and "not hotter" in warnings[0]


def test_energies_are_those_of_the_injected_coordinates_and_cached(tmp_path):
    prepared = testsystems.write_double_well_reservoir(
        tmp_path / "r", kind="boltzmann", n_frames=50, temperature_K=500)
    res = Reservoir.open(tmp_path / "r")
    ctx, _ = create_context(prepared.system,
                            make_integrator("langevin_middle", 300, 1, 2, 1),
                            platform="Reference", precision="double",
                            device=None, cpu_threads=None)

    calls = []

    def evaluate(pos, box):
        calls.append(1)
        ctx.setPositions(pos)
        return ctx.getState(getEnergy=True).getPotentialEnergy()._value

    key = {"system_sha256": "s", "platform": "Reference"}
    e = res.energies(evaluate, key_fields=key)["potential_kjmol"]
    x = np.asarray(res.positions[:, 0, :], dtype=float)
    exact = testsystems.double_well_energy(x[:, 0]) + 0.5 * testsystems.SPRING * (
        x[:, 1] ** 2 + x[:, 2] ** 2)
    assert np.allclose(e, exact, rtol=1e-10, atol=1e-8)
    n = len(calls)
    res.energies(evaluate, key_fields=key)
    assert len(calls) == n, "a matching cache is read, not recomputed"
    res.energies(evaluate, key_fields={**key, "platform": "CPU"})
    assert len(calls) == 2 * n, "a different platform recomputes"


def test_write_reservoir_from_arrays(tmp_path):
    prepared = testsystems.double_well()
    positions = np.zeros((3, 1, 3))
    positions[:, 0, 0] = [-1.0, 0.0, 1.0]
    meta = write_reservoir(tmp_path / "r", topology=prepared.topology,
                           positions=positions, kind="weighted",
                           temperature_K=500.0, weights=np.array([1, 1, 2]))
    res = Reservoir.open(tmp_path / "r")
    assert meta["n_frames"] == 3 and res.kind == "weighted"
    assert np.allclose(res.weights, [0.25, 0.25, 0.5])


def test_alanine_dipeptide_is_the_l_enantiomer():
    md = pytest.importorskip("mdtraj")
    top, pos = testsystems.alanine_dipeptide()
    t = md.Trajectory(pos[None], md.Topology.from_openmm(top))
    idx = {a.name: a.index for a in t.topology.atoms
           if a.residue.name == "ALA"}
    zeta = np.degrees(md.compute_dihedrals(
        t, [[idx["CA"], idx["N"], idx["C"], idx["CB"]]]))[0, 0]
    assert 25 < zeta < 45, "L-amino acids have zeta near +34 degrees"
    assert top.getNumAtoms() == 22
