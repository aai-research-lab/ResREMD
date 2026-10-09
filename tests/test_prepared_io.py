import numpy as np
import pytest

from resremd import testsystems
from resremd.system import load_prepared, write_prepared


def test_write_and_load_round_trip(tmp_path):
    p = testsystems.lj_box()
    write_prepared(tmp_path / "s", p.system, p.topology, p.positions, p.box)
    q = load_prepared(tmp_path / "s")
    assert q.n_atoms == p.n_atoms
    assert np.allclose(q.positions, p.positions)
    assert np.allclose(q.box, p.box)
    assert "CRYST1" in (tmp_path / "s/topology.pdb").read_text()


def test_prepare_peptide_implicit():
    top, pos = testsystems.alanine_dipeptide()
    system, topology, positions, box = testsystems.prepare_peptide(
        top, pos, solvent="implicit", platform="Reference")
    assert box is None and not system.usesPeriodicBoundaryConditions()
    assert positions.shape == (22, 3)
    assert system.getNumConstraints() > 0, "bonds to hydrogen constrained"
    with pytest.raises(ValueError):
        testsystems.prepare_peptide(top, pos, solvent="vacuum")


def test_proline_dipeptide_trans_and_cis():
    md = pytest.importorskip("mdtraj")
    for omega in (180.0, 0.0):
        top, pos = testsystems.proline_dipeptide(omega)
        t = md.Trajectory(pos[None], md.Topology.from_openmm(top))
        w = np.degrees(md.compute_dihedrals(
            t, [testsystems.omega_atoms(top)]))[0, 0]
        assert abs(((w - omega) + 180) % 360 - 180) < 15
        idx = {a.name: a.index for a in t.topology.atoms
               if a.residue.name == "PRO"}
        assert np.linalg.norm(pos[idx["CD"]] - pos[idx["N"]]) < 0.16
        zeta = np.degrees(md.compute_dihedrals(
            t, [[idx["CA"], idx["N"], idx["C"], idx["CB"]]]))[0, 0]
        assert 25 < zeta < 45


def test_a_damaged_prepared_directory_is_named(tmp_path):
    from resremd.errors import InputError

    box = testsystems.lj_box()
    write_prepared(tmp_path, box.system, box.topology, box.positions,
                   box.box)
    system = (tmp_path / "system.xml").read_text()
    (tmp_path / "system.xml").write_text(system[:200])
    with pytest.raises(InputError, match="system.xml could not be read"):
        load_prepared(tmp_path)
    (tmp_path / "system.xml").write_text(
        (tmp_path / "state.xml").read_text())
    with pytest.raises(InputError, match="holds a State, not a System"):
        load_prepared(tmp_path)
    # A state without positions.
    import openmm

    (tmp_path / "system.xml").write_text(system)
    context = openmm.Context(box.system, openmm.VerletIntegrator(0.001),
                             openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(box.positions)
    (tmp_path / "state.xml").write_text(openmm.XmlSerializer.serialize(
        context.getState(getEnergy=True)))
    with pytest.raises(InputError, match="state.xml could not be read"):
        load_prepared(tmp_path)


def test_a_system_in_memory_needs_no_box_given():
    """A periodic System's own box stands in, as OpenMM gives it; one given
    must be a box."""
    from resremd.errors import InputError
    from resremd.system import from_objects

    box = testsystems.lj_box()
    prepared = from_objects(box.system, box.topology, box.positions)
    assert np.allclose(prepared.box, box.box)
    with pytest.raises(InputError, match="The box is three vectors"):
        from_objects(box.system, box.topology, box.positions,
                     [[1.0, 2.0], [3.0, 4.0]])
    side = box.box[0, 0]
    for wrong in (-np.eye(3), [[side, 0, 0], [np.nan, side, 0],
                               [0, 0, side]],
                  # Not in OpenMM's reduced form.
                  [[side, 0, 0], [0.6 * side, side, 0], [0, 0, side]]):
        with pytest.raises(InputError, match="cannot be used"):
            from_objects(box.system, box.topology, box.positions, wrong)
    # A System that is not periodic ignores a box, whatever it is.
    well = testsystems.double_well()
    assert from_objects(well.system, well.topology, well.positions,
                        -np.eye(3)) is not None
