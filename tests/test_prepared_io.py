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
    with pytest.raises(InputError, match="not all finite"):
        from_objects(box.system, box.topology, box.positions,
                     [[side, 0, 0], [np.nan, side, 0], [0, 0, side]])
    from openmm.app.internal.unitcell import reducePeriodicBoxVectors

    # Off OpenMM's reduced form only by a shift of the lattice (b_x over
    # a_x/2; c_y over b_y/2, though under half of b's length): its helper
    # puts it right, and the message says so.
    for skewed in ([[side, 0, 0], [0.6 * side, side, 0], [0, 0, side]],
                   [[side, 0, 0], [0.4 * side, 0.6 * side, 0],
                    [0, 0.35 * side, side]]):
        with pytest.raises(InputError, match=r"(?s)reduced form.*\|c_y\| at "
                           r"most b_y/2.*reducePeriodicBoxVectors"):
            from_objects(box.system, box.topology, box.positions, skewed)
        assert from_objects(box.system, box.topology, box.positions,
                            reducePeriodicBoxVectors(skewed)) is not None
    # Each other way to be refused gets the remedy that fits, and none
    # that does not.
    from openmm import unit
    from openmm.app.internal.unitcell import computePeriodicBoxVectors

    octahedron = np.array(computePeriodicBoxVectors(
        side, side, side, 70.5288 * unit.degrees, 109.4712 * unit.degrees,
        70.5288 * unit.degrees).value_in_unit(unit.nanometer))
    cube = np.eye(3) * side
    tilted = cube.copy()
    tilted[0, 1] = 1e-17
    for wrong, said in ((octahedron.T, "if these are the vectors as columns"),
                        (-np.eye(3), "left-handed"),
                        (np.diag([side, side, 0.0]), "span no volume"),
                        (tilted, "round-off: set them to 0.")):
        with pytest.raises(InputError, match="cannot be used") as error:
            from_objects(box.system, box.topology, box.positions, wrong)
        message = str(error.value)
        assert said in message.lower(), message
        assert "reducePeriodicBoxVectors" not in message, message
    # A System that is not periodic ignores a box, whatever numbers it
    # holds.
    well = testsystems.double_well()
    assert from_objects(well.system, well.topology, well.positions,
                        -np.eye(3)) is not None
