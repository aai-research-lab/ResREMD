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
    from openmm import unit
    from openmm.app.internal.unitcell import (computePeriodicBoxVectors,
                                              reducePeriodicBoxVectors)

    def refused(wrong):
        with pytest.raises(InputError, match="cannot be used") as error:
            from_objects(box.system, box.topology, box.positions, wrong)
        return str(error.value)

    def taken(right):
        return from_objects(box.system, box.topology, box.positions,
                            np.asarray(right, float)).box

    def same_lattice(old, new):
        """new = M old with M whole numbers and |det M| = 1."""
        m = np.asarray(new, float) @ np.linalg.inv(np.asarray(old, float))
        assert np.allclose(m, np.round(m), atol=1e-6), m
        assert abs(round(np.linalg.det(np.round(m)))) == 1

    def nm(vectors):
        return np.array(vectors.value_in_unit(unit.nanometer))

    def into_limits(v):
        """Values left just past a limit by rounding, moved just inside."""
        v = np.array(v, float)
        for i, j, k in ((1, 0, 0), (2, 0, 0), (2, 1, 1)):
            limit = v[k, k] / 2
            if abs(v[i, j]) > limit:
                v[i, j] = np.copysign(np.nextafter(limit, 0), v[i, j])
        return v

    # Off the reduced form by a shift of the lattice (b_x over a_x/2; c_y
    # over b_y/2, though under half of b's length): the rule and the way
    # are said, and reducing gives the same lattice in OpenMM's form.
    for skewed in ([[side, 0, 0], [0.6 * side, side, 0], [0, 0, side]],
                   [[side, 0, 0], [0.4 * side, 0.6 * side, 0],
                    [0, 0.35 * side, side]]):
        message = refused(skewed)
        for said in ("|c_y| at most b_y/2", "give them as rows",
                     "not a reflection", "flip a if a_x < 0",
                     "whole multiples", "just past a limit"):
            assert said in message, message
        same_lattice(skewed, taken(nm(reducePeriodicBoxVectors(skewed))))
    # A shift that reducing leaves just past a limit, moved inside.
    dodecahedron = nm(computePeriodicBoxVectors(
        2.2, 2.2, 2.2, 60 * unit.degrees, 60 * unit.degrees,
        90 * unit.degrees))
    a, b, c = dodecahedron
    shifted = np.array([a, b + 2 * a, c - 3 * b + a])
    reduced = nm(reducePeriodicBoxVectors(shifted))
    refused(reduced)
    same_lattice(dodecahedron, taken(into_limits(reduced)))
    # Rotated (and negated): rotate back, flip, reduce.
    angle = np.radians(30)
    turn = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle), np.cos(angle)]])
    for wrong in (dodecahedron @ turn.T, -dodecahedron):
        refused(wrong)
        q, r = np.linalg.qr(wrong.T)  # wrong = r.T q.T
        if np.linalg.det(q) < 0:  # a rotation, not a reflection
            q[:, 0], r[0] = -q[:, 0], -r[0]
        lower = np.where(np.abs(r.T) < 1e-12 * side, 0.0, r.T)
        lower *= np.sign(np.diag(lower))[:, None]
        fixed = taken(into_limits(nm(reducePeriodicBoxVectors(lower))))
        same_lattice(wrong @ q, fixed)
    # Given as columns.
    octahedron = nm(computePeriodicBoxVectors(
        side, side, side, 70.5288 * unit.degrees, 109.4712 * unit.degrees,
        70.5288 * unit.degrees))
    refused(octahedron.T)
    taken(octahedron)
    # No volume, however oriented: no remedy, as none can work.
    flat = np.array([[10, 0, 0], [3, 10, 0], [6e-4, 8e-4, 0]]) @ turn.T
    for wrong in (np.diag([side, side, 0.0]), flat):
        assert refused(wrong) == (f"The box {np.asarray(wrong).tolist()} "
                                  "cannot be used: its vectors span no "
                                  "volume.")
    # A huge skew is no lack of volume, and a thin box OpenMM takes is
    # taken.
    assert "OpenMM needs" in refused([[3, 0, 0], [3e9 + 0.3, 3.3, 0],
                                      [0, 0, 2.7]])
    taken([[1, 0, 0], [0.5, 1e-12, 0], [0, 0, 1]])
    # A System that is not periodic ignores a box, whatever numbers it
    # holds.
    well = testsystems.double_well()
    assert from_objects(well.system, well.topology, well.positions,
                        -np.eye(3)) is not None
