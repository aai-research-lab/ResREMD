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
    """A periodic System's own box stands in, as OpenMM gives it; one that
    is not periodic ignores a box, whatever numbers it holds."""
    from resremd.system import from_objects

    box = testsystems.lj_box()
    prepared = from_objects(box.system, box.topology, box.positions)
    assert np.allclose(prepared.box, box.box)
    well = testsystems.double_well()
    assert from_objects(well.system, well.topology, well.positions,
                        -np.eye(3)) is not None


def test_a_refused_box_is_explained():
    """A box of the wrong shape or not finite is said to be; for one OpenMM
    refuses, each way the message offers works and keeps the lattice; a box
    of no volume is said to be one; OpenMM alone decides what is taken."""
    from openmm import unit
    from openmm.app.internal.unitcell import (computePeriodicBoxVectors,
                                              reducePeriodicBoxVectors)

    from resremd.errors import InputError
    from resremd.system import _box_refused, from_objects

    box = testsystems.lj_box()
    side = box.box[0, 0]

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
        """Values just past a limit, moved just inside."""
        v = np.array(v, float)
        for i, j, k in ((1, 0, 0), (2, 0, 0), (2, 1, 1)):
            limit = v[k, k] / 2
            if abs(v[i, j]) > limit:
                v[i, j] = np.copysign(np.nextafter(limit, 0), v[i, j])
        return v

    with pytest.raises(InputError, match="The box is three vectors"):
        from_objects(box.system, box.topology, box.positions,
                     [[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(InputError, match="not all finite"):
        from_objects(box.system, box.topology, box.positions,
                     [[side, 0, 0], [np.nan, side, 0], [0, 0, side]])
    # The whole message, once.
    skewed = [[2.0, 0.0, 0.0], [1.2, 2.0, 0.0], [0.0, 0.0, 2.0]]
    assert refused(skewed) == (
        f"The box {skewed} cannot be used: "
        f"{_box_refused(np.array(skewed))} OpenMM needs vector a along x "
        "and vector b in the xy plane, with a_x, b_y and c_z positive, "
        "|b_x| and |c_x| at most a_x/2, and |c_y| at most b_y/2. If you "
        "gave the vectors as columns, give them as rows. Otherwise: rotate "
        "the box, the positions, any reference positions or fixed "
        "directions in the System and any trajectory frames to be used "
        "with it together into that orientation (a rotation, not a "
        "reflection; values off that orientation only by round-off can be "
        "set to 0); flip a if a_x < 0, b if b_y < 0 and c if c_z < 0; add "
        "to c whole multiples of b, then of a; and add to b whole multiples "
        "of a. Flips and additions keep the lattice. openmm.app.internal."
        "unitcell.reducePeriodicBoxVectors does the adding, but may leave a "
        "value just past a limit, to be moved just inside.")
    # Off the reduced form by a shift of the lattice (b_x over a_x/2; c_y
    # over b_y/2, though under half of b's length): reduced, it is taken.
    for unreduced in ([[side, 0, 0], [0.6 * side, side, 0], [0, 0, side]],
                      [[side, 0, 0], [0.4 * side, 0.6 * side, 0],
                       [0, 0.35 * side, side]]):
        refused(unreduced)
        reduced = nm(reducePeriodicBoxVectors(unreduced))
        same_lattice(unreduced, taken(reduced))
    # Just past a limit, as reducing can leave it (c_x one step beyond
    # -a_x/2): moved inside.
    dodecahedron = nm(computePeriodicBoxVectors(
        2.2, 2.2, 2.2, 60 * unit.degrees, 60 * unit.degrees,
        90 * unit.degrees))
    past = dodecahedron.copy()
    past[2, 0] = -np.nextafter(past[0, 0] / 2, np.inf)
    refused(past)
    same_lattice(dodecahedron, taken(into_limits(past)))
    # Rotated, or negated: rotate back, flip, reduce.
    angle = np.radians(30)
    turn = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle), np.cos(angle)]])
    a, b, c = dodecahedron
    shifted = np.array([a, b + 2 * a, c - 3 * b + a])
    for wrong in (dodecahedron @ turn.T, -dodecahedron, shifted @ turn.T):
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
    # No volume, or too little to trust, however oriented: no remedy.
    flat = np.array([[10, 0, 0], [3, 10, 0], [6e-4, 8e-4, 0]]) @ turn.T
    thin = np.array([[1, 0, 0], [0.5, 1e-15, 0], [0, 0, 1]]) @ turn.T
    for wrong in (np.diag([side, side, 0.0]), flat, thin,
                  [[1, 1, 0], [0, 0, 0], [0, 0, 1]]):
        assert refused(wrong).endswith(
            "cannot be used: its vectors span no volume, or too little to "
            "trust.")
    # Thin, but with a along x and b in the xy plane (a zero volume is then
    # found exactly), or not too thin to trust: the remedy message.
    for wrong in ([[1, 0, 0], [0.6, 1e-13, 0], [0, 0, 1]],
                  np.array([[1, 0, 0], [0.5, 1e-11, 0], [0, 0, 1]]) @ turn.T):
        assert "OpenMM needs" in refused(wrong)
    # Either side of the line, turned.
    for b_y, trusted in ((0.9e-12, False), (1.1e-12, True)):
        wrong = np.array([[1, 0, 0], [1, b_y, 0], [0, 0, 1]]) @ turn.T
        assert ("OpenMM needs" in refused(wrong)) == trusted
    # Volume, skewed (well within 1e12), turned or of unequal vectors: the
    # remedy message, not "no volume", and never anything but an InputError.
    skew = np.array([[3, 0, 0], [3e9 + 0.3, 3.3, 0], [0, 0, 2.7]])
    unequal = np.array([[1e200, 1e-200, 0], [0, 1, 0], [0, 0, 1]])
    with np.errstate(all="raise"):
        for wrong in (skew, skew @ turn.T, unequal, unequal @ turn.T):
            assert "OpenMM needs" in refused(wrong)
    # A thin box OpenMM takes is taken.
    taken([[1, 0, 0], [0.5, 1e-12, 0], [0, 0, 1]])
