"""The benchmark systems: how each is prepared, and how its frames are read.

Every system says the same four things:

- how to prepare it (a directory in the FastMDXplora layout),
- its discrete states, and which state each frame is in,
- two continuous features, whose joint histogram is the finer comparison,
- where it can, the exact answer at a temperature.

States and features are functions of an MDTraj trajectory of the atoms a
run saved, so they read run output directly.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from resremd import testsystems
from resremd.system import write_prepared
from resremd.thermo import BOLTZ


class BenchSystem:
    name = ""
    states: tuple[str, ...] = ()
    feature_names: tuple[str, str] = ("", "")
    #: (low, high, bins) for each feature.
    feature_bins: tuple[tuple[float, float, int], ...] = ()
    #: Platform for preparation and for seeking starts. None: the spec's.
    platform: str | None = None

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def prepare(self, out: Path, *, temperature_K: float, platform: str,
                seed: int) -> None:
        raise NotImplementedError

    def features(self, traj, prepared: Path) -> np.ndarray:
        """(frames, 2) array of the two features."""
        raise NotImplementedError

    def labels_from_features(self, f: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def labels(self, traj, prepared: Path) -> np.ndarray:
        return self.labels_from_features(self.features(traj, prepared))

    def histogram(self, f: np.ndarray) -> np.ndarray:
        (a0, a1, n0), (b0, b1, n1) = self.feature_bins
        h, _, _ = np.histogram2d(f[:, 0], f[:, 1], bins=[n0, n1],
                                 range=[[a0, a1], [b0, b1]])
        return h

    def exact(self, temperature_K: float) -> dict[str, np.ndarray] | None:
        """Exact state populations and feature histogram, if known."""
        return None

    def default_starts(self) -> dict[str, dict[str, Any]]:
        return {"start": {}}

    def torsions(self, topology) -> dict[str, list[int]]:
        """Named torsions a spec can bias by name (``atoms: omega``)."""
        return {}


# ---------------------------------------------------------------------------
class DoubleWell(BenchSystem):
    """A particle in an asymmetric double well: every answer is exact."""

    name = "double_well"
    states = ("left", "right")
    feature_names = ("x_nm", "y_nm")
    feature_bins = ((-2.0, 2.0, 40), (-0.2, 0.2, 10))
    platform = "Reference"

    def prepare(self, out, *, temperature_K, platform, seed):
        for start, x in (("right", 1.0), ("left", -1.0)):
            p = testsystems.double_well(start_x=x)
            write_prepared(out / start, p.system, p.topology, p.positions)

    def default_starts(self):
        # Prepared directly, no seeking needed: the favoured well and the
        # disfavoured one.
        return {"right": {}, "left": {}}

    def features(self, traj, prepared):
        return np.asarray(traj.xyz[:, 0, :2], dtype=float)

    def labels_from_features(self, f):
        return (f[:, 0] >= 0).astype(int)

    def exact(self, temperature_K):
        kt = BOLTZ * temperature_K
        grid = testsystems.grid()
        u = testsystems.double_well_energy(grid)
        px = np.exp(-(u - u.min()) / kt)
        px /= px.sum()
        (a0, a1, n0), (b0, b1, n1) = self.feature_bins
        xedges = np.linspace(a0, a1, n0 + 1)
        hx = np.histogram(grid, bins=xedges, weights=px)[0]
        sigma = math.sqrt(kt / testsystems.SPRING)
        yedges = np.linspace(b0, b1, n1 + 1)
        cdf = np.array([0.5 * (1 + math.erf(e / (sigma * math.sqrt(2))))
                        for e in yedges])
        hy = np.diff(cdf)
        left = testsystems.left_fraction(temperature_K)
        return {"populations": np.array([left, 1.0 - left]),
                "histogram": np.outer(hx, hy)}

    def exact_reservoir(self, path: Path, cfg: dict[str, Any], seed: int
                        ) -> None:
        """An exactly drawn reservoir, flawed on purpose if asked.

        cfg keys: ``kind`` (boltzmann or non_boltzmann), ``temperature_K``
        (what the frames are labelled with), ``sampled_temperature_K`` (what
        they are really drawn at; defaults to the label), ``n_frames``,
        ``drop_state`` (a state whose frames are removed).
        """
        from resremd.reservoir import write_reservoir

        rng = np.random.default_rng([seed, 17])
        kind = cfg.get("kind", "boltzmann")
        n = int(cfg["n_frames"])
        label_t = cfg.get("temperature_K")
        drop = cfg.get("drop_state")
        xs: list[np.ndarray] = []
        while sum(len(x) for x in xs) < n:
            if kind == "boltzmann":
                t = cfg.get("sampled_temperature_K", label_t)
                x = testsystems.exact_x_samples(t, 4 * n, rng)
            else:
                x = rng.uniform(-2.0, 2.0, 4 * n)
            if drop is not None:
                x = x[self.labels_from_features(
                    np.column_stack([x, x])) != self.states.index(drop)]
            xs.append(x)
        x = np.concatenate(xs)[:n]
        t_yz = cfg.get("sampled_temperature_K", label_t)
        frames = testsystems.double_well_frames(
            x, t_yz, rng, uniform_yz=kind == "non_boltzmann")
        prepared = testsystems.double_well()
        write_reservoir(path, topology=prepared.topology, positions=frames,
                        kind=kind, temperature_K=label_t,
                        source={"method": "exact draws", "benchmark": cfg,
                                "seed": seed})


# ---------------------------------------------------------------------------
class AlanineDipeptide(BenchSystem):
    """ACE-ALA-NME with Amber14, in GBn2 or TIP3P-FB.

    States are four regions of the (phi, psi) map, our own division:
    alpha_R (phi < 0, -120 <= psi < 50), beta (phi < -100, otherwise),
    PPII (-100 <= phi < 0, otherwise) and alpha_L (phi >= 0, which also
    holds C7ax). The second start is sought in the alpha_L region, the
    least populated, so the two starts sit on either side of the slowest
    barrier.
    """

    name = "alanine_dipeptide"
    states = ("alpha_R", "beta", "PPII", "alpha_L")
    feature_names = ("phi_deg", "psi_deg")
    feature_bins = ((-180.0, 180.0, 24), (-180.0, 180.0, 24))

    def prepare(self, out, *, temperature_K, platform, seed):
        topology, positions = testsystems.alanine_dipeptide()
        system, top, pos, box = testsystems.prepare_peptide(
            topology, positions, solvent=self.config.get("solvent", "implicit"),
            padding_nm=float(self.config.get("padding_nm", 1.2)),
            npt_ns=float(self.config.get("npt_ns", 0.5)),
            temperature_K=temperature_K, platform=platform, seed=seed)
        write_prepared(out / "extended", system, top, pos, box)

    def default_starts(self):
        return {"extended": {},
                "alpha_L": {"seek": {"state": "alpha_L", "temperature_K": 600,
                                     "max_ns": 20, "check_ps": 1}}}

    def features(self, traj, prepared):
        import mdtraj as md

        _, phi = md.compute_phi(traj)
        _, psi = md.compute_psi(traj)
        return np.degrees(np.column_stack([phi[:, 0], psi[:, 0]]))

    def labels_from_features(self, f):
        phi, psi = f[:, 0], f[:, 1]
        alpha_r = (phi < 0) & (psi >= -120) & (psi < 50)
        labels = np.where(phi >= 0, 3, np.where(alpha_r, 0,
                                                np.where(phi < -100, 1, 2)))
        return labels.astype(int)


# ---------------------------------------------------------------------------
class Chignolin(BenchSystem):
    """CLN025 (YYDPETGTWY), from PDB 5AWL, with Amber14 in GBn2 or TIP3P-FB.

    Folded means a C-alpha RMSD to the crystal structure of at most
    ``folded_rmsd_nm`` (0.2 nm by default). The second feature is the C-alpha
    radius of gyration. The second start is sought by unfolding at high
    temperature until the RMSD passes ``unfolded_rmsd_nm``.
    """

    name = "chignolin"
    states = ("folded", "unfolded")
    feature_names = ("rmsd_ca_nm", "rg_ca_nm")
    feature_bins = ((0.0, 1.0, 25), (0.3, 1.3, 25))

    def prepare(self, out, *, temperature_K, platform, seed):
        from openmm import app, unit

        pdb_path = Path(self.config["pdb"])
        pdb = app.PDBFile(str(pdb_path))
        protein = app.Modeller(pdb.topology, pdb.positions)
        first = next(pdb.topology.chains())
        protein.delete([r for r in pdb.topology.residues()
                        if r.chain is not first or r.name == "HOH"
                        or r.name not in app.PDBFile._standardResidues])
        heavy = app.Modeller(protein.topology, protein.positions)
        heavy.delete([a for a in protein.topology.atoms()
                      if a.element is not None and a.element.symbol == "H"])
        pos = heavy.getPositions().value_in_unit(unit.nanometer)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "native.pdb", "w") as fh:
            app.PDBFile.writeFile(heavy.topology, np.asarray(pos) * 10.0, fh)
        system, top, positions, box = testsystems.prepare_peptide(
            heavy.topology, np.asarray(pos),
            solvent=self.config.get("solvent", "implicit"),
            padding_nm=float(self.config.get("padding_nm", 1.2)),
            npt_ns=float(self.config.get("npt_ns", 0.5)),
            temperature_K=temperature_K, platform=platform, seed=seed)
        write_prepared(out / "folded", system, top, positions, box)

    def default_starts(self):
        return {"folded": {},
                "unfolded": {"seek": {
                    "feature": "rmsd_ca_nm",
                    "above": float(self.config.get("unfolded_rmsd_nm", 0.5)),
                    "temperature_K": 600, "max_ns": 50, "check_ps": 10}}}

    def _native(self, prepared: Path):
        import mdtraj as md

        native = md.load(str(Path(prepared).parent / "native.pdb"))
        return native.atom_slice(native.topology.select("name CA"))

    def features(self, traj, prepared):
        import mdtraj as md

        ca = traj.topology.select("protein and name CA")
        sub = traj.atom_slice(ca)
        native = self._native(prepared)
        if native.n_atoms != sub.n_atoms:
            raise ValueError("The trajectory's C-alpha atoms do not match "
                             "the native structure's.")
        rmsd = md.rmsd(sub, native)
        rg = md.compute_rg(sub)
        return np.column_stack([rmsd, rg])

    def labels_from_features(self, f):
        cut = float(self.config.get("folded_rmsd_nm", 0.2))
        return (f[:, 0] > cut).astype(int)


# ---------------------------------------------------------------------------
class TorsionModel(BenchSystem):
    """Four atoms with an 80 kJ/mol cis/trans barrier: exact, and out of
    reach of temperature. Cis is phi within 90 degrees of 0."""

    name = "torsion_model"
    states = ("cis", "trans")
    feature_names = ("phi_deg", "angle_deg")
    feature_bins = ((-180.0, 180.0, 24), (85.0, 133.0, 12))
    platform = "Reference"

    def prepare(self, out, *, temperature_K, platform, seed):
        for start, cis in (("trans", False), ("cis", True)):
            p = testsystems.torsion_model(cis=cis)
            write_prepared(out / start, p.system, p.topology, p.positions)

    def default_starts(self):
        return {"trans": {}, "cis": {}}

    def torsions(self, topology):
        return {"phi": [0, 1, 2, 3]}

    def features(self, traj, prepared):
        import mdtraj as md

        phi = md.compute_dihedrals(traj, [[0, 1, 2, 3]])[:, 0]
        angle = md.compute_angles(traj, [[0, 1, 2]])[:, 0]
        return np.degrees(np.column_stack([phi, angle]))

    def labels_from_features(self, f):
        return (np.abs(f[:, 0]) >= 90.0).astype(int)

    def exact(self, temperature_K):
        kt = BOLTZ * temperature_K
        (a0, a1, n0), (b0, b1, n1) = self.feature_bins
        phi = testsystems.torsion_grid()
        p_phi = np.exp(-(testsystems.torsion_energy(phi)
                         - testsystems.torsion_energy(phi).min()) / kt)
        p_phi /= p_phi.sum()
        h_phi = np.histogram(np.degrees(phi), bins=np.linspace(a0, a1, n0 + 1),
                             weights=p_phi)[0]
        # A bond angle's density carries the sin(theta) Jacobian.
        th = np.radians(np.linspace(b0, b1, 4001))
        p_th = np.sin(th) * np.exp(-0.5 * testsystems._ANGLE_K
                                   * (th - testsystems._ANGLE_RAD) ** 2 / kt)
        h_th = np.histogram(np.degrees(th), bins=np.linspace(b0, b1, n1 + 1),
                            weights=p_th)[0]
        cis = testsystems.cis_fraction(temperature_K)
        return {"populations": np.array([cis, 1.0 - cis]),
                "histogram": np.outer(h_phi, h_th / h_th.sum())}


# ---------------------------------------------------------------------------
class ProlineDipeptide(BenchSystem):
    """Ac-Pro-NMe with Amber14, in GBn2 or TIP3P-FB: prolyl cis/trans.

    States by omega, the ACE-PRO peptide bond: cis within 90 degrees of 0.
    The second feature is the proline psi. The cis start is sought with the
    omega barrier lowered by a bias, which is removed before the start is
    written.
    """

    name = "proline_dipeptide"
    states = ("cis", "trans")
    feature_names = ("omega_deg", "psi_deg")
    feature_bins = ((-180.0, 180.0, 24), (-180.0, 180.0, 24))

    def prepare(self, out, *, temperature_K, platform, seed):
        topology, positions = testsystems.proline_dipeptide(180.0)
        system, top, pos, box = testsystems.prepare_peptide(
            topology, positions, solvent=self.config.get("solvent", "implicit"),
            padding_nm=float(self.config.get("padding_nm", 1.2)),
            npt_ns=float(self.config.get("npt_ns", 0.5)),
            temperature_K=temperature_K, platform=platform, seed=seed)
        write_prepared(out / "trans", system, top, pos, box)

    def default_starts(self):
        return {"trans": {},
                "cis": {"seek": {
                    "state": "cis", "temperature_K": 400, "max_ns": 20,
                    "check_ps": 2,
                    "bias_torsions": [{"atoms": "omega",
                                       "energy": "-k*sin(theta)^2",
                                       "parameters": {"k": 60.0}}]}}}

    def torsions(self, topology):
        return {"omega": testsystems.omega_atoms(topology)}

    def features(self, traj, prepared):
        import mdtraj as md

        top = traj.topology.to_openmm()
        omega = md.compute_dihedrals(traj, [testsystems.omega_atoms(top)])
        _, psi = md.compute_psi(traj)
        return np.degrees(np.column_stack([omega[:, 0], psi[:, 0]]))

    def labels_from_features(self, f):
        return (np.abs(f[:, 0]) >= 90.0).astype(int)


SYSTEMS = {cls.name: cls for cls in (DoubleWell, AlanineDipeptide, Chignolin,
                                     TorsionModel, ProlineDipeptide)}


def get(config: dict[str, Any]) -> BenchSystem:
    name = config.get("name")
    if name not in SYSTEMS:
        raise ValueError(f"Unknown benchmark system {name!r}; known: "
                         f"{', '.join(SYSTEMS)}.")
    return SYSTEMS[name](config)
