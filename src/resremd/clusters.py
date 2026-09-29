"""Reservoirs weighted by cluster populations.

A reservoir's frames can be grouped into clusters, for example the basins
of a few torsions, and each cluster given the population it has at the
reservoir temperature. Two builders do this, both writing a weighted
reservoir:

``frames`` (the default) keeps every frame and scales the weights of each
cluster's frames so that together they carry its population:
w_i = b_i P_c / sum_{j in c} b_j, with b the source's own weights. This is
exact when the frames within each cluster are a Boltzmann sample of that
cluster and P are the true populations: a simulation that equilibrates
within basins but crosses between them too rarely to weigh them, with the
populations from somewhere better (a longer or biased simulation, MBAR,
a Markov model). Frames of clusters without a population get no weight;
a cluster with a population but no frames is refused, since nothing could
carry it.

``representatives`` keeps one frame per cluster, weighted by the cluster's
population: a smaller reservoir, but an approximation. One structure
stands in for its whole cluster, so it is close only when clusters are
narrow next to the spread of the top replica's energies. The run's
reservoir checks then have little to judge by; prefer ``frames``.

Amber's ``clusterdihedral`` writes the populations it found to a
``clusterinfo`` file, which :func:`read_populations` reads.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .errors import InputError, ReservoirError

logger = logging.getLogger("resremd")


def read_amber_clusterinfo(path: str | Path) -> dict[int, float]:
    """Cluster populations from an Amber ``clusterdihedral`` clusterinfo
    file: the number of torsions, one line per torsion, the number of
    clusters, then one line per cluster starting with its number (from 1)
    and its count."""
    lines = [ln.split() for ln in Path(path).read_text().splitlines()
             if ln.strip()]
    try:
        n_torsions = int(lines[0][0])
        n_clusters = int(lines[1 + n_torsions][0])
        rows = lines[2 + n_torsions: 2 + n_torsions + n_clusters]
        counts = {int(r[0]): float(r[1]) for r in rows}
    except (IndexError, ValueError) as exc:
        raise InputError(f"{path} is not an Amber clusterinfo file ({exc}).",
                         code="resremd.input.clusters") from exc
    if len(counts) != n_clusters:
        raise InputError(f"{path} lists {len(counts)} clusters, not the "
                         f"{n_clusters} it announces.",
                         code="resremd.input.clusters")
    return counts


def read_populations(populations: Any) -> dict[int, float]:
    """Populations by cluster label, normalised: a dict, a JSON file of
    one, or an Amber clusterinfo file."""
    return _read_populations(populations)[0]


def _read_populations(populations: Any) -> tuple[dict[int, float], str]:
    """The populations, and where they came from: `mapping`, `json` or
    `clusterinfo`."""
    origin = "mapping"
    if isinstance(populations, (str, Path)):
        path = Path(populations)
        if not path.is_file():
            raise InputError(f"No populations file at {path}.",
                             code="resremd.input.missing")
        try:
            populations = json.loads(path.read_text())
            origin = "json"
        except json.JSONDecodeError:
            populations = read_amber_clusterinfo(path)
            origin = "clusterinfo"
    try:
        out = {int(k): float(v) for k, v in dict(populations).items()}
    except (TypeError, ValueError) as exc:
        raise InputError("Populations map cluster labels (integers) to "
                         "numbers.", code="resremd.input.clusters") from exc
    values = np.array(list(out.values()))
    if not out or not np.all(np.isfinite(values)) or np.any(values < 0) \
            or values.sum() <= 0:
        raise InputError("Populations must be finite, non-negative and not "
                         "all zero.", code="resremd.input.clusters")
    total = values.sum()
    return {k: v / total for k, v in out.items()}, origin


def read_labels(labels: Any, n_frames: int) -> np.ndarray:
    """One integer cluster label per frame: an array, a .npy file, or text
    with one label per line."""
    if isinstance(labels, (str, Path)):
        path = Path(labels)
        if not path.is_file():
            raise InputError(f"No labels file at {path}.",
                             code="resremd.input.missing")
        labels = np.load(path) if path.suffix == ".npy" \
            else np.loadtxt(path, ndmin=1)
    lab = np.asarray(labels)
    if lab.shape != (n_frames,):
        raise InputError(f"{lab.size} cluster labels for {n_frames} frames.",
                         code="resremd.input.clusters")
    if not np.all(np.equal(np.mod(lab, 1), 0)):
        raise InputError("Cluster labels are integers.",
                         code="resremd.input.clusters")
    return lab.astype(int)


def torsion_cells(angles_deg: np.ndarray, bins: int) -> np.ndarray:
    """A label per frame: the cell of a grid of ``bins`` arcs per torsion
    (arcs as in :func:`resremd.build.torsion_regions`), numbered from 0."""
    from .build import torsion_regions

    a = np.asarray(angles_deg, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    regions = np.stack([torsion_regions(a[:, j], bins)
                        for j in range(a.shape[1])], axis=1)
    return np.ravel_multi_index(regions.T, (bins,) * a.shape[1])


def cluster_weights(labels: np.ndarray, populations: dict[int, float],
                    base: np.ndarray | None = None) -> np.ndarray:
    """Frame weights that give each cluster its population, keeping the
    relative weights ``base`` within it."""
    labels = np.asarray(labels, dtype=int)
    b = np.ones(len(labels)) if base is None else np.asarray(base, float)
    held = {int(c): float(b[labels == c].sum()) for c in np.unique(labels)}
    missing = {c: p for c, p in populations.items()
               if p > 0 and held.get(c, 0.0) <= 0}
    if missing:
        shown = ", ".join(f"{c} ({p:.3g})" for c, p in
                          sorted(missing.items(), key=lambda x: -x[1])[:5])
        raise ReservoirError(
            f"{len(missing)} clusters with a population have no frames to "
            f"carry it, {sum(missing.values()):.3g} of the total: {shown}. "
            "Add frames from them, or give them no population if they are "
            "meant to be left out.", code="resremd.reservoir.clusters")
    w = np.zeros(len(labels))
    for c, p in populations.items():
        if p > 0:
            members = labels == c
            w[members] = b[members] * p / held[c]
    return w / w.sum()


def pick_representatives(labels: np.ndarray, base: np.ndarray,
                         angles_deg: np.ndarray | None,
                         rng: np.random.Generator) -> dict[int, int]:
    """One frame per cluster: the one nearest the cluster's weighted
    circular mean of the torsions, or without torsions a frame drawn by
    weight."""
    out = {}
    for c in np.unique(labels):
        members = np.flatnonzero((labels == c) & (base > 0))
        w = base[members]
        if w.sum() <= 0:
            continue
        if angles_deg is None:
            out[int(c)] = int(rng.choice(members, p=w / w.sum()))
            continue
        a = np.radians(np.atleast_2d(angles_deg.T).T[members])
        mean = np.arctan2((w[:, None] * np.sin(a)).sum(0),
                          (w[:, None] * np.cos(a)).sum(0))
        distance = (1.0 - np.cos(a - mean)).sum(axis=1)
        out[int(c)] = int(members[np.argmin(distance)])
    return out


def cluster_reservoir(**settings: Any) -> dict[str, Any]:
    """A weighted reservoir from ``source`` with its clusters given
    populations. Settings are those of :data:`resremd.options.CLUSTER`.

    Clusters are ``labels`` (one per frame) or the cells of a grid of
    ``bins`` arcs on each of ``torsions`` (atom index quadruples).
    ``populations`` maps cluster labels to populations (a dict, a JSON
    file, or an Amber clusterinfo file); by default each cluster keeps the
    population it has in ``source``, which only a ``representatives``
    reservoir can use. See the module's description for the two builds.
    """
    from .options import CLUSTER, resolve

    given = dict(settings)
    for key in ("source", "output", "labels", "populations"):
        if isinstance(given.get(key), Path):
            given[key] = str(given[key])
    if isinstance(given.get("labels"), np.ndarray):
        given["labels"] = given["labels"].tolist()
    return _cluster_reservoir(**resolve(CLUSTER, given))


def _cluster_reservoir(*, source: str, output: str, labels: Any,
                       torsions: list[list[int]] | None, bins: int,
                       populations: Any, representatives: bool,
                       random_seed: int | None) -> dict[str, Any]:
    from .build import torsion_angles_deg
    from .reservoir import Reservoir, ReservoirWriter, write_json

    src = Reservoir.open(source)
    out = Path(output)
    if out.resolve() == src.path.resolve():
        raise InputError("Write the clustered reservoir somewhere new, not "
                         "over its source.", code="resremd.input.output")
    if (out / "reservoir.json").exists():
        raise InputError(f"{out} already holds a reservoir.",
                         code="resremd.input.output")
    if src.kind == "non_boltzmann":
        raise ReservoirError(
            "A non-Boltzmann reservoir's frames are not a Boltzmann sample "
            "within their clusters, so cluster populations cannot make it "
            "one.", code="resremd.reservoir.kind")
    if (labels is None) == (torsions is None):
        raise InputError("Give the clusters as `labels` or as `torsions`, "
                         "one of the two.", code="resremd.input.clusters")
    n = src.n_frames
    angles = None
    if torsions is not None:
        if bins < 2:
            raise InputError("A torsion grid needs at least two arcs per "
                             "torsion.", code="resremd.input.clusters")
        watch = [[int(a) for a in t] for t in torsions]
        if any(len(t) != 4 or min(t) < 0 or max(t) >= src.n_atoms
               for t in watch):
            raise InputError(f"Torsions are four atom indices below "
                             f"{src.n_atoms}.", code="resremd.input.clusters")
        angles = np.array([torsion_angles_deg(src.frame(k)[0], watch)
                           for k in range(n)])
        lab = torsion_cells(angles, bins)
    else:
        lab = read_labels(labels, n)
    base = np.ones(n) / n if src.weights is None else \
        np.asarray(src.weights, dtype=float)
    own = {int(c): float(base[lab == c].sum()) for c in np.unique(lab)}
    if populations is None:
        if not representatives:
            raise InputError(
                "Keeping every frame with the clusters' own populations "
                "gives back the same reservoir; give `populations` to change "
                "them, or ask for `representatives`.",
                code="resremd.input.clusters")
        # Label -1 marks frames in no cluster; they get no representative.
        pops = {c: p for c, p in own.items() if c != -1}
        total = sum(pops.values())
        pops = {c: p / total for c, p in pops.items()}
    else:
        pops, origin = _read_populations(populations)
        if origin == "clusterinfo" and torsions is not None:
            raise InputError(
                "An Amber clusterinfo file numbers its own clusters, which "
                "are not the cells of this torsion grid. Give its per-frame "
                "cluster numbers as `labels`.",
                code="resremd.input.clusters")
    dropped = float(sum(base[lab == c].sum() for c in own
                        if pops.get(c, 0.0) <= 0))
    if dropped > 0:
        logger.warning(
            "Frames holding %.3g of the source's weight are in clusters "
            "without a population and are left out; the reservoir lacks "
            "whatever states they held.", dropped / base.sum())
    if representatives:
        chosen = pick_representatives(
            lab, base, angles, np.random.default_rng(random_seed))
        missing = [c for c, p in pops.items() if p > 0 and c not in chosen]
        if missing:
            raise ReservoirError(
                f"Clusters {missing[:5]} have a population but no frames.",
                code="resremd.reservoir.clusters")
        keep = np.array(sorted(chosen[c] for c, p in pops.items()
                               if p > 0))
        w = np.array([pops[int(lab[k])] for k in keep])
        logger.warning(
            "Each of the %d representatives stands in for its whole "
            "cluster, which is exact only for narrow clusters; keeping every "
            "frame (`representatives: false`) is exact.", len(keep))
    else:
        weights = cluster_weights(lab, pops, base)
        keep = np.flatnonzero(weights > 0)
        w = weights[keep]
    w = w / w.sum()
    periodic = src.box is not None
    writer = ReservoirWriter(out, n_frames=len(keep), n_atoms=src.n_atoms,
                             periodic=periodic)
    for i, k in enumerate(keep):
        pos, box = src.frame(int(k))
        writer.write(i, pos, box)
    writer.flush()
    np.save(out / "weights.npy", w)
    np.save(out / "cluster_labels.npy", lab[keep])
    built = src.path / "build_potential_kjmol.npy"
    if built.exists():
        np.save(out / "build_potential_kjmol.npy", np.load(built)[keep])
    shutil.copyfile(src.path / "topology.pdb", out / "topology.pdb")
    meta = {k: v for k, v in src.meta.items()
            if k not in ("convergence", "statistics")}
    previous = dict(src.meta.get("source") or {})
    meta.update({
        "kind": "weighted",
        "n_frames": int(len(keep)),
        "source": {
            "method": "cluster representatives" if representatives
            else "cluster populations",
            "from": str(src.path.resolve()),
            "from_frames_sha256": src.content_digest(),
            "clusters": ({"torsions": [[int(a) for a in t] for t in torsions],
                          "bins": int(bins)} if torsions is not None
                         else {"labels": "given"}),
            "populations": "given" if populations is not None
            else "the source's own",
            "n_clusters": int(sum(p > 0 for p in pops.values())),
            "dropped_source_weight": dropped / float(base.sum()),
            "effective_frames": float(1.0 / np.sum(w * w)),
            **({"system_sha256": previous["system_sha256"]}
               if "system_sha256" in previous else {}),
            "previous": previous,
        },
        "complete": True,
    })
    write_json(out / "reservoir.json", meta)
    Reservoir.open(out)
    logger.info("Clustered reservoir: %d frames in %d clusters, %.0f "
                "effective frames", len(keep), meta["source"]["n_clusters"],
                meta["source"]["effective_frames"])
    return meta
