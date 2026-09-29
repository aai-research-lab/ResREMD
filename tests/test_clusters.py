"""Reservoirs weighted by cluster populations, against exact answers.

The double well's two wells are the clusters. The frames are an exact
Boltzmann sample within each well at 520 K but hold the wells half and
half, far from their true populations; given those populations the
clustered reservoir is exact again.
"""

import json

import numpy as np
import pytest

import resremd
from resremd import clusters, testsystems
from resremd.errors import InputError, ReservoirError
from resremd.reservoir import Reservoir, write_reservoir

from test_exactness import TEMPERATURES, check_against_exact

T_R = 520.0


def _equal_wells(path, n=20000, seed=4):
    """Exact within each well, but half the frames in each."""
    rng = np.random.default_rng(seed)
    x = testsystems.exact_x_samples(T_R, 4 * n, rng)
    x = np.concatenate([x[x < 0][: n // 2], x[x >= 0][: n // 2]])
    frames = testsystems.double_well_frames(x, T_R, rng)
    write_reservoir(path, topology=testsystems.double_well().topology,
                    positions=frames, kind="boltzmann", temperature_K=T_R)
    return (x >= 0).astype(int)


def _true_populations():
    left = testsystems.left_fraction(T_R)
    return {0: left, 1: 1.0 - left}


def test_weights_give_each_cluster_its_population():
    labels = np.array([0, 0, 0, 1, 2, 2])
    base = np.array([1.0, 2.0, 1.0, 5.0, 1.0, 3.0])
    w = clusters.cluster_weights(labels, {0: 0.5, 1: 0.2, 2: 0.3}, base)
    assert w[labels == 0].sum() == pytest.approx(0.5)
    assert w[1] / w[0] == pytest.approx(2.0)        # kept within a cluster
    assert w[labels == 2].sum() == pytest.approx(0.3)
    # A cluster without a population is dropped; one without frames refused.
    w = clusters.cluster_weights(labels, {0: 1.0}, base)
    assert w[labels != 0].sum() == 0
    with pytest.raises(ReservoirError, match="no frames"):
        clusters.cluster_weights(labels, {0: 0.5, 7: 0.5}, base)


def test_amber_clusterinfo_is_read(tmp_path):
    info = tmp_path / "clusterinfo"
    info.write_text("2\n         5         7         9        15        10"
                    "  -180.000\n         7         9        15        17"
                    "        10  -180.000\n3\n         1        60   3   7\n"
                    "         2        30   4   7\n         3        10   3"
                    "   6\n")
    pops = clusters.read_populations(info)
    assert pops == pytest.approx({1: 0.6, 2: 0.3, 3: 0.1})
    (tmp_path / "p.json").write_text(json.dumps({"1": 3, "2": 1}))
    assert clusters.read_populations(tmp_path / "p.json") == \
        pytest.approx({1: 0.75, 2: 0.25})


def test_a_reservoir_with_its_clusters_reweighted_is_exact(tmp_path):
    labels = _equal_wells(tmp_path / "r")
    meta = resremd.cluster_reservoir(source=tmp_path / "r", output=tmp_path / "c",
                                      labels=labels,
                                      populations=_true_populations())
    assert meta["kind"] == "weighted" and meta["n_frames"] == labels.size
    r = Reservoir.open(tmp_path / "c")
    left = r.weights[np.asarray(r.positions[:, 0, 0]) < 0].sum()
    assert left == pytest.approx(testsystems.left_fraction(T_R))
    resremd.run(testsystems.double_well(), output=str(tmp_path / "run"),
                reservoir=str(tmp_path / "c"), temperatures_K=TEMPERATURES,
                production_steps=250 * 30000, exchange_interval_steps=250,
                trajectory_interval_steps=250, friction_per_ps=5.0,
                platform="Reference", random_seed=5, save_selection="all",
                equilibration_ns=0.0, minimize=False)
    check_against_exact(tmp_path / "run")


def test_representatives_are_one_frame_per_cluster(tmp_path):
    rng = np.random.default_rng(1)
    frames = testsystems.double_well_frames(
        testsystems.exact_x_samples(T_R, 2000, rng), T_R, rng)
    write_reservoir(tmp_path / "r", topology=testsystems.double_well().topology,
                    positions=frames, kind="boltzmann", temperature_K=T_R)
    labels = np.digitize(frames[:, 0, 0], np.linspace(-2, 2, 21))
    meta = resremd.cluster_reservoir(source=tmp_path / "r", output=tmp_path / "c",
                                      labels=labels, representatives=True,
                                      random_seed=3)
    r = Reservoir.open(tmp_path / "c")
    kept = np.load(tmp_path / "c/cluster_labels.npy")
    assert meta["source"]["method"] == "cluster representatives"
    assert sorted(kept.tolist()) == sorted(set(labels.tolist()))
    for c, w in zip(kept, r.weights):
        assert w == pytest.approx(np.mean(labels == c))


def test_torsion_cells_and_refusals(tmp_path):
    cells = clusters.torsion_cells(np.array([[0.0, 179.0], [0.0, -179.0],
                                             [90.0, 0.0]]), 4)
    assert cells[0] == cells[1] != cells[2]
    testsystems.write_double_well_reservoir(tmp_path / "u",
                                            kind="non_boltzmann",
                                            n_frames=10, temperature_K=None)
    with pytest.raises(ReservoirError, match="non-Boltzmann"):
        resremd.cluster_reservoir(source=tmp_path / "u", output=tmp_path / "x",
                                   labels=np.zeros(10, dtype=int))
    testsystems.write_double_well_reservoir(tmp_path / "b", kind="boltzmann",
                                            n_frames=10, temperature_K=T_R)
    with pytest.raises(InputError, match="same reservoir"):
        resremd.cluster_reservoir(source=tmp_path / "b", output=tmp_path / "x",
                                   labels=np.zeros(10, dtype=int))
    with pytest.raises(InputError, match="labels"):
        resremd.cluster_reservoir(source=tmp_path / "b", output=tmp_path / "x",
                                   labels=np.zeros(9, dtype=int), populations={0: 1})


def test_the_command_line_clusters_a_reservoir(tmp_path, capsys):
    from resremd.cli import main

    labels = _equal_wells(tmp_path / "r", n=200)
    np.save(tmp_path / "labels.npy", labels)
    (tmp_path / "p.json").write_text(json.dumps(
        {str(k): v for k, v in _true_populations().items()}))
    assert main(["-q", "reservoir", "cluster", "--source", str(tmp_path / "r"),
                 "--output", str(tmp_path / "c"), "--labels",
                 str(tmp_path / "labels.npy"), "--populations",
                 str(tmp_path / "p.json")]) == 0
    assert Reservoir.open(tmp_path / "c").kind == "weighted"
    assert main(["options", "cluster"]) == 0
    assert "representatives" in capsys.readouterr().out


def test_dropped_clusters_are_reported_and_zero_weights_never_represent(
        tmp_path, caplog):
    rng = np.random.default_rng(1)
    frames = testsystems.double_well_frames(
        testsystems.exact_x_samples(T_R, 40, rng), T_R, rng)
    weights = np.ones(40)
    weights[:10] = 0.0
    write_reservoir(tmp_path / "r", topology=testsystems.double_well().topology,
                    positions=frames, kind="weighted", temperature_K=T_R,
                    weights=weights)
    labels = np.repeat([0, 1, 2, -1], 10)
    meta = resremd.cluster_reservoir(source=tmp_path / "r",
                                     output=tmp_path / "c", labels=labels,
                                     populations={1: 0.5, 2: 0.5})
    assert meta["source"]["dropped_source_weight"] == pytest.approx(1 / 3)
    assert "left out" in caplog.text
    # Cluster 0 holds only zero-weight frames and -1 is no cluster: neither
    # gets a representative.
    meta = resremd.cluster_reservoir(source=tmp_path / "r",
                                     output=tmp_path / "d", labels=labels,
                                     representatives=True)
    kept = np.load(tmp_path / "d/cluster_labels.npy")
    assert sorted(kept.tolist()) == [1, 2]


def test_a_clusterinfo_file_is_not_for_a_torsion_grid(tmp_path):
    testsystems.write_double_well_reservoir(tmp_path / "b", kind="boltzmann",
                                            n_frames=10, temperature_K=T_R)
    (tmp_path / "ci").write_text("0\n1\n 1 10\n")
    with pytest.raises(InputError, match="clusterinfo"):
        resremd.cluster_reservoir(source=tmp_path / "b",
                                  output=tmp_path / "x",
                                  torsions=[[0, 0, 0, 0]],
                                  populations=str(tmp_path / "ci"))
