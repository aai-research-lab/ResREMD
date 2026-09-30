import json

from resremd import testsystems
from resremd.cli import main
from resremd.system import write_prepared
from resremd.throughput import measure


def test_throughput_reports_each_context_count():
    rows = measure(testsystems.lj_box(), n_replicas=4,
                   contexts_per_device=[1, 2], steps=20, cycles=2,
                   platform="Reference")
    assert [r["contexts"] for r in rows] == [1, 2]
    for r in rows:
        assert r["ns_per_day_total"] == \
            r["ns_per_day_per_replica"] * 4
        assert 0.0 < r["overhead_fraction"] <= 1.0
    assert not rows[0]["replicas_resident"]


def test_throughput_command_with_rest2(tmp_path, capsys):
    p = testsystems.lj_box()
    write_prepared(tmp_path / "setup", p.system, p.topology, p.positions,
                   p.box)
    assert main(["-q", "throughput", "--prepared", str(tmp_path / "setup"),
                 "--n-replicas", "2", "--contexts-per-device", "1",
                 "--steps", "10", "--cycles", "1", "--platform", "Reference",
                 "--rest2", "--rest2-selection", "all", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["contexts_per_device"] == 1


def test_the_cpu_runs_one_single_threaded_context_per_core():
    import numpy as np

    from resremd.engine import Engine, available_cores
    from resremd.thermo import ensemble_of

    p = testsystems.lj_box()
    ensemble, _ = ensemble_of(p.system)
    engine = Engine(system=p.system, ensemble=ensemble, n_replicas=16,
                    integrator="langevin_middle", timestep_fs=2.0,
                    friction_per_ps=1.0, temperature_K=100.0, platform="CPU",
                    precision="mixed", devices=None, contexts_per_device=None,
                    cpu_threads=None, seeds=np.random.default_rng(0))
    try:
        cores = available_cores()
        assert engine.describe()["contexts"] == min(16, cores)
        if cores > 1:
            threads = engine.slots[0].context.getPlatform().getPropertyValue(
                engine.slots[0].context, "Threads")
            assert int(threads) == 1
    finally:
        engine.close()
