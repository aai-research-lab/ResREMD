"""The harness, end to end on the smoke spec (a minute on a CPU)."""

import json
import shutil
from pathlib import Path

import pytest

from resbench import analyze, jobs, spec
from resbench.__main__ import main

HERE = Path(__file__).resolve().parent.parent


def test_every_shipped_spec_is_valid():
    for path in (HERE / "specs").glob("*.yml"):
        spec.load(path)


@pytest.mark.parametrize("change,match", [
    ({"surprise": 1}, "Unknown spec keys"),
    ({"run": {"random_seed": 3, "production_steps": 1000}}, "harness sets"),
    ({"run": {"production_stps": 1000}}, "production_steps"),
    ({"methods": [{"name": "a"}, {"name": "a"}]}, "used twice"),
    ({"methods": [{"name": "a", "reservoir": {}}]}, "either"),
    ({"reference": {"kind": "psychic"}}, "reference.kind"),
    ({"analysis": {"threshold_tvd": 0.1}}, "Unknown `analysis` keys"),
    ({"methods": [{"name": "a"}, {"name": "b", "runs": False, "reservoir": {
        "exact": {"temperature_K": 500, "n_frames": 10}}}],
      "reference": {"kind": "pooled", "methods": ["b"]}}, "makes none"),
    ({"reference": {"kind": "pooled"}}, "names the methods"),
    ({"seeds": [1, 1]}, "distinct"),
    ({"methods": [{"name": "a", "reservoir": {"exact": {
        "temperature_K": 500, "n_frames": 10,
        "sampled_temperture_K": 600}}}]}, "unknown exact"),
    ({"methods": [{"name": "a", "temperatures_K": [310.0, 400.0]}]},
     "ladder starts"),
])
def test_spec_mistakes_are_refused(change, match):
    raw = spec.load(HERE / "specs/smoke.yml")
    raw = {k: v for k, v in raw.items()}
    raw.update(change)
    with pytest.raises(spec.SpecError, match=match):
        spec.check(raw)


def test_smoke_end_to_end(tmp_path):
    out = tmp_path / "bench"
    assert main(["plan", str(HERE / "specs/smoke.yml"), str(out)]) == 0
    assert (out / "slurm/submit.sh").exists()
    sbatch = (out / "slurm/runs.sbatch").read_text()
    assert "resbench job" in sbatch
    jobs.run_all(out)
    assert all(state.startswith(("done", "complete"))
               for _, state in jobs.status(out))
    summary = analyze.analyze(out)
    assert summary["equal_cost"] and not summary["missing_runs"]
    plans = {m: summary["methods"][m]["cost_per_run_md_steps"]
             for m in summary["methods"]}
    assert max(plans.values()) - min(plans.values()) <= 3 * 250, \
        "equal cost within one exchange interval over the replicas"
    exact = summary["methods"]["resremd_exact"]
    assert summary["mbar"]
    for start in ("right", "left"):
        assert exact["starts"][start]["final_tv_error_mean"] < 0.05
        assert exact["starts"][start]["final_tv_error_mbar_mean"] < 0.05
        assert len(exact["starts"][start]["converged_tv_ns"]["values"]) == 2
    assert exact["reservoir_check_z_max_abs"] < 4
    assert exact["coverage_z_max_abs"] < 4
    assert not exact["coverage_unsupported_states"]
    # Each run has its own reservoir, made from its own start.
    assert (out / "reservoirs/resremd_generated/left/seed_1/reservoir.json"
            ).exists()
    # Reservoirs from opposite starts, compared seed by seed.
    assert len(summary["methods"]["resremd_generated"]
               ["reservoir_starts_tv"]) == 2
    for name in ("curves.csv", "convergence.csv", "agreement.csv",
                 "reservoirs.csv", "reservoir_agreement.csv", "summary.json"):
        assert (out / "analysis" / name).stat().st_size > 0
    json.loads((out / "analysis/summary.json").read_text())  # strict JSON
    # Running again redoes nothing.
    before = json.loads((out / "runs/remd/right/seed_1/manifest.json")
                        .read_text())["updated"]
    jobs.run_all(out)
    after = json.loads((out / "runs/remd/right/seed_1/manifest.json")
                       .read_text())["updated"]
    assert before == after
    _plots_run(out, tmp_path)
    # A reference file from these runs, used by another spec.
    assert main(["reference", str(out), str(tmp_path / "ref.json"),
                 "--methods", "resremd_exact"]) == 0
    ref = json.loads((tmp_path / "ref.json").read_text())
    assert abs(ref["populations"][0] - 0.912) < 0.05


def test_changed_settings_are_not_reused(tmp_path):
    out = tmp_path / "bench"
    main(["plan", str(HERE / "specs/smoke.yml"), str(out)])
    jobs.run_job(out, "prepare")
    jobs.run_job(out, "run/remd/right/1")
    text = (out / "spec.yml").read_text().replace("seeds: [1, 2]",
                                                  "seeds: [1, 2]\nequal_cost: false")
    (out / "spec.yml").write_text(text)
    with pytest.raises(RuntimeError, match="different settings"):
        jobs.run_job(out, "run/remd/right/1")


TORSION_SPEC = """
name: torsion_test
system: {name: torsion_model}
temperatures_K: [300.0, 400.0]
seeds: [1, 2]
run: {production_steps: 50000, exchange_interval_steps: 250,
      trajectory_interval_steps: 250, equilibration_ns: 0.0,
      minimize: false, save_selection: all, platform: Reference}
methods:
  - name: remd
  - name: at_300
    runs: false
    reservoir:
      generate: {temperature_K: 300.0, duration_ns: 8.0,
                 frame_interval_steps: 1000, equilibration_ns: 0.01,
                 platform: Reference,
                 bias_torsions: [{atoms: phi, energy: "-k*sin(theta)^2",
                                  parameters: {k: 70.0}}]}
reference: {kind: exact}
analysis: {threshold_tv: 0.05, points: 5}
slurm: {gres: null}
"""


def test_biased_reservoirs_give_the_exact_torsion_populations(tmp_path):
    """A named torsion bias, reservoir-only methods and a reference from
    reservoirs, on a model whose answer is exact."""
    from resremd import testsystems

    (tmp_path / "s.yml").write_text(TORSION_SPEC)
    out = tmp_path / "bench"
    assert main(["plan", str(tmp_path / "s.yml"), str(out)]) == 0
    runs = (out / "jobs/runs.txt").read_text().split()
    assert runs and not any("at_300" in r for r in runs)
    jobs.run_all(out)
    meta = json.loads((out / "reservoirs/at_300/trans/seed_1/reservoir.json")
                      .read_text())
    assert meta["kind"] == "weighted"
    assert meta["source"]["bias_torsions"][0]["atoms"] == [0, 1, 2, 3]
    assert main(["reference", str(out), str(tmp_path / "ref.json"),
                 "--methods", "at_300", "--from-reservoirs"]) == 0
    ref = json.loads((tmp_path / "ref.json").read_text())
    assert ref["from_reservoirs"] and ref["runs"] == 4
    assert abs(ref["populations"][0] - testsystems.cis_fraction(300.0)) \
        < 0.03
    summary = analyze.analyze(out)
    assert set(summary["methods"]) == {"remd"}


AUTO_SPEC = """
name: auto_test
system: {name: torsion_model}
temperatures_K: [300.0, 400.0]
seeds: [1]
starts: {trans: {}}
run: {production_steps: 50000, exchange_interval_steps: 250,
      trajectory_interval_steps: 250, equilibration_ns: 0.0,
      minimize: false, save_selection: all, platform: Reference}
methods:
  - name: remd
  - name: res_auto
    reservoir:
      generate: {temperature_K: 520.0, duration_ns: 8.0,
                 frame_interval_steps: 500, equilibration_ns: 0.01,
                 platform: Reference,
                 bias_torsions: [{atoms: phi, energy: "-k*sin(theta)^2",
                                  parameters: {k: 70.0}}],
                 convergence_torsions: [phi], convergence_tv: 0.3}
reference: {kind: exact}
analysis: {threshold_tv: 0.05, points: 5}
slurm: {gres: null}
"""


def test_a_self_stopping_reservoir_still_spends_the_budget(tmp_path):
    (tmp_path / "s.yml").write_text(AUTO_SPEC)
    out = tmp_path / "bench"
    assert main(["plan", str(tmp_path / "s.yml"), str(out)]) == 0
    jobs.run_all(out)
    meta = json.loads((out / "reservoirs/res_auto/trans/seed_1/"
                       "reservoir.json").read_text())
    assert meta["convergence"]["converged"]
    assert meta["cost"]["md_steps_total"] < 4_000_000 / 2
    summary = analyze.analyze(out)
    auto, remd = summary["methods"]["res_auto"], summary["methods"]["remd"]
    assert auto["reservoir_md_steps"] == meta["cost"]["md_steps_total"]
    assert abs(auto["cost_per_run_md_steps"]
               - remd["cost_per_run_md_steps"]) <= 2 * 250


def _plots_run(out, tmp_path):
    """Both figure scripts run on a finished benchmark (when matplotlib is
    there)."""
    import importlib.util
    import subprocess
    import sys

    if importlib.util.find_spec("matplotlib") is None:
        return
    plots = HERE / "plots"
    for cmd in (["plot_benchmark.py", str(out), "--format", "png"],
                ["plot_paper.py", "--tier1", str(out), "--budget", str(out),
                 "--out", str(tmp_path / "paper"), "--format", "png"]):
        subprocess.run([sys.executable, str(plots / cmd[0]), *cmd[1:]],
                       check=True, capture_output=True, text=True)
    assert (out / "analysis/figures/convergence.png").exists()
    assert (tmp_path / "paper/fig2_tier1_diagnostics.png").exists()
    assert (tmp_path / "paper/diagnostics_tier1.csv").exists()


REST2_SPEC = """
name: rest2_test
system: {name: torsion_model}
temperatures_K: [300.0, 900.0, 3000.0]
seeds: [1]
starts: {trans: {}}
run: {production_steps: 25000, exchange_interval_steps: 250,
      trajectory_interval_steps: 250, equilibration_ns: 0.0,
      minimize: false, save_selection: all, platform: Reference,
      rest2: true, rest2_selection: all}
methods:
  - name: rest2
  - name: rest2_res
    reservoir:
      generate: {temperature_K: 3000.0, rest2_run_temperature_K: 300.0,
                 rest2_selection: all, duration_ns: 0.5,
                 frame_interval_steps: 250, equilibration_ns: 0.01,
                 platform: Reference}
reference: {kind: exact}
analysis: {threshold_tv: 0.05, points: 5, mbar: true}
slurm: {gres: null}
"""


def test_rest2_methods_run_and_analyse(tmp_path):
    (tmp_path / "s.yml").write_text(REST2_SPEC)
    out = tmp_path / "bench"
    assert main(["plan", str(tmp_path / "s.yml"), str(out)]) == 0
    jobs.run_all(out)
    summary = analyze.analyze(out)
    res = summary["methods"]["rest2_res"]
    assert res["reservoir_acceptance"] == 1.0       # its own top state
    assert res["coverage_z_max_abs"] is not None
    assert summary["methods"]["rest2"]["starts"]["trans"][
        "final_tv_error_mbar_mean"] is not None
    man = json.loads((out / "runs/rest2/trans/seed_1/manifest.json")
                     .read_text())
    assert man["rest2"]["scales"][-1] < 0.33
