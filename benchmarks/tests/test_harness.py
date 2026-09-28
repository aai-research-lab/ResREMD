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
    for start in ("right", "left"):
        assert exact["starts"][start]["final_tv_error_mean"] < 0.05
        assert len(exact["starts"][start]["converged_tv_ns"]["values"]) == 2
    assert exact["reservoir_check_z_max_abs"] < 4
    # Each run has its own reservoir, made from its own start.
    assert (out / "reservoirs/resremd_generated/left/seed_1/reservoir.json"
            ).exists()
    for name in ("curves.csv", "convergence.csv", "agreement.csv",
                 "reservoirs.csv", "summary.json"):
        assert (out / "analysis" / name).stat().st_size > 0
    json.loads((out / "analysis/summary.json").read_text())  # strict JSON
    # Running again redoes nothing.
    before = json.loads((out / "runs/remd/right/seed_1/manifest.json")
                        .read_text())["updated"]
    jobs.run_all(out)
    after = json.loads((out / "runs/remd/right/seed_1/manifest.json")
                       .read_text())["updated"]
    assert before == after
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
