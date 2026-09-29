"""The paper's figures, from analysed benchmarks.

    python plot_paper.py --tier1 bench_exact --tier1b bench_torsion \\
        --budget bench_budget --out paper_figures [--format pdf png]

Each argument is a benchmark directory already run through
`resbench analyze`; any of the three may be left out. Writes, to --out:

    fig1_tier1_cost.*          tier 1: cost until both starts converged
    fig2_tier1_diagnostics.*   tier 1: which diagnostic flags which reservoir
    fig3_tier1b_convergence.*  tier 1b: convergence from each start
    fig4_tier1b_diagnostics.*  tier 1b: the same diagnostic table
    fig5_budget.*              budget: cost to converge and the reservoir's
                               halves, against what the reservoir cost
    fig5_budget_mbar.*         budget: convergence by MBAR over every
                               temperature
    diagnostics_tier1.csv, diagnostics_tier1b.csv
                               the tables behind figures 2 and 4

Figures share the style, colours and markers of plot_benchmark.py, so a
method looks the same everywhere.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_benchmark as pb  # noqa: E402

from resbench.analyze import RESERVOIR_Z  # noqa: E402
from resremd.analysis import ENSEMBLE_Z  # noqa: E402

plt = pb.plt

#: Columns of the diagnostic table: key, heading, kind. Each flags where
#: the package or the harness itself flags: the temperature check at
#: ENSEMBLE_Z, reservoirs from opposite starts at RESERVOIR_Z, coverage for
#: a state missing from the reservoir, distances at the benchmark's TV
#: threshold, and runs from opposite starts when a seed's never agree.
COLUMNS = [
    ("reference_tv", "error against\nthe reference", "tv"),
    ("temperature_z", "temperature\ncheck |z|", "temperature"),
    ("halves_tv", "reservoir halves\n(mean TV)", "tv"),
    ("coverage_z", "coverage\n|z|", "coverage"),
    ("res_starts_z", "reservoirs from\nopposite starts |z|", "reservoirs"),
    ("runs_agree", "runs from opposite\nstarts agree", "agree"),
]

FLAG_FILL = "#fbe2d5"
PASS_FILL = "#f4f3ef"


def load(bench: Path) -> tuple[dict, dict[str, list[dict]]]:
    a = bench / "analysis"
    summary = json.loads((a / "summary.json").read_text())
    tables = {name: pb.read_csv(a / f"{name}.csv") for name in
              ("curves", "convergence", "reservoirs", "agreement",
               "reservoir_agreement")}
    return summary, tables


def diagnostics(summary, tables) -> list[dict]:
    """One row per method: each diagnostic's worst value over its runs or
    reservoirs, and whether it flags the method."""
    threshold = summary["thresholds"]["tv"]
    res_z = defaultdict(list)
    for r in tables["reservoir_agreement"]:
        if r.get("max_abs_z") not in ("", None):
            res_z[r["method"]].append(pb.num(r["max_abs_z"]))
    rows = []
    for name, m in summary["methods"].items():
        finals = [s["final_tv_error_mean"] for s in m["starts"].values()
                  if s.get("final_tv_error_mean") is not None]
        agreed = m.get("starts_agree_ns")
        row = {
            "method": name,
            "reservoir": m.get("reservoir") or "none",
            "reference_tv": max(finals) if finals else None,
            "temperature_z": m.get("reservoir_check_z_max_abs"),
            # The mean over the method's reservoirs, as in the run summary:
            # the largest of many noisy halves would flag sound reservoirs.
            "halves_tv": m.get("reservoir_halves_tv_mean"),
            "coverage_z": m.get("coverage_z_max_abs"),
            "coverage_missing": " ".join(
                m.get("coverage_unsupported_states") or []),
            "res_starts_z": max(res_z[name]) if res_z[name] else None,
            "runs_agree": None if agreed is None else
            agreed.get("converged_fraction"),
        }
        for key, _, kind in COLUMNS:
            v = row[key]
            if v is None:
                row[f"{key}_flag"] = None
            elif kind == "temperature":
                row[f"{key}_flag"] = v >= ENSEMBLE_Z
            elif kind == "coverage":
                row[f"{key}_flag"] = bool(row["coverage_missing"])
            elif kind == "reservoirs":
                row[f"{key}_flag"] = v > RESERVOIR_Z
            elif kind == "tv":
                row[f"{key}_flag"] = v >= threshold
            else:                       # fraction of seeds whose starts agree
                row[f"{key}_flag"] = v < 1.0
        rows.append(row)
    return rows


def write_table(rows, path: Path) -> None:
    if not rows:
        print(f"no methods with finished runs; {path.name} not written")
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: "" if v is None else v for k, v in r.items()})
    print(f"wrote {path}")


def _cell(row, key, kind) -> str:
    v = row[key]
    if v is None:
        return "n/a"
    if kind == "agree":
        text = f"{v:.0%} of seeds"
    elif kind != "tv":
        text = ">999" if v >= 999 else f"{v:.1f}"
    else:
        text = f"{v:.3f}"
    if key == "coverage_z" and row["coverage_missing"]:
        text += f"\nmissing: {row['coverage_missing']}"
    return ("flag  " + text) if row[f"{key}_flag"] else text


def diagnostic_table(rows, summary, out, stem, fmt) -> None:
    """The table as a figure: flagged cells filled and marked "flag", so the
    fill never carries the meaning alone."""
    if not rows:
        return
    n, k = len(rows), len(COLUMNS)
    fig, ax = plt.subplots(figsize=(1.55 * k + 1.8, 0.46 * n + 1.3))
    ax.set_xlim(0, k)
    ax.set_ylim(n, -0.9)
    ax.axis("off")
    for j, (_, head, _) in enumerate(COLUMNS):
        ax.text(j + 0.5, -0.45, head, ha="center", va="center", fontsize=8,
                color=pb.INK)
    for i, row in enumerate(rows):
        ax.text(-0.08, i + 0.5, row["method"], ha="right", va="center",
                fontsize=8.5, color=pb.INK)
        for j, (key, _, kind) in enumerate(COLUMNS):
            flag = row[f"{key}_flag"]
            ax.add_patch(plt.Rectangle(
                (j + 0.03, i + 0.06), 0.94, 0.88, lw=0,
                color=FLAG_FILL if flag else PASS_FILL))
            ax.text(j + 0.5, i + 0.5, _cell(row, key, kind), ha="center",
                    va="center", fontsize=7.5,
                    color=pb.INK if flag else pb.MUTED,
                    fontweight="bold" if flag else "normal")
    ax.set_title(
        f"{summary['benchmark']}: which check flags which reservoir\n"
        f"flag: error or halves >= {summary['thresholds']['tv']:g}; "
        f"temperature |z| >= {ENSEMBLE_Z:g}; a state missing from the "
        f"reservoir; reservoirs from opposite starts |z| > {RESERVOIR_Z:g};"
        "\na seed whose runs from opposite starts never agree within the "
        "budget. n/a: no reservoir, or too few frames for the check",
        loc="left", fontsize=8.5)
    fig.tight_layout()
    pb.save(fig, out / stem, fmt)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tier1", type=Path)
    p.add_argument("--tier1b", type=Path)
    p.add_argument("--budget", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--format", nargs="+", default=["pdf", "png"])
    args = p.parse_args()
    if not (args.tier1 or args.tier1b or args.budget):
        p.error("give at least one of --tier1, --tier1b, --budget")
    args.out.mkdir(parents=True, exist_ok=True)
    pb.style()
    fmt = args.format
    if args.tier1:
        summary, t = load(args.tier1)
        styles = pb.method_styles(summary)
        pb.cost_to_converge(summary, styles, args.out, fmt,
                            stem="fig1_tier1_cost")
        rows = diagnostics(summary, t)
        write_table(rows, args.out / "diagnostics_tier1.csv")
        diagnostic_table(rows, summary, args.out, "fig2_tier1_diagnostics",
                         fmt)
    if args.tier1b:
        summary, t = load(args.tier1b)
        styles = pb.method_styles(summary)
        pb.convergence(t["curves"], summary, styles, args.out, fmt,
                       stem="fig3_tier1b_convergence")
        rows = diagnostics(summary, t)
        write_table(rows, args.out / "diagnostics_tier1b.csv")
        diagnostic_table(rows, summary, args.out, "fig4_tier1b_diagnostics",
                         fmt)
    if args.budget:
        summary, t = load(args.budget)
        styles = pb.method_styles(summary)
        pb.reservoir_budget(t["reservoirs"], summary, styles, args.out, fmt,
                            stem="fig5_budget")
        if not any(args.out.glob("fig5_budget.*")):
            print("fig5_budget not drawn: it needs at least two reservoir "
                  "methods of different cost")
        if summary.get("mbar"):
            pb.convergence(t["curves"], summary, styles, args.out, fmt,
                           column="tv_error_mbar", stem="fig5_budget_mbar",
                           what="convergence by MBAR over every temperature")


if __name__ == "__main__":
    main()
