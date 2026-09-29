"""Figures for one analysed benchmark.

    python plot_benchmark.py <benchmark-dir> [--format pdf png]

Reads <benchmark-dir>/analysis/ (from `resbench analyze`) and writes, to
<benchmark-dir>/analysis/figures/:

    convergence.*       population error against cost, one panel per start:
                        median over seeds, band from lowest to highest seed
    convergence_mbar.*  the same, estimated from every temperature by MBAR
                        (when the analysis ran with `mbar: true`)
    cost_to_converge.*  cost until both starts have converged, per method:
                        median and 95% bootstrap interval, every seed shown
    populations.*       final populations per method against the reference
    agreement.*         distance between runs from opposite starts
    reservoir_budget.*  cost to converge, and the distance between the
                        reservoir's halves, against what the reservoir cost
                        (when reservoir methods differ in cost)

Methods without a reservoir are baselines and are drawn in neutral ink
(solid, dashed, dotted). Reservoir methods take the categorical colours in
the order the spec lists them, with a marker each, so identity never rests
on colour alone and a method keeps its colour and marker in every figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

#: Categorical slots in fixed order, validated for colour-vision deficiency
#: on adjacent pairs. Three are light on white, so every figure also carries
#: a legend and markers.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
           "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
BASELINE_DASHES = ["-", "--", ":"]
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def style() -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 1.5,
        "savefig.bbox": "tight", "figure.dpi": 150,
    })


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def num(x: str) -> float:
    return float(x) if x not in ("", None) else np.nan


def method_styles(summary) -> dict[str, dict]:
    styles, colour, base = {}, 0, 0
    for name, m in summary["methods"].items():
        if m.get("reservoir") is None:
            if base >= len(BASELINE_DASHES):
                raise SystemExit("More than three baselines; plot fewer.")
            styles[name] = {"color": INK, "ls": BASELINE_DASHES[base],
                            "marker": "|", "baseline": True}
            base += 1
        else:
            if colour >= len(PALETTE):
                raise SystemExit(f"More than {len(PALETTE)} reservoir "
                                 "methods; plot fewer at once.")
            styles[name] = {"color": PALETTE[colour], "ls": "-",
                            "marker": MARKERS[colour], "baseline": False}
            colour += 1
    return styles


def convergence(curves, summary, styles, out, fmt, column="tv_error",
                stem="convergence", what="convergence"):
    starts = sorted({r["start"] for r in curves},
                    key=lambda s: [r["start"] for r in curves].index(s))
    fig, axes = plt.subplots(1, len(starts), figsize=(4.2 * len(starts), 3.8),
                             sharey=True, squeeze=False)
    runs = defaultdict(list)
    for r in curves:
        runs[(r["method"], r["start"], r["seed"])].append(
            (num(r["cost_md_steps"]), num(r.get(column, ""))))
    all_cost = [c for pts in runs.values() for c, _ in pts if c > 0]
    grid = np.logspace(np.log10(min(all_cost)), np.log10(max(all_cost)), 60)
    threshold = summary["thresholds"]["tv"]
    top = threshold
    for ax, start in zip(axes[0], starts):
        for name, st in styles.items():
            series = []
            for (m, s, _), pts in runs.items():
                if m != name or s != start:
                    continue
                pts = sorted(pts)
                c = np.log10([p[0] for p in pts])
                e = np.array([p[1] for p in pts])
                series.append(np.interp(np.log10(grid), c, e, left=np.nan,
                                        right=np.nan))
            if not series:
                continue
            arr = np.array(series)
            # Over the seeds present at each cost, where at least half are:
            # runs whose reservoirs stopped themselves start and end at
            # different costs.
            ok = np.isfinite(arr).sum(axis=0) >= math.ceil(len(arr) / 2)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.where(ok, np.nanmedian(arr, axis=0), np.nan)
                lo = np.where(ok, np.nanmin(arr, axis=0), np.nan)
                hi = np.where(ok, np.nanmax(arr, axis=0), np.nan)
            if not st["baseline"]:
                ax.fill_between(grid, lo, hi, color=st["color"], alpha=0.12,
                                lw=0)
            ax.plot(grid, np.maximum(med, 1e-4), color=st["color"],
                    ls=st["ls"], label=name)
            if np.isfinite(hi).any():
                top = max(top, float(np.nanmax(hi)))
            idx = np.flatnonzero(np.isfinite(med))
            if idx.size and not st["baseline"]:
                ax.plot(grid[idx[-1]], max(med[idx[-1]], 1e-4), st["marker"],
                        color=st["color"], ms=5)
        ax.axhline(threshold, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"started {start}", loc="left")
        ax.set_xlabel("MD steps over all replicas, reservoir included")
    # Set once, for both panels: a limit set per panel stops the shared
    # axis from growing to fit the next panel's data.
    axes[0][0].set_ylim(threshold / 20, top * 1.5)
    axes[0][0].set_ylabel("population error (TV distance)")
    axes[0][-1].legend(fontsize=7.5, loc="best")
    fig.suptitle(f"{summary['benchmark']}: {what} at "
                 f"{summary['temperature_K']:g} K (median over the seeds "
                 "present, where at least half are; "
                 f"dashed line: threshold {threshold:g})",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, out / stem, fmt)


def cost_to_converge(summary, styles, out, fmt, stem="cost_to_converge"):
    names = list(styles)
    fig, ax = plt.subplots(figsize=(6.0, 0.42 * len(names) + 1.2))
    y = np.arange(len(names))[::-1]
    xs = []
    for yi, name in zip(y, names):
        m = summary["methods"][name]
        c = m.get("both_starts_converged_tv_cost_md_steps") or \
            next(iter(m["starts"].values()))["converged_tv_cost_md_steps"]
        st = styles[name]
        limit = c["limit"]
        xs.append(limit)
        for v in c.get("values") or []:
            x = limit if v is None else v
            ax.plot(x, yi, "|", color=MUTED, ms=8, mew=1)
        done = round((c.get("converged_fraction") or 0.0) * c["repeats"])
        note = f"{done}/{c['repeats']} converged"
        if c["median"] is None:
            ax.plot(limit, yi, st["marker"] if not st["baseline"] else "o",
                    mfc="none", color=st["color"], ms=7)
            ax.text(limit, yi + 0.25, "median beyond the budget; " + note,
                    color=MUTED, fontsize=7.5, ha="right")
            continue
        lo = c["low"] if c["low"] is not None else c["median"]
        hi = c["high"] if c["high"] is not None else limit
        xs += [lo, hi]
        ax.plot([lo, hi], [yi, yi], color=st["color"], lw=2, ls=st["ls"])
        ax.plot(c["median"], yi, st["marker"] if not st["baseline"] else "o",
                color=st["color"], ms=7)
        ax.text(c["median"], yi + 0.25, note, color=MUTED,
                fontsize=7.5, ha="center")
    ax.set_yticks(y, names)
    ax.set_xscale("log")
    ax.set_xlim(min(xs) / 1.6, max(xs) * 1.6)
    ax.set_ylim(y.min() - 0.6, y.max() + 0.8)
    ax.set_xlabel("MD steps until both starts converged, reservoir included")
    ax.set_title("Cost to converge from either side\nmedian, 95% bootstrap "
                 "interval over seeds, ticks: each seed; open: not "
                 "converged within the budget", loc="left", fontsize=9)
    ax.grid(axis="y", visible=False)
    save(fig, out / stem, fmt)


def populations(summary, styles, out, fmt, stem="populations"):
    states = summary["states"]
    ref = summary["reference"]["populations"]
    names = list(styles)
    fig, axes = plt.subplots(1, len(states), figsize=(2.2 * len(states) + 1.4,
                                                      0.42 * len(names) + 1.4),
                             sharey=True, squeeze=False)
    y = np.arange(len(names))[::-1]
    for j, (ax, state) in enumerate(zip(axes[0], states)):
        ax.axvline(ref[j], color=MUTED, lw=1, ls=(0, (4, 3)))
        for yi, name in zip(y, names):
            m = summary["methods"][name]
            st = styles[name]
            for k, (start, s) in enumerate(m["starts"].items()):
                mean = s["final_populations_mean"][j]
                se = (s["final_populations_stderr"] or [0] * len(states))[j]
                ax.errorbar(mean, yi + (0.15 if k == 0 else -0.15),
                            xerr=1.96 * se,
                            fmt=st["marker"] if not st["baseline"] else "o",
                            color=st["color"], ms=5, lw=1.2,
                            mfc=st["color"] if k == 0 else "white")
        ax.set_title(state, loc="left")
        ax.set_xlabel("population")
        ax.grid(axis="y", visible=False)
    axes[0][0].set_yticks(y, names)
    first, second = (list(next(iter(summary["methods"].values()))["starts"])
                     + ["", ""])[:2]
    fig.suptitle(f"Final populations at {summary['temperature_K']:g} K\n"
                 f"filled: started {first}; open: started {second}; "
                 f"bars: 95% interval over seeds; dashed: "
                 f"{summary['reference']['kind']} reference",
                 x=0.01, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, out / stem, fmt)


def agreement(rows, summary, styles, out, fmt, stem="agreement"):
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["method"]][num(r["time_ns"])].append(num(r["tv_between_starts"]))
    for name, st in styles.items():
        if name not in by:
            continue
        t = np.array(sorted(by[name]))
        med = np.array([np.median(by[name][x]) for x in t])
        ax.plot(t, np.maximum(med, 1e-4), color=st["color"], ls=st["ls"],
                label=name)
        if not st["baseline"]:
            ax.plot(t[-1], max(med[-1], 1e-4), st["marker"],
                    color=st["color"], ms=5)
    ax.axhline(summary["thresholds"]["tv"], color=MUTED, lw=1,
               ls=(0, (4, 3)))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("simulated time per replica (ns)")
    ax.set_ylabel("distance between starts (TV)")
    ax.set_title("Runs from opposite starts, same seed (median)\n"
                 "agreement shows the starts were forgotten, not that the "
                 "answer is right", loc="left", fontsize=9)
    ax.legend(fontsize=7.5, loc="best")
    save(fig, out / stem, fmt)


def reservoir_budget(res_rows, summary, styles, out, fmt,
                     stem="reservoir_budget"):
    """Cost to converge and the reservoir's halves distance, against the
    reservoir's own cost: how much of a budget the reservoir should take."""
    names = [n for n, st in styles.items() if not st["baseline"]]
    costs = defaultdict(list)
    halves = defaultdict(list)
    for r in res_rows:
        if r["method"] in names:
            costs[r["method"]].append(num(r["md_steps"]) / 1e6)
            if r.get("halves_tv") not in ("", None):
                halves[r["method"]].append(num(r["halves_tv"]))
    names = [n for n in names if costs[n]]
    if len({round(np.mean(costs[n]), 3) for n in names}) < 2:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.0))
    threshold = summary["thresholds"]["tv"]
    handles = []
    for name in names:
        st = styles[name]
        c = np.array(costs[name])
        x, xlo, xhi = np.mean(c), np.min(c), np.max(c)
        m = summary["methods"][name]
        conv = m.get("both_starts_converged_tv_cost_md_steps") or \
            next(iter(m["starts"].values()))["converged_tv_cost_md_steps"]
        if conv["median"] is None:
            y = conv["limit"] / 1e6
            (h,) = ax1.plot(x, y, st["marker"], mfc="white", color=st["color"],
                            ms=7, label=f"{name} (not converged)")
        else:
            y = conv["median"] / 1e6
            lo = (conv["low"] if conv["low"] is not None
                  else conv["median"]) / 1e6
            hi = (conv["high"] if conv["high"] is not None
                  else conv["limit"]) / 1e6
            ax1.plot([x, x], [lo, hi], color=st["color"], lw=2)
            (h,) = ax1.plot(x, y, st["marker"], color=st["color"], ms=7,
                            label=name)
        handles.append(h)
        if xhi > xlo:            # a reservoir that stopped itself varies
            ax1.plot([xlo, xhi], [y, y], color=st["color"], lw=1)
        if halves[name]:
            hv = np.array(halves[name])
            ax2.plot(np.full(len(hv), x), hv, "_", color=MUTED, ms=8)
            ax2.plot(x, np.mean(hv), st["marker"], color=st["color"], ms=7)
    lim = [0, max(max(costs[n]) for n in names) * 1.1]
    ax1.plot(lim, lim, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax1.text(lim[1] * 0.97, lim[1] * 0.84, "reservoir cost alone",
             color=MUTED, fontsize=7.5, ha="right", va="top")
    for ax in (ax1, ax2):
        ax.set_xlim(*lim)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("MD steps spent on the reservoir (millions)")
    ax1.set_ylabel("MD steps until both starts converged (millions)")
    ax1.set_title("Cost to converge\nmedian, 95% interval over seeds; "
                  "open: not converged", loc="left", fontsize=9)
    ax2.axhline(threshold, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax2.set_ylabel("TV between the reservoir's halves")
    ax2.set_title("The reservoir's own convergence\nmean; ticks: each "
                  f"reservoir; dashed: threshold {threshold:g}", loc="left",
                  fontsize=9)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=7.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, out / stem, fmt)


def save(fig, stem: Path, fmt: list[str]) -> None:
    for f in fmt:
        fig.savefig(f"{stem}.{f}")
    plt.close(fig)
    print(f"wrote {stem}.{{{','.join(fmt)}}}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("benchmark")
    p.add_argument("--format", nargs="+", default=["pdf", "png"])
    args = p.parse_args()
    a = Path(args.benchmark) / "analysis"
    summary = json.loads((a / "summary.json").read_text())
    styles = method_styles(summary)
    out = a / "figures"
    out.mkdir(exist_ok=True)
    style()
    convergence(read_csv(a / "curves.csv"), summary, styles, out, args.format)
    cost_to_converge(summary, styles, out, args.format)
    populations(summary, styles, out, args.format)
    agreement(read_csv(a / "agreement.csv"), summary, styles, out,
              args.format)
    curves = read_csv(a / "curves.csv")
    if summary.get("mbar") and any(r.get("tv_error_mbar") for r in curves):
        convergence(curves, summary, styles, out, args.format,
                    column="tv_error_mbar", stem="convergence_mbar",
                    what="convergence by MBAR over every temperature")
    reservoir_budget(read_csv(a / "reservoirs.csv"), summary, styles, out,
                     args.format)


if __name__ == "__main__":
    main()
