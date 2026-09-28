"""resbench {plan,run,job,status,analyze,cost} ..."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import analyze as analysis
from . import jobs


def estimate(out: Path, ns_per_day: float | None) -> str:
    """What the plan costs, per method and in total."""
    from . import spec as specs

    spec = jobs.load_out(out)
    lines = []
    total_steps = 0.0
    total_ns = 0.0
    for m in spec["methods"]:
        p = spec["_plan"][m["name"]]
        n_runs = len(specs.method_starts(spec, m)) * len(spec["seeds"])
        reservoirs = 0
        if m.get("reservoir"):
            reservoirs = 1 if m["reservoir"].get("shared") else n_runs
        steps = n_runs * (p["total_steps"] - p["reservoir_steps"]) \
            + reservoirs * p["reservoir_steps"]
        # Reservoirs may use their own timestep; the plan's steps are what
        # count for cost, and nanoseconds are shown at the run's timestep.
        ns = steps * p["timestep_fs"] / 1e6
        total_steps += steps
        total_ns += ns
        lines.append(
            f"{m['name']:<22}{n_runs:>4} runs x {p['replicas']:>2} replicas"
            f"  {p['production_steps'] * p['timestep_fs'] / 1e6:>8.2f} ns "
            f"each  {p['total_steps'] / 1e6:>9.2f}e6 steps per run"
            f"  {reservoirs:>3} reservoirs")
    lines.append(f"{'total':<22}{total_steps / 1e6:>10.1f}e6 MD steps, "
                 f"about {total_ns:.1f} ns of dynamics")
    if spec["equal_cost"]:
        lines.append("equal cost: production lengthened for the cheaper "
                     "methods so every run costs the same")
    if ns_per_day:
        lines.append(f"at {ns_per_day:g} ns/day on one GPU: "
                     f"{total_ns / ns_per_day:.1f} GPU-days")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="resbench")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("plan", help="Write the job lists and SLURM scripts.")
    a.add_argument("spec")
    a.add_argument("out")
    a = sub.add_parser("run", help="Run every job here, in order.")
    a.add_argument("out")
    a = sub.add_parser("job", help="Run one job (what SLURM calls).")
    a.add_argument("out")
    a.add_argument("job")
    a = sub.add_parser("status", help="What is done.")
    a.add_argument("out")
    a = sub.add_parser("analyze", help="Tables in <out>/analysis.")
    a.add_argument("out")
    a = sub.add_parser("reference", help="Write a reference file from a "
                                         "benchmark of long runs.")
    a.add_argument("out")
    a.add_argument("file")
    a.add_argument("--methods", nargs="+")
    a.add_argument("--last-fraction", type=float, default=0.5)
    a = sub.add_parser("cost", help="How much dynamics the plan needs.")
    a.add_argument("out")
    a.add_argument("--ns-per-day", type=float,
                   help="Measured speed of one replica on one GPU.")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("resremd",):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    out = Path(args.out)
    if args.cmd == "plan":
        ids = jobs.plan(Path(args.spec), out)
        print(json.dumps({k: len(v) for k, v in ids.items()}))
        print(estimate(out, None))
        print(f"Run here: resbench run {out}")
        print(f"On SLURM: {out}/slurm/submit.sh")
    elif args.cmd == "run":
        jobs.run_all(out)
    elif args.cmd == "job":
        jobs.run_job(out, args.job)
    elif args.cmd == "status":
        for job, state in jobs.status(out):
            print(f"{job:<50}{state}")
    elif args.cmd == "analyze":
        print(analysis.format_summary(analysis.analyze(out)))
    elif args.cmd == "reference":
        data = analysis.write_reference(out, Path(args.file), args.methods,
                                        args.last_fraction)
        print(f"{args.file}: {data['runs']} runs, populations "
              + ", ".join(f"{s} {p:.4f} +- {e:.4f}" for s, p, e in zip(
                  data["states"], data["populations"], data["stderr"])))
    elif args.cmd == "cost":
        print(estimate(out, args.ns_per_day))
    return 0


if __name__ == "__main__":
    sys.exit(main())
