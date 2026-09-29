"""The `resremd` command.

Every flag is generated from :mod:`resremd.options`, so the command line
cannot offer a setting the Python API lacks, or the other way round.

    resremd reservoir generate --prepared setup --temperature-K 500 \\
        --duration-ns 200 --output reservoir
    resremd run --prepared setup --reservoir reservoir \\
        --temperature-min-K 300 --n-replicas 8 --duration-ns 100 --output remd
    resremd summary remd
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from pathlib import Path
from typing import Any

from . import __version__
from .errors import ResRemdError
from .options import GENERATE, IMPORT, RUN, Option, Schema

logger = logging.getLogger("resremd")


def flag(option: Option) -> str:
    return "--" + option.name.replace("_", "-")


def _add_options(parser: argparse.ArgumentParser, schema: Schema) -> None:
    parser.add_argument("--config", metavar="FILE",
                        help="Settings file (.json, or .yml/.yaml with "
                             "PyYAML). Flags given here override it.")
    for group_name, options in schema.groups():
        group = parser.add_argument_group(group_name)
        for option in options:
            kwargs: dict[str, Any] = {"dest": option.name,
                                      "default": argparse.SUPPRESS,
                                      "help": option.help.replace("%", "%%")}
            if option.type is bool:
                kwargs["action"] = argparse.BooleanOptionalAction
            elif option.type is list or (isinstance(option.type, tuple)
                                         and list in option.type):
                kwargs["nargs"] = "+"
                kwargs["metavar"] = "VALUE"
                # Mixed settings such as save_states take words too; they are
                # converted after parsing.
                if option.type is list:
                    # A mapping per item arrives as JSON text.
                    kwargs["type"] = json.loads if option.items is dict \
                        else (option.items or str)
            else:
                kwargs["type"] = option.type
                if option.choices:
                    kwargs["choices"] = option.choices
                kwargs["metavar"] = option.type.__name__.upper()
            group.add_argument(flag(option), **kwargs)


def _read_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    text = p.read_text()
    if p.suffix in (".yml", ".yaml"):
        from .errors import require

        yaml = require("yaml", "Reading a YAML settings file", "yaml")
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ResRemdError(f"{path} must hold a mapping of settings.",
                           code="resremd.input.config")
    return data


def _settings(args: argparse.Namespace, schema: Schema) -> dict[str, Any]:
    given = _read_config(getattr(args, "config", None))
    for option in schema.options:
        if option.name in vars(args):
            value = getattr(args, option.name)
            if isinstance(option.type, tuple) and isinstance(value, list):
                if len(value) == 1 and not value[0].lstrip("-").isdigit():
                    value = value[0]
                elif all(v.lstrip("-").isdigit() for v in value):
                    value = [int(v) for v in value]
                else:
                    raise ResRemdError(
                        f"`{flag(option)}` takes one word or a list of "
                        f"integers, not {' '.join(value)!r}.",
                        code="resremd.input.type")
            given[option.name] = value
    return given


def template(schema: Schema) -> str:
    """A settings file with every option, its help, and its default."""
    lines = [f"# resremd {schema.name}: {schema.description}", ""]
    for group_name, options in schema.groups():
        lines.append(f"# --- {group_name} " + "-" * max(3, 60 - len(group_name)))
        for option in options:
            for text in textwrap.wrap(option.help, 76):
                lines.append(f"# {text}")
            if option.choices:
                lines.append(f"# One of: {', '.join(option.choices)}")
            value = option.default if option.default is not None \
                else option.example
            rendered = json.dumps(value)
            if option.default is None:
                lines.append(f"# {option.name}: {rendered}")
            else:
                lines.append(f"{option.name}: {rendered}")
            lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="resremd",
        description="Reservoir replica exchange molecular dynamics for "
                    "OpenMM.")
    parser.add_argument("--version", action="version",
                        version=f"resremd {__version__}")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Only warnings and errors.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help=RUN.description,
                           description=RUN.description)
    _add_options(p_run, RUN)

    p_res = sub.add_parser("reservoir", help="Make a reservoir.")
    res_sub = p_res.add_subparsers(dest="action", required=True)
    p_gen = res_sub.add_parser("generate", help=GENERATE.description,
                               description=GENERATE.description)
    _add_options(p_gen, GENERATE)
    p_imp = res_sub.add_parser("import", help=IMPORT.description,
                               description=IMPORT.description)
    _add_options(p_imp, IMPORT)

    p_sum = sub.add_parser("summary", help="Exchange statistics of a run.")
    p_sum.add_argument("run_dir")
    p_sum.add_argument("--json", action="store_true",
                       help="Print the summary as JSON.")

    p_opt = sub.add_parser("options",
                           help="Print a settings file with every option.")
    p_opt.add_argument("which", nargs="?", default="run",
                       choices=("run", "generate", "import"))

    p_lad = sub.add_parser("ladder", help="Print a geometric temperature "
                                          "ladder.")
    p_lad.add_argument("--temperature-min-K", type=float, required=True)
    p_lad.add_argument("--temperature-max-K", type=float, required=True)
    p_lad.add_argument("--n-replicas", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        if args.command == "run":
            from .sampler import run

            run(**_settings(args, RUN))
        elif args.command == "reservoir":
            from .build import generate, import_trajectories

            if args.action == "generate":
                generate(**_settings(args, GENERATE))
            else:
                import_trajectories(**_settings(args, IMPORT))
        elif args.command == "summary":
            from .analysis import format_summary, summarize

            summary = summarize(args.run_dir)
            print(json.dumps(summary, indent=2) if args.json
                  else format_summary(summary))
        elif args.command == "options":
            schema = {"run": RUN, "generate": GENERATE,
                      "import": IMPORT}[args.which]
            print(template(schema))
        elif args.command == "ladder":
            from .ladder import geometric

            for t in geometric(args.temperature_min_K, args.temperature_max_K,
                               args.n_replicas):
                print(f"{t:.2f}")
    except ResRemdError as exc:
        logger.error("%s [%s]", exc, exc.code)
        return 2
    except KeyboardInterrupt:
        logger.error("Interrupted.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
