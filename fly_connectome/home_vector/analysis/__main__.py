"""Run reproducible offline diagnostics on explicitly selected saved bundles."""

import argparse
import importlib
import json

from .common import Audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("predictions", "perturbations", "statistics", "network"):
        sub = commands.add_parser(command)
        sub.add_argument("run", help="Completed training bundle (not a checkpoint alone)")
        sub.add_argument("--source", required=True, help="Matching prepared task bundle; may be relocated")
        sub.add_argument("--output", required=True, help="New, separate audit directory")
        if command == "perturbations":
            sub.add_argument("--family", default="challenge-pause-5",
                             help="Challenge family containing two opposite-history pause trajectories")
            sub.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == "perturbations" and args.threads < 1:
        parser.error("--threads must be positive")
    module = importlib.import_module(f"{__package__}.{args.command}")
    with Audit(args) as audit:
        module.analyze(audit)
    print(json.dumps(dict(output=str(audit.out), status="complete", audit=args.command)))


if __name__ == "__main__":
    main()
