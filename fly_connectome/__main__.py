"""Run with python -m fly_connectome extract or plot."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from dotenv import load_dotenv
from neuprint import Client, fetch_meta

from .connectivity import (
    DATASET, SERVER, ZERO_MEANING, REFERENCE_PRE, REFERENCE_POST,
    ValidationError, build_matrix, check_reference_annotations,
    check_reference_weight, check_type_inventory, fetch_annotations,
    fetch_connections, population_summary, reference_result, select_populations,
    type_inventory,
)
from .plotting import load_matrix, plot_heatmap
from .anatomy import anatomical_heatmaps
from .columns import column_analysis


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def code_revision():
    root = Path(__file__).resolve().parent.parent
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip())
        return {"commit": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def extract(output):
    # Explicit path prevents python-dotenv from searching unrelated parent directories.
    load_dotenv(Path.cwd() / ".env", override=False)
    token = os.environ.get("NEUPRINT_TOKEN", "").strip()
    if not token:
        raise ValidationError("Set NEUPRINT_TOKEN in the environment or the local .env file.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "server": SERVER, "dataset": DATASET, "status": "running",
        "selection": {"type": ["PFNd", "hDeltaB"], "regex": False,
                      "status_filter": None, "cropped_filter": None},
        "connectivity": {"weight_property": "weight", "min_total_weight": 1,
                         "omit_rois": True, "roi_filter": None, "threads": 1},
        "matrix": {"rows": "presynaptic body IDs", "columns": "postsynaptic body IDs",
                   "ordering": "ascending numeric body ID", "zero_meaning": ZERO_MEANING},
        "python": platform.python_version(),
        "packages": {name: version(name) for name in
                     ["neuprint-python", "pandas", "matplotlib", "numpy", "python-dotenv"]},
        "code": code_revision(),
    }
    save_json(output / "metadata.json", metadata)
    try:
        client = Client(SERVER, dataset=DATASET, token=token)
        metadata["dataset_metadata"] = fetch_meta(client=client)
        save_json(output / "metadata.json", metadata)
        inventory = type_inventory(client)
        inventory.to_csv(output / "type_inventory.csv", index=False)
        check_type_inventory(inventory)
        reference_neurons = fetch_annotations(client, [REFERENCE_PRE, REFERENCE_POST])
        reference_neurons.to_csv(output / "reference_neurons.csv", index=False)
        check_reference_annotations(reference_neurons)
        populations = select_populations(client)
        for label, neurons in populations.items():
            neurons.to_csv(output / f"{label}_neurons.csv", index=False)
        summary = population_summary(populations)
        save_json(output / "population_summary.json", summary)
        print("Selected populations (no reconstruction-status or cropped filters):")
        for label, details in summary.items():
            print(f"  {label}: types={details['matched_types']}, neurons={details['neuron_count']}")
            print("    " + "; ".join(
                f"{column}={counts}" for column, counts in details["reconstruction"].items()
            ))
        print(f"  PFNd web count sanity check (expected 40): "
              f"{'matched' if summary['PFNd']['web_count_sanity_check']['matches'] else 'DIFFERS'}")
        if not summary["PFNd"]["web_count_sanity_check"]["matches"]:
            print("WARNING: PFNd count differs from the web-interface sanity target of 40.", file=sys.stderr)
        reference_edges = fetch_connections(client, [REFERENCE_PRE], [REFERENCE_POST])
        reference_edges.to_csv(output / "reference_connections.csv", index=False)
        validation = reference_result(build_matrix(reference_edges, [REFERENCE_PRE], [REFERENCE_POST]))
        save_json(output / "reference_validation.json", validation)
        check_reference_weight(validation)
        print(f"Reference {REFERENCE_PRE} -> {REFERENCE_POST}: weight 137 verified.")
        for source, target in [("PFNd", "hDeltaB"), ("hDeltaB", "PFNd")]:
            edges = fetch_connections(client, populations[source]["bodyId"], populations[target]["bodyId"])
            stem = f"{source}_to_{target}"
            edges.to_csv(output / f"{stem}_connections.csv", index=False)
            matrix = build_matrix(edges, populations[source]["bodyId"], populations[target]["bodyId"])
            if source == "PFNd":
                # Detect a changed reference result between pair and population queries.
                validation = reference_result(matrix)
                save_json(output / "population_reference_validation.json", validation)
                check_reference_weight(validation)
            matrix.to_csv(output / f"{stem}_matrix.csv")
            plot_heatmap(matrix, output / stem, source, target)
        metadata["status"] = "complete"
    except ValidationError as exc:
        metadata.update(status="validation_failed", error=str(exc))
        raise
    except Exception as exc:
        # HTTP exception text could contain sensitive request data.
        metadata.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(output / "metadata.json", metadata)
    print(f"Saved data and heatmaps to {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="PFNd/hDeltaB MaleCNS raw connectivity analysis")
    commands = parser.add_subparsers(dest="command", required=True)
    extraction = commands.add_parser("extract", help="Validate, extract both directions, and plot")
    extraction.add_argument("--output", type=Path, required=True, help="New output directory")
    plotting = commands.add_parser("plot", help="Plot a saved matrix without a token or network")
    plotting.add_argument("matrix", type=Path)
    plotting.add_argument("--output-prefix", type=Path, required=True)
    plotting.add_argument("--source-type", choices=["PFNd", "hDeltaB"], required=True)
    plotting.add_argument("--target-type", choices=["PFNd", "hDeltaB"], required=True)
    anatomical = commands.add_parser("anatomical", help="Order a saved extraction by instance annotations, offline")
    anatomical.add_argument("source", type=Path, help="Complete extraction directory")
    anatomical.add_argument("--output", type=Path, required=True, help="New directory for additional artifacts")
    columns = commands.add_parser("columns", help="Aggregate PFNd → hDeltaB by annotated FB column, offline")
    columns.add_argument("source", type=Path, help="Complete extraction directory")
    columns.add_argument("--anatomical", type=Path, required=True, help="Complete anatomical ordering directory")
    columns.add_argument("--output", type=Path, required=True, help="New directory for column summaries")
    columns.add_argument("--show-column-caveat", action="store_true",
                         help="Show the annotation caveat above column heatmaps (hidden by default)")
    args = parser.parse_args(argv)
    try:
        if args.command == "extract":
            extract(args.output)
        elif args.command == "anatomical":
            result = anatomical_heatmaps(args.source, args.output)
            print(f"Saved anatomical tables and heatmaps to {args.output}")
            print(result["column_caveat"])
        elif args.command == "columns":
            result = column_analysis(args.source, args.anatomical, args.output,
                                     show_column_caveat=args.show_column_caveat)
            print(f"Saved column summaries and heatmaps to {args.output}")
            print(result["validation"])
            print(result["column_caveat"])
        else:
            plot_heatmap(load_matrix(args.matrix), args.output_prefix, args.source_type, args.target_type)
    except (ValidationError, FileExistsError) as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Stopped ({type(exc).__name__}). Check network access, dataset availability, "
              "local output paths, and credentials. No further queries will run.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
