import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re

import pandas as pd

from .connectivity import DATASET, SERVER, ValidationError
from .extension_analysis import markdown, validate_saved_census
from .inputs import save_json, sha256
from .vdeltae_inputs import verify as verify_vdeltae_census

AUDIT_KIND = "vdeltae_hdeltag_annotation_audit_v1"
AUDIT_FILES = ("body_annotations.csv", "annotation_label_counts.csv", "summary.md")
FIELD_DEFINITIONS = {
    "annotation_column": "Literal numeric C token; not a shared spatial coordinate.",
    "instance_index": "Separate hDeltaG instance index; not a column or angle.",
    "annotation_parse_status": "Name syntax: matched, missing_instance, or unrecognized_instance.",
    "source_file": "Resolved annotation CSV path used to load this body.",
    "anatomical_mapping_status": "unreviewed: axonal/dendritic positions and cross-type alignment unresolved.",
    "body_count": "Number of selected bodies, not synaptic contacts or spatial coverage.",
}


def load_population(path, expected_type):
    path = Path(path)
    table = pd.read_csv(
        path,
        dtype={"bodyId": "Int64", "type": "string", "instance": "string"},
    )
    required = {"bodyId", "type", "instance"}
    if not required.issubset(table.columns):
        raise ValidationError("Neuron table needs bodyId, type and instance.")

    selected = table.loc[table["type"].eq(expected_type)].copy()
    if selected.empty:
        raise ValidationError(f"No exact-type {expected_type} neurons found.")
    if selected["bodyId"].isna().any() or selected["bodyId"].le(0).any():
        raise ValidationError("Selected body IDs must be present and positive.")
    if selected["bodyId"].duplicated().any():
        raise ValidationError("Duplicate body IDs in selected population.")

    return selected.sort_values("bodyId").reset_index(drop=True)


def parse_annotations(table):
    required = {"type", "instance"}
    if not required.issubset(table.columns):
        raise ValidationError("Annotation parsing needs type and instance.")

    patterns = {
        "vDeltaE": r"vDeltaE_C(?P<column>[0-9]+)",
        "hDeltaG": r"hDeltaG_(?P<index>[0-9]{2})_C(?P<column>[0-9]+)",
    }
    if not table["type"].isin(patterns).all():
        raise ValidationError("This parser supports only vDeltaE and hDeltaG.")

    derived = {"annotation_column", "instance_index", "annotation_parse_status"}
    if derived.intersection(table.columns):
        raise ValidationError("Derived annotation fields already exist.")

    columns, indices, statuses = [], [], []
    for neuron in table.itertuples(index=False):
        column, instance_index = pd.NA, pd.NA
        name = neuron.instance
        if pd.isna(name) or (isinstance(name, str) and not name.strip()):
            status = "missing_instance"
        else:
            match = (
                re.fullmatch(patterns[neuron.type], name)
                if isinstance(name, str)
                else None
            )
            if match is None:
                status = "unrecognized_instance"
            else:
                column = int(match.group("column"))
                if neuron.type == "hDeltaG":
                    instance_index = int(match.group("index"))
                status = "matched"
        columns.append(column)
        indices.append(instance_index)
        statuses.append(status)
    result = table.copy()
    result["annotation_column"] = pd.array(columns, dtype="Int64")
    result["instance_index"] = pd.array(indices, dtype="Int64")
    result["annotation_parse_status"] = statuses
    return result


def assemble_body_table(vdeltae_path, hdeltag_path):
    populations = []
    for path, population in [(vdeltae_path, "vDeltaE"), (hdeltag_path, "hDeltaG")]:
        table = parse_annotations(load_population(path, population))
        added = {"source_file", "anatomical_mapping_status"}
        if added.intersection(table.columns):
            raise ValidationError("Audit provenance or mapping fields already exist.")
        table["source_file"] = str(Path(path).resolve())
        table["anatomical_mapping_status"] = "unreviewed"
        populations.append(table)

    bodies = pd.concat(populations, ignore_index=True, sort=False)
    if bodies["bodyId"].duplicated().any():
        raise ValidationError("A body ID occurs in both selected populations.")
    return bodies.sort_values(["type", "bodyId"]).reset_index(drop=True)


def collect_source_provenance(vdeltae_directory, hdeltab_directory):
    sources = {}
    specifications = [
        ("vdeltae_census", vdeltae_directory, "vDeltaE_neurons.csv"),
        ("hdeltab_census", hdeltab_directory, "candidate_neurons.csv"),
    ]
    identity_fields = ("uuid", "latestMutationId", "lastDatabaseEdit")

    for role, directory, filename in specifications:
        directory = Path(directory).resolve()
        manifest_path = directory / "manifest.json"
        annotation_path = directory / filename
        before = {
            "manifest_sha256": sha256(manifest_path),
            "annotation_sha256": sha256(annotation_path),
        }

        if role == "vdeltae_census":
            manifest = verify_vdeltae_census(directory)
        else:
            manifest, _, _, _ = validate_saved_census(directory)

        if (manifest.get("server")) != SERVER or manifest.get("dataset") != DATASET:
            raise ValidationError(
                "Source server or dataset differs from the configured dataset."
            )
        metadata = manifest.get("dataset_metadata", {})
        if any(metadata.get(field) is None for field in identity_fields):
            raise ValidationError("Source dataset identity fields are missing.")
        if not metadata["uuid"]:
            raise ValidationError("Source dataset UUID is empty.")
        if manifest.get("data_sha256", {}).get(filename) != before["annotation_sha256"]:
            raise ValidationError(
                "Annotation CSV is not covered by its source manifest hash."
            )
        if (
            sha256(manifest_path) != before["manifest_sha256"]
            or sha256(annotation_path) != before["annotation_sha256"]
        ):
            raise ValidationError(
                "Source manifest or annotation CSV changed during verification."
            )

        sources[role] = {
            "directory": str(directory),
            "manifest_file": str(manifest_path),
            "annotation_file": str(annotation_path),
            **before,
            "server": manifest["server"],
            "dataset": manifest["dataset"],
            "dataset_identity": {field: metadata[field] for field in identity_fields},
        }

    if (
        sources["vdeltae_census"]["dataset_identity"]
        != sources["hdeltab_census"]["dataset_identity"]
    ):
        raise ValidationError("Source censuses describe different dataset states.")

    return sources


def annotation_label_counts(bodies):
    return (
        bodies.groupby(
            ["type", "annotation_parse_status", "annotation_column"], dropna=False
        )
        .size()
        .rename("body_count")
        .reset_index()
    )


def check_source_files_unchanged(sources):
    for source in sources.values():
        for file_key, hash_key in [
            ("manifest_file", "manifest_sha256"),
            ("annotation_file", "annotation_sha256"),
        ]:
            path = Path(source[file_key])
            if not path.is_file() or sha256(path) != source[hash_key]:
                raise ValidationError(
                    "Source manifest or annotation CSV changed after verification."
                )


def audit_summary(bodies, counts, sources):
    populations = bodies.groupby("type").size().rename("body_count").reset_index()
    lines = [
        "# vDeltaE/hDeltaG annotation inventory",
        "",
        f"Server: {SERVER}; dataset: {DATASET}.",
        "", "## Selected bodies",
        "",
        markdown(populations),
        "", "## Bodies per annotation label",
        "",
        markdown(counts),
        "",
        "## Sources",
        "",
    ]
    for role, source in sources.items():
        lines.append(f"- {role}: `{source['annotation_file']}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "All source annotation columns are retained. Missing or unrecognized names remain visible.",
            "Column tokens are annotation labels. Equal tokens across types do not establish spatial alignment.",
            "Anatomical mapping is unreviewed: no axonal/dendritic positions, angles or eight-column mapping assigned.",
            "No neural dynamics or new connectivity queries were performed.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_audit(vdeltae_directory, hdeltab_directory, output):
    vdeltae_directory = Path(vdeltae_directory).resolve()
    hdeltab_directory = Path(hdeltab_directory).resolve()
    output = Path(output).resolve()
    for source in [vdeltae_directory, hdeltab_directory]:
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValidationError(
                "Audit output must be separate from source census directories."
            )

    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "kind": AUDIT_KIND,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_sources": {
            "vdeltae_census": str(vdeltae_directory),
            "hdeltab_census": str(hdeltab_directory),
        },
        "field_definitions": FIELD_DEFINITIONS,
        "settings": {
            "selected_types": ["vDeltaE", "hDeltaG"],
            "membership": "Exact source type; no instance-based reassignment.",
            "anatomical_mapping": "unreviewed",
        },
    }
    save_json(output / "manifest.json", manifest)
    try:
        module_directory = Path(__file__).resolve().parent
        manifest["code_sha256"] = {
            path.name: sha256(path) for path in sorted(module_directory.glob("*.py"))
        }
        manifest["python_version"] = platform.python_version()
        manifest["packages"] = {
            name: version(name)
            for name in ["numpy", "pandas", "neuprint-python", "matplotlib"]
        }
        sources = collect_source_provenance(vdeltae_directory, hdeltab_directory)
        manifest["sources"] = sources
        bodies = assemble_body_table(
            sources["vdeltae_census"]["annotation_file"],
            sources["hdeltab_census"]["annotation_file"],
        )
        counts = annotation_label_counts(bodies)
        bodies.to_csv(output / "body_annotations.csv", index=False)
        counts.to_csv(output / "annotation_label_counts.csv", index=False)
        (output / "summary.md").write_text(audit_summary(bodies, counts, sources))

        check_source_files_unchanged(sources)
        manifest["population_counts"] = {
            name: int(count) for name, count in bodies.groupby("type").size().items()
        }
        manifest["parse_status_counts"] = {
            name: int(count)
            for name, count in bodies["annotation_parse_status"].value_counts().items()
        }
        manifest["data_sha256"] = {name: sha256(output / name) for name in AUDIT_FILES}
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error_type"] = type(error).__name__
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(output / "manifest.json", manifest)
    return manifest


def verify_audit(output):
    output = Path(output).resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("kind") != AUDIT_KIND or manifest.get("status") != "complete":
        raise ValidationError("Verification requires a complete annotation audit.")
    hashes = manifest.get("data_sha256", {})
    if set(hashes) != set(AUDIT_FILES):
        raise ValidationError(
            "Audit artifact hash inventory is incomplete or unexpected."
        )
    for name in AUDIT_FILES:
        path = (output / name).resolve()
        if (
            not path.is_relative_to(output)
            or not path.is_file()
            or sha256(path) != hashes[name]
        ):
            raise ValidationError(
                "Audit artifact is missing, outside the run, or changed."
            )

    recorded_sources = manifest.get("sources", {})
    if set(recorded_sources) != {"vdeltae_census", "hdeltab_census"}:
        raise ValidationError("Audit source provenance is incomplete or unexpected.")
    check_source_files_unchanged(recorded_sources)
    sources = collect_source_provenance(
        recorded_sources["vdeltae_census"]["directory"],
        recorded_sources["hdeltab_census"]["directory"],
    )
    if sources != recorded_sources:
        raise ValidationError("Audit source provenance differs from verified sources.")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Offline vDeltaE/hDeltaG annotation audit."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", help="Verify source and save a new annotation inventory."
    )
    run.add_argument("--vdeltae-census", required=True, type=Path)
    run.add_argument("--hdeltab-census", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    verify = commands.add_parser(
        "verify", help="Verify a completed annotation inventory."
    )
    verify.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        run_audit(args.vdeltae_census, args.hdeltab_census, args.output)
        print("Annotation audit complete:", args.output)
    else:
        verify_audit(args.output)
        print("Annotation audit verified:", args.output)


if __name__ == "__main__":
    main()
