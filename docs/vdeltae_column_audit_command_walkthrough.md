# Hand-written implementation: the annotation-audit command

Prepared 29 October 2026. Continue after Step 6 of the
[annotation walkthrough](vdeltae_column_audit_walkthrough.md). The code below is
for you to transcribe into `fly_connectome/vdeltae_column_audit.py`, one piece at
a time. It does not alter neural dynamics or assign anatomical coordinates.

This stage packages the existing collector and table assembler. It saves a new
run with three artifacts and a manifest, checks that inputs stayed unchanged,
and provides an offline integrity-verification command. Existing census runs,
draft tables, simulation outputs and presentation exports remain untouched.

## Step 7a: small helpers and shared definitions

Add these imports, consolidating existing imports rather than duplicating them:

```python
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import platform

from .extension_analysis import markdown, validate_saved_census
from .inputs import save_json, sha256
```

`save_json` is the existing repository JSON writer. `markdown` formats a small
DataFrame as a Markdown table, showing missing entries as `undefined`.

Add these module-level constants and helpers below your existing functions:

```python
AUDIT_KIND = "vdeltae_hdeltag_annotation_audit_v1"
AUDIT_FILES = (
    "body_annotations.csv", "annotation_label_counts.csv", "summary.md",
)
FIELD_DEFINITIONS = {
    "annotation_column": "Literal numeric C token; not a shared spatial coordinate.",
    "instance_index": "Separate hDeltaG instance index; not a column or angle.",
    "annotation_parse_status": "Name syntax: matched, missing_instance, or unrecognized_instance.",
    "source_file": "Resolved annotation CSV path used to load this body.",
    "anatomical_mapping_status": "unreviewed: axonal/dendritic positions and cross-type alignment unresolved.",
    "body_count": "Number of selected bodies, not synaptic contacts or spatial coverage.",
}


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
                raise ValidationError("Source manifest or annotation CSV changed after verification.")


def audit_summary(bodies, counts, sources):
    populations = bodies.groupby("type").size().rename("body_count").reset_index()
    lines = [
        "# vDeltaE/hDeltaG annotation inventory", "",
        f"Server: {SERVER}; dataset: {DATASET}.", "",
        "## Selected bodies", "", markdown(populations), "",
        "## Bodies per annotation label", "", markdown(counts), "",
        "## Sources", "",
    ]
    for role, source in sources.items():
        lines.append(f"- {role}: `{source['annotation_file']}`")
    lines.extend([
        "", "## Interpretation", "",
        "All source annotation columns are retained. Missing or unrecognized names remain visible.",
        "Column tokens are annotation labels. Equal tokens across types do not establish spatial alignment.",
        "Anatomical mapping is unreviewed: no axonal/dendritic positions, angles or eight-column mapping assigned.",
        "No neural dynamics or new connectivity queries were performed.",
    ])
    return "\n".join(lines) + "\n"
```

The counts helper packages the same grouping used for your preliminary tables.
`dropna=False` remains essential: unresolved names must not disappear from the
summary. The source recheck compares the files used by this audit, including
the manifests that cover the wider census artifacts. It does not re-query the
live database or continuously monitor source files.

## Step 7b: write a complete or failed audit run

```python
def run_audit(vdeltae_directory, hdeltab_directory, output):
    vdeltae_directory = Path(vdeltae_directory).resolve()
    hdeltab_directory = Path(hdeltab_directory).resolve()
    output = Path(output).resolve()
    for source in [vdeltae_directory, hdeltab_directory]:
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValidationError("Audit output must be separate from source census directories.")

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
            name: version(name) for name in ["numpy", "pandas", "neuprint-python", "matplotlib"]
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
            name: int(count) for name, count in bodies["annotation_parse_status"].value_counts().items()
        }
        manifest["data_sha256"] = {
            name: sha256(output / name) for name in AUDIT_FILES
        }
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error_type"] = type(error).__name__
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(output / "manifest.json", manifest)
    return manifest
```

Read this function as a sequence of boundaries:

1. Reject overlapping source/output directory trees, then require a new output
   directory. An existing output is never overwritten, including a failed run.
2. Write a `running` manifest, then collect code hashes, package versions and
   verified sources. Hashing the package's Python files records more than just
   the new module because the reused helpers also affect behavior. This records
   code identity, not a copy of the code; preserve it in version control.
3. Assemble and save the tables and readable summary using the verified paths.
4. Recheck source files, hash outputs, and declare the run `complete`.
5. If an ordinary exception occurs after initialization, retain partial files
   and write `failed` with the exception class before propagating the exception.
   Only successful runs return normally.

Errors before directory/manifest creation cannot leave a failed manifest. An
unwritable filesystem or forcibly terminated process can also prevent the final
manifest write. A run marked `running` is incomplete, and verification rejects
it. There is no promise of an atomic filesystem snapshot: the final recheck
detects ordinary changes across assembly, rather than locking the source files.

No neuron counts are hardcoded. A completed inventory can contain unrecognized
names, provided they are retained and flagged. Completion means the annotation
audit succeeded; it does not mean anatomical mapping has been resolved.

## Step 7c: verify a saved audit

```python
def verify_audit(output):
    output = Path(output).resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("kind") != AUDIT_KIND or manifest.get("status") != "complete":
        raise ValidationError("Verification requires a complete annotation audit.")
    hashes = manifest.get("data_sha256", {})
    if set(hashes) != set(AUDIT_FILES):
        raise ValidationError("Audit artifact hash inventory is incomplete or unexpected.")
    for name in AUDIT_FILES:
        path = (output / name).resolve()
        if (not path.is_relative_to(output) or not path.is_file()
                or sha256(path) != hashes[name]):
            raise ValidationError("Audit artifact is missing, outside the run, or changed.")

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
```

This is an integrity check against the saved manifest plus a rerun of the
underlying source verifiers. It detects changed/missing outputs and sources.
It requires the original source directories to remain available at their
recorded paths. It does not require today's Python files to match the recorded
code hashes: those identify the run's code, while the current verifier checks
the saved artifacts. It does not regenerate tables or prove the biological
interpretation. A new run is the way to compare results after a code change.

Hashes are not signatures. Editing both an artifact and its recorded hash can
defeat this integrity check; the manifest is the trusted local record. We do
not put a self-referential hash of the manifest inside itself.

## Step 7d: expose the commands

Add this last, below the other functions:

```python
def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline vDeltaE/hDeltaG annotation audit.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Verify sources and save a new annotation inventory.")
    run.add_argument("--vdeltae-census", required=True, type=Path)
    run.add_argument("--hdeltab-census", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    verify = commands.add_parser("verify", help="Verify a completed annotation inventory.")
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
```

`argv=None` makes normal execution use the command-line arguments. Tests can
instead pass a list explicitly, without spawning a subprocess. The final guard
runs `main()` for `python -m ...` but not when tests import the module.

From the repository root, after transcription and tests:

```bash
python -m fly_connectome.vdeltae_column_audit run \
  --vdeltae-census results/vdeltae-inputs-live-20261024 \
  --hdeltab-census results/hdeltab-extension-census-final-20261024 \
  --output results/vdeltae-hdeltag-annotation-audit-20261029

python -m fly_connectome.vdeltae_column_audit verify \
  results/vdeltae-hdeltag-annotation-audit-20261029
```

Use another fresh output name if that directory exists. These commands are
new entry points; no existing command is replaced. Successful real-data runs
should retain 18 vDeltaE and eight hDeltaG bodies, with all current names matched
and anatomical mapping still `unreviewed`.

## Step 7e: tests using the existing provenance fixture

Append these tests to `tests/test_vdeltae_column_audit.py`. They reuse Step 6's
`provenance_sources` fixture: the tiny source files are sufficient for actual
loading/assembly, while the full census verifiers remain substituted.

```python
def test_audit_command_round_trip(provenance_sources, tmp_path, capsys):
    directories, _ = provenance_sources
    output = tmp_path / "audit"
    audit.main([
        "run", "--vdeltae-census", str(directories["vdeltae_census"]),
        "--hdeltab-census", str(directories["hdeltab_census"]),
        "--output", str(output),
    ])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["population_counts"] == {"vDeltaE": 1, "hDeltaG": 1}
    assert manifest["parse_status_counts"] == {"matched": 2}
    assert "vdeltae_column_audit.py" in manifest["code_sha256"]
    bodies = pd.read_csv(output / "body_annotations.csv")
    assert set(bodies["bodyId"]) == {1, 2}
    assert bodies["anatomical_mapping_status"].eq("unreviewed").all()
    audit.main(["verify", str(output)])
    assert "Annotation audit verified:" in capsys.readouterr().out


def test_audit_refuses_existing_output(provenance_sources, tmp_path):
    directories, _ = provenance_sources
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        audit.run_audit(directories["vdeltae_census"], directories["hdeltab_census"], output)
    assert marker.read_text() == "preserve"
    assert not (output / "manifest.json").exists()


def test_audit_refuses_output_inside_source(provenance_sources):
    directories, _ = provenance_sources
    output = directories["vdeltae_census"] / "audit"
    with pytest.raises(ValidationError, match="separate from source"):
        audit.run_audit(directories["vdeltae_census"], directories["hdeltab_census"], output)
    assert not output.exists()


def test_audit_records_verifier_failure(provenance_sources, tmp_path, monkeypatch):
    directories, _ = provenance_sources
    output = tmp_path / "failed"

    def failing_verifier(directory):
        raise ValidationError("Synthetic source failure.")

    monkeypatch.setattr(audit, "verify_vdeltae_census", failing_verifier)
    with pytest.raises(ValidationError, match="Synthetic source failure"):
        audit.run_audit(directories["vdeltae_census"], directories["hdeltab_census"], output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error_type"] == "ValidationError"
    assert manifest["finished_at_utc"]
    with pytest.raises(ValidationError, match="complete annotation audit"):
        audit.verify_audit(output)


def test_audit_rechecks_sources_after_assembly(provenance_sources, tmp_path, monkeypatch):
    directories, _ = provenance_sources
    output = tmp_path / "changed-source"
    original_assemble = audit.assemble_body_table

    def changing_assemble(v_path, h_path):
        bodies = original_assemble(v_path, h_path)
        path = directories["vdeltae_census"] / "vDeltaE_neurons.csv"
        path.write_text(path.read_text() + "\n")
        return bodies

    monkeypatch.setattr(audit, "assemble_body_table", changing_assemble)
    with pytest.raises(ValidationError, match="changed after verification"):
        audit.run_audit(directories["vdeltae_census"], directories["hdeltab_census"], output)
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("filename", audit.AUDIT_FILES)
@pytest.mark.parametrize("problem", ["changed", "missing"])
def test_audit_detects_changed_or_missing_outputs(
    provenance_sources, tmp_path, filename, problem
):
    directories, _ = provenance_sources
    output = tmp_path / "audit"
    audit.run_audit(directories["vdeltae_census"], directories["hdeltab_census"], output)
    path = output / filename
    if problem == "changed":
        path.write_text(path.read_text() + "\n")
    else:
        path.unlink()
    with pytest.raises(ValidationError, match="artifact is missing, outside the run, or changed"):
        audit.verify_audit(output)


def test_audit_verification_rechecks_saved_sources(provenance_sources, tmp_path):
    directories, _ = provenance_sources
    output = tmp_path / "audit"
    audit.run_audit(directories["vdeltae_census"], directories["hdeltab_census"], output)
    path = directories["hdeltab_census"] / "manifest.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValidationError, match="changed after verification"):
        audit.verify_audit(output)
```

There are 12 new cases: five initial tests, six artifact-change/removal cases
(two problems for three files), and one saved-source change. With the previous
35 cases, expect **47 passing cases**. Run:

```bash
python -m pytest -q tests/test_vdeltae_column_audit.py
```

Once these pass and the real-data run verifies, the annotation-inventory command
is reviewable. Review documentation and run the repository suite before a commit.
The next scientific task is physical column alignment using independent
anatomical evidence; successful software checks do not resolve that mapping.

Continue with the [spatial-audit walkthrough](vdeltae_spatial_audit_walkthrough.md),
starting with a read-only inventory of the verified sources' spatial metadata.
