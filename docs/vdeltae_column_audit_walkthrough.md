# Hand-written implementation: vDeltaE/hDeltaG column annotation audit

Prepared 28 October 2026. This is a walkthrough, not an executable implementation.
The user will write the production module in small steps. Suggested code lives
here; the authoritative implementation will live in the module once written.
No neural dynamics, connectivity extraction or biological coordinate assignments
are part of this first task.

## Collaboration and file layout

- Suggested production module: `fly_connectome/vdeltae_column_audit.py`.
- Later regression tests: `tests/test_vdeltae_column_audit.py`.
- Walkthrough: this document, which can be versioned with the project.
- Generated tables and run manifests: a new directory under ignored `results/`.

Transcribe or adapt one small piece, run it, and inspect its output before moving
on. We can review your code and discuss each operation rather than having the
assistant fill in the rest of the module. Examples here explain implementation
choices; they are not a second program that needs to remain synchronized with
every future version of the production module. No ignored code-suggestion
directory is necessary. Do not stage the temporary dissertation PDF.

## First milestone and evidence boundary

Produce one row per selected body, preserving its original annotations and
extracting the literal column token in its instance name. The reviewed inputs
contain 18 vDeltaE bodies and eight hDeltaG bodies. These counts are dataset
checks, not assumptions used to manufacture membership.

The column token must be named `annotation_column`, not `angle` or a shared
physical coordinate. Repeated labels are allowed; duplicate body IDs are not.
vDeltaE currently has C0–C9 labels, whereas hDeltaG has C1–C8. Do not renumber,
drop boundary labels, merge bodies, or assign an eight-column circular map.
Missing or unexpected names should remain visible and be flagged at the parsing
step. Axonal position, dendritic position and cross-type alignment remain unknown
until independently supported by morphology or synapse locations.

## Step 1: load one exact-type population

Create the production file yourself and write this first piece, or an equivalent
that you understand. It deliberately preserves all annotation columns.

```python
from pathlib import Path

import pandas as pd

from .connectivity import ValidationError


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
```

Things to understand in this piece:

- `path` is passed in rather than embedded in the function, so the same reader
  works with saved runs and small synthetic test files.
- Nullable `Int64` preserves integer IDs and permits explicit missing-ID
  validation. Fractional identifiers are rejected by the CSV dtype conversion.
- Exact equality selects the population. It does not use substring matching or
  infer type from the instance name.
- `.copy()` makes a separate selected table; the source file is never modified.
- Sorting provides reproducible row order. It does not assign anatomical order.
- `.reset_index(drop=True)` removes incidental row indices from the larger input
  table; those indices have no anatomical meaning.
- The hDeltaG input CSV contains other populations. Filtering is intentional;
  the selected bodies, not the entire file, must have unique IDs.
- This is only a reader with local checks. Complete source-manifest/hash
  verification belongs in the later command wrapper, before using the data to
  publish an audit result.

After writing the function, run this from the repository root with the existing
project environment active:

```bash
python - <<'PY'
from fly_connectome.vdeltae_column_audit import load_population

vdeltae = load_population(
    "results/vdeltae-inputs-live-20261024/vDeltaE_neurons.csv", "vDeltaE"
)
hdeltag = load_population(
    "results/hdeltab-extension-census-final-20261024/candidate_neurons.csv", "hDeltaG"
)
for table in [vdeltae, hdeltag]:
    print(table[["bodyId", "type", "instance"]].to_string(index=False))
    print("Selected bodies:", len(table))
PY
```

Expected result for the saved runs: 18 vDeltaE and eight hDeltaG rows, with the
original body IDs and instance names. This inspection writes no files. If your
environment does not expose `python`, activate the project's conda environment
as described in the README before running it.

## Step 2: parse literal instance annotations

Add `import re` alongside your other imports, then write a second function
below `load_population()`. This piece reads names already loaded into a table;
it does not read or write files. It accepts either population, or a table
containing both populations.

We want three additional fields:

| Field | Meaning |
| --- | --- |
| `annotation_column` | Numeric token following `_C`, not an inferred anatomical coordinate |
| `instance_index` | Separate hDeltaG index preceding `_C`; not a column or angle |
| `annotation_parse_status` | `matched`, `missing_instance`, or `unrecognized_instance` |

The source `instance` column and every original row remain unchanged. Parsing
status describes recognition of a name's syntax, not validation of its anatomy.

```python
import re


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
            match = re.fullmatch(patterns[neuron.type], name) if isinstance(name, str) else None
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
```

Things to understand in this piece:

- `re` is Python's standard regular-expression library. The `r` before each
  pattern declares a raw string. The two patterns describe the reviewed naming
  formats; the hDeltaG index has two digits, while the column token has one or
  more digits. No current column-range limit is imposed by the parser.
- `(?P<column>...)` and `(?P<index>...)` name captured pieces of text. For
  `hDeltaG_14_C8`, these capture `8` and `14`, respectively. Calling `int(...)`
  converts them to numbers while the original full name remains available.
- `re.fullmatch(...)` requires the entire name to match. A suffix-only search
  could wrongly accept `prefix_vDeltaE_C3` or `vDeltaE_C3_extra`.
- The pattern is chosen using the exact `type` field. A vDeltaE row named
  `hDeltaG_01_C1` is unrecognized; its type is not silently changed.
- `pd.NA` records an unresolved value. `Int64`, unlike ordinary NumPy `int64`,
  can store missing integers without converting the whole column to floats.
- The three lists accumulate one result for every input row. Assigning arrays
  and a list preserves their positional correspondence even if the original
  DataFrame has nonconsecutive index labels.
- Missing or unrecognized names are retained and flagged, not discarded. No
  value of zero is used to represent missing data: C0 is a real observed label.
- Unsupported population types are rejected as a caller/selection error.
  Existing derived fields are rejected to avoid silently overwriting an earlier
  interpretation. To reparse, start from the original loaded table.
- The function does not repeat body-ID validation: call it on the tables loaded
  in Step 1. It does not merge neurons that share an annotation label.

### Inspect the real annotation table

After transcribing the parser, paste this complete block into Bash from the
repository root with the project environment active:

```bash
python - <<'PY'
from fly_connectome.vdeltae_column_audit import load_population, parse_annotations

sources = [
    ("results/vdeltae-inputs-live-20261024/vDeltaE_neurons.csv", "vDeltaE"),
    ("results/hdeltab-extension-census-final-20261024/candidate_neurons.csv", "hDeltaG"),
]
for path, population in sources:
    table = parse_annotations(load_population(path, population))
    print("\nPopulation:", population)
    print(table[["bodyId", "type", "instance", "annotation_column",
                 "instance_index", "annotation_parse_status"]].to_string(index=False))
    print("\nParsing status:")
    print(table["annotation_parse_status"].value_counts().to_string())
    print("\nBodies per annotation label (including unresolved labels):")
    print(table.groupby("annotation_column", dropna=False).size().to_string())
PY
```

Expected results for the saved files: all 18 vDeltaE and eight hDeltaG names
match the reviewed formats. vDeltaE body counts per annotation label are
`C0:2, C1:1, C2:1, C3:3, C4:2, C5:3, C6:1, C7:3, C8:1, C9:1`.
hDeltaG has one body for each C1–C8 label. This is an annotation inventory;
neither equal labels across types nor differences in label range establish a
shared anatomical coordinate system. This inspection still writes no files.

### Small regression checks you can write by hand

The real files contain only recognized names, so inspecting them alone would
not check our unresolved-name behavior. Add these tests to
`tests/test_vdeltae_column_audit.py` after implementing the parser:

```python
import pandas as pd
import pytest

from fly_connectome.vdeltae_column_audit import parse_annotations


def test_parser_retains_body_rows_and_distinguishes_index_from_column():
    original = pd.DataFrame({
        "bodyId": [1, 2, 3, 4, 5],
        "type": ["vDeltaE"] * 4 + ["hDeltaG"],
        "instance": ["vDeltaE_C0", "vDeltaE_C9", "vDeltaE_C3",
                     "vDeltaE_C3", "hDeltaG_14_C8"],
    }, index=[10, 20, 30, 40, 50])
    before = original.copy(deep=True)
    result = parse_annotations(original)
    pd.testing.assert_frame_equal(original, before)
    pd.testing.assert_frame_equal(result[original.columns], original)
    assert result["annotation_column"].tolist() == [0, 9, 3, 3, 8]
    assert result["instance_index"].iloc[:4].isna().all()
    assert result["instance_index"].iloc[4] == 14
    assert str(result["annotation_column"].dtype) == "Int64"
    assert result["annotation_parse_status"].eq("matched").all()


@pytest.mark.parametrize("name, status", [
    (None, "missing_instance"),
    ("", "missing_instance"),
    ("   ", "missing_instance"),
    ("vDeltaE_C3_extra", "unrecognized_instance"),
    ("prefix_vDeltaE_C3", "unrecognized_instance"),
    ("hDeltaG_01_C1", "unrecognized_instance"),
    (17, "unrecognized_instance"),
])
def test_unresolved_names_are_preserved_without_inventing_columns(name, status):
    original = pd.DataFrame({"bodyId": [1], "type": ["vDeltaE"], "instance": [name]})
    result = parse_annotations(original)
    pd.testing.assert_frame_equal(result[original.columns], original)
    assert len(result) == 1
    assert result["annotation_column"].isna().all()
    assert result["instance_index"].isna().all()
    assert result["annotation_parse_status"].iloc[0] == status
```

Run these after writing the code and tests:

```bash
python -m pytest -q tests/test_vdeltae_column_audit.py
```

Stop here to inspect and discuss the parsed table before assembling it below.

## Step 3: assemble and save a preliminary body-level table

Add this function below the parser. It uses the two functions you have already
written, preserves the union of the source annotation columns, and attaches
the source CSV path to each selected body.

```python
def assemble_body_table(vdeltae_path, hdeltag_path):
    populations = []
    for path, population in [
        (vdeltae_path, "vDeltaE"),
        (hdeltag_path, "hDeltaG"),
    ]:
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
```

Things to understand in this piece:

- Each input is selected by its expected exact type before concatenation.
  Passing the hDeltaG candidate census does not import its other populations.
- `pd.concat(...)` stacks rows and retains all source columns. If a column
  exists in only one source, the other population gets missing values in that
  column; annotations are not guessed or filled from the other population.
- The final duplicate check also catches a body appearing under conflicting
  types across the two files. Duplicate column labels such as C3 remain valid.
- `source_file` records the resolved local path. A path identifies the file
  used, but does not certify its contents or dataset provenance. Source census
  verification and hashes will be added in the later command wrapper.
- `anatomical_mapping_status` is deliberately `unreviewed`, even for a
  successfully parsed name. We have not assigned axonal/dendritic positions,
  angles, or a common coordinate system. Those fields can be added when there
  is evidence to populate them.
- Sorting by type and body ID makes the output reproducible without claiming
  anatomical order. There is no hardcoded requirement to produce 26 rows.

### Inspect first, then save to a fresh directory

After writing the function, run the following block from the repository root.
It prints the assembly result and writes two CSV files to a **new** directory.
If you run it again, choose a different output directory; existing directories
are refused. Neither input census is modified.

```bash
python - <<'PY'
from pathlib import Path

from fly_connectome.vdeltae_column_audit import assemble_body_table

bodies = assemble_body_table(
    "results/vdeltae-inputs-live-20261024/vDeltaE_neurons.csv",
    "results/hdeltab-extension-census-final-20261024/candidate_neurons.csv",
)
counts = (
    bodies.groupby(
        ["type", "annotation_parse_status", "annotation_column"], dropna=False
    )
    .size()
    .rename("body_count")
    .reset_index()
)
print(bodies[["bodyId", "type", "instance", "annotation_column",
              "instance_index", "anatomical_mapping_status"]].to_string(index=False))
print("\nBodies per annotation label:")
print(counts.to_string(index=False))

output = Path("results/vdeltae-hdeltag-annotation-draft-20261028")
output.mkdir(parents=True, exist_ok=False)
bodies.to_csv(output / "body_annotations.csv", index=False)
counts.to_csv(output / "annotation_label_counts.csv", index=False)
print("\nSaved preliminary annotation tables to:", output)
PY
```

The expected body table has 26 rows: 18 vDeltaE and eight hDeltaG, all with
`matched` parsing status and `unreviewed` anatomical mapping status. Check these
as observations of this saved dataset, not as rules for assigning membership.
The counts table describes numbers of bodies, not synapses or spatial coverage.
`dropna=False` keeps unresolved column labels visible if a later input contains
an unrecognized or missing name.

CSV represents missing values as empty cells. When loading the saved body table
for analysis, explicitly recover its nullable integer columns:

```python
bodies = pd.read_csv(
    "results/vdeltae-hdeltag-annotation-draft-20261028/body_annotations.csv",
    dtype={"bodyId": "Int64", "annotation_column": "Int64", "instance_index": "Int64"},
)
```

These are preliminary inspection artifacts. The short saving block is not yet
the final extraction/verification command: it has no complete/failed manifest
or source-hash checks. If writing is interrupted, keep that directory as an
incomplete attempt and use a fresh directory on the next run.

### Check assembly without depending on local census files

Append this test to your test file after transcribing the function. It checks
exact-type filtering, retention of source-specific fields, unresolved-name
retention, provenance, and conflicting body IDs using tiny temporary CSVs.

```python
from fly_connectome.connectivity import ValidationError
from fly_connectome.vdeltae_column_audit import assemble_body_table


def test_assembly_preserves_annotations_and_rejects_conflicting_ids(tmp_path):
    v_path = tmp_path / "v.csv"
    h_path = tmp_path / "h.csv"
    pd.DataFrame({
        "bodyId": [1, 2], "type": ["vDeltaE", "other"],
        "instance": ["unexpected_name", "other_C1"],
        "v_only_note": ["keep this", "not selected"],
    }).to_csv(v_path, index=False)
    hdeltag = pd.DataFrame({
        "bodyId": [3], "type": ["hDeltaG"], "instance": ["hDeltaG_14_C8"],
        "h_only_note": ["keep this too"],
    })
    hdeltag.to_csv(h_path, index=False)

    result = assemble_body_table(v_path, h_path).set_index("bodyId")
    assert set(result.index) == {1, 3}
    assert result.loc[1, "v_only_note"] == "keep this"
    assert result.loc[3, "h_only_note"] == "keep this too"
    assert pd.isna(result.loc[3, "v_only_note"])
    assert result.loc[1, "annotation_parse_status"] == "unrecognized_instance"
    assert pd.isna(result.loc[1, "annotation_column"])
    assert result.loc[3, "annotation_column"] == 8
    assert result.loc[1, "source_file"] == str(v_path.resolve())
    assert result.loc[3, "source_file"] == str(h_path.resolve())
    assert result["anatomical_mapping_status"].eq("unreviewed").all()

    hdeltag["bodyId"] = 1
    hdeltag.to_csv(h_path, index=False)
    with pytest.raises(ValidationError, match="both selected populations"):
        assemble_body_table(v_path, h_path)
```

Run `python -m pytest -q tests/test_vdeltae_column_audit.py` after adding it.

## Step 4: test source-table validation

The nine cases written so far cover annotation parsing and assembly. The next
tests exercise `load_population()` directly: malformed source tables should
fail before becoming an anatomical inventory. These tests still use synthetic
CSV files under `tmp_path`; no saved results or network access are needed.

Add `load_population` to the existing imports at the top of your test file:

```python
from fly_connectome.vdeltae_column_audit import (
    assemble_body_table,
    load_population,
    parse_annotations,
)
```

Keep your existing `pandas`, `pytest`, and `ValidationError` imports. Append the
following functions below your current tests.

### Missing required columns

```python
@pytest.mark.parametrize("missing_column", ["bodyId", "type", "instance"])
def test_loader_rejects_missing_required_columns(tmp_path, missing_column):
    path = tmp_path / "neurons.csv"
    table = pd.DataFrame({
        "bodyId": [1], "type": ["vDeltaE"], "instance": ["vDeltaE_C1"],
    })
    table.drop(columns=[missing_column]).to_csv(path, index=False)

    with pytest.raises(ValidationError, match="Neuron table needs"):
        load_population(path, "vDeltaE")
```

This produces three test cases, each removing a different required field from
an otherwise valid CSV. `drop(...)` returns a new table; it does not replace a
missing field with an empty value. A missing column and a missing value are
different input problems.

### No exact-type population

```python
def test_loader_rejects_absent_exact_type(tmp_path):
    path = tmp_path / "neurons.csv"
    pd.DataFrame({
        "bodyId": [1], "type": ["vDeltaE_other"], "instance": ["vDeltaE_C1"],
    }).to_csv(path, index=False)

    with pytest.raises(ValidationError, match="No exact-type vDeltaE"):
        load_population(path, "vDeltaE")
```

The related type name and plausible instance name deliberately provide a
temptation to infer membership. Neither is an exact `type="vDeltaE"` match,
so the loader must reject this selection rather than return an empty inventory
or silently include the related type.

### Missing or nonpositive selected body IDs

```python
@pytest.mark.parametrize("bad_id", [None, 0, -1])
def test_loader_rejects_missing_or_nonpositive_ids(tmp_path, bad_id):
    path = tmp_path / "neurons.csv"
    pd.DataFrame({
        "bodyId": [bad_id], "type": ["vDeltaE"], "instance": ["vDeltaE_C1"],
    }).to_csv(path, index=False)

    with pytest.raises(ValidationError, match="present and positive"):
        load_population(path, "vDeltaE")
```

These three cases distinguish an absent identifier from invalid numeric IDs.
`None` becomes an empty CSV cell, which the reader loads as a missing value.
Zero is invalid as a **body ID**, even though zero is a valid observed
**annotation column**. Those fields have different meanings and constraints.

### Duplicate IDs within a selected population

```python
def test_loader_rejects_duplicate_selected_ids(tmp_path):
    path = tmp_path / "neurons.csv"
    pd.DataFrame({
        "bodyId": [1, 1], "type": ["vDeltaE", "vDeltaE"],
        "instance": ["vDeltaE_C1", "vDeltaE_C2"],
    }).to_csv(path, index=False)

    with pytest.raises(ValidationError, match="Duplicate body IDs"):
        load_population(path, "vDeltaE")
```

Different instance names do not make a repeated body ID into two neurons. This
checks duplicates within one selected population; the assembly test already
checks conflicting IDs across the two populations. Repeated column labels
on distinct body IDs remain allowed and are covered by the parser test.

### Fractional identifiers

```python
def test_loader_rejects_fractional_ids(tmp_path):
    path = tmp_path / "neurons.csv"
    pd.DataFrame({
        "bodyId": [1.5], "type": ["vDeltaE"], "instance": ["vDeltaE_C1"],
    }).to_csv(path, index=False)

    with pytest.raises((TypeError, ValueError)):
        load_population(path, "vDeltaE")
```

This case checks that an identifier such as `1.5` cannot be silently rounded
or truncated to body `1`. Unlike the other failures, it occurs during pandas'
CSV conversion to nullable `Int64`, before our explicit validation checks.
The current project environment raises `TypeError`; the test permits the two
conversion exception classes without depending on pandas' exact error wording.
It does not require a `ValidationError` because the current loader does not
wrap conversion errors. No loader change is needed for this test.

CSV dtype conversion applies to the entire source table before exact-type
selection. A fractional ID anywhere in that source therefore prevents loading;
the subsequent positive-ID and duplicate checks apply to selected rows. This
distinction describes the current reader's behavior.

### Run the expanded tests

```bash
python -m pytest -q tests/test_vdeltae_column_audit.py
```

These additions contribute nine cases: three missing-column cases, one absent
type, three missing/nonpositive IDs, one duplicate-ID case, and one fractional
ID case. Together with the previous nine, expect **18 passing cases**.

This covers the principal source-validation failures identified at this stage.
Command-line behavior, source-hash verification, manifests and output-directory
handling will need checks when that wrapper is implemented.

## Step 5: verify source censuses and collect provenance

This is the first piece of the eventual command wrapper. Keep the table-loading
functions unchanged: the new function verifies the two source **directories**
and returns a small provenance record. It does not assemble or save an audit
run yet. No authentication or network access is required.

Add these imports to the production module, consolidating `ValidationError`
with its existing import rather than importing it twice:

```python
from .connectivity import DATASET, SERVER, ValidationError
from .extension_analysis import validate_saved_census
from .inputs import sha256
from .vdeltae_inputs import verify as verify_vdeltae_census
```

The two verifiers already exist:

- `verify_vdeltae_census(directory)` checks the completed vDeltaE input census,
  verifies its recorded artifact hashes, and reconstructs its input summaries
  and selected model-overlap matrices. It returns the source manifest.
- `validate_saved_census(directory)` checks the completed hDeltaB extension
  census, its recorded artifact hashes, outgoing summaries and declared motif
  matrices. It returns a tuple whose first element is the source manifest.
- `sha256(path)` computes the repository's existing SHA-256 file digest. A
  digest detects content changes relative to the recorded value; it is not an
  independent certification of the biological annotations.

Add this function below the assembly function:

```python
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

        if (manifest.get("server") != SERVER
                or manifest.get("dataset") != DATASET):
            raise ValidationError("Source server or dataset differs from the configured dataset.")
        metadata = manifest.get("dataset_metadata", {})
        if any(metadata.get(field) is None for field in identity_fields):
            raise ValidationError("Source dataset identity fields are missing.")
        if not metadata["uuid"]:
            raise ValidationError("Source dataset UUID is empty.")
        if manifest.get("data_sha256", {}).get(filename) != before["annotation_sha256"]:
            raise ValidationError("Annotation CSV is not covered by its source manifest hash.")
        if (sha256(manifest_path) != before["manifest_sha256"]
                or sha256(annotation_path) != before["annotation_sha256"]):
            raise ValidationError("Source manifest or annotation CSV changed during verification.")

        sources[role] = {
            "directory": str(directory),
            "manifest_file": str(manifest_path),
            "annotation_file": str(annotation_path),
            **before,
            "server": manifest["server"],
            "dataset": manifest["dataset"],
            "dataset_identity": {field: metadata[field] for field in identity_fields},
        }

    if (sources["vdeltae_census"]["dataset_identity"]
            != sources["hdeltab_census"]["dataset_identity"]):
        raise ValidationError("Source censuses describe different dataset states.")
    return sources
```

Things to understand in this piece:

- Arguments are census directories, unlike the CSV paths accepted by
  `assemble_body_table()`. File names are fixed by the existing census formats.
- `specifications` connects each source role to its expected annotation file.
  We verify both whole censuses even though this audit uses only their body
  annotations. A hash check alone would not reconstruct their summaries.
- `before` captures the manifest and annotation bytes before invoking the
  verifier. The second hash calculation checks that these files did not change
  during verification. Missing files naturally raise a filesystem error.
- Requiring the annotation file's digest in `data_sha256` prevents accepting a
  CSV simply because it happens to sit next to a valid census. The existing
  verifiers check the source manifests' completion and validation status.
- Matching the configured server and dataset is separate from matching dataset
  state. UUID, mutation ID and database-edit marker must be present and equal
  across the two runs. These checks compare recorded provenance; they do not
  ask the live database whether it has subsequently changed.
- `**before` copies both hashes into the returned record. Other fields record
  resolved file locations and the common dataset identity. The entire source
  metadata object is not duplicated in our smaller audit record; its manifest
  remains identified by path and hash.
- The function verifies sources at this point in time. The later wrapper must
  check their recorded hashes again after assembly and before declaring the
  audit complete. Keeping this step separate does not make subsequent reads
  atomic or prevent another process from changing files.

### Inspect provenance without writing files

After transcribing the imports and function, run this from the repository root
with the project environment active:

```bash
python - <<'PY'
import json

from fly_connectome.vdeltae_column_audit import collect_source_provenance

sources = collect_source_provenance(
    "results/vdeltae-inputs-live-20261024",
    "results/hdeltab-extension-census-final-20261024",
)
print(json.dumps(sources, indent=2))
PY
```

Expect two source records for `neuprint.janelia.org`, `male-cns:v1.0`, with
matching dataset identity records, resolved paths and file digests. This command
only reads the existing runs and prints provenance; it does not create a new
audit directory or modify the preliminary tables. Verifying a census also does
not establish what any column token means anatomically.

Before incorporating this function into the command wrapper, add focused tests
for incompatible dataset states, an unrecorded/changed annotation file, and a
manifest changed during verification. Those tests can substitute tiny verifier
stubs to exercise this function's checks; they do not need to reproduce the
whole census workflow. The existing census tests cover the verifiers themselves.

## Step 6: regression tests for provenance collection

Prepared 29 October 2026. This is a single testing milestone before writing the
command wrapper. Append the snippets below to your existing test file, moving
the new imports to its top. Leave the production module unchanged.

### Set up two tiny source directories

```python
import json

from fly_connectome import vdeltae_column_audit as audit


@pytest.fixture
def provenance_sources(tmp_path, monkeypatch):
    directories = {}
    calls = []
    for role, filename, csv_text in [
        ("vdeltae_census", "vDeltaE_neurons.csv",
         "bodyId,type,instance\n1,vDeltaE,vDeltaE_C1\n"),
        ("hdeltab_census", "candidate_neurons.csv",
         "bodyId,type,instance\n2,hDeltaG,hDeltaG_01_C1\n"),
    ]:
        directory = tmp_path / role
        directory.mkdir()
        annotation = directory / filename
        annotation.write_text(csv_text)
        manifest = {
            "status": "complete",
            "server": audit.SERVER,
            "dataset": audit.DATASET,
            "dataset_metadata": {
                "uuid": "synthetic-dataset", "latestMutationId": 1,
                "lastDatabaseEdit": "synthetic-edit-marker",
            },
            "data_sha256": {filename: audit.sha256(annotation)},
        }
        (directory / "manifest.json").write_text(json.dumps(manifest))
        directories[role] = directory

    def verify_vdeltae(directory):
        calls.append(("vdeltae_census", directory))
        return json.loads((directory / "manifest.json").read_text())

    def verify_hdeltab(directory):
        calls.append(("hdeltab_census", directory))
        manifest = json.loads((directory / "manifest.json").read_text())
        return manifest, None, None, None

    monkeypatch.setattr(audit, "verify_vdeltae_census", verify_vdeltae)
    monkeypatch.setattr(audit, "validate_saved_census", verify_hdeltab)
    return directories, calls
```

This introduces two pytest features:

- `@pytest.fixture` makes `provenance_sources` available as an argument to tests,
  just as pytest supplies `tmp_path`. Without a broader `scope`, each test case
  gets a fresh invocation and its own files and call list.
- `monkeypatch` temporarily substitutes the two verifier functions **where the
  collector looks them up**, in the `audit` module. Pytest restores them after
  each case. We preserve their return conventions, including the four-element
  hDeltaB tuple, but these stubs only read the synthetic manifests and record
  calls. They do not reconstruct or validate a real census.

This keeps the tests focused on our collector's additional checks. Tests of
the real verifiers remain separate; the live saved-census inspection from
Step 5 exercises their integration with this collector. Do not pass these
synthetic directories to the real census verifiers: they intentionally contain
only the files needed for this unit test.

### Successful collection and the hash-comparison regression

```python
def test_provenance_records_each_files_own_hash(provenance_sources):
    directories, calls = provenance_sources
    sources = audit.collect_source_provenance(
        directories["vdeltae_census"], directories["hdeltab_census"]
    )
    assert calls == list(directories.items())
    assert set(sources) == set(directories)
    for role, filename in [
        ("vdeltae_census", "vDeltaE_neurons.csv"),
        ("hdeltab_census", "candidate_neurons.csv"),
    ]:
        directory = directories[role].resolve()
        record = sources[role]
        assert record["directory"] == str(directory)
        assert record["manifest_file"] == str(directory / "manifest.json")
        assert record["annotation_file"] == str(directory / filename)
        assert record["manifest_sha256"] == audit.sha256(directory / "manifest.json")
        assert record["annotation_sha256"] == audit.sha256(directory / filename)
        assert record["manifest_sha256"] != record["annotation_sha256"]
        assert record["server"] == audit.SERVER
        assert record["dataset"] == audit.DATASET
    assert sources["vdeltae_census"]["dataset_identity"] == {
        "uuid": "synthetic-dataset", "latestMutationId": 1,
        "lastDatabaseEdit": "synthetic-edit-marker",
    }
    assert (sources["vdeltae_census"]["dataset_identity"]
            == sources["hdeltab_census"]["dataset_identity"])
```

The manifest and annotation hashes deliberately differ. The function must
nevertheless succeed, because each file is compared with its **own** earlier
hash. Comparing the manifest to the annotation's earlier hash reproduces the
transcription error and makes this test fail. The recorded call list also
checks that the collector invokes both verifiers with the expected directories.

### Different or incomplete dataset identities

```python
@pytest.mark.parametrize("field", ["uuid", "latestMutationId", "lastDatabaseEdit"])
def test_provenance_rejects_different_dataset_states(provenance_sources, field):
    directories, _ = provenance_sources
    path = directories["hdeltab_census"] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["dataset_metadata"][field] = "different-value"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValidationError, match="different dataset states"):
        audit.collect_source_provenance(
            directories["vdeltae_census"], directories["hdeltab_census"]
        )


@pytest.mark.parametrize("field", ["uuid", "latestMutationId", "lastDatabaseEdit"])
def test_provenance_rejects_missing_identity(provenance_sources, field):
    directories, _ = provenance_sources
    path = directories["vdeltae_census"] / "manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["dataset_metadata"][field]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValidationError, match="identity fields are missing"):
        audit.collect_source_provenance(
            directories["vdeltae_census"], directories["hdeltab_census"]
        )


@pytest.mark.parametrize("field", ["server", "dataset"])
def test_provenance_rejects_wrong_dataset_location(provenance_sources, field):
    directories, _ = provenance_sources
    path = directories["vdeltae_census"] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = "different-value"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValidationError, match="Source server or dataset differs"):
        audit.collect_source_provenance(
            directories["vdeltae_census"], directories["hdeltab_census"]
        )
```

These changes happen **before** calling the collector. The first group checks
incompatible provenance, the second checks missing evidence, and the third
checks the configured dataset location. None represents a file changing during
verification; the next tests exercise that separate failure mode.

### Missing or incorrect recorded annotation hashes

```python
@pytest.mark.parametrize("problem", ["missing", "incorrect"])
def test_provenance_requires_recorded_annotation_hash(provenance_sources, problem):
    directories, _ = provenance_sources
    path = directories["vdeltae_census"] / "manifest.json"
    manifest = json.loads(path.read_text())
    if problem == "missing":
        del manifest["data_sha256"]["vDeltaE_neurons.csv"]
    else:
        manifest["data_sha256"]["vDeltaE_neurons.csv"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValidationError, match="not covered by its source manifest hash"):
        audit.collect_source_provenance(
            directories["vdeltae_census"], directories["hdeltab_census"]
        )
```

The stub deliberately lets these manifests through, so this test can reach the
collector's own coverage check. The real verifier may reject an incorrect hash
earlier; either route prevents a completed audit from accepting that source.

### Files changed during verification

```python
@pytest.mark.parametrize("role, filename, verifier_name", [
    ("vdeltae_census", "manifest.json", "verify_vdeltae_census"),
    ("vdeltae_census", "vDeltaE_neurons.csv", "verify_vdeltae_census"),
    ("hdeltab_census", "manifest.json", "validate_saved_census"),
    ("hdeltab_census", "candidate_neurons.csv", "validate_saved_census"),
])
def test_provenance_rejects_changes_during_verification(
    provenance_sources, monkeypatch, role, filename, verifier_name
):
    directories, _ = provenance_sources
    original_verifier = getattr(audit, verifier_name)

    def changing_verifier(directory):
        result = original_verifier(directory)
        path = directory / filename
        path.write_text(path.read_text() + "\n")
        return result

    monkeypatch.setattr(audit, verifier_name, changing_verifier)
    with pytest.raises(ValidationError, match="changed during verification"):
        audit.collect_source_provenance(
            directories["vdeltae_census"], directories["hdeltab_census"]
        )
    assert (role, directories[role]) in provenance_sources[1]
```

The replacement verifier reads the manifest, changes one source file, and then
returns. This places the change between the collector's initial and final hash
checks without threads or timing-dependent behavior. Appending a newline changes
the bytes even if parsed CSV or JSON content would remain equivalent. The final
assertion checks the fixture's call list to confirm the intended source's
verifier was reached.

### Failures from either real-verifier boundary must propagate

```python
@pytest.mark.parametrize("verifier_name", [
    "verify_vdeltae_census", "validate_saved_census",
])
def test_provenance_propagates_verifier_failure(
    provenance_sources, monkeypatch, verifier_name
):
    directories, _ = provenance_sources

    def failing_verifier(directory):
        raise ValidationError("Synthetic census verification failed.")

    monkeypatch.setattr(audit, verifier_name, failing_verifier)
    with pytest.raises(ValidationError, match="Synthetic census verification failed"):
        audit.collect_source_provenance(
            directories["vdeltae_census"], directories["hdeltab_census"]
        )
```

This checks that the collector does not suppress verifier failures and return
a provenance record anyway. It does not independently test how the underlying
verifier detects a failed census.

### Run this milestone

```bash
python -m pytest -q tests/test_vdeltae_column_audit.py
```

These snippets add 17 cases: one success, three different identity fields,
three missing identity fields, two dataset-location fields, two annotation-hash
problems, four mid-verification file changes, and two verifier failures.
Together with your previous 18, expect **35 passing cases**.

Stop here to review the fixture and its substitutions before adding the command
wrapper. The wrapper will still need its own tests for output-directory refusal,
complete/failed manifests, final source rechecks and output verification.

## Later pieces: agree and write them one at a time

1. **Add the reproducible command and saved-run verification.** Continue with
   [Step 7 in the command walkthrough](vdeltae_column_audit_command_walkthrough.md).
   It uses Step 5's verified source records, saves tables and a readable summary,
   records code/source/output hashes, and adds complete/failed run manifests.
   Its regression tests reuse the fixture from Step 6. Results stay separate
   from input censuses; no neuron-count assumptions are added to the parser.

The immediate goal ends at the annotation table. Ordered connectivity heatmaps,
anatomical compartment checks and shared-coordinate hypotheses come afterward.
The existing PFN-specific annotation parser must not be reused with an assumed
vDeltaE/hDeltaG column convention.
