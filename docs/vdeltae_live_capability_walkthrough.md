# Hand-written implementation: live spatial data availability probe

Prepared 29 October 2026. Continue after the offline selection/query builders in
the [spatial-query walkthrough](vdeltae_spatial_capability_walkthrough.md). Add the
suggestions below to `fly_connectome/vdeltae_spatial_audit.py`, one piece at a
time. The runner and tests are now implemented, and the bounded saved run has
been verified. These snippets preserve the implementation guide; the production
module and tests are authoritative.

This is a two-body data-access probe. It saves the four existing query results,
requests two unhealed skeletons and the FB mesh, and records what was accessible.
It does not extract all synapse coordinates, assign columns, validate axon versus
dendrite labels, or alter neural dynamics. A successful download means a payload
was retrieved, not that its geometry has been biologically validated.

Here, “spatial data availability probe” means checking which data the selected
neurons expose and which resources can be retrieved. The runner is named
`run_spatial_data_probe`, its verifier `verify_spatial_data_probe`, the test
fixture `spatial_probe_environment`, and the format constant `SPATIAL_PROBE_KIND`.
The existing filename `capability.json` and format identifier string remain
unchanged for saved-run compatibility; neither denotes a special neuPrint query
category. Function names are independent of those saved-format identifiers.

## 1. Imports and metadata compatibility

Consolidate these with the imports already in the module:

```python
from datetime import datetime, timezone
from importlib.metadata import version
import platform

from neuprint import fetch_meta
from requests.exceptions import HTTPError, RequestException, Timeout

from .connectivity import DATASET, SERVER, ValidationError
from .extension_analysis import markdown
from .inputs import integers, save_json, sha256
```

Add these constants and helpers. Metadata comparison deliberately covers both
dataset state and coordinates; it never converts positions or adopts a filter.

```python
SPATIAL_PROBE_KIND = "vdeltae_hdeltag_spatial_capability_v1"
PROBE_COUNT_COLUMNS = (
    "all_contained_sites", "sites_marked_FB", "sites_with_location", "sites_with_confidence",
)


def check_probe_metadata(metadata, inventory):
    for source in inventory["sources"].values():
        expected = source["spatial_metadata"]
        fields = {
            **expected["dataset_identity"],
            **expected["coordinate_metadata"],
            "postHighAccuracyThreshold": expected["recorded_post_high_accuracy_threshold"],
        }
        for field, value in fields.items():
            if value is None or metadata.get(field) != value:
                raise ValidationError(f"Live dataset metadata differs or is missing: {field}.")


def require_probe_columns(table, columns):
    if not set(columns).issubset(table.columns):
        raise ValidationError("Spatial data table is missing required columns.")


def validate_probe_tables(tables, probe):
    expected_ids = set(int(value) for value in probe["bodyId"])
    annotations = tables["body_annotations"]
    availability = tables["site_availability"]
    properties = tables["site_property_inventory"]
    categories = tables["fb_site_categories"]
    require_probe_columns(annotations, ["bodyId", "type", "instance"])
    require_probe_columns(availability, ["bodyId", *PROBE_COUNT_COLUMNS])
    require_probe_columns(properties, ["bodyId", "synapse_type", "property", "sites_with_property"])
    require_probe_columns(categories, ["bodyId", "synapse_type", "compartment", "split", "site_count"])

    for table in [annotations, availability, properties, categories]:
        ids = integers(table["bodyId"], "Returned body IDs", 1)
        if not set(ids).issubset(expected_ids):
            raise ValidationError("Spatial data query returned an unselected body.")
    for table in [annotations, availability]:
        if table["bodyId"].duplicated().any() or set(table["bodyId"]) != expected_ids:
            raise ValidationError("Spatial data query is missing or duplicates a selected body.")
    columns = ["bodyId", "type", "instance"]
    try:
        pd.testing.assert_frame_equal(
            annotations[columns].sort_values("bodyId").reset_index(drop=True),
            probe[columns].sort_values("bodyId").reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError as error:
        raise ValidationError("Live body annotations differ from the saved audit.") from error

    counts = availability.set_index("bodyId")
    for name in PROBE_COUNT_COLUMNS:
        integers(counts[name], name)
        if counts[name].gt(counts["all_contained_sites"]).any():
            raise ValidationError("Site availability exceeds its all-site denominator.")
    for table, keys, count_column in [
        (properties, ["bodyId", "synapse_type", "property"], "sites_with_property"),
        (categories, ["bodyId", "synapse_type", "compartment", "split"], "site_count"),
    ]:
        integers(table[count_column], count_column, 1)
        if table.duplicated(keys).any() or not table["synapse_type"].isin(["pre", "post"]).all():
            raise ValidationError("Duplicate categories or unexpected synapse polarity; review raw data.")
    if properties["property"].isna().any():
        raise ValidationError("A site property name is missing.")
    for name, group in properties.groupby("property"):
        totals = group.groupby("bodyId")["sites_with_property"].sum().reindex(counts.index, fill_value=0)
        if totals.gt(counts["all_contained_sites"]).any():
            raise ValidationError("Property coverage exceeds its all-site denominator.")
    for property_name, count_column in [
        ("location", "sites_with_location"), ("confidence", "sites_with_confidence"),
    ]:
        totals = (
            properties.loc[properties["property"].eq(property_name)]
            .groupby("bodyId")["sites_with_property"].sum()
            .reindex(counts.index, fill_value=0)
        )
        if not totals.eq(counts[count_column]).all():
            raise ValidationError("Property inventory does not reconcile with availability.")
    fb_totals = categories.groupby("bodyId")["site_count"].sum().reindex(counts.index, fill_value=0)
    if not fb_totals.eq(counts["sites_marked_FB"]).all():
        raise ValidationError("FB categories do not reconcile with marked-FB site counts.")
```

The saved Meta node names this dataset `male-cns`, while the client selects the
versioned name `male-cns:v1.0`. The runner checks the versioned client target;
this helper compares UUID, mutation/edit markers and spatial metadata. It must
not require the Meta node's base name to equal the versioned client name.

The checks require the same two bodies, the same type/instance names, integer
counts and compatible denominators. Empty property/category tables are allowed
when their corresponding counts are zero; they still need their declared
columns. Null compartment/split values remain valid groups. An unexpected
polarity causes a stop for review, without deleting the raw query results.

These checks cannot prove that Cypher was implemented correctly by a server,
or that every graph node corresponds to a true biological site. They compare
independently grouped answers under the same unfiltered query scope. No recorded
confidence cutoff has been applied, and neuron `pre`/`post` totals are not used
as interchangeable denominators.

## 2. Retrieve optional resources without hiding failures

```python
def retrieve_probe_resource(fetch, path, text=False):
    try:
        payload = fetch()
        if text:
            if not isinstance(payload, str):
                raise TypeError("Expected text resource.")
            data = payload.encode("utf-8")
        else:
            if not isinstance(payload, bytes):
                raise TypeError("Expected byte resource.")
            data = payload
        path.write_bytes(data)
        return {
            "status": "retrieved" if data else "empty_response",
            "file": path.name, "bytes": len(data),
        }
    except HTTPError as error:
        code = error.response.status_code if error.response is not None else None
        if code == 404:
            status = "http_not_found"
        elif code in [401, 403]:
            status = "authentication_or_access_error"
        else:
            status = "http_error"
        return {"status": status, "error_type": type(error).__name__, "http_status": code}
    except Timeout as error:
        return {"status": "timeout", "error_type": type(error).__name__}
    except RequestException as error:
        return {"status": "request_error", "error_type": type(error).__name__}
    except Exception as error:
        return {"status": "unexpected_error", "error_type": type(error).__name__}
```

This helper always returns a resource record. The runner below decides which
statuses permit completion. It does not silently swallow access or parser
errors: those become explicit resource records followed by a failed run.

`http_not_found` records the endpoint response, not a conclusion that the
biological neuron lacks a skeleton or that the resource is permanently absent.
`empty_response` means zero returned bytes; a nonempty comment-only SWC would
still be `retrieved`. Validating morphology format/content is a later step.
Only exception classes/status codes are recorded, never credentials, response
bodies or arbitrary exception messages. Filesystem failures also remain failures.

## 3. Execute, preserve raw results, and finalize provenance

```python
def run_spatial_data_probe(client, audit_directory, output):
    audit_directory = Path(audit_directory).resolve()
    output = Path(output).resolve()
    bodies, inventory = inspect_spatial_sources(audit_directory)
    probe = select_spatial_probe(bodies)
    queries = spatial_probe_queries(probe["bodyId"])
    source_directories = [audit_directory] + [
        Path(source["source_manifest_file"]).parent for source in inventory["sources"].values()
    ]
    for source in source_directories:
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValidationError("Spatial probe output must be separate from its source directories.")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "kind": SPATIAL_PROBE_KIND, "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "server": SERVER, "dataset": DATASET,
        "source_inventory": inventory,
        "selection_rule": "Smallest audited body ID per population; no geometry/connectivity selection.",
        "selected_body_ids": [int(value) for value in probe["bodyId"]],
        "queries": queries,
        "settings": {
            "confidence_filter": None, "sites": "All contained sites; FB category query uses s.FB = true.",
            "site_denominator": "Distinct neuron/Synapse-node pairs; not contacts.",
            "skeleton_heal": False, "mesh_purpose": "Visualization context only.",
            "anatomical_mapping": "unreviewed", "resource_validation": "Payload retrieval only.",
        },
        "resources": {},
    }
    save_json(output / "capability.json", manifest)
    try:
        manifest["code_sha256"] = {
            path.name: sha256(path) for path in sorted(Path(__file__).resolve().parent.glob("*.py"))
        }
        manifest["python_version"] = platform.python_version()
        manifest["packages"] = {
            name: version(name) for name in ["neuprint-python", "numpy", "pandas", "requests"]
        }
        # Client.server is normalized by neuprint-python; check the actual target too.
        if client.server.rstrip("/") != f"https://{SERVER}" or client.dataset != DATASET:
            raise ValidationError("Client target differs from the configured server/dataset.")
        before = fetch_meta(client=client)
        save_json(output / "metadata_before.json", before)
        check_probe_metadata(before, inventory)
        probe.to_csv(output / "probe_bodies.csv", index=False)
        tables = {}
        for name, query in queries.items():
            tables[name] = client.fetch_custom(query)
            tables[name].to_csv(output / f"{name}.csv", index=False)
        validate_probe_tables(tables, probe)

        for body_id in probe["bodyId"]:
            body_id = int(body_id)
            key = f"skeleton_{body_id}"
            manifest["resources"][key] = retrieve_probe_resource(
                lambda body_id=body_id: client.fetch_skeleton(body_id, heal=False, format="swc"),
                output / f"{key}.swc", text=True,
            )
            save_json(output / "capability.json", manifest)
            if manifest["resources"][key]["status"] not in ["retrieved", "empty_response", "http_not_found"]:
                raise ValidationError("Optional resource request failed; inspect resource status.")
        manifest["resources"]["FB_mesh"] = retrieve_probe_resource(
            lambda: client.fetch_roi_mesh("FB"), output / "FB.obj"
        )
        save_json(output / "capability.json", manifest)
        if manifest["resources"]["FB_mesh"]["status"] not in ["retrieved", "empty_response", "http_not_found"]:
            raise ValidationError("Optional resource request failed; inspect resource status.")

        after = fetch_meta(client=client)
        save_json(output / "metadata_after.json", after)
        check_probe_metadata(after, inventory)
        _, final_inventory = inspect_spatial_sources(audit_directory)
        if final_inventory != inventory:
            raise ValidationError("Saved spatial sources changed during spatial data availability probing.")
        resource_table = pd.DataFrame([
            {"resource": key, **record} for key, record in manifest["resources"].items()
        ])
        (output / "summary.md").write_text("\n".join([
            "# Spatial data availability probe", "", markdown(probe[["bodyId", "type", "instance"]]),
            "", "## Site availability", "", markdown(tables["site_availability"]),
            "", "## Resource responses", "", markdown(resource_table),
            "", "Column meanings, compartment meanings and axis orientation remain unresolved.",
            "Resource retrieval does not validate morphology. No confidence filter or neural dynamics applied.", "",
        ]))
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error_type"] = type(error).__name__
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["artifact_sha256"] = {
            path.name: sha256(path) for path in sorted(output.iterdir())
            if path.is_file() and path.name != "capability.json"
        }
        save_json(output / "capability.json", manifest)
    return manifest
```

The runner accepts an already-created client, making fake-client tests possible.
It checks the actual client target and fresh metadata rather than trusting a
cached `client.meta` value. Queries are retained in the manifest before requests
start; each returned table is saved before validation, preserving diagnostics.
Requests stop at the first unexpected resource failure, so later resources may
remain unattempted. Complete means the bounded probe finished consistently;
404/empty optional responses are still visible and may limit further analysis.

The `lambda body_id=body_id: ...` binds the current loop value as an argument
default. It avoids depending on a changing outer loop variable. `fetch` is a
callable passed to the retrieval helper, which invokes it inside its error handler.

As with the annotation runner, missing/unverified input before directory creation
cannot produce a failed manifest. Disk failures or forced termination can prevent
finalization. A manifest still marked `running` is incomplete. Source checks
detect ordinary drift; they do not create a locked database/filesystem snapshot.

## 4. Verify saved results without network access

```python
def verify_spatial_data_probe(output):
    output = Path(output).resolve()
    manifest = json.loads((output / "capability.json").read_text())
    if manifest.get("kind") != SPATIAL_PROBE_KIND or manifest.get("status") != "complete":
        raise ValidationError("Verification requires a complete spatial data availability probe.")
    required = {
        "metadata_before.json", "metadata_after.json", "probe_bodies.csv", "summary.md",
        "body_annotations.csv", "site_availability.csv", "site_property_inventory.csv", "fb_site_categories.csv",
    }
    hashes = manifest.get("artifact_sha256", {})
    if not required.issubset(hashes):
        raise ValidationError("Spatial probe artifact hash inventory is incomplete.")
    for name, digest in hashes.items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or not path.is_file() or sha256(path) != digest:
            raise ValidationError("Spatial probe artifact is missing, outside the run, or changed.")
    probe = pd.read_csv(output / "probe_bodies.csv", dtype={"bodyId": "Int64"})
    queries = spatial_probe_queries(probe["bodyId"])
    if manifest.get("queries") != queries:
        raise ValidationError("Saved probe queries differ from the declared query design.")
    tables = {name: pd.read_csv(output / f"{name}.csv") for name in queries}
    validate_probe_tables(tables, probe)
    inventory = manifest["source_inventory"]
    for name in ["metadata_before.json", "metadata_after.json"]:
        check_probe_metadata(json.loads((output / name).read_text()), inventory)
    bodies, current_inventory = inspect_spatial_sources(inventory["annotation_audit_directory"])
    if current_inventory != inventory:
        raise ValidationError("Saved spatial probe source provenance changed.")
    expected = select_spatial_probe(bodies)
    pd.testing.assert_frame_equal(probe, expected, check_dtype=False)
    expected_resources = {f"skeleton_{int(body)}" for body in probe["bodyId"]} | {"FB_mesh"}
    if set(manifest.get("resources", {})) != expected_resources:
        raise ValidationError("Spatial probe resource inventory is incomplete or unexpected.")
    for record in manifest["resources"].values():
        if record["status"] not in ["retrieved", "empty_response", "http_not_found"]:
            raise ValidationError("Completed spatial data availability probe contains a resource error.")
        if record["status"] in ["retrieved", "empty_response"]:
            if record.get("file") not in hashes:
                raise ValidationError("Resource payload is not covered by artifact hashes.")
            size = (output / record["file"]).stat().st_size
            if size != record["bytes"] or (record["status"] == "empty_response") != (size == 0):
                raise ValidationError("Resource byte count or empty status differs from payload.")
    return manifest
```

This uses the original annotation audit and source censuses locally, but never
contacts neuPrint. The manifest is a trusted local record, not a signature.
Recorded code hashes identify run provenance; they do not require today's code
to be byte-identical. Query-definition changes can nevertheless require using
the run's original code for this strict query-design verification.

## 5. Focused offline tests before using a real client

Append these to `tests/test_vdeltae_spatial_audit.py`. Add imports at the top.
These tests inject a fake client and fake metadata fetcher; **no credentials or
network** are used. The fake-client fixture supplies consistent site counts,
including null compartment/split values, and tiny raw geometry payloads.

```python
from copy import deepcopy

from requests import Response
from requests.exceptions import HTTPError, Timeout

from fly_connectome import vdeltae_spatial_audit as spatial


@pytest.fixture
def spatial_probe_environment(tmp_path, monkeypatch):
    probe = pd.DataFrame({
        "bodyId": [10, 1], "type": ["vDeltaE", "hDeltaG"],
        "instance": ["vDeltaE_C6", "hDeltaG_14_C8"], "annotation_column": [6, 8],
    })
    metadata = {
        "dataset": "male-cns", "uuid": "test", "latestMutationId": 1,
        "lastDatabaseEdit": "test", "voxelSize": [8, 8, 8],
        "voxelUnits": "nanometers", "postHighAccuracyThreshold": .5,
    }
    source = {
        "source_manifest_file": str(tmp_path / "source" / "manifest.json"),
        "source_manifest_sha256": "synthetic-hash",
        "spatial_metadata": spatial.spatial_metadata_inventory(metadata),
    }
    inventory = {
        "annotation_audit_directory": str(tmp_path / "audit"),
        "annotation_audit_manifest_sha256": "synthetic-audit-hash",
        "sources": {"vdeltae_census": deepcopy(source), "hdeltab_census": deepcopy(source)},
    }
    tables = {
        "body_annotations": probe[["bodyId", "type", "instance"]].copy(),
        "site_availability": pd.DataFrame({
            "bodyId": [10, 1], "all_contained_sites": [1, 1], "sites_marked_FB": [1, 1],
            "sites_with_location": [1, 1], "sites_with_confidence": [1, 1],
        }),
        "site_property_inventory": pd.DataFrame([
            {"bodyId": body, "synapse_type": polarity, "property": prop, "sites_with_property": 1}
            for body, polarity in [(10, "pre"), (1, "post")]
            for prop in ["location", "confidence"]
        ]),
        "fb_site_categories": pd.DataFrame({
            "bodyId": [10, 1], "synapse_type": ["pre", "post"],
            "compartment": [None, None], "split": [None, None], "site_count": [1, 1],
        }),
    }

    class FakeClient:
        server = f"https://{spatial.SERVER}"
        dataset = spatial.DATASET

        def fetch_custom(self, query):
            name = next(name for name, text in spatial.spatial_probe_queries(probe["bodyId"]).items() if text == query)
            return tables[name].copy(deep=True)

        def fetch_skeleton(self, body, heal, format):
            assert body in [10, 1] and heal is False and format == "swc"
            return "1 0 0 0 0 1 -1\n"

        def fetch_roi_mesh(self, roi):
            assert roi == "FB"
            return b"v 0 0 0\n"

    monkeypatch.setattr(spatial, "inspect_spatial_sources", lambda path: (probe.copy(), deepcopy(inventory)))
    monkeypatch.setattr(spatial, "fetch_meta", lambda **kwargs: deepcopy(metadata))
    return FakeClient(), probe, inventory, metadata, tables


def test_spatial_probe_run_and_offline_verification(spatial_probe_environment, tmp_path):
    client, probe, inventory, _, _ = spatial_probe_environment
    output = tmp_path / "probe"
    result = spatial.run_spatial_data_probe(client, inventory["annotation_audit_directory"], output)
    assert result["status"] == "complete"
    assert result["selected_body_ids"] == [10, 1]
    assert set(result["resources"]) == {"skeleton_10", "skeleton_1", "FB_mesh"}
    assert spatial.verify_spatial_data_probe(output)["status"] == "complete"


def test_spatial_probe_refuses_existing_output(spatial_probe_environment, tmp_path):
    client, _, inventory, _, _ = spatial_probe_environment
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        spatial.run_spatial_data_probe(client, inventory["annotation_audit_directory"], output)
    assert list(output.iterdir()) == []


def test_spatial_probe_records_metadata_drift(spatial_probe_environment, tmp_path):
    client, _, inventory, metadata, _ = spatial_probe_environment
    metadata["latestMutationId"] = 2
    output = tmp_path / "drift"
    with pytest.raises(ValidationError, match="metadata differs"):
        spatial.run_spatial_data_probe(client, inventory["annotation_audit_directory"], output)
    result = json.loads((output / "capability.json").read_text())
    assert result["status"] == "failed"
    assert "metadata_before.json" in result["artifact_sha256"]
    assert not (output / "body_annotations.csv").exists()


def test_spatial_probe_saves_raw_tables_before_reconciliation_failure(spatial_probe_environment, tmp_path):
    client, _, inventory, _, tables = spatial_probe_environment
    tables["fb_site_categories"].loc[0, "site_count"] = 2
    output = tmp_path / "bad-counts"
    with pytest.raises(ValidationError, match="FB categories do not reconcile"):
        spatial.run_spatial_data_probe(client, inventory["annotation_audit_directory"], output)
    result = json.loads((output / "capability.json").read_text())
    assert result["status"] == "failed"
    assert "fb_site_categories.csv" in result["artifact_sha256"]


@pytest.mark.parametrize("code, expected_status", [
    (404, "http_not_found"), (401, "authentication_or_access_error"), (500, "http_error"),
])
def test_resource_http_failures_remain_distinct(tmp_path, code, expected_status):
    def fetch():
        response = Response()
        response.status_code = code
        raise HTTPError("not recorded", response=response)

    result = spatial.retrieve_probe_resource(fetch, tmp_path / "resource.obj")
    assert result["status"] == expected_status
    assert result["http_status"] == code
    assert "not recorded" not in json.dumps(result)


def test_resource_timeout_is_not_an_absence(tmp_path):
    def fetch():
        raise Timeout("not recorded")

    assert spatial.retrieve_probe_resource(fetch, tmp_path / "resource.obj")["status"] == "timeout"


def test_empty_resource_payload_is_preserved(tmp_path):
    path = tmp_path / "resource.obj"
    record = spatial.retrieve_probe_resource(lambda: b"", path)
    assert record["status"] == "empty_response"
    assert path.is_file() and record["bytes"] == 0


def test_resource_access_error_fails_the_run(spatial_probe_environment, tmp_path, monkeypatch):
    client, _, inventory, _, _ = spatial_probe_environment

    def denied(*args, **kwargs):
        response = Response()
        response.status_code = 403
        raise HTTPError("not recorded", response=response)

    monkeypatch.setattr(client, "fetch_skeleton", denied)
    output = tmp_path / "denied"
    with pytest.raises(ValidationError, match="Optional resource request failed"):
        spatial.run_spatial_data_probe(client, inventory["annotation_audit_directory"], output)
    result = json.loads((output / "capability.json").read_text())
    assert result["status"] == "failed"
    assert result["resources"]["skeleton_10"]["http_status"] == 403
    assert "FB_mesh" not in result["resources"]


def test_spatial_probe_detects_changed_saved_output(spatial_probe_environment, tmp_path):
    client, _, inventory, _, _ = spatial_probe_environment
    output = tmp_path / "probe"
    spatial.run_spatial_data_probe(client, inventory["annotation_audit_directory"], output)
    (output / "site_availability.csv").write_text("changed")
    with pytest.raises(ValidationError, match="artifact is missing, outside the run, or changed"):
        spatial.verify_spatial_data_probe(output)


@pytest.mark.parametrize("problem", ["missing_body", "changed_instance"])
def test_spatial_probe_rejects_live_identity_changes(spatial_probe_environment, tmp_path, problem):
    client, _, inventory, _, tables = spatial_probe_environment
    if problem == "missing_body":
        tables["body_annotations"] = tables["body_annotations"].iloc[:1].copy()
    else:
        tables["body_annotations"].loc[0, "instance"] = "changed"
    output = tmp_path / "identity-change"
    with pytest.raises(ValidationError):
        spatial.run_spatial_data_probe(client, inventory["annotation_audit_directory"], output)
    assert json.loads((output / "capability.json").read_text())["status"] == "failed"


def test_spatial_probe_rechecks_live_metadata_at_end(spatial_probe_environment, tmp_path, monkeypatch):
    client, _, inventory, metadata, _ = spatial_probe_environment
    calls = []

    def metadata_fetcher(**kwargs):
        result = deepcopy(metadata)
        if calls:
            result["latestMutationId"] = 2
        calls.append(True)
        return result

    monkeypatch.setattr(spatial, "fetch_meta", metadata_fetcher)
    output = tmp_path / "late-metadata-drift"
    with pytest.raises(ValidationError, match="metadata differs"):
        spatial.run_spatial_data_probe(client, inventory["annotation_audit_directory"], output)
    assert len(calls) == 2
    result = json.loads((output / "capability.json").read_text())
    assert result["status"] == "failed"
    assert "metadata_after.json" in result["artifact_sha256"]


def test_spatial_probe_rechecks_saved_source_inventory(spatial_probe_environment, tmp_path, monkeypatch):
    client, probe, inventory, _, _ = spatial_probe_environment
    calls = []

    def inspection(path):
        result = deepcopy(inventory)
        if calls:
            result["annotation_audit_manifest_sha256"] = "changed"
        calls.append(True)
        return probe.copy(), result

    monkeypatch.setattr(spatial, "inspect_spatial_sources", inspection)
    output = tmp_path / "late-source-drift"
    with pytest.raises(ValidationError, match="sources changed"):
        spatial.run_spatial_data_probe(client, inventory["annotation_audit_directory"], output)
    assert len(calls) == 2
    assert json.loads((output / "capability.json").read_text())["status"] == "failed"


def test_spatial_probe_preserves_optional_not_found_response(spatial_probe_environment, tmp_path, monkeypatch):
    client, _, inventory, _, _ = spatial_probe_environment

    def missing_mesh(roi):
        response = Response()
        response.status_code = 404
        raise HTTPError("not recorded", response=response)

    monkeypatch.setattr(client, "fetch_roi_mesh", missing_mesh)
    output = tmp_path / "missing-mesh"
    result = spatial.run_spatial_data_probe(client, inventory["annotation_audit_directory"], output)
    assert result["status"] == "complete"
    assert result["resources"]["FB_mesh"]["status"] == "http_not_found"
    assert not (output / "FB.obj").exists()
    assert spatial.verify_spatial_data_probe(output)["status"] == "complete"


def test_zero_site_rows_are_retained_without_inventing_categories(spatial_probe_environment, tmp_path):
    client, _, inventory, _, tables = spatial_probe_environment
    tables["site_availability"].loc[:, list(spatial.PROBE_COUNT_COLUMNS)] = 0
    tables["site_property_inventory"] = tables["site_property_inventory"].iloc[:0].copy()
    tables["fb_site_categories"] = tables["fb_site_categories"].iloc[:0].copy()
    output = tmp_path / "zero-sites"
    spatial.run_spatial_data_probe(client, inventory["annotation_audit_directory"], output)
    assert spatial.verify_spatial_data_probe(output)["status"] == "complete"
    assert len(pd.read_csv(output / "site_availability.csv")) == 2
```

These add 17 cases to the existing 15 spatial tests: **32 spatial cases**. Run:

```bash
python -m pytest -q tests/test_vdeltae_spatial_audit.py
```

The fake client tests runner behavior, not server-side Cypher syntax or real
resource availability. The final cases explicitly check changed body identity,
late live/saved-source drift, a 404 optional-resource response and zero-site
populations. These are separate outcomes, not reasons to discard bodies or
silently alter the query scope.

## 6. Run the real bounded probe only after transcription and review

Use the existing local `.env` or exported `NEUPRINT_TOKEN`, as in the README.
The block below prints only safe statuses. Do not paste tokens into the block.
Choose a new output directory; existing runs are refused.

```bash
python - <<'PY'
import os
from pathlib import Path

from dotenv import load_dotenv
from neuprint import Client

from fly_connectome.connectivity import DATASET, SERVER, ValidationError
from fly_connectome.vdeltae_spatial_audit import run_spatial_data_probe

load_dotenv(Path.cwd() / ".env", override=False)
token = os.environ.get("NEUPRINT_TOKEN", "").strip()
if not token:
    raise ValidationError("Set NEUPRINT_TOKEN locally as described in the README.")
try:
    client = Client(SERVER, dataset=DATASET, token=token)
    result = run_spatial_data_probe(
        client,
        "results/vdeltae-hdeltag-annotation-audit-20261029",
        "results/vdeltae-hdeltag-spatial-capability-20261029",
    )
except Exception as error:
    print(f"Spatial data availability probe stopped ({type(error).__name__}); inspect any saved capability.json and raw tables.")
    raise SystemExit(1) from None
print("Spatial data availability probe:", result["status"])
for name, record in result["resources"].items():
    print(name, record["status"])
PY
```

A client-construction/authentication failure may occur before the runner creates
its output directory; such a failure has no run manifest. Within the runner,
failed queries leave prior raw results and a failed manifest where possible.
Review unexpected data/property values and errors before retrying in a fresh
directory. Do not loosen metadata checks to bypass a changed dataset.

Then verify locally, without a client or credentials:

```bash
python - <<'PY'
from fly_connectome.vdeltae_spatial_audit import verify_spatial_data_probe

result = verify_spatial_data_probe("results/vdeltae-hdeltag-spatial-capability-20261029")
print("Saved spatial data availability probe verified:", result["status"])
PY
```

Review the site-availability/property/category tables and each resource status.
Only after this data availability checkpoint should we design XYZ extraction and
anatomical visualization. Leave the original body annotation audit unchanged.

Continue with the [two-body coordinate and morphology pilot](vdeltae_coordinate_pilot_walkthrough.md),
starting with point-query construction, coordinate validation, geometry parsing
and offline plotting helpers before adding the saved extraction wrapper.
