from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform

from neuprint import fetch_meta
import pandas as pd
from requests.exceptions import HTTPError, RequestException, Timeout

from .connectivity import DATASET, SERVER, ValidationError
from .extension_analysis import markdown
from .inputs import integers, save_json, sha256
from .vdeltae_column_audit import verify_audit

SPATIAL_PROBE_KIND = "vdeltae_hdeltag_spatial_capability_v1"
PROBE_COUNT_COLUMNS = (
    "all_contained_sites",
    "sites_marked_FB",
    "sites_with_location",
    "sites_with_confidence",
)


def check_probe_metadata(metadata, inventory):
    for source in inventory["sources"].values():
        expected = source["spatial_metadata"]
        fields = {
            **expected["dataset_identity"],
            **expected["coordinate_metadata"],
            "postHighAccuracyThreshold": expected[
                "recorded_post_high_accuracy_threshold"
            ],
        }
        for field, value in fields.items():
            if value is None or metadata.get(field) != value:
                raise ValidationError(
                    f"Live dataset metadata differs or is missing: {field}."
                )


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
    require_probe_columns(
        properties, ["bodyId", "synapse_type", "property", "sites_with_property"]
    )
    require_probe_columns(
        categories, ["bodyId", "synapse_type", "compartment", "split", "site_count"]
    )

    for table in [annotations, availability, properties, categories]:
        ids = integers(table["bodyId"], "Returned body IDs", 1)
        if not set(ids).issubset(expected_ids):
            raise ValidationError("Spatial data query returned an unselected body.")
    for table in [annotations, availability]:
        if table["bodyId"].duplicated().any() or set(table["bodyId"]) != expected_ids:
            raise ValidationError(
                "Spatial data query is missing or duplicates a selected body."
            )
    columns = ["bodyId", "type", "instance"]
    try:
        pd.testing.assert_frame_equal(
            annotations[columns].sort_values("bodyId").reset_index(drop=True),
            probe[columns].sort_values("bodyId").reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError as error:
        raise ValidationError(
            "Live body annotations differ from the saved audit."
        ) from error

    counts = availability.set_index("bodyId")
    for name in PROBE_COUNT_COLUMNS:
        integers(counts[name], name)
        if counts[name].gt(counts["all_contained_sites"]).any():
            raise ValidationError("Site availability exceeds its all-site denominator.")
    for table, keys, count_column in [
        (properties, ["bodyId", "synapse_type", "property"], "sites_with_property"),
        (
            categories,
            ["bodyId", "synapse_type", "compartment", "split"],
            "site_count",
        ),
    ]:
        integers(table[count_column], count_column, 1)
        if (
            table.duplicated(keys).any()
            or not table["synapse_type"].isin(["pre", "post"]).all()
        ):
            raise ValidationError(
                "Duplicate categories or unexpected synapse polarity; review raw data."
            )
        if properties["property"].isna().any():
            raise ValidationError("A site property name is missing.")
        for name, group in properties.groupby("property"):
            totals = (
                group.groupby("bodyId")["sites_with_property"]
                .sum()
                .reindex(counts.index, fill_value=0)
            )
            if totals.gt(counts["all_contained_sites"]).any():
                raise ValidationError(
                    "Property coverage exceeds its all-site denominator."
                )
        for property_name, count_column in [
            ("location", "sites_with_location"),
            ("confidence", "sites_with_confidence"),
        ]:
            totals = (
                properties.loc[properties["property"].eq(property_name)]
                .groupby("bodyId")["sites_with_property"]
                .sum()
                .reindex(counts.index, fill_value=0)
            )
            if not totals.eq(counts[count_column]).all():
                raise ValidationError(
                    "Property inventory does not reconcile with availability."
                )
        fb_totals = (
            categories.groupby("bodyId")["site_count"]
            .sum()
            .reindex(counts.index, fill_value=0)
        )
        if not fb_totals.eq(counts["sites_marked_FB"]).all():
            raise ValidationError(
                "FB categories do not reconcile with marked-FB site counts."
            )


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
            "file": path.name,
            "bytes": len(data),
        }
    except HTTPError as error:
        code = error.response.status_code if error.response is not None else None
        if code == 404:
            status = "http_not_found"
        elif code in [401, 403]:
            status = "authentication_or_access_error"
        else:
            status = "http_error"
        return {
            "status": status,
            "error_type": type(error).__name__,
            "http_status": code,
        }
    except Timeout as error:
        return {"status": "timeout", "error_type": type(error).__name__}
    except RequestException as error:
        return {"status": "request_error", "error_type": type(error).__name__}
    except Exception as error:
        return {"status": "unexpected_error", "error_type": type(error).__name__}


def run_spatial_data_probe(client, audit_directory, output, *, full_population=False):
    audit_directory = Path(audit_directory).resolve()
    output = Path(output).resolve()
    bodies, inventory = inspect_spatial_sources(audit_directory)
    probe = select_spatial_population(bodies) if full_population else select_spatial_probe(bodies)
    queries = spatial_probe_queries(probe["bodyId"], full_population=full_population)
    source_directories = [audit_directory] + [
        Path(source["source_manifest_file"]).parent
        for source in inventory["sources"].values()
    ]
    for source in source_directories:
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValidationError(
                "Spatial probe output must be separate from its source directories."
            )
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "kind": SPATIAL_PROBE_KIND,
        "full_population": full_population,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "server": SERVER,
        "dataset": DATASET,
        "source_inventory": inventory,
        "selection_rule": ("All audited vDeltaE/hDeltaG bodies; no geometry/connectivity selection."
                           if full_population else "Smallest audited body ID per population; no geometry/connectivity selection."),
        "selected_body_ids": [int(value) for value in probe["bodyId"]],
        "queries": queries,
        "settings": {
            "confidence_filter": None,
            "sites": "All contained sites; FB category query uses s.FB = true.",
            "site_denominator": "Distinct neuron/Synapse-node pairs; not contacts.",
            "skeleton_heal": False,
            "mesh_purpose": "Visualization context only.",
            "anatomical_mapping": "unreviewed",
            "resource_validation": "Payload retrieval only.",
        },
        "resources": {},
    }
    save_json(output / "capability.json", manifest)
    try:
        manifest["code_sha256"] = {
            path.name: sha256(path)
            for path in sorted(Path(__file__).resolve().parent.glob("*.py"))
        }
        manifest["python_version"] = platform.python_version()
        manifest["packages"] = {
            name: version(name)
            for name in ["neuprint-python", "numpy", "pandas", "requests"]
        }
        # Client.server is normalized by neuprint-python; check the actual target too.
        if (
            client.server.rstrip("/") != f"https://{SERVER}"
            or client.dataset != DATASET
        ):
            raise ValidationError(
                "Client target differs from the configured server/dataset."
            )
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
                lambda body_id=body_id: client.fetch_skeleton(
                    body_id, heal=False, format="swc"
                ),
                output / f"{key}.swc",
                text=True,
            )
            save_json(output / "capability.json", manifest)
            if manifest["resources"][key]["status"] not in [
                "retrieved",
                "empty_response",
                "http_not_found",
            ]:
                raise ValidationError(
                    "Optional resource request failed; inspect resource status."
                )
        manifest["resources"]["FB_mesh"] = retrieve_probe_resource(
            lambda: client.fetch_roi_mesh("FB"), output / "FB.obj"
        )
        save_json(output / "capability.json", manifest)
        if manifest["resources"]["FB_mesh"]["status"] not in [
            "retrieved",
            "empty_response",
            "http_not_found",
        ]:
            raise ValidationError(
                "Optional resource request failed; inspect resource status."
            )

        after = fetch_meta(client=client)
        save_json(output / "metadata_after.json", after)
        check_probe_metadata(after, inventory)
        _, final_inventory = inspect_spatial_sources(audit_directory)
        if final_inventory != inventory:
            raise ValidationError(
                "Saved spatial sources changed during spatial data availability probing."
            )
        resource_table = pd.DataFrame(
            [
                {"resource": key, **record}
                for key, record in manifest["resources"].items()
            ]
        )
        (output / "summary.md").write_text(
            "\n".join(
                [
                    "# Spatial data availability probe",
                    "",
                    markdown(probe[["bodyId", "type", "instance"]]),
                    "",
                    "## Site availability",
                    "",
                    markdown(tables["site_availability"]),
                    "",
                    "## Resource responses",
                    "",
                    markdown(resource_table),
                    "",
                    "Column meanings, compartment meanings and axis orientation remain unresolved.",
                    "Resource retrieval does not validate morphology. No confidence filter or neural dynamics applied.",
                    "",
                ]
            )
        )
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error_type"] = type(error).__name__
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["artifact_sha256"] = {
            path.name: sha256(path)
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "capability.json"
        }
        save_json(output / "capability.json", manifest)
    return manifest


def verify_spatial_data_probe(output):
    output = Path(output).resolve()
    manifest = json.loads((output / "capability.json").read_text())
    if (
        manifest.get("kind") != SPATIAL_PROBE_KIND
        or manifest.get("status") != "complete"
    ):
        raise ValidationError(
            "Verification requires a complete spatial data availability probe."
        )
    required = {
        "metadata_before.json",
        "metadata_after.json",
        "probe_bodies.csv",
        "summary.md",
        "body_annotations.csv",
        "site_availability.csv",
        "site_property_inventory.csv",
        "fb_site_categories.csv",
    }
    hashes = manifest.get("artifact_sha256", {})
    if not required.issubset(hashes):
        raise ValidationError("Spatial probe artifact hash inventory is incomplete.")
    for name, digest in hashes.items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or not path.is_file() or sha256(path) != digest:
            raise ValidationError("Spatial probe artifact is missing, outside the run, or changed.")
    probe = pd.read_csv(output / "probe_bodies.csv", dtype={"bodyId": "Int64"})
    full_population = manifest.get("full_population", False)
    queries = spatial_probe_queries(probe["bodyId"], full_population=full_population)
    if manifest.get("selected_body_ids") != probe["bodyId"].tolist():
        raise ValidationError("Saved selection differs from the body table.")
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
    expected = select_spatial_population(bodies) if full_population else select_spatial_probe(bodies)
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

def spatial_metadata_inventory(metadata):
    properties = metadata.get("neuronProperties", {})
    if isinstance(properties, str):
        try:
            properties = json.loads(properties)
        except json.JSONDecodeError as error:
            raise ValidationError(
                "Unrecognized neuronProperties metadata encoding."
            ) from error
    if not isinstance(properties, dict):
        raise ValidationError("Expected a neuronProperties metadata dictionary.")
    roi_info = metadata.get("roiInfo", {})
    if not isinstance(roi_info, dict):
        raise ValidationError("Expected an ROI metadata dictionary.")

    return {
        "dataset_identity": {
            field: metadata.get(field)
            for field in ("uuid", "latestMutationId", "lastDatabaseEdit")
        },
        "coordinate_metadata": {
            field: metadata.get(field) for field in ("voxelSize", "voxelUnits")
        },
        "recorded_post_high_accuracy_threshold": metadata.get(
            "postHighAccuracyThreshold"
        ),
        "fb_roi_names_in_metadata": sorted(
            name for name in roi_info if name == "FB" or name.startswith("FBl")
        ),
        "declared_neuron_properties": {
            name: properties[name]
            for name in ("instance", "type", "somaLocation", "somaSide", "roiInfo")
            if name in properties
        },
        "column_annotation_semantics": "unresolved",
        "anatomical_axis_orientation": "unverified",
        "synapse_schema_and_compartments": "not inspected by this metadata inventory",
        "skeleton_and_mesh_availability_for_selected_bodies": "not checked",
    }


def inspect_spatial_sources(audit_directory):
    audit_directory = Path(audit_directory).resolve()
    manifest_path = audit_directory / "manifest.json"
    before = sha256(manifest_path)
    manifest = verify_audit(audit_directory)
    bodies = pd.read_csv(
        audit_directory / "body_annotations.csv",
        dtype={
            "bodyId": "Int64",
            "annotation_column": "Int64",
            "instance_index": "Int64",
        },
    )
    inventories = {}
    for role, source in manifest["sources"].items():
        source_manifest = json.loads(Path(source["manifest_file"]).read_text())
        inventories[role] = {
            "source_manifest_file": source["manifest_file"],
            "source_manifest_sha256": source["manifest_sha256"],
            "spatial_metadata": spatial_metadata_inventory(
                source_manifest.get("dataset_metadata", {})
            ),
        }

    verify_audit(audit_directory)
    if sha256(manifest_path) != before:
        raise ValidationError("Annotation audit manifest changed during inspection.")
    return bodies, {
        "annotation_audit_directory": str(audit_directory),
        "annotation_audit_manifest_sha256": before,
        "sources": inventories,
    }


def select_spatial_probe(bodies):
    required = {"bodyId", "type", "instance", "annotation_column"}
    if not required.issubset(bodies.columns):
        raise ValidationError(
            "Probe selection needs the audited body annotation fields."
        )
    if bodies["bodyId"].isna().any() or bodies["bodyId"].duplicated().any():
        raise ValidationError("Probe selection requires present, unique body IDs.")
    if not bodies["type"].isin(["vDeltaE", "hDeltaG"]).all():
        raise ValidationError(
            "Probe selection supports only the audited vDeltaE/hDeltaG core."
        )

    rows = []
    for population in ["vDeltaE", "hDeltaG"]:
        selected = bodies.loc[bodies["type"].eq(population)].sort_values("bodyId")
        if selected.empty:
            raise ValidationError("Probe selection needs both audited populations.")
        rows.append(selected.iloc[[0]])
    return pd.concat(rows, ignore_index=True).copy()


def select_spatial_population(bodies):
    """Keep every audited body, including repeated labels and edge columns."""
    select_spatial_probe(bodies)  # Reuse schema, identity and population checks.
    if bodies.groupby("type").size().to_dict() != {"hDeltaG": 8, "vDeltaE": 18}:
        raise ValidationError("Full population requires the audited 18 vDeltaE and eight hDeltaG bodies.")
    integers(bodies["bodyId"], "Population body IDs", 1)
    return pd.concat([
        bodies.loc[bodies["type"].eq(name)].sort_values("bodyId")
        for name in ["vDeltaE", "hDeltaG"]
    ], ignore_index=True).copy()


def spatial_probe_queries(body_ids, *, full_population=False):
    if type(full_population) is not bool:
        raise ValidationError("full_population must be a boolean.")
    ids = integers(pd.Series(list(body_ids), dtype=object), "Probe body IDs", 1)
    if len(ids) != (26 if full_population else 2) or ids.duplicated().any():
        raise ValidationError(
            "Spatial selection requires exactly 26 distinct body IDs." if full_population
            else "Spatial capability probe requires exactly any two distinct body IDs."
        )
    ids = sorted(int(value) for value in ids)
    selection = f"MATCH (n:Neuron) WHERE n.bodyId IN {ids}"
    contained_sites = """
        MATCH (n)-[:Contains]->(:SynapseSet)-[:Contains]->(s:Synapse)
        WITH DISTINCT n, s
    """
    return {
        "body_annotations": selection
        + """
            RETURN n.bodyId AS bodyId, n.type AS type, n.instance AS instance,
                   n.roiInfo AS roiInfo, n.status AS status,
                   n.statusLabel AS statusLabel, n.cropped AS cropped
            ORDER BY bodyId
        """,
        "site_availability": selection
        + """
            OPTIONAL MATCH (n)-[:Contains]->(:SynapseSet)-[:Contains]->(s:Synapse)
            WITH DISTINCT n, s
            RETURN n.bodyId as bodyId,
                   count(s) as all_contained_sites,
                   sum(CASE WHEN s.FB = true THEN 1 ELSE 0 END) AS sites_marked_FB,
                   sum(CASE WHEN s.location IS NOT NULL THEN 1 else 0 END) AS sites_with_location,
                   sum(CASE WHEN s.confidence IS NOT NULL THEN 1 ELSE 0 END) AS sites_with_confidence
            ORDER BY bodyId
        """,
        "site_property_inventory": selection
        + contained_sites
        + """
            UNWIND keys(s) AS property
            RETURN n.bodyId AS bodyId, s.type AS synapse_type,
                   property, count(*) AS sites_with_property
            ORDER BY bodyId, synapse_type, property
        """,
        "fb_site_categories": selection
        + contained_sites
        + """
            WHERE s.FB = true
            RETURN n.bodyId AS bodyId, s.type AS synapse_type,
                   s.compartment AS compartment, s.split AS split,
                   count(*) AS site_count
            ORDER BY bodyId, synapse_type, compartment, split
        """,
    }
