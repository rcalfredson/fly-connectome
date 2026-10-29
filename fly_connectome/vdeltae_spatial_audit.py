import json
from pathlib import Path

import pandas as pd

from .connectivity import ValidationError
from .inputs import integers, sha256
from .vdeltae_column_audit import verify_audit


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


def spatial_probe_queries(body_ids):
    ids = integers(pd.Series(list(body_ids), dtype=object), "Probe body IDs", 1)
    if len(ids) != 2 or ids.duplicated().any():
        raise ValidationError(
            "Spatial capability probe requires exactly any two distinct body IDs."
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
