import json
from pathlib import Path

import pandas as pd

from .connectivity import ValidationError
from .inputs import sha256
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
