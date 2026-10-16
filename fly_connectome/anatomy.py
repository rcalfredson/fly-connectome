"""Offline annotation ordering, independent of connection weights."""

import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from .connectivity import (
    POPULATIONS, ValidationError, build_matrix, check_reference_annotations,
    check_reference_weight, reference_result,
)
from .plotting import load_matrix, order_matrix, plot_heatmap


COLUMN_CAVEAT = (
    "PFNd C1–C8 and hDeltaB C1–C12 are within-type annotation indices; "
    "they have not been established as directly comparable spatial coordinates."
)


def annotation_order(neurons, neuron_type):
    """Return one row per neuron; body ID only provides a deterministic tie break."""
    patterns = {
        "PFNd": r"PFNd\(PB04\)_([LR])([2-9])_C([1-8])",
        "hDeltaB": r"hDeltaB_(\d{2})_C([1-9]|1[0-2])",
    }
    if neuron_type not in patterns:
        raise ValidationError(f"Unsupported annotation type: {neuron_type}")
    required = {"bodyId", "type", "instance"}
    if not required.issubset(neurons.columns) or neurons.empty:
        raise ValidationError(f"Missing population or required annotations for {neuron_type}.")
    if neurons[list(required)].isna().any().any() or neurons.bodyId.duplicated().any():
        raise ValidationError(f"Missing annotations or duplicate body IDs in {neuron_type}.")
    if set(neurons.type) != {neuron_type}:
        raise ValidationError(f"Unexpected types in {neuron_type} population.")
    rows = []
    for neuron in neurons.itertuples(index=False):
        match = re.fullmatch(patterns[neuron_type], neuron.instance)
        if not match:
            raise ValidationError(f"Unrecognized {neuron_type} instance: {neuron.instance!r}")
        if neuron_type == "PFNd":
            side, glomerulus, column = match.groups()
            glomerulus = int(glomerulus)
            if int(column) != (glomerulus - 1 if side == "L" else 10 - glomerulus):
                raise ValidationError(f"Unexpected PB/FB mapping: {neuron.instance}")
            instance_index = None
        else:
            instance_index, column = map(int, match.groups())
            side, glomerulus = None, None
        rows.append({
            "bodyId": neuron.bodyId, "type": neuron_type, "instance": neuron.instance,
            "fb_column": int(column), "pb_side": side, "pb_glomerulus": glomerulus,
            "instance_index": instance_index, "ordering_source": "instance_annotation",
            "label": f"{neuron.instance} | {neuron.bodyId}",
        })
    table = pd.DataFrame(rows)
    keys = ["fb_column", "pb_side", "bodyId"] if neuron_type == "PFNd" else ["fb_column", "bodyId"]
    table = table.sort_values(keys).reset_index(drop=True)
    for column in ["pb_glomerulus", "instance_index"]:
        table[column] = table[column].astype("Int64")
    table.insert(0, "order", range(1, len(table) + 1))
    return table


def anatomical_heatmaps(source, output):
    """Validate a complete saved extraction and write additional review artifacts."""
    source, output = Path(source).resolve(), Path(output).resolve()
    metadata_path = source / "metadata.json"
    extraction_metadata = json.loads(metadata_path.read_text())
    if extraction_metadata.get("status") != "complete":
        raise ValidationError("Anatomical plotting requires a complete extraction.")
    tables = {}
    inputs = [metadata_path]
    for typ in POPULATIONS:
        path = source / f"{typ}_neurons.csv"
        tables[typ] = annotation_order(pd.read_csv(path), typ)
        inputs.append(path)
    check_reference_annotations(pd.concat(tables.values()).query("bodyId in [26913, 515192]"))
    matrices = {}
    for pre, post in [("PFNd", "hDeltaB"), ("hDeltaB", "PFNd")]:
        stem = f"{pre}_to_{post}"
        matrix_path = source / f"{stem}_matrix.csv"
        edges_path = source / f"{stem}_connections.csv"
        matrix = load_matrix(matrix_path)
        # Reconstruct from the saved edge table to validate every value and axis.
        expected = build_matrix(pd.read_csv(edges_path), tables[pre].bodyId, tables[post].bodyId)
        pd.testing.assert_frame_equal(matrix, expected, check_names=False)
        ordered = order_matrix(matrix, tables[pre].bodyId, tables[post].bodyId)
        pd.testing.assert_frame_equal(
            ordered.reindex(index=matrix.index, columns=matrix.columns), matrix
        )
        matrices[stem] = (pre, post, matrix, ordered)
        inputs.extend([matrix_path, edges_path])
    reference = reference_result(matrices["PFNd_to_hDeltaB"][3])
    check_reference_weight(reference)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running", "source_directory": str(source), "input_sha256": hashes,
        "extraction_metadata": extraction_metadata,
        "ordering_source": "instance_annotation", "column_caveat": COLUMN_CAVEAT,
        "sort_keys": {"PFNd": ["fb_column", "pb_side", "bodyId"],
                      "hDeltaB": ["fb_column", "bodyId"]},
        "tie_break": "Ascending body ID is deterministic, not anatomical.",
        "aggregation": False, "connectivity_used_to_order": False,
        "validation": {"complete_populations": True, "all_weights_unchanged": True,
                       "saved_edges_match_matrices": True, "reference": reference},
    }
    manifest_path = output / "anatomical_ordering.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for typ, table in tables.items():
        table.to_csv(output / f"{typ}_anatomical_order.csv", index=False)
    for stem, (pre, post, matrix, ordered) in matrices.items():
        ordered.to_csv(output / f"{stem}_anatomical_matrix.csv")
        plot_heatmap(
            matrix, output / f"{stem}_anatomical", pre, post,
            row_order=tables[pre].bodyId, column_order=tables[post].bodyId,
            row_labels=tables[pre].set_index("bodyId").label.to_dict(),
            column_labels=tables[post].set_index("bodyId").label.to_dict(),
            row_groups=tables[pre].set_index("bodyId").fb_column.to_dict(),
            column_groups=tables[post].set_index("bodyId").fb_column.to_dict(),
            ordering_label="annotation order",
            note="Column indices are type-specific; shared spatial alignment is unverified.",
        )
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != hashes[p.name] for p in inputs):
        raise ValidationError("Source inputs changed during anatomical plotting.")
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
