"""PFNv expansion of a validated PFNd -> hDeltaB extraction."""

from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform

import pandas as pd
from neuprint import NeuronCriteria, fetch_meta

from .anatomy import annotation_order
from .connectivity import (
    DATASET, SERVER, ZERO_MEANING, ValidationError, build_matrix,
    fetch_annotations, fetch_connections, population_summary, type_inventory,
)
from .inputs import save_json, sha256
from .plotting import load_matrix, plot_heatmap


SOURCE_TYPES = ("PFNd", "PFNv")
MAPPING_NOTE = (
    "PB side/glomerulus and FB column are literal instance annotations. PFNv uses "
    "a different mapping from PFNd, including C9 and shared column labels across "
    "different PB glomeruli. No PFNd mapping formula, common spatial coordinate, "
    "or functional angle is assigned to PFNv. All contact regions are pooled."
)


def pfnv_mapping(order):
    """Report observed annotations, without inferring a PB-to-FB mapping rule."""
    rows = []
    for (side, glomerulus), group in order.groupby(["pb_side", "pb_glomerulus"], sort=True):
        rows.append({"pb_side": side, "pb_glomerulus": int(glomerulus),
                     "neuron_count": len(group),
                     "fb_columns": ",".join(str(v) for v in sorted(group.fb_column.unique())),
                     "body_ids": ",".join(str(v) for v in sorted(group.bodyId))})
    return pd.DataFrame(rows)


def match_metadata(actual, expected):
    if not actual.get("uuid") or actual.get("uuid") != expected.get("uuid"):
        raise ValidationError("PFNv expansion dataset UUID differs from the baseline or is missing.")
    for field in ["latestMutationId", "lastDatabaseEdit"]:
        if actual.get(field) != expected.get(field):
            raise ValidationError(f"PFNv expansion dataset {field} differs from the baseline.")


def run_pfn_expansion(client, source, output, code):
    # Local import keeps the extraction module usable by the shared model loader.
    from .rate_model import load_network

    source, output = Path(source), Path(output)
    counts, populations, _, baseline, before = load_network(source)
    if baseline.get("network_kind") == "pfn_feedforward":
        raise ValidationError("Expand the original PFNd extraction, not an already expanded run.")
    metadata = fetch_meta(client=client)
    match_metadata(metadata, baseline.get("dataset_metadata", {}))
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "server": SERVER, "dataset": DATASET, "dataset_metadata": metadata,
        "network_kind": "pfn_feedforward", "code": code,
        "code_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ["neuprint-python", "numpy", "pandas", "matplotlib"]},
        "source_directory": str(source.resolve()), "source_sha256": before,
        "selection": {"source_types": list(SOURCE_TYPES), "target_type": "hDeltaB",
                      "regex": False, "status_filter": None, "cropped_filter": None},
        "connectivity": {"weight_property": "weight", "min_total_weight": 1,
                         "omit_rois": True, "roi_filter": None, "threads": 1},
        "mapping_note": MAPPING_NOTE, "zero_meaning": ZERO_MEANING,
    }
    save_json(output / "metadata.json", manifest)
    try:
        inventory = type_inventory(client)
        related = set(inventory.type.dropna().loc[
            inventory.type.dropna().str.contains("pfnd|pfnv|hdeltab", case=False, regex=True)])
        if related != {"PFNd", "PFNv", "hDeltaB"}:
            raise ValidationError("Unexpected related PFN/hDeltaB type labels; review the type inventory.")
        inventory.to_csv(output / "type_inventory.csv", index=False)
        for label in ["PFNd", "PFNv", "hDeltaB"]:
            live = fetch_annotations(client, NeuronCriteria(type=label, regex=False, client=client))
            if label in populations:
                pd.testing.assert_frame_equal(live.sort_values("bodyId").reset_index(drop=True).convert_dtypes(),
                                              populations[label].sort_values("bodyId").reset_index(drop=True).convert_dtypes(),
                                              check_dtype=False)
            populations[label] = live
        if len(populations["PFNv"]) != 20:
            raise ValidationError("Expected the reviewed 20 PFNv neurons; review a changed population.")
        combined_ids = pd.concat([populations[t].bodyId for t in [*SOURCE_TYPES, "hDeltaB"]])
        if combined_ids.duplicated().any():
            raise ValidationError("Population body IDs overlap.")
        orders = {t: annotation_order(populations[t], t) for t in populations}
        matrices = {}
        for label in SOURCE_TYPES:
            edges = fetch_connections(client, populations[label].bodyId, populations["hDeltaB"].bodyId)
            matrix = build_matrix(edges, populations[label].bodyId, populations["hDeltaB"].bodyId)
            if label == "PFNd":
                pd.testing.assert_frame_equal(matrix, counts, check_names=False)
            matrices[label] = matrix
            edges.to_csv(output / f"{label}_to_hDeltaB_connections.csv", index=False)
            matrix.to_csv(output / f"{label}_to_hDeltaB_matrix.csv")
        combined = pd.concat([matrices[t] for t in SOURCE_TYPES]).sort_index()
        combined.to_csv(output / "PFN_to_hDeltaB_matrix.csv")
        for label, neurons in populations.items():
            neurons.to_csv(output / f"{label}_neurons.csv", index=False)
            orders[label].to_csv(output / f"{label}_anatomical_order.csv", index=False)
        pfnv_mapping(orders["PFNv"]).to_csv(output / "PFNv_annotation_mapping.csv", index=False)
        save_json(output / "population_summary.json", population_summary(populations))
        rows = []
        for label, matrix in matrices.items():
            rows.append({"population": label, "source_neurons": len(matrix),
                         "target_neurons": len(matrix.columns),
                         "positive_pairs": int(matrix.gt(0).to_numpy().sum()),
                         "raw_contact_total": int(matrix.to_numpy().sum()),
                         "disconnected_sources": int(matrix.sum(axis=1).eq(0).sum()),
                         "disconnected_targets": int(matrix.sum(axis=0).eq(0).sum())})
            plot_heatmap(matrix, output / f"{label}_to_hDeltaB_anatomical", label, "hDeltaB",
                         row_order=orders[label].bodyId, column_order=orders["hDeltaB"].bodyId,
                         row_labels=orders[label].set_index("bodyId").label.to_dict(),
                         column_labels=orders["hDeltaB"].set_index("bodyId").label.to_dict(),
                         ordering_label="instance annotation; no angular calibration")
        pd.DataFrame(rows).to_csv(output / "connectivity_summary.csv", index=False)
        match_metadata(fetch_meta(client=client), metadata)
        if any(sha256(source / name) != digest for name, digest in before.items()):
            raise ValidationError("Original PFNd extraction changed during expansion.")
        manifest["validation"] = {"PFNd": len(populations["PFNd"]), "PFNv": len(populations["PFNv"]),
                                  "hDeltaB": len(populations["hDeltaB"]),
                                  "PFNd_annotations_and_weights_unchanged": True,
                                  "hDeltaB_annotations_unchanged": True, "source_unchanged": True,
                                  "dataset_metadata_unchanged": True, "connectivity": rows}
        manifest["data_sha256"] = {p.name: sha256(p) for p in sorted(output.iterdir())
                                   if p.suffix in [".csv", ".json"] and p.name != "metadata.json"}
        manifest["data_validated"] = True
        manifest["artifact_sha256"] = {p.name: sha256(p) for p in sorted(output.iterdir())
                                       if p.is_file() and p.name != "metadata.json"}
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="validation_failed" if isinstance(exc, (ValidationError, AssertionError)) else "failed",
                        error_type=type(exc).__name__)
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(output / "metadata.json", manifest)
    return manifest


def load_pfn_network(source):
    from .rate_model import load_network, verify_model_data

    source = Path(source)
    manifest = json.loads((source / "metadata.json").read_text())
    if manifest.get("network_kind") != "pfn_feedforward":
        raise ValidationError("Use a validated PFNv-expanded extraction for simulate-pfn.")
    verify_model_data(source, manifest)
    counts, populations, orders, metadata, _ = load_network(source)
    populations["PFNv"] = pd.read_csv(source / "PFNv_neurons.csv")
    orders["PFNv"] = annotation_order(populations["PFNv"], "PFNv")
    if len(populations["PFNv"]) != 20:
        raise ValidationError("Expected 20 PFNv neurons.")
    ids = pd.concat([p.bodyId for p in populations.values()])
    if ids.duplicated().any():
        raise ValidationError("Population body IDs overlap.")
    for label in populations:
        pd.testing.assert_frame_equal(pd.read_csv(source / f"{label}_anatomical_order.csv").convert_dtypes(),
                                      orders[label].convert_dtypes(), check_dtype=False)
    pd.testing.assert_frame_equal(pd.read_csv(source / "PFNv_annotation_mapping.csv", dtype={"fb_columns": str, "body_ids": str}), pfnv_mapping(orders["PFNv"]),
                                  check_dtype=False)
    extra = build_matrix(pd.read_csv(source / "PFNv_to_hDeltaB_connections.csv"),
                         populations["PFNv"].bodyId, populations["hDeltaB"].bodyId)
    pd.testing.assert_frame_equal(extra, load_matrix(source / "PFNv_to_hDeltaB_matrix.csv"), check_names=False)
    combined = pd.concat([counts, extra]).sort_index()
    pd.testing.assert_frame_equal(combined, load_matrix(source / "PFN_to_hDeltaB_matrix.csv"), check_names=False)
    hashes = {**manifest["data_sha256"], "metadata.json": sha256(source / "metadata.json")}
    return combined, populations, orders, metadata, hashes


def pfn_experiments(orders):
    from .rate_model import default_experiments

    spec = default_experiments(orders["PFNd"])

    def group(side, glomerulus):
        order = orders["PFNv"]
        ids = sorted(int(v) for v in order.loc[(order.pb_side == side) & (order.pb_glomerulus == glomerulus), "bodyId"])
        if not ids:
            raise ValidationError(f"PFNv stimulus group {side}{glomerulus} is missing.")
        return ids

    def event(side, glomerulus, start=.2, stop=.3):
        return {"body_ids": group(side, glomerulus), "start_s": start, "stop_s": stop,
                "amplitude": 1., "label": f"PFNv {side}{glomerulus}"}

    spec["experiments"].extend([
        {"name": "pfnv_group_pulse", "events": [event("L", 2)]},
        {"name": "pfnv_right_group_pulse", "events": [event("R", 2)]},
        {"name": "pfnv_sustained", "events": [event("L", 2, stop=1.2)]},
        {"name": "pfnv_sequence_forward", "events": [event("L", g, a, b) for g, a, b in
                                                      zip([2, 5, 8], [.2, .3, .4], [.3, .4, .5])]},
        {"name": "pfnv_sequence_reverse", "events": [event("L", g, a, b) for g, a, b in
                                                      zip([8, 5, 2], [.2, .3, .4], [.3, .4, .5])]},
        {"name": "combined_group_pulse", "events": [*spec["experiments"][2]["events"], event("L", 2)]},
    ])
    return spec
