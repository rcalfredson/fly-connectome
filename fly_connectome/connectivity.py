"""Exact population selection and raw directed synapse-count matrices."""

import pandas as pd
from neuprint import NeuronCriteria, fetch_adjacencies, fetch_neurons

SERVER = "neuprint.janelia.org"
DATASET = "male-cns:v1.0"
REFERENCE_PRE = 26913
REFERENCE_POST = 515192
REFERENCE_WEIGHT = 137
EXPECTED_PFND_COUNT = 40
POPULATIONS = ("PFNd", "hDeltaB")
ANNOTATION_COLUMNS = ["bodyId", "type", "instance", "status", "statusLabel", "cropped"]
EDGE_COLUMNS = ["bodyId_pre", "bodyId_post", "weight"]
ZERO_MEANING = (
    "Zero means no connection was reported in the queried dataset; "
    "it does not establish biological absence."
)


class ValidationError(ValueError):
    """The dataset does not match the explicit selection or reference criteria."""


def fetch_annotations(client, criteria):
    return fetch_neurons(
        criteria, omit_rois=True, returned_columns=ANNOTATION_COLUMNS, client=client
    ).sort_values("bodyId").reset_index(drop=True)


def type_inventory(client):
    """Inspect annotations independently of instance naming conventions."""
    return client.fetch_custom(
        "MATCH (n:Neuron) WHERE n.type IS NOT NULL "
        "RETURN n.type AS type, count(n) AS neuron_count ORDER BY type"
    )


def check_type_inventory(inventory):
    # Related labels are diagnostic only; never expand membership with a regex.
    labels = inventory["type"].dropna()
    candidates = labels[labels.str.contains("pfnd|hdeltab", case=False, regex=True)]
    if set(candidates) != set(POPULATIONS):
        raise ValidationError(
            f"Unexpected related type annotations: {sorted(candidates.tolist())}; "
            f"expected exactly {list(POPULATIONS)}. Review annotations before proceeding."
        )


def check_reference_annotations(neurons):
    expected = {REFERENCE_PRE: "PFNd", REFERENCE_POST: "hDeltaB"}
    if neurons["bodyId"].duplicated().any():
        raise ValidationError("Duplicate reference body IDs returned.")
    actual = neurons.set_index("bodyId")["type"].to_dict()
    if actual != expected:
        raise ValidationError(f"Reference annotations differ: expected {expected}, observed {actual}.")


def select_populations(client):
    populations = {}
    for label in POPULATIONS:
        criteria = NeuronCriteria(type=label, regex=False, client=client)
        neurons = fetch_annotations(client, criteria)
        if neurons.empty or set(neurons["type"]) != {label}:
            raise ValidationError(f"Empty or unexpected annotations for {label}.")
        if neurons["bodyId"].isna().any() or neurons["bodyId"].duplicated().any():
            raise ValidationError(f"Missing or duplicate body IDs in {label}.")
        populations[label] = neurons
    for body, label in [(REFERENCE_PRE, "PFNd"), (REFERENCE_POST, "hDeltaB")]:
        if body not in set(populations[label]["bodyId"]):
            raise ValidationError(f"Reference body {body} absent from {label} selection.")
    return populations


def population_summary(populations):
    summary = {}
    for label, neurons in populations.items():
        reconstruction = {}
        for column in ["status", "statusLabel", "cropped"]:
            if column in neurons:
                values = neurons[column].map(lambda v: "unknown" if pd.isna(v) else str(v))
                reconstruction[column] = {str(k): int(v) for k, v in values.value_counts().items()}
            else:
                reconstruction[column] = {"unavailable": len(neurons)}
        summary[label] = {
            "matched_types": sorted(neurons["type"].unique().tolist()),
            "neuron_count": len(neurons),
            "reconstruction": reconstruction,
        }
    summary["PFNd"]["web_count_sanity_check"] = {
        "expected": EXPECTED_PFND_COUNT,
        "matches": len(populations["PFNd"]) == EXPECTED_PFND_COUNT,
    }
    return summary


def fetch_connections(client, source_ids, target_ids):
    _, connections = fetch_adjacencies(
        sources=list(source_ids), targets=list(target_ids),
        min_total_weight=1, omit_rois=True, weight_props=["weight"],
        threads=1, client=client,
    )
    return connections[EDGE_COLUMNS].sort_values(EDGE_COLUMNS[:2]).reset_index(drop=True)


def build_matrix(connections, source_ids, target_ids):
    """Use full population axes, including neurons with no returned connections."""
    sources = pd.Index(sorted(source_ids), name="bodyId_pre")
    targets = pd.Index(sorted(target_ids), name="bodyId_post")
    if sources.empty or targets.empty or not sources.is_unique or not targets.is_unique:
        raise ValidationError("Matrix axes must contain nonempty, unique body IDs.")
    if connections[EDGE_COLUMNS].isna().any().any():
        raise ValidationError("Connection table contains missing values.")
    if connections.duplicated(EDGE_COLUMNS[:2]).any():
        raise ValidationError("Duplicate connection pairs; expected total weights, not ROI rows.")
    if not connections["bodyId_pre"].isin(sources).all() or not connections["bodyId_post"].isin(targets).all():
        raise ValidationError("Returned connection lies outside the selected populations.")
    weights = pd.to_numeric(connections["weight"], errors="raise")
    if ((weights < 0) | (weights % 1 != 0)).any():
        raise ValidationError("Synapse counts must be nonnegative integers.")
    if connections.empty:
        return pd.DataFrame(0, index=sources, columns=targets, dtype="int64")
    matrix = connections.pivot(index="bodyId_pre", columns="bodyId_post", values="weight")
    return matrix.reindex(index=sources, columns=targets).fillna(0).astype("int64")


def reference_result(matrix):
    observed = int(matrix.loc[REFERENCE_PRE, REFERENCE_POST])
    return {
        "bodyId_pre": REFERENCE_PRE, "bodyId_post": REFERENCE_POST,
        "expected_weight": REFERENCE_WEIGHT, "observed_weight": observed,
        "passed": observed == REFERENCE_WEIGHT,
    }


def check_reference_weight(result):
    if not result["passed"]:
        raise ValidationError(
            f"Reference {REFERENCE_PRE} -> {REFERENCE_POST}: expected weight "
            f"{REFERENCE_WEIGHT}, observed {result['observed_weight']}."
        )
