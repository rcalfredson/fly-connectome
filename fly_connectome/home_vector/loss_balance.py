"""Frozen training-only group scales for the optional balanced task loss."""

import numpy as np

from .data import sample_protocol


def trajectory_group(row):
    # Historical challenge kind='long' is not part of the added training group.
    return "long" if row.get("kind") in ("long_training", "long_holdout") else "original"


def loss_plan(mode, training, dt):
    if mode not in ("original", "balanced"):
        raise ValueError("Require original or balanced loss mode.")
    plan = dict(mode=mode, selection_metric="equal_trajectory_scaled_mse")
    if mode == "original":
        return plan
    energies = {name: [] for name in ("original", "long")}
    for row in training:
        _, target = sample_protocol(row["protocol"], dt)
        energies[trajectory_group(row)].append(float(np.mean(np.sum(target ** 2, axis=-1))))
    counts = {name: len(values) for name, values in energies.items()}
    if not all(counts.values()) or counts["original"] != counts["long"]:
        raise ValueError("Balanced loss requires equal nonempty original and long training groups.")
    energy = {name: float(np.mean(values)) for name, values in energies.items()}
    if not all(np.isfinite(v) and v > 0 for v in energy.values()):
        raise ValueError("Balanced loss requires finite positive training target energies.")
    inverse = {name: 1 / value for name, value in energy.items()}
    normalizer = float(np.mean(list(inverse.values())))
    plan.update(selection_metric="balanced_group_scaled_mse", training_counts=counts,
                training_target_energy_mm2=energy,
                training_target_rms_mm={k: float(np.sqrt(v)) for k, v in energy.items()},
                weights={k: v / normalizer for k, v in inverse.items()},
                derivation="Inverse equal-trajectory mean target energy; mean training weight = 1",
                scale_source="Training targets only; frozen before optimization",
                validation_aggregation="Equal mean of weighted original and long group means")
    return plan


def example_weights(records, plan, reference):
    if plan["mode"] == "original":
        return None
    return reference.new_tensor([plan["weights"][trajectory_group(row)] for row in records])


def validation_scores(result, records, scale, plan):
    """Keep historical raw scoring and explicitly separate checkpoint selection."""
    rows = result["per_trajectory"]
    if len(rows) != len(records):
        raise ValueError("Evaluation rows do not match validation trajectories.")
    subsets = {}
    for name in ("original", "long"):
        values = [r["mse_mm2"] / scale ** 2 for r, source in zip(rows, records)
                  if trajectory_group(source) == name]
        if values:
            subsets[name] = dict(count=len(values), scaled_mse=float(np.mean(values)))
    score = result["equal_trajectory_scaled_mse"]
    if plan["mode"] == "balanced":
        if len(subsets) != 2:
            raise ValueError("Balanced validation requires both original and long groups.")
        score = float(np.mean([plan["weights"][k] * v["scaled_mse"] for k, v in subsets.items()]))
    return dict(validation_scaled_mse=result["equal_trajectory_scaled_mse"],
                validation_subsets=subsets, validation_selection_score=score,
                validation_selection_metric=plan["selection_metric"])
