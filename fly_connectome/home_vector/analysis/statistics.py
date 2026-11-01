"""Descriptive pooled-time direction and error statistics, with fixed thresholds."""

import numpy as np

from ..data import write_json


def direction_statistics(predictions, targets):
    pred = np.concatenate(predictions).astype(float)
    target = np.concatenate(targets).astype(float)
    length = np.linalg.norm(target, axis=1)
    valid = length >= 1
    pred, target, length = pred[valid], target[valid], length[valid]
    names = ("median_angle_error_deg", "angle_p90_deg", "fraction_angle_within15deg",
             "parallel_rmse_mm", "perpendicular_rmse_mm", "parallel_fraction_squared_error",
             "fraction_vector_error_over1mm", "fraction_angle_within15deg_among_over1mm_errors")
    result = dict(trajectories=len(predictions), non_near_home_samples=int(valid.sum()))
    if not valid.any():
        return dict(result, **dict.fromkeys(names))
    error = pred - target
    parallel = np.sum(error * (target / length[:, None]), axis=1)
    perpendicular_squared = np.maximum(0, np.sum(error ** 2, axis=1) - parallel ** 2)
    angle = np.rad2deg(np.abs(np.angle(np.exp(1j * (
        np.arctan2(pred[:, 1], pred[:, 0]) - np.arctan2(target[:, 1], target[:, 0]))))))
    angle[np.linalg.norm(pred, axis=1) < 1e-6] = 180
    miss = np.linalg.norm(error, axis=1) > 1
    error_energy = float(np.sum(error ** 2))
    result.update(
        median_angle_error_deg=float(np.median(angle)), angle_p90_deg=float(np.quantile(angle, .9)),
        fraction_angle_within15deg=float(np.mean(angle <= 15)),
        parallel_rmse_mm=float(np.sqrt(np.mean(parallel ** 2))),
        perpendicular_rmse_mm=float(np.sqrt(np.mean(perpendicular_squared))),
        parallel_fraction_squared_error=float(np.sum(parallel ** 2) / error_energy) if error_energy else None,
        fraction_vector_error_over1mm=float(np.mean(miss)),
        fraction_angle_within15deg_among_over1mm_errors=float(np.mean(angle[miss] <= 15)) if miss.any() else None)
    return result


def analyze(audit):
    predictions, targets = audit.predictions("test")
    rows = audit.groups["test"]
    report = {}
    for kind in ["all"] + sorted({row["kind"] for row in rows}):
        indices = [i for i, row in enumerate(rows) if kind == "all" or row["kind"] == kind]
        report[kind] = direction_statistics([predictions[i] for i in indices], [targets[i] for i in indices])
    write_json(audit.out / "statistics.json", dict(
        definitions=dict(
            non_near_home="True home-vector length >=1 mm; predicted length <1e-6 mm assigned 180-degree error.",
            angular_threshold="15 degrees, selected descriptively after training; not a preregistered success gate.",
            miss_threshold="Cartesian vector error >1 mm, descriptive post-hoc subset.",
            parallel="Projection of prediction-minus-target error onto true home-vector unit direction.",
            perpendicular="Residual error orthogonal to true home-vector direction.",
            weighting="Pooled timestamps, not equal trajectories; correlated samples, no independent-sample uncertainty claim."),
        groups=report))
