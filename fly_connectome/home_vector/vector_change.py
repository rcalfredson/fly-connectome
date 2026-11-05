"""Optional fixed-window home-vector supervision, with a training-only scale."""

import numpy as np

from .data import sample_protocol
from .loss_balance import trajectory_group


def window_steps(window_s, dt):
    if not np.isfinite(window_s) or window_s <= 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Require a finite positive vector-change window and timestep.")
    ticks = window_s / dt
    if not np.isfinite(ticks) or ticks < 1 or not np.isclose(ticks, round(ticks), rtol=0, atol=1e-7):
        raise ValueError("Vector-change window must be a positive integer number of simulation steps.")
    return int(round(ticks))


def require_windows(records, dt, steps):
    if not records or any(round(row["protocol"][-1][0] / dt) < steps for row in records):
        raise ValueError("Every trajectory must contain at least one full vector-change window.")


def change_plan(weight, window_s, training, dt):
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("Require a finite nonnegative vector-change weight.")
    if not np.isfinite(window_s) or window_s <= 0:
        raise ValueError("Require a finite positive vector-change window.")
    plan = dict(enabled=bool(weight), weight=weight, window_s=window_s)
    if not weight:
        return plan
    steps = window_steps(window_s, dt)
    require_windows(training, dt, steps)
    energy = []
    for row in training:
        _, target = sample_protocol(row["protocol"], dt)
        delta = target[steps:] - target[:-steps]
        energy.append(float(np.mean(np.sum(delta ** 2, axis=-1))))
    mean_energy = float(np.mean(energy))
    if not np.isfinite(mean_energy) or mean_energy <= 0:
        raise ValueError("Require positive training vector-change target energy.")
    plan.update(window_steps=steps, scale_mm=float(np.sqrt(mean_energy)),
                training_target_energy_mm2=mean_energy, training_trajectories=len(training),
                normalization="One frozen RMS target-change scale; training trajectories equally weighted",
                aggregation="All overlapping valid windows, then equal trajectory mean; no absolute group weights",
                checkpoint_selection="Unchanged absolute-vector validation criterion")
    return plan


def change_loss(prediction, target, mask, plan):
    """Both window endpoints retain gradients; masks are valid trajectory prefixes."""
    steps = plan["window_steps"]
    if prediction.shape[1] <= steps:
        raise ValueError("Every trajectory must contain at least one full vector-change window.")
    valid = mask[:, steps:] * mask[:, :-steps]
    if not (valid.sum(1) > 0).all():
        raise ValueError("Every trajectory must contain at least one full vector-change window.")
    predicted_delta = prediction[:, steps:] - prediction[:, :-steps]
    target_delta = target[:, steps:] - target[:, :-steps]
    error = ((predicted_delta - target_delta) / plan["scale_mm"]).square().sum(-1)
    return ((error * valid).sum(1) / valid.sum(1)).mean()


def change_metrics(predictions, targets, records, dt, plan):
    """Diagnostics only: never used to select a checkpoint or derive scales."""
    steps = window_steps(plan["window_s"], dt)
    errors = []
    for prediction, target in zip(predictions, targets):
        if len(target) <= steps:
            raise ValueError("Every trajectory must contain at least one full vector-change window.")
        delta_error = (prediction[steps:] - prediction[:-steps]) - (target[steps:] - target[:-steps])
        errors.append(float(np.mean(np.sum(delta_error ** 2, axis=-1))))

    def summary(values):
        mse = float(np.mean(values))
        return dict(count=len(values), mse_mm2=mse, scaled_mse=mse / plan["scale_mm"] ** 2)

    result = summary(errors)
    result["window_s"] = plan["window_s"]
    result["subsets"] = {name: summary([v for row, v in zip(records, errors) if trajectory_group(row) == name])
                         for name in ("original", "long")
                         if any(trajectory_group(row) == name for row in records)}
    return result
