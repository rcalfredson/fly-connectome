"""Continuous replay of prescribed PFN inputs; geometry is external to the network."""

from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .connectivity import ValidationError
from .movement import (MovementParameters, encode_pfn, movement_weights,
                       _save_figure, solve_output)
from .rate_model import grid_index, relative_error, save_json, sha256, verify_model_data


CONVENTIONS = (
    "Seconds, degrees, mm/s and mm. Heading 0 faces world +x; clockwise positive. "
    "World +y points right/down on the plot. Head and body aligned; forward and right "
    "body velocities positive. vx=vf*cos(theta)-vr*sin(theta); "
    "vy=vf*sin(theta)+vr*cos(theta). Inputs use zero-order hold (no angular or velocity "
    "interpolation) on [time_s[k], time_s[k+1]); boundaries must lie on the dt grid. "
    "At a boundary the new input applies, but the state is the previous interval's "
    "terminal state. The last row is a terminal input sample, with no interval after it. "
    "No extrapolation. Instantaneous turns are prescribed input changes, not turn dynamics."
)


def default_protocol():
    """One uninterrupted sequence; square starts at the original origin at t=5 s."""
    return pd.DataFrame([
        (0., 0., 0., 0., "initial_stop"),
        (.5, 0., 1., 0., "straight"),
        (1.5, 90., 1., 0., "right_angle_turn"),
        (2.5, 90., 0., 1., "sideways"),
        (3.5, 90., -1., 0., "backward_reversal"),
        (4.5, 90., 0., 0., "stop_at_origin"),
        (5., 0., 1., 0., "square_east"),
        (6., 90., 1., 0., "square_south"),
        (7., 180., 1., 0., "square_west"),
        (8., 270., 1., 0., "square_north"),
        (9., 270., 0., 0., "final_stop"),
        (10., 270., 0., 0., "end"),
    ], columns=["time_s", "heading_deg", "forward_mm_s", "right_mm_s", "label"])


def sample_inputs(protocol, dt_s):
    """Validate an explicit timestamped input table and sample its half-open intervals."""
    columns = ["time_s", "heading_deg", "forward_mm_s", "right_mm_s"]
    if not set(columns).issubset(protocol.columns) or len(protocol) < 2:
        raise ValidationError("Trajectory needs at least two rows and time/heading/forward/right columns.")
    values = protocol[columns].to_numpy(dtype=float)
    if not np.isfinite(dt_s) or dt_s <= 0 or not np.isfinite(values).all():
        raise ValidationError("Trajectory inputs must be finite and dt positive.")
    if values[0, 0] != 0 or np.any(np.diff(values[:, 0]) <= 0):
        raise ValidationError("Trajectory timestamps must start at zero and strictly increase.")
    boundaries = np.array([grid_index(t, dt_s) for t in values[:, 0]])
    ticks = np.arange(boundaries[-1] + 1)
    segment = np.searchsorted(boundaries, ticks, side="right") - 1
    return pd.DataFrame(dict(time_s=ticks * dt_s, segment=segment,
                             heading_deg=values[segment, 1], forward_mm_s=values[segment, 2],
                             right_mm_s=values[segment, 3]))


def world_velocity(heading_deg, forward_mm_s, right_mm_s):
    angle = np.deg2rad(heading_deg)
    return np.stack((forward_mm_s * np.cos(angle) - right_mm_s * np.sin(angle),
                     forward_mm_s * np.sin(angle) + right_mm_s * np.cos(angle)), axis=-1)


def geometric_reference(inputs):
    """Exact integral of held velocities, from a zero origin; never fed into PFN/hDeltaB."""
    velocity = world_velocity(inputs.heading_deg, inputs.forward_mm_s, inputs.right_mm_s)
    position = np.vstack((np.zeros(2), np.cumsum(velocity[:-1] * np.diff(inputs.time_s)[:, None], axis=0)))
    return pd.DataFrame(dict(world_vx_mm_s=velocity[:, 0], world_vy_mm_s=velocity[:, 1],
                             displacement_x_mm=position[:, 0], displacement_y_mm=position[:, 1],
                             return_to_origin_x_mm=-position[:, 0], return_to_origin_y_mm=-position[:, 1]))


def solve_continuous(drive, dt_s, tau_s, initial=None):
    """Euler and exact held-drive solutions, sharing the same supplied initial state."""
    drive = np.asarray(drive, dtype=float)
    if (drive.ndim != 2 or len(drive) < 2 or not np.isfinite(drive).all()
            or not np.isfinite([dt_s, tau_s]).all() or min(dt_s, tau_s) <= 0
            or dt_s > tau_s / 20):
        raise ValidationError("Use finite 2D drive and 0 < dt <= tau/20.")
    initial = np.zeros(drive.shape[1]) if initial is None else np.asarray(initial, dtype=float)
    if initial.shape != (drive.shape[1],) or not np.isfinite(initial).all() or (initial < 0).any():
        raise ValidationError("Initial activity must be finite, nonnegative and match targets.")
    activity, exact = np.zeros_like(drive), np.zeros_like(drive)
    activity[0] = exact[0] = initial
    fraction = -np.expm1(-dt_s / tau_s)
    for k in range(len(drive) - 1):
        target = np.maximum(0., drive[k])
        activity[k + 1] = activity[k] + dt_s / tau_s * (target - activity[k])
        exact[k + 1] = exact[k] + fraction * (target - exact[k])
    return activity, exact


def replay(protocol, mapping, weights, parameters, initial=None):
    inputs = sample_inputs(protocol, parameters.dt_s)
    pfn = np.array([encode_pfn(mapping, h, f, r, 1., parameters)
                    for h, f, r in inputs[["heading_deg", "forward_mm_s", "right_mm_s"]].itertuples(index=False, name=None)])
    drive = pfn @ weights
    activity, exact = solve_continuous(drive, parameters.dt_s, parameters.tau_s, initial)
    return inputs, pfn, drive, activity, exact


def render_trajectory(source, output):
    source, output = Path(source), Path(output)
    config = json.loads((source / "trajectory_config.json").read_text())
    table = pd.read_csv(source / "trajectory.csv")
    protocol = pd.read_csv(source / "protocol.csv")
    with np.load(source / "activity.npz") as trace:
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
        try:
            axes[0].plot(table.displacement_x_mm, table.displacement_y_mm, color="0.6")
            points = axes[0].scatter(table.displacement_x_mm[::50], table.displacement_y_mm[::50],
                                     c=table.time_s[::50], s=12, cmap="viridis")
            axes[0].scatter([0], [0], marker="x", c="red", label="Origin; start/end")
            axes[0].invert_yaxis()
            axes[0].set_aspect("equal", adjustable="datalim")
            axes[0].set(xlabel="World x (mm)", ylabel="World y (mm; down)", title="External geometric trajectory")
            axes[0].legend(fontsize=8)
            fig.colorbar(points, ax=axes[0], label="Time (s)")
            axes[1].step(table.time_s, table.heading_deg, where="post")
            axes[1].set_ylabel("Heading (° clockwise)")
            for column in ["forward_mm_s", "right_mm_s", "world_vx_mm_s", "world_vy_mm_s"]:
                axes[2].step(table.time_s, table[column], where="post", label=column)
            axes[2].set(xlabel="Time (s)", ylabel="Velocity (mm/s)")
            axes[2].legend(ncol=4, fontsize=8)
            _save_figure(fig, output / "trajectory_inputs")
        finally:
            plt.close(fig)
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True, constrained_layout=True)
        try:
            for ax, pop in zip(axes, ["PFNd", "PFNv", "hDeltaB"]):
                order = pd.read_csv(source / f"{pop}_anatomical_order.csv")
                ids = config["target_ids"] if pop == "hDeltaB" else config["source_ids"]
                values = trace["hdeltab_activity" if pop == "hDeltaB" else "pfn_activity"]
                values = values[:, [ids.index(int(body)) for body in order.bodyId]].T
                im = ax.imshow(values, extent=[0, table.time_s.iloc[-1], len(order) - .5, -.5],
                               aspect="auto", interpolation="nearest", vmin=0)
                ax.set_yticks(range(len(order)), order.label, fontsize=6)
                ax.set_ylabel(pop)
                for t in protocol.time_s.iloc[1:-1]:
                    ax.axvline(t, color="white", ls="--", lw=.6)
                fig.colorbar(im, ax=ax, label="Raw activity (a.u.; shared scale within population)")
            axes[0].set_title("Prescribed PFN → continuous hDeltaB; b=1, group-normalized PFNd ×6/PFNv ×1\nNo row normalization or stationary subtraction")
            axes[-1].set_xticks(protocol.time_s,
                                [f"{t:g}\n{label}" for t, label in zip(protocol.time_s, protocol.label)],
                                rotation=40, ha="right", fontsize=8)
            axes[-1].set_xlabel("Time (s); interval starts")
            _save_figure(fig, output / "raw_activity")
        finally:
            plt.close(fig)


def run_trajectory(source, output, code, protocol_path=None, dt_s=.001):
    """Reuse a validated selected movement run without modifying any source artifacts."""
    source, output = Path(source), Path(output)
    manifest = json.loads((source / "movement_model.json").read_text())
    if manifest.get("status") != "complete":
        raise ValidationError("Trajectory replay requires a complete movement run.")
    verify_model_data(source, manifest)
    config = json.loads((source / "movement_config.json").read_text())
    p = MovementParameters(**config["parameters"])
    expected = dict(tau_s=.1, dt_s=.001, gain=.001, normalize_incoming_by_group=True,
                    pfnd_weight_multiplier=6., pfnv_weight_multiplier=1.,
                    reference_speed_mm_s=1., velocity_modulation=5.)
    if any(getattr(p, k) != v for k, v in expected.items()) or 1. not in config["baseline_levels"]:
        raise ValidationError("Source must use the selected group-normalized PFNd=6/PFNv=1 reference model.")
    p = replace(p, dt_s=dt_s)
    p.validate()
    mapping = pd.read_csv(source / "PFN_encoding_map.csv")
    id_column = "bodyId" if "bodyId" in mapping else "bodyId_pre"
    mapping = mapping.set_index(id_column).loc[config["source_ids"]].reset_index()
    counts = pd.read_csv(source / "raw_contact_matrix.csv", index_col=0)
    counts.columns = counts.columns.astype(int)
    counts = counts.loc[config["source_ids"], config["target_ids"]]
    weights = movement_weights(counts, mapping, p)
    saved_weights = pd.read_csv(source / "effective_weight_matrix.csv", index_col=0)
    saved_weights.columns = saved_weights.columns.astype(int)
    np.testing.assert_allclose(weights, saved_weights.loc[counts.index, counts.columns], rtol=1e-12, atol=1e-15)
    protocol = pd.read_csv(protocol_path) if protocol_path else default_protocol()
    if "label" not in protocol:
        protocol["label"] = [f"segment_{i}" for i in range(len(protocol))]
    inputs, pfn, drive, activity, exact = replay(protocol, mapping, weights, p)
    geometry = geometric_reference(inputs)
    _, _, _, fine, fine_exact = replay(protocol, mapping, weights, replace(p, dt_s=p.dt_s / 2))
    _, _, _, repeated, _ = replay(protocol, mapping, weights, p)
    validation = dict(relative_exact_error=relative_error(activity, exact),
                      relative_half_dt_error=relative_error(activity, fine[::2]),
                      half_dt_exact_error=relative_error(fine[::2], exact),
                      exact_grid_agreement=relative_error(fine_exact[::2], exact),
                      deterministic_replay=bool(np.array_equal(activity, repeated)))
    # Independently compare the existing fixed-condition solver, with zero initial state.
    old_time, old_activity, _ = solve_output(np.array([1.]), np.array([2.]), p)
    old_drive = np.ones((len(old_time), 1))
    old_drive[grid_index(p.onset_s, p.dt_s):grid_index(p.offset_s, p.dt_s)] += 2
    matched, _ = solve_continuous(old_drive, p.dt_s, p.tau_s)
    validation["existing_solver_max_abs_error"] = float(np.max(np.abs(matched - old_activity)))
    # Analytic polygon vertices, independent of the sampled cumulative integral.
    vertices = np.vstack((np.zeros(2), np.cumsum(
        world_velocity(protocol.heading_deg, protocol.forward_mm_s, protocol.right_mm_s)[:-1]
        * np.diff(protocol.time_s)[:, None], axis=0)))
    boundary_indices = [grid_index(v, p.dt_s) for v in protocol.time_s]
    positions = geometry[["displacement_x_mm", "displacement_y_mm"]].to_numpy()
    validation["geometric_boundary_max_abs_error_mm"] = float(np.max(np.abs(positions[boundary_indices] - vertices)))
    validation["terminal_displacement_mm"] = positions[-1].tolist()
    validation["passed"] = bool(validation["relative_exact_error"] < .02
        and validation["relative_half_dt_error"] < .02
        and validation["half_dt_exact_error"] <= validation["relative_exact_error"] + 1e-12
        and validation["exact_grid_agreement"] < 1e-12
        and validation["deterministic_replay"] and validation["existing_solver_max_abs_error"] < 1e-14
        and validation["geometric_boundary_max_abs_error_mm"] < 1e-10)
    output.mkdir(parents=True, exist_ok=False)
    record = dict(status="running", started_at_utc=datetime.now(timezone.utc).isoformat(), code=code,
                  source_directory=str(source.resolve()), source_sha256=manifest["data_sha256"],
                  code_sha256={f.name: sha256(f) for f in sorted(Path(__file__).parent.glob("*.py"))},
                  python=platform.python_version(), packages={v: version(v) for v in ["numpy", "pandas", "matplotlib"]})
    try:
        save_json(output / "trajectory_config.json", dict(parameters=asdict(p), baseline=1.,
                  source_ids=config["source_ids"], target_ids=config["target_ids"], conventions=CONVENTIONS,
                  initial_state="hDeltaB zero at t=0 only; carried across every interval",
                  encoding=config["encoding"], heading_mapping=config["heading_mapping"],
                  disabled=["recurrence", "noise", "feedback", "plasticity", "decoder", "neural path integration"],
                  activity_reference="Raw only; no fixed-heading rest subtraction or row normalization."))
        protocol.to_csv(output / "protocol.csv", index=False)
        trajectory = pd.concat([inputs, geometry], axis=1)
        trajectory.to_csv(output / "trajectory.csv", index=False)
        boundaries = trajectory.iloc[boundary_indices].copy()
        boundaries["label"] = protocol.label.to_numpy()
        boundaries["raw_hdeltab_mean"] = activity[boundary_indices].mean(axis=1)
        boundaries.to_csv(output / "segment_boundaries.csv", index=False)
        mapping.to_csv(output / "PFN_encoding_map.csv", index=False)
        pd.DataFrame(weights, index=counts.index, columns=counts.columns).to_csv(output / "effective_weight_matrix.csv")
        for pop in ["PFNd", "PFNv", "hDeltaB"]:
            (output / f"{pop}_anatomical_order.csv").write_bytes((source / f"{pop}_anatomical_order.csv").read_bytes())
        np.savez_compressed(output / "activity.npz", time_s=inputs.time_s, pfn_activity=pfn,
                            hdeltab_drive=drive, hdeltab_activity=activity, hdeltab_exact=exact)
        save_json(output / "numerical_validation.json", validation)
        (output / "experiment_notes.md").write_text(NOTES + "\n\n" + CONVENTIONS + "\n")
        if not validation["passed"]:
            raise ValidationError("Trajectory numerical validation failed.")
        render_trajectory(output, output)
        verify_model_data(source, manifest)
        record.update(status="complete", validation=validation, data_validated=True,
                      data_sha256={str(f.relative_to(output)): sha256(f) for f in sorted(output.rglob("*"))
                                   if f.is_file() and f.name != "trajectory_model.json"})
    except Exception as exc:
        record.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(output / "trajectory_model.json", record)
    return record


NOTES = """# Continuous prescribed-PFN trajectory replay

This milestone establishes continuous feedforward replay with explicit movement inputs,
fixed anatomical neuron identities, and state carried through movement and heading changes.
The saved selected model supplies the literature-based heading/translation preferences,
per-target incoming normalization within each PFNd-L/R and PFNv-L/R group, PFNd ×6,
PFNv ×1, common gain 0.001, baseline b=1, tau=0.1 s, and default dt=0.001 s.
PFN rates are prescribed; no recurrence, noise, feedback, or plasticity is present.

The default protocol includes straight travel, a right-angle turn, sideways travel,
backward travel reversing the earlier southward leg, stops, and a unit square.
The first four moving legs close at t=4.5 s; the square runs from t=5 to 9 s
and closes at the same origin. State is never reset during this sequence.

World velocity and its exact held-input integral are external geometry in mm/s and mm.
Return-to-origin is defined separately as minus displacement from the starting origin;
its direction is undefined at the origin. None of these signals enters the neural model,
and none is neural memory. Exactness is relative to the prescribed held-input protocol.

All temporal activity exports are raw arbitrary rate units. Heatmaps use a single scale
over all neurons and times within each population, with no independently peak-normalized
rows. No movement-evoked subtraction is performed: heading-dependent baseline PFN activity
changes during turns, so fixed-heading stationary subtraction would be inappropriate.
A future evoked comparison would require a separate continuous zero-translation replay
with the identical heading history, initial state, and timing, explicitly interpreted as
a counterfactual input decomposition, not a stored displacement signal.

Euler activity is compared with the exact held-drive solution, half-dt replay,
deterministic replay and the existing fixed-trial solver under matched initial conditions.
Tests separately check coordinate signs, known displacements, closure, stationary
equilibrium and segment continuation. Numerical agreement does not validate biology.

The leaky hDeltaB stage follows recent input and relaxes toward heading-dependent rest.
Its decay is not displacement integration. No angular decoder is fitted or applied.
Before neural home-vector modeling, an explicit neural accumulation mechanism, calibrated
coordinate/readout mapping independent of test trajectories, retention and reset rules,
and tests of duration/speed scaling, heading changes, closure and robustness are needed.
Upstream PFN generation, biological time constants, and behavioral homing also remain unvalidated.
"""
