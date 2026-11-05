"""Compare matched pilot runs using explicit, descriptive failure diagnostics."""

import argparse
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import Audit
from ..__main__ import metrics
from ..data import digest, sample_protocol, write_json


def vector_rmse(prediction, target):
    return float(np.sqrt(np.mean(np.sum((prediction - target) ** 2, axis=1))))


def sustained_underestimate(distance, magnitude, dt, tolerance, minimum=2., hold=.5):
    """First onset of >=hold seconds below (1-tolerance)*truth, beyond minimum mm.

    This measures sustained magnitude underestimation, not an overall accuracy
    limit: overestimation, angle errors and initial transients are separate.
    """
    bad = (distance >= minimum) & (magnitude < (1 - tolerance) * distance)
    count = max(2, int(np.ceil(hold / dt)) + 1)
    if len(bad) < count:
        return None
    starts = np.flatnonzero(np.convolve(bad.astype(int), np.ones(count, dtype=int), mode="valid") == count)
    return float(distance[starts[0]]) if len(starts) else None


def trajectory_summary(row, prediction, target, dt):
    p, y = prediction.astype(float), target.astype(float)
    error = p - y
    result = dict(family=row["family"], kind=row["kind"],
                  vector_rmse_mm=vector_rmse(p, y),
                  component_rmse_mm=np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
                  terminal_error_mm=float(np.linalg.norm(error[-1])),
                  terminal_prediction_mm=p[-1].tolist(), terminal_target_mm=y[-1].tolist())
    if row["kind"] == "long":
        distance, magnitude = np.linalg.norm(y, axis=1), np.linalg.norm(p, axis=1)
        result["long_path"] = dict(
            terminal_true_length_mm=float(distance[-1]), terminal_predicted_length_mm=float(magnitude[-1]),
            underestimate_onset_distance_mm={str(t): sustained_underestimate(distance, magnitude, dt, t)
                                              for t in (.1, .2, .3)},
            late_10s_predicted_magnitude_slope_mm_s=float(np.polyfit(
                np.arange(len(magnitude[-round(10 / dt) - 1:])) * dt,
                magnitude[-round(10 / dt) - 1:], 1)[0]))
    if row["kind"] == "matched_pause":
        j = round(row["suffix_start_s"] / dt)
        delta = p[j:] - p[j]
        result["pause"] = dict(
            movement="backward" if row["protocol"][0][2] < 0 else "forward",
            duration_s=row["protocol"][-1][0] - row["suffix_start_s"],
            onset_error_mm=float(np.linalg.norm(error[j])),
            endpoint_drift_mm=float(np.linalg.norm(delta[-1])),
            endpoint_component_drift_mm=delta[-1].tolist(),
            max_drift_from_onset_mm=float(np.max(np.linalg.norm(delta, axis=1))),
            decoded_path_length_during_pause_mm=float(np.linalg.norm(np.diff(p[j:], axis=0), axis=1).sum()),
            vector_rmse_mm=vector_rmse(p[j:], y[j:]),
            final_second_vector_rmse_mm=vector_rmse(p[-round(1 / dt) - 1:], y[-round(1 / dt) - 1:]))
    if row["family"] == "challenge-world-equivalent":
        j = round(row["protocol"][1][0] / dt)
        result["world_path"] = dict(moving_vector_rmse_mm=vector_rmse(p[:j], y[:j]),
                                      pause_onset_error_mm=float(np.linalg.norm(error[j])),
                                      pause_vector_rmse_mm=vector_rmse(p[j:], y[j:]))
    if row["kind"] == "reversal":
        onset, returned = [round(r[0] / dt) for r in row["protocol"][1:3]]
        local_minimum = onset + int(np.argmin(p[onset:returned + 1, 0]))
        result["reversal"] = dict(
            turnaround_lag_s=float((local_minimum - onset) * dt),
            additional_negative_x_excursion_mm=float(p[onset, 0] - p[local_minimum, 0]),
            error_at_true_return_mm=float(np.linalg.norm(error[returned])),
            final_pause_error_mm=float(np.linalg.norm(error[-1])))
    return result


def replay_long_dynamics(audit):
    """Check hidden-state settling rather than inferring it from decoder saturation."""
    import torch
    from ..__main__ import batch
    from ..model import HomeVectorNetwork
    torch.set_num_threads(1)
    network = HomeVectorNetwork(audit.circuit, seed=audit.manifest["settings"]["seed"])
    checkpoint = torch.load(audit.run / "best.pt", weights_only=True, map_location="cpu")
    network.load_state_dict(checkpoint["model"])
    i = next(i for i, row in enumerate(audit.groups["challenge"]) if row["family"] == "challenge-long-64")
    row = audit.groups["challenge"][i]
    x, _, _ = batch(audit.circuit, [row], audit.task["dt_s"], "cpu", torch.float32)
    with torch.no_grad():
        prediction, rates = network(x, audit.task["dt_s"])
    saved, _ = audit.predictions("challenge")
    replay_error = float(np.max(np.abs(prediction[0].numpy() - saved[i])))
    if replay_error >= .002:
        raise ValueError("CPU long-path replay differs materially from saved predictions.")
    r = rates[0].numpy().astype(float)
    weights = network.weights().detach().numpy().astype(float)
    decoder = network.D.detach().numpy().astype(float)
    tau = network.tau.detach().numpy()[network.type_index.numpy()]
    bias = network.bias.detach().numpy()[network.type_index.numpy()]
    drive = x[0, 0].numpy().astype(float) @ weights[:network.input_size]
    total = bias + drive + r[-1] @ weights[network.input_size:]
    derivative = (-r[-1] + np.maximum(total, 0)) / tau
    jacobian = (-np.eye(network.size) + (total > 0)[:, None] * weights[network.input_size:].T) / tau[:, None]
    spectral_abscissa = float(np.max(np.linalg.eigvals(jacobian).real))
    return dict(
        cpu_vs_saved_max_abs_error_mm=replay_error,
        final_state_norm=float(np.linalg.norm(r[-1])),
        final_state_derivative_norm_per_s=float(np.linalg.norm(derivative)),
        final_decoded_velocity_mm_s=(decoder @ derivative).tolist(),
        state_fractional_change_last10s=float(np.linalg.norm(r[-1] - r[-round(10 / audit.task["dt_s"]) - 1])
                                              / np.linalg.norm(r[-1])),
        final_active_set_continuous_jacobian_largest_real_eigenvalue_per_s=spectral_abscissa,
        local_linear_relaxation_time_s=float(-1 / spectral_abscissa) if spectral_abscissa < 0 else None,
        caveat="Local derivative at the final state; not a proof of global stability or a universal memory time constant.")


def training_distribution(audit):
    distances, durations, moving_intervals = [], [], []
    for row in audit.groups["train"]:
        _, target = sample_protocol(row["protocol"], audit.task["dt_s"])
        distances.append(float(np.max(np.linalg.norm(target, axis=1))))
        durations.append(row["protocol"][-1][0])
        for a, b in zip(row["protocol"][:-1], row["protocol"][1:]):
            if np.linalg.norm(a[2:]) > 0:
                moving_intervals.append(b[0] - a[0])
    return dict(trajectories=len(distances), max_distance_mm_quantiles=dict(
        zip(["median", "p95", "max"], map(float, np.quantile(distances, [.5, .95, 1])))),
        duration_s_max=float(max(durations)), longest_moving_control_interval_s=float(max(moving_intervals)),
        kinds={kind: sum(row["kind"] == kind for row in audit.groups["train"])
               for kind in sorted({row["kind"] for row in audit.groups["train"]})})


def summarize(audit, replay):
    p, y = audit.predictions("challenge")
    rows, dt = audit.groups["challenge"], audit.task["dt_s"]
    saved = json.loads((audit.run / "metrics.json").read_text())
    history = json.loads((audit.run / "history.json").read_text())
    result = dict(selected_update=saved["selected_update"], challenge=[
        trajectory_summary(row, a, b, dt) for row, a, b in zip(rows, p, y)],
        split_metrics={s: {k: saved[s][k] for k in ("normalized_rmse", "cartesian_rmse_mm", "equal_trajectory_scaled_mse")}
                       for s in ("validation", "test", "challenge")},
        validation_history=[dict(update=r["update"], scaled_mse=r["validation_scaled_mse"])
                            for r in history if "validation_scaled_mse" in r], test_groups={})
    tp, ty = audit.predictions("test")
    for kind in sorted({row["kind"] for row in audit.groups["test"]}):
        ix = [i for i, row in enumerate(audit.groups["test"]) if row["kind"] == kind]
        result["test_groups"][kind] = metrics([tp[i] for i in ix], [ty[i] for i in ix])
    indices = [i for i, row in enumerate(rows) if row["family"] == "challenge-world-equivalent"]
    if len(indices) != 2:
        raise ValueError("Require two world-equivalent challenge trajectories.")
    a, b = indices
    np.testing.assert_allclose(y[a], y[b], atol=1e-6)
    separation = np.linalg.norm(p[a].astype(float) - p[b], axis=1)
    moving_end = round(rows[a]["protocol"][1][0] / dt)
    result["world_path_prediction_separation"] = dict(
        moving_rms_mm=float(np.sqrt(np.mean(separation[:moving_end] ** 2))),
        terminal_mm=float(separation[-1]))
    if replay:
        result["long_path_state_diagnostics"] = replay_long_dynamics(audit)
    return result, (p, y)


def validate_challenge_comparison(reference, current, allow_task_differences):
    if not allow_task_differences:
        for name in ("trajectories.json", "task.json"):
            if digest(reference.source / name) != digest(current.source / name):
                raise ValueError("Comparison requires identical task and trajectory files.")
    else:
        if (digest(reference.run / "circuit.json") != digest(current.run / "circuit.json")
                or reference.task["dt_s"] != current.task["dt_s"]
                or reference.task["length_scale_mm"] != current.task["length_scale_mm"]):
            raise ValueError("Different-task challenge comparison requires identical circuits, time steps and loss scales.")
    if reference.groups["challenge"] != current.groups["challenge"]:
        raise ValueError("Challenge protocols must match exactly.")


def plot_comparison(audits, arrays, report):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for model_index, (label, audit, (predictions, targets)) in enumerate(zip(report["model_labels"].values(), audits, arrays)):
        rows, dt = audit.groups["challenge"], audit.task["dt_s"]
        long = next(i for i, row in enumerate(rows) if row["family"] == "challenge-long-64")
        t = np.arange(len(predictions[long])) * dt
        color = f"C{model_index}"
        axes[0, 0].plot(t, np.linalg.norm(predictions[long], axis=1), label=label, color=color)
        if model_index == 0:
            axes[0, 0].plot(t, np.linalg.norm(targets[long], axis=1), "k--", label="True")
        for i, row in enumerate(rows):
            if row["family"] == "challenge-pause-5":
                onset = round(row["suffix_start_s"] / dt)
                time = np.arange(len(predictions[i]) - onset) * dt
                direction = "backward" if row["protocol"][0][2] < 0 else "forward"
                ax = axes[0, 1] if direction == "backward" else axes[0, 2]
                ax.plot(time, np.linalg.norm(predictions[i][onset:].astype(float) - targets[i][onset:], axis=1), label=label, color=color)
            if row["kind"] == "closed_loop":
                time = np.arange(len(predictions[i])) * dt
                for j, component in enumerate(("x", "y")):
                    axes[1, j].plot(time, predictions[i][:, j] - targets[i][:, j], label=label, color=color)
            if row["family"] == "challenge-world-equivalent":
                time = np.arange(len(predictions[i])) * dt
                axes[1, 2].plot(time, np.linalg.norm(predictions[i].astype(float) - targets[i], axis=1),
                                 label=f"{label}, {row['kind']}", color=color,
                                 linestyle="-" if row["kind"] == "forward" else "--")
    titles = ["Long straight path", "Pause after backward walking", "Pause after forward walking",
              "Square-loop x error", "Square-loop y error", "Same world path: vector errors"]
    labels = ["Predicted magnitude (mm)", "Vector error (mm)", "Vector error (mm)",
              "Signed component error (mm)", "Signed component error (mm)", "Vector error (mm)"]
    for ax, title, ylabel in zip(axes.flat, titles, labels):
        ax.set(title=title, xlabel="Time (s)", ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0, 1].set_xlabel("Time since pause onset (s)")
    axes[0, 2].set_xlabel("Time since pause onset (s)")
    fig.savefig(audits[1].out / "comparison.png", dpi=180)
    fig.savefig(audits[1].out / "comparison.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for key, label in report["model_labels"].items():
        h = report[key]["validation_history"]
        ax.plot([r["update"] for r in h], [r["scaled_mse"] for r in h], label=label)
    ax.set(xlabel="Training update", ylabel="Validation equal-trajectory MSE / L²", title="Historical validation score (unweighted)")
    if not report.get("validation_sets_match", not report["task_differences_allowed"]):
        ax.set_title("Different validation sets: compare trends within each curve")
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(audits[1].out / "validation_history.png", dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("run", "source", "reference-run", "reference-source", "output"):
        parser.add_argument(f"--{option}", required=True)
    parser.add_argument("--replay-dynamics", action="store_true", help="Also replay the two long challenges on CPU")
    parser.add_argument("--allow-task-differences", action="store_true",
                        help="Allow different datasets, requiring identical circuits, dt, loss scale and challenge protocols")
    parser.add_argument("--reference-label", default="Original")
    parser.add_argument("--label", default="+hDeltaC")
    args = parser.parse_args(argv)
    args.command = "comparison"
    reference_args = copy.copy(args)
    reference_args.run, reference_args.source = args.reference_run, args.reference_source
    reference, current = Audit(reference_args), Audit(args)
    validate_challenge_comparison(reference, current, args.allow_task_differences)
    with current:
        current.record["reference_source_sha256"] = reference.hashes
        a, ap = summarize(reference, args.replay_dynamics)
        b, bp = summarize(current, args.replay_dynamics)
        keys = ("reference", "current") if args.allow_task_differences else ("original", "hdeltac")
        report = dict(zip(keys, (a, b)))
        report.update(model_labels=dict(zip(keys, (args.reference_label, args.label))),
                      task_differences_allowed=args.allow_task_differences,
                      validation_sets_match=reference.groups["validation"] == current.groups["validation"],
                      training_distribution=training_distribution(reference),
                      definitions=dict(
                          exploratory="Thresholds selected descriptively after inspecting these pilots; no preregistered gate or seed uncertainty claim.",
                          distance="First >=0.5 s of magnitude underestimation beyond 2 mm, at 10/20/30 percent tolerances; not a longest fully accurate distance.",
                          drift="Distance between pause-end and pause-onset predictions; target error is reported separately.",
                          reversal="Delay to the minimum predicted x within the backward-walk interval after the 2 s switch; not a fit to a pure delay model.",
                          cross_task="Only identical challenge protocols are directly matched; aggregate validation/test scores use different datasets when task differences are allowed."))
        write_json(current.out / "comparison.json", report)
        plot_comparison((reference, current), (ap, bp), report)
        after = {str(p): digest(p) for folder in (reference.run, reference.source)
                 for p in folder.iterdir() if p.is_file()}
        if after != reference.hashes:
            raise ValueError("Reference inputs changed during comparison.")
    print(json.dumps(dict(output=str(current.out), status="complete")))


if __name__ == "__main__":
    main()
