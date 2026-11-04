"""Frozen-checkpoint straight walks across speed, heading and body direction."""

import argparse
import copy
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import Audit
from .compare import sustained_underestimate
from ..data import digest, encode, write_json


def conditions(speeds, headings, directions):
    rows = []
    for speed in speeds:
        for heading in headings:
            for direction in directions:
                rows.append(dict(family="factorial", speed_mm_s=float(speed),
                                 heading_deg=float(heading), body_direction_deg=float(direction)))
        # Keep the world path at 45 degrees while changing the body motion.
        for direction in directions:
            rows.append(dict(family="same_world", speed_mm_s=float(speed),
                             heading_deg=float((45 - direction) % 360), body_direction_deg=float(direction)))
    return rows


def controls_and_velocity(rows):
    speed = np.array([r["speed_mm_s"] for r in rows])
    body = np.deg2rad([r["body_direction_deg"] for r in rows])
    heading = np.array([r["heading_deg"] for r in rows])
    controls = np.column_stack((heading, speed * np.cos(body), speed * np.sin(body)))
    world = np.deg2rad(heading) + body
    velocity = speed[:, None] * np.column_stack((np.cos(world), np.sin(world)))
    return controls, velocity


def constant_rollout(network, pfn, dt, duration):
    """Batched NumPy float32 replay of the model's held-input update, without a graph.

    Store decoded vectors and state norms, plus the state ten seconds before
    the endpoint; avoid retaining every neuron's full trace for every trial.
    """
    weight = network.weights().detach().cpu().numpy()
    index = network.type_index.cpu().numpy()
    bias = network.bias.detach().cpu().numpy()[index]
    tau = network.tau.detach().cpu().numpy()[index]
    decoder = network.D.detach().cpu().numpy()
    intercept = network.d.detach().cpu().numpy()
    drive = np.asarray(pfn, dtype=np.float32) @ weight[:network.input_size]
    alpha = -np.expm1(-np.float32(dt) / tau)
    recurrent = weight[network.input_size:]
    count = round(duration / dt) + 1
    state = np.zeros((len(pfn), network.size), dtype=np.float32)
    prediction = np.empty((len(pfn), count, 2), dtype=np.float32)
    norms = np.empty((len(pfn), count), dtype=np.float32)
    late_index = max(0, count - 1 - round(10 / dt))
    for tick in range(count):
        prediction[:, tick] = state @ decoder.T + intercept
        norms[:, tick] = np.linalg.norm(state, axis=1)
        if tick == late_index:
            late_state = state.copy()
        if tick + 1 < count:
            state += alpha * (np.maximum(0, bias + drive + state @ recurrent) - state)
    if not np.isfinite(prediction).all() or not np.isfinite(norms).all():
        raise ValueError("Nonfinite frozen rollout.")
    derivative = (-state + np.maximum(0, bias + drive + state @ recurrent)) / tau
    diagnostics = dict(
        state_fractional_change_last10s=np.linalg.norm(state - late_state, axis=1)
        / np.maximum(np.linalg.norm(state, axis=1), np.finfo(np.float32).tiny),
        final_decoded_velocity=derivative @ decoder.T)
    return prediction, norms, diagnostics


def summary(row, prediction, velocity, dt):
    time = np.arange(len(prediction)) * dt
    distance = row["speed_mm_s"] * time
    target = -time[:, None] * velocity
    unit = -velocity / row["speed_mm_s"]
    along = prediction @ unit
    error = np.linalg.norm(prediction - target, axis=1)
    magnitude = np.linalg.norm(prediction, axis=1)
    onset = sustained_underestimate(distance, magnitude, dt, .1)
    # Reuse the sustained-threshold operator on full vector error, including
    # transverse error and overestimation, rather than magnitude alone.
    vector_onset = sustained_underestimate(distance, distance - error, dt, .1)
    angle = None if magnitude[-1] < 1e-8 else float(np.rad2deg(np.arccos(
        np.clip(along[-1] / magnitude[-1], -1, 1))))
    late = max(0, len(time) - 1 - round(10 / dt))
    return dict(row, duration_s=float(time[-1]), distance_mm=float(distance[-1]),
                final_vector_error_mm=float(error[-1]), final_magnitude_mm=float(magnitude[-1]),
                final_along_track_mm=float(along[-1]), final_direction_error_deg=angle,
                final_gain=float(along[-1] / distance[-1]),
                final_transverse_error_mm=float(np.linalg.norm(prediction[-1] - along[-1] * unit)),
                vector_rmse_mm=float(np.sqrt(np.mean(error ** 2))),
                underestimate_10pct_distance_mm=onset,
                underestimate_10pct_time_s=None if onset is None else onset / row["speed_mm_s"],
                vector_error_10pct_distance_mm=vector_onset,
                vector_error_10pct_time_s=None if vector_onset is None else vector_onset / row["speed_mm_s"],
                late_along_track_gain=float(np.polyfit(time[late:], along[late:], 1)[0] / row["speed_mm_s"]))


def replay(audit, rows, duration):
    import torch
    from ..model import HomeVectorNetwork
    torch.set_num_threads(1)
    network = HomeVectorNetwork(audit.circuit, seed=audit.manifest["settings"]["seed"])
    network.load_state_dict(torch.load(audit.run / "best.pt", map_location="cpu", weights_only=True)["model"])
    network.eval()
    controls, velocity = controls_and_velocity(rows)
    dt = audit.task["dt_s"]
    pfn = encode(audit.circuit, controls)
    prediction, norms, diagnostics = constant_rollout(network, pfn, dt, duration)
    # Verify the optimized replay against the saved 64-second challenge, not
    # merely a second invocation of the same integrator.
    saved, _ = audit.predictions("challenge")
    long = next(i for i, r in enumerate(audit.groups["challenge"])
                if r["family"] == "challenge-long-64")
    protocol = audit.groups["challenge"][long]["protocol"]
    check_pfn = encode(audit.circuit, [protocol[0][1:]])
    check, _, _ = constant_rollout(network, check_pfn, dt, protocol[-1][0])
    difference = float(np.max(np.abs(check[0] - saved[long])))
    if difference >= .002:
        raise ValueError(f"Frozen replay differs from saved predictions by {difference} mm.")
    results = []
    for i, row in enumerate(rows):
        item = summary(row, prediction[i].astype(float), velocity[i], dt)
        item["state_fractional_change_last10s"] = float(diagnostics["state_fractional_change_last10s"][i])
        item["final_decoded_speed_mm_s"] = float(np.linalg.norm(diagnostics["final_decoded_velocity"][i]))
        results.append(item)
    return results, prediction, norms, difference


def plots(output, rows, predictions, dt):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    colors = plt.get_cmap("tab10")
    speeds = sorted({r["speed_mm_s"] for r in rows})
    comparison_speed = min(speeds, key=lambda speed: abs(speed - 1))
    time = np.arange(predictions[0].shape[1]) * dt
    for model, prediction in enumerate(predictions):
        for j, speed in enumerate(speeds):
            i = next(i for i, r in enumerate(rows) if r["family"] == "factorial"
                     and r["heading_deg"] == 45 and r["body_direction_deg"] == 0 and r["speed_mm_s"] == speed)
            along = prediction[i] @ (-np.ones(2) / np.sqrt(2))
            style = "-" if model == 0 else "--"
            label = f"{'Original' if model == 0 else '+hDeltaC'}, {speed:g} mm/s"
            axes[0, 0].plot(time, along, style, color=colors(j), label=label)
            axes[0, 1].plot(time * speed, along, style, color=colors(j), label=label)
        for family, ax in [("same_world", axes[1, 0]), ("factorial", axes[1, 1])]:
            selected = [i for i, r in enumerate(rows) if r["family"] == family and r["speed_mm_s"] == comparison_speed
                        and (family == "same_world" or r["body_direction_deg"] == 0)]
            x = [rows[i]["body_direction_deg" if family == "same_world" else "heading_deg"] for i in selected]
            endpoint = prediction[selected, -1].astype(float)
            _, velocity = controls_and_velocity([rows[i] for i in selected])
            ax.plot(x, np.linalg.norm(endpoint + time[-1] * velocity, axis=1), "o-",
                    label="Original" if model == 0 else "+hDeltaC")
    axes[0, 0].set(title="Forward walking, heading 45°", xlabel="Elapsed time (s)", ylabel="Along-track prediction (mm)")
    axes[0, 1].set(title="Same traces against travelled distance", xlabel="True distance (mm)", ylabel="Along-track prediction (mm)")
    axes[0, 1].plot([0, max(speeds) * time[-1]], [0, max(speeds) * time[-1]], "k:", label="Ideal")
    axes[1, 0].set(title=f"Same world path, {comparison_speed:g} mm/s", xlabel="Body walking direction (degrees)", ylabel=f"{time[-1]:g}-second vector error (mm)")
    axes[1, 1].set(title=f"Forward walking, {comparison_speed:g} mm/s", xlabel="Heading (degrees)", ylabel=f"{time[-1]:g}-second vector error (mm)")
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.savefig(output / "straight_sweep.png", dpi=180)
    fig.savefig(output / "straight_sweep.pdf")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("run", "source", "reference-run", "reference-source", "output"):
        parser.add_argument(f"--{option}", required=True)
    parser.add_argument("--duration", type=float, default=64.)
    parser.add_argument("--speeds", type=float, nargs="+", default=[.25, .5, 1., 2.])
    args = parser.parse_args(argv)
    if (not np.isfinite(args.duration) or args.duration < 10 or
            not np.isfinite(args.speeds).all() or min(args.speeds) <= 0 or len(set(args.speeds)) != len(args.speeds)):
        parser.error("Require duration >=10 s and distinct finite positive speeds.")
    args.command = "straight"
    reference_args = copy.copy(args)
    reference_args.run, reference_args.source = args.reference_run, args.reference_source
    reference, current = Audit(reference_args), Audit(args)
    for name in ("task.json", "trajectories.json"):
        if digest(reference.source / name) != digest(current.source / name):
            raise ValueError("Require matched task and trajectory bundles.")
    dt = current.task["dt_s"]
    if not np.isclose(args.duration / dt, round(args.duration / dt), rtol=0, atol=1e-7):
        parser.error("Duration must fall on the task time grid.")
    rows = conditions(args.speeds, range(0, 360, 45), [0, 45, 90, 127, 135, 180, 225, 233, 270, 315])
    with current:
        current.record["reference_source_sha256"] = reference.hashes
        current.record["backend"] = "NumPy float32 CPU; trained Torch weights; unchanged model update"
        arrays, report, table = [], {}, []
        for label, audit in [("original", reference), ("hdeltac", current)]:
            print(f"Replaying {label}: {len(rows)} straight trajectories", flush=True)
            results, prediction, norms, difference = replay(audit, rows, args.duration)
            report[label] = dict(saved_long_replay_max_abs_error_mm=difference, trajectories=results)
            table.extend(dict(model=label, **r) for r in results)
            arrays.append(prediction)
            np.savez_compressed(current.out / f"{label}_traces.npz", prediction=prediction, state_norm=norms)
        report["conditions"] = rows
        report["definitions"] = dict(
            thresholds="First >=0.5 s breach beyond 2 mm, at 10%; null means not observed within the trial, not indefinitely accurate.",
            angle="Body direction is atan2(right, forward); heading + body direction gives world movement direction.",
            initialization="Zero dynamic rates per trajectory; no fitting or parameter changes.",
            scope="One checkpoint per architecture; descriptive grid, not seed uncertainty or a causal attribution to hDeltaC.",
            late_gain="Slope of signed along-track prediction over final 10 s divided by walking speed; ideal is 1.")
        write_json(current.out / "straight_sweep.json", report)
        with (current.out / "straight_sweep.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        plots(current.out, rows, arrays, dt)
        after = {str(p): digest(p) for folder in (reference.run, reference.source)
                 for p in folder.iterdir() if p.is_file()}
        if after != reference.hashes:
            raise ValueError("Reference inputs changed during sweep.")
    print(f"Completed: {current.out}")


if __name__ == "__main__":
    main()
