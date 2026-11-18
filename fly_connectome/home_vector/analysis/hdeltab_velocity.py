"""Probe current world-direction tuning and history dependence in frozen hDeltaB."""

import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ..__main__ import begin, fail, finish
from ..data import digest, encode, verify_bundle, write_json
from ..model import HomeVectorNetwork
from .sugar_retrained_loops import load_selected_models


TRAINING = "results/cx-sugar-reliable-return-training-seed0-20261113"
LEGACY = "results/pfn-hdeltab-movement-groupnorm-pfnd6-v1.0"
MAP = "results/pfn-hdeltab-movement-groupnorm-pfnd6-decoder-v1.0/hDeltaB_decoder_map.csv"
HISTORY = "results/cx-standard-displacement-retention-20261117"
WINDOWS = {"early": (0.8, 1.0), "late": (2.8, 3.0)}
CONTROL_COLUMNS = ["heading_deg", "forward_mm_s", "right_mm_s"]


def rollout(network, circuit, controls, dt, duration, initial=None):
    encoded = encode(circuit, controls).astype(np.float32)
    ticks = round(duration / dt) + 1
    inputs = torch.from_numpy(np.repeat(encoded[:, None, :], ticks, axis=1))
    with torch.no_grad():
        _, rates = network(inputs, dt, initial=initial)
    return rates.numpy()


def wrap_degrees(angle):
    return (np.asarray(angle) + 180) % 360 - 180


def decode_profiles(profiles, mapping):
    """Average within the published anatomical positions before vector averaging."""
    values = np.asarray(profiles)
    bins = np.column_stack([values[:, mapping.position.eq(k)].mean(axis=1) for k in range(14)])
    angles = mapping.groupby("position").published_angle_deg.first().to_numpy()
    vector = bins @ np.exp(1j * np.deg2rad(angles))
    total = bins.sum(axis=1)
    valid = (total > 1e-12) & (np.abs(vector) > 1e-12 * np.maximum(total, 1.))
    return pd.DataFrame(dict(published_decoded_deg=np.where(valid, np.rad2deg(np.angle(vector)), np.nan),
                             resultant_length=np.divide(np.abs(vector), total, out=np.zeros_like(total), where=total > 0),
                             valid=valid), index=profiles.index)


def pattern_comparison(profiles, grid):
    values = np.asarray(profiles, dtype=float)
    centered = values - values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    valid = norms > 1e-12
    unit = np.divide(centered, norms[:, None], out=np.zeros_like(centered), where=valid[:, None])
    similarity = np.clip(unit @ unit.T, -1, 1)
    similarity[~(valid[:, None] & valid[None, :])] = np.nan
    heading, world = grid.heading_deg.to_numpy(), grid.world_direction_deg.to_numpy()
    triangle = np.triu(np.isfinite(similarity), 1)
    same = triangle & (world[:, None] == world) & (heading[:, None] != heading)
    different = triangle & (world[:, None] != world) & (heading[:, None] == heading)
    a = float(similarity[same].mean()) if same.any() else None
    b = float(similarity[different].mean()) if different.any() else None
    return dict(same_world_mean_centered_cosine=a, different_world_mean_centered_cosine=b,
                contrast=a - b if a is not None and b is not None else None,
                same_world_valid_pairs=int(same.sum()), different_world_valid_pairs=int(different.sum())), similarity


def measure(profiles, grid, mapping):
    phase = decode_profiles(profiles, mapping)
    world_error = wrap_degrees(phase.published_decoded_deg.to_numpy() - grid.world_direction_deg)
    heading_error = wrap_degrees(phase.published_decoded_deg.to_numpy() - grid.heading_deg)
    metrics, similarity = pattern_comparison(profiles, grid)
    valid = phase.valid.to_numpy()
    metrics.update(
        world_angle_mae_deg=float(np.abs(world_error[valid]).mean()) if valid.any() else None,
        heading_angle_mae_deg=float(np.abs(heading_error[valid]).mean()) if valid.any() else None,
        median_resultant_length=float(phase.resultant_length.median()),
    )
    details = phase.assign(world_error_deg=world_error, heading_error_deg=heading_error)
    return metrics, details, similarity


def draw(output, grid, reference, current, reference_similarity, current_similarity, history):
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.8), layout="constrained")
    for column, (label, phase, similarity) in enumerate((
        ("Earlier PFN → hDeltaB", reference, reference_similarity),
        ("Trained recurrent network", current, current_similarity),
    )):
        ax = axes[0, column]
        for heading, rows in grid.groupby("heading_deg"):
            ax.scatter(rows.world_direction_deg, phase.loc[rows.index].published_decoded_deg % 360,
                       s=16, label=f"{heading:g}°")
        ax.plot([0, 360], [0, 360], "k--", lw=1)
        ax.set(title=label, xlabel="True world travel direction (°)", ylabel="hDeltaB population angle (°)",
               xlim=(-8, 360), ylim=(-8, 360), xticks=[0, 90, 180, 270, 360],
               yticks=[0, 90, 180, 270, 360])
        if column == 1:
            ax.legend(title="Heading", ncol=2, fontsize=7, loc="upper left")
        ax = axes[1, column]
        im = ax.imshow(similarity, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set(title="Matched-rest activity pattern similarity", xlabel="World direction, grouped",
               ylabel="World direction, grouped", xticks=[3.5, 19.5, 35.5, 51.5],
               xticklabels=[0, 90, 180, 270], yticks=[3.5, 19.5, 35.5, 51.5],
               yticklabels=[0, 90, 180, 270])
    fig.colorbar(im, ax=axes[1, :2].tolist(), shrink=.8, label="Centered cosine")
    for name, group in history.groupby("prefix", sort=False):
        if name == "rest":
            continue
        axes[0, 2].plot(group.world_direction_deg, group.phase_shift_deg, "o-", label=name)
        axes[1, 2].plot(group.world_direction_deg, group.relative_pattern_change, "o-")
    axes[0, 2].set(title="Same input after different histories", ylabel="Angle shift from rest history (°)")
    axes[0, 2].legend(fontsize=8)
    axes[1, 2].set(ylabel="Pattern change / rest-history pattern norm")
    for ax in axes[:, 2]:
        ax.set(xlabel="World travel direction (°)", xticks=[0, 90, 180, 270])
        ax.grid(alpha=.2)
    fig.suptitle("hDeltaB direction signal in the feedforward and recurrent models • 0.8–1.0 s into movement")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"hdeltab_direction_probe.{suffix}", dpi=160)
    plt.close(fig)


def draw_slide(output, summary):
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.8, 4.2), layout="constrained")
    labels = ["Earlier\nfeedforward", "Recurrent\n0.8–1.0 s", "Recurrent\n2.8–3.0 s"]
    errors = [summary["reference_raw"]["world_angle_mae_deg"],
              summary["early"]["raw"]["world_angle_mae_deg"],
              summary["late"]["raw"]["world_angle_mae_deg"]]
    bars = left.bar(labels, errors, color=["#8a98a5", "#1765aa", "#1765aa"], width=.62)
    left.bar_label(bars, fmt="%.1f°", padding=4)
    left.set(ylabel="Mean absolute travel-angle error (°)", ylim=(0, 36))

    prefixes = ["forward_4mm", "right_4mm", "closed_square"]
    positions = np.arange(len(prefixes))
    for offset, window, label, color in ((-.18, "early", "0.8–1.0 s", "#1765aa"),
                                         (.18, "late", "2.8–3.0 s", "#7351a4")):
        shifts = [summary["history"][window][name]["median_absolute_angle_shift_deg"] for name in prefixes]
        right.bar(positions + offset, shifts, width=.34, label=label, color=color)
    right.set(xticks=positions, xticklabels=["4 mm forward", "4 mm rightward", "Closed square"],
              ylabel="Median angle shift vs. stationary history (°)", ylim=(0, 104))
    right.legend(frameon=False)
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e3e7ec")
        ax.set_axisbelow(True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / f"hdeltab_direction_slide.{suffix}", dpi=240)
    plt.close(fig)


def analyze(training, legacy, history, output, map_path):
    training, legacy, history, output = (Path(path).resolve() for path in (training, legacy, history, output))
    for source in (training, legacy, history):
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Choose a separate output directory.")
    models, circuit, _, config, selection = load_selected_models(training)
    prior = verify_bundle(history)
    if prior["selection"] != selection["original"]:
        raise ValueError("History states are from another checkpoint.")
    legacy_manifest = json.loads((legacy / "movement_model.json").read_text())
    if not legacy_manifest["data_validated"]:
        raise ValueError("Require verified legacy simulation data.")
    for relative in ("conditions.csv", "hDeltaB_anatomical_order.csv", "baseline_1/steady_activity.csv",
                     "baseline_1/movement_evoked_activity.csv"):
        if digest(legacy / relative) != legacy_manifest["data_sha256"][relative]:
            raise ValueError(f"Legacy activity or metadata changed: {relative}")
    map_path = Path(map_path).resolve()
    mapping = pd.read_csv(map_path)
    inventory = pd.read_csv(legacy / "hDeltaB_anatomical_order.csv")
    recovered = mapping[["bodyId", "instance"]].set_index("bodyId").sort_index()
    pd.testing.assert_frame_equal(recovered, inventory[["bodyId", "instance"]].set_index("bodyId").sort_index())
    if mapping.bodyId.duplicated().any() or set(mapping.position) != set(range(14)):
        raise ValueError("Require 19 unique bodies in the 14 published positions.")
    idx = [circuit.dynamic_ids.index(body) for body in mapping.bodyId]
    if any(circuit.dynamic_types[i] != "hDeltaB" for i in idx) or len(idx) != circuit.dynamic_types.count("hDeltaB"):
        raise ValueError("Checkpoint and reference hDeltaB populations must match exactly.")
    cols = [str(body) for body in mapping.bodyId]
    conditions = pd.read_csv(legacy / "conditions.csv").set_index("trial")
    grid = conditions.loc[conditions.primary_grid].sort_values(["world_direction_deg", "heading_deg"])
    control = grid[CONTROL_COLUMNS].to_numpy()
    rest_control = np.column_stack((control[:, 0], np.zeros((len(control), 2))))
    np.testing.assert_allclose(wrap_degrees(control[:, 0] + np.rad2deg(np.arctan2(control[:, 2], control[:, 1]))
                                                - grid.world_direction_deg.to_numpy()), 0, atol=1e-9)
    if len(grid) != 64 or len(grid[["heading_deg", "world_direction_deg"]].drop_duplicates()) != 64:
        raise ValueError("Require the full set of 64 distinct heading and travel conditions.")
    np.testing.assert_allclose(np.linalg.norm(control[:, 1:], axis=1), 1., atol=1e-12, rtol=0)
    script_hash = digest(Path(__file__))
    source_paths = [training / "manifest.json", training / "original_best.pt", history / "manifest.json",
                    history / "prefixes.npz", map_path, legacy / "movement_model.json", legacy / "conditions.csv",
                    legacy / "hDeltaB_anatomical_order.csv", legacy / "baseline_1/steady_activity.csv",
                    legacy / "baseline_1/movement_evoked_activity.csv"]
    frozen = {str(path): digest(path) for path in source_paths}
    output, manifest = begin(output, "home_vector_hdeltab_velocity_v1", training=str(training),
                             legacy=str(legacy), history=str(history))
    manifest.update(immutable_input_sha256=frozen, selection=selection["original"], training=False,
                    refitting=False, analysis_sha256=script_hash)
    write_json(output / "manifest.json", manifest)
    try:
        shutil.copyfile(__file__, output / Path(__file__).name)
        mapping.to_csv(output / "hdeltab_map.csv", index=False)
        dt = config["dt_s"]
        torch.set_num_threads(1)
        network = HomeVectorNetwork(circuit).eval()
        network.load_state_dict(models["original"].state_dict())
        pre = rollout(network, circuit, rest_control, dt, 0.2)[:, -1].copy()
        rates = rollout(network, circuit, np.concatenate((control, rest_control)), dt, 3.0,
                        torch.tensor(np.tile(pre, (2, 1))))[:, :, idx]
        reference_raw = pd.read_csv(legacy / "baseline_1/steady_activity.csv", index_col=0).loc[grid.index, cols]
        reference_evoked = pd.read_csv(legacy / "baseline_1/movement_evoked_activity.csv", index_col=0).loc[grid.index, cols]
        reference_metrics, reference_phase, _ = measure(reference_raw, grid, mapping)
        ref_evoked, reference_similarity = pattern_comparison(reference_evoked, grid)
        summary = {"reference_raw": reference_metrics, "reference_matched_rest": ref_evoked}
        details = [grid.join(reference_phase).assign(window="reference")]
        for window, (start, stop) in WINDOWS.items():
            raw, stationary = np.split(rates[:, round(start / dt):round(stop / dt)].mean(axis=1, dtype=float), 2)
            metrics, phase, _ = measure(pd.DataFrame(raw, columns=cols, index=grid.index), grid, mapping)
            evoked_metrics, similarity = pattern_comparison(raw - stationary, grid)
            summary[window] = {"raw": metrics, "matched_rest": evoked_metrics}
            details.append(grid.join(phase).assign(window=window))
            if window == "early":
                current_phase, current_similarity = phase, similarity
        pd.concat(details).to_csv(output / "direction_trials.csv")

        fixed = grid.loc[grid.heading_deg.eq(45)].copy()
        names = ["rest", "forward_4mm", "right_4mm", "closed_square"]
        with np.load(history / "prefixes.npz") as saved:
            states = saved["rates"][[saved["names"].tolist().index(name) for name in names], -1]
            terminal = saved["controls"][[saved["names"].tolist().index(name) for name in names], -1]
        np.testing.assert_allclose(terminal, np.tile([45., 0., 0.], (len(names), 1)), atol=0, rtol=0)
        probe = np.r_[fixed[CONTROL_COLUMNS].to_numpy(), [[45., 0., 0.]]]
        history_rates = rollout(network, circuit, np.tile(probe, (len(names), 1)), dt, 3.0,
                                torch.tensor(np.repeat(states, len(probe), axis=0)))[:, :, idx]
        history_details = []
        for window, (start, stop) in WINDOWS.items():
            profile = history_rates[:, round(start / dt):round(stop / dt)].mean(axis=1, dtype=float)
            profile = profile.reshape(len(names), len(probe), len(idx))
            for k, name in enumerate(names):
                raw = profile[k, :-1]
                phase = decode_profiles(pd.DataFrame(raw, columns=cols), mapping)
                if k == 0:
                    rest_angle = phase.published_decoded_deg.to_numpy()
                centered = profile[0, :-1] - profile[0, :-1].mean(axis=1, keepdims=True)
                relative_change = np.linalg.norm(raw - profile[0, :-1], axis=1) / np.linalg.norm(centered, axis=1)
                evoked = raw - profile[k, -1]
                reference = profile[0, :-1] - profile[0, -1]
                history_details.append(pd.DataFrame(dict(prefix=name, window=window,
                    world_direction_deg=fixed.world_direction_deg.to_numpy(),
                    angle_deg=phase.published_decoded_deg.to_numpy(),
                    phase_shift_deg=wrap_degrees(phase.published_decoded_deg.to_numpy() - rest_angle),
                    relative_pattern_change=relative_change,
                    movement_response_change_norm=np.linalg.norm(evoked-reference, axis=1))))
        history_frame = pd.concat(history_details)
        history_frame.to_csv(output / "history_trials.csv", index=False)
        for (window, prefix), frame in history_frame.groupby(["window", "prefix"]):
            summary.setdefault("history", {}).setdefault(window, {})[prefix] = {
                "median_absolute_angle_shift_deg": float(frame.phase_shift_deg.abs().median()),
                "median_relative_pattern_change": float(frame.relative_pattern_change.median()),
                "max_absolute_angle_shift_deg": float(frame.phase_shift_deg.abs().max()),
            }
        np.savez_compressed(output / "responses.npz", rates=rates, history_rates=history_rates,
                            dt_s=dt, body_ids=mapping.bodyId, grid_controls=control,
                            stationary_controls=rest_control, history_probe_controls=probe, prefixes=names)
        draw(output, grid, reference_phase, current_phase, reference_similarity, current_similarity,
             history_frame.loc[history_frame.window.eq("early")])
        draw_slide(output, summary)
        write_json(output / "summary.json", summary)
        if any(digest(path) != expected for path, expected in frozen.items()):
            raise ValueError("An archived input changed.")
        finish(output, manifest)
    except BaseException as error:
        fail(output, manifest, error)
        raise
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", default=TRAINING)
    parser.add_argument("--legacy", default=LEGACY)
    parser.add_argument("--history", default=HISTORY)
    parser.add_argument("--map", default=MAP)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.training, args.legacy, args.history, args.output, args.map), indent=2))


if __name__ == "__main__":
    main()
