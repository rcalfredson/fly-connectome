"""Measure hDeltaC readout contributions and acute effects on displacement retention."""

import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..__main__ import begin, fail, finish
from ..data import digest, verify_bundle, write_json
from ..model import HomeVectorNetwork
from .sugar_retrained_loops import load_selected_models

TRAINING = "results/cx-sugar-reliable-return-training-seed0-20261113"
RETENTION = "results/cx-standard-displacement-retention-20261117"
BRANCHES = {"path_15": "Repeated circles", "pause_32s": "Pause"}
PAIRS = ("forward_4mm", "right_4mm")
CLAMP_S = 2.


@torch.no_grad()
def acute_replay(network, inputs, initial, dt, mask, release):
    """Clamp hDeltaC through the boundary sample, then release from that state."""
    _, clamped = network(inputs[:, :release + 1], dt, initial=initial, silence_mask=mask)
    _, released = network(inputs[:, release:], dt, initial=clamped[:, -1])
    return torch.cat((clamped[:, :-1], released), dim=1)


def run_branch(source, network, mask, dt):
    with np.load(source) as saved:
        names = saved["prefix_names"].tolist()
        indices = [names.index(name) for name in ("rest", *PAIRS)]
        time = saved["time_s"]
        pfn = saved["pfn"]
        baseline = saved["rates"][indices]
        predicted = saved["prediction"][indices]
        target = saved["target"][indices]
    inputs = torch.from_numpy(pfn).expand(len(indices), -1, -1)
    initial = torch.from_numpy(baseline[:, 0])
    release = round(CLAMP_S / dt)
    np.testing.assert_allclose(time[release], CLAMP_S)
    with torch.no_grad():
        replay_prediction, replay_rates = network(inputs, dt, initial=initial)
    np.testing.assert_allclose(replay_rates.numpy(), baseline, atol=5e-5, rtol=1e-5)
    np.testing.assert_allclose(replay_prediction.numpy(), predicted, atol=5e-5, rtol=1e-5)
    changed = acute_replay(network, inputs, initial, dt, mask, release).numpy()
    assert np.all(changed[:, :release + 1, mask] == 0)
    np.testing.assert_array_equal(changed[:, 0, ~mask], baseline[:, 0, ~mask])
    baseline_delta = baseline[1:].astype(float) - baseline[0]
    changed_delta = changed[1:].astype(float) - changed[0]
    readout = network.D.detach().numpy().T.astype(float)
    direct = baseline_delta[:, :, mask] @ readout[mask]
    others = baseline_delta[:, :, ~mask] @ readout[~mask]
    changed_direct = changed_delta[:, :, mask] @ readout[mask]
    changed_others = changed_delta[:, :, ~mask] @ readout[~mask]
    true = target[1:].astype(float) - target[0]
    np.testing.assert_allclose(true, np.broadcast_to(true[:, :1], true.shape), atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(true, axis=-1), 4., atol=1e-10)
    np.testing.assert_allclose(direct + others, predicted[1:].astype(float) - predicted[0],
                               atol=2e-5, rtol=2e-5)
    unit = true[:, :1] / 4.
    vectors = dict(intact=direct + others, hdeltac_direct=direct, others=others,
                   silenced=changed_direct + changed_others,
                   other_population_change=changed_others - others)
    series = {f"{name}_along_target_mm": (v * unit).sum(axis=-1) for name, v in vectors.items()}
    series["intact_error_mm"] = np.linalg.norm(vectors["intact"] - true, axis=-1)
    series["silenced_error_mm"] = np.linalg.norm(vectors["silenced"] - true, axis=-1)
    series["hdeltac_direct_norm_mm"] = np.linalg.norm(direct, axis=-1)
    series["hdeltac_state_distance"] = np.linalg.norm(baseline_delta[:, :, mask], axis=-1)
    series["other_population_change_norm_mm"] = np.linalg.norm(vectors["other_population_change"], axis=-1)
    return time, series


def figure(output, branches):
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.4), sharex=True, sharey=True,
                             layout="constrained")
    for row, (branch, label) in enumerate(BRANCHES.items()):
        time, data = branches[branch]
        window = time <= 16.
        for col, name in enumerate(PAIRS):
            ax = axes[row, col]
            ax.axvspan(0, CLAMP_S, color="#e1e5eb", alpha=.75, label="Silencing window")
            ax.axhline(4, color=".3", ls=":", label="Correct separation")
            for key, color, style, legend in (("intact", "#1765aa", "-", "Intact"),
                                              ("silenced", "#7351a4", "-", "Brief silencing"),
                                              ("hdeltac_direct", "#e37026", "--", "Direct hDeltaC addend")):
                ax.plot(time[window], data[f"{key}_along_target_mm"][col, window],
                        style, color=color, label=legend)
            direction = "Forward" if name == "forward_4mm" else "Rightward"
            ax.set_title(f"{label} · {direction} history")
            ax.grid(alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(facecolor="white", framealpha=1., edgecolor="white", fontsize=8)
    fig.supylabel("Decoded separation along correct offset (mm)")
    fig.supxlabel("Time under identical inputs (s)")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"hdeltac_retention.{suffix}", dpi=220)
    plt.close(fig)


def analyze(training, retention, output):
    training, retention, output = (Path(p).resolve() for p in (training, retention, output))
    for source in (training, retention):
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Choose a separate output directory.")
    models, circuit, _, config, selection = load_selected_models(training)
    prior = verify_bundle(retention)
    if prior["selection"] != selection["original"]:
        raise ValueError("Archived continuations use another checkpoint.")
    dt = config["dt_s"]
    network = HomeVectorNetwork(circuit).eval()
    network.load_state_dict(models["original"].state_dict())
    mask = torch.tensor([kind == "hDeltaC" for kind in circuit.dynamic_types])
    if not mask.any() or mask.all():
        raise ValueError("Require hDeltaC and other dynamic cells.")
    paths = [training / "manifest.json", training / "original_best.pt", retention / "manifest.json",
             *(retention / f"{branch}.npz" for branch in BRANCHES)]
    frozen = {str(path): digest(path) for path in paths}
    output, manifest = begin(output, "home_vector_hdeltac_retention_v1",
                             training=str(training), retention=str(retention), clamp_duration_s=CLAMP_S)
    manifest.update(immutable_input_sha256=frozen, selection=selection["original"], pairs=list(PAIRS))
    write_json(output / "manifest.json", manifest)
    try:
        torch.set_num_threads(1)
        branches = {}
        rows = []
        for branch in BRANCHES:
            time, series = run_branch(retention / f"{branch}.npz", network, mask, dt)
            branches[branch] = time, series
            np.savez_compressed(output / f"{branch}_metrics.npz", time_s=time, **series)
            for i, name in enumerate(PAIRS):
                for seconds in (0., 2., 4., 8., 16., 32., 128.):
                    if seconds > time[-1]:
                        continue
                    k = round(seconds / dt)
                    rows.append(dict(branch=branch, prefix=name, seconds=seconds,
                                     **{key: float(values[i, k]) for key, values in series.items()}))
        with (output / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        figure(output, branches)
        shutil.copyfile(__file__, output / Path(__file__).name)
        if any(digest(path) != expected for path, expected in frozen.items()):
            raise ValueError("An archived input changed.")
        finish(output, manifest)
    except BaseException as error:
        fail(output, manifest, error)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", default=TRAINING)
    parser.add_argument("--retention", default=RETENTION)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    analyze(args.training, args.retention, args.output)


if __name__ == "__main__":
    main()
