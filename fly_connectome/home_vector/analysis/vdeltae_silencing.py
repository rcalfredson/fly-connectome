"""Compare hDeltaG recovery with and without vDeltaE in a frozen sugar model."""

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
from ..data import digest, encode, sample_protocol, verify_bundle, write_json
from ..sugar_training import sugar_block
from .sugar_response import event_gate
from .sugar_retrained_loops import load_selected_models

TRAINING = "results/cx-sugar-reliable-return-training-seed0-20261113"
LOOPS = "results/cx-sugar-long-loops-20261112"
EVENT_S = 8.
PERIOD_S = 8.


@torch.no_grad()
def replay(model, pfn, gates, dt, silence):
    """Keep selected cells at zero from initialization using the native sugar update."""
    weights = model.weights()
    drives = pfn @ weights[:model.input_size]
    recurrent = weights[model.input_size:]
    bias = model.bias[model.type_index]
    if silence is not None:
        drives = drives.masked_fill(silence, 0.)
        recurrent = recurrent.masked_fill(silence, 0.)
        bias = bias.masked_fill(silence, 0.)
    initial = drives.new_zeros(drives.shape[0], model.size)
    _, rates = sugar_block(initial, drives[:, :-1].unbind(1), gates[:, :-1].unbind(1),
                           recurrent, bias, model.tau[model.type_index],
                           torch.tensor(model._sugar_mask), model.response.relaxation_rate_per_s,
                           model.response.reference_activity, dt)
    return torch.cat((initial[:, None], rates), dim=1)


def scores(relative, mean, dt):
    """Report the raw trajectory and the change relative to matched no sugar."""
    raw = mean[:, 0]
    contrast = raw - mean[:, 1]
    first = np.flatnonzero((relative >= 0) & (relative < PERIOD_S))
    trough = first[np.argmin(raw[:, first], axis=1)]
    result = []
    for i, name in enumerate(("intact", "vDeltaE_silenced")):
        before = (relative >= -1.) & (relative < 0)
        result.append(dict(condition=name, mean_before=float(raw[i, before].mean()),
                           first_loop_minimum=float(raw[i, trough[i]]),
                           minimum_time_s=float(relative[trough[i]]),
                           drop_from_pre_event=float(raw[i, before].mean() - raw[i, trough[i]])))
    step = round(PERIOD_S / dt)
    pre = np.argmin(abs(relative))
    raw_cycles = raw[:, pre:-1].reshape(2, -1, step).mean(axis=-1)
    contrast_cycles = contrast[:, pre:-1].reshape(2, -1, step).mean(axis=-1)
    cycles = []
    for k in range(raw_cycles.shape[1]):
        cycles.append(dict(start_after_sugar_s=k * PERIOD_S, stop_after_sugar_s=(k + 1) * PERIOD_S,
                           intact_mean=float(raw_cycles[0, k]),
                           silenced_mean=float(raw_cycles[1, k]),
                           intact_sugar_minus_no_sugar=float(contrast_cycles[0, k]),
                           silenced_sugar_minus_no_sugar=float(contrast_cycles[1, k])))
    return result, cycles


def plot(output, time, mean, cycles, hg_rates, hg_ids):
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.7), layout="constrained")
    early = (time >= -4) & (time <= 24)
    for i, (label, color, key) in enumerate((("Intact", "#1765aa", "intact"),
                                              ("vDeltaE silenced", "#7351a4", "silenced"))):
        axes[0].plot(time[early], mean[i, 0, early], color=color, label=label)
        axes[0].plot(time[early], mean[i, 1, early], color=color, ls="--", alpha=.7)
        axes[1].plot([r["start_after_sugar_s"] + 4 for r in cycles],
                     [r[f"{key}_mean"] for r in cycles], "o-", color=color, label=label)
        axes[2].plot([r["start_after_sugar_s"] + 4 for r in cycles],
                     [r[f"{key}_sugar_minus_no_sugar"] for r in cycles], "o-", color=color)
    axes[0].axvline(0, color=".35", lw=1)
    axes[0].set(title="Around sugar delivery", xlim=(-4, 24), ylabel="Mean hDeltaG activity (model units)")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=".35", ls="--"))
    axes[0].legend(handles, labels + ["Dashed: no sugar"], frameon=False, fontsize=8)
    axes[1].set(title="Average over each 8 s circle", ylabel="Mean hDeltaG activity (model units)")
    axes[2].axhline(0, color=".35", lw=1)
    axes[2].set(title="Sugar effect relative to no sugar", ylabel="Change in mean activity (model units)")
    for ax in axes:
        ax.set_xlabel("Time from sugar delivery (s)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.2)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"vdeltae_hdeltag_recovery.{suffix}", dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharex=True, sharey=True, layout="constrained")
    for i, (ax, label) in enumerate(zip(axes, ("Intact", "vDeltaE silenced"))):
        ax.plot(time[early], hg_rates[i, 0, early], lw=1)
        ax.axvline(0, color=".35", lw=1)
        ax.set(title=label, xlabel="Time from sugar delivery (s)")
    axes[0].set_ylabel("hDeltaG activity (model units)")
    axes[1].legend(hg_ids, title="Body ID", fontsize=7, ncol=2, frameon=False)
    fig.savefig(output / "hdeltag_cell_traces.png", dpi=200)
    plt.close(fig)


def analyze(training, loops, output):
    training, loops, output = (Path(p).resolve() for p in (training, loops, output))
    for source in (training, loops):
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Choose a separate output directory.")
    models, circuit, response, config, selection = load_selected_models(training)
    verify_bundle(loops)
    inputs = [training / "manifest.json", training / "sugar_best.pt", loops / "manifest.json",
              loops / "protocols.json"]
    frozen = {str(p): digest(p) for p in inputs}
    protocol = json.loads((loops / "protocols.json").read_text())["path_08"]["protocol"]
    dt = config["dt_s"]
    controls, _ = sample_protocol(protocol, dt)
    time = np.arange(len(controls)) * dt
    np.testing.assert_allclose(time[-1], 128.)
    samples_per_loop = round(PERIOD_S / dt)
    loop_controls = controls[:-1].reshape(-1, samples_per_loop, 3)
    np.testing.assert_allclose(loop_controls, np.broadcast_to(loop_controls[0], loop_controls.shape),
                               atol=1e-10, rtol=0)
    pfn = torch.from_numpy(encode(circuit, controls).astype(np.float32)).expand(2, -1, -1)
    gate = event_gate(time, [EVENT_S], response)
    gates = torch.from_numpy(np.stack((gate, np.zeros_like(gate))).astype(np.float32))
    vd = torch.tensor([kind == "vDeltaE" for kind in circuit.dynamic_types])
    hg = torch.tensor([kind == "hDeltaG" for kind in circuit.dynamic_types])
    if not vd.any() or not hg.any():
        raise ValueError("Require vDeltaE and hDeltaG cells.")
    output, manifest = begin(output, "home_vector_vdeltae_silencing_v1", training=str(training),
                             movement_source=str(loops), protocol="path_08", sugar_event_s=EVENT_S)
    manifest.update(immutable_input_sha256=frozen, selection=selection["sugar"],
                    condition_order=["intact", "vDeltaE_silenced"], event_order=["sugar", "no_sugar"])
    write_json(output / "manifest.json", manifest)
    try:
        torch.set_num_threads(1)
        model = models["sugar"]
        intact = replay(model, pfn, gates, dt, None)
        silenced = replay(model, pfn, gates, dt, vd)
        with torch.no_grad():
            n = round(10 / dt) + 1
            _, reference = model(pfn[:, :n], dt, gates[:, :n])
        torch.testing.assert_close(intact[:, :n], reference, atol=5e-5, rtol=1e-5)
        assert torch.count_nonzero(silenced[:, :, vd]) == 0
        traces = torch.stack((intact, silenced))
        if not torch.isfinite(traces).all() or traces.max() > 1e6:
            raise ValueError("Activity exceeded the numerical guard.")
        event = round(EVENT_S / dt)
        torch.testing.assert_close(traces[:, 0, :event+1], traces[:, 1, :event+1], rtol=0, atol=0)
        hg_rates = traces[:, :, :, hg].numpy()
        mean = hg_rates.mean(axis=-1, dtype=np.float64)
        relative = (np.arange(len(time)) - event) * dt
        prepost, cycles = scores(relative, mean, dt)
        hg_ids = np.asarray(circuit.dynamic_ids)[hg.numpy()]
        plot(output, relative, mean, cycles, hg_rates, hg_ids)
        np.savez_compressed(output / "responses.npz", time_from_sugar_s=relative, controls=controls,
                            gate=gate, hDeltaG_rates=hg_rates, vDeltaE_mean=traces[:, :, :, vd].mean(-1).numpy(),
                            hDeltaG_ids=hg_ids)
        with (output / "cycle_means.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(cycles[0]))
            writer.writeheader()
            writer.writerows(cycles)
        summary = dict(dt_s=dt, prepost=prepost, cycles=cycles, intact_matches_checkpoint=True,
                       silenced_vDeltaE_exactly_zero=True, pre_event_sugar_control_matches=True)
        write_json(output / "summary.json", summary)
        shutil.copyfile(__file__, output / Path(__file__).name)
        if any(digest(path) != expected for path, expected in frozen.items()):
            raise ValueError("An archived input changed.")
        finish(output, manifest)
    except BaseException as error:
        fail(output, manifest, error)
        raise
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", default=TRAINING)
    parser.add_argument("--loops", default=LOOPS)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.training, args.loops, args.output), indent=2))


if __name__ == "__main__":
    main()
