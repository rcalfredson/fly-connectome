"""Offline PFNd -> hDeltaB rate experiments with explicit, provisional assumptions."""

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .anatomy import annotation_order
from .connectivity import (
    DATASET, SERVER, ValidationError, build_matrix, check_reference_annotations,
    check_reference_weight, reference_result,
)
from .plotting import load_matrix


@dataclass(frozen=True)
class Parameters:
    tau_s: float = 0.1
    dt_s: float = 0.001
    gain: float = 0.001
    duration_s: float = 3.0

    def validate(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValidationError("Model parameters must be finite.")
        if min(self.tau_s, self.dt_s, self.duration_s) <= 0 or self.gain < 0:
            raise ValidationError("Times must be positive and coupling gain nonnegative.")
        if self.dt_s > self.tau_s / 20:
            raise ValidationError("Use dt <= tau/20 for this explicit Euler model.")
        grid_index(self.duration_s, self.dt_s)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grid_index(time_s, dt_s):
    """Avoid silently rounding a stimulus boundary to a different time."""
    if not np.isfinite(time_s):
        raise ValidationError("Stimulus times must be finite.")
    index = round(time_s / dt_s)
    if not np.isclose(time_s / dt_s, index, rtol=0, atol=1e-8):
        raise ValidationError("Duration and stimulus boundaries must be multiples of dt.")
    return index


def load_network(source):
    source = Path(source)
    names = ["metadata.json", "PFNd_neurons.csv", "hDeltaB_neurons.csv",
             "PFNd_to_hDeltaB_connections.csv", "PFNd_to_hDeltaB_matrix.csv"]
    before = {name: sha256(source / name) for name in names}
    metadata = json.loads((source / "metadata.json").read_text())
    if (metadata.get("status") != "complete" or metadata.get("dataset") != DATASET
            or metadata.get("server") != SERVER):
        raise ValidationError("Simulation requires a complete baseline from the configured server/dataset.")
    populations = {name: pd.read_csv(source / f"{name}_neurons.csv") for name in ["PFNd", "hDeltaB"]}
    orders = {name: annotation_order(table, name) for name, table in populations.items()}
    if len(populations["PFNd"]) != 40 or len(populations["hDeltaB"]) != 19:
        raise ValidationError("This initial model expects the reviewed 40 PFNd and 19 hDeltaB neurons.")
    check_reference_annotations(pd.concat(populations.values()).query("bodyId in [26913, 515192]"))
    edges = pd.read_csv(source / "PFNd_to_hDeltaB_connections.csv")
    counts = load_matrix(source / "PFNd_to_hDeltaB_matrix.csv")
    expected = build_matrix(edges, populations["PFNd"].bodyId, populations["hDeltaB"].bodyId)
    pd.testing.assert_frame_equal(counts, expected, check_names=False)
    check_reference_weight(reference_result(counts))
    return counts, populations, orders, metadata, before


def default_experiments(order):
    """Anatomical source groups, with no assigned heading or travel angles."""
    groups = {}
    for glomerulus in [2, 5, 8]:
        selected = order[(order.pb_side == "L") & (order.pb_glomerulus == glomerulus)]
        if selected.empty:
            raise ValidationError(f"Default stimulus group PB L{glomerulus} is missing.")
        groups[f"L{glomerulus}"] = sorted(int(v) for v in selected.bodyId)

    def event(ids, start, stop, label):
        return {"body_ids": ids, "start_s": start, "stop_s": stop, "amplitude": 1.0, "label": label}

    def sequence(labels):
        return [event(groups[label], start, stop, label) for label, start, stop in
                zip(labels, [.2, .3, .4], [.3, .4, .5])]

    return {"experiments": [
        {"name": "unstimulated", "events": []},
        {"name": "reference_pulse", "events": [event([26913], .2, .3, "PFNd 26913")]},
        {"name": "group_pulse", "events": [event(groups["L2"], .2, .3, "L2")]},
        {"name": "sustained", "events": [event(groups["L2"], .2, 1.2, "L2")]},
        {"name": "sequence_forward", "events": sequence(["L2", "L5", "L8"])},
        {"name": "sequence_reverse", "events": sequence(["L8", "L5", "L2"])},
    ]}


def validate_experiments(spec, source_ids, parameters):
    parameters.validate()
    if not isinstance(spec, dict) or set(spec) != {"experiments"}:
        raise ValidationError("Stimulus JSON must contain only an experiments list.")
    experiments = spec["experiments"]
    if not isinstance(experiments, list) or not experiments:
        raise ValidationError("Supply at least one experiment.")
    names = set()
    normalized = []
    for experiment in experiments:
        if not isinstance(experiment, dict) or set(experiment) != {"name", "events"}:
            raise ValidationError("Each experiment needs name and events fields only.")
        name = experiment["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name) or name in names:
            raise ValidationError("Experiment names must be unique lowercase filename-safe identifiers.")
        names.add(name)
        if not isinstance(experiment["events"], list):
            raise ValidationError("Experiment events must be a list.")
        events = []
        for event in experiment["events"]:
            required = {"body_ids", "start_s", "stop_s", "amplitude"}
            if not isinstance(event, dict) or not required.issubset(event) or set(event) - required - {"label"}:
                raise ValidationError("Each event needs body_ids, start_s, stop_s and amplitude; label is optional.")
            ids = event["body_ids"]
            if (not isinstance(ids, list) or not ids
                    or any(type(body) is not int or body not in source_ids for body in ids)
                    or len(set(ids)) != len(ids)):
                raise ValidationError("Stimulus body_ids must be unique integer IDs from the source population.")
            for field in ["start_s", "stop_s", "amplitude"]:
                if type(event[field]) not in [int, float] or not np.isfinite(event[field]):
                    raise ValidationError("Stimulus times and amplitudes must be finite numbers.")
            start, stop, amplitude = (float(event[k]) for k in ["start_s", "stop_s", "amplitude"])
            if not 0 <= start < stop <= parameters.duration_s or amplitude < 0:
                raise ValidationError("Events require 0 <= start < stop <= duration and nonnegative amplitude.")
            grid_index(start, parameters.dt_s)
            grid_index(stop, parameters.dt_s)
            label = event.get("label", name)
            if not isinstance(label, str):
                raise ValidationError("Event labels must be strings.")
            events.append({"body_ids": sorted(ids), "start_s": start, "stop_s": stop,
                           "amplitude": amplitude, "label": label})
        normalized.append({"name": name, "events": events})
    return {"experiments": normalized}


def make_drive(source_ids, events, parameters):
    n = grid_index(parameters.duration_s, parameters.dt_s)
    times = np.arange(n + 1) * parameters.dt_s
    drive = np.zeros((n + 1, len(source_ids)))
    index = {int(body): j for j, body in enumerate(source_ids)}
    for event in events:
        start = grid_index(event["start_s"], parameters.dt_s)
        stop = grid_index(event["stop_s"], parameters.dt_s)
        columns = [index[body] for body in event["body_ids"]]
        # Overlapping events add; each event is active on [start, stop).
        drive[start:stop, columns] += event["amplitude"]
    return times, drive


def integrate(weights, drive, parameters, recurrent_weights=None):
    """Simultaneous explicit Euler updates; W rows are sources, columns targets."""
    parameters.validate()
    weights, drive = np.asarray(weights, dtype=float), np.asarray(drive, dtype=float)
    if weights.ndim != 2 or drive.ndim != 2 or weights.shape[0] != drive.shape[1] or len(drive) < 2:
        raise ValidationError("Weights and drive have incompatible axes.")
    if not np.isfinite(weights).all() or not np.isfinite(drive).all() or (weights < 0).any() or (drive < 0).any():
        raise ValidationError("This initial excitatory model requires finite nonnegative weights and drive.")
    if recurrent_weights is not None:
        recurrent_weights = np.asarray(recurrent_weights, dtype=float)
        if (recurrent_weights.shape != (weights.shape[0], weights.shape[0])
                or not np.isfinite(recurrent_weights).all() or (recurrent_weights < 0).any()):
            raise ValidationError("PFNd recurrent weights must be a finite nonnegative square source matrix.")
    pre = np.zeros_like(drive)
    post = np.zeros((len(drive), weights.shape[1]))
    alpha = parameters.dt_s / parameters.tau_s
    for k in range(len(drive) - 1):
        input_pre = drive[k] if recurrent_weights is None else drive[k] + pre[k] @ recurrent_weights
        pre[k + 1] = pre[k] + alpha * (-pre[k] + np.maximum(0, input_pre))
        post[k + 1] = post[k] + alpha * (-post[k] + np.maximum(0, pre[k] @ weights))
    if not np.isfinite(pre).all() or not np.isfinite(post).all():
        raise ValidationError("Nonfinite simulated activity; review gain and stimulus amplitudes.")
    return pre, post


def analytical_response(weights, source_ids, events, times, tau_s):
    """Independent continuous-time solution for equal-tau, zero-baseline feedforward rates."""
    pre = np.zeros((len(times), len(source_ids)))
    post = np.zeros((len(times), weights.shape[1]))
    index = {int(body): j for j, body in enumerate(source_ids)}
    for event in events:
        columns = [index[body] for body in event["body_ids"]]
        target_weights = weights[columns].sum(axis=0)
        for boundary, sign in [(event["start_s"], 1), (event["stop_s"], -1)]:
            u = np.maximum(0, times - boundary) / tau_s
            first_stage = -np.expm1(-u)
            second_stage = first_stage - u * np.exp(-u)
            pre[:, columns] += (sign * event["amplitude"] * first_stage)[:, None]
            post += (sign * event["amplitude"] * second_stage)[:, None] * target_weights
    # Tiny negative roundoff can occur after subtracting two step responses.
    return np.maximum(0, pre), np.maximum(0, post)


def relative_error(actual, expected):
    scale = max(float(np.max(np.abs(expected))), 1e-12)
    return float(np.max(np.abs(actual - expected))) / scale


def response_summary(name, times, drive, pre, post, events, source_groups=None):
    peak = float(post.max())
    mean_post = post.mean(axis=1)
    result = {
        "experiment": name, "stimulated_neurons": int((drive.max(axis=0) > 0).sum()),
        "injected_drive_au_s": sum(len(e["body_ids"]) * e["amplitude"] * (e["stop_s"] - e["start_s"]) for e in events),
        "hDeltaB_peak_activity": peak,
        "hDeltaB_peak_population_mean": float(mean_post.max()),
        "hDeltaB_population_peak_time_s": float(times[mean_post.argmax()]) if peak else None,
        "hDeltaB_final_max_activity": float(post[-1].max()),
        "hDeltaB_final_over_peak": float(post[-1].max()) / peak if peak else None,
    }
    for pop, mask in (source_groups or {"PFNd": np.ones(pre.shape[1], dtype=bool)}).items():
        result[f"{pop}_peak_activity"] = float(pre[:, mask].max())
    return result


def render_rate_model(source, output, *, heatmaps_only=False):
    source, output = Path(source), Path(output)
    config = json.loads((source / "model_config.json").read_text())
    experiments = json.loads((source / "stimuli.json").read_text())["experiments"]
    source_types = config.get("source_populations", ["PFNd"])
    population_types = [*source_types, "hDeltaB"]
    source_label = " + ".join(source_types)
    orders = {pop: pd.read_csv(source / f"{pop}_anatomical_order.csv") for pop in population_types}
    peak_profiles = []
    for experiment in experiments:
        name = experiment["name"]
        activity = pd.read_csv(source / "experiments" / name / "activity.csv")
        times = activity.time_s.to_numpy()
        ratios = [*[3] * len(source_types), 2]
        if not heatmaps_only:
            ratios = [1, 1, *ratios]
        height = (10 if heatmaps_only else 13) + 3 * (len(source_types) - 1)
        fig, axes = plt.subplots(len(ratios), 1, figsize=(14, height), sharex=True,
                                 gridspec_kw={"height_ratios": ratios}, constrained_layout=True)
        heatmap_axes = axes if heatmaps_only else axes[2:]
        try:
            if not heatmaps_only:
                drive = pd.read_csv(source / "experiments" / name / "drive.csv")
                axes[0].step(times, drive.drop(columns="time_s").sum(axis=1), where="post", color="black")
                axes[0].set_ylabel("Total external\ndrive (a.u.)")
                for event in experiment["events"]:
                    axes[0].axvspan(event["start_s"], event["stop_s"], alpha=.12, color="grey")
                    axes[0].text((event["start_s"] + event["stop_s"]) / 2, 1.02, event["label"],
                                 transform=axes[0].get_xaxis_transform(), ha="center", fontsize=8)
            for pop, ax in zip(population_types, heatmap_axes):
                order = orders[pop]
                values = activity[[f"{pop}_{body}" for body in order.bodyId]].to_numpy()
                if not heatmaps_only:
                    axes[1].plot(times, values.mean(axis=1), label=f"{pop} mean per neuron")
                im = ax.imshow(values.T, origin="upper", aspect="auto", interpolation="nearest",
                               extent=[times[0], times[-1], len(order) - .5, -.5],
                               vmin=0, vmax=max(float(values.max()), 1e-12))
                ax.set_yticks(range(len(order)), order.label, fontsize=6)
                ax.set_ylabel(f"{pop} neuron\n(annotation order)")
                fig.colorbar(im, ax=ax, label="Activity (a.u.; scale per panel)")
                if pop == "hDeltaB":
                    peak_profiles.append(values.max(axis=0))
            if not heatmaps_only:
                axes[1].legend(fontsize=8)
                axes[1].set_ylabel("Population mean\nactivity (a.u.)")
            axes[-1].set_xlabel("Time (s)")
            recurrent_gain = config.get("recurrent_gain", 0.)
            model_name = "recurrent" if recurrent_gain else "feedforward"
            recurrence_label = f"recurrent gain = {recurrent_gain:g}/contact" if recurrent_gain else "no recurrence"
            fig.suptitle(f"{name} | illustrative {source_label} → hDeltaB {model_name} rate model\n"
                         f"τ = {config['tau_s']:g} s; forward gain = {config['gain']:g}/contact; {recurrence_label}")
            for ext in ["png", "pdf"]:
                fig.savefig(output / f"{name}.{ext}", dpi=150)
        finally:
            plt.close(fig)
    if heatmaps_only:
        return
    fig, ax = plt.subplots(figsize=(12, max(3, len(experiments) * .5)), constrained_layout=True)
    try:
        image = ax.imshow(peak_profiles, aspect="auto", vmin=0)
        ax.set_yticks(range(len(experiments)), [e["name"] for e in experiments])
        ax.set_xticks(range(len(orders["hDeltaB"])), orders["hDeltaB"].label, rotation=90, fontsize=7)
        ax.set_title("Peak activity of each hDeltaB neuron during each experiment")
        fig.colorbar(image, ax=ax, label="Peak activity (a.u.; neuron peaks may occur at different times)")
        for ext in ["png", "pdf"]:
            fig.savefig(output / f"hDeltaB_peak_profiles.{ext}", dpi=150)
    finally:
        plt.close(fig)


def verify_model_data(source, manifest):
    source = Path(source).resolve()
    if not manifest.get("data_validated") or not manifest.get("data_sha256"):
        raise ValidationError("Rate-model plotting requires validated data.")
    for name, digest in manifest["data_sha256"].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or not path.is_file() or sha256(path) != digest:
            raise ValidationError(f"Rate-model data changed, missing or outside run: {name}")


def replot_rate_model(source, output, code, *, heatmaps_only=False):
    source, output = Path(source), Path(output)
    manifest = json.loads((source / "rate_model.json").read_text())
    verify_model_data(source, manifest)
    output.mkdir(parents=True, exist_ok=False)
    render_rate_model(source, output, heatmaps_only=heatmaps_only)
    verify_model_data(source, manifest)
    save_json(output / "replot.json", {"status": "complete", "source": str(source.resolve()),
                                      "data_sha256": manifest["data_sha256"], "code": code,
                                      "plot_settings": {"heatmaps_only": heatmaps_only}})


def run_rate_model(source, output, code, parameters=None, stimuli=None, *, include_pfnv=False):
    parameters = parameters or Parameters()
    parameters.validate()
    source, output = Path(source), Path(output)
    if include_pfnv:
        from .pfn import load_pfn_network, pfn_experiments
        counts, populations, orders, baseline, source_hashes = load_pfn_network(source)
        default_spec = pfn_experiments(orders) if stimuli is None else None
    else:
        counts, populations, orders, baseline, source_hashes = load_network(source)
        if baseline.get("network_kind") == "pfn_feedforward":
            raise ValidationError("Use simulate-pfn to include both populations in an expanded extraction.")
        default_spec = default_experiments(orders["PFNd"]) if stimuli is None else None
    source_types = ["PFNd", "PFNv"] if include_pfnv else ["PFNd"]
    source_label = " + ".join(source_types)
    source_groups = {pop: counts.index.isin(populations[pop].bodyId) for pop in source_types}
    body_types = {int(body): pop for pop in source_types for body in populations[pop].bodyId}
    source_columns = [f"{body_types[body]}_{body}" for body in counts.index]
    spec = default_spec if stimuli is None else json.loads(Path(stimuli).read_text())
    spec = validate_experiments(spec, set(counts.index), parameters)
    output.mkdir(parents=True, exist_ok=False)
    config = {
        **asdict(parameters), "baseline_drive": 0.0, "initial_activity": 0.0,
        "nonlinearity": "ReLU", "noise": "none", "activity_units": "arbitrary normalized rate units; not Hz and not bounded at 1",
        "coupling": "W[pre,post] = gain * raw_contact_count; all included connections assumed excitatory",
        "weight_axes": f"Presynaptic {source_label} rows, postsynaptic hDeltaB columns; post input = pre_activity @ W",
        "equations": ["tau * d(PFNd)/dt = -PFNd + ReLU(external_drive)",
                      "tau * d(hDeltaB)/dt = -hDeltaB + ReLU(PFNd @ W)"],
        "solver": "simultaneous explicit Euler; drive[k] active over [t[k], t[k+1])",
        "disabled": ["PFNd recurrence", "hDeltaB feedback", "hDeltaB recurrence", "all other cell types"],
        "interpretation": "Tests propagation through a specified feedforward model. No heading/travel angles, movement encoding, displacement memory or steering implemented.",
        "validation_tolerance": "Max absolute error / block peak <= 0.02 versus continuous-time solution and half-dt; half-dt must not worsen analytical error.",
    }
    if include_pfnv:
        config["source_populations"] = source_types
        config["disabled"][0] = "PFN recurrence and cross-population coupling"
        config["coupling"] += "; identical gain per contact for PFNd and PFNv; all hDeltaB input compartments pooled"
        config["equations"] = [
            "tau * d(PFNd)/dt = -PFNd + ReLU(I_d)",
            "tau * d(PFNv)/dt = -PFNv + ReLU(I_v)",
            "tau * d(hDeltaB)/dt = -hDeltaB + ReLU(PFNd @ W_d + PFNv @ W_v)",
        ]
    manifest = {
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(source.resolve()), "source_sha256": source_hashes,
        "source_metadata": baseline, "server": SERVER, "dataset": DATASET,
        "code": code, "code_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ["numpy", "pandas", "matplotlib"]},
        "network": {**{pop: len(populations[pop]) for pop in source_types}, "hDeltaB": len(counts.columns),
                    "positive_pairs": int(counts.gt(0).to_numpy().sum()), "raw_contact_total": int(counts.to_numpy().sum())},
    }
    save_json(output / "rate_model.json", manifest)
    try:
        save_json(output / "model_config.json", config)
        save_json(output / "stimuli.json", spec)
        counts.to_csv(output / "raw_contact_matrix.csv")
        weights = counts.to_numpy(dtype=float) * parameters.gain
        pd.DataFrame(weights, index=counts.index, columns=counts.columns).to_csv(output / "effective_weight_matrix.csv")
        for pop in populations:
            populations[pop].to_csv(output / f"{pop}_neurons.csv", index=False)
            orders[pop].to_csv(output / f"{pop}_anatomical_order.csv", index=False)
        summaries, neuron_summaries, checks, sensitivity = [], [], [], []
        for experiment in spec["experiments"]:
            name, events = experiment["name"], experiment["events"]
            print(f"Simulating {name}", flush=True)
            destination = output / "experiments" / name
            destination.mkdir(parents=True)
            times, drive = make_drive(counts.index, events, parameters)
            pre, post = integrate(weights, drive, parameters)
            columns = source_columns + [f"hDeltaB_{body}" for body in counts.columns]
            pd.DataFrame(np.column_stack([times, pre, post]), columns=["time_s", *columns]).to_csv(destination / "activity.csv", index=False)
            pd.DataFrame(np.column_stack([times, drive]), columns=["time_s", *source_columns]).to_csv(destination / "drive.csv", index=False)
            summaries.append(response_summary(name, times, drive, pre, post, events, source_groups))
            blocks = [(pop, counts.index[mask], pre[:, mask]) for pop, mask in source_groups.items()]
            for pop, ids, activity in [*blocks, ("hDeltaB", counts.columns, post)]:
                for j, body in enumerate(ids):
                    neuron_summaries.append({"experiment": name, "population": pop, "bodyId": int(body),
                                             "peak_activity": float(activity[:, j].max()),
                                             "peak_time_s": float(times[activity[:, j].argmax()]) if activity[:, j].max() else None,
                                             "final_activity": float(activity[-1, j]),
                                             "activity_area_au_s": float((activity[1:, j] + activity[:-1, j]).sum() * parameters.dt_s / 2)})
            fine_parameters = replace(parameters, dt_s=parameters.dt_s / 2)
            fine_times, fine_drive = make_drive(counts.index, events, fine_parameters)
            fine = integrate(weights, fine_drive, fine_parameters)
            exact = analytical_response(weights, counts.index, events, times, parameters.tau_s)
            replay = integrate(weights, drive, parameters)
            validation_blocks = [(pop, pre[:, mask], fine[0][:, mask], exact[0][:, mask], replay[0][:, mask])
                                 for pop, mask in source_groups.items()]
            validation_blocks.append(("hDeltaB", post, fine[1], exact[1], replay[1]))
            for pop, coarse, refined, theory, repeated in validation_blocks:
                exact_error = relative_error(coarse, theory)
                fine_error = relative_error(refined[::2], theory)
                half_dt_error = relative_error(coarse, refined[::2])
                passed = (exact_error <= .02 and half_dt_error <= .02
                          and fine_error <= exact_error + 1e-10 and np.array_equal(coarse, repeated))
                checks.append({"experiment": name, "population": pop, "relative_analytical_error": exact_error,
                               "relative_half_dt_error": half_dt_error, "half_dt_analytical_error": fine_error,
                               "deterministic_replay": np.array_equal(coarse, repeated), "passed": bool(passed)})
            # One parameter at a time; all sweeps use half dt. Base case included once.
            for tau_factor, gain_factor in [(1, 1), (.5, 1), (2, 1), (1, .5), (1, 2)]:
                swept = replace(fine_parameters, tau_s=parameters.tau_s * tau_factor, gain=parameters.gain * gain_factor)
                swept_pre, swept_post = integrate(counts.to_numpy(dtype=float) * swept.gain, fine_drive, swept)
                sensitivity.append({**response_summary(name, fine_times, fine_drive, swept_pre, swept_post, events, source_groups),
                                    "tau_s": swept.tau_s, "gain": swept.gain, "dt_s": swept.dt_s})
        pd.DataFrame(summaries).to_csv(output / "experiment_summary.csv", index=False)
        pd.DataFrame(neuron_summaries).to_csv(output / "neuron_summary.csv", index=False)
        pd.DataFrame(checks).to_csv(output / "numerical_validation.csv", index=False)
        pd.DataFrame(sensitivity).to_csv(output / "parameter_sensitivity.csv", index=False)
        if not all(row["passed"] for row in checks):
            raise ValidationError("Rate-model validation failed; inspect numerical_validation.csv.")
        if any(sha256(source / name) != digest for name, digest in source_hashes.items()):
            raise ValidationError("Baseline data changed during simulation.")
        manifest["validation"] = {"analytical_solution": True, "half_timestep": True, "deterministic_replay": True,
                                  "source_unchanged": True, "reference_26913_to_515192": 137,
                                  "max_relative_analytical_error": max(r["relative_analytical_error"] for r in checks),
                                  "max_relative_half_dt_error": max(r["relative_half_dt_error"] for r in checks)}
        manifest["data_sha256"] = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file() and p.suffix in [".csv", ".json"] and p.name != "rate_model.json"}
        manifest["data_validated"] = True
        save_json(output / "rate_model.json", manifest)
        render_rate_model(output, output)
        verify_model_data(output, manifest)
        manifest["artifact_sha256"] = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "rate_model.json"}
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="validation_failed" if isinstance(exc, (ValidationError, AssertionError)) else "failed",
                        error_type=type(exc).__name__)
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(output / "rate_model.json", manifest)
    return manifest
