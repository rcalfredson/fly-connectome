"""Matched, subcritical PFNd recurrence sweeps using saved feedforward experiments."""

from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .connectivity import DATASET, SERVER, ValidationError, build_matrix, check_reference_weight, reference_result
from .inputs import verify_input_data
from .rate_model import (
    Parameters, analytical_response, integrate, make_drive, relative_error,
    render_rate_model, response_summary, save_json, sha256, validate_experiments,
    verify_model_data,
)


RECRUITMENT_THRESHOLD = .001


def load_recurrence(census, feedforward, source_ids):
    census, feedforward = Path(census), Path(feedforward)
    cm = json.loads((census / "inputs.json").read_text())
    fm = json.loads((feedforward / "rate_model.json").read_text())
    if cm.get("status") != "complete" or fm.get("status") != "complete":
        raise ValidationError("Use complete census and feedforward simulation runs.")
    verify_input_data(census, cm)
    verify_model_data(feedforward, fm)
    if any(m.get("dataset") != DATASET or m.get("server") != SERVER for m in [cm, fm]):
        raise ValidationError("Census and feedforward data must use the configured server/dataset.")
    a = cm["dataset_metadata"]
    b = fm["source_metadata"]["dataset_metadata"]
    if not a.get("uuid") or a.get("uuid") != b.get("uuid"):
        raise ValidationError("Census and feedforward dataset UUIDs differ or are missing.")
    for field in ["latestMutationId", "lastDatabaseEdit"]:
        if a.get(field) != b.get(field):
            raise ValidationError(f"Census and feedforward dataset {field} values differ.")
    targets = pd.read_csv(census / "PFNd_neurons.csv")
    baseline_targets = pd.read_csv(feedforward / "PFNd_neurons.csv")
    columns = ["bodyId", "type", "instance"]
    pd.testing.assert_frame_equal(targets[columns].sort_values("bodyId").reset_index(drop=True),
                                  baseline_targets[columns].sort_values("bodyId").reset_index(drop=True))
    if set(targets.bodyId) != set(source_ids) or targets.bodyId.duplicated().any():
        raise ValidationError("PFNd population differs from the feedforward matrix axes.")
    upstream = pd.read_csv(census / "upstream_neurons.csv")
    selected = upstream[upstream.bodyId.isin(source_ids)]
    if not selected.type.eq("PFNd").all() or selected.bodyId.duplicated().any():
        raise ValidationError("Recurrent source annotations do not agree with PFNd membership.")
    raw = pd.read_csv(census / "incoming_connections.csv")
    recurrent = raw[raw.bodyId_pre.isin(source_ids)].copy()
    counts = build_matrix(recurrent, source_ids, source_ids)
    return counts, cm, fm


def stability(counts, gain, parameters):
    if not np.isfinite(gain) or gain < 0:
        raise ValidationError("Recurrent gains must be finite and nonnegative.")
    values = np.asarray(counts, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or not np.isfinite(values).all() or (values < 0).any():
        raise ValidationError("Recurrent counts must be a finite nonnegative square matrix.")
    eigenvalues = np.linalg.eigvals(values)
    rho = float(np.abs(eigenvalues).max())
    feedback = gain * rho
    euler_rho = float(np.abs(1 + parameters.dt_s / parameters.tau_s * (gain * eigenvalues - 1)).max())
    if feedback >= 1 or euler_rho >= 1:
        raise ValidationError("Recurrent gain is not subcritical or the Euler update is unstable.")
    return {"recurrent_gain": float(gain), "raw_spectral_radius": rho,
            "feedback_spectral_radius": feedback, "euler_spectral_radius": euler_rho,
            "critical_gain": 1 / rho if rho else None,
            "slowest_PFNd_mode_tau_s": parameters.tau_s / (1 - feedback),
            "continuous_stable": True, "euler_stable": True}


def matrix_exponential(matrix):
    """Scaling/squaring Taylor reference, independent of the Euler solver; no SciPy dependency."""
    matrix = np.asarray(matrix, dtype=float)
    norm = float(np.linalg.norm(matrix, ord=np.inf))
    squarings = max(0, int(np.ceil(np.log2(norm / .5)))) if norm > .5 else 0
    small = matrix / 2 ** squarings
    result = np.eye(len(matrix))
    term = result.copy()
    for k in range(1, 200):
        term = term @ small / k
        result += term
        if np.linalg.norm(term, ord=np.inf) < 1e-16 * max(1, np.linalg.norm(result, ord=np.inf)):
            break
    else:
        raise ValidationError("Matrix-exponential reference failed to converge.")
    for _ in range(squarings):
        result = result @ result
    return result


def reference_operators(weights, recurrent_weights, parameters):
    """Exact interval propagator for piecewise-constant drive and nonnegative activity."""
    n, h = weights.shape
    size = n + h
    generator = np.zeros((size + n, size + n))
    generator[:n, :n] = recurrent_weights - np.eye(n)
    generator[:n, n:size] = weights
    generator[n:size, n:size] = -np.eye(h)
    generator[size:, :n] = np.eye(n)
    exponential = matrix_exponential(generator * parameters.dt_s / parameters.tau_s)
    return exponential[:size, :size], exponential[size:, :size]


def continuous_reference(drive, operators):
    state_operator, input_operator = operators
    activity = np.zeros((len(drive), len(state_operator)))
    for k in range(len(drive) - 1):
        activity[k + 1] = activity[k] @ state_operator + drive[k] @ input_operator
    return activity[:, :drive.shape[1]], activity[:, drive.shape[1]:]


def decay_metrics(times, activity, last_stop):
    peak = float(activity.max())
    if not peak:
        return {"decay_to_1pct_after_offset_s": None, "decay_observed": False, "final_over_peak": None}
    envelope = activity.max(axis=1)
    active = np.flatnonzero((times >= last_stop) & (envelope > .01 * peak))
    crossing = active[-1] + 1 if len(active) else int(np.searchsorted(times, last_stop))
    observed = crossing < len(times)
    return {"decay_to_1pct_after_offset_s": float(times[crossing] - last_stop) if observed else None,
            "decay_observed": bool(observed), "final_over_peak": float(envelope[-1] / peak)}


def comparison_metrics(name, times, drive, pre, post, events, baseline):
    result = response_summary(name, times, drive, pre, post, events)
    stimulated = drive.max(axis=0) > 0
    result["recruited_unstimulated_PFNd"] = int((pre[:, ~stimulated].max(axis=0) > RECRUITMENT_THRESHOLD).sum())
    result["unstimulated_PFNd_count"] = int((~stimulated).sum())
    last_stop = max((e["stop_s"] for e in events), default=0.)
    result["last_stimulus_offset_s"] = last_stop
    for pop, activity in [("PFNd", pre), ("hDeltaB", post)]:
        result.update({f"{pop}_{key}": value for key, value in decay_metrics(times, activity, last_stop).items()})
    old_pre, old_post = baseline
    for pop, activity, old in [("PFNd", pre, old_pre), ("hDeltaB", post, old_post)]:
        result[f"{pop}_peak_ratio_to_feedforward"] = float(activity.max() / old.max()) if old.max() else None
    profile, old_profile = post.max(axis=0), old_post.max(axis=0)
    result["hDeltaB_peak_profile_cosine_to_feedforward"] = (
        float(np.dot(profile, old_profile) / (np.linalg.norm(profile) * np.linalg.norm(old_profile)))
        if profile.any() and old_profile.any() else None)
    return result


def render_comparisons(source, output):
    source, output = Path(source), Path(output)
    manifest = json.loads((source / "recurrent_model.json").read_text())
    spec = json.loads((source / "stimuli.json").read_text())
    for experiment in spec["experiments"]:
        name = experiment["name"]
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
        try:
            for condition in manifest["conditions"]:
                frame = pd.read_csv(source / condition["directory"] / "experiments" / name / "activity.csv")
                pre = frame.filter(regex="^PFNd_").to_numpy()
                post = frame.filter(regex="^hDeltaB_").to_numpy()
                baseline = pd.read_csv(source / condition["directory"] / "experiments" / name / "drive.csv")
                unstimulated = baseline.drop(columns="time_s").max(axis=0).to_numpy() == 0
                label = f"recurrent gain {condition['recurrent_gain']:g}"
                axes[0].plot(frame.time_s, pre.mean(axis=1), label=label)
                axes[1].plot(frame.time_s, post.mean(axis=1), label=label)
                values = pre[:, unstimulated].mean(axis=1) if unstimulated.any() else np.zeros(len(pre))
                axes[2].plot(frame.time_s, values, label=label)
            for event in experiment["events"]:
                for ax in axes:
                    ax.axvspan(event["start_s"], event["stop_s"], color="grey", alpha=.1)
            axes[0].legend(fontsize=8)
            for ax, label in zip(axes, ["PFNd mean activity", "hDeltaB mean activity", "Unstimulated PFNd mean activity"]):
                ax.set_ylabel(f"{label}\n(a.u.)")
            axes[-1].set_xlabel("Time (s); shaded intervals indicate external stimulation")
            fig.suptitle(f"{name} | same stimulus and feedforward weights; varying PFNd recurrence")
            for ext in ["png", "pdf"]:
                fig.savefig(output / f"{name}_comparison.{ext}", dpi=150)
        finally:
            plt.close(fig)
    table = pd.read_csv(source / "comparison_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    metrics = ["hDeltaB_peak_ratio_to_feedforward", "recruited_unstimulated_PFNd",
               "PFNd_decay_to_1pct_after_offset_s", "hDeltaB_decay_to_1pct_after_offset_s"]
    for name, group in table.groupby("experiment", sort=False):
        if name == "unstimulated":
            continue
        for ax, metric in zip(axes.flat, metrics):
            ax.plot(group.recurrent_gain, group[metric], marker="o", label=name)
            ax.set_xlabel("Recurrent gain per contact")
            ax.set_ylabel(metric.replace("_", " "))
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Subcritical recurrence: amplification, recruitment and decay")
    try:
        for ext in ["png", "pdf"]:
            fig.savefig(output / f"recurrence_summary.{ext}", dpi=150)
    finally:
        plt.close(fig)


def run_recurrent_model(feedforward, census, output, code, gains=(0., .0005, .001, .002)):
    feedforward, census, output = Path(feedforward), Path(census), Path(output)
    fm = json.loads((feedforward / "rate_model.json").read_text())
    verify_model_data(feedforward, fm)
    config = json.loads((feedforward / "model_config.json").read_text())
    if config.get("recurrent_gain", 0) or config.get("baseline_drive") != 0 or config.get("initial_activity") != 0 or config.get("nonlinearity") != "ReLU" or config.get("noise") != "none":
        raise ValidationError("Use a zero-baseline, zero-initial, noiseless feedforward ReLU run as the comparison source.")
    parameters = Parameters(**{key: config[key] for key in asdict(Parameters())})
    parameters.validate()
    counts = pd.read_csv(feedforward / "raw_contact_matrix.csv", index_col=0)
    counts.index = counts.index.astype(int)
    counts.columns = counts.columns.astype(int)
    check_reference_weight(reference_result(counts))
    recurrent_counts, cm, fm = load_recurrence(census, feedforward, counts.index)
    weights = counts.to_numpy(dtype=float) * parameters.gain
    saved_weights = pd.read_csv(feedforward / "effective_weight_matrix.csv", index_col=0)
    np.testing.assert_array_equal(saved_weights.index, counts.index)
    np.testing.assert_array_equal(saved_weights.columns.astype(int), counts.columns)
    np.testing.assert_allclose(saved_weights, weights, rtol=1e-14, atol=1e-15)
    spec = validate_experiments(json.loads((feedforward / "stimuli.json").read_text()), set(counts.index), parameters)
    gains = [float(g) for g in gains]
    if not gains or len(gains) != len(set(gains)) or 0. not in gains:
        raise ValidationError("Supply unique recurrent gains including zero for the matched control.")
    gains.sort()
    conditions = [{**stability(recurrent_counts, gain, parameters),
                   "directory": "gain_" + f"{gain:.12g}".replace(".", "p").replace("-", "m")} for gain in gains]
    if len({c["directory"] for c in conditions}) != len(conditions):
        raise ValidationError("Gain values collide at the saved directory precision.")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "feedforward_directory": str(feedforward.resolve()), "census_directory": str(census.resolve()),
                "feedforward_manifest_sha256": sha256(feedforward / "rate_model.json"),
                "census_manifest_sha256": sha256(census / "inputs.json"),
                "dataset": DATASET, "server": SERVER, "dataset_metadata": cm["dataset_metadata"],
                "conditions": conditions, "parameters": asdict(parameters), "code": code,
                "code_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                "python": platform.python_version(), "packages": {p: version(p) for p in ["numpy", "pandas", "matplotlib"]},
                "recurrent_pairs": int((recurrent_counts > 0).to_numpy().sum()),
                "recurrent_contacts": int(recurrent_counts.to_numpy().sum()),
                "self_contacts": int(np.trace(recurrent_counts)),
                "definitions": {"recruitment": f"Never externally stimulated PFNd neurons whose peak exceeds {RECRUITMENT_THRESHOLD} a.u.; operational threshold, not a biological firing criterion.",
                                "decay": "Time after last stimulus offset until maximum neuron activity stays <=1% of that population's whole-trial peak. Blank if unobserved or zero response.",
                                "profile": "Cosine similarity of vectors of per-neuron hDeltaB temporal maxima; maxima need not occur simultaneously.",
                                "reference": "Continuous-time linear propagator via scaling/squaring matrix exponential; valid because all states, inputs and couplings are nonnegative and ReLU is therefore identity.",
                                "limits": "Single activity value pools all anatomical regions. No saturation, explicit delay, inhibitory populations, hDeltaB feedback, angle decoder or displacement memory."}}
    save_json(output / "recurrent_model.json", manifest)
    all_checks, all_metrics, neuron_rows = [], [], []
    try:
        save_json(output / "stimuli.json", spec)
        pd.DataFrame(conditions).to_csv(output / "stability.csv", index=False)
        recurrent_counts.to_csv(output / "raw_recurrent_matrix.csv")
        for condition in conditions:
            gain = condition["recurrent_gain"]
            print(f"Simulating recurrent gain {gain:g}", flush=True)
            directory = output / condition["directory"]
            directory.mkdir()
            r = recurrent_counts.to_numpy(dtype=float) * gain
            child_config = {**config, "recurrent_gain": gain,
                            "equations": ["tau * d(PFNd)/dt = -PFNd + ReLU(external_drive + PFNd @ R)", config["equations"][1]],
                            "recurrent_coupling": "R[pre,post] = recurrent_gain * recurrent_contact_count; assumed excitatory; all regions pooled",
                            "disabled": ["hDeltaB feedback", "hDeltaB recurrence", "all other cell types"],
                            "validation_tolerance": "Relative block-peak error <=2% versus matrix-exponential continuous-time reference and half dt; refined error must not increase.",
                            "interpretation": "Matched recurrent propagation experiment; transient persistence is not displacement memory."}
            save_json(directory / "model_config.json", child_config)
            save_json(directory / "stimuli.json", spec)
            counts.to_csv(directory / "raw_contact_matrix.csv")
            pd.DataFrame(weights, index=counts.index, columns=counts.columns).to_csv(directory / "effective_weight_matrix.csv")
            recurrent_counts.to_csv(directory / "raw_recurrent_matrix.csv")
            pd.DataFrame(r, index=counts.index, columns=counts.index).to_csv(directory / "effective_recurrent_weight_matrix.csv")
            for pop in ["PFNd", "hDeltaB"]:
                for suffix in ["neurons", "anatomical_order"]:
                    (directory / f"{pop}_{suffix}.csv").write_bytes((feedforward / f"{pop}_{suffix}.csv").read_bytes())
            operators = reference_operators(weights, r, parameters)
            child_checks, child_summaries = [], []
            child = {"status": "running", "parameters": asdict(parameters), "stability": condition,
                     "code": code, "code_sha256": manifest["code_sha256"],
                     "source_feedforward_manifest_sha256": manifest["feedforward_manifest_sha256"],
                     "source_census_manifest_sha256": manifest["census_manifest_sha256"]}
            save_json(directory / "rate_model.json", child)
            for experiment in spec["experiments"]:
                name, events = experiment["name"], experiment["events"]
                times, drive = make_drive(counts.index, events, parameters)
                old = pd.read_csv(feedforward / "experiments" / name / "activity.csv")
                np.testing.assert_allclose(times, old.time_s, rtol=0, atol=1e-12)
                old_drive = pd.read_csv(feedforward / "experiments" / name / "drive.csv")
                cols_pre = [f"PFNd_{body}" for body in counts.index]
                cols_post = [f"hDeltaB_{body}" for body in counts.columns]
                np.testing.assert_allclose(drive, old_drive[cols_pre], rtol=0, atol=1e-14)
                baseline = (old[cols_pre].to_numpy(), old[cols_post].to_numpy())
                pre, post = integrate(weights, drive, parameters, r)
                fine_parameters = replace(parameters, dt_s=parameters.dt_s / 2)
                _, fine_drive = make_drive(counts.index, events, fine_parameters)
                fine = integrate(weights, fine_drive, fine_parameters, r)
                exact = continuous_reference(drive, operators)
                repeated = integrate(weights, drive, parameters, r)
                for pop, activity, refined, theory, replay, old_activity in zip(["PFNd", "hDeltaB"], [pre, post], fine, exact, repeated, baseline):
                    analytic_error = relative_error(activity, theory)
                    fine_error = relative_error(refined[::2], theory)
                    dt_error = relative_error(activity, refined[::2])
                    matches = bool(np.allclose(activity, old_activity, rtol=1e-12, atol=1e-14)) if gain == 0 else None
                    passed = analytic_error <= .02 and dt_error <= .02 and fine_error <= analytic_error + 1e-10 and np.array_equal(activity, replay) and matches is not False
                    child_checks.append({"recurrent_gain": gain, "experiment": name, "population": pop,
                                         "relative_analytical_error": analytic_error, "half_dt_analytical_error": fine_error,
                                         "relative_half_dt_error": dt_error, "deterministic_replay": np.array_equal(activity, replay),
                                         "zero_gain_matches_saved_feedforward": matches, "passed": bool(passed)})
                if gain == 0:
                    closed_form = analytical_response(weights, counts.index, events, times, parameters.tau_s)
                    for actual, expected in zip(exact, closed_form):
                        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)
                destination = directory / "experiments" / name
                destination.mkdir(parents=True)
                pd.DataFrame(np.column_stack([times, pre, post]), columns=["time_s", *cols_pre, *cols_post]).to_csv(destination / "activity.csv", index=False)
                pd.DataFrame(np.column_stack([times, drive]), columns=["time_s", *cols_pre]).to_csv(destination / "drive.csv", index=False)
                metrics = {"recurrent_gain": gain, **comparison_metrics(name, times, drive, pre, post, events, baseline)}
                child_summaries.append(metrics)
                stimulated = drive.max(axis=0) > 0
                for pop, ids, activity in [("PFNd", counts.index, pre), ("hDeltaB", counts.columns, post)]:
                    for j, body in enumerate(ids):
                        neuron_rows.append({"recurrent_gain": gain, "experiment": name, "population": pop,
                                            "bodyId": int(body), "peak_activity": float(activity[:, j].max()),
                                            "externally_stimulated": bool(stimulated[j]) if pop == "PFNd" else False})
            pd.DataFrame(child_summaries).to_csv(directory / "experiment_summary.csv", index=False)
            pd.DataFrame(child_checks).to_csv(directory / "numerical_validation.csv", index=False)
            all_metrics.extend(child_summaries)
            all_checks.extend(child_checks)
            if not all(c["passed"] for c in child_checks):
                raise ValidationError("Recurrent numerical validation failed; inspect condition numerical_validation.csv.")
            child["data_sha256"] = {str(p.relative_to(directory)): sha256(p) for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "rate_model.json"}
            child["data_validated"] = True
            save_json(directory / "rate_model.json", child)
            render_rate_model(directory, directory)
            verify_model_data(directory, child)
            child["status"] = "complete"
            child["artifact_sha256"] = {str(p.relative_to(directory)): sha256(p) for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "rate_model.json"}
            save_json(directory / "rate_model.json", child)
        pd.DataFrame(all_metrics).to_csv(output / "comparison_summary.csv", index=False)
        pd.DataFrame(all_checks).to_csv(output / "numerical_validation.csv", index=False)
        pd.DataFrame(neuron_rows).to_csv(output / "neuron_comparison.csv", index=False)
        verify_input_data(census, cm)
        verify_model_data(feedforward, fm)
        if (sha256(feedforward / "rate_model.json") != manifest["feedforward_manifest_sha256"]
                or sha256(census / "inputs.json") != manifest["census_manifest_sha256"]):
            raise ValidationError("Source manifests changed during the recurrence sweep.")
        manifest["validation"] = {"conditions": len(conditions), "experiments_per_condition": len(spec["experiments"]),
                                  "all_subcritical": True, "source_unchanged": True, "zero_gain_matches_saved_feedforward": True,
                                  "max_relative_analytical_error": max(c["relative_analytical_error"] for c in all_checks),
                                  "max_relative_half_dt_error": max(c["relative_half_dt_error"] for c in all_checks)}
        manifest["data_sha256"] = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file() and p.suffix in [".csv", ".json"] and p.name != "recurrent_model.json"}
        manifest["data_validated"] = True
        save_json(output / "recurrent_model.json", manifest)
        render_comparisons(output, output)
        verify_model_data(output, manifest)
        manifest["artifact_sha256"] = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "recurrent_model.json"}
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="validation_failed" if isinstance(exc, (ValidationError, AssertionError)) else "failed", error_type=type(exc).__name__)
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(output / "recurrent_model.json", manifest)
    return manifest


def replot_recurrent_model(source, output, code, *, heatmaps_only=False):
    source, output = Path(source), Path(output)
    manifest = json.loads((source / "recurrent_model.json").read_text())
    verify_model_data(source, manifest)
    output.mkdir(parents=True, exist_ok=False)
    if heatmaps_only:
        for condition in manifest["conditions"]:
            destination = output / condition["directory"]
            destination.mkdir()
            render_rate_model(source / condition["directory"], destination, heatmaps_only=True)
    else:
        render_comparisons(source, output)
    verify_model_data(source, manifest)
    save_json(output / "replot.json", {"status": "complete", "source": str(source.resolve()),
                                      "data_sha256": manifest["data_sha256"], "code": code,
                                      "plot_settings": {"heatmaps_only": heatmaps_only}})
