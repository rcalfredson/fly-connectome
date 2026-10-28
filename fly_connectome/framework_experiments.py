"""Offline, separate-output pilot of the common rate framework.

python -m fly_connectome.framework_experiments --help
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .connectivity import ValidationError
from .inputs import save_json, sha256
from .janke_reference import (JankeParameters, discrete_reference, absolute_input_control,
                              rate_candidate, bump_readout)
from .movement import MovementParameters, movement_weights
from .rate_framework import RateNetwork, coupling_from_counts
from .rate_model import relative_error, verify_model_data
from .trajectory import replay


def _begin(output, kind, source_files=()):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="running", kind=kind, started_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(Path(p).resolve()): sha256(p) for p in source_files},
        code_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        packages=dict(numpy=np.__version__, pandas=pd.__version__),
        scope="Offline pilot; existing commands/data unchanged; no weight learning or inferred functional mapping")
    save_json(output / "framework_run.json", manifest)
    return output, manifest


def _finish(output, manifest, validation):
    if not all(validation.values()):
        raise ValidationError("Framework numerical/reproduction check failed; inspect saved comparison tables.")
    for name, digest in manifest["source_sha256"].items():
        if sha256(name) != digest:
            raise ValidationError("Source file changed during the pilot.")
    manifest.update(status="complete", data_validated=True,
        finished_at_utc=datetime.now(timezone.utc).isoformat(), validation=validation,
        data_sha256={str(p.relative_to(output)): sha256(p) for p in output.rglob("*")
                     if p.is_file() and p.name != "framework_run.json"})
    save_json(output / "framework_run.json", manifest)
    return manifest


def _failed(output, manifest, exc):
    manifest.update(status="failed", error_type=type(exc).__name__,
                    finished_at_utc=datetime.now(timezone.utc).isoformat())
    save_json(output / "framework_run.json", manifest)


def _plot(output, name, render):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    try:
        render(axes)
        for extension in ["png", "pdf"]:
            fig.savefig(output / f"{name}.{extension}")
    finally:
        plt.close(fig)


def run_pfn(movement, trajectory, output):
    """Reconstruct weights and replay the existing selected continuous trajectory."""
    movement, trajectory = Path(movement), Path(trajectory)
    records = []
    for directory, filename in [(movement, "movement_model.json"), (trajectory, "trajectory_model.json")]:
        record = json.loads((directory / filename).read_text())
        if record.get("status") != "complete":
            raise ValidationError("PFN comparison needs complete saved movement and trajectory runs.")
        verify_model_data(directory, record)
        records.append((directory, record))
    config = json.loads((trajectory / "trajectory_config.json").read_text())
    movement_config = json.loads((movement / "movement_config.json").read_text())
    p = MovementParameters(**config["parameters"])
    p.validate()
    selected = dict(tau_s=.1, gain=.001, normalize_incoming_by_group=True,
                    pfnd_weight_multiplier=6., pfnv_weight_multiplier=1.)
    if (config["baseline"] != 1 or any(getattr(p, k) != v for k, v in selected.items())
            or config["source_ids"] != movement_config["source_ids"]
            or config["target_ids"] != movement_config["target_ids"]
            or config["parameters"] != movement_config["parameters"]):
        raise ValidationError("Use matching selected b=1, PFNd×6/PFNv×1 group-normalized runs.")
    mapping = pd.read_csv(movement / "PFN_encoding_map.csv")
    column = "bodyId" if "bodyId" in mapping else "bodyId_pre"
    mapping = mapping.set_index(column).loc[config["source_ids"]].reset_index()
    counts = pd.read_csv(movement / "raw_contact_matrix.csv", index_col=0)
    counts.columns = counts.columns.astype(int)
    counts = counts.loc[config["source_ids"], config["target_ids"]]
    weights = movement_weights(counts, mapping, p)
    neurons = pd.DataFrame(dict(neuron_id=[str(v) for v in config["source_ids"] + config["target_ids"]],
        cell_type=list(mapping.type) + ["hDeltaB"] * len(config["target_ids"]), tau_s=p.tau_s,
        input_group=[f"{t}-{side}" for t, side in zip(mapping.type, mapping.pb_side)]
                    + ["hDeltaB"] * len(config["target_ids"])))
    contacts = pd.DataFrame([dict(source_id=str(a), target_id=str(b), contacts=int(counts.loc[a, b]))
        for a in counts.index for b in counts.columns if counts.loc[a, b] > 0])
    rules = pd.DataFrame([dict(source_type=t, target_type="hDeltaB", channel="rate", sign=1,
        strength=p.gain * multiplier, sign_evidence="assumed",
        sign_detail="Existing PFN model positive coupling assumption; not a per-connection receptor measurement")
        for t, multiplier in [("PFNd", p.pfnd_weight_multiplier), ("PFNv", p.pfnv_weight_multiplier)]])
    coupling = coupling_from_counts(neurons, contacts, rules, normalize_incoming=True)
    network = RateNetwork(neurons, coupling, rate_unit="arbitrary_rate_units")
    reconstructed = np.zeros_like(weights)
    pre_index = {str(v): i for i, v in enumerate(counts.index)}
    post_index = {str(v): i for i, v in enumerate(counts.columns)}
    for edge in coupling.itertuples():
        reconstructed[pre_index[edge.source_id], post_index[edge.target_id]] = edge.effective_weight
    np.testing.assert_allclose(reconstructed, weights, rtol=1e-12, atol=1e-15)
    for directory in [movement, trajectory]:
        saved = pd.read_csv(directory / "effective_weight_matrix.csv", index_col=0)
        saved.columns = saved.columns.astype(int)
        np.testing.assert_allclose(weights, saved.loc[counts.index, counts.columns], rtol=1e-12, atol=1e-15)
    source_files = [directory / name for directory, record in records for name in record["data_sha256"]]
    source_files += [directory / name for directory, name in [(movement, "movement_model.json"), (trajectory, "trajectory_model.json")]]
    output, manifest = _begin(output, "pfn_existing_model_reproduction", source_files)
    try:
        protocol = pd.read_csv(trajectory / "protocol.csv")
        inputs, prescribed, drive, legacy, exact = replay(protocol, mapping, weights, p)
        with np.load(trajectory / "activity.npz") as saved:
            np.testing.assert_allclose(prescribed, saved["pfn_activity"], rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(legacy, saved["hdeltab_activity"], rtol=1e-12, atol=1e-14)
        size, sources = len(neurons), len(mapping)
        def solve(values, dt):
            return network.solve(np.zeros((len(values), size)), dt,
                prescribed={str(body): values[:, i] for i, body in enumerate(config["source_ids"])})
        trace = solve(prescribed, p.dt_s)
        repeated = solve(prescribed, p.dt_s)
        fine_input = np.vstack((np.repeat(prescribed[:-1], 2, axis=0), prescribed[-1]))
        fine = solve(fine_input, p.dt_s / 2)
        activity = trace.rates[:, sources:]
        metrics = dict(legacy_max_abs_error=float(np.max(np.abs(activity - legacy))),
            relative_exact_error=relative_error(activity, exact),
            relative_half_dt_error=relative_error(activity, fine.rates[::2, sources:]))
        validation = dict(weights_reconstructed=True, saved_encoding_and_legacy_trace_match=True,
            common_solver_matches_legacy=bool(np.allclose(activity, legacy, rtol=1e-12, atol=1e-14)),
            exact_error_below_2_percent=metrics["relative_exact_error"] < .02,
            half_dt_error_below_2_percent=metrics["relative_half_dt_error"] < .02,
            deterministic=bool(np.array_equal(trace.rates, repeated.rates)))
        network.neurons.to_csv(output / "neurons.csv", index=False)
        counts.to_csv(output / "raw_contact_matrix.csv")
        coupling.to_csv(output / "coupling.csv", index=False)
        rules.to_csv(output / "coupling_rules.csv", index=False)
        inputs.to_csv(output / "sampled_inputs.csv", index=False)
        np.savez_compressed(output / "activity.npz", time_s=trace.time_s,
            common_rates=trace.rates, legacy_hdeltab=legacy, exact_hdeltab=exact)
        inherited = []
        for representation in ["raw", "movement_evoked"]:
            pairs = pd.read_csv(movement / "baseline_1" / f"{representation}_pair_comparisons.csv")
            same = pairs[pairs.relation == "same_world_different_heading"]
            inherited.append(dict(representation=representation, pairs=len(same),
                mean_centered_cosine=float(same.centered_cosine.mean()),
                min_centered_cosine=float(same.centered_cosine.min())))
        pd.DataFrame(inherited).to_csv(output / "inherited_heading_differences.csv", index=False)
        save_json(output / "parameters.json", dict(parameters=asdict(p), source_ids=config["source_ids"],
            target_ids=config["target_ids"], rate_unit=network.rate_unit,
            assumptions="Legacy encoding/normalization retained; PFNs clamped; no new recurrence or memory",
            numerical_metrics=metrics))
        def render(axes):
            axes[0].plot(trace.time_s, trace.rates[:, :sources].mean(axis=1))
            axes[0].set_ylabel("PFN mean (a.u.)")
            axes[1].plot(trace.time_s, activity.mean(axis=1), label="Common solver")
            axes[1].plot(trace.time_s, legacy.mean(axis=1), "--", label="Existing solver")
            axes[1].set_ylabel("hDeltaB mean (a.u.)")
            axes[1].legend()
            axes[2].plot(trace.time_s, np.max(np.abs(activity - legacy), axis=1))
            axes[2].set(xlabel="Time (s)", ylabel="Max absolute difference")
        _plot(output, "pfn_comparison", render)
        (output / "summary.md").write_text(
            "# Common-framework PFN compatibility check\n\n"
            "Existing prescribed PFNd/PFNv→hDeltaB behavior is reproduced with the same body IDs, "
            "b=1, group normalization, multipliers, gain, time constant and continuous trajectory state. "
            "Rates remain arbitrary units, not calibrated Hz. No new recurrence is enabled in this compatibility case.\n\n"
            f"Maximum difference from the existing solver: {metrics['legacy_max_abs_error']:.6g}.\n\n"
            "Inherited heading differences are copied statistics from the saved independent movement grid, "
            "not newly fitted or repaired by this replay. See inherited_heading_differences.csv. "
            "Geometry is not fed into the dynamics.\n")
        for directory, record in records:
            verify_model_data(directory, record)
        return _finish(output, manifest, validation)
    except Exception as exc:
        _failed(output, manifest, exc)
        raise


def default_janke_config():
    return dict(parameters=asdict(JankeParameters()), input_mode="centered", dt_s=.005, rate_tau_s=.1, duration_s=20,
        reset_times_s=[0, 15], experiments=[
            dict(name="uniform_control", events=[]),
            dict(name="pulse", events=[dict(start_s=2, stop_s=5, amplitude=.05, phase_deg=0)]),
            dict(name="sustained", events=[dict(start_s=2, stop_s=12, amplitude=.05, phase_deg=0)]),
            dict(name="opposite_inputs", events=[dict(start_s=2, stop_s=7, amplitude=.05, phase_deg=0),
                dict(start_s=7, stop_s=12, amplitude=.05, phase_deg=180)])])


def _demo_inputs(config, p):
    from .rate_model import grid_index
    duration = grid_index(config["duration_s"], p.reference_dt_s)
    if duration < 1:
        raise ValidationError("Demo needs at least one reference interval.")
    resets = np.zeros(duration + 1, dtype=bool)
    for t in config["reset_times_s"]:
        if not 0 <= t <= config["duration_s"]:
            raise ValidationError("Reset outside demo duration.")
        resets[grid_index(t, p.reference_dt_s)] = True
    names, experiments = set(), []
    for case in config["experiments"]:
        name = case["name"]
        if not isinstance(name, str) or not name or not name.isascii() or not name.replace("_", "").isalnum() or name in names:
            raise ValidationError("Experiment names must be unique ASCII letters/digits/underscores.")
        names.add(name)
        values = np.ones((duration + 1, 8))
        for event in case["events"]:
            if (set(event) - {"start_s", "stop_s", "amplitude", "phase_deg", "uniform_offset"}
                    or not ({"amplitude", "uniform_offset"} & set(event))):
                raise ValidationError("Events require amplitude and/or uniform_offset, with supported fields only.")
            start, stop = event["start_s"], event["stop_s"]
            amplitude, phase = event.get("amplitude", 0.), event.get("phase_deg", 0.)
            uniform = event.get("uniform_offset", 0.)
            if not np.isfinite([start, stop, amplitude, phase, uniform]).all() or not 0 <= start < stop <= config["duration_s"] or amplitude < 0:
                raise ValidationError("Invalid prescribed demo event.")
            on, off = grid_index(start, p.reference_dt_s), grid_index(stop, p.reference_dt_s)
            values[on:off] += uniform + amplitude * np.cos(np.arange(8) * np.pi / 4 - np.deg2rad(phase))
        if (values < 0).any():
            raise ValidationError("Events produced negative prescribed column signals.")
        experiments.append((name, values))
    if not experiments:
        raise ValidationError("Provide at least one experiment.")
    return experiments, resets


def run_janke(output, config=None, config_path=None):
    config = dict(default_janke_config() if config is None else config)
    # Old saved configurations keep their original centered behavior.
    mode = config.setdefault("input_mode", "centered")
    if mode not in {"centered", "absolute"}:
        raise ValidationError("Candidate input_mode must be centered or absolute.")
    p = JankeParameters(**config["parameters"])
    p.validate()
    experiments, resets = _demo_inputs(config, p)
    output, manifest = _begin(output, "janke_reference_and_hypothetical_rate_candidate",
                              [] if config_path is None else [config_path])
    try:
        save_json(output / "config.json", config)
        rows = []
        for name, values in experiments:
            reference = discrete_reference(values, p, resets=resets)
            control = reference["fluorescence"] if mode == "centered" else absolute_input_control(values, p, resets=resets)["latent_output"]
            candidate = rate_candidate(values, p, dt_s=config["dt_s"], rate_tau_s=config["rate_tau_s"], resets=resets, input_mode=mode)
            fine = rate_candidate(values, p, dt_s=config["dt_s"] / 2, rate_tau_s=config["rate_tau_s"], resets=resets, input_mode=mode)
            reference_error = float(np.max(np.abs(candidate["latent_proxy"] - reference["fluorescence"])))
            control_error = float(np.max(np.abs(candidate["latent_proxy"] - control)))
            fine_error = relative_error(candidate["rates"], fine["rates"][::2])
            rows.append(dict(experiment=name, input_mode=mode, latent_reference_max_abs_error=reference_error,
                latent_control_max_abs_error=control_error,
                candidate_rate_half_dt_error=fine_error,
                rate_fraction_at_zero=float(np.mean(candidate["rates"] == 0)),
                reference_fraction_negative=float(np.mean(reference["fluorescence"] < 0))))
            np.savez_compressed(output / f"{name}_activity.npz", native_time_s=reference["time_s"],
                prescribed_column_signal=values, reference_mean=reference["mean"],
                reference_integrated=reference["integrated"], reference_fluorescence=reference["fluorescence"],
                time_s=candidate["time_s"], candidate_rate=candidate["rates"],
                candidate_excitability=candidate["excitability"], fluorescence_proxy=candidate["fluorescence_proxy"])
            np.savez_compressed(output / f"{name}_latent_control.npz", time_s=reference["time_s"],
                                candidate_latent_proxy=candidate["latent_proxy"], mathematical_control=control)
            for label, time, signal in [("reference_fluorescence", reference["time_s"], reference["fluorescence"]),
                                         ("candidate_rate", candidate["time_s"], candidate["rates"])]:
                readout = bump_readout(signal)
                readout.insert(0, "time_s", time)
                readout.to_csv(output / f"{name}_{label}_readout.csv", index=False)
            candidate["neurons"].to_csv(output / "idealized_nodes.csv", index=False)
            candidate["coupling"].to_csv(output / "hypothetical_column_coupling.csv", index=False)
            save_json(output / "candidate_assumptions.json", candidate["assumptions"])
            def render(axes):
                axes[0].step(reference["time_s"], values[:, 0] - 1, where="post")
                axes[0].set_ylabel("Column 0 input deviation")
                axes[1].plot(reference["time_s"], reference["fluorescence"])
                axes[1].set_ylabel("Reference fluorescence (a.u.)")
                axes[2].plot(candidate["time_s"], candidate["rates"])
                axes[2].set(xlabel="Time (s)", ylabel="Candidate rate (a.u.)")
            _plot(output, name, render)
        metrics = pd.DataFrame(rows)
        metrics.to_csv(output / "numerical_comparisons.csv", index=False)
        (output / "summary.md").write_text(
            "# Janke reference and candidate rate dynamics\n\n"
            "Controlled idealized eight-column inputs; no movement encoding, MaleCNS body mapping or raw anatomical weights. "
            "Figure 4.1 example coefficients are declared on a 1 s basis by default; this is not a fit or a reproduction of recorded flies.\n\n"
            f"Candidate input mode: **{mode}**. The thesis fluorescence reference always remains spatially centered. "
            "Absolute mode removes only the candidate's centering corrections; its signs, strengths and imposed baseline are unchanged. "
            "This is a new hypothesis, not an absolute-rate interpretation validated by the thesis.\n\n"
            "The reference is the signed discrete fluorescence recurrence. The candidate embeds its latent leak exactly "
            "under held inputs and adds a provisional nonnegative rate stage, using the common solver's slow excitability state. "
            "The mean ramp is imposed; centered mode also imposes a spatial mean correction. "
            "The signed latent proxy is checked against the matching discrete mathematical control. "
            "Its difference from the unchanged centered thesis reference is separately reported and is not an error gate in absolute mode. "
            "Neither agreement establishes firing-rate validity. "
            "Rates can differ because of rectification and added rate relaxation; zero fractions and negative-reference fractions are reported.\n\n"
            "Predictions: uniform input produces no spatial modulation. Centered mode cancels uniform input changes; "
            "absolute mode can change the population mean. The slow component decays by k2 per reference interval "
            "when its selected input is zero; direct input is not accumulated. Opposite inputs act on the residual state and do not imply "
            "perfect displacement cancellation. Reset overrides the predicted states/output; its timing convention is explicit.\n\n"
            "Numerical failure: latent proxy/control error >1e-12 or candidate half-dt error >=2%. "
            "No biological acceptance gate, home-vector claim, fitted decoder, learning or explicit network feedback is introduced. "
            "The source parameter/time-unit inconsistencies remain unresolved; do not reuse these coefficients at another timestep without conversion.\n")
        check_name = "latent_embedding_matches_reference" if mode == "centered" else "latent_embedding_matches_absolute_control"
        return _finish(output, manifest, {
            check_name: bool(metrics.latent_control_max_abs_error.le(1e-12).all()),
            "candidate_half_dt_error_below_2_percent": bool(metrics.candidate_rate_half_dt_error.lt(.02).all())})
    except Exception as exc:
        _failed(output, manifest, exc)
        raise


def verify(output):
    directory = Path(output)
    manifest = json.loads((directory / "framework_run.json").read_text())
    if manifest.get("status") != "complete":
        raise ValidationError("Framework run must be complete.")
    verify_model_data(directory, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pfn = sub.add_parser("pfn-reference", help="Replay the selected saved PFN trajectory through the common solver")
    pfn.add_argument("--movement", type=Path, required=True)
    pfn.add_argument("--trajectory", type=Path, required=True)
    pfn.add_argument("--output", type=Path, required=True)
    janke = sub.add_parser("janke-comparison", help="Controlled column-signal reference/candidate comparison")
    janke.add_argument("--config", type=Path)
    janke.add_argument("--input-mode", choices=["centered", "absolute"],
                       help="Override candidate input processing; the thesis reference stays centered")
    janke.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("verify", help="Verify saved output hashes offline")
    check.add_argument("source", type=Path)
    args = parser.parse_args()
    if args.command == "pfn-reference":
        result = run_pfn(args.movement, args.trajectory, args.output)
    elif args.command == "janke-comparison":
        config = json.loads(args.config.read_text()) if args.config else None
        if args.input_mode is not None:
            config = default_janke_config() if config is None else config
            config["input_mode"] = args.input_mode
        result = run_janke(args.output, config, args.config)
    else:
        result = verify(args.source)
    print(json.dumps(result["validation"], indent=2))


if __name__ == "__main__":
    main()
