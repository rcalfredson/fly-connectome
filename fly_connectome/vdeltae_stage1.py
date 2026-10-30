"""Body-aligned prescribed-input experiments; no movement encoding or home-vector decoder."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .connectivity import ValidationError
from .inputs import save_json, sha256
from .janke_reference import JankeParameters
from .rate_framework import RateNetwork, RateState, RateTrace
from .rate_model import grid_index, relative_error
from .vdeltae_contact_audit import csv_comparable, standard_mask, verify_contact_audit
from .vdeltae_spatial_audit import select_spatial_population
from .vdeltae_spatial_pilot import pilot_code_provenance, record_artifact_hashes

KIND = "vdeltae_hdeltag_stage1_v1"
SCENARIOS = ("fast_control", "slow_candidate")
READOUT_COLUMNS = [
    "population_mean", "peak", "trough", "spatial_rms", "delta_mean", "delta_spatial_rms",
    "slow_mean", "delta_slow_spatial_rms", "fraction_rate_at_zero", "fraction_rectified_target",
]


def default_config():
    return dict(parameters=asdict(JankeParameters()), dt_s=.005, sample_dt_s=.01,
                rate_tau_s=.1, per_contact_gain=.001, source_tonic_rate=0., total_increment=1.,
                onset_s=30., pulse_s=1., sustained_s=10., duration_s=70., numerical_tolerance=.02)


def prepare_config(config=None):
    defaults = default_config()
    config = {} if config is None else dict(config)
    if set(config) - set(defaults):
        raise ValidationError("Unknown stage-1 configuration fields.")
    config = {**defaults, **config}
    try:
        if not isinstance(config["parameters"], dict) or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)
                for value in config["parameters"].values()):
            raise ValidationError("Reference parameters must be finite numbers.")
        p = JankeParameters(**config["parameters"])
        p.validate()
    except (TypeError, ValueError) as error:
        raise ValidationError("Invalid reference parameter block.") from error
    config["parameters"] = asdict(p)
    for name in set(config) - {"parameters"}:
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValidationError(f"{name} must be a finite number.")
    positive = ["dt_s", "sample_dt_s", "rate_tau_s", "per_contact_gain", "total_increment",
                "onset_s", "pulse_s", "sustained_s", "duration_s", "numerical_tolerance"]
    if any(config[name] <= 0 for name in positive) or config["source_tonic_rate"] < 0:
        raise ValidationError("Use positive scales/times and nonnegative tonic source activity.")
    if not 0 < p.k1 < 1 or not 0 < p.k2 < 1 or p.a <= 0 or p.b <= 0 or p.c <= 0:
        raise ValidationError("This candidate requires positive increments and retention strictly between zero and one.")
    if config["dt_s"] > config["rate_tau_s"] / 20 + 1e-14:
        raise ValidationError("Stage 1 requires at least 20 integration steps per rate time constant.")
    if config["onset_s"] < 5 * config["rate_tau_s"]:
        raise ValidationError("Allow at least five rate time constants before stimulation.")
    if config["pulse_s"] > config["sustained_s"] or config["numerical_tolerance"] >= 1:
        raise ValidationError("Pulse duration must not exceed sustained duration; tolerance must be below one.")
    slow_tau = -p.reference_dt_s / np.log(p.k2)
    if config["duration_s"] < config["onset_s"] + config["sustained_s"] + 5 * slow_tau:
        raise ValidationError("Require at least five slow time constants after sustained stimulation.")
    stride = grid_index(config["sample_dt_s"], config["dt_s"])
    if stride < 1:
        raise ValidationError("Recording interval must be an integer multiple of the integration step.")
    for name in ["onset_s", "pulse_s", "sustained_s", "duration_s"]:
        grid_index(config[name], config["sample_dt_s"])
    return config, p


def load_stage1_source(source):
    source = Path(source).resolve()
    before = sha256(source / "contacts.json")
    record = verify_contact_audit(source)
    bodies = pd.read_csv(source / "selected_bodies.csv")
    bodies = select_spatial_population(bodies)
    contacts = pd.read_csv(source / "contacts_raw.csv", low_memory=False)
    if sha256(source / "contacts.json") != before:
        raise ValidationError("Contact source changed while loading stage 1.")
    return bodies, contacts, record, before


def projection_counts(bodies, contacts, threshold):
    selected = contacts.loc[standard_mask(contacts, threshold) & contacts.FB_post.eq(True)]
    if not selected.compartment_post.isin(["axon", "dendrite"]).all():
        raise ValidationError("Selected contacts have unassigned receiving compartments; specify their treatment first.")
    index = pd.MultiIndex.from_product([
        sorted(bodies.loc[bodies.type.eq("vDeltaE"), "bodyId"]),
        sorted(bodies.loc[bodies.type.eq("hDeltaG"), "bodyId"]), ["axon", "dendrite"],
    ], names=["bodyId_pre", "bodyId_post", "compartment_post"])
    counts = selected.groupby(index.names).size().reindex(index, fill_value=0).rename("contacts").reset_index()
    if int(counts.contacts.sum()) != len(selected):
        raise ValidationError("Contact allocation failed conservation.")
    return counts


def build_network(bodies, counts, config, scenario):
    if scenario not in SCENARIOS:
        raise ValidationError("Unknown stage-1 scenario.")
    p = JankeParameters(**config["parameters"])
    slow_tau = -p.reference_dt_s / np.log(p.k2)
    neurons = bodies[["bodyId", "type", "instance", "annotation_column"]].copy()
    neurons["neuron_id"] = neurons.bodyId.astype(str)
    neurons["cell_type"] = neurons.type
    neurons["tau_s"] = config["rate_tau_s"]
    neurons["slow_tau_s"] = np.where(neurons.type.eq("hDeltaG") & (scenario == "slow_candidate"), slow_tau, 0.)
    neurons["bias"], neurons["threshold"], neurons["gain"] = 0., 0., 1.
    coupling = counts.copy()
    dendrite = coupling.compartment_post.eq("dendrite")
    coupling["source_id"] = coupling.bodyId_pre.astype(str)
    coupling["target_id"] = coupling.bodyId_post.astype(str)
    coupling["channel"] = np.where(dendrite & (scenario == "slow_candidate"), "excitability", "rate")
    coupling["sign"] = np.where(dendrite, -1, 1)
    coupling["strength"] = config["per_contact_gain"] * np.where(dendrite, p.a / (1 - p.k2), p.b)
    coupling["effective_weight"] = coupling.contacts * coupling.sign * coupling.strength
    coupling["normalization"] = "none"
    coupling["sign_evidence"] = "assumed"
    coupling["sign_detail"] = np.where(dendrite,
        "Hypothesized negative receiving-dendrite effect; slow candidate and DC-matched fast control.",
        "Hypothesized positive fast effect onto receiving axonal compartment.")
    # Two disjoint anatomical components may share a pair/channel in the fast control.
    # Retain both signed rows rather than duplicate a contact or erase its provenance.
    network = RateNetwork(neurons, coupling, rate_unit="arbitrary_rate_units")
    return network


def experiment_specs(bodies, config):
    ids = sorted(int(v) for v in bodies.loc[bodies.type.eq("vDeltaE"), "bodyId"])
    specs = [dict(name="tonic_control", kind="control", body_ids=[], start_s=None, stop_s=None)]
    for kind in ["pulse", "sustained"]:
        stop = config["onset_s"] + config[f"{kind}_s"]
        for body in ids:
            specs.append(dict(name=f"body_{body}_{kind}", kind=kind, body_ids=[body],
                              start_s=config["onset_s"], stop_s=stop))
        specs.append(dict(name=f"population_{kind}", kind=kind, body_ids=ids,
                          start_s=config["onset_s"], stop_s=stop))
    return specs


def prescribed_inputs(times, source_ids, spec, config):
    values = np.full((len(times), len(source_ids)), config["source_tonic_rate"], dtype=float)
    if spec["body_ids"]:
        active = (times >= spec["start_s"] - 1e-12) & (times < spec["stop_s"] - 1e-12)
        columns = [source_ids.index(body) for body in spec["body_ids"]]
        values[np.ix_(active, columns)] += config["total_increment"] / len(columns)
    return values


def baseline(times, config):
    p = JankeParameters(**config["parameters"])
    return p.c / (1 - p.k1) * -np.expm1(np.asarray(times) * np.log(p.k1) / p.reference_dt_s)


def solve_trial(network, config, spec, *, half_dt=False):
    dt = config["dt_s"] / (2 if half_dt else 1)
    times = np.arange(grid_index(config["duration_s"], dt) + 1) * dt
    source_ids = [int(v) for v in network.neurons.loc[network.neurons.cell_type.eq("vDeltaE"), "bodyId"]]
    values = prescribed_inputs(times, source_ids, spec, config)
    targets = network.neurons.cell_type.eq("hDeltaG").to_numpy()
    drive = np.zeros((len(times), len(network.neuron_ids)))
    drive[:, targets] = baseline(times, config)[:, None]
    trace = network.solve(drive, dt, prescribed={str(body): values[:, i] for i, body in enumerate(source_ids)},
                          record_every=grid_index(config["sample_dt_s"], dt), method="exponential_euler")
    return trace


def diagnostics(network, trace, control_rates, control_slow, config):
    target = np.flatnonzero(network.neurons.cell_type.eq("hDeltaG"))
    rates, slow = trace.rates[:, target], trace.excitability[:, target]
    delta, delta_slow = rates - control_rates[:, target], slow - control_slow[:, target]
    fast = np.zeros_like(rates)
    lookup = {int(index): i for i, index in enumerate(target)}
    for row in network.coupling.loc[network.coupling.channel.eq("rate")].itertuples():
        fast[:, lookup[network.index[row.target_id]]] += row.effective_weight * trace.rates[:, network.index[row.source_id]]
    preactivation = baseline(trace.time_s, config)[:, None] + fast + slow
    spatial = lambda array: np.sqrt(np.mean((array - array.mean(axis=1, keepdims=True)) ** 2, axis=1))
    readouts = np.column_stack([
        rates.mean(axis=1), rates.max(axis=1), rates.min(axis=1), spatial(rates),
        delta.mean(axis=1), spatial(delta), slow.mean(axis=1), spatial(delta_slow),
        (rates == 0).mean(axis=1), (preactivation <= 0).mean(axis=1),
    ])
    return delta, delta_slow, readouts, preactivation


def trial_metrics(network, trace, fine, controls, config, spec, scenario):
    delta, delta_slow, readouts, preactivation = diagnostics(network, trace, *controls[0], config)
    fine_delta, fine_slow, _, _ = diagnostics(network, fine, *controls[1], config)
    if not np.array_equal(trace.time_s, fine.time_s):
        raise ValidationError("Coarse and fine recording grids differ.")
    targets = network.neurons.cell_type.eq("hDeltaG").to_numpy()
    if spec["kind"] == "control":
        rate_error = relative_error(trace.rates[:, targets], fine.rates[:, targets])
        slow_error = relative_error(trace.excitability[:, targets], fine.excitability[:, targets])
    else:
        rate_error, slow_error = relative_error(delta, fine_delta), relative_error(delta_slow, fine_slow)
    tail_error = 0.
    if scenario == "slow_candidate" and spec["kind"] != "control":
        off = grid_index(spec["stop_s"], config["sample_dt_s"])
        tau = network.slow_tau_s[np.flatnonzero(targets)[0]]
        expected = delta_slow[off] * np.exp(-(trace.time_s[off:] - trace.time_s[off])[:, None] / tau)
        tail_error = float(np.max(np.abs(delta_slow[off:] - expected)))
    validation = dict(scenario=scenario, experiment=spec["name"],
                      rate_half_dt_relative_error=rate_error, slow_half_dt_relative_error=slow_error,
                      post_stimulus_slow_decay_max_abs_error=tail_error)
    body_rows = []
    start = grid_index(config["onset_s"], config["sample_dt_s"])
    off = None if spec["stop_s"] is None else grid_index(spec["stop_s"], config["sample_dt_s"])
    for i, body in enumerate(network.neurons.loc[targets].itertuples()):
        row = dict(scenario=scenario, experiment=spec["name"], kind=spec["kind"], bodyId=int(body.bodyId),
                   annotation_column=body.annotation_column, peak_delta=float(delta[start:, i].max()),
                   trough_delta=float(delta[start:, i].min()),
                   delta_at_offset=float(delta[off, i]) if off is not None else np.nan,
                   slow_delta_at_offset=float(delta_slow[off, i]) if off is not None else np.nan,
                   final_delta=float(delta[-1, i]), final_slow_delta=float(delta_slow[-1, i]),
                   fraction_rate_at_zero=float((trace.rates[start:, np.flatnonzero(targets)[i]] == 0).mean()),
                   fraction_rectified_target=float((preactivation[start:, i] <= 0).mean()))
        for delay in [1., 5.]:
            name = f"delta_{int(delay)}s_after_offset"
            checkpoint = None if spec["stop_s"] is None else spec["stop_s"] + delay
            row[name] = (float(delta[grid_index(checkpoint, config["sample_dt_s"]), i])
                         if checkpoint is not None and checkpoint <= config["duration_s"] else np.nan)
        body_rows.append(row)
    return validation, body_rows, readouts


def _protection(source, source_record, output):
    paths = [source, Path(source_record["sources"]["coordinates"]), Path(source_record["sources"]["census"])]
    geometry = json.loads((paths[1] / "coordinates.json").read_text())
    inventory = geometry["source_inventory"]
    paths += [Path(geometry["source_probe"]), Path(inventory["annotation_audit_directory"])]
    paths += [Path(item["source_manifest_file"]).parent for item in inventory["sources"].values()]
    # All source ancestors already exist; fresh mkdir also prevents replacing them.
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in paths):
        raise ValidationError("Stage-1 outputs must be separate from source directories.")


def run_stage1(source, output, config=None, *, config_path=None):
    config, _ = prepare_config(config)
    source, output = Path(source).resolve(), Path(output).resolve()
    bodies, contacts, source_record, source_hash = load_stage1_source(source)
    _protection(source, source_record, output)
    threshold = source_record["standard_confidence_threshold"]
    counts = projection_counts(bodies, contacts, threshold)
    specs = experiment_specs(bodies, config)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(kind=KIND, status="running", started_at_utc=datetime.now(timezone.utc).isoformat(),
                    source=str(source), source_manifest_sha256=source_hash,
                    source_config=None if config_path is None else str(Path(config_path).resolve()),
                    source_config_sha256=None if config_path is None else sha256(config_path),
                    assumptions=dict(rate_unit="arbitrary_rate_units", input_centering=False,
                        normalization="none; one fixed global per-contact scale",
                        signs="assumed: positive fast receiving-axon; negative receiving-dendrite",
                        control="Dendritic steady-state gain matched; fast versus slow channel only.",
                        baseline="Imposed Figure 4.1 demonstration ramp; not generated by modeled circuitry.",
                        initial_state="Zero dynamic rates and slow states; sources clamped to declared tonic activity.",
                        feedback="Omitted; prescribed sources cannot receive feedback.",
                        resets="None after initialization", movement_encoding="None", angular_readout="None"))
    save_json(output / "stage1.json", manifest)
    try:
        manifest.update(pilot_code_provenance())
        save_json(output / "config.json", config)
        save_json(output / "experiments.json", specs)
        bodies.to_csv(output / "selected_bodies.csv", index=False)
        counts.to_csv(output / "projection_counts.csv", index=False)
        allocation = contacts[["contact_relationship_id", "bodyId_pre", "bodyId_post", "compartment_post"]].copy()
        allocation["standard_confidence"] = standard_mask(contacts, threshold)
        allocation["receiving_FB_marked"] = contacts.FB_post
        allocation["selected"] = allocation.standard_confidence & contacts.FB_post.eq(True)
        allocation.to_csv(output / "contact_allocation.csv", index=False)
        validations, responses = [], []
        for scenario in SCENARIOS:
            network = build_network(bodies, counts, config, scenario)
            network.neurons.to_csv(output / f"{scenario}_neurons.csv", index=False)
            network.coupling.to_csv(output / f"{scenario}_coupling.csv", index=False)
            coarse_control = solve_trial(network, config, specs[0])
            fine_control = solve_trial(network, config, specs[0], half_dt=True)
            controls = [(coarse_control.rates, coarse_control.excitability),
                        (fine_control.rates, fine_control.excitability)]
            for spec in specs:
                trace = coarse_control if spec["kind"] == "control" else solve_trial(network, config, spec)
                fine = fine_control if spec["kind"] == "control" else solve_trial(network, config, spec, half_dt=True)
                validation, rows, readouts = trial_metrics(network, trace, fine, controls, config, spec, scenario)
                validations.append(validation)
                responses.extend(rows)
                np.savez_compressed(output / f"{scenario}__{spec['name']}.npz",
                    time_s=trace.time_s, neuron_ids=np.array(trace.neuron_ids), rates=trace.rates,
                    excitability=trace.excitability, fine_rates=fine.rates, fine_excitability=fine.excitability,
                    readout_columns=np.array(READOUT_COLUMNS), readouts=readouts,
                    imposed_baseline=baseline(trace.time_s, config))
            print(f"Stage 1: {scenario}, {len(specs)} trials saved.", flush=True)
        metrics, response = pd.DataFrame(validations), pd.DataFrame(responses)
        metrics.to_csv(output / "numerical_validation.csv", index=False)
        response.to_csv(output / "response_by_body.csv", index=False)
        valid = (metrics.rate_half_dt_relative_error.le(config["numerical_tolerance"]).all()
                 and metrics.slow_half_dt_relative_error.le(config["numerical_tolerance"]).all()
                 and metrics.post_stimulus_slow_decay_max_abs_error.le(1e-10).all())
        if not valid:
            raise ValidationError("Stage-1 convergence or declared slow-state decay check failed.")
        verify_contact_audit(source)
        if sha256(source / "contacts.json") != source_hash:
            raise ValidationError("Contact source changed during simulation.")
        if config_path is not None and sha256(config_path) != manifest["source_config_sha256"]:
            raise ValidationError("Configuration changed during simulation.")
        (output / "summary.md").write_text("\n".join([
            "# Body-aligned vDeltaE → hDeltaG stage 1", "",
            f"{len(specs)} stimuli × two scenarios; {int(counts.contacts.sum())} allocated contacts.",
            "Prescribed absolute source rates; total added activity matched across single-body/population stimuli.",
            "Pulse and sustained conditions have different total doses; durations are not dose-matched.",
            "Both scenarios use the same signed steady-state anatomical gains and imposed rising baseline.",
            "The slow candidate assigns a phenomenological negative excitability state to receiving dendritic contacts.",
            "Numerical convergence and post-stimulus slow-state leak checks passed; no biological acceptance test is claimed.",
            "Body response CSV reports raw and time-matched-control-relative behavior; NPZ files retain all rates/slow states and fine-step traces.",
            "Population-mean subtraction is analysis only; no input centering, incoming normalization or angular decoder.",
            "Rectified target fractions describe the ReLU argument, separately from actual rate-zero fractions.",
            "No fitted trajectories, movement encoding, plasticity, feedback or home-vector claim.", "",
        ]))
        manifest.update(status="complete", numerical_validated=True)
    except Exception as error:
        manifest.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["artifact_sha256"] = record_artifact_hashes(output, "stage1.json")
        save_json(output / "stage1.json", manifest)
    return manifest


def _same_table(actual, expected, name):
    actual, expected = actual.copy(), expected.copy()
    for column in ["neuron_id", "source_id", "target_id"]:
        if column in expected:
            actual[column] = actual[column].astype(str)
            expected[column] = expected[column].astype(str)
    try:
        pd.testing.assert_frame_equal(csv_comparable(actual), csv_comparable(expected),
                                      check_dtype=False, atol=1e-12, rtol=1e-10)
    except AssertionError as error:
        raise ValidationError(f"Saved {name} differs from its reconstructed specification.") from error


def _saved_trace(data, network, fine=False):
    prefix = "fine_" if fine else ""
    rates, slow = data[prefix + "rates"], data[prefix + "excitability"]
    return RateTrace(data["time_s"], network.neuron_ids, rates, slow,
                     RateState(network.neuron_ids, rates[-1], slow[-1]))


def verify_stage1(directory):
    """Recheck anatomy, hashes, stimulus clamps and derived metrics without rerunning dynamics."""
    directory = Path(directory).resolve()
    record = json.loads((directory / "stage1.json").read_text())
    if record.get("kind") != KIND or record.get("status") != "complete" or not record.get("numerical_validated"):
        raise ValidationError("Stage-1 run is not complete and numerically validated.")
    if record.get("artifact_sha256") != record_artifact_hashes(directory, "stage1.json"):
        raise ValidationError("Stage-1 artifact hashes or file inventory changed.")
    bodies, contacts, source, source_hash = load_stage1_source(record["source"])
    if source_hash != record["source_manifest_sha256"]:
        raise ValidationError("Stage-1 contact source manifest changed.")
    config, _ = prepare_config(json.loads((directory / "config.json").read_text()))
    specs = experiment_specs(bodies, config)
    if specs != json.loads((directory / "experiments.json").read_text()):
        raise ValidationError("Saved stimulus specification changed.")
    counts = projection_counts(bodies, contacts, source["standard_confidence_threshold"])
    _same_table(pd.read_csv(directory / "selected_bodies.csv"), bodies, "body inventory")
    _same_table(pd.read_csv(directory / "projection_counts.csv"), counts, "contact counts")
    allocation = contacts[["contact_relationship_id", "bodyId_pre", "bodyId_post", "compartment_post"]].copy()
    allocation["standard_confidence"] = standard_mask(contacts, source["standard_confidence_threshold"])
    allocation["receiving_FB_marked"] = contacts.FB_post
    allocation["selected"] = allocation.standard_confidence & contacts.FB_post.eq(True)
    _same_table(pd.read_csv(directory / "contact_allocation.csv"), allocation, "contact allocation")
    validations, responses = [], []
    times = np.arange(grid_index(config["duration_s"], config["sample_dt_s"]) + 1) * config["sample_dt_s"]
    expected_files = {"config.json", "experiments.json", "selected_bodies.csv", "projection_counts.csv",
                      "contact_allocation.csv", "summary.md", "numerical_validation.csv", "response_by_body.csv"}
    for scenario in SCENARIOS:
        network = build_network(bodies, counts, config, scenario)
        for suffix, table in [("neurons", network.neurons), ("coupling", network.coupling)]:
            filename = f"{scenario}_{suffix}.csv"
            expected_files.add(filename)
            _same_table(pd.read_csv(directory / filename), table, filename)
        source_mask = network.neurons.cell_type.eq("vDeltaE").to_numpy()
        source_ids = network.neurons.loc[source_mask, "bodyId"].astype(int).tolist()
        controls = None
        for spec in specs:
            filename = f"{scenario}__{spec['name']}.npz"
            expected_files.add(filename)
            with np.load(directory / filename, allow_pickle=False) as data:
                if (tuple(data["neuron_ids"]) != network.neuron_ids
                        or list(data["readout_columns"]) != READOUT_COLUMNS
                        or not np.allclose(data["time_s"], times, rtol=0, atol=1e-12)):
                    raise ValidationError("Saved trace identity or recording grid changed.")
                expected_inputs = prescribed_inputs(times, source_ids, spec, config)
                for prefix in ["", "fine_"]:
                    rates, slow = data[prefix + "rates"], data[prefix + "excitability"]
                    if (rates.shape != (len(times), len(network.neuron_ids)) or slow.shape != rates.shape
                            or not np.isfinite(rates).all() or not np.isfinite(slow).all()
                            or (rates < 0).any() or np.any(rates[0, ~source_mask] != 0)
                            or np.any(slow[0] != 0) or np.any(slow[:, source_mask] != 0)
                            or not np.array_equal(rates[:, source_mask], expected_inputs)):
                        raise ValidationError("Saved rates, initial states or prescribed inputs are invalid.")
                    if scenario == "fast_control" and np.any(slow != 0):
                        raise ValidationError("Fast control must have no slow-state activity.")
                if not np.allclose(data["imposed_baseline"], baseline(times, config), atol=1e-14, rtol=0):
                    raise ValidationError("Saved imposed baseline changed.")
                trace, fine = _saved_trace(data, network), _saved_trace(data, network, True)
                if controls is None:
                    controls = [(trace.rates.copy(), trace.excitability.copy()),
                                (fine.rates.copy(), fine.excitability.copy())]
                validation, rows, readouts = trial_metrics(network, trace, fine, controls, config, spec, scenario)
                if not np.allclose(data["readouts"], readouts, atol=1e-12, rtol=1e-10):
                    raise ValidationError("Saved readouts differ from raw states.")
                validations.append(validation)
                responses.extend(rows)
    if expected_files != set(record["artifact_sha256"]):
        raise ValidationError("Unexpected stage-1 artifact inventory.")
    metrics = pd.DataFrame(validations)
    _same_table(pd.read_csv(directory / "numerical_validation.csv"), metrics, "numerical validation")
    _same_table(pd.read_csv(directory / "response_by_body.csv"), pd.DataFrame(responses), "body responses")
    if (not metrics.rate_half_dt_relative_error.le(config["numerical_tolerance"]).all()
            or not metrics.slow_half_dt_relative_error.le(config["numerical_tolerance"]).all()
            or not metrics.post_stimulus_slow_decay_max_abs_error.le(1e-10).all()):
        raise ValidationError("Saved numerical checks fail.")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--contacts", required=True)
    run.add_argument("--config")
    run.add_argument("--output", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--run", required=True)
    export = sub.add_parser("export")
    export.add_argument("--run", required=True)
    export.add_argument("--output", required=True)
    verify_figures = sub.add_parser("verify-figures")
    verify_figures.add_argument("--run", required=True)
    args = parser.parse_args()
    if args.command == "run":
        config = None if args.config is None else json.loads(Path(args.config).read_text())
        run_stage1(args.contacts, args.output, config, config_path=args.config)
    elif args.command == "verify":
        verify_stage1(args.run)
        print("Stage-1 saved run verified.")
    elif args.command == "export":
        from .vdeltae_stage1_figures import export_stage1_figures
        export_stage1_figures(args.run, args.output)
    else:
        from .vdeltae_stage1_figures import verify_stage1_figures
        verify_stage1_figures(args.run)
        print("Stage-1 figures verified.")


if __name__ == "__main__":
    main()
