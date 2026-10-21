"""Prescribed literature-based PFN encoding; independent movement comparisons."""

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .connectivity import DATASET, SERVER, ValidationError
from .pfn import load_pfn_network
from .rate_model import grid_index, relative_error, save_json, sha256, verify_model_data


PAPER = "https://doi.org/10.1038/s41586-021-04191-x"
METHODS = "https://wilson.hms.harvard.edu/sites/g/files/omnuum8421/files/wilson-lab/files/lu_et_al.pdf"
HEADINGS = np.arange(0, 360, 45)
PREFERRED_TRANSLATION = {("PFNd", "L"): -31., ("PFNd", "R"): 31.,
                         ("PFNv", "L"): 137., ("PFNv", "R"): -137.}


@dataclass(frozen=True)
class MovementParameters:
    tau_s: float = .1
    dt_s: float = .001
    gain: float = .001
    pfnd_weight_multiplier: float = 1.
    pfnv_weight_multiplier: float = 1.
    normalize_incoming_by_group: bool = False
    duration_s: float = 2.
    onset_s: float = .2
    offset_s: float = 1.2
    analysis_start_s: float = 1.
    analysis_stop_s: float = 1.2
    reference_speed_mm_s: float = 1.
    velocity_modulation: float = 5.
    null_repeats: int = 199
    seed: int = 20261020

    def validate(self):
        if type(self.normalize_incoming_by_group) is not bool:
            raise ValidationError("Incoming normalization must be a boolean.")
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValidationError("Movement parameters must be finite.")
        if min(self.tau_s, self.dt_s, self.duration_s, self.reference_speed_mm_s) <= 0:
            raise ValidationError("Time and reference speed must be positive.")
        if min(self.gain, self.velocity_modulation, self.pfnd_weight_multiplier, self.pfnv_weight_multiplier) < 0 or self.dt_s > self.tau_s / 20:
            raise ValidationError("Use nonnegative gains and dt <= tau/20.")
        if not 0 <= self.onset_s < self.analysis_start_s < self.analysis_stop_s <= self.offset_s < self.duration_s:
            raise ValidationError("Analysis must follow onset and end no later than offset, before trial end.")
        if type(self.null_repeats) is not int or self.null_repeats < 1 or type(self.seed) is not int or self.seed < 0:
            raise ValidationError("Use a positive integer null count and nonnegative integer seed.")
        for t in [self.duration_s, self.onset_s, self.offset_s, self.analysis_start_s, self.analysis_stop_s]:
            grid_index(t, self.dt_s)


def wrap_degrees(angle):
    return (np.asarray(angle) + 180) % 360 - 180


def heading_mapping(orders, source_ids):
    """Fixed PB map, not fitted to hDeltaB responses or PFN connection weights."""
    table = pd.concat([orders[t] for t in ["PFNd", "PFNv"]]).set_index("bodyId").loc[source_ids].copy()
    if not table.pb_side.isin(["L", "R"]).all() or not table.pb_glomerulus.isin(range(2, 10)).all():
        raise ValidationError("Expected receiving PB L/R2–9 annotations for every PFN.")
    sign = np.where(table.pb_side.eq("L"), 1., -1.)
    table["preferred_heading_deg"] = wrap_degrees(sign * (table.pb_glomerulus.to_numpy(dtype=float) - 5) * 45)
    table["preferred_translation_deg"] = [PREFERRED_TRANSLATION[(t, side)] for t, side in zip(table.type, table.pb_side)]
    table["heading_map_source"] = "Lu et al. 2022 Methods: L5/R5=0; L7/R3=+90; 45-degree steps, wrapped"
    return table.reset_index()


def encode_pfn(mapping, heading_deg, forward_mm_s, right_mm_s, baseline, parameters):
    """Arbitrary rate units; prescribed activity, not current into PFN dynamics."""
    heading = .5 * (1 + np.cos(np.deg2rad(heading_deg - mapping.preferred_heading_deg.to_numpy())))
    angle = np.deg2rad(mapping.preferred_translation_deg.to_numpy())
    projection = forward_mm_s * np.cos(angle) + right_mm_s * np.sin(angle)
    return heading * (baseline + parameters.velocity_modulation * np.maximum(0., projection / parameters.reference_speed_mm_s))


def movement_weights(counts, mapping, parameters):
    """Optionally normalize each target's incoming counts within PFN type/PB side."""
    id_column = "bodyId" if "bodyId" in mapping else "bodyId_pre"
    aligned = mapping.set_index(id_column).loc[counts.index]
    types = aligned.type
    scales = types.map({"PFNd": parameters.pfnd_weight_multiplier,
                       "PFNv": parameters.pfnv_weight_multiplier})
    if scales.isna().any():
        raise ValidationError("Movement weights require PFNd/PFNv source types.")
    weights = counts.to_numpy(dtype=float).copy()
    if parameters.normalize_incoming_by_group:
        if not aligned.pb_side.isin(["L", "R"]).all():
            raise ValidationError("Incoming normalization requires PFN PB sides L/R.")
        for positions in aligned.groupby(["type", "pb_side"]).indices.values():
            group = weights[positions]
            totals = group.sum(axis=0, keepdims=True)
            # No incoming contacts: retain zeros rather than inventing a connection.
            weights[positions] = np.divide(group, totals, out=np.zeros_like(group), where=totals > 0)
    return weights * parameters.gain * scales.to_numpy()[:, None]


def weight_description(config):
    p = config["parameters"]
    prefix = "Group-normalized; " if p.get("normalize_incoming_by_group", False) else ""
    return prefix + f"PFNd weights ×{p.get('pfnd_weight_multiplier', 1):g}; PFNv ×{p.get('pfnv_weight_multiplier', 1):g}"


def movement_conditions():
    rows = []
    for heading in HEADINGS:
        rows.append({"trial": f"rest_heading{heading:03d}", "heading_deg": int(heading),
                     "world_direction_deg": None, "body_direction_deg": None,
                     "speed_mm_s": 0., "forward_mm_s": 0., "right_mm_s": 0., "primary_grid": False})
        for world in HEADINGS:
            speeds = [.5, 1., 2.] if heading == world == 0 else [1.]
            body = float(wrap_degrees(world - heading))
            for speed in speeds:
                angle = np.deg2rad(body)
                rows.append({"trial": f"world{world:03d}_heading{heading:03d}_speed{speed:g}".replace(".", "p"),
                             "heading_deg": int(heading), "world_direction_deg": int(world),
                             "body_direction_deg": body, "speed_mm_s": speed,
                             "forward_mm_s": float(speed * np.cos(angle)), "right_mm_s": float(speed * np.sin(angle)),
                             "primary_grid": speed == 1.})
    return pd.DataFrame(rows)


def solve_output(baseline_input, movement_increment, parameters):
    """Euler plus independent continuous-time solution of a single rate stage."""
    p = parameters
    n = grid_index(p.duration_s, p.dt_s)
    times = np.arange(n + 1) * p.dt_s
    on, off = grid_index(p.onset_s, p.dt_s), grid_index(p.offset_s, p.dt_s)
    drive = np.broadcast_to(baseline_input, (n + 1, len(baseline_input))).copy()
    drive[on:off] += movement_increment
    activity = np.zeros_like(drive)
    for k in range(n):
        activity[k + 1] = activity[k] + p.dt_s / p.tau_s * (-activity[k] + np.maximum(0., drive[k]))

    def step(boundary):
        return -np.expm1(-np.maximum(0., times - boundary) / p.tau_s)

    exact = step(0)[:, None] * baseline_input + (step(p.onset_s) - step(p.offset_s))[:, None] * movement_increment
    return times, activity, exact


def normalized_patterns(profiles, *, centered):
    values = np.asarray(profiles, dtype=float)
    if centered:
        values = values - values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(values, axis=1)
    valid = norms > 1e-12
    unit = np.zeros_like(values)
    np.divide(values, norms[:, None], out=unit, where=valid[:, None])
    similarity = np.clip(unit @ unit.T, -1, 1)
    similarity[~(valid[:, None] & valid[None, :])] = np.nan
    return similarity, valid


def pattern_comparison(profiles, conditions, *, include_pairs=True):
    """No alignment, decoder fitting, angular assignment, or neuron permutation."""
    similarity, valid = normalized_patterns(profiles, centered=True)
    raw_cosine, _ = normalized_patterns(profiles, centered=False)
    headings = conditions.heading_deg.to_numpy()
    worlds = conditions.world_direction_deg.to_numpy()
    triangle = np.triu(np.ones_like(similarity, dtype=bool), 1)
    same = triangle & (worlds[:, None] == worlds[None, :]) & (headings[:, None] != headings[None, :])
    different = triangle & (worlds[:, None] != worlds[None, :]) & (headings[:, None] == headings[None, :])

    def mean(mask):
        values = similarity[mask]
        return float(np.nanmean(values)) if np.isfinite(values).any() else None

    within, between = mean(same), mean(different)
    result = {"same_world_mean_centered_cosine": within,
              "different_world_mean_centered_cosine": between,
              "contrast": within - between if within is not None and between is not None else None,
              "valid_directional_patterns": int(valid.sum()), "grid_conditions": len(profiles),
              "same_world_valid_pairs": int(np.isfinite(similarity[same]).sum()),
              "different_world_valid_pairs": int(np.isfinite(similarity[different]).sum())}
    rows = []
    if not include_pairs:
        return result, pd.DataFrame(), similarity
    for relation, mask in [("same_world_different_heading", same), ("different_world_same_heading", different)]:
        for i, j in zip(*np.where(mask)):
            rows.append({"trial_a": conditions.iloc[i].trial, "trial_b": conditions.iloc[j].trial,
                         "relation": relation, "centered_cosine": similarity[i, j], "raw_cosine": raw_cosine[i, j]})
    return result, pd.DataFrame(rows), similarity


def shuffled_sources(mapping, rng):
    """Reassign source rows independently within each PFN type × PB side."""
    permutation = np.arange(len(mapping))
    for _, group in mapping.groupby(["type", "pb_side"], sort=True):
        indices = group.index.to_numpy()
        permutation[indices] = rng.permutation(indices)
    return permutation


def shuffled_labels(conditions, rng):
    shuffled = conditions.copy()
    for _, group in conditions.groupby("heading_deg", sort=True):
        shuffled.loc[group.index, "world_direction_deg"] = rng.permutation(group.world_direction_deg.to_numpy())
    return shuffled


def render_movement(source, output):
    source, output = Path(source), Path(output)
    config = json.loads((source / "movement_config.json").read_text())
    conditions = pd.read_csv(source / "conditions.csv")
    target_order = pd.read_csv(source / "hDeltaB_anatomical_order.csv")
    target_columns = [str(body) for body in target_order.bodyId]
    grid = conditions[conditions.primary_grid].reset_index(drop=True)
    for baseline in config["baseline_levels"]:
        tag = "baseline_" + f"{baseline:g}".replace(".", "p")
        destination = output / tag
        destination.mkdir(parents=True, exist_ok=True)
        raw = pd.read_csv(source / tag / "steady_activity.csv", index_col=0)
        evoked = pd.read_csv(source / tag / "movement_evoked_activity.csv", index_col=0)
        selections = {
            "world_invariance": conditions[(conditions.world_direction_deg == 0) & (conditions.speed_mm_s == 1)],
            "heading_sweep": conditions[(conditions.body_direction_deg == 0) & (conditions.speed_mm_s == 1)],
            "body_direction_sweep": conditions[(conditions.heading_deg == 0) & conditions.primary_grid],
            "speed_sweep": conditions[(conditions.heading_deg == 0) & ((conditions.world_direction_deg == 0) | (conditions.speed_mm_s == 0))].sort_values("speed_mm_s"),
        }
        for name, selected in selections.items():
            fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
            try:
                for ax, frame, label in [(axes[0], raw, "Raw hDeltaB activity"), (axes[1], evoked, "Movement-evoked activity (matched rest subtracted)")]:
                    values = frame.loc[selected.trial, target_columns].to_numpy()
                    image = ax.imshow(values, aspect="auto", vmin=0, vmax=max(float(values.max()), 1e-12))
                    ax.set_yticks(range(len(selected)), [f"heading {r.heading_deg:g}°; body {r.body_direction_deg:g}°; speed {r.speed_mm_s:g}" if r.speed_mm_s else f"heading {r.heading_deg:g}°; stationary" for r in selected.itertuples()], fontsize=8)
                    ax.set_xticks(range(len(target_order)), target_order.label, rotation=90, fontsize=6)
                    ax.set_title(label)
                    fig.colorbar(image, ax=ax, label="Activity (a.u.; scale per panel)")
                fig.suptitle(f"{name} | PFN encoding baseline {baseline:g}; {weight_description(config)}\nFixed hDeltaB identities; no fitted angles or pattern alignment")
                _save_figure(fig, destination / name)
            finally:
                plt.close(fig)
        similarity = pd.read_csv(source / tag / "pattern_similarity.csv", index_col=0)
        display = grid.sort_values(["world_direction_deg", "heading_deg"]).trial.tolist()
        fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
        try:
            image = ax.imshow(similarity.loc[display, display], vmin=-1, vmax=1, cmap="coolwarm")
            ax.set_xticks(np.arange(8) * 8 + 3.5, [f"{a}°" for a in HEADINGS])
            ax.set_yticks(np.arange(8) * 8 + 3.5, [f"{a}°" for a in HEADINGS])
            ax.set_xlabel("World travel direction; eight headings within each block")
            ax.set_ylabel("World travel direction; eight headings within each block")
            ax.set_title(f"hDeltaB movement-evoked pattern similarity | baseline {baseline:g}\nCentered cosine; {weight_description(config)}")
            fig.colorbar(image, ax=ax, label="Centered cosine")
            _save_figure(fig, destination / "pattern_similarity")
        finally:
            plt.close(fig)
        for heading in [0, 90, 180, 270]:
            name = f"world000_heading{heading:03d}_speed1"
            with np.load(source / tag / "trials" / f"{name}.npz") as trace:
                fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True, constrained_layout=True,
                                         gridspec_kw={"height_ratios": [3, 2, 2]})
                try:
                    for pop, ax in zip(["PFNd", "PFNv", "hDeltaB"], axes):
                        order = pd.read_csv(source / f"{pop}_anatomical_order.csv")
                        if pop == "hDeltaB":
                            ids = config["target_ids"]
                            values = trace["hdeltab_activity"]
                        else:
                            ids = config["source_ids"]
                            values = trace["pfn_activity"]
                        indices = [ids.index(int(body)) for body in order.bodyId]
                        data = values[:, indices].T
                        image = ax.imshow(data, aspect="auto", extent=[trace["time_s"][0], trace["time_s"][-1], len(order) - .5, -.5], vmin=0, vmax=max(float(data.max()), 1e-12))
                        ax.set_yticks(range(len(order)), order.label, fontsize=6)
                        ax.set_ylabel(pop)
                        ax.axvline(config["parameters"]["onset_s"], color="white", linestyle="--", linewidth=.7)
                        ax.axvline(config["parameters"]["offset_s"], color="white", linestyle="--", linewidth=.7)
                        fig.colorbar(image, ax=ax, label="Activity (a.u.; scale per panel)")
                    axes[-1].set_xlabel("Time (s)")
                    fig.suptitle(f"World travel 0°; heading {heading}° | baseline {baseline:g}\n{weight_description(config)}; no recurrence")
                    _save_figure(fig, destination / f"time_{name}")
                finally:
                    plt.close(fig)
    nulls = pd.read_csv(source / "null_comparisons.csv")
    observed = pd.read_csv(source / "comparison_summary.csv")
    fig, axes = plt.subplots(1, len(config["baseline_levels"]), figsize=(6 * len(config["baseline_levels"]), 4), squeeze=False, constrained_layout=True)
    try:
        for ax, baseline in zip(axes.flat, config["baseline_levels"]):
            subset = nulls[nulls.baseline_level == baseline]
            for name, group in subset.groupby("control"):
                ax.hist(group.contrast.dropna(), bins=20, alpha=.5, label=name)
            actual = observed[(observed.baseline_level == baseline) & observed.representation.eq("movement_evoked")].iloc[0].contrast
            if pd.notna(actual):
                ax.axvline(actual, color="black", linewidth=2, label="Anatomical network")
            ax.set_title(f"PFN baseline {baseline:g}")
            ax.set_xlabel("Same-world minus different-world mean centered cosine")
            ax.set_ylabel("Shuffle realizations")
            ax.legend(fontsize=7)
        _save_figure(fig, output / "control_comparison")
    finally:
        plt.close(fig)


def _save_figure(fig, prefix):
    for ext in ["png", "pdf"]:
        fig.savefig(str(prefix) + f".{ext}", dpi=150)


def write_experiment_notes(output, config, comparisons, null_rows):
    p = config["parameters"]
    lines = [
        "# First movement-driven PFN → hDeltaB experiment", "",
        "This is a conditional test of the extracted MaleCNS feedforward network given imposed PFN encoding. "
        "It does not test upstream generation of that encoding or decode an absolute hDeltaB travel angle.", "",
        f"Reference: [Lu et al., 2022]({PAPER}); [Methods]({METHODS}). Walking physiology is transferred "
        "to this connectome as a modeling assumption. No parameters are fitted to hDeltaB responses.", "",
        "## Fixed specification", "",
        "- Network: 40 PFNd + 20 PFNv → 19 hDeltaB; contact-derived weights × base gain × source-type multiplier; all input compartments pooled.",
        f"- Coupling: {weight_description(config)}. Multipliers act on weights, not prescribed PFN activity.",
        ("- Incoming normalization: for each receiving hDeltaB neuron, divide contacts within each PFN type/PB-side group by that group's total incoming contacts, then apply type multiplier and base gain. Zero-total groups remain zero. "
         "This matches the Lu released weight preprocessing up to our retained common base gain; it is not an exact reproduction of their other simulation settings."
         if p.get('normalize_incoming_by_group', False) else
         "- No incoming-weight normalization. The Lu released code also normalizes incoming contacts within each PFN type/side group; multipliers alone do not reproduce that preprocessing."),
        "- Heading zero is arbitrary; clockwise angles and rightward lateral velocity are positive; head and body are aligned.",
        "- Heading preferences: PB L5/R5 = 0°, L7/R3 = +90°; 45° steps, wrapping at 180°. "
        "L9/R9 are assigned −180°. See PFN_encoding_map.csv for all individual assignments. FB C indices do not set angles.",
        "- Preferred body movement: PFNd L/R = −31°/+31°; PFNv L/R = +137°/−137°.",
        "- Heading response H = (1 + cos(heading − preferred_heading))/2. Prescribed PFN activity "
        "is H × [b + alpha × max(0, velocity projection / reference speed)]. Both PFN types have unit per-neuron gain.",
        f"- alpha = {p['velocity_modulation']:g}; reference speed = {p['reference_speed_mm_s']:g} mm/s; "
        f"PFN baseline offsets b = {config['baseline_levels']}. These are arbitrary rate units, not measured Hz.",
        "- PFN activity is prescribed directly; it is not passed through PFN rate dynamics. "
        "hDeltaB follows tau × dr/dt = −r + ReLU(PFN @ W), initialized to zero.",
        f"- hDeltaB tau = {p['tau_s']:g} s; dt = {p['dt_s']:g} s; common base gain = {p['gain']:g}. "
        "These retain the earlier illustrative model settings, not fitted physiological time constants.",
        "- No recurrence, feedback, noise, output angle fitting, per-condition alignment, or displacement memory.", "",
        "## Conditions and timing", "",
        "Each baseline uses 74 independent trials: an 8-world-direction × 8-heading grid at 1 mm/s; "
        "8 stationary heading controls; and 0.5 and 2 mm/s trials at heading/world direction 0°. "
        "The speed comparisons supplement the primary 64-condition grid.", "",
        f"Trials last {p['duration_s']:g} s. Heading is constant; movement is active on "
        f"[{p['onset_s']:g}, {p['offset_s']:g}) s. Readouts are temporal means over "
        f"[{p['analysis_start_s']:g}, {p['analysis_stop_s']:g}) s. "
        "Stationary PFN baseline is active from the start and remains after translation stops. "
        "For b>0, zero movement therefore does not imply neural silence. World travel angle is undefined at rest.", "",
        "## Comparisons", "",
        "Raw profiles retain all activity. Movement-evoked profiles subtract the matched stationary trial "
        "at the same heading, baseline and analysis window. All 19 target identities remain fixed. "
        "Pattern similarity is cosine similarity after subtracting the profile's across-neuron mean; "
        "this rejects a spatially uniform increase as directional structure. Zero or constant centered profiles "
        "with L2 norm ≤1e−12 are undefined. Uncentered cosines and amplitude measurements are also saved.", "",
        "The primary contrast is mean similarity for same-world/different-heading pairs minus mean similarity "
        "for different-world/same-heading pairs. Each comparison has 224 unordered pairs at matched speed. "
        "These conditions and pairs are not independent biological replicates.", "",
        f"Two controls use {p['null_repeats']} realizations each (seed {p['seed']}): "
        "source-row permutations within each PFN type × PB side, and independent world-direction label "
        "permutations within each heading. Each connectivity realization is held fixed across conditions. "
        "The former preserves group-wise target totals and whole row profiles as a multiset; "
        "the latter changes only comparison labels, not simulated movement or activity. "
        "These are distinct controls, not temporal shuffling of a continuous trajectory. Permutations are saved.", "",
        "## Results", "",
        "| PFN baseline | Profile | Same-world mean | Different-world mean | Contrast |",
        "| --- | --- | --- | --- | --- |",
    ]
    def number(v):
        return "undefined" if v is None else f"{v:.6f}"
    for row in comparisons:
        lines.append(f"| {row['baseline_level']:g} | {row['representation']} | "
                     f"{number(row['same_world_mean_centered_cosine'])} | "
                     f"{number(row['different_world_mean_centered_cosine'])} | {number(row['contrast'])} |")
    lines.extend(["", "| PFN baseline | Control | Contrast mean | Minimum | Maximum |",
                  "| --- | --- | --- | --- | --- |"])
    for (baseline, name), group in pd.DataFrame(null_rows).groupby(["baseline_level", "control"]):
        values = group.contrast.dropna()
        fields = [f"{value:.6f}" for value in [values.mean(), values.min(), values.max()]] if len(values) else ["undefined"] * 3
        lines.append(f"| {baseline:g} | {name} | {' | '.join(fields)} |")
    lines.extend(["", "Numerical validation is separate from the biological hypothesis: Euler traces are checked "
                  "against a continuous-time solution, half-dt integration and deterministic replay. "
                  "Source hashes are checked before and after the run. With this linear output stage, "
                  "matched-rest subtraction removes the baseline contribution; identical evoked results "
                  "across baseline offsets are expected, not independent replication.", "",
                  "No output preferred directions were estimated from the tested weights. "
                  "Imperfect invariance is retained; anatomical sampling, input assumptions and simple dynamics "
                  "remain possible sources of variation. Shuffle distributions are conditional controls, not a population inference across flies.", ""])
    (output / "experiment_notes.md").write_text("\n".join(lines))


def run_movement(source, output, code, parameters=None, baseline_levels=(1., 0.)):
    p = parameters or MovementParameters()
    p.validate()
    baseline_levels = [float(v) for v in baseline_levels]
    if (not baseline_levels or len(set(baseline_levels)) != len(baseline_levels)
            or any(not np.isfinite(v) or v < 0 for v in baseline_levels)):
        raise ValidationError("Use unique finite nonnegative PFN baseline levels.")
    tags = ["baseline_" + f"{v:g}".replace(".", "p") for v in baseline_levels]
    if len(set(tags)) != len(tags):
        raise ValidationError("Baseline levels collide at the saved directory precision.")
    source, output = Path(source), Path(output)
    counts, populations, orders, metadata, hashes = load_pfn_network(source)
    mapping = heading_mapping(orders, counts.index)
    conditions = movement_conditions()
    weights = movement_weights(counts, mapping, p)
    config = {
        "parameters": asdict(p), "baseline_levels": baseline_levels,
        "source_ids": [int(v) for v in counts.index], "target_ids": [int(v) for v in counts.columns],
        "encoding": "r_i = 0.5*(1+cos(theta-theta_i)) * (b + alpha*ReLU((vf*cos(phi_i)+vr*sin(phi_i))/v0)); alpha=velocity_modulation; angles evaluated in radians",
        "activity_units": "Arbitrary rates; no conversion to Hz. Unit per-neuron scale for both PFN types.",
        "coordinates": "Heading zero is an arbitrary world reference; clockwise positive; rightward body velocity positive; head/body aligned. World direction = heading + atan2(vr,vf). Undefined at zero speed.",
        "heading_mapping": "Lu Methods anchors L5/R5=0, L7/R3=+90; 45-degree steps; wrap L9/R9 to -180. Transferred to MaleCNS using literal receiving PB annotations, not FB C labels. A modeling assumption, not MaleCNS physiological calibration.",
        "sources": [PAPER, METHODS, "https://github.com/druckmann-lab/Translational-velocity-and-heading-model"],
        "literature_transfer": "Walking tuning and encoding transferred from published experimental preparations to MaleCNS. This is not an exact reproduction of the published model or its weight preprocessing.",
        "equations": ["PFN activity prescribed algebraically; no PFN rate filtering", "tau*d(hDeltaB)/dt = -hDeltaB + ReLU(PFN @ W)"],
        "coupling": "W[pre,post] = gain*source_type_multiplier*contacts (optionally divided by sum of contacts from the same PFN type/PB-side group to that target). Normalization precedes multiplication; zero-total groups stay zero. All compartments pooled; excitatory assumption. Defaults: no normalization, multipliers=1. Normalized PFNd=6/PFNv=1 matches Lu released weight preprocessing up to the retained common gain, not other model settings.",
        "disabled": ["recurrence", "feedback", "noise", "hDeltaB angle decoder", "displacement integration"],
        "timing": "Heading fixed throughout each trial; translation active on [onset, offset); hDeltaB initialized to zero. Analysis window [analysis_start, analysis_stop). Nonzero PFN baseline is active from t=0 and remains after translation stops.",
        "readout": "Raw temporal mean, and temporal mean minus matched stationary trial at same heading/baseline/time. No per-trial pattern shift, output angle fitting, or activity-dependent neuron reordering.",
        "primary_metric": "Centered cosine across 19 fixed hDeltaB neurons, unit-normalized after subtracting each pattern's across-neuron mean. Constant/zero patterns (centered L2 <=1e-12) undefined. Contrast = mean same-world/different-heading similarity minus mean different-world/same-heading similarity; speed=1 mm/s only. Also save uncentered cosines and raw amplitude.",
        "nulls": "Connectivity: independently permute whole source rows within each type x PB side, preserving group-wise target totals and row profiles as a multiset. Movement labels: independently permute world-direction labels within each heading; responses unchanged. All permutations saved. Null metrics evaluated from the same steady-window linear response factors; no dynamic null traces.",
        "interpretation": "Conditional test of anatomical transformation given imposed PFN encoding. Does not independently validate upstream encoding, preferred hDeltaB angles, biological time constants, or homing. Shuffle realizations are not independent flies.",
    }
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_directory": str(source.resolve()), "source_sha256": hashes,
                "server": SERVER, "dataset": DATASET, "dataset_metadata": metadata["dataset_metadata"], "code": code,
                "code_sha256": {f.name: sha256(f) for f in sorted(Path(__file__).parent.glob("*.py"))},
                "python": platform.python_version(), "packages": {v: version(v) for v in ["numpy", "pandas", "matplotlib"]}}
    save_json(output / "movement_model.json", manifest)
    try:
        save_json(output / "movement_config.json", config)
        conditions.to_csv(output / "conditions.csv", index=False)
        mapping.to_csv(output / "PFN_encoding_map.csv", index=False)
        counts.to_csv(output / "raw_contact_matrix.csv")
        pd.DataFrame(weights, index=counts.index, columns=counts.columns).to_csv(output / "effective_weight_matrix.csv")
        for pop in populations:
            populations[pop].to_csv(output / f"{pop}_neurons.csv", index=False)
            orders[pop].to_csv(output / f"{pop}_anatomical_order.csv", index=False)
        grid = conditions[conditions.primary_grid].reset_index(drop=True)
        grid_indices = np.flatnonzero(conditions.primary_grid)
        encoding_grid = np.array([encode_pfn(mapping, r.heading_deg, r.forward_mm_s, r.right_mm_s, 0., p) for r in grid.itertuples()])
        rng_connectivity, rng_labels = [np.random.default_rng(s) for s in np.random.SeedSequence(p.seed).spawn(2)]
        permutations = np.array([shuffled_sources(mapping, rng_connectivity) for _ in range(p.null_repeats)])
        label_permutations = [shuffled_labels(grid, rng_labels) for _ in range(p.null_repeats)]
        np.savez_compressed(output / "null_permutations.npz", source_row_indices=permutations,
                            grid_world_direction_deg=np.array([v.world_direction_deg.to_numpy() for v in label_permutations]))
        comparisons, null_rows, validations, amplitudes = [], [], [], []
        for baseline, tag in zip(baseline_levels, tags):
            print(f"Simulating movement encoding baseline {baseline:g}: {len(conditions)} trials", flush=True)
            destination = output / tag
            (destination / "trials").mkdir(parents=True)
            steady = []
            for trial in conditions.itertuples():
                base = encode_pfn(mapping, trial.heading_deg, 0., 0., baseline, p)
                active = encode_pfn(mapping, trial.heading_deg, trial.forward_mm_s, trial.right_mm_s, baseline, p)
                increment = active - base
                target_base, target_increment = base @ weights, increment @ weights
                times, activity, exact = solve_output(target_base, target_increment, p)
                _, fine, _ = solve_output(target_base, target_increment, replace(p, dt_s=p.dt_s / 2))
                _, replay, _ = solve_output(target_base, target_increment, p)
                error = relative_error(activity, exact)
                fine_error = relative_error(fine[::2], exact)
                half_error = relative_error(activity, fine[::2])
                passed = error <= .02 and half_error <= .02 and fine_error <= error + 1e-10 and np.array_equal(activity, replay)
                validations.append({"baseline_level": baseline, "trial": trial.trial, "relative_analytical_error": error,
                                    "half_dt_analytical_error": fine_error, "relative_half_dt_error": half_error,
                                    "deterministic_replay": np.array_equal(activity, replay), "passed": bool(passed)})
                mask = (times >= p.onset_s) & (times < p.offset_s)
                prescribed = np.broadcast_to(base, (len(times), len(base))).copy()
                prescribed[mask] += increment
                np.savez_compressed(destination / "trials" / f"{trial.trial}.npz", time_s=times,
                                    pfn_activity=prescribed, hdeltab_activity=activity,
                                    heading_deg=np.full(len(times), trial.heading_deg),
                                    forward_mm_s=np.where(mask, trial.forward_mm_s, 0.), right_mm_s=np.where(mask, trial.right_mm_s, 0.))
                window = slice(grid_index(p.analysis_start_s, p.dt_s), grid_index(p.analysis_stop_s, p.dt_s))
                steady.append(activity[window].mean(axis=0))
            steady = np.array(steady)
            rest = {r.heading_deg: steady[i] for i, r in enumerate(conditions.itertuples()) if r.speed_mm_s == 0}
            evoked = steady - np.array([rest[h] for h in conditions.heading_deg])
            for label, data in [("steady_activity", steady), ("movement_evoked_activity", evoked)]:
                pd.DataFrame(data, index=pd.Index(conditions.trial, name="trial"), columns=counts.columns).to_csv(destination / f"{label}.csv")
            for representation, data in [("raw", steady), ("movement_evoked", evoked)]:
                summary, pairs, similarity = pattern_comparison(data[grid_indices], grid)
                comparisons.append({"baseline_level": baseline, "representation": representation, **summary})
                pairs.to_csv(destination / f"{representation}_pair_comparisons.csv", index=False)
                if representation == "movement_evoked":
                    pd.DataFrame(similarity, index=grid.trial, columns=grid.trial).to_csv(destination / "pattern_similarity.csv")
            for i, trial in enumerate(conditions.itertuples()):
                amplitudes.append({"baseline_level": baseline, **trial._asdict(), "raw_mean_activity": float(steady[i].mean()),
                                   "raw_peak_activity": float(steady[i].max()), "evoked_l2": float(np.linalg.norm(evoked[i])),
                                   "evoked_centered_l2": float(np.linalg.norm(evoked[i] - evoked[i].mean()))})
            # The matched-rest subtraction removes baseline exactly in this linear system.
            _, unit_response, _ = solve_output(np.zeros(1), np.ones(1), p)
            factor = float(unit_response[window].mean())
            np.testing.assert_allclose(evoked[grid_indices], factor * (encoding_grid @ weights), rtol=1e-11, atol=1e-12)
            for replicate, permutation in enumerate(permutations):
                shuffled = factor * (encoding_grid @ weights[permutation])
                score, _, _ = pattern_comparison(shuffled, grid, include_pairs=False)
                null_rows.append({"baseline_level": baseline, "control": "shuffled_connectivity", "replicate": replicate, **score})
                score, _, _ = pattern_comparison(evoked[grid_indices], label_permutations[replicate], include_pairs=False)
                null_rows.append({"baseline_level": baseline, "control": "shuffled_movement_labels", "replicate": replicate, **score})
        pd.DataFrame(comparisons).to_csv(output / "comparison_summary.csv", index=False)
        pd.DataFrame(null_rows).to_csv(output / "null_comparisons.csv", index=False)
        pd.DataFrame(amplitudes).drop(columns="Index").to_csv(output / "activity_summary.csv", index=False)
        pd.DataFrame(validations).to_csv(output / "numerical_validation.csv", index=False)
        write_experiment_notes(output, config, comparisons, null_rows)
        if not all(row["passed"] for row in validations):
            raise ValidationError("Movement numerical validation failed; inspect numerical_validation.csv.")
        if any(sha256(source / name) != digest for name, digest in hashes.items()):
            raise ValidationError("Source extraction changed during movement experiments.")
        manifest["validation"] = {"trials": len(conditions) * len(baseline_levels), "primary_grid_conditions_per_baseline": len(grid),
                                  "all_numerical_checks_passed": True, "source_unchanged": True,
                                  "steady_null_shortcut_matches_simulation": True,
                                  "max_relative_analytical_error": max(r["relative_analytical_error"] for r in validations)}
        manifest["data_sha256"] = {str(f.relative_to(output)): sha256(f) for f in sorted(output.rglob("*")) if f.is_file() and f.name != "movement_model.json"}
        manifest["data_validated"] = True
        save_json(output / "movement_model.json", manifest)
        render_movement(output, output)
        verify_model_data(output, manifest)
        manifest["artifact_sha256"] = {str(f.relative_to(output)): sha256(f) for f in sorted(output.rglob("*")) if f.is_file() and f.name != "movement_model.json"}
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="validation_failed" if isinstance(exc, (ValidationError, AssertionError)) else "failed", error_type=type(exc).__name__)
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(output / "movement_model.json", manifest)
    return manifest


def render_normalized_evoked(source, output):
    """Compare pattern shape by dividing each matched-rest-subtracted row by its peak."""
    source, output = Path(source), Path(output)
    config = json.loads((source / "movement_config.json").read_text())
    conditions = pd.read_csv(source / "conditions.csv")
    order = pd.read_csv(source / "hDeltaB_anatomical_order.csv")
    columns = order.bodyId.astype(str).tolist()
    selections = {
        "world_invariance": conditions[(conditions.world_direction_deg == 0) & (conditions.speed_mm_s == 1)],
        "world_invariance_minus090deg": conditions[(conditions.world_direction_deg == 270) & (conditions.speed_mm_s == 1)],
        "world_invariance_plus090deg": conditions[(conditions.world_direction_deg == 90) & (conditions.speed_mm_s == 1)],
        "heading_sweep": conditions[(conditions.body_direction_deg == 0) & (conditions.speed_mm_s == 1)],
        "body_direction_sweep": conditions[(conditions.heading_deg == 0) & conditions.primary_grid],
    }
    for baseline in config["baseline_levels"]:
        tag = "baseline_" + f"{baseline:g}".replace(".", "p")
        destination = output / tag
        destination.mkdir(parents=True, exist_ok=True)
        evoked = pd.read_csv(source / tag / "movement_evoked_activity.csv", index_col=0)
        for name, selected in selections.items():
            values = evoked.loc[selected.trial, columns].to_numpy(dtype=float)
            peaks = values.max(axis=1, keepdims=True)
            normalized = np.divide(values, peaks, out=np.zeros_like(values), where=peaks > 1e-12)
            stem = destination / f"{name}_normalized_evoked"
            pd.DataFrame(normalized, index=selected.trial, columns=columns).to_csv(stem.with_suffix(".csv"))
            pd.DataFrame({"trial": selected.trial, "peak_evoked_activity": peaks[:, 0]}).to_csv(
                destination / f"{name}_normalization_peaks.csv", index=False)
            fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
            try:
                image = ax.imshow(normalized, aspect="auto", vmin=0, vmax=1)
                ax.set_yticks(range(len(selected)),
                              [f"heading {r.heading_deg:g}°; body {r.body_direction_deg:g}°; speed {r.speed_mm_s:g}"
                               for r in selected.itertuples()], fontsize=8)
                ax.set_xticks(range(len(order)), order.label, rotation=90, fontsize=6)
                title = name
                if name.startswith("world_invariance"):
                    world = float(wrap_degrees(selected.world_direction_deg.iloc[0]))
                    title = f"World invariance: world-relative travel {world:+g}°"
                ax.set_title(f"{title} | PFN baseline {baseline:g}; {weight_description(config)}\nMatched rest subtracted; each row divided by its peak")
                fig.colorbar(image, ax=ax, label="Activity / row maximum")
                _save_figure(fig, stem)
            finally:
                plt.close(fig)


def replot_movement(source, output, code, *, normalized_evoked_only=False):
    source, output = Path(source), Path(output)
    manifest = json.loads((source / "movement_model.json").read_text())
    verify_model_data(source, manifest)
    output.mkdir(parents=True, exist_ok=False)
    if normalized_evoked_only:
        render_normalized_evoked(source, output)
    else:
        render_movement(source, output)
    verify_model_data(source, manifest)
    save_json(output / "replot.json", {"status": "complete", "code": code, "source": str(source.resolve()),
                                      "data_sha256": manifest["data_sha256"],
                                      "plot_settings": {"normalized_evoked_only": normalized_evoked_only,
                                                        "normalization": "Per-row maximum after matched-rest subtraction; peak <= 1e-12 gives a zero row" if normalized_evoked_only else None}})
