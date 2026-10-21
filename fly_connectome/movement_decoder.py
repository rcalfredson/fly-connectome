"""Offline transfer of the Lu et al. published model's hDeltaB decoder."""

from collections import Counter
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .connectivity import ValidationError
from .movement import wrap_degrees, weight_description
from .rate_model import save_json, sha256, verify_model_data


AUTHORS_NOTEBOOK = "https://github.com/druckmann-lab/Translational-velocity-and-heading-model/blob/main/Translational%20Velocity%20and%20Heading%20Simulation%20Runs.ipynb"
AUTHORS_LABELS = "https://github.com/druckmann-lab/Translational-velocity-and-heading-model/blob/main/connectivity_matrices/NEURON_LABELS.csv"
REFERENCE_SHA256 = {"notebook": "7884c26e08ae2aa911b702cbb0b4a90523a6de43928eca84a398c32524674fc9",
                    "neuron_labels": "f1cb4e92a74be0a48f913cf6316cdb174bffa5c8bddb60329910a11e24aa9071"}
# make_delb_readout: reorder 19 cells, then average cells sharing these names.
# These are 14 named positions, NOT the 12 MaleCNS C suffixes or 8 imaging ROIs.
POSITIONS = [(8, 7, 1), (9, 7, 1), (10, 8, 1), (11, 9, 2),
             (12, 10, 2), (13, 11, 1), (14, 12, 1), (1, 1, 1),
             (2, 2, 1), (3, 3, 3), (4, 4, 2), (5, 5, 1),
             (6, 6, 1), (7, 6, 1)]
# Notebook calc_PVA uses angles linspace(-pi-b, pi-b, 14, endpoint=False).
# b is a fit to their model. Preserve it explicitly; never refit to our trials.
PUBLISHED_OFFSET_DEG = -10.62784572061546


def decoder_mapping(order):
    expected = {f"hDeltaB_{i:02d}_C{c}": n for i, c, n in POSITIONS}
    if order.bodyId.duplicated().any() or Counter(order.instance) != Counter(expected):
        raise ValidationError("Decoder requires the published 19-cell anatomical-name inventory, with matching multiplicities.")
    records = []
    for position, (i, column, count) in enumerate(POSITIONS):
        name = f"hDeltaB_{i:02d}_C{column}"
        for body in order.loc[order.instance.eq(name), "bodyId"]:
            records.append({"bodyId": int(body), "instance": name, "position": position,
                            "cells_in_position": count, "phase_angle_deg": -180 + position * 360 / 14,
                            "published_angle_deg": -180 + position * 360 / 14 - PUBLISHED_OFFSET_DEG})
    return pd.DataFrame(records)


def decode_profiles(profiles, mapping):
    columns = [str(body) for body in mapping.bodyId]
    if len(profiles.columns) != len(columns) or set(profiles.columns) != set(columns):
        raise ValidationError("Activity identities do not match the decoder mapping.")
    activity = profiles.loc[:, columns].to_numpy(dtype=float)
    if not np.isfinite(activity).all() or (activity < -1e-10).any():
        raise ValidationError("Population-vector weights must be finite and nonnegative.")
    activity = np.maximum(activity, 0.)  # tolerate roundoff in matched-rest subtraction
    bins = np.column_stack([activity[:, mapping.position.eq(k)].mean(axis=1) for k in range(14)])
    angles = np.deg2rad(-180 + np.arange(14) * 360 / 14)
    vector = bins @ np.exp(1j * angles)
    total = bins.sum(axis=1)
    valid = (total > 1e-12) & (np.abs(vector) > 1e-12 * np.maximum(total, 1.))
    phase = np.where(valid, np.rad2deg(np.angle(vector)), np.nan)
    result = pd.DataFrame({"phase_deg": phase,
                           "published_decoded_deg": wrap_degrees(phase - PUBLISHED_OFFSET_DEG),
                           "resultant_length": np.divide(np.abs(vector), total, out=np.zeros_like(total), where=total > 0),
                           "valid": valid}, index=profiles.index)
    return result, pd.DataFrame(bins, index=profiles.index, columns=[f"position_{k:02d}" for k in range(14)])


def run_decoder(source, output, code):
    source, output = Path(source), Path(output)
    original = json.loads((source / "movement_model.json").read_text())
    verify_model_data(source, original)
    config = json.loads((source / "movement_config.json").read_text())
    conditions = pd.read_csv(source / "conditions.csv").set_index("trial")
    mapping = decoder_mapping(pd.read_csv(source / "hDeltaB_anatomical_order.csv"))
    output.mkdir(parents=True, exist_ok=False)
    mapping.to_csv(output / "hDeltaB_decoder_map.csv", index=False)
    manifest = {"status": "running", "source": str(source.resolve()), "code": code,
                "code_sha256": sha256(Path(__file__)), "source_data_sha256": original["data_sha256"],
                "sources": [AUTHORS_NOTEBOOK, AUTHORS_LABELS],
                "reference_sha256_verified_2026_10_20": REFERENCE_SHA256,
                "published_offset_deg": PUBLISHED_OFFSET_DEG,
                "transfer": "Exact anatomical-name and multiplicity match; average within 14 named positions. Cross-dataset functional correspondence is a modeling assumption.",
                "calibration": "Published model offset preserved, not fitted to MaleCNS. Absolute angular errors include an uncalibrated cross-dataset phase offset.",
                "formula": "phase=arg(sum_k mean_activity_k*exp(i*(-pi+2*pi*k/14))); published_angle=wrap(phase-b), b=-10.62784572061546 deg",
                "scope": "Published model decoder (Fig. 4d analogue), not the eight-ROI calcium-imaging processing used for Fig. 4h. Existing simulated trials reused without changing inputs or dynamics."}
    summaries = []
    try:
        for baseline in config["baseline_levels"]:
            tag = "baseline_" + f"{baseline:g}".replace(".", "p")
            destination = output / tag
            destination.mkdir()
            for readout, filename in [("raw", "steady_activity.csv"), ("movement_evoked", "movement_evoked_activity.csv")]:
                profiles = pd.read_csv(source / tag / filename, index_col=0)
                decoded, bins = decode_profiles(profiles, mapping)
                frame = conditions.join(decoded, how="left", validate="one_to_one")
                frame["angular_error_deg"] = wrap_degrees(frame.published_decoded_deg - frame.world_direction_deg)
                frame.loc[frame.speed_mm_s.eq(0), "angular_error_deg"] = np.nan
                frame.to_csv(destination / f"{readout}_decoded_trials.csv")
                bins.to_csv(destination / f"{readout}_position_activity.csv")
                grid = frame[frame.primary_grid & frame.valid]
                error = grid.angular_error_deg.to_numpy()
                summaries.append({"baseline": baseline, "readout": readout, "valid_primary_trials": len(grid),
                                  "mean_absolute_error_published_convention_deg": float(np.abs(error).mean()) if len(error) else np.nan,
                                  "circular_bias_deg_diagnostic_only": float(np.rad2deg(np.angle(np.exp(1j * np.deg2rad(error)).mean()))) if len(error) else np.nan})
                fig, ax = plt.subplots(figsize=(7.5, 6), layout="constrained")
                fig.get_layout_engine().set(rect=(0, .05, 1, .95))
                for heading, group in grid.groupby("heading_deg"):
                    x = wrap_degrees(group.world_direction_deg.to_numpy())
                    y = group.published_decoded_deg.to_numpy()
                    indices = np.argsort(x)
                    # Scatter avoids joining across the angular wrap discontinuity.
                    ax.scatter(x[indices], y[indices], s=24, alpha=.75, label=f"{heading:g}°")
                ax.plot([-180, 180], [-180, 180], "--", color="0.6", linewidth=1)
                ax.set(xlim=(-190, 190), ylim=(-190, 190), xlabel="World-relative travel direction (°)",
                       ylabel="Decoded hΔB angle, published convention (°)",
                       title=f"Lu model decoder transferred by anatomical name\n{readout.replace('_', ' ')}; baseline {baseline:g}; {weight_description(config)}")
                if len(grid):
                    ax.legend(title="Heading", bbox_to_anchor=(1.02, 1), loc="upper left")
                fig.text(.02, .01, "Published angular offset retained; no MaleCNS angle fitting.", fontsize=8)
                for suffix in ["png", "pdf"]:
                    fig.savefig(destination / f"{readout}_direction_decode.{suffix}", dpi=180, bbox_inches="tight")
                plt.close(fig)
        pd.DataFrame(summaries).to_csv(output / "decoder_summary.csv", index=False)
        (output / "decoder_notes.md").write_text(
            "# Published hDeltaB model decoder transfer\n\n"
            f"Sources: [authors' notebook]({AUTHORS_NOTEBOOK}), [neuron labels]({AUTHORS_LABELS}).\n\n"
            "The notebook's make_delb_readout reorders 19 neurons into 14 named positions, averaging duplicate names. "
            "All names and multiplicities match our MaleCNS inventory. This correspondence is an anatomical transfer assumption, not a physiological calibration. "
            "C1–C12 suffixes alone are not the decoder bins.\n\n"
            "For bin k=0,...,13, alpha_k=-180+360*k/14 degrees. Compute phase=atan2(sum a_k sin alpha_k, sum a_k cos alpha_k). "
            "The notebook additionally subtracts b=-10.62784572061546 degrees, fitted to its original model. We preserve that convention without refitting. "
            "Both the unshifted phase and published-convention angle are saved. Zero or cancelled vectors have undefined angles; resultant length measures vector concentration, not accuracy.\n\n"
            "This is a model comparison analogous to Fig. 4d. Fig. 4h uses calcium imaging with eight spatial ROIs and per-fly offset correction. "
            "We do not reproduce that imaging pipeline or claim an experimentally known absolute MaleCNS angle map.\n\n"
            "Both raw and matched-rest-subtracted profiles are decoded; the latter is our additional control. "
            "All 74 conditions per baseline are saved, but plots and angular-error summaries use only the 64 primary moving conditions. "
            "Stationary raw profiles can have a phase but no true travel direction. Reported circular bias is diagnostic only and is never subtracted. "
            "Mean absolute error uses the frozen published convention and includes any cross-dataset phase mismatch. "
            "No new simulations, fits, or changes to the original results are made.\n")
        verify_model_data(source, original)
        manifest["status"] = "complete"
        manifest["artifact_sha256"] = {str(f.relative_to(output)): sha256(f) for f in sorted(output.rglob("*")) if f.is_file()}
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        save_json(output / "decoder_manifest.json", manifest)
    return manifest
