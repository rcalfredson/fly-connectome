"""Offline conservation checks and readable summaries for circuit selection."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .circuit_extension import PAIR, summarize_outputs
from .connectivity import ValidationError, build_matrix
from .inputs import save_json, sha256, verify_input_data
from .rate_model import verify_model_data


def markdown(table):
    def cell(v):
        if pd.isna(v):
            return "undefined"
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v).replace("|", "/")
    return "\n".join(["| " + " | ".join(table.columns) + " |",
        "| " + " | ".join(["---"] * len(table.columns)) + " |"]
        + ["| " + " | ".join(map(cell, row)) + " |" for row in table.itertuples(index=False, name=None)])


def validate_saved_census(source):
    source = Path(source)
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValidationError("Census must be complete.")
    verify_input_data(source, manifest)
    raw = pd.read_csv(source / "raw_outputs.csv")
    raw["labels"] = raw.labels.map(json.loads)
    neurons = pd.read_csv(source / "candidate_neurons.csv")
    hdeltab = pd.read_csv(source / "hDeltaB_neurons.csv")
    tables = summarize_outputs(raw, hdeltab)
    for table, name in zip(tables, ["downstream_partners.csv", "output_denominators.csv",
                                   "population_by_source.csv", "population_outputs.csv"]):
        pd.testing.assert_frame_equal(table.reset_index(drop=True), pd.read_csv(source / name), check_dtype=False)
    edges = pd.read_csv(source / "outgoing_connections.csv")
    pd.testing.assert_frame_equal(raw[edges.columns], edges, check_dtype=False)
    summary = pd.read_csv(source / "motif_summary.csv")
    if set(zip(summary.source, summary.target)) != set(map(tuple, manifest["motifs"])):
        raise ValidationError("Motif summary differs from declared whitelist.")
    for row in summary.itertuples():
        src = neurons.loc[neurons.type == row.source, "bodyId"]
        dst = neurons.loc[neurons.type == row.target, "bodyId"]
        if row.query_status != "queried":
            if len(src) and len(dst):
                raise ValidationError("Existing population was not queried.")
            continue
        prefix = row.source + "_to_" + row.target
        conn = pd.read_csv(source / (prefix + "_connections.csv"))
        matrix = build_matrix(conn, src, dst)
        saved = pd.read_csv(source / (prefix + "_matrix.csv"), index_col=0)
        saved.columns = saved.columns.astype(int)
        pd.testing.assert_frame_equal(matrix, saved, check_names=False)
        denominator = int(neurons.loc[neurons.bodyId.isin(dst), "post"].sum())
        expected_counts = {
            "weight": matrix.to_numpy().sum(),
            "positive_pairs": matrix.gt(0).to_numpy().sum(),
            "possible_pairs": len(src) * len(dst),
            "all_target_post_denominator": denominator,
        }
        for field, expected_count in expected_counts.items():
            if getattr(row, field) != expected_count:
                raise ValidationError(
                    f"Motif {row.source}→{row.target}: {field} differs from reconstructed value."
                )
        if denominator:
            np.testing.assert_allclose(row.target_input_fraction, row.weight / denominator, rtol=1e-12)
        if row.source == "hDeltaB":
            expected = build_matrix(raw[raw.bodyId_post.isin(dst)], src, dst)
            pd.testing.assert_frame_equal(matrix, expected)
    return manifest, neurons, tables, summary


def analyze(census, movement, trajectory, output):
    census, movement, trajectory, output = map(Path, [census, movement, trajectory, output])
    manifest, neurons, (partners, den, per, summary), motifs = validate_saved_census(census)
    mm = json.loads((movement / "movement_model.json").read_text())
    tm = json.loads((trajectory / "trajectory_model.json").read_text())
    for source, record in [(movement, mm), (trajectory, tm)]:
        if record.get("status") != "complete":
            raise ValidationError("Movement and trajectory references must be complete.")
        verify_model_data(source, record)
    mc = json.loads((movement / "movement_config.json").read_text())
    tc = json.loads((trajectory / "trajectory_config.json").read_text())
    expected = set(neurons.loc[neurons.type == "hDeltaB", "bodyId"])
    if set(mc["target_ids"]) != expected or set(tc["target_ids"]) != expected:
        raise ValidationError("Model hDeltaB identities differ from census.")
    output.mkdir(parents=True, exist_ok=False)
    candidate_rows = []
    for row in motifs[motifs.source == "hDeltaB"].itertuples():
        if row.query_status == "population_absent":
            continue
        conn = pd.read_csv(census / f"hDeltaB_to_{row.target}_connections.csv")
        target = neurons[neurons.type == row.target].copy()
        target["hdeltab_weight"] = target.bodyId.map(conn.groupby("bodyId_post").weight.sum()).fillna(0).astype(int)
        target["all_recorded_post_denominator"] = target.post
        target["hdeltab_input_fraction"] = target.hdeltab_weight / target.post.replace(0, np.nan)
        candidate_rows.append(target)
    targets = pd.concat(candidate_rows, ignore_index=True)
    targets.to_csv(output / "hdeltab_input_by_candidate_neuron.csv", index=False)
    categories = partners.groupby("category", as_index=False).agg(partner_bodies=("bodyId", "size"))
    weights = pd.read_csv(census / "outgoing_connections.csv").merge(partners[["bodyId", "category"]],
        left_on="bodyId_post", right_on="bodyId", validate="many_to_one").groupby("category").weight.sum()
    categories["weight"] = categories.category.map(weights)
    categories["all_output_denominator"] = int(den.all_outgoing_segment_weight.sum())
    categories["output_share"] = categories.weight / categories.all_output_denominator
    categories.to_csv(output / "annotation_categories.csv", index=False)
    reconstruction = []
    for field in ["status", "statusLabel", "cropped", "consensusNt", "is_neuron", "status_flagged"]:
        merged = pd.read_csv(census / "outgoing_connections.csv").merge(partners[["bodyId", field]],
            left_on="bodyId_post", right_on="bodyId", validate="many_to_one")
        for value, group in merged.groupby(field, dropna=False):
            reconstruction.append(dict(field=field, value="unknown" if pd.isna(value) else value,
                partner_bodies=group.bodyId_post.nunique(), weight=int(group.weight.sum())))
    pd.DataFrame(reconstruction).to_csv(output / "partner_reconstruction.csv", index=False)
    differences = []
    for representation in ["raw", "movement_evoked"]:
        pairs = pd.read_csv(movement / "baseline_1" / f"{representation}_pair_comparisons.csv")
        same = pairs[pairs.relation == "same_world_different_heading"]
        values = same.centered_cosine
        worst = same.loc[values.idxmin()]
        differences.append(dict(representation=representation, same_world_pairs=len(same),
            mean_centered_cosine=float(values.mean()), min_centered_cosine=float(values.min()),
            median_centered_cosine=float(values.median()), max_centered_cosine=float(values.max()),
            worst_trial_a=worst.trial_a, worst_trial_b=worst.trial_b))
    differences = pd.DataFrame(differences)
    differences.to_csv(output / "inherited_heading_differences.csv", index=False)
    ordered = summary.head(20).copy()
    ordered["output_percent"] = ordered.output_share * 100
    subset = motifs[["source", "target", "query_status", "source_neurons", "target_neurons", "positive_pairs", "weight", "all_target_post_denominator", "target_input_fraction"]]
    text = ("# Connectivity summary\n\n"
        "All values are raw standard synaptic contact counts in male-cns:v1.0. "
        "No status, ROI, hemisphere, cropping or partner-type filter enters the outgoing census. "
        "Zero is no reported edge, not biological absence.\n\n"
        f"Selected hDeltaB: {len(den)}; outgoing weight: {int(den.all_outgoing_segment_weight.sum()):,}; "
        f"connected partner bodies: {len(partners):,}.\n\n"
        "Top 20 populations (all populations retained in census CSV):\n\n"
        + markdown(ordered[["population", "connected_partner_bodies", "weight", "all_outgoing_segment_denominator", "output_percent", "covered_sources", "selected_sources"]])
        + "\n\nAnnotation categories (disjoint; untyped and flagged properties are additional overlapping labels):\n\n"
        + markdown(categories) + "\n\nExplicitly whitelisted motifs:\n\n" + markdown(subset)
        + "\n\nMotifs marked population_absent were not queried because a selected population "
        "was absent; their contact counts and fractions are undefined, not measured zeros. "
        "Absent populations have no rows in the per-neuron input table. "
        "Target input fractions divide by the sum of recorded post counts of ALL selected target neurons. "
        "This denominator is external to the motif subgraph; a complete candidate input census was not run. "
        "Individual target denominators and body IDs are in hdeltab_input_by_candidate_neuron.csv. "
        "Outgoing fractions use contact weights, never presynaptic site counts.\n\n"
        "Inherited same-world/different-heading differences (b=1, 224 pairs per representation):\n\n"
        + markdown(differences) + "\n\n"
        "All saved matrices for queried motifs were reconstructed from edges over complete population axes; "
        "all outgoing hDeltaB candidate subsets match the independent Segment census. "
        "Raw untyped partners, annotations and edge ROI JSON remain in the extraction directory. "
        "No shared physical columns, functional angles, physiological signs, integration or homing are inferred.\n")
    (output / "connectivity_summary.md").write_text(text)
    # Hash verification repeated after analysis; no source changes are permitted.
    verify_input_data(census, manifest)
    verify_model_data(movement, mm)
    verify_model_data(trajectory, tm)
    save_json(output / "analysis.json", dict(status="complete", data_validated=True,
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
        source_manifest_sha256={str(p): sha256(p) for p in [census / "manifest.json", movement / "movement_model.json", trajectory / "trajectory_model.json"]},
        source_directories=[str(p.resolve()) for p in [census, movement, trajectory]],
        code_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        validation=dict(raw_aggregation_reproduced=True, matrices_reconstructed=True,
            complete_axes=True, motif_denominators_checked=True, model_body_ids_match=True,
            source_hashes_unchanged=True),
        data_sha256={p.name: sha256(p) for p in output.iterdir() if p.is_file()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("census", type=Path)
    parser.add_argument("--movement", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.census, args.movement, args.trajectory, args.output)


if __name__ == "__main__":
    main()
