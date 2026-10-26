"""One-hop all-Segment input census of vDeltaE; anatomical counts, no dynamics."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from neuprint import Client, fetch_meta

from .circuit_extension import neuron_query, save_raw, validate_edges
from .connectivity import DATASET, SERVER, ValidationError, build_matrix, fetch_connections, type_inventory
from .extension_analysis import markdown
from .inputs import UPSTREAM, UNTYPED, PAIR, integers, save_json, sha256, upstream_query, verify_input_data
from .pfn import match_metadata
from .rate_model import verify_model_data


def prepare(raw, targets):
    if targets.empty or targets.bodyId.duplicated().any() or not targets.type.eq("vDeltaE").all():
        raise ValidationError("Expected nonempty unique exact-type vDeltaE targets.")
    integers(targets.bodyId, "Target IDs", 1)
    integers(targets.post, "Target post counts")
    raw = validate_edges(raw, raw.bodyId_pre.unique(), targets.bodyId)
    if not raw.labels.map(lambda v: isinstance(v, list) and "Segment" in v).all():
        raise ValidationError("All sources must be Segment nodes.")
    raw["labels"] = raw.labels.map(lambda v: json.dumps(sorted(v)))
    sources = raw.rename(columns={"bodyId_pre": "bodyId"})[UPSTREAM].drop_duplicates()
    if sources.bodyId.duplicated().any() or sources.type.eq(UNTYPED).any():
        raise ValidationError("Inconsistent source annotations or reserved type label.")
    typed = sources.type.notna() & sources.type.fillna("").str.strip().ne("")
    sources["population"] = sources.type.where(typed, UNTYPED)
    sources["is_neuron"] = sources.labels.map(lambda v: "Neuron" in json.loads(v))
    sources["category"] = np.where(sources.is_neuron,
        np.where(typed, "annotated_neuron", "untyped_neuron"), "non_neuron_segment")
    sources["status_flagged"] = sources.status.isin(["Glia", "Orphan", "Unimportant"]) | sources.statusLabel.isin(["Glia", "Orphan", "Unimportant"])
    sources = sources.sort_values("bodyId").reset_index(drop=True)
    edges = raw[[*PAIR, "weight", "roiInfo"]].sort_values(PAIR).reset_index(drop=True)
    enriched = edges.merge(sources[["bodyId", "population", "category", "status_flagged"]],
        left_on="bodyId_pre", right_on="bodyId", validate="many_to_one")
    ids = sorted(targets.bodyId)
    totals = edges.groupby("bodyId_post").weight.sum().reindex(ids, fill_value=0)
    den = targets.set_index("bodyId")[["post"]].reindex(ids).rename(columns={"post": "recorded_post"})
    den["all_segment_weight"] = totals
    den["post_minus_edge_residual"] = den.recorded_post - totals
    for category in ["annotated_neuron", "untyped_neuron", "non_neuron_segment"]:
        den[category + "_weight"] = enriched[enriched.category == category].groupby("bodyId_post").weight.sum().reindex(ids, fill_value=0)
    for name, mask in [("untyped", enriched.population.eq(UNTYPED)), ("flagged", enriched.status_flagged)]:
        den[name + "_weight"] = enriched[mask].groupby("bodyId_post").weight.sum().reindex(ids, fill_value=0)
    index = pd.MultiIndex.from_product([sorted(sources.population.unique()), ids], names=["population", "bodyId_post"])
    per = enriched.groupby(["population", "bodyId_post"]).weight.sum().reindex(index, fill_value=0).rename("weight").reset_index()
    per["all_segment_denominator"] = per.bodyId_post.map(totals)
    per["input_fraction"] = per.weight / per.all_segment_denominator.replace(0, np.nan)
    rows = []
    grand = int(totals.sum())
    for pop, group in per.groupby("population"):
        rows.append(dict(population=pop, connected_source_bodies=int(sources.population.eq(pop).sum()),
            weight=int(group.weight.sum()), all_segment_denominator=grand,
            input_share=group.weight.sum()/grand if grand else np.nan,
            covered_targets=int(group.weight.gt(0).sum()), selected_targets=len(ids),
            coverage=float(group.weight.gt(0).mean()), weight_min=int(group.weight.min()),
            weight_median=float(group.weight.median()), weight_max=int(group.weight.max()),
            fraction_min=group.input_fraction.min(), fraction_max=group.input_fraction.max()))
    summary = pd.DataFrame(rows, columns=["population", "connected_source_bodies", "weight", "all_segment_denominator",
        "input_share", "covered_targets", "selected_targets", "coverage", "weight_min", "weight_median",
        "weight_max", "fraction_min", "fraction_max"]).sort_values(["weight", "population"], ascending=[False, True])
    if int(summary.weight.sum()) != grand or int(per.weight.sum()) != grand:
        raise ValidationError("Input aggregation did not conserve contacts.")
    return {"upstream_neurons.csv": sources, "incoming_connections.csv": edges,
        "input_denominators.csv": den.rename_axis("bodyId").reset_index(),
        "population_by_target.csv": per, "population_inputs.csv": summary}


def overlap_tables(edges, mapping, targets):
    rows, matrices = [], {}
    denominator = int(targets.post.sum())
    for pop in ["PFNd", "PFNv", "hDeltaB"]:
        ids = sorted(mapping.loc[mapping.type == pop, "bodyId"])
        if not ids:
            raise ValidationError("Modeled populations must be nonempty.")
        selected = edges[edges.bodyId_pre.isin(ids)]
        matrix = build_matrix(selected, ids, targets.bodyId)
        matrices[pop] = matrix
        rows.append(dict(population=pop, modeled_sources=len(ids),
            connected_sources=selected.bodyId_pre.nunique(), positive_pairs=len(selected),
            weight=int(selected.weight.sum()), all_vdeltae_post_denominator=denominator,
            input_share=selected.weight.sum()/denominator if denominator else np.nan,
            covered_targets=selected.bodyId_post.nunique(), selected_targets=len(targets)))
    return pd.DataFrame(rows), matrices


def verify(source):
    source = Path(source)
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValidationError("Verification requires a complete census.")
    verify_input_data(source, manifest)
    raw = pd.read_csv(source / "raw_connections.csv")
    raw["labels"] = raw.labels.map(json.loads)
    targets = pd.read_csv(source / "vDeltaE_neurons.csv")
    tables = prepare(raw, targets)
    for name, table in tables.items():
        pd.testing.assert_frame_equal(table.reset_index(drop=True), pd.read_csv(source / name), check_dtype=False)
    if tables["input_denominators.csv"].post_minus_edge_residual.ne(0).any():
        raise ValidationError("Incoming weights do not reconcile with target post counts.")
    mapping = pd.read_csv(source / "modeled_neurons.csv")
    summary, matrices = overlap_tables(tables["incoming_connections.csv"], mapping, targets)
    pd.testing.assert_frame_equal(summary, pd.read_csv(source / "model_overlap_summary.csv"), check_dtype=False)
    for pop, expected in matrices.items():
        actual = pd.read_csv(source / f"{pop}_to_vDeltaE_matrix.csv", index_col=0)
        actual.columns = actual.columns.astype(int)
        pd.testing.assert_frame_equal(expected, actual, check_names=False)
    verify_input_data(source, manifest)
    return manifest


def run(client, baseline_dir, movement_dir, output):
    baseline_dir, movement_dir, output = map(Path, [baseline_dir, movement_dir, output])
    baseline = json.loads((baseline_dir / "manifest.json").read_text())
    model = json.loads((movement_dir / "movement_model.json").read_text())
    if (baseline.get("status") != "complete" or baseline.get("dataset") != DATASET
            or baseline.get("server") != SERVER or model.get("status") != "complete"):
        raise ValidationError("Use complete configured dataset census and movement model.")
    verify_input_data(baseline_dir, baseline)
    verify_model_data(movement_dir, model)
    previous = pd.read_csv(baseline_dir / "candidate_neurons.csv")
    saved_targets = previous[previous.type == "vDeltaE"]
    config = json.loads((movement_dir / "movement_config.json").read_text())
    encoding = pd.read_csv(movement_dir / "PFN_encoding_map.csv")
    encoding = encoding.rename(columns={"bodyId_pre": "bodyId"})
    mapping = pd.concat([encoding[["bodyId", "type"]], pd.DataFrame(
        dict(bodyId=config["target_ids"], type="hDeltaB"))], ignore_index=True).sort_values("bodyId").reset_index(drop=True)
    if mapping.bodyId.duplicated().any() or set(encoding.bodyId) != set(config["source_ids"]):
        raise ValidationError("Invalid model identity map.")
    output.mkdir(parents=True, exist_ok=False)
    source_manifests = [baseline_dir / "manifest.json", movement_dir / "movement_model.json"]
    manifest = dict(status="running", server=SERVER, dataset=DATASET,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        source_manifest_sha256={str(p.resolve()): sha256(p) for p in source_manifests},
        code_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        packages={n: version(n) for n in ["neuprint-python", "numpy", "pandas"]},
        settings=dict(target_type="vDeltaE", source_label="Segment", hops=1, min_weight=1,
            weight_property="weight", status_filter=None, cropped_filter=None, roi_filter=None,
            self_and_recurrent_connections="included", untyped_partners="included"), queries={},
        definitions=dict(input_share="Population weight / all incoming Segment weights across all selected vDeltaE targets.",
            input_fraction="Population weight to a target / all incoming Segment weights to that target.",
            coverage="Targets reached / complete selected vDeltaE count, including zero-input targets.",
            zero_denominator="Undefined fractions saved blank; target retained.",
            reconciliation="Recorded target post count minus incoming Segment weight must equal zero for every target.",
            roi="Raw edge roiInfo preserved only; overlapping regions not summed or used to infer columns.",
            zero="No reported connection in dataset, not proof of biological absence.",
            annotations="Untyped/flagged totals overlap categories; unknown reconstruction fields do not imply completeness."))
    save_json(output / "manifest.json", manifest)
    try:
        meta = fetch_meta(client=client)
        match_metadata(meta, baseline["dataset_metadata"])
        manifest["dataset_metadata"] = meta
        inventory = type_inventory(client)
        inventory.to_csv(output / "type_inventory.csv", index=False)
        related = inventory.type.dropna().loc[inventory.type.dropna().str.contains("vdeltae", case=False)]
        if set(related) != {"vDeltaE"}:
            raise ValidationError("Changed or ambiguous vDeltaE nomenclature; review type inventory.")
        query = neuron_query(["vDeltaE"])
        manifest["queries"]["targets"] = query
        targets = client.fetch_custom(query).sort_values("bodyId").reset_index(drop=True)
        save_raw(output / "vDeltaE_neurons.csv", targets)
        pd.testing.assert_frame_equal(targets[["bodyId", "type", "instance"]],
            saved_targets[["bodyId", "type", "instance"]].sort_values("bodyId").reset_index(drop=True), check_dtype=False)
        reference = fetch_connections(client, [26913], [515192])
        reference.to_csv(output / "reference_connections.csv", index=False)
        if int(build_matrix(reference, [26913], [515192]).iloc[0, 0]) != 137:
            raise ValidationError("Reference connection changed.")
        query = upstream_query(targets.bodyId)
        manifest["queries"]["inputs"] = query
        save_json(output / "manifest.json", manifest)
        raw = client.fetch_custom(query)
        save_raw(output / "raw_connections.csv", raw)
        tables = prepare(raw, targets)
        for name, table in tables.items():
            table.to_csv(output / name, index=False)
        edges = tables["incoming_connections.csv"]
        for pop in ["hDeltaB", "hDeltaG", "hDeltaC", "vDeltaE"]:
            ids = previous.loc[previous.type == pop, "bodyId"]
            actual = edges[edges.bodyId_pre.isin(ids)][PAIR + ["weight"]].sort_values(PAIR).reset_index(drop=True)
            expected = pd.read_csv(baseline_dir / f"{pop}_to_vDeltaE_connections.csv")[PAIR + ["weight"]].sort_values(PAIR).reset_index(drop=True)
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
        mapping.to_csv(output / "modeled_neurons.csv", index=False)
        overlap, matrices = overlap_tables(edges, mapping, targets)
        overlap.to_csv(output / "model_overlap_summary.csv", index=False)
        for pop, matrix in matrices.items():
            matrix.to_csv(output / f"{pop}_to_vDeltaE_matrix.csv")
        den = tables["input_denominators.csv"]
        if den.post_minus_edge_residual.ne(0).any():
            raise ValidationError("Inspect per-target input residuals; post counts did not reconcile.")
        summary = tables["population_inputs.csv"].head(25).copy()
        summary["input_percent"] = summary.input_share * 100
        (output / "summary.md").write_text("# vDeltaE input census\n\n"
            f"{len(targets)} selected vDeltaE neurons, {len(tables['upstream_neurons.csv']):,} source bodies, "
            f"{len(edges):,} positive pairs and {int(edges.weight.sum()):,} contacts. "
            "Every target's incoming Segment weight reconciles with its recorded post count.\n\n"
            "## Leading inputs\n\n" + markdown(summary[["population", "connected_source_bodies", "weight", "all_segment_denominator", "input_percent", "covered_targets", "selected_targets"]])
            + "\n\nAll populations, raw body IDs, annotations and ROI JSON are retained in CSV. "
            "Untyped fragments are included. Fractions use all incoming Segment contacts; source-body counts can include fragments.\n\n"
            "## Existing modeled populations\n\n" + markdown(overlap)
            + "\n\nThe three matrices retain every modeled source and every selected target, including zeros. "
            "hDeltaB/hDeltaG/hDeltaC/vDeltaE subsets match the earlier independent candidate queries. "
            "No additional hop is extracted. Counts establish anatomical inputs, not their physiological signs, "
            "tuning, efficacy or contribution to integration.\n")
        verify_input_data(baseline_dir, baseline)
        verify_model_data(movement_dir, model)
        match_metadata(fetch_meta(client=client), meta)
        if any(sha256(Path(p)) != h for p, h in manifest["source_manifest_sha256"].items()):
            raise ValidationError("Source manifest changed during extraction.")
        manifest.update(status="complete", data_validated=True,
            validation=dict(targets=len(targets), upstream_bodies=len(tables["upstream_neurons.csv"]),
                positive_pairs=len(edges), total_weight=int(edges.weight.sum()),
                post_counts_reconcile=True, prior_subsets_match=True, source_hashes_unchanged=True),
            data_sha256={p.name: sha256(p) for p in output.iterdir() if p.suffix in [".csv", ".md"]})
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(output / "manifest.json", manifest)
    try:
        return verify(output)
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__)
        save_json(output / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Prior candidate census, or census to verify")
    parser.add_argument("--movement", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify(args.source)["validation"], indent=2))
        return 0
    if args.movement is None or args.output is None:
        parser.error("Extraction requires --movement and --output.")
    load_dotenv(Path.cwd() / ".env", override=False)
    token = os.environ.get("NEUPRINT_TOKEN", "").strip()
    if not token:
        raise ValidationError("Set NEUPRINT_TOKEN locally.")
    try:
        result = run(Client(SERVER, dataset=DATASET, token=token), args.source, args.movement, args.output)
    except Exception as exc:
        print(f"Extraction stopped ({type(exc).__name__}); inspect safe manifest and raw diagnostics.")
        return 1
    print(json.dumps(result["validation"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
