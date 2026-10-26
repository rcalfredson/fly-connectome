"""Bounded hDeltaB output census and explicitly selected motif queries; no dynamics.

Run ``python -m fly_connectome.circuit_extension --help``. Authentication follows
the existing extraction command. Raw responses and safe failure metadata survive.
"""
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

from .connectivity import DATASET, SERVER, ValidationError, build_matrix, fetch_connections
from .inputs import integers, save_json, sha256, UNTYPED
from .pfn import match_metadata
from .rate_model import load_network

ANNOTATIONS = ["type", "instance", "status", "statusLabel", "cropped",
               "hemibrainType", "flywireType", "consensusNt", "pre", "post"]
PAIR = ["bodyId_pre", "bodyId_post"]
# Fixed before extraction: travel relay, two memory candidates, anatomical input
# control for vDeltaE, and goal/steering endpoints. No discovered-node expansion.
TYPES = ["hDeltaB", "PFR_a", "PFR_b", "hDeltaA", "hDeltaG", "vDeltaE",
         "hDeltaC", "FC2A", "FC2B", "FC2C", "PFL3"]
CORE = ["PFR_a", "PFR_b", "hDeltaA", "hDeltaG", "vDeltaE"]
MOTIFS = sorted(set(
    [("hDeltaB", "hDeltaB")]
    + [(a, b) for t in CORE for a, b in [("hDeltaB", t), (t, "hDeltaB"), (t, t)]]
    + [("vDeltaE", "hDeltaG"), ("hDeltaG", "vDeltaE"),
       ("hDeltaC", "vDeltaE"), ("vDeltaE", "hDeltaC")]
    + [(a, b) for t in ["hDeltaA", "hDeltaG", "vDeltaE", "hDeltaB"]
       for f in ["FC2A", "FC2B", "FC2C"] for a, b in [(t, f), (f, t)]]
    + [(f, "PFL3") for f in ["FC2A", "FC2B", "FC2C", "hDeltaB", "hDeltaA", "hDeltaG"]]
    + [("PFL3", t) for t in ["hDeltaA", "hDeltaG"]]
    + [("PFR_a", "PFR_b"), ("PFR_b", "PFR_a")]
))


def neuron_query(types):
    fields = ", ".join(f"n.{c} AS {c}" for c in ANNOTATIONS)
    return (f"MATCH (n:Neuron) WHERE n.type IN {json.dumps(types)} "
            f"RETURN n.bodyId AS bodyId, {fields}, labels(n) AS labels ORDER BY bodyId")


def output_query(ids):
    fields = ", ".join(f"t.{c} AS {c}" for c in ANNOTATIONS)
    return f"""MATCH (s:Neuron) WHERE s.bodyId IN {sorted(map(int, ids))}
WITH s MATCH (s)-[e:ConnectsTo]->(t:Segment) WHERE e.weight >= 1
RETURN s.bodyId AS bodyId_pre, t.bodyId AS bodyId_post, e.weight AS weight,
       {fields}, labels(t) AS labels, e.roiInfo AS roiInfo
ORDER BY bodyId_pre, bodyId_post"""


def validate_edges(edges, sources, targets=None):
    edges = edges.copy()
    for c in PAIR:
        edges[c] = integers(edges[c], c, 1)
    edges["weight"] = integers(edges.weight, "weight", 1)
    if edges.duplicated(PAIR).any() or not edges.bodyId_pre.isin(sources).all():
        raise ValidationError("Duplicate pairs or source outside selection.")
    if targets is not None and not edges.bodyId_post.isin(targets).all():
        raise ValidationError("Target outside selection.")
    return edges


def summarize_outputs(raw, sources):
    """Outgoing weight counts postsynaptic contacts, NOT unique presynaptic sites."""
    if sources.bodyId.duplicated().any() or sources.empty:
        raise ValidationError("Sources must be nonempty and unique.")
    integers(sources.bodyId, "Source IDs", 1)
    edges = validate_edges(raw, sources.bodyId)
    if not edges.labels.map(lambda x: isinstance(x, list) and "Segment" in x).all():
        raise ValidationError("Every partner must be a Segment.")
    edges["labels"] = edges.labels.map(lambda x: json.dumps(sorted(x)))
    partners = edges.rename(columns={"bodyId_post": "bodyId"})[
        ["bodyId", *ANNOTATIONS, "labels"]].drop_duplicates()
    if partners.bodyId.duplicated().any() or partners.type.eq(UNTYPED).any():
        raise ValidationError("Inconsistent partner annotations or reserved type label.")
    typed = partners.type.notna() & partners.type.fillna("").str.strip().ne("")
    partners["population"] = partners.type.where(typed, UNTYPED)
    partners["is_neuron"] = partners.labels.map(lambda x: "Neuron" in json.loads(x))
    partners["category"] = np.where(partners.is_neuron,
        np.where(typed, "annotated_neuron", "untyped_neuron"), "non_neuron_segment")
    partners["status_flagged"] = partners.status.isin(["Glia", "Orphan", "Unimportant"]) | partners.statusLabel.isin(["Glia", "Orphan", "Unimportant"])
    enriched = edges.merge(partners[["bodyId", "population", "category", "status_flagged"]],
                           left_on="bodyId_post", right_on="bodyId", validate="many_to_one")
    ids = sorted(sources.bodyId)
    totals = edges.groupby("bodyId_pre").weight.sum().reindex(ids, fill_value=0)
    den = totals.rename("all_outgoing_segment_weight").rename_axis("bodyId").reset_index()
    # pre is a count of sites; outgoing weights count polyadic contacts. No equality assertion.
    den = den.merge(sources[["bodyId", "pre"]], validate="one_to_one")
    for label in ["annotated_neuron", "untyped_neuron", "non_neuron_segment"]:
        den[label + "_weight"] = den.bodyId.map(enriched[enriched.category == label].groupby("bodyId_pre").weight.sum()).fillna(0).astype(int)
    index = pd.MultiIndex.from_product([sorted(partners.population.unique()), ids], names=["population", "bodyId_pre"])
    per = enriched.groupby(["population", "bodyId_pre"]).weight.sum().reindex(index, fill_value=0).rename("weight").reset_index()
    per["all_outgoing_segment_denominator"] = per.bodyId_pre.map(totals)
    per["output_fraction"] = per.weight / per.all_outgoing_segment_denominator.replace(0, np.nan)
    rows = []
    grand = int(totals.sum())
    for pop, group in per.groupby("population"):
        rows.append(dict(population=pop, connected_partner_bodies=int(partners.population.eq(pop).sum()),
            weight=int(group.weight.sum()), all_outgoing_segment_denominator=grand,
            output_share=group.weight.sum() / grand if grand else np.nan,
            covered_sources=int(group.weight.gt(0).sum()), selected_sources=len(ids),
            source_coverage=float(group.weight.gt(0).mean()),
            weight_min=int(group.weight.min()), weight_max=int(group.weight.max())))
    summary = pd.DataFrame(rows, columns=["population", "connected_partner_bodies", "weight",
        "all_outgoing_segment_denominator", "output_share", "covered_sources", "selected_sources",
        "source_coverage", "weight_min", "weight_max"]).sort_values(["weight", "population"], ascending=[False, True])
    if int(summary.weight.sum()) != grand or int(per.weight.sum()) != grand:
        raise ValidationError("Output aggregation failed conservation.")
    return partners.sort_values("bodyId"), den, per, summary


def save_raw(path, table):
    table = table.copy()
    if "labels" in table:
        table["labels"] = table.labels.map(json.dumps)
    table.to_csv(path, index=False)


def run(client, source, output):
    source, output = Path(source), Path(output)
    _, populations, _, baseline, source_hashes = load_network(source)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="running", started_at_utc=datetime.now(timezone.utc).isoformat(),
        server=SERVER, dataset=DATASET, source_directory=str(source.resolve()), source_sha256=source_hashes,
        code_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        packages={n: version(n) for n in ["neuprint-python", "pandas", "numpy"]},
        types=TYPES, motifs=MOTIFS, queries={},
        settings=dict(min_weight=1, weight_property="weight", roi_filter=None,
            status_filter=None, cropped_filter=None, output_partner_label="Segment",
            expansion="one outgoing census plus fixed directed pair whitelist"),
        definitions=dict(output_share="Population contact weight / all positive outgoing Segment contact weights from all 19 selected hDeltaB.",
            output_fraction="Population weight from a source / all positive outgoing Segment contact weights from that source.",
            source_coverage="Sources reaching population / all 19 selected hDeltaB; zero-output sources retained.",
            target_input_fraction="Whitelisted source population weight / recorded postsynaptic count of complete target population; not fraction of this queried subgraph.",
            zero="No reported connection, not biological absence; failed query never becomes zero.",
            pre="Unique presynaptic sites; NOT a valid outgoing contact-weight denominator (polyadic synapses).",
            roi="Raw edge roiInfo preserved; overlapping ROI values never summed.",
            self="Self edges included; recurrence reported separately from off-diagonal recurrence."))
    save_json(output / "manifest.json", manifest)
    try:
        meta = fetch_meta(client=client)
        match_metadata(meta, baseline["dataset_metadata"])
        manifest["dataset_metadata"] = meta
        query = neuron_query(TYPES)
        manifest["queries"]["neurons"] = query
        neurons = client.fetch_custom(query)
        save_raw(output / "candidate_neurons.csv", neurons)
        integers(neurons.bodyId, "Candidate IDs", 1)
        if neurons.bodyId.duplicated().any() or not neurons.type.isin(TYPES).all():
            raise ValidationError("Invalid candidate population annotations.")
        integers(neurons.post, "Target post counts")
        live = neurons[neurons.type == "hDeltaB"].sort_values("bodyId").reset_index(drop=True)
        pd.testing.assert_frame_equal(live[["bodyId", "type", "instance"]],
            populations["hDeltaB"][["bodyId", "type", "instance"]].sort_values("bodyId").reset_index(drop=True), check_dtype=False)
        save_raw(output / "hDeltaB_neurons.csv", live)
        reference = fetch_connections(client, [26913], [515192])
        reference.to_csv(output / "reference_connections.csv", index=False)
        if int(build_matrix(reference, [26913], [515192]).iloc[0, 0]) != 137:
            raise ValidationError("Reference weight changed.")
        query = output_query(live.bodyId)
        manifest["queries"]["outputs"] = query
        save_json(output / "manifest.json", manifest)
        raw = client.fetch_custom(query)
        save_raw(output / "raw_outputs.csv", raw)
        partners, den, per, summary = summarize_outputs(raw, live)
        partners.to_csv(output / "downstream_partners.csv", index=False)
        raw[PAIR + ["weight", "roiInfo"]].to_csv(output / "outgoing_connections.csv", index=False)
        den.to_csv(output / "output_denominators.csv", index=False)
        per.to_csv(output / "population_by_source.csv", index=False)
        summary.to_csv(output / "population_outputs.csv", index=False)
        rows, all_edges = [], []
        for pre, post in MOTIFS:
            src = neurons.loc[neurons.type == pre, "bodyId"].tolist()
            dst = neurons.loc[neurons.type == post, "bodyId"].tolist()
            if not src or not dst:
                rows.append(dict(source=pre, target=post, query_status="population_absent", source_neurons=len(src), target_neurons=len(dst)))
                continue
            query = f"""MATCH (s:Neuron)-[e:ConnectsTo]->(t:Neuron)
WHERE s.bodyId IN {sorted(src)} AND t.bodyId IN {sorted(dst)} AND e.weight >= 1
RETURN s.bodyId AS bodyId_pre, t.bodyId AS bodyId_post, e.weight AS weight, e.roiInfo AS roiInfo
ORDER BY bodyId_pre, bodyId_post"""
            manifest["queries"][pre + "_to_" + post] = query
            save_json(output / "manifest.json", manifest)
            edges = client.fetch_custom(query)
            edges.to_csv(output / f"{pre}_to_{post}_connections.csv", index=False)
            edges = validate_edges(edges, src, dst)
            matrix = build_matrix(edges, src, dst)
            matrix.to_csv(output / f"{pre}_to_{post}_matrix.csv")
            # Independent census cross-check for all hDeltaB outgoing subsets.
            if pre == "hDeltaB":
                expected = raw[raw.bodyId_post.isin(dst)][PAIR + ["weight"]].sort_values(PAIR).reset_index(drop=True)
                pd.testing.assert_frame_equal(edges[PAIR + ["weight"]].sort_values(PAIR).reset_index(drop=True), expected, check_dtype=False)
            total = int(edges.weight.sum())
            denominator = int(neurons.loc[neurons.bodyId.isin(dst), "post"].sum())
            if total > denominator:
                raise ValidationError("Subset weight exceeds target post count.")
            outgoing_denominator = int(den.all_outgoing_segment_weight.sum()) if pre == "hDeltaB" else None
            rows.append(dict(source=pre, target=post, query_status="queried", source_neurons=len(src),
                target_neurons=len(dst), possible_pairs=len(src)*len(dst), positive_pairs=len(edges),
                weight=total, self_weight=int(edges.loc[edges.bodyId_pre == edges.bodyId_post, "weight"].sum()),
                off_diagonal_weight=int(edges.loc[edges.bodyId_pre != edges.bodyId_post, "weight"].sum()),
                covered_sources=edges.bodyId_pre.nunique(), covered_targets=edges.bodyId_post.nunique(),
                all_target_post_denominator=denominator, target_input_fraction=total/denominator if denominator else None,
                all_hdeltab_output_denominator=outgoing_denominator,
                hdeltab_output_share=total/outgoing_denominator if outgoing_denominator else None))
            all_edges.append(edges.assign(source_type=pre, target_type=post))
        pd.DataFrame(rows).to_csv(output / "motif_summary.csv", index=False)
        pd.concat(all_edges, ignore_index=True).to_csv(output / "motif_connections.csv", index=False)
        inventory_query = "MATCH (n:Neuron) WHERE n.type IS NOT NULL RETURN n.type AS type, count(n) AS neuron_count ORDER BY type"
        manifest["queries"]["type_inventory"] = inventory_query
        client.fetch_custom(inventory_query).to_csv(output / "type_inventory.csv", index=False)
        match_metadata(fetch_meta(client=client), meta)
        if any(sha256(source / n) != h for n, h in source_hashes.items()):
            raise ValidationError("Baseline changed during extraction.")
        manifest.update(status="complete", data_validated=True,
            validation=dict(selected_hdeltab=len(live), positive_output_pairs=len(raw),
                downstream_bodies=len(partners), total_output_weight=int(raw.weight.sum()),
                raw_summary_conservation=True, census_motif_agreement=True,
                baseline_hashes_unchanged=True, metadata_unchanged=True),
            data_sha256={p.name: sha256(p) for p in output.glob("*.csv")})
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original complete PFNd/hDeltaB extraction")
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing directories refused")
    args = parser.parse_args()
    load_dotenv(Path.cwd() / ".env", override=False)
    token = os.environ.get("NEUPRINT_TOKEN", "").strip()
    if not token:
        raise ValidationError("Set NEUPRINT_TOKEN locally.")
    try:
        result = run(Client(SERVER, dataset=DATASET, token=token), args.source, args.output)
    except Exception as exc:
        # Suppress arbitrary HTTP exception strings, which can contain request data.
        print(f"Extraction stopped ({type(exc).__name__}); inspect safe manifest/diagnostics.")
        return 1
    print(json.dumps(result["validation"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
