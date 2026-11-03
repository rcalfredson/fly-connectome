"""Extract and validate only the anatomical projections added by hDeltaC."""

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Circuit, TYPES, digest, verify_bundle, write_json


ADDED_PROJECTIONS = (
    [(t, "hDeltaC") for t in TYPES]
    + [("hDeltaC", t) for t in TYPES]
    + [("hDeltaC", "hDeltaC"), ("PFNd", "hDeltaC"), ("PFNv", "hDeltaC")]
)
PAIR = ["bodyId_pre", "bodyId_post"]


def contact_matrix(edges, sources, targets):
    """Reconstruct a full matrix; successful empty responses retain both axes."""
    if set(edges.columns) != set(PAIR + ["weight"]):
        raise ValueError("Require body-ID pairs and standard contact weight only.")
    if edges.duplicated(PAIR).any():
        raise ValueError("Duplicate contact pairs.")
    for name in PAIR + ["weight"]:
        values = pd.to_numeric(edges[name], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 1).any() or (values % 1).any():
            raise ValueError("Require positive integer body IDs and contact counts.")
    if not edges.bodyId_pre.isin(sources).all() or not edges.bodyId_post.isin(targets).all():
        raise ValueError("Contacts outside the selected projection.")
    matrix = pd.DataFrame(0, index=pd.Index(sorted(sources), name="bodyId_pre"),
                          columns=pd.Index(sorted(targets), name="bodyId_post"), dtype="int64")
    for row in edges.itertuples(index=False):
        matrix.loc[row.bodyId_pre, row.bodyId_post] = int(row.weight)
    return matrix


def validate(source):
    """Offline verification, including all zero blocks and raw-response agreement."""
    source = Path(source)
    manifest = verify_bundle(source)
    if manifest.get("kind") != "home_vector_hdeltac_anatomy_v1":
        raise ValueError("Require a complete hDeltaC anatomy bundle.")
    if manifest.get("projections") != [list(pair) for pair in ADDED_PROJECTIONS]:
        raise ValueError("hDeltaC projection whitelist differs from the declared boundary.")
    base = Circuit(**json.loads((source / "base_circuit.json").read_text()))
    if base.population_types != TYPES or len(base.dynamic_ids) != 45 or len(base.input_ids) != 60:
        raise ValueError("hDeltaC extraction requires the original 60-input/45-dynamic core.")
    neurons = pd.read_csv(source / "neurons.csv")
    if neurons.bodyId.duplicated().any() or neurons.bodyId.isna().any():
        raise ValueError("Invalid selected neuron identities.")
    expected = dict(zip(base.input_ids + base.dynamic_ids, base.input_types + base.dynamic_types))
    hc = sorted(neurons.loc[neurons.type.eq("hDeltaC"), "bodyId"].tolist())
    if len(hc) != 20 or set(hc) & set(expected):
        raise ValueError("Require 20 distinct additional hDeltaC neurons.")
    expected.update({i: "hDeltaC" for i in hc})
    if dict(zip(neurons.bodyId, neurons.type)) != expected:
        raise ValueError("Selected body/type identities differ from the core plus hDeltaC.")
    populations = {t: sorted(neurons.loc[neurons.type.eq(t), "bodyId"])
                   for t in ["PFNd", "PFNv", *TYPES, "hDeltaC"]}
    incoming = pd.read_csv(source / "incoming_connections.csv")
    outgoing = pd.read_csv(source / "outgoing_connections.csv")
    contact_matrix(incoming, list(expected), hc)
    contact_matrix(outgoing, hc, base.dynamic_ids)
    raw = pd.concat([incoming, outgoing], ignore_index=True)
    rows = []
    for pre, post in ADDED_PROJECTIONS:
        prefix = f"{pre}_to_{post}"
        edges = pd.read_csv(source / f"{prefix}_connections.csv")
        selected = raw[raw.bodyId_pre.isin(populations[pre]) & raw.bodyId_post.isin(populations[post])]
        pd.testing.assert_frame_equal(edges.sort_values(PAIR).reset_index(drop=True),
                                      selected.sort_values(PAIR).reset_index(drop=True), check_dtype=False)
        matrix = contact_matrix(edges, populations[pre], populations[post])
        saved = pd.read_csv(source / f"{prefix}_matrix.csv", index_col=0)
        saved.columns = saved.columns.astype(int)
        pd.testing.assert_frame_equal(matrix, saved, check_names=False)
        rows.append(dict(source_type=pre, target_type=post, contacts=int(matrix.to_numpy().sum()),
                         positive_pairs=len(edges)))
    if manifest.get("summary") != rows:
        raise ValueError("Projection summaries disagree with the raw contacts.")
    return manifest, base, hc, rows


def extend(base, provenance, anatomy):
    """Append hDeltaC nodes and edges without perturbing any old block or its RNG draws."""
    anatomy = Path(anatomy)
    manifest_hash = digest(anatomy / "manifest.json")
    manifest, saved_base, hc, rows = validate(anatomy)
    if saved_base.as_dict() != base.as_dict():
        raise ValueError("hDeltaC anatomy was extracted for a different core circuit.")
    if list(manifest["snapshot"]) != list(provenance["snapshot"]):
        raise ValueError("hDeltaC and core anatomical snapshots differ.")
    result = copy.deepcopy(base.as_dict())
    result["dynamic_ids"] += hc
    result["dynamic_types"] += ["hDeltaC"] * len(hc)
    pre_index = {i: n for n, i in enumerate(result["input_ids"] + result["dynamic_ids"])}
    post_index = {i: n for n, i in enumerate(result["dynamic_ids"])}
    annotations = dict(zip(result["input_ids"] + result["dynamic_ids"],
                           result["input_types"] + result["dynamic_types"]))
    provenance = copy.deepcopy(provenance)
    for row in rows:
        pre, post = row["source_type"], row["target_type"]
        receivers = sum(t == post for t in result["dynamic_types"])
        denominator = row["contacts"] / receivers
        path = anatomy / f"{pre}_to_{post}_connections.csv"
        for edge in pd.read_csv(path).itertuples(index=False):
            if annotations[edge.bodyId_pre] != pre or annotations[edge.bodyId_post] != post:
                raise ValueError("Contact type disagrees with selected body annotations.")
            result["edges"].append(dict(pre=pre_index[edge.bodyId_pre], post=post_index[edge.bodyId_post],
                                        contacts=int(edge.weight), denominator=denominator,
                                        source_type=pre, target_type=post))
        provenance["projections"].append(dict(row, denominator=denominator,
                                               source_file=str(path.resolve())))
    verify_bundle(anatomy)
    if digest(anatomy / "manifest.json") != manifest_hash:
        raise ValueError("hDeltaC anatomy changed during assembly.")
    provenance["source_sha256"].update({str(p.resolve()): digest(p) for p in anatomy.iterdir() if p.is_file()})
    provenance.update(circuit_variant="hdeltac", added_types=["hDeltaC"],
                      input_mapping="Unchanged prescribed PFNd/PFNv rates; added direct inputs only if reported.")
    return Circuit(**result), provenance


def extract(client, source, census, output):
    """Two disjoint, bounded connection queries; failures remain failed bundles."""
    from neuprint import fetch_meta
    from fly_connectome.circuit_extension import neuron_query, save_raw
    from fly_connectome.connectivity import SERVER, DATASET
    from fly_connectome.extension_analysis import validate_saved_census
    from fly_connectome.pfn import match_metadata
    from .__main__ import begin, finish, load

    source, census = Path(source), Path(census)
    base, _, _ = load(source)
    if base.population_types != TYPES or len(base.edges) != 1075:
        raise ValueError("Require the unchanged original core as extraction source.")
    cm, candidates, _, _ = validate_saved_census(census)
    census_hash = digest(census / "manifest.json")
    provenance = json.loads((source / "sources.json").read_text())
    snapshot = [cm["server"], cm["dataset"], cm["dataset_metadata"]["uuid"],
                cm["dataset_metadata"]["latestMutationId"]]
    if provenance["snapshot"] != snapshot or snapshot[:2] != [SERVER, DATASET]:
        raise ValueError("Prepared core and census dataset snapshots differ.")
    hc = sorted(candidates.loc[candidates.type.eq("hDeltaC"), "bodyId"].tolist())
    if len(hc) != 20:
        raise ValueError("Expected the reviewed full population of 20 hDeltaC neurons.")
    output, record = begin(output, "home_vector_hdeltac_anatomy_v1", source=str(source.resolve()),
                            source_manifest_sha256=digest(source / "manifest.json"),
                            census=str(census.resolve()),
                            census_manifest_sha256=census_hash)
    record.update(server=SERVER, dataset=DATASET, snapshot=snapshot,
                  projections=[list(pair) for pair in ADDED_PROJECTIONS], queries={},
                  selection="Exact types, full populations; no status, ROI, cropped or confidence filters.",
                  weight="Standard all-ROI integer weight >=1; self-connections included.")
    write_json(output / "manifest.json", record)
    try:
        meta = fetch_meta(client=client)
        match_metadata(meta, cm["dataset_metadata"])
        record["dataset_metadata"] = meta
        types = ["PFNd", "PFNv", *TYPES, "hDeltaC"]
        record["queries"]["neurons"] = neuron_query(types)
        neurons = client.fetch_custom(record["queries"]["neurons"])
        save_raw(output / "neurons.csv", neurons)
        expected = dict(zip(base.input_ids + base.dynamic_ids, base.input_types + base.dynamic_types))
        expected.update({i: "hDeltaC" for i in hc})
        if neurons.bodyId.duplicated().any() or dict(zip(neurons.bodyId, neurons.type)) != expected:
            raise ValueError("Live full-population identities differ from the saved selection.")
        write_json(output / "base_circuit.json", base.as_dict())
        bounds = [("incoming", list(expected), hc), ("outgoing", hc, base.dynamic_ids)]
        responses = []
        for name, sources, targets in bounds:
            query = f"""MATCH (s:Neuron)-[e:ConnectsTo]->(t:Neuron)
WHERE s.bodyId IN {sorted(sources)} AND t.bodyId IN {sorted(targets)} AND e.weight >= 1
RETURN s.bodyId AS bodyId_pre, t.bodyId AS bodyId_post, e.weight AS weight
ORDER BY bodyId_pre, bodyId_post"""
            record["queries"][name] = query
            write_json(output / "manifest.json", record)
            edges = client.fetch_custom(query)
            edges.to_csv(output / f"{name}_connections.csv", index=False)
            contact_matrix(edges, sources, targets)
            responses.append(edges)
        raw = pd.concat(responses, ignore_index=True)
        populations = {t: sorted(neurons.loc[neurons.type.eq(t), "bodyId"]) for t in types}
        rows = []
        for pre, post in ADDED_PROJECTIONS:
            prefix = f"{pre}_to_{post}"
            selected = raw[raw.bodyId_pre.isin(populations[pre]) & raw.bodyId_post.isin(populations[post])]
            selected.to_csv(output / f"{prefix}_connections.csv", index=False)
            matrix = contact_matrix(selected, populations[pre], populations[post])
            matrix.to_csv(output / f"{prefix}_matrix.csv")
            # Cross-check both saved hDeltaC/vDeltaE projections and the full hDeltaB output census.
            if (pre, post) in [("hDeltaC", "vDeltaE"), ("vDeltaE", "hDeltaC")]:
                old = pd.read_csv(census / f"{prefix}_matrix.csv", index_col=0)
                old.columns = old.columns.astype(int)
                pd.testing.assert_frame_equal(matrix, old, check_names=False)
            if pre == "hDeltaB":
                old = pd.read_csv(census / "outgoing_connections.csv")
                old = old[old.bodyId_post.isin(hc)][PAIR + ["weight"]]
                pd.testing.assert_frame_equal(matrix, contact_matrix(old, populations[pre], hc))
            rows.append(dict(source_type=pre, target_type=post, contacts=int(selected.weight.sum()),
                             positive_pairs=len(selected)))
        record["summary"] = rows
        match_metadata(fetch_meta(client=client), meta)
        validate_saved_census(census)
        if digest(census / "manifest.json") != census_hash:
            raise ValueError("Census changed during extraction.")
        finish(output, record)
        validate(output)
    except Exception as error:
        # HTTP errors can contain request details: persist only the exception class.
        record.update(status="failed", error_type=type(error).__name__)
        write_json(output / "manifest.json", record)
        raise
    return dict(output=str(output), added_neurons=len(hc), projections=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Original complete core prepared bundle")
    parser.add_argument("--census", required=True, help="Verified census containing full hDeltaC identities")
    parser.add_argument("--output", required=True, help="New anatomy output directory")
    args = parser.parse_args(argv)
    from dotenv import load_dotenv
    from neuprint import Client
    from fly_connectome.connectivity import SERVER, DATASET
    load_dotenv(Path.cwd() / ".env", override=False)
    token = os.environ.get("NEUPRINT_TOKEN", "").strip()
    if not token:
        print("Set NEUPRINT_TOKEN locally before extraction.")
        return 1
    try:
        result = extract(Client(SERVER, dataset=DATASET, token=token),
                         args.source, args.census, args.output)
    except Exception as error:
        print(f"hDeltaC extraction stopped ({type(error).__name__}); inspect the safe manifest.")
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
