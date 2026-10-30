"""Bounded vDeltaE -> hDeltaG contacts, with explicit confidence and ROI scopes.

No shared column grid, compartment-specific sign or neural dynamics is inferred.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from neuprint import fetch_meta

from .connectivity import DATASET, SERVER, ValidationError
from .extension_analysis import markdown, validate_saved_census
from .inputs import integers, save_json, sha256
from .vdeltae_spatial_audit import check_probe_metadata, select_spatial_population, spatial_probe_queries
from .vdeltae_spatial_pilot import (
    XYZ, LAYERS, pilot_code_provenance, record_artifact_hashes,
    verify_coordinate_pilot,
)

KIND = "vdeltae_hdeltag_contact_audit_v1"
PAIR = ["bodyId_pre", "bodyId_post"]
SITE_FIELDS = ["synapse_node_id", "synapse_type", *XYZ, "confidence", "compartment", "split", "FB", *LAYERS]
RAW_COLUMNS = [*PAIR, "contact_relationship_id", *[
    f"{field}_{side}" for side in ["pre", "post"] for field in SITE_FIELDS
]]


def csv_comparable(table):
    """Normalize only null spelling for comparison across a CSV round trip."""
    table = table.copy()
    for column in table:
        if table[column].dtype == object or isinstance(table[column].dtype, pd.StringDtype):
            values = table[column].astype(object)
            table[column] = values.where(values.notna(), np.nan)
    return table


def contact_query(bodies):
    bodies = select_spatial_population(bodies)
    pre = sorted(int(v) for v in bodies.loc[bodies.type.eq("vDeltaE"), "bodyId"])
    post = sorted(int(v) for v in bodies.loc[bodies.type.eq("hDeltaG"), "bodyId"])
    fields = []
    for side, variable in [("pre", "a"), ("post", "b")]:
        for field in SITE_FIELDS:
            expression = (f"id({variable})" if field == "synapse_node_id" else
                          f"{variable}.type" if field == "synapse_type" else
                          f"{variable}.location.{field}" if field in XYZ else f"{variable}.{field}")
            fields.append(f"{expression} AS {field}_{side}")
    return f"""MATCH (n:Neuron)-[:Contains]->(:SynapseSet)-[:Contains]->(a:Synapse)
        -[contact:SynapsesTo]->(b:Synapse)<-[:Contains]-(:SynapseSet)<-[:Contains]-(m:Neuron)
WHERE n.bodyId IN {pre} AND m.bodyId IN {post}
WITH DISTINCT n, m, a, b, contact
RETURN n.bodyId AS bodyId_pre, m.bodyId AS bodyId_post,
       id(contact) AS contact_relationship_id, {', '.join(fields)}
ORDER BY bodyId_pre, bodyId_post, synapse_node_id_pre, synapse_node_id_post"""


def load_sources(coordinates, census):
    """Verify both sources and their common dataset identity, before and after reading."""
    coordinates, census = Path(coordinates).resolve(), Path(census).resolve()
    before = {str(path): sha256(path) for path in [coordinates / "coordinates.json", census / "manifest.json"]}
    geometry = verify_coordinate_pilot(coordinates)
    if geometry.get("full_population") is not True:
        raise ValidationError("Contact audit requires the verified full-population coordinate run.")
    cm, neurons, _, _ = validate_saved_census(census)
    if cm.get("server") != SERVER or cm.get("dataset") != DATASET:
        raise ValidationError("Census target differs from the configured dataset.")
    check_probe_metadata(cm["dataset_metadata"], geometry["source_inventory"])
    bodies = pd.read_csv(coordinates / "probe_bodies.csv")
    select_spatial_population(bodies)
    columns = ["bodyId", "type", "instance"]
    expected = neurons.loc[neurons.type.isin(["vDeltaE", "hDeltaG"]), columns].sort_values("bodyId").reset_index(drop=True)
    pd.testing.assert_frame_equal(bodies[columns].sort_values("bodyId").reset_index(drop=True), expected, check_dtype=False)
    points = pd.read_csv(coordinates / "fb_sites_raw.csv", low_memory=False)
    edges = pd.read_csv(census / "vDeltaE_to_hDeltaG_connections.csv")
    verify_coordinate_pilot(coordinates)
    validate_saved_census(census)
    if before != {path: sha256(Path(path)) for path in before}:
        raise ValidationError("Source manifests changed during contact-source verification.")
    provenance = {
        "coordinates": str(coordinates), "census": str(census), "manifest_sha256": before,
        "connections_sha256": sha256(census / "vDeltaE_to_hDeltaG_connections.csv"),
    }
    return bodies, points, edges, geometry, provenance


def standard_mask(raw, threshold):
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValidationError("Invalid standard confidence threshold.")
    # Explicitly follow installed neuprint-python SynapseCriteria's strict > rule
    # for BOTH endpoints. Missing confidence is retained in raw, excluded here.
    return raw.confidence_pre.gt(threshold) & raw.confidence_post.gt(threshold)


def contact_scopes(raw, threshold):
    for confidence, mask in [("all_reported", pd.Series(True, index=raw.index)),
                             ("standard", standard_mask(raw, threshold))]:
        for roi, roi_mask in [("all_ROIs", pd.Series(True, index=raw.index)),
                              ("post_FB", raw.FB_post.eq(True)),
                              ("both_FB", raw.FB_post.eq(True) & raw.FB_pre.eq(True))]:
            yield f"{confidence}__{roi}", raw.loc[mask & roi_mask]


def pair_counts(raw, bodies):
    index = pd.MultiIndex.from_product([
        sorted(bodies.loc[bodies.type.eq("vDeltaE"), "bodyId"]),
        sorted(bodies.loc[bodies.type.eq("hDeltaG"), "bodyId"]),
    ], names=PAIR)
    return raw.groupby(PAIR).size().reindex(index, fill_value=0).rename("contacts")


def validate_contacts(raw, bodies, points, edges, threshold):
    if not set(RAW_COLUMNS).issubset(raw.columns):
        raise ValidationError("Raw contact table is missing required columns.")
    for column in [*PAIR, "contact_relationship_id", "synapse_node_id_pre", "synapse_node_id_post"]:
        integers(raw[column], column, 1 if column in PAIR else 0)
    for side, population in [("pre", "vDeltaE"), ("post", "hDeltaG")]:
        if not raw[f"bodyId_{side}"].isin(bodies.loc[bodies.type.eq(population), "bodyId"]).all():
            raise ValidationError("Contact endpoint is outside the selected directed populations.")
        if not raw[f"synapse_type_{side}"].eq(side).all():
            raise ValidationError("Unexpected contact synapse polarity.")
        if not np.isfinite(raw[[f"{axis}_{side}" for axis in XYZ]].to_numpy(dtype=float)).all():
            raise ValidationError("Missing or nonfinite contact coordinates.")
        scores = pd.to_numeric(raw[f"confidence_{side}"], errors="raise").dropna()
        if not np.isfinite(scores).all() or not scores.between(0, 1).all():
            raise ValidationError("Invalid nonmissing contact confidence.")
        for flag in ["FB", *LAYERS]:
            values = raw[f"{flag}_{side}"].dropna()
            if not values.map(lambda value: isinstance(value, (bool, np.bool_))).all():
                raise ValidationError("ROI flags must be boolean or missing.")
        columns = ["bodyId", *SITE_FIELDS]
        sites = raw[[f"{c}_{side}" for c in columns]].rename(columns={f"{c}_{side}": c for c in columns}).drop_duplicates()
        keys = ["bodyId", "synapse_node_id"]
        if sites.duplicated(keys).any():
            raise ValidationError("A shared contact site has inconsistent properties.")
        # Snapshot-local graph IDs cross-check the actual site, not nearest geometry.
        existing = pd.MultiIndex.from_frame(points[keys])
        membership = pd.MultiIndex.from_frame(sites[keys]).isin(existing)
        if not np.array_equal(membership, sites.FB.eq(True).to_numpy()):
            raise ValidationError("Contact FB membership differs from the saved site inventory.")
        actual = sites.loc[sites.FB.eq(True), columns].sort_values(keys).reset_index(drop=True)
        expected = points[columns].merge(actual[keys], on=keys, validate="one_to_one").sort_values(keys).reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(csv_comparable(actual), csv_comparable(expected),
                                          check_dtype=False, rtol=1e-12, atol=1e-12)
        except AssertionError as error:
            raise ValidationError("Contact FB site properties differ from the saved anatomy.") from error
    if raw.contact_relationship_id.duplicated().any() or raw.duplicated([
        *PAIR, "synapse_node_id_pre", "synapse_node_id_post"
    ]).any():
        raise ValidationError("Duplicate contact identity; do not silently deduplicate contacts.")
    standard = raw.loc[standard_mask(raw, threshold)]
    observed = pair_counts(standard, bodies)
    expected = edges.set_index(PAIR).weight.reindex(observed.index, fill_value=0)
    if not observed.eq(expected).all():
        raise ValidationError("Standard-confidence contact counts differ from saved body-pair weights.")
    fb_expected = []
    for record in edges.itertuples():
        try:
            info = json.loads(record.roiInfo)
            value = info.get("FB", {}).get("post", 0)
            count = integers(pd.Series([value]), "Saved FB post count").iloc[0]
            if count > record.weight:
                raise ValueError("FB count exceeds edge total")
        except (ValueError, TypeError, AttributeError) as error:
            raise ValidationError("Invalid saved edge ROI counts.") from error
        fb_expected.append((*[getattr(record, key) for key in PAIR], count))
    expected_fb = pd.DataFrame(fb_expected, columns=[*PAIR, "contacts"]).set_index(PAIR).contacts
    observed_fb = pair_counts(standard.loc[standard.FB_post.eq(True)], bodies)
    if not observed_fb.eq(expected_fb.reindex(observed_fb.index, fill_value=0)).all():
        raise ValidationError("Standard FB-post counts differ from saved edge ROI counts.")


def contact_tables(raw, bodies, threshold):
    enriched = raw.copy()
    for side in ["pre", "post"]:
        enriched = enriched.merge(bodies[["bodyId", "type", "instance", "annotation_column"]].rename(
            columns={name: f"{name}_{side}" for name in ["bodyId", "type", "instance", "annotation_column"]}
        ), on=f"bodyId_{side}", validate="many_to_one")
    enriched["standard_confidence"] = standard_mask(enriched, threshold)
    all_pairs = pair_counts(raw, bodies)
    pair_rows, compartment_rows, scope_rows = [], [], []
    tables = {"contacts_annotated.csv": enriched}
    body_lookup = bodies.set_index("bodyId")
    # None is retained as a distinct missing annotation, never renamed to axon/dendrite.
    compartments = ["axon", "dendrite", None] + sorted(set(raw.compartment_post.dropna()) - {"axon", "dendrite"})
    tables["compartment_inventory.csv"] = pd.DataFrame({
        "compartment_index": range(1, len(compartments) + 1), "compartment_post": compartments,
    })
    for scope, selected in contact_scopes(raw, threshold):
        counts = pair_counts(selected, bodies)
        matrix = counts.unstack("bodyId_post")
        # Matrix rows are sources, columns are receivers, as in the census.
        tables[f"{scope}_matrix.csv"] = matrix.reset_index()
        scope_rows.append({
            "scope": scope, "contacts": len(selected),
            "unique_pre_sites": len(selected[["bodyId_pre", "synapse_node_id_pre"]].drop_duplicates()),
            "unique_post_sites": len(selected[["bodyId_post", "synapse_node_id_post"]].drop_duplicates()),
            "all_reported_core_contacts": len(raw),
            "fraction_of_all_reported_core_contacts": len(selected) / len(raw) if len(raw) else np.nan,
            "missing_FB_pre_contacts": int(selected.FB_pre.isna().sum()),
            "missing_FB_post_contacts": int(selected.FB_post.isna().sum()),
            "missing_confidence_pre_contacts": int(selected.confidence_pre.isna().sum()),
            "missing_confidence_post_contacts": int(selected.confidence_post.isna().sum()),
        })
        unique_pre = selected.groupby(PAIR).synapse_node_id_pre.nunique()
        unique_post = selected.groupby(PAIR).synapse_node_id_post.nunique()
        for (pre, post), count in counts.items():
            denominator = int(all_pairs.loc[(pre, post)])
            pair_rows.append({
                "scope": scope, "bodyId_pre": pre, "bodyId_post": post,
                "annotation_column_pre": body_lookup.loc[pre, "annotation_column"],
                "annotation_column_post": body_lookup.loc[post, "annotation_column"],
                "contacts": int(count), "unique_pre_sites": int(unique_pre.get((pre, post), 0)),
                "unique_post_sites": int(unique_post.get((pre, post), 0)),
                "all_reported_body_pair_contacts": denominator,
                "fraction_of_all_reported_body_pair_contacts": count / denominator if denominator else np.nan,
            })
        for index, compartment in enumerate(compartments, 1):
            mask = selected.compartment_post.isna() if compartment is None else selected.compartment_post.eq(compartment)
            group = selected.loc[mask]
            cc = pair_counts(group, bodies)
            tables[f"{scope}_compartment_{index:02d}_matrix.csv"] = cc.unstack("bodyId_post").reset_index()
            compartment_rows.append({
                "scope": scope, "compartment_index": index, "compartment_post": compartment,
                "contacts": len(group), "scope_contacts": len(selected),
                "fraction_of_scope_contacts": len(group) / len(selected) if len(selected) else np.nan,
            })
    tables["body_pair_counts.csv"] = pd.DataFrame(pair_rows)
    tables["compartment_counts.csv"] = pd.DataFrame(compartment_rows)
    tables["scope_counts.csv"] = pd.DataFrame(scope_rows)
    for table in tables.values():
        table.columns = table.columns.map(str)
        table.columns.name = None
    return tables


def run_contact_audit(client, coordinates, census, output):
    bodies, points, edges, geometry, provenance = load_sources(coordinates, census)
    output = Path(output).resolve()
    protected = [Path(coordinates).resolve(), Path(census).resolve(), Path(geometry["source_probe"]),
                 Path(geometry["source_inventory"]["annotation_audit_directory"])] + [
        Path(source["source_manifest_file"]).parent for source in geometry["source_inventory"]["sources"].values()
    ]
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in protected):
        raise ValidationError("Contact output must be separate from every source directory.")
    if client.server.rstrip("/") != f"https://{SERVER}" or client.dataset != DATASET:
        raise ValidationError("Client target differs from the configured dataset.")
    threshold = json.loads((Path(coordinates) / "metadata_before.json").read_text())["postHighAccuracyThreshold"]
    queries = {"annotations": spatial_probe_queries(bodies.bodyId, full_population=True)["body_annotations"],
               "contacts": contact_query(bodies)}
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "kind": KIND, "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "server": SERVER, "dataset": DATASET, "sources": provenance, "queries": queries,
        "standard_confidence_threshold": threshold,
        "settings": {"raw_confidence_filter": None, "raw_roi_filter": None,
                     "standard_comparison": "Both endpoint confidence scores strictly greater than the recorded threshold.",
                     "FB_comparison": "post_FB follows receiving-site ROI; both_FB requires both endpoints.",
                     "identity": "Snapshot-local graph relationship/site IDs; one row per directed contact.",
                     "signs": "unassigned", "column_mapping": "unresolved", "normalization": None},
    }
    save_json(output / "contacts.json", manifest)
    try:
        manifest.update(pilot_code_provenance())
        before = fetch_meta(client=client)
        save_json(output / "metadata_before.json", before)
        check_probe_metadata(before, geometry["source_inventory"])
        bodies.to_csv(output / "selected_bodies.csv", index=False)
        live = client.fetch_custom(queries["annotations"])
        live.to_csv(output / "live_annotations.csv", index=False)
        columns = ["bodyId", "type", "instance"]
        pd.testing.assert_frame_equal(live[columns].sort_values("bodyId").reset_index(drop=True),
                                      bodies[columns].sort_values("bodyId").reset_index(drop=True), check_dtype=False)
        raw = client.fetch_custom(queries["contacts"])
        raw.to_csv(output / "contacts_raw.csv", index=False)
        validate_contacts(raw, bodies, points, edges, threshold)
        for name, table in contact_tables(raw, bodies, threshold).items():
            table.to_csv(output / name, index=False)
        after = fetch_meta(client=client)
        save_json(output / "metadata_after.json", after)
        check_probe_metadata(after, geometry["source_inventory"])
        if load_sources(coordinates, census)[4] != provenance:
            raise ValidationError("Contact sources changed during extraction.")
        scopes = pd.read_csv(output / "scope_counts.csv")
        compartments = pd.read_csv(output / "compartment_counts.csv")
        (output / "summary.md").write_text("\n".join([
            "# Directed vDeltaE → hDeltaG contact audit", "", markdown(scopes), "",
            "## Receiving compartments", "", markdown(compartments), "",
            "Every matrix retains all 18 source bodies and eight receivers, including zero-report pairs.",
            "C labels are annotations, not angles or shared physical columns. C0 bodies remain separate.",
            "Confidence and ROI scopes overlap: never add their totals together.",
            "A contact is a pre/post site pair; unique sites and contact counts differ for polyadic synapses.",
            "Raw confidence and ROI annotations remain unfiltered. Missing flags are not converted to false.",
            "Standard-confidence body-pair and FB-post counts reconcile to the saved census.",
            "Compartment labels do not establish synaptic sign, channel, physiological effect or integration.", "",
        ]))
        manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["artifact_sha256"] = record_artifact_hashes(output, "contacts.json")
        save_json(output / "contacts.json", manifest)
    return manifest


def verify_contact_audit(output):
    output = Path(output).resolve()
    record = json.loads((output / "contacts.json").read_text())
    if record.get("kind") != KIND or record.get("status") != "complete":
        raise ValidationError("Verification requires a complete directed contact audit.")
    sources = record["sources"]
    bodies, points, edges, geometry, current = load_sources(sources["coordinates"], sources["census"])
    if current != sources:
        raise ValidationError("Contact source provenance changed.")
    threshold = json.loads((Path(sources["coordinates"]) / "metadata_before.json").read_text())["postHighAccuracyThreshold"]
    if record["standard_confidence_threshold"] != threshold or record["queries"] != {
        "annotations": spatial_probe_queries(bodies.bodyId, full_population=True)["body_annotations"],
        "contacts": contact_query(bodies),
    }:
        raise ValidationError("Saved contact query or confidence convention differs from its sources.")
    raw = pd.read_csv(output / "contacts_raw.csv", low_memory=False)
    validate_contacts(raw, bodies, points, edges, threshold)
    tables = contact_tables(raw, bodies, threshold)
    required = {"contacts_raw.csv", "selected_bodies.csv", "live_annotations.csv", "metadata_before.json",
                "metadata_after.json", "summary.md", *tables}
    if set(record.get("artifact_sha256", {})) != required:
        raise ValidationError("Contact artifact inventory differs from expected outputs.")
    for name, digest in record["artifact_sha256"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or not path.is_file() or sha256(path) != digest:
            raise ValidationError("Contact artifact is missing, outside the run, or changed.")
    pd.testing.assert_frame_equal(bodies, pd.read_csv(output / "selected_bodies.csv"), check_dtype=False)
    columns = ["bodyId", "type", "instance"]
    live = pd.read_csv(output / "live_annotations.csv")
    pd.testing.assert_frame_equal(live[columns].sort_values("bodyId").reset_index(drop=True),
                                  bodies[columns].sort_values("bodyId").reset_index(drop=True), check_dtype=False)
    for name in ["metadata_before.json", "metadata_after.json"]:
        check_probe_metadata(json.loads((output / name).read_text()), geometry["source_inventory"])
    for name, expected in tables.items():
        pd.testing.assert_frame_equal(csv_comparable(expected), csv_comparable(pd.read_csv(output / name, low_memory=False)),
                                      check_dtype=False, rtol=1e-12, atol=1e-12)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract", help="Retrieve only audited vDeltaE → hDeltaG contacts.")
    extract.add_argument("--coordinates", required=True)
    extract.add_argument("--census", required=True)
    extract.add_argument("--output", required=True)
    verify = commands.add_parser("verify", help="Verify a saved contact audit offline.")
    verify.add_argument("--run", required=True)
    export = commands.add_parser("export", help="Render verified contact matrices and receiver overlays offline.")
    export.add_argument("--run", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--source-compartments", action="store_true",
                        help="Add source vDeltaE axon/dendrite heatmaps with one shared colorbar.")
    export.add_argument("--receiving-compartments", action="store_true",
                        help="Add receiving hDeltaG axon/dendrite heatmaps with one shared colorbar.")
    figures = commands.add_parser("verify-figures", help="Verify saved contact figures offline.")
    figures.add_argument("--run", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "extract":
            import os
            from dotenv import load_dotenv
            from neuprint import Client

            load_dotenv(Path.cwd() / ".env", override=False)
            token = os.environ.get("NEUPRINT_TOKEN", "").strip()
            if not token:
                raise ValidationError("Configure NEUPRINT_TOKEN locally.")
            client = Client(SERVER, dataset=DATASET, token=token)
            result = run_contact_audit(client, args.coordinates, args.census, args.output)
            verify_contact_audit(args.output)
        elif args.command == "verify":
            result = verify_contact_audit(args.run)
        else:
            from .vdeltae_contact_figures import export_contact_figures, verify_contact_figures

            if args.command == "export":
                result = export_contact_figures(args.run, args.output,
                    source_compartment_panels=args.source_compartments,
                    receiving_compartment_panels=args.receiving_compartments)
                verify_contact_figures(args.output)
            else:
                result = verify_contact_figures(args.run)
    except Exception as error:
        # HTTP exceptions can contain request details. Persist/print classes only.
        print(f"Contact {args.command} stopped ({type(error).__name__}); inspect local sources/manifests.")
        return 1
    print(f"Contact {args.command}: {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
