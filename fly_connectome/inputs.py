"""One-hop PFNd input census, including untyped segments and all input regions."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from neuprint import NeuronCriteria, fetch_meta, fetch_neurons

from .anatomy import annotation_order
from .connectivity import (
    ANNOTATION_COLUMNS, DATASET, SERVER, ValidationError, build_matrix,
    check_reference_annotations, check_reference_weight, check_type_inventory,
    fetch_annotations, fetch_connections, reference_result, type_inventory,
)


UNTYPED = "<untyped>"
PAIR = ["bodyId_pre", "bodyId_post"]
ANATOMY = ["pb_side", "pb_glomerulus", "fb_column"]
UPSTREAM = ["bodyId", "type", "instance", "status", "statusLabel", "cropped",
            "hemibrainType", "flywireType", "consensusNt", "labels"]
CAVEAT = (
    "PB side/glomerulus and FB column describe the receiving PFNd neuron. "
    "They do not locate a contact. Contact regions use edge roiInfo post counts. "
    "Column indices are not functional angles. Counts do not specify physiological efficacy."
)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def integers(values, name, minimum=0):
    numbers = pd.to_numeric(values, errors="raise")
    if not np.isfinite(numbers).all() or (numbers < minimum).any() or (numbers % 1 != 0).any():
        raise ValidationError(f"{name} must contain finite integers >= {minimum}.")
    return numbers.astype("int64")


def upstream_query(target_ids):
    """Target-bounded query; default API sources=None would restrict to Neuron nodes."""
    ids = sorted(int(body) for body in target_ids)
    return f"""MATCH (t:Neuron) WHERE t.bodyId IN {ids}
WITH t MATCH (s:Segment)-[e:ConnectsTo]->(t)
WHERE e.weight >= 1
RETURN s.bodyId AS bodyId_pre, t.bodyId AS bodyId_post, e.weight AS weight,
       s.type AS type, s.instance AS instance, s.status AS status,
       s.statusLabel AS statusLabel, s.cropped AS cropped,
       s.hemibrainType AS hemibrainType, s.flywireType AS flywireType,
       s.consensusNt AS consensusNt, labels(s) AS labels, e.roiInfo AS roiInfo
ORDER BY bodyId_pre, bodyId_post"""


def prepare_inputs(raw, targets, primary_rois):
    """Validate raw edges and unpack ROI counts without summing overlapping ROIs."""
    if not primary_rois or "NotPrimary" in primary_rois:
        raise ValidationError("Expected a nonempty primary ROI inventory without NotPrimary.")
    if targets.bodyId.duplicated().any():
        raise ValidationError("Duplicate PFNd target IDs.")
    integers(targets.bodyId, "Target IDs", 1)
    integers(targets.post, "Target postsynaptic counts")
    # Parsing validates exact PFNd type, instances and the inspected PB/FB mapping.
    anatomy = annotation_order(targets, "PFNd")
    raw = raw.copy()
    for column in PAIR:
        raw[column] = integers(raw[column], column, 1)
    raw["weight"] = integers(raw.weight, "Connection weights", 1)
    if raw.duplicated(PAIR).any() or not raw.bodyId_post.isin(targets.bodyId).all():
        raise ValidationError("Duplicate input pairs or targets outside the selected PFNd population.")
    if not raw.labels.map(lambda v: isinstance(v, list) and "Segment" in v).all():
        raise ValidationError("Upstream labels must identify Segment nodes.")
    raw["labels"] = raw.labels.map(lambda v: json.dumps(sorted(v)))
    upstream = raw.rename(columns={"bodyId_pre": "bodyId"})[UPSTREAM].drop_duplicates()
    if upstream.bodyId.duplicated().any():
        raise ValidationError("Inconsistent annotations for an upstream body.")
    if upstream.type.eq(UNTYPED).any():
        raise ValidationError("Type label conflicts with the reserved untyped display label.")
    typed = upstream.type.notna() & upstream.type.fillna("").str.strip().ne("")
    upstream["population"] = upstream.type.where(typed, UNTYPED)
    upstream["is_neuron"] = upstream.labels.map(lambda v: "Neuron" in json.loads(v))
    upstream["source_category"] = np.where(
        upstream.is_neuron, np.where(typed, "annotated_neuron", "untyped_neuron"),
        "non_neuron_segment",
    )
    upstream["status_flagged"] = upstream.status.isin(["Glia", "Orphan", "Unimportant"]) | upstream.statusLabel.isin(["Glia", "Orphan", "Unimportant"])
    upstream = upstream.sort_values("bodyId").reset_index(drop=True)
    edges = raw[[*PAIR, "weight", "roiInfo"]].sort_values(PAIR).reset_index(drop=True)
    roi_rows = []
    for edge in edges.itertuples(index=False):
        try:
            info = json.loads(edge.roiInfo)
            if not isinstance(info, dict):
                raise ValueError("ROI object is not a dictionary")
            primary_total = 0
            for roi, counts in sorted(info.items()):
                count = int(integers(pd.Series([counts.get("post", 0)]), "ROI post counts").iloc[0])
                if count > edge.weight:
                    raise ValidationError("ROI count exceeds the whole connection weight.")
                if count:
                    primary = roi in primary_rois
                    roi_rows.append([edge.bodyId_pre, edge.bodyId_post, roi, count, primary])
                    primary_total += count if primary else 0
            if primary_total > edge.weight:
                raise ValidationError("Primary ROI counts overlap or exceed the whole connection weight.")
            if primary_total < edge.weight:
                roi_rows.append([edge.bodyId_pre, edge.bodyId_post, "NotPrimary",
                                 edge.weight - primary_total, True])
        except (TypeError, ValueError, AttributeError) as exc:
            if isinstance(exc, ValidationError):
                raise
            raise ValidationError("Invalid connection roiInfo; unavailable ROI data is not zero.") from exc
    rois = pd.DataFrame(roi_rows, columns=[*PAIR, "roi", "weight", "additive"])
    if not rois.empty:
        summed = rois[rois.additive].groupby(PAIR).weight.sum()
        expected = edges.set_index(PAIR).weight
        pd.testing.assert_series_equal(summed.sort_index(), expected.sort_index(), check_names=False)
    return upstream, edges, rois, anatomy


def summarize_inputs(upstream, edges, rois, targets, anatomy):
    """Keep all 40 targets in distributions, including zero-input targets."""
    ids = sorted(targets.bodyId.tolist())
    enriched = edges.merge(upstream, left_on="bodyId_pre", right_on="bodyId", validate="many_to_one")
    populations = sorted(upstream.population.unique())
    totals = enriched.groupby("bodyId_post").weight.sum().reindex(ids, fill_value=0)
    denominators = targets.set_index("bodyId")[["post"]].reindex(ids).rename(columns={"post": "post_sites"})
    denominators["all_segment_weight"] = totals
    denominators["post_minus_edge_residual"] = denominators.post_sites - totals
    for category in ["annotated_neuron", "untyped_neuron", "non_neuron_segment"]:
        denominators[f"{category}_weight"] = enriched[enriched.source_category == category].groupby("bodyId_post").weight.sum().reindex(ids, fill_value=0)
    denominators["flagged_weight"] = enriched[enriched.status_flagged].groupby("bodyId_post").weight.sum().reindex(ids, fill_value=0)
    denominators["untyped_weight"] = enriched[enriched.population == UNTYPED].groupby("bodyId_post").weight.sum().reindex(ids, fill_value=0)
    index = pd.MultiIndex.from_product([populations, ids], names=["population", "bodyId_post"])
    per_target = enriched.groupby(["population", "bodyId_post"]).weight.sum().reindex(index, fill_value=0).rename("weight").reset_index()
    per_target["all_segment_denominator"] = per_target.bodyId_post.map(totals)
    per_target["input_fraction"] = per_target.weight / per_target.all_segment_denominator.replace(0, np.nan)
    per_target = per_target.merge(anatomy[["bodyId", "instance", *ANATOMY]], left_on="bodyId_post", right_on="bodyId", validate="many_to_one").drop(columns="bodyId")
    rows = []
    total = int(totals.sum())
    for pop, group in per_target.groupby("population", sort=True):
        sources = upstream[upstream.population == pop]
        rows.append({
            "population": pop, "source_bodies": len(sources), "weight": int(group.weight.sum()),
            "share": group.weight.sum() / total if total else np.nan,
            "covered_targets": int(group.weight.gt(0).sum()), "target_count": len(ids),
            "coverage": float(group.weight.gt(0).mean()),
            "weight_min": int(group.weight.min()), "weight_median": float(group.weight.median()),
            "weight_mean": float(group.weight.mean()), "weight_max": int(group.weight.max()),
            "fraction_min": group.input_fraction.min(), "fraction_median": group.input_fraction.median(),
            "fraction_max": group.input_fraction.max(),
            "targets_with_defined_fraction": int(group.input_fraction.notna().sum()),
        })
    summary_columns = ["population", "source_bodies", "weight", "share", "covered_targets", "target_count", "coverage", "weight_min", "weight_median", "weight_mean", "weight_max", "fraction_min", "fraction_median", "fraction_max", "targets_with_defined_fraction"]
    summary = pd.DataFrame(rows, columns=summary_columns).sort_values(["weight", "population"], ascending=[False, True])
    grouped = per_target.groupby(["population", *ANATOMY], as_index=False).agg(
        weight=("weight", "sum"), target_neurons=("bodyId_post", "size"),
        covered_targets=("weight", lambda x: int(x.gt(0).sum())),
        all_segment_denominator=("all_segment_denominator", "sum"),
    )
    grouped["weight_per_target"] = grouped.weight / grouped.target_neurons
    grouped["input_fraction"] = grouped.weight / grouped.all_segment_denominator.replace(0, np.nan)
    roi_enriched = rois.merge(upstream[["bodyId", "population"]], left_on="bodyId_pre", right_on="bodyId", validate="many_to_one")
    roi_summary = roi_enriched.groupby(["population", "roi", "additive"], as_index=False).weight.sum()
    roi_anatomy = roi_enriched.merge(anatomy[["bodyId", *ANATOMY]], left_on="bodyId_post", right_on="bodyId", validate="many_to_one")
    roi_groups = roi_anatomy.groupby(["population", *ANATOMY, "roi", "additive"], as_index=False).weight.sum()
    reconstruction = []
    for column in ["source_category", "status", "statusLabel", "cropped", "status_flagged"]:
        for value, group in enriched.groupby(column, dropna=False):
            reconstruction.append({"field": column, "value": "unknown" if pd.isna(value) else str(value),
                                   "source_bodies": group.bodyId_pre.nunique(), "weight": int(group.weight.sum())})
    if not (int(summary.weight.sum()) == total == int(grouped.weight.sum())
            == int(roi_summary.loc[roi_summary.additive, "weight"].sum())):
        raise ValidationError("Population, anatomical or additive ROI aggregation changed total weight.")
    return {
        "input_denominators.csv": denominators.rename_axis("bodyId").reset_index(),
        "population_inputs.csv": summary, "population_by_target.csv": per_target,
        "anatomical_group_inputs.csv": grouped, "population_roi_inputs.csv": roi_summary,
        "anatomical_roi_inputs.csv": roi_groups,
        "reconstruction_summary.csv": pd.DataFrame(reconstruction, columns=["field", "value", "source_bodies", "weight"]),
    }


def render_inputs(source, output, *, show_fraction_caveat=False):
    """Render saved tables only. Plot selection never changes saved census membership."""
    source, output = Path(source), Path(output)
    summary = pd.read_csv(source / "population_inputs.csv").head(20)
    per_target = pd.read_csv(source / "population_by_target.csv")
    order = pd.read_csv(source / "PFNd_anatomical_order.csv")
    roi = pd.read_csv(source / "population_roi_inputs.csv")
    if summary.empty:
        return
    names = summary.population.tolist()
    plots = []
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(names[::-1], summary.weight.to_numpy()[::-1])
    ax.set_xlabel("Incoming synaptic contacts (raw standard weight)")
    ax.set_title("Leading PFNd inputs by total weight; all populations saved in CSV")
    plots.append((fig, "population_totals"))
    matrix = per_target.pivot(index="population", columns="bodyId_post", values="input_fraction").reindex(index=names, columns=order.bodyId)
    fig, ax = plt.subplots(figsize=(15, 8))
    im = ax.imshow(np.ma.masked_invalid(matrix.to_numpy(dtype=float) * 100), aspect="auto", vmin=0)
    ax.set_yticks(range(len(names)), names)
    ax.set_xticks(range(len(order)), order.label, rotation=90, fontsize=6)
    ax.set_xlabel("Receiving PFNd neuron, ordered by FB annotation then PB side; contact location is separate")
    ax.set_title("Fraction of each PFNd neuron's all-segment input; grey cells undefined")
    im.cmap = im.cmap.copy()
    im.cmap.set_bad("#cccccc")
    fig.colorbar(im, ax=ax, label="% of target's incoming contacts")
    plots.append((fig, "population_fractions"))
    fig, ax = plt.subplots(figsize=(11, 7))
    values = [per_target.loc[per_target.population == name, "input_fraction"].dropna().to_numpy() * 100 for name in names[::-1]]
    ax.boxplot(values, tick_labels=names[::-1], orientation="horizontal", showfliers=True)
    ax.set_xlabel("% of individual PFNd neuron's all-segment input (zeros included)")
    ax.set_title("Distribution across receiving PFNd neurons")
    plots.append((fig, "input_distributions"))
    additive = roi[roi.additive].copy()
    additive["region"] = additive.roi.where(additive.roi.isin(["PB", "NO", "FB"]), "Other / unspecified")
    region = additive.groupby(["population", "region"]).weight.sum().unstack(fill_value=0).reindex(names, fill_value=0)
    region = region.reindex(columns=["PB", "NO", "FB", "Other / unspecified"], fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 7))
    region.iloc[::-1].plot.barh(stacked=True, ax=ax)
    ax.set_xlabel("Incoming contacts, partitioned by primary ROI at receiving endpoint")
    ax.set_title("Contact region is distinct from the receiving neuron's column annotation")
    plots.append((fig, "input_regions"))
    try:
        for fig, name in plots:
            show_caveat = name != "population_fractions" or show_fraction_caveat
            if show_caveat:
                fig.text(.5, .005, "Anatomical counts do not specify physiological efficacy. No functional angles assigned.", ha="center", fontsize=8)
            fig.tight_layout(rect=(0, .025 if show_caveat else 0, 1, 1))
            for extension in ["png", "pdf"]:
                fig.savefig(output / f"{name}.{extension}", dpi=160)
    finally:
        for fig, _ in plots:
            plt.close(fig)


def verify_input_data(source, manifest):
    source = Path(source).resolve()
    if not manifest.get("data_validated") or not manifest.get("data_sha256"):
        raise ValidationError("Offline plotting requires validated input census data.")
    for name, digest in manifest["data_sha256"].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or not path.is_file() or sha256(path) != digest:
            raise ValidationError(f"Input census artifact missing, outside run, or changed: {name}")


def replot_inputs(source, output, code, *, show_fraction_caveat=False):
    source, output = Path(source), Path(output)
    manifest = json.loads((source / "inputs.json").read_text())
    verify_input_data(source, manifest)
    output.mkdir(parents=True, exist_ok=False)
    render_inputs(source, output, show_fraction_caveat=show_fraction_caveat)
    verify_input_data(source, manifest)
    save_json(output / "replot.json", {"status": "complete", "source": str(source.resolve()),
                                      "data_sha256": manifest["data_sha256"], "code": code,
                                      "plot_settings": {"show_fraction_caveat": show_fraction_caveat}})


def run_inputs(client, source, output, code, *, show_fraction_caveat=False):
    """Extract to a new directory; preserve raw diagnostics and safe failure metadata."""
    source, output = Path(source), Path(output)
    paths = [source / name for name in ["metadata.json", "PFNd_neurons.csv"]]
    source_hashes = {p.name: sha256(p) for p in paths}
    baseline = json.loads(paths[0].read_text())
    if baseline.get("status") != "complete" or baseline.get("dataset") != DATASET or baseline.get("server") != SERVER:
        raise ValidationError("Inputs require a complete baseline extraction from the configured server/dataset.")
    saved = pd.read_csv(paths[1])
    annotation_order(saved, "PFNd")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "server": SERVER, "dataset": DATASET, "code": code,
        "code_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ["neuprint-python", "pandas", "numpy", "matplotlib", "python-dotenv"]},
        "source_directory": str(source.resolve()), "source_metadata": baseline,
        "source_sha256": source_hashes, "caveat": CAVEAT,
        "plot_settings": {"show_fraction_caveat": show_fraction_caveat},
        "query_settings": {"target_type": "PFNd", "exact_type": True, "source_label": "Segment",
                           "status_filter": None, "cropped_filter": None, "roi_filter": None,
                           "min_weight": 1, "weight_property": "weight", "hops": 1,
                           "self_connections": "included", "recurrent_connections": "included"},
        "definitions": {
            "share": "Population weight / all incoming positive Segment edge weights to the complete PFNd population.",
            "input_fraction": "Population weight to target / all incoming positive Segment edge weights to that target.",
            "coverage": "Targets with positive population input / all selected PFNd targets.",
            "zero_denominator": "Fractions undefined (blank CSV cells); zero-weight targets remain in coverage and weight distributions.",
            "post_minus_edge_residual": "Recorded target post sites minus summed incoming Segment edge weights; required to equal zero for completion.",
            "additive": "Primary ROI post counts plus explicit NotPrimary residual. Other ROI rows overlap; never sum them into totals.",
            "flagged_weight": "Status/statusLabel Glia, Orphan or Unimportant; overlaps other annotation categories.",
            "anatomical_groups": "Joint PB side, PB glomerulus and FB column of the receiving neuron; not physical contact columns.",
            "roi_tables": "Positive ROI counts only (sparse); no coordinates, compartment inference or morphology fetched.",
            "zero": "No connection reported in this dataset, not proof of biological absence.",
        },
    }
    save_json(output / "inputs.json", manifest)
    try:
        manifest["dataset_metadata"] = fetch_meta(client=client)
        primary_rois = sorted(client.primary_rois)
        manifest["primary_rois"] = primary_rois
        inventory = type_inventory(client)
        inventory.to_csv(output / "type_inventory.csv", index=False)
        check_type_inventory(inventory)
        targets = fetch_neurons(NeuronCriteria(type="PFNd", regex=False, client=client),
                                returned_columns=[*ANNOTATION_COLUMNS, "post"], omit_rois=True, client=client)
        targets = targets.sort_values("bodyId").reset_index(drop=True)
        pd.testing.assert_frame_equal(targets[["bodyId", "type", "instance"]],
                                      saved[["bodyId", "type", "instance"]].sort_values("bodyId").reset_index(drop=True), check_dtype=False)
        targets.to_csv(output / "PFNd_neurons.csv", index=False)
        reference = fetch_annotations(client, [26913, 515192])
        reference.to_csv(output / "reference_neurons.csv", index=False)
        check_reference_annotations(reference)
        ref_edges = fetch_connections(client, [26913], [515192])
        ref_edges.to_csv(output / "reference_connections.csv", index=False)
        ref = reference_result(build_matrix(ref_edges, [26913], [515192]))
        manifest["reference_validation"] = ref
        check_reference_weight(ref)
        query = upstream_query(targets.bodyId)
        manifest["upstream_query"] = query
        save_json(output / "inputs.json", manifest)
        raw = client.fetch_custom(query)
        # Store diagnostic raw data before any downstream validation.
        raw.assign(labels=raw.labels.map(json.dumps)).to_csv(output / "raw_connections.csv", index=False)
        upstream, edges, rois, anatomy = prepare_inputs(raw, targets, primary_rois)
        tables = {"upstream_neurons.csv": upstream, "incoming_connections.csv": edges,
                  "connection_roi_counts.csv": rois, "PFNd_anatomical_order.csv": anatomy,
                  **summarize_inputs(upstream, edges, rois, targets, anatomy)}
        for name, table in tables.items():
            table.to_csv(output / name, index=False)
        denominators = tables["input_denominators.csv"]
        if denominators.post_minus_edge_residual.ne(0).any():
            raise ValidationError("Incoming Segment weights do not reconcile with recorded PFNd post counts; inspect input_denominators.csv.")
        if any(sha256(p) != source_hashes[p.name] for p in paths):
            raise ValidationError("Baseline source files changed during extraction.")
        manifest["validation"] = {
            "targets": len(targets), "expected_40_sanity_check": len(targets) == 40,
            "upstream_bodies": len(upstream), "positive_pairs": len(edges),
            "total_weight": int(edges.weight.sum()), "post_counts_reconcile": True,
            "primary_rois_with_residual_conserve_weights": True,
            "not_primary_weight": int(rois.loc[rois.roi == "NotPrimary", "weight"].sum()),
            "all_target_axes_retained": True, "saved_annotations_match": True,
        }
        manifest["data_sha256"] = {p.name: sha256(p) for p in sorted(output.glob("*.csv"))}
        manifest["data_validated"] = True
        save_json(output / "inputs.json", manifest)
        render_inputs(output, output, show_fraction_caveat=show_fraction_caveat)
        verify_input_data(output, manifest)
        manifest["artifact_sha256"] = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.suffix in [".csv", ".png", ".pdf"]}
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="validation_failed" if isinstance(exc, (ValidationError, AssertionError)) else "failed",
                        error_type=type(exc).__name__)
        # Never serialize arbitrary server/HTTP exception text (may contain request data).
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(output / "inputs.json", manifest)
    return manifest
