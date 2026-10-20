"""Offline PFN → hDeltaB summaries in type-specific annotated FB columns."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from .anatomy import COLUMN_CAVEAT, annotation_order
from .connectivity import ValidationError, ZERO_MEANING, build_matrix
from .plotting import load_matrix, order_matrix, plot_heatmap


def aggregate_columns(matrix, source_order, target_order, *, source_type="PFNd"):
    """Include every possible neuron pair, including unconnected neurons."""
    if source_type not in ["PFNd", "PFNv"]:
        raise ValidationError("Column summaries support PFNd or PFNv sources.")
    source_count, source_columns = (40, range(1, 9)) if source_type == "PFNd" else (20, range(1, 10))
    source_key = f"{source_type.lower()}_column"
    for table, axis, count, columns in [
        (source_order, matrix.index, source_count, source_columns),
        (target_order, matrix.columns, 19, range(1, 13)),
    ]:
        if (len(table) != count or table.bodyId.isna().any()
                or table.bodyId.duplicated().any() or not axis.is_unique
                or len(axis) != count or set(table.bodyId) != set(axis)
                or table.fb_column.isna().any() or set(table.fb_column) != set(columns)):
            raise ValidationError(f"Expected complete unique {source_count}/19 populations and all {len(source_columns)}/12 columns.")
    values = matrix.to_numpy()
    if matrix.isna().any().any() or (values < 0).any() or (values % 1 != 0).any():
        raise ValidationError("Matrix weights must be nonnegative integer synapse counts.")
    rows = []
    for pre in source_columns:
        sources = source_order.loc[source_order.fb_column == pre, "bodyId"]
        for post in range(1, 13):
            targets = target_order.loc[target_order.fb_column == post, "bodyId"]
            block = matrix.loc[sources, targets].to_numpy()
            total, possible = int(block.sum()), len(sources) * len(targets)
            connected = int((block > 0).sum())
            rows.append({
                source_key: pre, "hdeltab_column": post,
                "total_weight": total, "mean_weight_per_possible_pair": total / possible,
                "source_neurons": len(sources), "target_neurons": len(targets),
                "possible_pairs": possible, "connected_pairs": connected,
                "connection_density": connected / possible,
            })
    summary = pd.DataFrame(rows)
    for metric, rank in [("total_weight", "total_rank"),
                         ("mean_weight_per_possible_pair", "mean_rank")]:
        summary[rank] = summary.groupby(source_key)[metric].rank(
            method="min", ascending=False).astype(int)
    if (int(summary.total_weight.sum()) != int(values.sum())
            or int(summary.possible_pairs.sum()) != values.size
            or int(summary.connected_pairs.sum()) != int((values > 0).sum())):
        raise ValidationError("Column aggregation failed conservation checks.")
    return summary


def ranked_table(summary, *, source_type="PFNd"):
    """Top three positions by each metric, retaining ties at the cutoff."""
    caveat = COLUMN_CAVEAT if source_type == "PFNd" else (
        "PFNv C1–C9 and hDeltaB C1–C12 are within-type annotation indices; "
        "they have not been established as directly comparable spatial coordinates.")
    lines = [f"# {source_type} → hDeltaB strongest annotated column partners", "", caveat, "",
             "Anatomical connectivity summary; no directional transformation is inferred.", "",
             "Top three positions per metric; ties share competition rank and are included at the cutoff.",
             "Entries show hDeltaB column, rank, total synapses, and mean synapses per possible pair.", "",
             f"| {source_type} column | By total weight | By mean weight per possible pair |",
             "| --- | --- | --- |"]
    for pre, group in summary.groupby(f"{source_type.lower()}_column", sort=True):
        cells = []
        for rank in ["total_rank", "mean_rank"]:
            top = group.loc[group[rank] <= 3].sort_values([rank, "hdeltab_column"])
            cells.append("; ".join(
                f"C{r.hdeltab_column} (#{getattr(r, rank)}; {r.total_weight}; "
                f"{r.mean_weight_per_possible_pair:.3f})" for r in top.itertuples()))
        lines.append(f"| C{pre} | {cells[0]} | {cells[1]} |")
    return "\n".join(lines) + "\n"


def column_analysis(source, anatomical, output, *, show_column_caveat=False):
    """Validate saved inputs, then write a new set of column-level artifacts."""
    source, anatomical, output = map(lambda p: Path(p).resolve(), [source, anatomical, output])
    # Snapshot both directions and ordering tables before reading any inputs.
    inputs = sorted({p for directory in [source, anatomical] for p in directory.iterdir()
                     if p.is_file() and p.suffix in {".csv", ".json"}})
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    metadata = json.loads((source / "metadata.json").read_text())
    ordering_metadata = json.loads((anatomical / "anatomical_ordering.json").read_text())
    if metadata.get("status") != "complete" or ordering_metadata.get("status") != "complete":
        raise ValidationError("Column analysis requires complete extraction and anatomical ordering.")
    tables = {}
    for typ in ["PFNd", "hDeltaB"]:
        expected = annotation_order(pd.read_csv(source / f"{typ}_neurons.csv"), typ)
        tables[typ] = pd.read_csv(anatomical / f"{typ}_anatomical_order.csv")
        pd.testing.assert_frame_equal(tables[typ].convert_dtypes(), expected.convert_dtypes(),
                                      check_dtype=False)
    stem = "PFNd_to_hDeltaB"
    matrix = load_matrix(source / f"{stem}_matrix.csv")
    expected = build_matrix(pd.read_csv(source / f"{stem}_connections.csv"),
                            tables["PFNd"].bodyId, tables["hDeltaB"].bodyId)
    pd.testing.assert_frame_equal(matrix, expected, check_names=False)
    pd.testing.assert_frame_equal(
        load_matrix(anatomical / f"{stem}_anatomical_matrix.csv"),
        order_matrix(matrix, tables["PFNd"].bodyId, tables["hDeltaB"].bodyId))
    summary = aggregate_columns(matrix, tables["PFNd"], tables["hDeltaB"])
    manifest = {
        "status": "running", "source_directory": str(source),
        "anatomical_directory": str(anatomical), "input_sha256": hashes,
        "dataset": metadata.get("dataset"), "column_caveat": COLUMN_CAVEAT,
        "show_column_caveat": show_column_caveat,
        "zero_meaning": ZERO_MEANING, "ordering": "Numeric PFNd C1–C8; hDeltaB C1–C12",
        "formulas": {"total_weight": "Sum of all directed neuron-pair weights in the block",
                     "possible_pairs": "source_neurons * target_neurons",
                     "mean_weight_per_possible_pair": "total_weight / possible_pairs (including zeros)",
                     "connected_pairs": "Number of neuron pairs with weight > 0",
                     "connection_density": "connected_pairs / possible_pairs"},
        "ranking": "Descending within PFNd column; competition rank; numerical display order for ties",
        "validation": {"pfnd_neurons": len(tables["PFNd"]),
                       "hdeltab_neurons": len(tables["hDeltaB"]),
                       "each_neuron_represented_exactly_once": True,
                       "ordering_matches_annotations": True, "saved_edges_match_matrix": True,
                       "anatomical_matrix_matches_original": True,
                       "original_total_weight": int(matrix.to_numpy().sum()),
                       "aggregate_total_weight": int(summary.total_weight.sum()),
                       "possible_pairs": int(summary.possible_pairs.sum()),
                       "connected_pairs": int(summary.connected_pairs.sum())},
    }
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "column_aggregation.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    summary.to_csv(output / f"{stem}_column_summary.csv", index=False)
    (output / f"{stem}_column_rankings.md").write_text(ranked_table(summary))
    for metric, suffix, label in [
        ("total_weight", "total", "Total synapse count"),
        ("mean_weight_per_possible_pair", "mean", "Mean synapse count per possible neuron pair"),
    ]:
        grouped = summary.pivot(index="pfnd_column", columns="hdeltab_column", values=metric)
        prefix = output / f"{stem}_column_{suffix}"
        grouped.to_csv(str(prefix) + "_matrix.csv")
        plot_heatmap(
            grouped, prefix, "PFNd", "hDeltaB",
            row_labels={i: f"C{i}" for i in grouped.index},
            column_labels={i: f"C{i}" for i in grouped.columns},
            xlabel="Postsynaptic hDeltaB annotated FB column",
            ylabel="Presynaptic PFNd annotated FB column",
            title=f"PFNd → hDeltaB | {label}", colorbar_label=label,
            vmax=float(grouped.to_numpy().max()) or 1,
            note=("Column indices are type-specific; shared spatial alignment is unverified."
                  if show_column_caveat else None),
        )
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p)] for p in inputs):
        raise ValidationError("Source matrices or ordering inputs changed during column analysis.")
    manifest["validation"]["source_files_unchanged"] = True
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def pfnv_column_analysis(source, output, *, show_column_caveat=False):
    """Use validated expanded extraction and its literal PFNv column annotations."""
    from .pfn import load_pfn_network, MAPPING_NOTE
    from .rate_model import save_json, sha256

    source, output = Path(source).resolve(), Path(output).resolve()
    combined, populations, orders, metadata, hashes = load_pfn_network(source)
    matrix = combined.loc[sorted(populations["PFNv"].bodyId)]
    summary = aggregate_columns(matrix, orders["PFNv"], orders["hDeltaB"], source_type="PFNv")
    manifest = {
        "status": "running", "source_directory": str(source), "input_sha256": hashes,
        "dataset": metadata["dataset"], "mapping_note": MAPPING_NOTE,
        "show_column_caveat": show_column_caveat, "zero_meaning": ZERO_MEANING,
        "ordering": "Numeric PFNv C1–C9; hDeltaB C1–C12",
        "formulas": {"total_weight": "Sum of all directed neuron-pair weights in the block",
                     "mean_weight_per_possible_pair": "total_weight / (source_neurons * target_neurons), including zeros",
                     "connection_density": "connected_pairs / possible_pairs"},
        "ranking": "Descending within PFNv column; competition rank; numerical display order for ties",
        "validation": {"pfnv_neurons": len(populations["PFNv"]), "hdeltab_neurons": len(populations["hDeltaB"]),
                       "each_neuron_represented_exactly_once": True, "ordering_matches_annotations": True,
                       "saved_edges_match_matrix": True,
                       "original_total_weight": int(matrix.to_numpy().sum()),
                       "aggregate_total_weight": int(summary.total_weight.sum()),
                       "possible_pairs": int(summary.possible_pairs.sum()),
                       "connected_pairs": int(summary.connected_pairs.sum())},
    }
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "column_aggregation.json", manifest)
    stem = "PFNv_to_hDeltaB"
    summary.to_csv(output / f"{stem}_column_summary.csv", index=False)
    (output / f"{stem}_column_rankings.md").write_text(ranked_table(summary, source_type="PFNv"))
    for metric, suffix, label in [
        ("total_weight", "total", "Total synapse count"),
        ("mean_weight_per_possible_pair", "mean", "Mean synapse count per possible neuron pair"),
    ]:
        grouped = summary.pivot(index="pfnv_column", columns="hdeltab_column", values=metric)
        prefix = output / f"{stem}_column_{suffix}"
        grouped.to_csv(str(prefix) + "_matrix.csv")
        plot_heatmap(grouped, prefix, "PFNv", "hDeltaB",
                     row_labels={i: f"C{i}" for i in grouped.index},
                     column_labels={i: f"C{i}" for i in grouped.columns},
                     xlabel="Postsynaptic hDeltaB annotated FB column",
                     ylabel="Presynaptic PFNv annotated FB column",
                     title=f"PFNv → hDeltaB | {label}", colorbar_label=label,
                     vmax=float(grouped.to_numpy().max()) or 1,
                     note=("Column indices are type-specific; shared spatial alignment is unverified."
                           if show_column_caveat else None))
    if any(sha256(source / name) != digest for name, digest in hashes.items()):
        raise ValidationError("Source inputs changed during PFNv column analysis.")
    manifest["validation"]["source_files_unchanged"] = True
    manifest["status"] = "complete"
    save_json(output / "column_aggregation.json", manifest)
    return manifest
