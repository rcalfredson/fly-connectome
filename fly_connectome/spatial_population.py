"""All reported PFNd -> hDeltaB contacts, organized by annotated FB columns."""

from datetime import datetime, timezone
import html
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from neuprint import SynapseCriteria, fetch_meta, fetch_neurons, fetch_synapse_connections

from .anatomy import COLUMN_CAVEAT
from .connectivity import ANNOTATION_COLUMNS, DATASET, SERVER, ValidationError
from .spatial import (
    LAYERS, XYZ, UM_PER_VOXEL, REFERENCE_COUNTS, load_source, hashes, save_json,
    fetch_fb_synapses, prepare_contacts, validate_points, spatial_summary,
    contact_summary, parse_obj, plot_pilot,
)

COMPARTMENTS = ("axon", "dendrite", "linker", "unknown", "cell-body-fiber")
COLORS = {"axon": "#d97706", "dendrite": "#0891b2", "linker": "#6b7280",
          "unknown": "#a855f7", "cell-body-fiber": "#db2777"}
GALLERY_TABLES = ("column_coverage.csv", "contact_arbor_summary.csv", "arbor_spatial_summary.csv")


def verify_run_data(run, metadata):
    """Check files in this run, including legacy manifests containing old paths.

    Legacy population manifests list all data files, including root-level CSVs;
    their common parent identifies the recorded run root without opening it.
    New manifests explicitly store paths relative to the run directory.
    """
    run = Path(run).resolve()
    recorded = metadata.get("data_sha256", {})
    if not recorded or "validation" not in metadata:
        raise ValidationError("Replotting requires fully extracted and validated population data.")
    try:
        base = (Path(".") if metadata.get("data_sha256_base") == "." else
                Path(os.path.commonpath([str(Path(name).parent) for name in recorded])))
        local = {}
        for name, expected in recorded.items():
            relative = Path(name).relative_to(base)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Data path escapes run directory")
            path = (run / relative).resolve()
            path.relative_to(run)
            if str(path) in local:
                raise ValueError("Duplicate mapped data path")
            if not path.is_file() or hashes([path])[str(path)] != expected:
                raise ValidationError(f"Population spatial data missing or changed: {relative}.")
            local[str(path)] = expected
        return local
    except ValidationError:
        raise
    except (ValueError, TypeError) as exc:
        raise ValidationError("Invalid population data paths in manifest.") from exc


def annotate_contacts(contacts, annotations):
    contacts = contacts.copy()
    columns = annotations.set_index("bodyId").fb_column
    for side in ("pre", "post"):
        contacts[f"column_{side}"] = contacts[f"bodyId_{side}"].map(columns)
        if contacts[f"column_{side}"].isna().any():
            raise ValidationError("Unannotated contact body.")
    return contacts


def column_coverage(contacts, annotations):
    """All 96 combinations, including explicit zero-report cells; no thresholding."""
    rows = []
    for pre in range(1, 9):
        for post in range(1, 13):
            group = contacts[(contacts.column_pre == pre) & (contacts.column_post == post)]
            fb = group[group.FB_post]
            npre = int(((annotations.type == "PFNd") & (annotations.fb_column == pre)).sum())
            npost = int(((annotations.type == "hDeltaB") & (annotations.fb_column == post)).sum())
            row = {"column_pre": pre, "column_post": post, "source_neurons": npre,
                   "target_neurons": npost, "possible_pairs": npre * npost,
                   "connected_pairs": len(group[["bodyId_pre", "bodyId_post"]].drop_duplicates()),
                   "total_contacts": len(group), "fb_post_contacts": len(fb),
                   "both_fb_contacts": int(group.both_in_FB.sum())}
            for compartment in COMPARTMENTS:
                row[f"fb_post_{compartment}_contacts"] = int((fb.compartment_post == compartment).sum())
            if not group.empty:
                ranked = (group.groupby(["bodyId_pre", "bodyId_post"]).size().rename("weight").reset_index()
                          .sort_values(["weight", "bodyId_pre", "bodyId_post"], ascending=[False, True, True]))
                first = ranked.iloc[0]
                row.update(representative_pre=int(first.bodyId_pre), representative_post=int(first.bodyId_post),
                           representative_weight=int(first.weight))
            rows.append(row)
    coverage = pd.DataFrame(rows)
    for col in ("representative_pre", "representative_post", "representative_weight"):
        coverage[col] = coverage[col].astype("Int64")
    return coverage


def arbor_contacts(contacts):
    """Receiving arbor contact counts/positions; FB-only and all-ROI kept separate."""
    rows = []
    keys = ["bodyId_pre", "bodyId_post", "column_pre", "column_post", "compartment_post"]
    for scope, points in (("total", contacts), ("post_in_FB", contacts[contacts.FB_post])):
        for values, group in points.groupby(keys, observed=True):
            row = dict(zip(keys, values))
            row.update(scope=scope, contacts=len(group))
            for axis in XYZ:
                row[f"{axis}_mean_um"] = float(group[f"{axis}_post"].mean() * UM_PER_VOXEL)
                row[f"{axis}_median_um"] = float(group[f"{axis}_post"].median() * UM_PER_VOXEL)
            rows.append(row)
    return pd.DataFrame(rows)


def validate_population(contacts, synapses, roi_counts, matrix):
    actual = contacts.groupby(["bodyId_pre", "bodyId_post"]).size()
    expected = matrix.stack()
    expected = expected[expected > 0]
    if actual.to_dict() != expected.to_dict():
        raise ValidationError("Live contacts differ from saved positive pairs/weights.")
    validate_points(synapses, roi_counts, contacts)
    counts = contact_summary(contacts)
    ref = counts[(counts.bodyId_pre == 26913) & (counts.bodyId_post == 515192)]
    observed = ref.set_index("scope").contacts.astype(int).to_dict()
    if observed != REFERENCE_COUNTS:
        raise ValidationError("Reference spatial counts changed.")
    return counts


def compatible_cache(cache, meta):
    """Only reuse a completed pilot from the identical dataset metadata snapshot."""
    if cache is None:
        return None
    cache = Path(cache).resolve()
    manifest = json.loads((cache / "spatial_pilot.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("dataset") != DATASET or manifest.get("server") != SERVER:
        raise ValidationError("Cache must be a completed pilot from the configured dataset.")
    old = manifest["dataset_metadata"]
    for key in ("uuid", "lastDatabaseEdit", "latestMutationId", "postHighAccuracyThreshold", "voxelSize", "voxelUnits"):
        if old.get(key) != meta.get(key):
            raise ValidationError(f"Pilot cache dataset metadata differs ({key}); omit --cache to fetch fresh data.")
    return cache, manifest


def cache_file(cache_info, relative):
    if cache_info is None:
        return None
    cache, manifest = cache_info
    path = cache / relative
    if not path.is_file():
        return None
    expected = next((digest for key, digest in manifest.get("artifact_sha256", {}).items()
                     if Path(key).resolve() == path), None)
    if expected is None or hashes([path])[str(path)] != expected:
        raise ValidationError(f"Pilot cache artifact not verified: {relative}.")
    return path


def run_population(client, source, output, code, cache=None):
    annotations, matrix, source_hashes = load_source(source, all_connected=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pairs = [(int(pre), int(post)) for pre in matrix.index for post in matrix.columns if matrix.loc[pre, post] > 0]
    manifest = {"status": "running", "analysis": "all connected PFNd -> hDeltaB column combinations",
                "started_at_utc": datetime.now(timezone.utc).isoformat(), "server": SERVER, "dataset": DATASET,
                "source_sha256": source_hashes, "code": code,
                "code_sha256": hashes(sorted(Path(__file__).parent.glob("*.py"))),
                "packages": {p: version(p) for p in ("neuprint-python", "pandas", "numpy", "matplotlib")},
                "pairs": pairs, "column_caveat": COLUMN_CAVEAT,
                "coordinates": {"space": "native MaleCNS EM", "raw_unit": "8 nm voxel", "summary_unit": "µm",
                                "um_per_voxel": UM_PER_VOXEL, "anatomical_axis_orientation": "unverified"},
                "visualization": {"column_views": "all contacts and all members of each annotated column group",
                                  "detailed_pair_selection": "largest whole-connection weight per column pair; body IDs break ties",
                                  "absent_cells": "no connection reported in saved matrix, not biological absence"},
                "caveats": [COLUMN_CAVEAT, "Synapse polarity and arbor identity are separate.",
                            "Compartment labels are dataset assignments, not independently verified ground truth.",
                            "ROI meshes provide visual context only; no boundary distances or normalized coordinate inferred.",
                            "Multiple target C labels do not by themselves imply unrelated spatial annotations.",
                            "An aggregate contact position may mix arbors; inspect compartment-specific positions."]}
    path = output / "spatial_population.json"
    save_json(path, manifest)
    try:
        meta = fetch_meta(client=client)
        manifest["dataset_metadata"] = meta
        if meta.get("voxelSize") != [8, 8, 8] or meta.get("voxelUnits") != "nanometers":
            raise ValidationError("Unexpected coordinate units.")
        confidence = float(meta["postHighAccuracyThreshold"])
        if confidence != .5 or not {"FB", *LAYERS}.issubset(client.all_rois):
            raise ValidationError("Review changed confidence/ROI configuration.")
        manifest["query"] = {"confidence": confidence, "comparison": ">", "weight": "weight",
                             "contacts": "all ROIs, restricted to saved PFNd and hDeltaB populations",
                             "skeleton_heal": False, "population_filters": None}
        cache_info = compatible_cache(cache, meta)
        manifest["cache"] = {"source": str(cache_info[0]) if cache_info else None, "reused_sha256": {}}

        def cached(relative):
            found = cache_file(cache_info, relative)
            if found is not None:
                manifest["cache"]["reused_sha256"].update(hashes([found]))
            return found

        save_json(path, manifest)
        ids = sorted(annotations.bodyId.tolist())
        live, roi_counts = fetch_neurons(ids, returned_columns=ANNOTATION_COLUMNS, client=client)
        pd.testing.assert_frame_equal(
            annotations[["bodyId", "type", "instance"]].sort_values("bodyId").reset_index(drop=True),
            live[["bodyId", "type", "instance"]].sort_values("bodyId").reset_index(drop=True), check_dtype=False)
        annotations.to_csv(output / "neurons.csv", index=False)
        live.to_csv(output / "live_annotations.csv", index=False)
        roi_counts.to_csv(output / "neuron_roi_counts.csv", index=False)
        batches = []
        old_synapses = cached("fb_synapses.csv")
        if old_synapses is not None:
            points = pd.read_csv(old_synapses)
            batches.append(points[points.bodyId.isin(ids)])
        reused_ids = set(batches[0].bodyId) if batches else set()
        missing = [body for body in ids if body not in reused_ids]
        for start in range(0, len(missing), 8):
            bodies = missing[start:start + 8]
            print(f"Fetching FB sites: bodies {start + 1}–{start + len(bodies)} of {len(missing)} uncached", flush=True)
            batches.append(fetch_fb_synapses(client, bodies, confidence, allow_empty=True))
        synapses = pd.concat(batches, ignore_index=True).sort_values(["bodyId", "type", *XYZ])
        if not set(synapses.bodyId).issubset(ids) or synapses.duplicated(["bodyId", "type", *XYZ]).any():
            raise ValidationError("Unexpected bodies or duplicate population synapse sites.")
        synapses.to_csv(output / "fb_synapses.csv", index=False)
        batches = []
        pre_ids = annotations.loc[annotations.type == "PFNd", "bodyId"].tolist()
        post_ids = annotations.loc[annotations.type == "hDeltaB", "bodyId"].tolist()
        for i, pre in enumerate(pre_ids):
            print(f"Fetching contacts for PFNd {pre} ({i + 1}/{len(pre_ids)})", flush=True)
            raw = fetch_synapse_connections(int(pre), post_ids,
                    SynapseCriteria(primary_only=False, confidence=confidence, client=client), client=client)
            for post, pair in raw.groupby("bodyId_post", observed=True):
                batches.append(prepare_contacts(pair, int(pre), int(post)))
        if not batches:
            raise ValidationError("No population contacts returned.")
        contacts = annotate_contacts(pd.concat(batches, ignore_index=True), annotations)
        counts = validate_population(contacts, synapses, roi_counts, matrix)
        coverage = column_coverage(contacts, annotations)
        if (int(coverage.total_contacts.sum()) != int(matrix.to_numpy().sum())
                or int(coverage.connected_pairs.sum()) != len(pairs)
                or int(coverage.possible_pairs.sum()) != matrix.size):
            raise ValidationError("Column coverage failed count conservation.")
        for side in ("pre", "post"):
            contacts[f"roi_{side}"] = contacts[f"roi_{side}"].map(lambda v: json.dumps(v if isinstance(v, list) else [v]))
        contacts.to_csv(output / "contacts.csv", index=False)
        counts.to_csv(output / "pair_counts.csv", index=False)
        coverage.to_csv(output / "column_coverage.csv", index=False)
        arbor_contacts(contacts).to_csv(output / "contact_arbor_summary.csv", index=False)
        spatial_summary(synapses).merge(annotations[["bodyId", "instance", "type", "fb_column"]].rename(columns={"type": "neuron_type"}), on="bodyId").to_csv(output / "arbor_spatial_summary.csv", index=False)
        manifest["validation"] = {"neurons": len(ids), "connected_neuron_pairs": len(pairs),
                                  "possible_neuron_pairs": int(coverage.possible_pairs.sum()),
                                  "connected_column_pairs": int((coverage.total_contacts > 0).sum()),
                                  "unreported_column_pairs": int((coverage.total_contacts == 0).sum()),
                                  "contacts": len(contacts), "FB_post_contacts": int(contacts.FB_post.sum()),
                                  "FB_sites": len(synapses), "reference_counts": REFERENCE_COUNTS,
                                  "all_pair_counts_match_saved_weights": True, "roi_counts_match_points": True,
                                  "FB_contact_endpoints_match_neuron_points": True, "column_counts_conserved": True}
        for dirname in ("skeletons", "meshes"):
            (output / dirname).mkdir()
        for i, body in enumerate(ids):
            relative = f"skeletons/{body}.csv"
            previous = cached(relative)
            if previous is not None:
                shutil.copyfile(previous, output / relative)
            else:
                print(f"Fetching skeleton {body} ({i + 1}/{len(ids)})", flush=True)
                client.fetch_skeleton(int(body), heal=False).to_csv(output / relative, index=False)
        for roi in ("FB", "FBl3", "FBl4"):
            relative = f"meshes/{roi}.obj"
            previous = cached(relative)
            if previous is not None:
                shutil.copyfile(previous, output / relative)
            else:
                (output / relative).write_bytes(client.fetch_roi_mesh(roi))
        manifest["status"] = "data_complete"
        manifest["data_sha256_base"] = "."
        manifest["data_sha256"] = {str(Path(p).relative_to(output)): digest for p, digest in
                                   hashes(sorted(p for p in output.rglob("*") if p.is_file() and p.name != path.name)).items()}
        save_json(path, manifest)
        render_population(output)
        if hashes([Path(p) for p in source_hashes]) != source_hashes:
            raise ValidationError("Saved source inputs changed during extraction.")
        manifest["artifact_sha256"] = hashes(sorted(p for p in output.rglob("*") if p.is_file() and p.name != path.name))
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="validation_failed" if isinstance(exc, (ValidationError, AssertionError)) else "failed", error_type=type(exc).__name__)
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(path, manifest)
    return manifest


def render_population(run, output=None):
    """Render saved population data, with identical view limits for all comparisons."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    run = Path(run)
    output = Path(output) if output is not None else run
    for directory in ("columns", "pairs", "atlas"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    if output.resolve() != run.resolve():
        for name in GALLERY_TABLES:
            shutil.copyfile(run / name, output / name)
    annotations = pd.read_csv(run / "neurons.csv")
    synapses = pd.read_csv(run / "fb_synapses.csv")
    contacts = pd.read_csv(run / "contacts.csv")
    coverage = pd.read_csv(run / "column_coverage.csv")
    columns = annotations.set_index("bodyId").fb_column
    types = annotations.set_index("bodyId").type
    synapses["column"] = synapses.bodyId.map(columns)
    synapses["neuron_type"] = synapses.bodyId.map(types)
    vertices, _ = parse_obj((run / "meshes" / "FB.obj").read_bytes())
    vertices *= UM_PER_VOXEL
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    padding = (high - low) * .04
    low, high = low - padding, high + padding

    def context(ax, dims):
        ax.scatter(*vertices[::3, dims].T, s=.6, c="#aaa", alpha=.13, rasterized=True)
        ax.set_xlim(low[dims[0]], high[dims[0]])
        ax.set_ylim(low[dims[1]], high[dims[1]])
        ax.set_xlabel(f"{XYZ[dims[0]]} (µm)")
        ax.set_ylabel(f"{XYZ[dims[1]]} (µm)")
        ax.set_aspect("equal")

    def points(ax, table, dims, **kwargs):
        ax.scatter(*(table[XYZ].to_numpy()[:, dims] * UM_PER_VOXEL).T, rasterized=True, **kwargs)

    for row in coverage[coverage.total_contacts > 0].itertuples():
        print(f"Plotting columns C{row.column_pre} -> C{row.column_post}", flush=True)
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), layout="constrained")
        source = synapses[(synapses.neuron_type == "PFNd") & (synapses.column == row.column_pre) & (synapses.type == "pre")]
        target = synapses[(synapses.neuron_type == "hDeltaB") & (synapses.column == row.column_post)]
        pair = contacts[(contacts.column_pre == row.column_pre) & (contacts.column_post == row.column_post)]
        for ax, dims in zip(axes, ((0, 1), (0, 2), (1, 2))):
            context(ax, dims)
            points(ax, source, dims, s=6, c="#7851a9", alpha=.3, label="PFNd output sites (all partners)")
            for (compartment, polarity), group in target.groupby(["compartment", "type"], observed=True):
                points(ax, group, dims, s=5 if polarity == "post" else 12,
                       marker="." if polarity == "post" else "^", c=COLORS.get(compartment, "#a855f7"), alpha=.3,
                       label=f"hDeltaB {compartment}: {'receives input' if polarity == 'post' else 'provides output'}")
            ax.scatter(*(pair[[f"{a}_post" for a in XYZ]].to_numpy()[:, dims] * UM_PER_VOXEL).T,
                       s=13, facecolors="none", edgecolors="#dc2626", linewidths=.55,
                       label="Selected column contacts: receiving endpoints", rasterized=True)
        axes[0].legend(fontsize=5.5, loc="best")
        fig.suptitle(f"PFNd C{row.column_pre} → hDeltaB C{row.column_post} | all {row.connected_pairs} connected neuron pairs\n"
                     f"{row.total_contacts} contacts overall; {row.fb_post_contacts} FB-post | native XYZ", fontsize=12)
        fig.supxlabel("All neurons in each annotated column group shown. Grey FB mesh is visual context only; no inferred column boundaries.", fontsize=8)
        for ext in ("png", "pdf"):
            fig.savefig(output / "columns" / f"C{row.column_pre}_C{row.column_post}.{ext}", dpi=150)
        plt.close(fig)

    for pre in range(1, 9):
        fig, axes = plt.subplots(3, 4, figsize=(16, 13), layout="constrained")
        for post, ax in enumerate(axes.flat, start=1):
            context(ax, (0, 2))
            group = contacts[(contacts.column_pre == pre) & (contacts.column_post == post)]
            row = coverage[(coverage.column_pre == pre) & (coverage.column_post == post)].iloc[0]
            if group.empty:
                ax.text(.5, .5, "No reported connection", ha="center", va="center", transform=ax.transAxes, fontsize=9)
            else:
                target = synapses[(synapses.neuron_type == "hDeltaB") & (synapses.column == post) & (synapses.type == "post")]
                for compartment, background in target.groupby("compartment", observed=True):
                    points(ax, background, (0, 2), s=2, c=COLORS.get(compartment, "#a855f7"), alpha=.12)
                for compartment, selected in group.groupby("compartment_post", observed=True):
                    ax.scatter(selected.x_post * UM_PER_VOXEL, selected.z_post * UM_PER_VOXEL,
                               s=9, c=COLORS.get(compartment, "#a855f7"), alpha=.8, rasterized=True)
            ax.set_title(f"hDeltaB C{post}: {int(row.total_contacts)} total / {int(row.fb_post_contacts)} FB-post\n{int(row.connected_pairs)} connected neuron pairs", fontsize=9)
        handles = [Line2D([], [], marker="o", linestyle="", color=color, label=comp) for comp, color in COLORS.items()]
        fig.legend(handles=handles, loc="outside lower center", ncol=5, title="Contact receiving compartment; faint points show hDeltaB FB input sites")
        fig.suptitle(f"PFNd C{pre} → all hDeltaB columns | native XZ, identical scale\nAll reported contact endpoints; column labels are type-specific annotations", fontsize=14)
        for ext in ("png", "pdf"):
            fig.savefig(output / "atlas" / f"PFNd_C{pre}.{ext}", dpi=150)
        plt.close(fig)

    plot_column_locations(run, output)

    representatives = [(int(row.representative_pre), int(row.representative_post))
                       for row in coverage[coverage.total_contacts > 0].itertuples()]
    plot_pilot(run, output / "pairs", pairs=representatives)
    write_index(output, coverage, annotations, contacts)


def landmark_tables(synapses, summary, annotations):
    """PFNd landmarks pool all output sites per body, not compartment centroids."""
    by_id = annotations.set_index("bodyId")
    points = synapses[(synapses.bodyId.map(by_id.type) == "PFNd") & (synapses.type == "pre")]
    source = (points.groupby("bodyId")[XYZ].mean().mul(UM_PER_VOXEL)
              .rename(columns={axis: f"{axis}_mean_um" for axis in XYZ}).reset_index())
    source["fb_column"] = source.bodyId.map(by_id.fb_column)
    return [("PFNd output-site centroids", source),
            ("hDeltaB dendritic input-site centroids", summary[(summary.neuron_type == "hDeltaB") & (summary.type == "post") & (summary.compartment == "dendrite") & (summary.roi == "FB")]),
            ("hDeltaB axonal output-site centroids", summary[(summary.neuron_type == "hDeltaB") & (summary.type == "pre") & (summary.compartment == "axon") & (summary.roi == "FB")])]


def plot_column_locations(run, output=None):
    """Render only the landmark overview, independently of expensive pair plots."""
    import matplotlib.pyplot as plt
    run = Path(run)
    output = Path(output) if output is not None else run
    synapses = pd.read_csv(run / "fb_synapses.csv")
    summary = pd.read_csv(run / "arbor_spatial_summary.csv")
    annotations = pd.read_csv(run / "neurons.csv")
    groups = landmark_tables(synapses, summary, annotations)
    vertices, _ = parse_obj((run / "meshes" / "FB.obj").read_bytes())
    vertices *= UM_PER_VOXEL
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    padding = (high - low) * .04
    low, high = low - padding, high + padding
    records = []
    for label, table in groups:
        record = table[["bodyId", "fb_column", "x_mean_um", "y_mean_um", "z_mean_um"]].copy()
        record["landmark"] = label
        records.append(record)
    pd.concat(records, ignore_index=True).to_csv(output / "column_landmarks.csv", index=False)
    fig, axes = plt.subplots(3, 3, figsize=(15, 13), layout="constrained")
    for rownum, (label, table) in enumerate(groups):
        for ax, dims in zip(axes[rownum], ((0, 1), (0, 2), (1, 2))):
            ax.scatter(*vertices[::3, dims].T, s=.6, c="#aaa", alpha=.13, rasterized=True)
            ax.set_xlim(low[dims[0]], high[dims[0]])
            ax.set_ylim(low[dims[1]], high[dims[1]])
            ax.set_xlabel(f"{XYZ[dims[0]]} (µm)")
            ax.set_ylabel(f"{XYZ[dims[1]]} (µm)")
            ax.set_aspect("equal")
            for item in table.itertuples():
                x, y = getattr(item, f"{XYZ[dims[0]]}_mean_um"), getattr(item, f"{XYZ[dims[1]]}_mean_um")
                ax.scatter(x, y, s=18, c="#333")
                ax.annotate(f"C{item.fb_column}", (x, y), xytext=(2, 2), textcoords="offset points", fontsize=6)
            ax.set_title(label, fontsize=10)
    fig.suptitle("One point per neuron/arbor: FB synapse centroids across all partners\nNative XYZ; C labels do not define a shared coordinate or a shaft location", fontsize=13)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"column_locations.{ext}", dpi=160)
    plt.close(fig)

def write_index(output, coverage, annotations, contacts):
    """Local browsable report; no hosted service or external assets required."""
    output = Path(output)
    cells = []
    for pre in range(1, 9):
        row_cells = [f"<th>PFNd C{pre}<br><a href='atlas/PFNd_C{pre}.png'>12-panel atlas</a></th>"]
        for post in range(1, 13):
            row = coverage[(coverage.column_pre == pre) & (coverage.column_post == post)].iloc[0]
            if not row.total_contacts:
                row_cells.append("<td class='empty'>No reported connection</td>")
            else:
                rep = f"{int(row.representative_pre)}_{int(row.representative_post)}"
                row_cells.append(f"<td><a href='columns/C{pre}_C{post}.png'>All contacts</a> · "
                                 f"<a href='columns/C{pre}_C{post}.pdf'>PDF</a><br>{int(row.total_contacts)} contacts; "
                                 f"{int(row.connected_pairs)} neuron pairs<br>{int(row.fb_post_contacts)} FB-post: "
                                 f"{int(row.fb_post_axon_contacts)} axon / {int(row.fb_post_dendrite_contacts)} dendrite<br>"
                                 f"<a href='pairs/pair_{rep}.png'>Representative pair</a></td>")
        cells.append("<tr>" + "".join(row_cells) + "</tr>")
    by_id = annotations.set_index("bodyId")
    pair_rows = []
    for (pre, post), group in contacts.groupby(["bodyId_pre", "bodyId_post"]):
        cp, ct = int(group.column_pre.iloc[0]), int(group.column_post.iloc[0])
        counts = group.compartment_post.value_counts().to_dict()
        description = ", ".join(f"{html.escape(str(comp))}: {count}" for comp, count in counts.items())
        pair_rows.append(f"<tr><td>{pre} → {post}</td><td>{html.escape(by_id.loc[pre, 'instance'])}</td>"
                         f"<td>{html.escape(by_id.loc[post, 'instance'])}</td><td>{len(group)}</td>"
                         f"<td>{description}</td><td><a href='columns/C{cp}_C{ct}.png'>Column view</a></td></tr>")
    document = """<!doctype html><html lang='en'><meta charset='utf-8'>
<title>PFNd → hDeltaB spatial column atlas</title>
<style>body{font:15px system-ui;margin:24px;color:#222}table{border-collapse:collapse;margin:20px 0}th,td{border:1px solid #bbb;padding:9px;vertical-align:top;min-width:125px}th{background:#edf1f5}.empty{background:#eee;color:#666}a{color:#145ca2}.scroll{overflow-x:auto}p{max-width:1000px}</style>
<h1>PFNd → hDeltaB spatial column atlas</h1>
<p>Every connected annotated column combination, including all reported positive neuron pairs, without a minimum-weight cutoff beyond one contact. Column figures pool all contacts; detailed pair figures show the strongest whole-connection pair per cell (body IDs break ties).</p>
<p>Native MaleCNS coordinates in µm. Orange = labelled axon; cyan = labelled dendrite. “Pre” means provides output, “post” means receives input; both can occur on an axonal arbor. The grey FB mesh is visual context only. No shared column coordinate or shaft-based indexing rule is inferred. Missing cells mean no connection reported in the saved matrix, not biological absence.</p>
<p><a href='column_locations.png'>Population arbor positions</a> · <a href='column_locations.pdf'>PDF</a> · <a href='column_coverage.csv'>Coverage and counts</a> · <a href='contact_arbor_summary.csv'>Contact positions by arbor</a> · <a href='arbor_spatial_summary.csv'>Neuron arbor positions</a></p>
<div class='scroll'><table><thead><tr><th></th>"""
    document += "".join(f"<th>hDeltaB C{post}</th>" for post in range(1, 13))
    document += "</tr></thead><tbody>" + "".join(cells) + "</tbody></table></div>"
    document += "<h2>All connected neuron pairs</h2><p>Counts and receiving compartment labels below cover all ROIs; the column index above reports FB-post arbor counts separately.</p><table><thead><tr><th>Pair</th><th>PFNd</th><th>hDeltaB</th><th>Contacts</th><th>Receiving compartment</th><th>Figure</th></tr></thead><tbody>"
    document += "".join(pair_rows) + "</tbody></table></html>"
    (output / "index.html").write_text(document)
