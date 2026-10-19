"""Targeted FB spatial pilot; native coordinates and arbor labels stay explicit."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd
from neuprint import SynapseCriteria, fetch_meta, fetch_neurons, fetch_synapse_connections

from .anatomy import annotation_order, COLUMN_CAVEAT
from .connectivity import (
    ANNOTATION_COLUMNS, DATASET, SERVER, ValidationError, build_matrix,
    check_reference_annotations, check_reference_weight, reference_result,
)
from .plotting import load_matrix


PAIRS = ((26913, 515192), (77360, 515192), (26913, 27109),
         (26913, 513702), (25982, 11512))
LAYERS = tuple(f"FBl{i}" for i in range(1, 10))
XYZ = ["x", "y", "z"]
UM_PER_VOXEL = 0.008
REFERENCE_COUNTS = {"total": 137, "post_in_FB": 135, "both_in_FB": 133}


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def hashes(paths):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def load_source(source, *, all_connected=False):
    """Reuse complete saved populations and verify the matrix against saved edges."""
    source = Path(source).resolve()
    paths = [source / name for name in (
        "metadata.json", "PFNd_neurons.csv", "hDeltaB_neurons.csv",
        "PFNd_to_hDeltaB_connections.csv", "PFNd_to_hDeltaB_matrix.csv",
    )]
    before = hashes(paths)
    meta = json.loads(paths[0].read_text())
    if meta.get("status") != "complete" or meta.get("dataset") != DATASET or meta.get("server") != SERVER:
        raise ValidationError("Spatial pilot requires a complete extraction from the configured server/dataset.")
    populations = {typ: pd.read_csv(source / f"{typ}_neurons.csv") for typ in ("PFNd", "hDeltaB")}
    annotations = pd.concat([annotation_order(populations[t], t) for t in populations], ignore_index=True)
    matrix = load_matrix(paths[-1])
    expected = build_matrix(pd.read_csv(paths[-2]), populations["PFNd"].bodyId, populations["hDeltaB"].bodyId)
    pd.testing.assert_frame_equal(matrix, expected, check_names=False)
    check_reference_weight(reference_result(matrix))
    pairs = ([(int(pre), int(post)) for pre in matrix.index for post in matrix.columns
              if matrix.loc[pre, post] > 0] if all_connected else PAIRS)
    ids = (sorted(annotations.bodyId.tolist()) if all_connected
           else sorted({body for pair in pairs for body in pair}))
    annotations = annotations[annotations.bodyId.isin(ids)].copy()
    if set(annotations.bodyId) != set(ids) or annotations.bodyId.duplicated().any():
        raise ValidationError("Pilot neurons missing or duplicated in saved populations.")
    for pre, post in pairs:
        if int(matrix.loc[pre, post]) <= 0:
            raise ValidationError(f"Pilot pair {pre} -> {post} is absent from saved connectivity.")
    return annotations, matrix, before


def fetch_fb_synapses(client, body_ids, confidence, *, allow_empty=False):
    """Fetch each point once and retain every layer flag, avoiding ROI row expansion."""
    ids = [int(body) for body in body_ids]
    layer_fields = ", ".join(f"coalesce(s.{roi}, false) AS {roi}" for roi in LAYERS)
    query = f"""
        MATCH (n:Neuron) WHERE n.bodyId IN {ids}
        MATCH (n)-[:Contains]->(:SynapseSet)-[:Contains]->(s:Synapse)
        WHERE s.FB AND s.confidence > {float(confidence)}
        WITH DISTINCT n, s
        RETURN n.bodyId AS bodyId, s.type AS type,
               s.location.x AS x, s.location.y AS y, s.location.z AS z,
               s.confidence AS confidence,
               coalesce(s.compartment, 'unknown') AS compartment, s.split AS split,
               {layer_fields}
        ORDER BY bodyId, type, x, y, z
    """
    synapses = client.fetch_custom(query)
    key = ["bodyId", "type", *XYZ]
    if (synapses.empty and not allow_empty) or synapses[key].isna().any().any() or synapses.duplicated(key).any():
        raise ValidationError("Missing or duplicate FB synapse points.")
    if not set(synapses.bodyId).issubset(ids) or not set(synapses.type).issubset({"pre", "post"}):
        raise ValidationError("Unexpected bodies or synapse polarities.")
    if not np.isfinite(synapses[XYZ].to_numpy(dtype=float)).all():
        raise ValidationError("Nonfinite synapse coordinates.")
    return synapses


def prepare_contacts(contacts, pre, post):
    """Keep all contacts; ROI flags do not discard boundary or unassigned points."""
    contacts = contacts.copy()
    key = ["bodyId_pre", "bodyId_post", *[f"{a}_{side}" for side in ("pre", "post") for a in XYZ]]
    if contacts.empty or contacts[key].isna().any().any() or contacts.duplicated(key).any():
        raise ValidationError(f"Missing or duplicate contacts for {pre} -> {post}.")
    if set(contacts.bodyId_pre) != {pre} or set(contacts.bodyId_post) != {post}:
        raise ValidationError("Contact query returned an unexpected pair.")
    if not np.isfinite(contacts[key[2:]].to_numpy(dtype=float)).all():
        raise ValidationError("Nonfinite contact coordinates.")
    for side in ("pre", "post"):
        if f"compartment_{side}" not in contacts:
            contacts[f"compartment_{side}"] = "unknown"
        contacts[f"compartment_{side}"] = contacts[f"compartment_{side}"].astype(object).fillna("unknown")
        for roi in ("FB", *LAYERS):
            contacts[f"{roi}_{side}"] = contacts[f"roi_{side}"].map(
                lambda rois: roi in rois if isinstance(rois, (list, tuple, set)) else rois == roi
            )
    contacts["both_in_FB"] = contacts.FB_pre & contacts.FB_post
    return contacts.sort_values(key).reset_index(drop=True)


def validate_points(synapses, roi_counts, contacts):
    """Compare point counts with neuron ROI statistics and match both contact endpoints."""
    for row in roi_counts[roi_counts.roi.isin(["FB", *LAYERS])].itertuples():
        points = synapses[synapses.bodyId == row.bodyId]
        if row.roi != "FB":
            points = points[points[row.roi]]
        for polarity in ("pre", "post"):
            if int((points.type == polarity).sum()) != int(getattr(row, polarity)):
                raise ValidationError(f"Synapse/ROI count mismatch: {row.bodyId}, {row.roi}, {polarity}.")
    point_keys = set(synapses[["bodyId", "type", *XYZ]].itertuples(index=False, name=None))
    for side, polarity in (("pre", "pre"), ("post", "post")):
        selected = contacts[contacts[f"FB_{side}"]]
        for row in selected[[f"bodyId_{side}", *[f"{a}_{side}" for a in XYZ]]].itertuples(index=False, name=None):
            if (row[0], polarity, *row[1:]) not in point_keys:
                raise ValidationError("FB contact endpoint missing from individual-neuron points.")


def spatial_summary(synapses):
    """Summarize unique sites separately by body, polarity, compartment and layer."""
    rows = []
    for roi in ("FB", *LAYERS):
        points = synapses if roi == "FB" else synapses[synapses[roi]]
        for keys, group in points.groupby(["bodyId", "type", "compartment"], observed=True):
            row = dict(zip(["bodyId", "type", "compartment"], keys))
            row.update(roi=roi, count=len(group))
            for axis in XYZ:
                values = group[axis] * UM_PER_VOXEL
                for stat, value in (("mean", values.mean()), ("median", values.median()),
                                    ("std", values.std(ddof=0)), ("min", values.min()),
                                    ("max", values.max()), ("q10", values.quantile(.1)),
                                    ("q90", values.quantile(.9))):
                    row[f"{axis}_{stat}_um"] = float(value)
            rows.append(row)
    return pd.DataFrame(rows)


def contact_summary(contacts):
    rows = []
    for (pre, post), group in contacts.groupby(["bodyId_pre", "bodyId_post"]):
        for scope, mask in (("total", np.ones(len(group), dtype=bool)),
                            ("post_in_FB", group.FB_post), ("both_in_FB", group.both_in_FB)):
            points = group[mask]
            rows.append({"bodyId_pre": pre, "bodyId_post": post, "scope": scope,
                         "contacts": len(points),
                         "unique_pre_sites": len(points[[f"{a}_pre" for a in XYZ]].drop_duplicates()),
                         "unique_post_sites": len(points[[f"{a}_post" for a in XYZ]].drop_duplicates())})
    return pd.DataFrame(rows)


def parse_obj(data):
    """Read only geometry; OBJ indices are one-based and may contain slash fields."""
    vertices, faces = [], []
    for line in data.decode().splitlines():
        fields = line.split()
        if fields and fields[0] == "v":
            vertices.append([float(x) for x in fields[1:4]])
        elif fields and fields[0] == "f":
            face = [int(x.split('/')[0]) for x in fields[1:]]
            face = [x - 1 if x > 0 else len(vertices) + x for x in face]
            faces.extend([[face[0], face[i], face[i + 1]] for i in range(1, len(face) - 1)])
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def plot_pilot(run, output=None, *, pairs=None):
    """Replot saved data offline in common native XYZ axes (micrometers)."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    run = Path(run)
    output = Path(output) if output is not None else run
    output.mkdir(parents=True, exist_ok=True)
    synapses = pd.read_csv(run / "fb_synapses.csv")
    contacts = pd.read_csv(run / "contacts.csv")
    annotations = pd.read_csv(run / "neurons.csv").set_index("bodyId")
    if pairs is None:
        pairs = json.loads((run / "spatial_pilot.json").read_text()).get("pairs", PAIRS)
    vertices, faces = parse_obj((run / "meshes" / "FB.obj").read_bytes())
    vertices *= UM_PER_VOXEL
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    pad = (high - low) * .04
    low, high = low - pad, high + pad
    colors = {"axon": "#d97706", "dendrite": "#0891b2", "linker": "#6b7280", "unknown": "#a855f7"}
    for pre, post in pairs:
        print(f"Plotting {pre} -> {post}", flush=True)
        fig = plt.figure(figsize=(13, 10), layout="constrained")
        axes = [fig.add_subplot(2, 2, i + 1, **({"projection": "3d"} if i == 3 else {})) for i in range(4)]
        dims_list = [(0, 1), (0, 2), (1, 2), (0, 1, 2)]
        try:
            for ax, dims in zip(axes, dims_list):
                is3d = len(dims) == 3
                if is3d:
                    ax.add_collection3d(Poly3DCollection(vertices[faces], facecolor="#b0b0b0", edgecolor="none", alpha=.035, rasterized=True))
                else:
                    ax.scatter(*vertices[:, dims].T, s=.4, c="#c0c0c0", alpha=.12, rasterized=True)
                # The view box clips skeletons for display; it is not an FB membership test.
                for body, color in ((pre, "#7851a9"), (post, "#374151")):
                    sk = pd.read_csv(run / "skeletons" / f"{body}.csv").set_index("rowId")
                    child = sk[sk.link.isin(sk.index)]
                    segments = np.stack([child[XYZ].to_numpy(), sk.loc[child.link, XYZ].to_numpy()], axis=1) * UM_PER_VOXEL
                    visible = ((segments >= low) & (segments <= high)).all(axis=(1, 2))
                    segments = segments[visible][:, :, list(dims)]
                    collection = (Line3DCollection if is3d else LineCollection)(segments, colors=color, linewidths=.4, alpha=.3, rasterized=True)
                    ax.add_collection(collection)
                pfnd = synapses[(synapses.bodyId == pre) & (synapses.type == "pre")]
                ax.scatter(*(pfnd[XYZ].to_numpy()[:, dims] * UM_PER_VOXEL).T,
                           s=8, c="#7851a9", alpha=.4, label="PFNd pre sites", rasterized=True)
                hd = synapses[synapses.bodyId == post]
                for (compartment, polarity), group in hd.groupby(["compartment", "type"], observed=True):
                    ax.scatter(*(group[XYZ].to_numpy()[:, dims] * UM_PER_VOXEL).T,
                               s=5 if polarity == "post" else 14, marker="." if polarity == "post" else "^",
                               c=colors.get(compartment, "#a855f7"), alpha=.4,
                               label=f"hDeltaB {compartment} {polarity}", rasterized=True)
                pair = contacts[(contacts.bodyId_pre == pre) & (contacts.bodyId_post == post)]
                ax.scatter(*(pair[[f"{a}_post" for a in XYZ]].to_numpy()[:, dims] * UM_PER_VOXEL).T,
                           s=18, facecolors="none", edgecolors="#dc2626", linewidths=.7,
                           label="Pair post endpoints (all ROIs)", rasterized=True)
                for dim, setter in zip(dims, [ax.set_xlim, ax.set_ylim] + ([ax.set_zlim] if is3d else [])):
                    setter(low[dim], high[dim])
                ax.set_xlabel(f"{XYZ[dims[0]]} (µm)")
                ax.set_ylabel(f"{XYZ[dims[1]]} (µm)")
                if is3d:
                    ax.set_zlabel("z (µm)")
                    ax.set_box_aspect(high - low)
                else:
                    ax.set_aspect("equal")
            axes[0].legend(fontsize=6, loc="best")
            nfb = int(pair.FB_post.sum())
            fig.suptitle(f"{annotations.loc[pre, 'instance']} ({pre}) → {annotations.loc[post, 'instance']} ({post})\n"
                         f"{len(pair)} contacts overall; {nfb} FB-post | native EM axes, no anatomical axis assignment", fontsize=11)
            fig.supxlabel("Grey FB mesh is visual context only. Skeletons clipped to view box; synapse membership uses dataset ROI flags.", fontsize=8)
            for ext in ("png", "pdf"):
                fig.savefig(output / f"pair_{pre}_{post}.{ext}", dpi=170)
        finally:
            plt.close(fig)


def run_pilot(client, source, output, code):
    """Retrieve only seven bodies, five pairs and three visual-context meshes."""
    annotations, matrix, source_hashes = load_source(source)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "server": SERVER, "dataset": DATASET, "source_sha256": source_hashes,
                "code": code, "code_sha256": hashes(sorted(Path(__file__).parent.glob("*.py"))),
                "packages": {p: version(p) for p in ("neuprint-python", "pandas", "numpy", "matplotlib")},
                "pairs": [list(p) for p in PAIRS], "column_caveat": COLUMN_CAVEAT,
                "coordinates": {"space": "native MaleCNS EM", "raw_unit": "8 nm voxel", "summary_unit": "µm",
                                "um_per_voxel": UM_PER_VOXEL, "anatomical_axis_orientation": "unverified"},
                "caveats": ["Compartment labels are dataset assignments, checked visually, not validated ground truth.",
                            "Synapse polarity is distinct from arbor identity; axonal arbors receive input.",
                            "FB mesh is for visualization, not quantitative boundary measurements.",
                            "No normalized FB coordinate or annotation-to-position conversion is inferred.",
                            "Layer rows overlap their parent FB; do not sum them with FB rows."]}
    save_json(output / "spatial_pilot.json", manifest)
    try:
        meta = fetch_meta(client=client)
        manifest["dataset_metadata"] = meta
        if meta.get("voxelSize") != [8, 8, 8] or meta.get("voxelUnits") != "nanometers":
            raise ValidationError("Unexpected coordinate units.")
        if not {"FB", *LAYERS}.issubset(client.all_rois):
            raise ValidationError("Expected FB/layer ROIs unavailable.")
        confidence = float(meta["postHighAccuracyThreshold"])
        if confidence != .5:
            raise ValidationError("Review changed standard confidence threshold before comparing saved weights.")
        manifest["query"] = {"confidence": confidence, "comparison": ">", "weight": "weight",
                             "synapses": "FB-restricted; explicit layer flags; DISTINCT n,s",
                             "contacts": "all ROIs; both endpoints retained; primary_only=False",
                             "skeleton_heal": False, "population_filters": None}
        save_json(output / "spatial_pilot.json", manifest)
        ids = annotations.bodyId.tolist()
        live, roi_counts = fetch_neurons(ids, returned_columns=ANNOTATION_COLUMNS, client=client)
        pd.testing.assert_frame_equal(
            annotations[["bodyId", "type", "instance"]].sort_values("bodyId").reset_index(drop=True),
            live[["bodyId", "type", "instance"]].sort_values("bodyId").reset_index(drop=True), check_dtype=False)
        check_reference_annotations(live[live.bodyId.isin([26913, 515192])])
        annotations.to_csv(output / "neurons.csv", index=False)
        live.to_csv(output / "live_annotations.csv", index=False)
        roi_counts.to_csv(output / "neuron_roi_counts.csv", index=False)
        print(f"Fetching FB synapses for {len(ids)} neurons", flush=True)
        synapses = fetch_fb_synapses(client, ids, confidence)
        if set(synapses.bodyId) != set(ids):
            raise ValidationError("A pilot neuron has no returned FB points.")
        synapses.to_csv(output / "fb_synapses.csv", index=False)
        batches = []
        for pre, post in PAIRS:
            print(f"Fetching contacts {pre} -> {post}", flush=True)
            pair = fetch_synapse_connections(pre, post, SynapseCriteria(primary_only=False, confidence=confidence, client=client), client=client)
            pair = prepare_contacts(pair, pre, post)
            if len(pair) != int(matrix.loc[pre, post]):
                raise ValidationError(f"Contact count differs from saved weight for {pre} -> {post}.")
            batches.append(pair)
        contacts = pd.concat(batches, ignore_index=True)
        validate_points(synapses, roi_counts, contacts)
        summary = contact_summary(contacts)
        ref = summary[(summary.bodyId_pre == 26913) & (summary.bodyId_post == 515192)]
        observed = ref.set_index("scope").contacts.astype(int).to_dict()
        manifest["validation"] = {"reference_counts": observed, "expected_reference_counts": REFERENCE_COUNTS,
                                  "all_pair_counts_match_saved_weights": True, "roi_counts_match_points": True,
                                  "FB_contact_endpoints_match_neuron_points": True}
        if observed != REFERENCE_COUNTS:
            raise ValidationError("Reference spatial counts changed; review ROI membership before continuing.")
        for side in ("pre", "post"):
            contacts[f"roi_{side}"] = contacts[f"roi_{side}"].map(lambda v: json.dumps(v if isinstance(v, list) else [v]))
        contacts.to_csv(output / "contacts.csv", index=False)
        summary.to_csv(output / "pair_counts.csv", index=False)
        spatial_summary(synapses).merge(annotations[["bodyId", "instance", "type", "fb_column"]].rename(columns={"type": "neuron_type"}), on="bodyId").to_csv(output / "arbor_spatial_summary.csv", index=False)
        for dirname in ("skeletons", "meshes"):
            (output / dirname).mkdir()
        for body in ids:
            print(f"Fetching skeleton {body}", flush=True)
            client.fetch_skeleton(int(body), heal=False).to_csv(output / "skeletons" / f"{body}.csv", index=False)
        for roi in ("FB", "FBl3", "FBl4"):
            print(f"Fetching mesh {roi}", flush=True)
            (output / "meshes" / f"{roi}.obj").write_bytes(client.fetch_roi_mesh(roi))
        plot_pilot(output)
        if hashes([Path(p) for p in source_hashes]) != source_hashes:
            raise ValidationError("Saved source inputs changed during pilot.")
        manifest["artifact_sha256"] = hashes(sorted(p for p in output.rglob("*") if p.is_file() and p.name != "spatial_pilot.json"))
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="validation_failed" if isinstance(exc, (ValidationError, AssertionError)) else "failed", error_type=type(exc).__name__)
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(output / "spatial_pilot.json", manifest)
    return manifest
