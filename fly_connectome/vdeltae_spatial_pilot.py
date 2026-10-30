"""Bounded FB coordinate extraction, descriptive summaries and offline figures."""

from datetime import datetime, timezone
from importlib.metadata import version
from io import StringIO
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from neuprint import fetch_meta

from .connectivity import DATASET, SERVER, ValidationError
from .extension_analysis import markdown
from .inputs import integers, save_json, sha256
from .spatial import parse_obj
from .vdeltae_spatial_audit import (
    check_probe_metadata, spatial_probe_queries, verify_spatial_data_probe,
)

XYZ = ["x", "y", "z"]
LAYERS = [f"FBl{layer}" for layer in range(1, 10)]
SWC_COLUMNS = ["node_id", "node_type", "x", "y", "z", "radius", "parent_id"]
CATEGORY_KEYS = ["bodyId", "synapse_type", "compartment", "split"]
COORDINATE_KIND = "vdeltae_hdeltag_coordinate_pilot_v1"
ANALYSIS_FILES = (
    "site_counts.csv", "site_spatial_summary.csv", "layer_summary.csv", "coordinate_ranges.csv",
)


def fb_site_query(body_ids, *, full_population=False):
    body_ids = list(body_ids)
    spatial_probe_queries(body_ids, full_population=full_population)
    ids = sorted(int(value) for value in body_ids)
    layers = ", ".join(f"s.{name} AS {name}" for name in LAYERS)
    return f"""
        MATCH (n:Neuron) WHERE n.bodyId IN {ids}
        MATCH (n)-[:Contains]->(:SynapseSet)-[:Contains]->(s:Synapse)
        WHERE s.FB = true
        WITH DISTINCT n, s
        RETURN n.bodyId AS bodyId, id(s) AS synapse_node_id,
               s.type AS synapse_type,
               s.location.x AS x, s.location.y AS y, s.location.z AS z,
               s.confidence AS confidence, s.compartment AS compartment,
               s.split AS split, s.FB AS FB, {layers}
        ORDER BY bodyId, synapse_node_id
    """


def native_scale_um(metadata):
    if metadata.get("voxelUnits") != "nanometers" or metadata.get("voxelSize") != [
        8,
        8,
        8,
    ]:
        raise ValidationError(
            "Review changed coordinate units before reusing native geometry."
        )
    return np.asarray(metadata["voxelSize"], dtype=float) / 1000


def category_counts(points):
    return (
        points.groupby(CATEGORY_KEYS, dropna=False)
        .size()
        .rename("site_count")
        .reset_index()
    )


def validate_fb_points(points, probe, availability, expected_categories):
    required = {
        "bodyId",
        "synapse_node_id",
        "synapse_type",
        *XYZ,
        "confidence",
        "compartment",
        "split",
        "FB",
        *LAYERS,
    }
    if not required.issubset(points.columns):
        raise ValidationError("FB point table is missing required columns.")
    integers(points["bodyId"], "Point body IDs", 1)
    integers(points["synapse_node_id"], "Graph synapse IDs", 0)
    if not set(points["bodyId"]).issubset(set(probe["bodyId"])):
        raise ValidationError("Point query returned an unselected body.")
    if points.duplicated(["bodyId", "synapse_node_id"]).any():
        raise ValidationError("Repeated body/synapse-node identity.")
    if points.duplicated(["bodyId", "synapse_type", *XYZ]).any():
        raise ValidationError(
            "Repeated site coordinates within body/polarity; review raw nodes."
        )
    if (
        not points["synapse_type"].isin(["pre", "post"]).all()
        or not points["FB"].eq(True).all()
    ):
        raise ValidationError("Unexpected site polarity or FB flag.")
    if not np.isfinite(points[XYZ].to_numpy(dtype=float)).all():
        raise ValidationError("Missing or nonfinite site coordinates.")
    confidence = pd.to_numeric(points["confidence"], errors="raise").dropna()
    if not np.isfinite(confidence).all() or not confidence.between(0, 1).all():
        raise ValidationError("Invalid nonmissing confidence score.")
    expected = availability.set_index("bodyId")["sites_marked_FB"]
    actual = points.groupby("bodyId").size().reindex(expected.index, fill_value=0)
    if not actual.eq(expected).all():
        raise ValidationError(
            "FB point counts differ from the saved probe denominator."
        )
    columns = [*CATEGORY_KEYS, "site_count"]
    try:
        pd.testing.assert_frame_equal(
            category_counts(points)[columns]
            .sort_values(CATEGORY_KEYS)
            .reset_index(drop=True),
            expected_categories[columns]
            .sort_values(CATEGORY_KEYS)
            .reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError as error:
        raise ValidationError(
            "Point categories differ from the saved unfiltered categories."
        ) from error


def read_native_swc(path):
    text = Path(path).read_text()
    comments = [line for line in text.splitlines() if line.lstrip().startswith("#")]
    lines = [
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines or any(len(line.split()) != 7 for line in lines):
        raise ValidationError("Expected nonempty seven-column SWC geometry.")
    table = pd.read_csv(StringIO("\n".join(lines)), sep=r"\s+", names=SWC_COLUMNS)
    for name, minimum in [("node_id", 1), ("node_type", 0), ("parent_id", -1)]:
        table[name] = integers(table[name], name, minimum)
    if (
        not np.isfinite(table[[*XYZ, "radius"]].to_numpy(dtype=float)).all()
        or table["radius"].lt(0).any()
    ):
        raise ValidationError("Invalid skeleton coordinates or radii.")
    if table["node_id"].duplicated().any():
        raise ValidationError("Duplicate skeleton node IDs.")
    ids = set(table["node_id"])
    parents = dict(zip(table["node_id"], table["parent_id"]))
    if not set(table["parent_id"]).issubset(ids | {-1}):
        raise ValidationError("Skeleton contains an absent parent node.")
    finished = set()
    for node in parents:
        path_nodes = set()
        while node != -1 and node not in finished:
            if node in path_nodes:
                raise ValidationError("Skeleton contains a parent cycle.")
            path_nodes.add(node)
            node = parents[node]
        finished.update(path_nodes)
    return table, comments


def skeleton_segments(table):
    indexed = table.set_index("node_id")
    children = table.loc[table["parent_id"].ne(-1)]
    return np.stack(
        [
            children[XYZ].to_numpy(dtype=float),
            indexed.loc[children["parent_id"], XYZ].to_numpy(dtype=float),
        ],
        axis=1,
    )


def read_context_mesh(path):
    vertices, faces = parse_obj(Path(path).read_bytes())
    validate_context_mesh(vertices, faces)
    return vertices, faces


def validate_context_mesh(vertices, faces):
    """Validate triangle geometry without treating it as an anatomical boundary."""
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not len(vertices)
        or not np.isfinite(vertices).all()
    ):
        raise ValidationError("Invalid or empty FB mesh vertices.")
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
        or not np.issubdtype(faces.dtype, np.integer)
        or faces.min() < 0
        or faces.max() >= len(vertices)
    ):
        raise ValidationError("Invalid or empty FB mesh faces.")


def body_projection_figure(points, skeleton, scale_um, title, *, mesh=None):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection

    scale = np.asarray(scale_um, dtype=float)
    if scale.shape != (3,) or not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValidationError(
            "Expected three positive finite coordinate scale factors."
        )
    segments = skeleton_segments(skeleton) * scale
    context = skeleton[XYZ].to_numpy(dtype=float) * scale
    if len(points):
        context = np.concatenate([context, points[XYZ].to_numpy(dtype=float) * scale])
    mesh_triangles = None
    if mesh is not None:
        vertices, faces = mesh
        validate_context_mesh(vertices, faces)
        vertices_um = vertices * scale
        mesh_triangles = vertices_um[faces]
        context = np.concatenate([context, vertices_um])
    views = [(0, 1), (0, 2), (1, 2)]
    colors = {"axon": "#d97706", "dendrite": "#0891b2", "linker": "#6b7280"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), layout="constrained")
    try:
        for column, (a, b) in enumerate(views):
            for row in [0, 1]:
                ax = axes[row, column]
                if mesh_triangles is not None:
                    ax.add_collection(PolyCollection(
                        mesh_triangles[:, :, [a, b]], facecolors="#aeb6bf",
                        edgecolors="none", alpha=.12, rasterized=True, zorder=0,
                        label="FB mesh (context)",
                    ))
                if row == 1:
                    ax.add_collection(
                        LineCollection(
                            segments[:, :, [a, b]],
                            colors="#777777",
                            linewidths=0.4,
                            alpha=0.45,
                            zorder=1,
                        )
                    )
                for (compartment, polarity), group in points.groupby(
                    ["compartment", "synapse_type"], dropna=False
                ):
                    label = "unannotated" if pd.isna(compartment) else str(compartment)
                    coordinates = group[XYZ].to_numpy(dtype=float) * scale
                    ax.scatter(
                        coordinates[:, a],
                        coordinates[:, b],
                        s=5,
                        marker="^" if polarity == "pre" else ".",
                        color=colors.get(label, "#a855f7"),
                        alpha=0.45,
                        label=f"{label} / {polarity} (n={len(group)})",
                        rasterized=True,
                        zorder=3,
                    )
                for axis, setter in [(a, ax.set_xlim), (b, ax.set_ylim)]:
                    low, high = context[:, axis].min(), context[:, axis].max()
                    padding = max((high - low) * 0.05, 0.1)
                    setter(low - padding, high + padding)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlabel(f"Native {XYZ[a]} (µm; orientation unverified)")
                ax.set_ylabel(f"Native {XYZ[b]} (µm)")
                ax.set_title(
                    "FB sites" if row == 0 else "Same sites + coarse unhealed skeleton"
                )
        if len(points) or mesh is not None:
            axes[0, 0].legend(fontsize=7, loc="upper left" if mesh is not None else "best")
        fig.suptitle(
            title
            + "\nDataset compartment colors; separate polarity markers; columns unresolved."
            + ("\nFB mesh uses native coordinates for visual context only." if mesh is not None else "")
        )
        return fig
    except Exception:
        plt.close(fig)
        raise


def mesh_preview_figure(vertices):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    try:
        for ax, (a, b) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
            ax.scatter(
                vertices[:, a],
                vertices[:, b],
                s=0.2,
                color="#777777",
                alpha=0.2,
                rasterized=True,
            )
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(f"OBJ {XYZ[a]} (raw units)")
            ax.set_ylabel(f"OBJ {XYZ[b]} (raw units)")
        fig.suptitle(
            "FB mesh vertex preview only; no containment or registration inferred"
        )
        return fig
    except Exception:
        plt.close(fig)
        raise


def mesh_context_ranges(points, probe, skeletons, vertices, scale_um):
    """Describe native-range compatibility; bounding-box counts are not containment."""
    scale = np.asarray(scale_um, dtype=float)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    rows = []
    objects = [(None, "FB_mesh_vertices", vertices)]
    for body in probe["bodyId"]:
        body = int(body)
        sites = points.loc[points["bodyId"].eq(body), XYZ].to_numpy(dtype=float)
        if len(sites) and ((sites.max(axis=0) < low) | (sites.min(axis=0) > high)).any():
            raise ValidationError("FB site and mesh coordinate ranges do not overlap; review frame and units.")
        objects.extend([
            (body, "FB_sites", sites),
            (body, "skeleton_nodes", skeletons[body][XYZ].to_numpy(dtype=float)),
        ])
    for body, name, coordinates in objects:
        within = ((coordinates >= low) & (coordinates <= high)).all(axis=1)
        row = {
            "bodyId": body, "source": name, "point_count": len(coordinates),
            "within_mesh_xyz_bounds_count": int(within.sum()),
            "within_mesh_xyz_bounds_fraction": float(within.mean()) if len(coordinates) else np.nan,
        }
        for index, axis in enumerate(XYZ):
            for stat, operation in [("min", np.min), ("max", np.max)]:
                value = float(operation(coordinates[:, index])) if len(coordinates) else np.nan
                row[f"{axis}_{stat}_native"] = value
                row[f"{axis}_{stat}_um"] = value * scale[index]
        rows.append(row)
    return pd.DataFrame(rows)


def coordinate_analysis_tables(points, probe, skeletons, scale_um):
    """Describe validated sites without dropping null labels or overlapping layers."""
    spatial_rows, layer_rows, range_rows = [], [], []
    scale = np.asarray(scale_um, dtype=float)
    body_totals = points.groupby("bodyId").size()
    for (body, polarity, compartment), group in points.groupby(
        ["bodyId", "synapse_type", "compartment"], dropna=False
    ):
        base = {"bodyId": int(body), "synapse_type": polarity, "compartment": compartment}
        row = {
            **base, "site_count": len(group), "body_FB_sites": int(body_totals.loc[body]),
            "fraction_of_body_FB_sites": len(group) / body_totals.loc[body],
            "confidence_missing": int(group["confidence"].isna().sum()),
        }
        for index, axis in enumerate(XYZ):
            values = group[axis].to_numpy(dtype=float) * scale[index]
            for name, value in [
                ("mean", values.mean()), ("median", np.median(values)),
                ("q10", np.quantile(values, .1)), ("q90", np.quantile(values, .9)),
            ]:
                row[f"{axis}_{name}_um"] = float(value)
        spatial_rows.append(row)
        for layer in LAYERS:
            flags = group[layer]
            if not flags.dropna().isin([True, False]).all():
                raise ValidationError("Layer flags must be boolean or missing.")
            marked = int(flags.eq(True).sum())
            layer_rows.append({
                **base, "layer": layer, "sites_true": marked,
                "sites_false": int(flags.eq(False).sum()), "sites_unknown": int(flags.isna().sum()),
                "group_FB_sites": len(group), "fraction_marked_of_group_FB_sites": marked / len(group),
            })
    for body in probe["bodyId"]:
        body = int(body)
        for name, coordinates in [
            ("FB_sites", points.loc[points["bodyId"].eq(body), XYZ].to_numpy(dtype=float)),
            ("skeleton_nodes", skeletons[body][XYZ].to_numpy(dtype=float)),
        ]:
            row = {"bodyId": body, "source": name, "count": len(coordinates)}
            for index, axis in enumerate(XYZ):
                for stat, operation in [("min", np.min), ("max", np.max)]:
                    value = float(operation(coordinates[:, index])) if len(coordinates) else np.nan
                    row[f"{axis}_{stat}_native"] = value
                    row[f"{axis}_{stat}_um"] = value * scale[index]
            range_rows.append(row)
    spatial_columns = [
        "bodyId", "synapse_type", "compartment", "site_count", "body_FB_sites",
        "fraction_of_body_FB_sites", "confidence_missing",
        *[f"{axis}_{stat}_um" for axis in XYZ for stat in ["mean", "median", "q10", "q90"]],
    ]
    layer_columns = [
        "bodyId", "synapse_type", "compartment", "layer", "sites_true", "sites_false",
        "sites_unknown", "group_FB_sites", "fraction_marked_of_group_FB_sites",
    ]
    return {
        "site_counts.csv": category_counts(points),
        "site_spatial_summary.csv": pd.DataFrame(spatial_rows, columns=spatial_columns),
        "layer_summary.csv": pd.DataFrame(layer_rows, columns=layer_columns),
        "coordinate_ranges.csv": pd.DataFrame(range_rows),
    }


def pilot_code_provenance():
    return {
        "code_sha256": {
            path.name: sha256(path) for path in sorted(Path(__file__).resolve().parent.glob("*.py"))
        },
        "python_version": platform.python_version(),
        "packages": {name: version(name) for name in ["neuprint-python", "numpy", "pandas", "matplotlib"]},
    }


def record_artifact_hashes(directory, manifest_name):
    return {
        path.name: sha256(path) for path in sorted(directory.iterdir())
        if path.is_file() and path.name != manifest_name
    }


def run_coordinate_pilot(client, source_probe, output):
    """Save selected coordinates; require a fresh directory and stable sources."""
    source_probe, output = Path(source_probe).resolve(), Path(output).resolve()
    source_hash = sha256(source_probe / "capability.json")
    source_manifest = verify_spatial_data_probe(source_probe)
    inventory = source_manifest["source_inventory"]
    if sha256(source_probe / "capability.json") != source_hash:
        raise ValidationError("Source probe manifest changed during initial verification.")
    protected = [source_probe, Path(inventory["annotation_audit_directory"])] + [
        Path(source["source_manifest_file"]).parent for source in inventory["sources"].values()
    ]
    for directory in protected:
        if output.is_relative_to(directory) or directory.is_relative_to(output):
            raise ValidationError("Coordinate output must be separate from all source directories.")
    if client.server.rstrip("/") != f"https://{SERVER}" or client.dataset != DATASET:
        raise ValidationError("Client target differs from the configured dataset.")
    probe = pd.read_csv(source_probe / "probe_bodies.csv", dtype={"bodyId": "Int64"})
    full_population = source_manifest.get("full_population", False)
    query = fb_site_query(probe["bodyId"], full_population=full_population)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "kind": COORDINATE_KIND, "status": "running",
        "full_population": full_population,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_probe": str(source_probe), "source_manifest_sha256": source_hash,
        "source_artifact_sha256": source_manifest["artifact_sha256"],
        "source_inventory": inventory, "source_resources": source_manifest["resources"],
        "selected_body_ids": [int(value) for value in probe["bodyId"]], "query": query,
        "settings": {
            "confidence_filter": None, "site_scope": "Distinct FB-marked Synapse nodes.",
            "coordinates": "Native MaleCNS XYZ; anatomical axis orientation unverified.",
            "unit_basis": "Saved 8 nm metadata and native MaleCNS skeleton documentation.",
            "skeleton_healing": False, "column_mapping": "unresolved",
            "analyses": "Descriptive groups, spatial quantiles, nullable layer flags and coordinate ranges.",
        },
    }
    save_json(output / "coordinates.json", manifest)
    try:
        manifest.update(pilot_code_provenance())
        before = fetch_meta(client=client)
        save_json(output / "metadata_before.json", before)
        check_probe_metadata(before, inventory)
        scale = native_scale_um(before)
        probe.to_csv(output / "probe_bodies.csv", index=False)
        points = client.fetch_custom(query)
        points.to_csv(output / "fb_sites_raw.csv", index=False)
        validate_fb_points(
            points, probe, pd.read_csv(source_probe / "site_availability.csv"),
            pd.read_csv(source_probe / "fb_site_categories.csv"),
        )
        skeletons = {}
        for body in probe["bodyId"]:
            body = int(body)
            resource = source_manifest["resources"][f"skeleton_{body}"]
            if resource["status"] != "retrieved":
                raise ValidationError("Morphology inspection requires every selected skeleton to be retrieved.")
            raw = source_probe / resource["file"]
            skeletons[body], comments = read_native_swc(raw)
            (output / f"skeleton_{body}.swc").write_bytes(raw.read_bytes())
            skeletons[body].to_csv(output / f"skeleton_{body}_nodes.csv", index=False)
            save_json(output / f"skeleton_{body}_comments.json", comments)
        mesh = source_manifest["resources"]["FB_mesh"]
        if mesh["status"] == "retrieved":
            mesh_path = source_probe / mesh["file"]
            read_context_mesh(mesh_path)
            (output / "FB.obj").write_bytes(mesh_path.read_bytes())
        for filename, table in coordinate_analysis_tables(points, probe, skeletons, scale).items():
            table.to_csv(output / filename, index=False)
        after = fetch_meta(client=client)
        save_json(output / "metadata_after.json", after)
        check_probe_metadata(after, inventory)
        verify_spatial_data_probe(source_probe)
        if sha256(source_probe / "capability.json") != source_hash:
            raise ValidationError("Saved source probe changed during coordinate extraction.")
        manifest["native_to_um_per_axis"] = scale.tolist()
        manifest["point_count"] = len(points)
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error_type"] = type(error).__name__
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["artifact_sha256"] = record_artifact_hashes(output, "coordinates.json")
        save_json(output / "coordinates.json", manifest)
    return manifest


def verify_coordinate_pilot(run):
    """Check saved hashes, source provenance, morphology and reconstructed analyses offline."""
    run = Path(run).resolve()
    manifest = json.loads((run / "coordinates.json").read_text())
    if manifest.get("kind") != COORDINATE_KIND or manifest.get("status") != "complete":
        raise ValidationError("Verification requires a complete coordinate pilot.")
    ids = manifest["selected_body_ids"]
    required = {"metadata_before.json", "metadata_after.json", "probe_bodies.csv", "fb_sites_raw.csv", *ANALYSIS_FILES}
    for body in ids:
        required.update({f"skeleton_{body}.swc", f"skeleton_{body}_nodes.csv", f"skeleton_{body}_comments.json"})
    if manifest["source_resources"]["FB_mesh"]["status"] == "retrieved":
        required.add("FB.obj")
    hashes = manifest.get("artifact_sha256", {})
    if not required.issubset(hashes):
        raise ValidationError("Coordinate artifact inventory is incomplete.")
    for name, digest in hashes.items():
        path = (run / name).resolve()
        if not path.is_relative_to(run) or not path.is_file() or sha256(path) != digest:
            raise ValidationError("Coordinate artifact is missing, outside the run, or changed.")
    source = Path(manifest["source_probe"])
    source_manifest = verify_spatial_data_probe(source)
    if sha256(source / "capability.json") != manifest["source_manifest_sha256"]:
        raise ValidationError("Coordinate source probe manifest changed.")
    for key, source_key in [
        ("source_inventory", "source_inventory"), ("source_artifact_sha256", "artifact_sha256"),
        ("source_resources", "resources"),
    ]:
        if manifest[key] != source_manifest[source_key]:
            raise ValidationError("Coordinate provenance differs from its verified source.")
    probe = pd.read_csv(run / "probe_bodies.csv")
    pd.testing.assert_frame_equal(probe, pd.read_csv(source / "probe_bodies.csv"), check_dtype=False)
    full_population = source_manifest.get("full_population", False)
    if manifest.get("full_population", False) != full_population:
        raise ValidationError("Coordinate scope differs from its verified source.")
    if ids != probe["bodyId"].tolist() or manifest["query"] != fb_site_query(ids, full_population=full_population):
        raise ValidationError("Coordinate selection or query differs from the source probe.")
    points = pd.read_csv(run / "fb_sites_raw.csv", low_memory=False)
    validate_fb_points(points, probe, pd.read_csv(source / "site_availability.csv"),
                       pd.read_csv(source / "fb_site_categories.csv"))
    if len(points) != manifest["point_count"]:
        raise ValidationError("Recorded site count differs from raw points.")
    for filename in ["metadata_before.json", "metadata_after.json"]:
        metadata = json.loads((run / filename).read_text())
        check_probe_metadata(metadata, manifest["source_inventory"])
        scale = native_scale_um(metadata)
        if scale.tolist() != manifest["native_to_um_per_axis"]:
            raise ValidationError("Coordinate scale differs from recorded metadata.")
    skeletons = {}
    for body in ids:
        path = run / f"skeleton_{body}.swc"
        resource = source_manifest["resources"][f"skeleton_{body}"]
        if sha256(path) != source_manifest["artifact_sha256"][resource["file"]]:
            raise ValidationError("Copied skeleton differs from the source geometry.")
        skeletons[body], comments = read_native_swc(path)
        pd.testing.assert_frame_equal(skeletons[body], pd.read_csv(run / f"skeleton_{body}_nodes.csv"), check_dtype=False)
        if comments != json.loads((run / f"skeleton_{body}_comments.json").read_text()):
            raise ValidationError("Saved skeleton comments differ from raw SWC headers.")
    if "FB.obj" in required:
        resource = source_manifest["resources"]["FB_mesh"]
        if sha256(run / "FB.obj") != source_manifest["artifact_sha256"][resource["file"]]:
            raise ValidationError("Copied mesh differs from the source geometry.")
        read_context_mesh(run / "FB.obj")
    for filename, expected in coordinate_analysis_tables(points, probe, skeletons, scale).items():
        pd.testing.assert_frame_equal(expected, pd.read_csv(run / filename), check_dtype=False, rtol=1e-12, atol=1e-12)
    return manifest


def axis_distribution_figure(points, probe, scale_um, bins=40):
    import matplotlib.pyplot as plt

    scale = np.asarray(scale_um)
    colors = {"axon": "#d97706", "dendrite": "#0891b2", "linker": "#6b7280"}
    fig, axes = plt.subplots(len(probe), 3, figsize=(14, max(7, 2.8 * len(probe))), squeeze=False, layout="constrained")
    try:
        for row, neuron in enumerate(probe.itertuples(index=False)):
            selected = points.loc[points["bodyId"].eq(neuron.bodyId)]
            for index, axis in enumerate(XYZ):
                ax = axes[row, index]
                all_values = points[axis].to_numpy(dtype=float) * scale[index]
                edges = np.histogram_bin_edges(all_values, bins=bins)
                for (compartment, polarity), group in selected.groupby(["compartment", "synapse_type"], dropna=False):
                    label = "unannotated" if pd.isna(compartment) else str(compartment)
                    ax.hist(group[axis].to_numpy(dtype=float) * scale[index], bins=edges,
                            histtype="step", color=colors.get(label, "#a855f7"),
                            linestyle="-" if polarity == "pre" else "--",
                            label=f"{label}/{polarity}, n={len(group)}")
                ax.set_xlabel(f"Native {axis} (µm; orientation unverified)")
                ax.set_ylabel("Site count per bin")
                ax.set_title(f"{neuron.type} {int(neuron.bodyId)}")
            if len(selected):
                axes[row, 0].legend(fontsize=6)
        fig.suptitle("Native-axis distributions; shared bins, raw counts, no column interpretation")
        return fig
    except Exception:
        plt.close(fig)
        raise


def export_coordinate_pilot(run, output, *, mesh_overlay=False):
    """Render a verified pilot offline into a separate, new export directory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run, output = Path(run).resolve(), Path(output).resolve()
    source_hash = sha256(run / "coordinates.json")
    manifest = verify_coordinate_pilot(run)
    full_population = manifest.get("full_population", False)
    if type(mesh_overlay) is not bool:
        raise ValidationError("mesh_overlay must be a boolean.")
    if mesh_overlay and manifest["source_resources"]["FB_mesh"]["status"] != "retrieved":
        raise ValidationError("FB context overlay requires a retrieved, verified mesh.")
    if full_population and not mesh_overlay:
        raise ValidationError("Full-population export requires mesh_overlay=True for anatomical context.")
    protected = [run, Path(manifest["source_probe"]), Path(manifest["source_inventory"]["annotation_audit_directory"])] + [
        Path(source["source_manifest_file"]).parent for source in manifest["source_inventory"]["sources"].values()
    ]
    for directory in protected:
        if output.is_relative_to(directory) or directory.is_relative_to(output):
            raise ValidationError("Plot directory must be separate from coordinate and source directories.")
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "kind": "vdeltae_hdeltag_coordinate_figures_v1", "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "source": str(run),
        "source_manifest_sha256": source_hash,
        "full_population": full_population,
        "settings": {"formats": ["png", "pdf"], "dpi": 160, "histogram_bins": 40,
                     "subsampling": None, "mesh_overlay": mesh_overlay, "anatomical_axes": "unverified"},
    }
    save_json(output / "plots.json", record)
    try:
        record.update(pilot_code_provenance())
        points = pd.read_csv(run / "fb_sites_raw.csv", low_memory=False)
        probe = pd.read_csv(run / "probe_bodies.csv")
        scale = np.asarray(manifest["native_to_um_per_axis"])
        skeletons = {
            int(body): read_native_swc(run / f"skeleton_{int(body)}.swc")[0]
            for body in probe["bodyId"]
        }
        mesh = None
        if mesh_overlay:
            mesh = read_context_mesh(run / "FB.obj")
            mesh_context_ranges(points, probe, skeletons, mesh[0], scale).to_csv(
                output / "mesh_coordinate_ranges.csv", index=False
            )
            record["mesh_context"] = {
                "source_mesh_sha256": manifest["artifact_sha256"]["FB.obj"],
                "native_to_um_per_axis": scale.tolist(), "translation_um": [0., 0., 0.],
                "axis_order": XYZ,
                "coordinate_basis": "Native identity overlay supported by source provenance and coordinate ranges; no fitted registration.",
                "range_check": "Axis-aligned bounds only; not mesh containment or a column assignment.",
                "display": {"facecolor": "#aeb6bf", "alpha": .12, "edges": False,
                            "clipping_to_mesh": False, "all_sites_and_skeleton_segments": True,
                            "legend_location": "upper left"},
            }
            save_json(output / "plots.json", record)

        def save_figure(figure, stem):
            try:
                for extension in record["settings"]["formats"]:
                    figure.savefig(output / f"{stem}.{extension}", dpi=record["settings"]["dpi"])
            finally:
                plt.close(figure)

        for neuron in probe.itertuples(index=False):
            body = int(neuron.bodyId)
            save_figure(body_projection_figure(
                points.loc[points["bodyId"].eq(body)], skeletons[body], scale,
                f"{neuron.type} {body}: {neuron.instance}", mesh=mesh,
            ), f"body_{body}")
        if full_population:
            from .vdeltae_population import (
                column_profile_figure, population_figure_specs,
                population_projection_figure, population_spatial_summary,
            )
            spatial_summary = population_spatial_summary(points, probe, scale)
            spatial_summary.to_csv(output / "column_spatial_summary.csv", index=False)
            for stem, kind, bodies, compartment in population_figure_specs(points, probe):
                selected = points.loc[points["bodyId"].isin(bodies["bodyId"])]
                if kind == "histogram":
                    figure = axis_distribution_figure(selected, bodies, scale, bins=40)
                elif kind == "profiles":
                    figure = column_profile_figure(spatial_summary, bodies)
                else:
                    figure = population_projection_figure(selected, bodies, scale, mesh, compartment)
                save_figure(figure, stem)
        else:
            save_figure(axis_distribution_figure(points, probe, scale, bins=40), "axis_distributions")
        if manifest["source_resources"]["FB_mesh"]["status"] == "retrieved":
            vertices, _ = read_context_mesh(run / "FB.obj")
            save_figure(mesh_preview_figure(vertices), "FB_context")
        (output / "summary.md").write_text("\n".join([
            "# Full-population coordinate inspection" if full_population else "# Two-body coordinate inspection",
            "", markdown(probe[["bodyId", "type", "instance", "annotation_column"]]),
            "", "## Raw site categories", "", markdown(pd.read_csv(run / "site_counts.csv")),
            "", "## Interpretation boundaries", "",
            "All FB-marked sites are shown without a confidence cutoff or spatial subsampling.",
            "Individual-body colors denote dataset compartments; population-overview colors denote annotation column labels. Polarity uses separate markers/line styles.",
            ("Skeletons are coarse and unhealed. The FB mesh is drawn as faint projected triangles behind every site; no points or branches are clipped."
             if mesh_overlay else "Skeletons are coarse and unhealed. The mesh is a separate raw-coordinate vertex preview."),
            ("The native identity overlay uses the recorded scale without fitted rotation or translation. Bounding-range agreement is not containment or independently calibrated registration."
             if mesh_overlay else "No FB mesh is overlaid on the body figures."),
            "Means/quantiles describe distributions, not column positions or physiological signals.",
            "Layer fractions have group-specific denominators and retain unknown flags; layers can overlap.",
            "Column meanings, anatomical axis orientation and compartment provenance remain unresolved.",
            "Population summaries keep bodies separate, preserve repeated/edge labels, and include absent compartments with zero counts. Site quantiles are not physical column centers.",
            "", f"Descriptive CSV tables and raw sources: {run}", "",
            *( ["## Population figure index", ""] + [
                f"- [{stem}.png]({stem}.png): {bodies['type'].iloc[0]}, {kind}"
                + (f", compartment {compartment}" if compartment is not None else "")
                for stem, kind, bodies, compartment in population_figure_specs(points, probe)
            ] if full_population else [] ),
        ]))
        verify_coordinate_pilot(run)
        if sha256(run / "coordinates.json") != source_hash:
            raise ValidationError("Coordinate manifest changed during figure export.")
        record["status"] = "complete"
    except Exception as error:
        record["status"] = "failed"
        record["error_type"] = type(error).__name__
        raise
    finally:
        record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        record["artifact_sha256"] = record_artifact_hashes(output, "plots.json")
        save_json(output / "plots.json", record)
    return record


def verify_coordinate_exports(output):
    """Verify the complete figure inventory and its coordinate source without network access."""
    output = Path(output).resolve()
    record = json.loads((output / "plots.json").read_text())
    if record.get("kind") != "vdeltae_hdeltag_coordinate_figures_v1" or record.get("status") != "complete":
        raise ValidationError("Verification requires complete coordinate figures.")
    run = Path(record["source"])
    manifest = verify_coordinate_pilot(run)
    if sha256(run / "coordinates.json") != record["source_manifest_sha256"]:
        raise ValidationError("Figure source manifest changed.")
    full_population = manifest.get("full_population", False)
    if record.get("full_population", False) != full_population:
        raise ValidationError("Figure scope differs from its coordinate source.")
    stems = [f"body_{body}" for body in manifest["selected_body_ids"]]
    if full_population:
        from .vdeltae_population import population_figure_specs, population_spatial_summary
        points = pd.read_csv(run / "fb_sites_raw.csv", low_memory=False)
        probe = pd.read_csv(run / "probe_bodies.csv")
        stems.extend(spec[0] for spec in population_figure_specs(points, probe))
    else:
        stems.append("axis_distributions")
    if manifest["source_resources"]["FB_mesh"]["status"] == "retrieved":
        stems.append("FB_context")
    required = {"summary.md"} | {f"{stem}.{extension}" for stem in stems for extension in ["png", "pdf"]}
    overlay = record.get("settings", {}).get("mesh_overlay", False)
    if full_population:
        if not overlay:
            raise ValidationError("Full-population figures require mesh context.")
        required.add("column_spatial_summary.csv")
    if type(overlay) is not bool:
        raise ValidationError("Saved mesh_overlay setting must be a boolean.")
    if overlay:
        required.add("mesh_coordinate_ranges.csv")
    hashes = record.get("artifact_sha256", {})
    if set(hashes) != required:
        raise ValidationError("Figure artifact inventory differs from expected exports.")
    for name, digest in hashes.items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or not path.is_file() or sha256(path) != digest:
            raise ValidationError("Figure artifact is missing, outside the run, or changed.")
    if full_population:
        expected = population_spatial_summary(points, probe, manifest["native_to_um_per_axis"])
        pd.testing.assert_frame_equal(expected, pd.read_csv(output / "column_spatial_summary.csv"),
                                      check_dtype=False, rtol=1e-12, atol=1e-12)
    if overlay:
        if manifest["source_resources"]["FB_mesh"]["status"] != "retrieved":
            raise ValidationError("Overlay source has no retrieved mesh.")
        context = record.get("mesh_context", {})
        if (context.get("source_mesh_sha256") != manifest["artifact_sha256"]["FB.obj"]
                or context.get("native_to_um_per_axis") != manifest["native_to_um_per_axis"]
                or context.get("translation_um") != [0., 0., 0.]
                or context.get("axis_order") != XYZ):
            raise ValidationError("Mesh overlay provenance or coordinate transform differs from its source.")
        points = pd.read_csv(run / "fb_sites_raw.csv", low_memory=False)
        probe = pd.read_csv(run / "probe_bodies.csv")
        skeletons = {int(body): read_native_swc(run / f"skeleton_{int(body)}.swc")[0]
                     for body in probe["bodyId"]}
        vertices, _ = read_context_mesh(run / "FB.obj")
        expected = mesh_context_ranges(points, probe, skeletons, vertices, manifest["native_to_um_per_axis"])
        pd.testing.assert_frame_equal(
            expected, pd.read_csv(output / "mesh_coordinate_ranges.csv"), check_dtype=False,
            rtol=1e-12, atol=1e-12,
        )
    return record
