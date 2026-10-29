from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from .connectivity import ValidationError
from .inputs import integers
from .spatial import parse_obj
from .vdeltae_spatial_audit import spatial_probe_queries

XYZ = ["x", "y", "z"]
LAYERS = [f"FBl{layer}" for layer in range(1, 10)]
SWC_COLUMNS = ["node_id", "node_type", "x", "y", "z", "radius", "parent_id"]
CATEGORY_KEYS = ["bodyId", "synapse_type", "compartment", "split"]


def fb_site_query(body_ids):
    body_ids = list(body_ids)
    spatial_probe_queries(body_ids)  # Reuse the existing two-ID validation.
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
        or faces.min() < 0
        or faces.max() >= len(vertices)
    ):
        raise ValidationError("Invalid or empty FB mesh faces.")
    return vertices, faces


def body_projection_figure(points, skeleton, scale_um, title):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    scale = np.asarray(scale_um, dtype=float)
    if scale.shape != (3,) or not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValidationError(
            "Expected three positive finite coordinate scale factors."
        )
    segments = skeleton_segments(skeleton) * scale
    context = skeleton[XYZ].to_numpy(dtype=float) * scale
    if len(points):
        context = np.concatenate([context, points[XYZ].to_numpy(dtype=float) * scale])
    views = [(0, 1), (0, 2), (1, 2)]
    colors = {"axon": "#d97706", "dendrite": "#0891b2", "linker": "#6b7280"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), layout="constrained")
    try:
        for column, (a, b) in enumerate(views):
            for row in [0, 1]:
                ax = axes[row, column]
                if row == 1:
                    ax.add_collection(
                        LineCollection(
                            segments[:, :, [a, b]],
                            colors="#777777",
                            linewidths=0.4,
                            alpha=0.45,
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
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(
            title
            + "\nDataset compartment colors; separate polarity markers; columns unresolved."
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
