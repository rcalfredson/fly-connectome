"""Descriptive population views; annotation labels are not physical column assignments."""

import numpy as np
import pandas as pd

from .vdeltae_spatial_pilot import XYZ, validate_context_mesh


def population_spatial_summary(points, bodies, scale_um):
    """One row per body/compartment, including absent and unannotated compartments.

    Site-weighted distributions describe arbors, not estimated column centers.
    No bodies are pooled together just because they share an annotation label.
    """
    rows = []
    labels = points["compartment"].fillna("unannotated")
    compartments = sorted({"axon", "dendrite", "linker", "unannotated", *labels.unique()})
    scale = np.asarray(scale_um)
    for body in bodies.itertuples(index=False):
        selected = points["bodyId"].eq(body.bodyId)
        denominator = int(selected.sum())
        for compartment in compartments:
            group = points.loc[selected & labels.eq(compartment)]
            row = {
                "bodyId": int(body.bodyId), "type": body.type, "instance": body.instance,
                "annotation_column": body.annotation_column, "compartment": compartment,
                "site_count": len(group), "body_FB_sites": denominator,
                "fraction_of_body_FB_sites": len(group) / denominator if denominator else np.nan,
                "pre_sites": int(group["synapse_type"].eq("pre").sum()),
                "post_sites": int(group["synapse_type"].eq("post").sum()),
            }
            for index, axis in enumerate(XYZ):
                values = group[axis].to_numpy(dtype=float) * scale[index]
                for name, quantile in [("q10", .1), ("median", .5), ("q90", .9)]:
                    row[f"{axis}_{name}_um"] = float(np.quantile(values, quantile)) if len(values) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def population_projection_figure(points, bodies, scale_um, mesh, compartment):
    """All sites in one compartment, colored by unchanged source column labels."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D

    vertices, faces = mesh
    validate_context_mesh(vertices, faces)
    scale = np.asarray(scale_um)
    vertices_um = vertices * scale
    triangles = vertices_um[faces]
    labels = points["compartment"].fillna("unannotated")
    selected = points.loc[labels.eq(compartment)].merge(
        bodies[["bodyId", "annotation_column"]], on="bodyId", validate="many_to_one"
    )
    columns = sorted(int(value) for value in bodies["annotation_column"].dropna().unique())
    colors = {column: plt.get_cmap("tab20")(index % 20) for index, column in enumerate(columns)}
    context = np.concatenate([vertices_um, points[XYZ].to_numpy(dtype=float) * scale])
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    try:
        for ax, (a, b) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
            ax.add_collection(PolyCollection(triangles[:, :, [a, b]], facecolors="#aeb6bf",
                                            alpha=.12, edgecolors="none", rasterized=True, zorder=0))
            for column, group in selected.groupby("annotation_column", dropna=False):
                color = "#666666" if pd.isna(column) else colors[int(column)]
                for polarity, subset in group.groupby("synapse_type"):
                    xyz = subset[XYZ].to_numpy(dtype=float) * scale
                    ax.scatter(xyz[:, a], xyz[:, b], s=2, alpha=.35,
                               marker="o" if polarity == "pre" else "x", color=color,
                               linewidths=.3, rasterized=True, zorder=2)
            for index, setter in [(a, ax.set_xlim), (b, ax.set_ylim)]:
                low, high = context[:, index].min(), context[:, index].max()
                margin = max((high - low) * .04, .01)
                setter(low - margin, high + margin)
            ax.set_xlabel(f"native {XYZ[a]} (µm)")
            ax.set_ylabel(f"native {XYZ[b]} (µm)")
            ax.set_aspect("equal", adjustable="box")
        handles = []
        for column in columns:
            count = int(bodies["annotation_column"].eq(column).sum())
            handles.append(Line2D([], [], marker="o", linestyle="", color=colors[column],
                                  label=f"C{column}: {count} {'body' if count == 1 else 'bodies'}"))
        if bodies["annotation_column"].isna().any():
            handles.append(Line2D([], [], marker="o", linestyle="", color="#666666", label="Unresolved label"))
        axes[-1].legend(handles=handles, loc="upper left", fontsize=7)
        figure.suptitle(f"{bodies['type'].iloc[0]}: {compartment}, {len(selected):,} FB sites"
                         "\nColor = annotation label; pre = circle, post = cross. Native axes; no column mapping inferred.")
        return figure
    except Exception:
        plt.close(figure)
        raise


def population_figure_specs(points, bodies):
    """Deterministic pages: six bodies per histogram, one overview per compartment."""
    for population in ["vDeltaE", "hDeltaG"]:
        group = bodies.loc[bodies["type"].eq(population)].sort_values(
            ["annotation_column", "bodyId"], na_position="last"
        )
        for start in range(0, len(group), 6):
            yield f"{population}_axis_distributions_{start // 6 + 1}", "histogram", group.iloc[start:start + 6], None
        present = points.loc[points["bodyId"].isin(group["bodyId"]), "compartment"].fillna("unannotated")
        for index, compartment in enumerate(sorted({"axon", "dendrite", *present.unique()})):
            # Numeric suffix avoids embedding unchecked source strings in paths.
            yield f"{population}_compartment_{index + 1}", "overview", group, compartment
        yield f"{population}_column_profiles", "profiles", group, None


def column_profile_figure(summary, bodies):
    """Compare axonal/dendritic site distributions per body without fitting a grid."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(14, max(6, .35 * len(bodies))),
                                squeeze=False, layout="constrained")
    try:
        for ax, axis in zip(axes[0], XYZ):
            for compartment, color, offset in [("axon", "#d97706", -.12), ("dendrite", "#0891b2", .12)]:
                rows = summary.loc[summary.compartment.eq(compartment)].set_index("bodyId").reindex(bodies.bodyId)
                medians = rows[f"{axis}_median_um"].to_numpy(dtype=float)
                low = rows[f"{axis}_q10_um"].to_numpy(dtype=float)
                high = rows[f"{axis}_q90_um"].to_numpy(dtype=float)
                ax.errorbar(medians, np.arange(len(bodies)) + offset,
                            xerr=np.array([medians - low, high - medians]),
                            fmt="o", markersize=3, linewidth=1, color=color, label=compartment)
            ax.set_yticks(np.arange(len(bodies)), [
                f"C{int(row.annotation_column) if pd.notna(row.annotation_column) else '?'} / {row.bodyId}"
                for row in bodies.itertuples()
            ])
            ax.invert_yaxis()
            ax.set_xlabel(f"native {axis} (µm): median and 10–90% site range")
            ax.grid(axis="x", alpha=.2)
        axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, fontsize=8)
        figure.suptitle(f"{bodies['type'].iloc[0]}: each row is a body, ordered by annotation label"
                         "\nSite distributions, not estimated column centers; absent compartments have no marker.")
        return figure
    except Exception:
        plt.close(figure)
        raise
