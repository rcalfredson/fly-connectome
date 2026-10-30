"""Offline matrix and contact-location views of a verified directed contact audit."""

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .connectivity import ValidationError
from .inputs import save_json, sha256
from .vdeltae_contact_audit import load_sources, pair_counts, standard_mask, verify_contact_audit
from .vdeltae_spatial_pilot import (
    XYZ, pilot_code_provenance, read_context_mesh, read_native_swc,
    record_artifact_hashes, skeleton_segments,
)


def coupling_figure(raw, bodies, threshold):
    import matplotlib.pyplot as plt

    sources = bodies.loc[bodies.type.eq("vDeltaE")].sort_values(["annotation_column", "bodyId"])
    receivers = bodies.loc[bodies.type.eq("hDeltaG")].sort_values(["annotation_column", "bodyId"])
    standard = raw.loc[standard_mask(raw, threshold)]
    selections = [
        ("All reported, all ROIs", raw), ("Standard confidence, all ROIs", standard),
        ("Standard, FB-post → axon", standard.loc[standard.FB_post.eq(True) & standard.compartment_post.eq("axon")]),
        ("Standard, FB-post → dendrite", standard.loc[standard.FB_post.eq(True) & standard.compartment_post.eq("dendrite")]),
    ]
    matrices = [pair_counts(selected, bodies).unstack("bodyId_post").reindex(
        index=sources.bodyId, columns=receivers.bodyId) for _, selected in selections]
    maximum = max(1, max(int(matrix.to_numpy().max()) for matrix in matrices))
    figure, axes = plt.subplots(2, 2, figsize=(13, 13), layout="constrained")
    try:
        for ax, (title, selected), matrix in zip(axes.flat, selections, matrices):
            artist = ax.imshow(matrix, vmin=0, vmax=maximum, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(receivers)), [f"C{int(r.annotation_column) if pd.notna(r.annotation_column) else '?'}\n{r.bodyId}" for r in receivers.itertuples()], fontsize=8)
            ax.set_yticks(range(len(sources)), [f"C{int(r.annotation_column) if pd.notna(r.annotation_column) else '?'} / {r.bodyId}" for r in sources.itertuples()], fontsize=8)
            ax.set_title(f"{title}: {len(selected):,} contacts")
            ax.set_xlabel("hDeltaG receiving body / annotation label")
            ax.set_ylabel("vDeltaE source body / annotation label")
            figure.colorbar(artist, ax=ax, label="Contacts (shared scale)", shrink=.7)
        figure.suptitle("Directed anatomical counts; repeated C labels remain separate bodies."
                         "\nNo normalization, synaptic signs or shared angular grid. Other/missing compartments remain in totals and tables.")
        return figure
    except Exception:
        plt.close(figure)
        raise


def source_compartment_matrices(raw, bodies, threshold):
    """Standard FB-post contacts, split by SOURCE compartment; receivers unrestricted."""
    return _compartment_matrices(raw, bodies, threshold, "pre")


def receiving_compartment_matrices(raw, bodies, threshold):
    """Standard FB-post contacts, split by RECEIVING compartment; sources unrestricted."""
    return _compartment_matrices(raw, bodies, threshold, "post")


def _compartment_matrices(raw, bodies, threshold, side):
    selected = raw.loc[standard_mask(raw, threshold) & raw.FB_post.eq(True)]
    sources = bodies.loc[bodies.type.eq("vDeltaE")].sort_values(["annotation_column", "bodyId"])
    receivers = bodies.loc[bodies.type.eq("hDeltaG")].sort_values(["annotation_column", "bodyId"])
    tables = {}
    for compartment in ["axon", "dendrite"]:
        group = selected.loc[selected[f"compartment_{side}"].eq(compartment)]
        matrix = pair_counts(group, bodies).unstack("bodyId_post").reindex(
            index=sources.bodyId, columns=receivers.bodyId
        ).rename_axis(index="bodyId_pre", columns="bodyId_post").reset_index()
        matrix.columns = matrix.columns.map(str)
        matrix.columns.name = None
        prefix = "source" if side == "pre" else "receiving"
        tables[f"{prefix}_{compartment}_matrix.csv"] = matrix
    return tables


def source_compartment_figure(raw, bodies, threshold):
    return _compartment_figure(raw, bodies, threshold, "pre")


def receiving_compartment_figure(raw, bodies, threshold):
    """The lower two original coupling panels, side by side with one colorbar."""
    return _compartment_figure(raw, bodies, threshold, "post")


def _compartment_figure(raw, bodies, threshold, side):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    sources = bodies.loc[bodies.type.eq("vDeltaE")].sort_values(["annotation_column", "bodyId"])
    receivers = bodies.loc[bodies.type.eq("hDeltaG")].sort_values(["annotation_column", "bodyId"])
    matrices = [table.set_index("bodyId_pre") for table in _compartment_matrices(raw, bodies, threshold, side).values()]
    norm = Normalize(vmin=0, vmax=max(1, max(int(m.to_numpy().max()) for m in matrices)))
    selected = raw.loc[standard_mask(raw, threshold) & raw.FB_post.eq(True)]
    other = int((~selected[f"compartment_{side}"].isin(["axon", "dendrite"])).sum())
    figure, axes = plt.subplots(1, 2, figsize=(14, 8), layout="constrained")
    try:
        for ax, compartment, matrix in zip(axes, ["axon", "dendrite"], matrices):
            artist = ax.imshow(matrix, norm=norm, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(receivers)), [f"C{int(r.annotation_column) if pd.notna(r.annotation_column) else '?'}\n{r.bodyId}" for r in receivers.itertuples()], fontsize=8)
            ax.set_yticks(range(len(sources)), [f"C{int(r.annotation_column) if pd.notna(r.annotation_column) else '?'} / {r.bodyId}" for r in sources.itertuples()], fontsize=8)
            ax.set_xlabel("hDeltaG receiving body / annotation label")
            title = (f"vDeltaE {compartment} sites → hDeltaG" if side == "pre"
                     else f"vDeltaE → hDeltaG {compartment} sites")
            ax.set_title(f"{title}: {int(matrix.to_numpy().sum()):,} contacts")
        figure.supylabel("vDeltaE source body / annotation label")
        figure.colorbar(artist, ax=list(axes), label="Contacts (one shared scale)", shrink=.75)
        if side == "pre":
            figure.suptitle("Source-compartment comparison: standard-confidence, FB-marked receiving endpoints"
                             "\nAll hDeltaG receiving compartments included; raw counts, no normalization."
                             + (f"\nOther/missing source compartments: {other} contacts (retained in the audit)." if other else ""))
        else:
            figure.suptitle("Receiving hDeltaG compartments: standard-confidence, FB-marked receiving endpoints"
                             "\nAll vDeltaE source compartments included; raw counts, no normalization."
                             + (f"\nOther/missing receiving compartments: {other} contacts (retained in the audit)." if other else ""))
        return figure
    except Exception:
        plt.close(figure)
        raise


def receiver_contact_figure(selected, receiver, bodies, context, skeleton, mesh, scale_um):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.lines import Line2D

    scale = np.asarray(scale_um)
    vertices, faces = mesh
    triangles = (vertices * scale)[faces]
    skeleton_xyz = skeleton[XYZ].to_numpy(dtype=float) * scale
    context_xyz = context[XYZ].to_numpy(dtype=float) * scale
    xyz = selected[[f"{axis}_post" for axis in XYZ]].to_numpy(dtype=float) * scale
    bounds = np.concatenate([vertices * scale, skeleton_xyz, context_xyz, xyz])
    columns = sorted(int(v) for v in bodies.loc[bodies.type.eq("vDeltaE"), "annotation_column"].dropna().unique())
    colors = {column: plt.get_cmap("tab20")(index % 20) for index, column in enumerate(columns)}
    segments = skeleton_segments(skeleton) * scale
    figure, axes = plt.subplots(1, 3, figsize=(16, 6), layout="constrained")
    try:
        for ax, (a, b) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
            ax.add_collection(PolyCollection(triangles[:, :, [a, b]], facecolors="#aeb6bf",
                                            edgecolors="none", alpha=.12, rasterized=True, zorder=0))
            ax.add_collection(LineCollection(segments[:, :, [a, b]], colors="#777777", linewidths=.4, alpha=.35, zorder=1))
            ax.scatter(context_xyz[:, a], context_xyz[:, b], color="#999999", s=2, alpha=.12, rasterized=True, zorder=2)
            for (column, compartment), group in selected.groupby(["annotation_column_pre", "compartment_post"], dropna=False):
                values = group[[f"{axis}_post" for axis in XYZ]].to_numpy(dtype=float) * scale
                color = "#666666" if pd.isna(column) else colors[int(column)]
                marker = "o" if compartment == "dendrite" else "^" if compartment == "axon" else "x"
                ax.scatter(values[:, a], values[:, b], color=color, marker=marker, s=9,
                           alpha=.65, linewidths=.5, rasterized=True, zorder=3)
            for index, setter in [(a, ax.set_xlim), (b, ax.set_ylim)]:
                low, high = bounds[:, index].min(), bounds[:, index].max()
                margin = max(.01, (high - low) * .04)
                setter(low - margin, high + margin)
            ax.set_xlabel(f"native {XYZ[a]} (µm)")
            ax.set_ylabel(f"native {XYZ[b]} (µm)")
            ax.set_aspect("equal", adjustable="box")
        handles = []
        for column, group in selected.groupby("annotation_column_pre", dropna=False):
            color = "#666666" if pd.isna(column) else colors[int(column)]
            label = "Unresolved C" if pd.isna(column) else f"vDeltaE C{int(column)}"
            handles.append(Line2D([], [], marker="o", linestyle="", color=color,
                                  label=f"{label}: {len(group)} contacts"))
        if handles:
            axes[-1].legend(handles=handles, loc="upper left", fontsize=7)
        unique = selected[["bodyId_post", "synapse_node_id_post"]].drop_duplicates().shape[0]
        figure.suptitle(f"hDeltaG {receiver.bodyId} ({receiver.instance}): standard-confidence FB-post contacts"
                         f"\n{len(selected)} contacts / {unique} unique receiving sites. Color = vDeltaE C annotation."
                         "\nReceiving compartment: dendrite = circle, axon = triangle, other/missing = cross; gray = all receiver FB sites and full skeleton.")
        return figure
    except Exception:
        plt.close(figure)
        raise


def export_contact_figures(run, output, *, source_compartment_panels=False, receiving_compartment_panels=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if type(source_compartment_panels) is not bool:
        raise ValidationError("source_compartment_panels must be a boolean.")
    if type(receiving_compartment_panels) is not bool:
        raise ValidationError("receiving_compartment_panels must be a boolean.")
    run, output = Path(run).resolve(), Path(output).resolve()
    source_hash = sha256(run / "contacts.json")
    audit = verify_contact_audit(run)
    bodies, points, _, geometry, _ = load_sources(audit["sources"]["coordinates"], audit["sources"]["census"])
    protected = [run, Path(audit["sources"]["coordinates"]), Path(audit["sources"]["census"]),
                 Path(geometry["source_probe"]), Path(geometry["source_inventory"]["annotation_audit_directory"])] + [
        Path(source["source_manifest_file"]).parent for source in geometry["source_inventory"]["sources"].values()
    ]
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in protected):
        raise ValidationError("Contact figures must be separate from all sources.")
    coords = Path(audit["sources"]["coordinates"])
    if geometry["source_resources"]["FB_mesh"]["status"] != "retrieved":
        raise ValidationError("Contact overlays require a verified FB mesh.")
    output.mkdir(parents=True, exist_ok=False)
    record = {"kind": "vdeltae_hdeltag_contact_figures_v1", "status": "running",
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "source": str(run),
              "source_manifest_sha256": source_hash, "native_to_um_per_axis": geometry["native_to_um_per_axis"],
              "source_compartment_panels": source_compartment_panels,
              "receiving_compartment_panels": receiving_compartment_panels,
              "overlay_scope": "standard__post_FB", "translation_um": [0., 0., 0.],
              "axis_order": XYZ, "subsampling": None}
    save_json(output / "plots.json", record)
    try:
        record.update(pilot_code_provenance())
        mesh = read_context_mesh(coords / "FB.obj")
        raw = pd.read_csv(run / "contacts_annotated.csv", low_memory=False)
        threshold = audit["standard_confidence_threshold"]

        def save(figure, stem):
            try:
                for suffix in ["png", "pdf"]:
                    figure.savefig(output / f"{stem}.{suffix}", dpi=160)
            finally:
                plt.close(figure)

        save(coupling_figure(raw, bodies, threshold), "body_pair_coupling")
        if source_compartment_panels:
            save(source_compartment_figure(raw, bodies, threshold), "source_compartment_coupling")
            for name, table in source_compartment_matrices(raw, bodies, threshold).items():
                table.to_csv(output / name, index=False)
        if receiving_compartment_panels:
            save(receiving_compartment_figure(raw, bodies, threshold), "receiving_compartment_coupling")
            for name, table in receiving_compartment_matrices(raw, bodies, threshold).items():
                table.to_csv(output / name, index=False)
        selected = raw.loc[standard_mask(raw, threshold) & raw.FB_post.eq(True)]
        receivers = bodies.loc[bodies.type.eq("hDeltaG")].sort_values(["annotation_column", "bodyId"])
        links = []
        for receiver in receivers.itertuples(index=False):
            body = int(receiver.bodyId)
            skeleton, _ = read_native_swc(coords / f"skeleton_{body}.swc")
            save(receiver_contact_figure(selected.loc[selected.bodyId_post.eq(body)], receiver, bodies,
                                         points.loc[points.bodyId.eq(body)], skeleton, mesh,
                                         geometry["native_to_um_per_axis"]), f"receiver_{body}")
            links.append(f"- [{receiver.instance}, body {body}](receiver_{body}.png)")
        (output / "summary.md").write_text("\n".join([
            "# Directed contact views", "", "[Body-pair count comparison](body_pair_coupling.png)", "", *links, "",
            *( ["[Source axon/dendrite comparison, shared scale](source_compartment_coupling.png)",
                "Source comparison filters compartment_pre; all receiving compartments are included. Matrix CSVs retain every body pair.", ""]
               if source_compartment_panels else [] ),
            *( ["[Receiving hDeltaG axon/dendrite comparison, shared scale](receiving_compartment_coupling.png)",
                "Receiving comparison filters compartment_post; all source compartments are included. Matrix CSVs retain every body pair.", ""]
               if receiving_compartment_panels else [] ),
            "Overlays show standard-confidence FB postsynaptic contact positions; both endpoints remain in the raw audit.",
            "Gray context includes all receiver FB sites and its full unhealed skeleton. No fitted transform or clipping.",
            "Repeated contacts at a shared site can overlap; plotted contact counts are not unique-site counts.",
            "Colors group source C labels for display only; matrices retain both C0 bodies and all repeated labels separately.",
            "Missing/other receiving compartments use crosses and remain separate in the saved audit tables.",
            "Native axes and source compartments do not establish shared angular columns, signs or neural dynamics.", "",
        ]))
        verify_contact_audit(run)
        if sha256(run / "contacts.json") != source_hash:
            raise ValidationError("Contact source changed during figure export.")
        record["status"] = "complete"
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        record["artifact_sha256"] = record_artifact_hashes(output, "plots.json")
        save_json(output / "plots.json", record)
    return record


def verify_contact_figures(output):
    output = Path(output).resolve()
    record = json.loads((output / "plots.json").read_text())
    if record.get("kind") != "vdeltae_hdeltag_contact_figures_v1" or record.get("status") != "complete":
        raise ValidationError("Verification requires complete contact figures.")
    run = Path(record["source"])
    audit = verify_contact_audit(run)
    bodies, _, _, geometry, _ = load_sources(audit["sources"]["coordinates"], audit["sources"]["census"])
    if sha256(run / "contacts.json") != record["source_manifest_sha256"]:
        raise ValidationError("Contact figure source changed.")
    if (record.get("native_to_um_per_axis") != geometry["native_to_um_per_axis"]
            or record.get("translation_um") != [0., 0., 0.] or record.get("axis_order") != XYZ
            or record.get("overlay_scope") != "standard__post_FB" or record.get("subsampling") is not None):
        raise ValidationError("Contact figure transform or scope differs from the declared design.")
    stems = ["body_pair_coupling", *[f"receiver_{int(body)}" for body in bodies.loc[bodies.type.eq("hDeltaG"), "bodyId"]]]
    required = {"summary.md", *[f"{stem}.{suffix}" for stem in stems for suffix in ["png", "pdf"]]}
    source_panels = record.get("source_compartment_panels", False)
    if type(source_panels) is not bool:
        raise ValidationError("Saved source_compartment_panels must be a boolean.")
    if source_panels:
        required.update({"source_compartment_coupling.png", "source_compartment_coupling.pdf",
                         "source_axon_matrix.csv", "source_dendrite_matrix.csv"})
    receiving_panels = record.get("receiving_compartment_panels", False)
    if type(receiving_panels) is not bool:
        raise ValidationError("Saved receiving_compartment_panels must be a boolean.")
    if receiving_panels:
        required.update({"receiving_compartment_coupling.png", "receiving_compartment_coupling.pdf",
                         "receiving_axon_matrix.csv", "receiving_dendrite_matrix.csv"})
    if set(record.get("artifact_sha256", {})) != required:
        raise ValidationError("Contact figure artifact inventory differs from expected outputs.")
    for name, digest in record["artifact_sha256"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or not path.is_file() or sha256(path) != digest:
            raise ValidationError("Contact figure artifact is missing, outside the run, or changed.")
    if source_panels:
        raw = pd.read_csv(run / "contacts_raw.csv", low_memory=False)
        for name, expected in source_compartment_matrices(raw, bodies, audit["standard_confidence_threshold"]).items():
            pd.testing.assert_frame_equal(expected, pd.read_csv(output / name), check_dtype=False)
    if receiving_panels:
        raw = pd.read_csv(run / "contacts_raw.csv", low_memory=False)
        for name, expected in receiving_compartment_matrices(raw, bodies, audit["standard_confidence_threshold"]).items():
            pd.testing.assert_frame_equal(expected, pd.read_csv(output / name), check_dtype=False)
    return record
