# Hand-written implementation: two-body coordinates and morphology pilot

Prepared 29 October 2026. Continue after the completed
[live spatial data probe](vdeltae_live_capability_walkthrough.md). Suggested new
module: `fly_connectome/vdeltae_spatial_pilot.py`; suggested tests:
`tests/test_vdeltae_spatial_pilot.py`. Transcribe one part at a time. No production
code or live extraction was added while preparing this guide.

## Scope and coordinate conventions

Keep vDeltaE 54907 and hDeltaG 19798. Reuse the verified saved probe's body
selection, site-count summaries and raw morphology. Do not select a new pair,
download all 26 skeletons, fit a column mapping or change neural dynamics.

The saved metadata records 8 nm voxels. The
[official MaleCNS download guide](https://male-cns.janelia.org/download/)
also shows native skeletons with 8 nm units. This initial pilot requires that
recorded configuration and converts both native synapse and SWC XYZ to µm by
multiplying by 0.008. The conversion is explicit and must not be applied to
arbitrary SWC files or transformed/template coordinates. Anatomical axis
orientation is unverified: plots use native X/Y/Z, without mirroring or rotation.

The downloaded SWCs contain coarse-generation comments and their own UUID and
mutation fields. Preserve these comments verbatim. Those identifiers describe
skeleton-generation provenance; do not automatically equate them with the
neuPrint dataset UUID/mutation markers. Successful retrieval does not establish
skeleton freshness or fine-branch completeness.

Keep the FB OBJ as separate context, initially plotted in raw OBJ units. Its
vertices are not a validated quantitative boundary. Do not use the mesh to crop
points, test containment or optimize an alignment to match an expected pattern.

The first increment below prepares point querying/validation and pure geometry
helpers. The saved extraction wrapper comes after reviewing those helpers.

## 1. New module, point query and explicit scale

```python
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
    if metadata.get("voxelUnits") != "nanometers" or metadata.get("voxelSize") != [8, 8, 8]:
        raise ValidationError("Review changed coordinate units before reusing native geometry.")
    return np.asarray(metadata["voxelSize"], dtype=float) / 1000


def category_counts(points):
    return points.groupby(CATEGORY_KEYS, dropna=False).size().rename("site_count").reset_index()


def validate_fb_points(points, probe, availability, expected_categories):
    required = {"bodyId", "synapse_node_id", "synapse_type", *XYZ,
                "confidence", "compartment", "split", "FB", *LAYERS}
    if not required.issubset(points.columns):
        raise ValidationError("FB point table is missing required columns.")
    integers(points["bodyId"], "Point body IDs", 1)
    integers(points["synapse_node_id"], "Graph synapse IDs", 0)
    if not set(points["bodyId"]).issubset(set(probe["bodyId"])):
        raise ValidationError("Point query returned an unselected body.")
    if points.duplicated(["bodyId", "synapse_node_id"]).any():
        raise ValidationError("Repeated body/synapse-node identity.")
    if points.duplicated(["bodyId", "synapse_type", *XYZ]).any():
        raise ValidationError("Repeated site coordinates within body/polarity; review raw nodes.")
    if not points["synapse_type"].isin(["pre", "post"]).all() or not points["FB"].eq(True).all():
        raise ValidationError("Unexpected site polarity or FB flag.")
    if not np.isfinite(points[XYZ].to_numpy(dtype=float)).all():
        raise ValidationError("Missing or nonfinite site coordinates.")
    confidence = pd.to_numeric(points["confidence"], errors="raise").dropna()
    if not np.isfinite(confidence).all() or not confidence.between(0, 1).all():
        raise ValidationError("Invalid nonmissing confidence score.")
    expected = availability.set_index("bodyId")["sites_marked_FB"]
    actual = points.groupby("bodyId").size().reindex(expected.index, fill_value=0)
    if not actual.eq(expected).all():
        raise ValidationError("FB point counts differ from the saved probe denominator.")
    columns = [*CATEGORY_KEYS, "site_count"]
    try:
        pd.testing.assert_frame_equal(
            category_counts(points)[columns].sort_values(CATEGORY_KEYS).reset_index(drop=True),
            expected_categories[columns].sort_values(CATEGORY_KEYS).reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError as error:
        raise ValidationError("Point categories differ from the saved unfiltered categories.") from error
```

`id(s)` is a graph-node identifier for this database snapshot, not a permanent
biological ID. Retain it with dataset identity and the natural
body/polarity/XYZ key; never join it across rebuilt databases. Repeated
coordinates trigger review rather than automatic deduplication.

No confidence threshold is applied: the denominator is the saved unfiltered
`sites_marked_FB`, currently 1,010 for 54907 and 4,047 for 19798. These totals
are observations, not hardcoded requirements. Missing confidence and unknown
compartment/split values are preserved; null groups count in reconciliation.
Layer flags remain nullable. Do not sum overlapping layers to reconstruct FB
totals. This point inventory counts sites, not synaptic contacts or partners.

The offline query tests cannot establish live Cypher syntax or server semantics.
Keep raw answers if a future live validation fails.

## 2. Read morphology without healing or dropping fields

The installed neuprint-python SWC reader discards node-type values. This reader
retains them and raw comments. Node-type codes are not automatically translated
into biological axon/dendrite labels.

```python
def read_native_swc(path):
    text = Path(path).read_text()
    comments = [line for line in text.splitlines() if line.lstrip().startswith("#")]
    lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not lines or any(len(line.split()) != 7 for line in lines):
        raise ValidationError("Expected nonempty seven-column SWC geometry.")
    table = pd.read_csv(StringIO("\n".join(lines)), sep=r"\s+", names=SWC_COLUMNS)
    for name, minimum in [("node_id", 1), ("node_type", 0), ("parent_id", -1)]:
        table[name] = integers(table[name], name, minimum)
    if (not np.isfinite(table[[*XYZ, "radius"]].to_numpy(dtype=float)).all()
            or table["radius"].lt(0).any()):
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
    return np.stack([
        children[XYZ].to_numpy(dtype=float),
        indexed.loc[children["parent_id"], XYZ].to_numpy(dtype=float),
    ], axis=1)


def read_context_mesh(path):
    vertices, faces = parse_obj(Path(path).read_bytes())
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or not np.isfinite(vertices).all()):
        raise ValidationError("Invalid or empty FB mesh vertices.")
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValidationError("Invalid or empty FB mesh faces.")
    return vertices, faces
```

Multiple SWC roots/components are retained. Cycles and missing parents fail
explicitly. There is no fragment healing, smoothing, resampling or nearest-node
attachment. Radii remain raw and are not used to assert anatomical diameters.
The reused OBJ parser is generic geometry code, not PFN-specific body selection.

## 3. Plot helpers: native projections, all points and full skeleton bounds

These helpers return figures rather than writing files. The later export wrapper
will own fresh output directories and manifests. This keeps rendering testable
without mixing it with queries or silently replacing earlier exports.

```python
def body_projection_figure(points, skeleton, scale_um, title):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    scale = np.asarray(scale_um, dtype=float)
    if scale.shape != (3,) or not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValidationError("Expected three positive finite coordinate scale factors.")
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
                    ax.add_collection(LineCollection(
                        segments[:, :, [a, b]], colors="#777777", linewidths=.4, alpha=.45,
                    ))
                for (compartment, polarity), group in points.groupby(
                    ["compartment", "synapse_type"], dropna=False
                ):
                    label = "unannotated" if pd.isna(compartment) else str(compartment)
                    coordinates = group[XYZ].to_numpy(dtype=float) * scale
                    ax.scatter(coordinates[:, a], coordinates[:, b], s=5,
                               marker="^" if polarity == "pre" else ".",
                               color=colors.get(label, "#a855f7"), alpha=.45,
                               label=f"{label} / {polarity} (n={len(group)})", rasterized=True)
                for axis, setter in [(a, ax.set_xlim), (b, ax.set_ylim)]:
                    low, high = context[:, axis].min(), context[:, axis].max()
                    padding = max((high - low) * .05, .1)
                    setter(low - padding, high + padding)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlabel(f"Native {XYZ[a]} (µm; orientation unverified)")
                ax.set_ylabel(f"Native {XYZ[b]} (µm)")
                ax.set_title("FB sites" if row == 0 else "Same sites + coarse unhealed skeleton")
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(title + "\nDataset compartment colors; separate polarity markers; columns unresolved.")
        return fig
    except Exception:
        plt.close(fig)
        raise


def mesh_preview_figure(vertices):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    try:
        for ax, (a, b) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
            ax.scatter(vertices[:, a], vertices[:, b], s=.2, color="#777777", alpha=.2, rasterized=True)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(f"OBJ {XYZ[a]} (raw units)")
            ax.set_ylabel(f"OBJ {XYZ[b]} (raw units)")
        fig.suptitle("FB mesh vertex preview only; no containment or registration inferred")
        return fig
    except Exception:
        plt.close(fig)
        raise
```

Pass one body's validated points and parsed skeleton at a time. Plot colors
describe source `compartment` labels; they are not a physiological sign or a
newly established arbor classification. Markers distinguish pre/post sites.
Unknown compartments remain visible. All points are shown without spatial
subsampling. Top-row panels omit the skeleton to reduce occlusion; the bottom
adds it. Bounds include the whole skeleton and all plotted sites, without an FB
mesh crop or automatic centering. Empty site clouds are permitted if the saved
denominator is zero; they still show the skeleton.

The caller closes every returned figure with `plt.close(fig)`, including after
save failures. Source point tables, skeleton tables and mesh arrays are never
modified by these helpers. No neuron-to-neuron registration is fitted.

## 4. First tests to transcribe

Start the new test file with this block. Configure the noninteractive backend
before importing pyplot, as the earlier plotting workflows do.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from fly_connectome.connectivity import ValidationError
from fly_connectome.vdeltae_spatial_pilot import (
    XYZ, LAYERS, fb_site_query, native_scale_um, read_native_swc,
    skeleton_segments, read_context_mesh, validate_fb_points, category_counts,
    body_projection_figure, mesh_preview_figure,
)


def test_native_units_require_explicit_matching_metadata():
    scale = native_scale_um({"voxelSize": [8, 8, 8], "voxelUnits": "nanometers"})
    np.testing.assert_allclose(np.array([[1000, 2000, 3000]]) * scale, [[8, 16, 24]])
    with pytest.raises(ValidationError):
        native_scale_um({"voxelSize": [8, 8, 8], "voxelUnits": "micrometers"})


def test_swc_retains_types_comments_and_multiple_components(tmp_path):
    path = tmp_path / "geometry.swc"
    path.write_text("# coarse geometry\n1 0 0 0 0 1 -1\n2 2 8 0 0 1 1\n3 3 0 8 0 1 -1\n")
    table, comments = read_native_swc(path)
    assert comments == ["# coarse geometry"]
    assert table["node_type"].tolist() == [0, 2, 3]
    assert table["parent_id"].eq(-1).sum() == 2
    np.testing.assert_array_equal(skeleton_segments(table), [[[8, 0, 0], [0, 0, 0]]])


@pytest.mark.parametrize("text", [
    "1 0 0 0 0 1 2\n2 0 1 0 0 1 1\n",  # cycle
    "1 0 0 0 0 1 5\n",                    # absent parent
    "1 0 0 0 0 1 -1\n1 0 1 0 0 1 -1\n", # duplicate ID
    "1 0 nan 0 0 1 -1\n",                 # nonfinite position
    "1 0 0 0 0 -1 -1\n",                  # negative radius
])
def test_invalid_skeletons_fail_without_healing(tmp_path, text):
    path = tmp_path / "bad.swc"
    path.write_text(text)
    with pytest.raises(ValidationError):
        read_native_swc(path)


def test_mesh_faces_are_validated(tmp_path):
    path = tmp_path / "FB.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    vertices, faces = read_context_mesh(path)
    np.testing.assert_array_equal(faces, [[0, 1, 2]])
    path.write_text("v 0 0 0\nf 1 2 3\n")
    with pytest.raises(ValidationError):
        read_context_mesh(path)


def test_query_retains_raw_site_information_without_a_confidence_filter():
    query = fb_site_query([54907, 19798])
    assert "n.bodyId IN [19798, 54907]" in query
    assert "id(s) AS synapse_node_id" in query and "WHERE s.FB = true" in query
    assert "WITH DISTINCT n, s" in query
    assert "s.compartment AS compartment" in query and "s.split AS split" in query
    assert "coalesce" not in query and "s.confidence >" not in query
    assert "SynapsesTo" not in query and "ConnectsTo" not in query


@pytest.fixture
def point_example():
    points = pd.DataFrame({
        "bodyId": [10, 1], "synapse_node_id": [100, 200], "synapse_type": ["pre", "post"],
        "x": [8., 16.], "y": [0., 8.], "z": [0., 0.],
        "confidence": [.9, np.nan], "compartment": ["axon", None], "split": [None, None],
        "FB": [True, True], **{layer: [None, None] for layer in LAYERS},
    })
    probe = pd.DataFrame({"bodyId": [10, 1]})
    availability = pd.DataFrame({"bodyId": [10, 1], "sites_marked_FB": [1, 1]})
    return points, probe, availability, category_counts(points)


def test_points_reconcile_with_unknown_labels_and_missing_confidence(point_example):
    points, probe, availability, categories = point_example
    before = points.copy(deep=True)
    validate_fb_points(points, probe, availability, categories)
    pd.testing.assert_frame_equal(points, before)


@pytest.mark.parametrize("problem", ["body", "node_id", "coordinate", "count"])
def test_point_failures_are_not_silently_filtered(point_example, problem):
    points, probe, availability, categories = point_example
    if problem == "body":
        points.loc[0, "bodyId"] = 999
    elif problem == "node_id":
        points = pd.concat([points, points.iloc[[0]]], ignore_index=True)
    elif problem == "coordinate":
        points.loc[0, "x"] = np.inf
    else:
        availability.loc[0, "sites_marked_FB"] = 2
    with pytest.raises(ValidationError):
        validate_fb_points(points, probe, availability, categories)


def test_offline_figures_preserve_points_and_render(tmp_path, point_example):
    points = point_example[0].iloc[[0]].copy()
    before = points.copy(deep=True)
    path = tmp_path / "body.swc"
    path.write_text("1 0 0 0 0 1 -1\n2 0 16 16 0 1 1\n")
    skeleton, _ = read_native_swc(path)
    figures = [
        body_projection_figure(points, skeleton, [.008] * 3, "Synthetic body"),
        mesh_preview_figure(np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])),
    ]
    try:
        assert len(figures[0].axes) == 6
        assert len(figures[1].axes) == 3
        for index, figure in enumerate(figures):
            destination = tmp_path / f"figure_{index}.png"
            figure.savefig(destination)
            assert destination.stat().st_size > 0
        pd.testing.assert_frame_equal(points, before)
    finally:
        for figure in figures:
            plt.close(figure)
```

There are 15 cases. Run after transcription:

```bash
python -m pytest -q tests/test_vdeltae_spatial_pilot.py
```

## 5. Inspect the cached geometry and query without a live request

After writing the preceding pieces, paste this complete block into Bash from
the repository root. It verifies the completed source run, parses the two
skeletons and mesh, reports native bounds and prints the proposed query. It
does not query neuPrint or write new outputs.

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
from fly_connectome.vdeltae_spatial_audit import verify_spatial_data_probe
from fly_connectome.vdeltae_spatial_pilot import XYZ, fb_site_query, read_native_swc, read_context_mesh

source = Path("results/vdeltae-hdeltag-spatial-capability-20261029")
manifest = verify_spatial_data_probe(source)
probe = pd.read_csv(source / "probe_bodies.csv")
for body in probe["bodyId"]:
    resource = manifest["resources"][f"skeleton_{int(body)}"]
    if resource["status"] != "retrieved":
        raise RuntimeError("Required skeleton was not retrieved; inspect source status.")
    skeleton, comments = read_native_swc(source / resource["file"])
    print("\nBody:", body, "nodes:", len(skeleton), "roots:", skeleton["parent_id"].eq(-1).sum())
    print(skeleton[XYZ].agg(["min", "max"]).to_string())
    print("Preserved header lines:", len(comments))
mesh = manifest["resources"]["FB_mesh"]
if mesh["status"] == "retrieved":
    vertices, faces = read_context_mesh(source / mesh["file"])
    print("\nFB mesh vertices:", len(vertices), "triangles:", len(faces))
print("\nProposed FB site query:\n", fb_site_query(probe["bodyId"]))
verify_spatial_data_probe(source)
PY
```

## 6. Next increment: saved extraction and export wrappers

Do this after reviewing the pure helpers and tests. Use separate fresh
directories for the coordinate run and PNG/PDF exports. The next wrapper should:

1. Verify the saved probe; record its manifest hash, selected bodies, source
   geometry hashes, query text, unit basis, code/package identity and settings.
   Refuse existing output directories and overlap with any source directory.
2. Use the existing authentication pattern, with safe error reporting. Check
   the actual client target and live metadata against the saved source snapshot
   before querying. Do not relax dataset/coordinate checks to bypass drift.
3. Save the returned raw point table before validation. Reconcile body/polarity/
   compartment/split counts with the saved unfiltered FB categories. Preserve
   failed raw results with explicit failed status; never discard inconvenient
   sites to make counts agree.
4. Copy the verified raw SWCs, headers and optional OBJ into the new run and
   save parsed skeleton tables and site-count summaries. Record per-body XYZ
   ranges for skeleton nodes and sites, in raw coordinates and explicit µm.
   Preserve unknown values and all original resource provenance.
5. Recheck live metadata and saved source hashes; finalize complete/failed
   status and artifact hashes. Add offline verification that reconstructs the
   key count checks and reads/parses all geometry without contacting the server.
6. Render one six-panel body figure per neuron and a separate mesh preview.
   Save figures to a new export directory with their own running/complete/failed
   manifest, source and code hashes, plot settings, and output hashes. Recheck
   source files across rendering. Do not replace previous presentation exports.

Test the wrappers with a fake client, including source drift, malformed points,
count mismatch, invalid morphology, overwrite refusal, output tampering and an
offline render round trip. The offline tests here do not prove that the new
point query runs on the live server; retain its first integration result.

After those checks, run the two-body extraction and inspect both figures and
coordinate ranges. Gross scale/translation mismatches are reasons to revisit
provenance, not reasons to optimize a transform. Coarse skeletons can omit fine
arbors, so local offsets do not automatically invalidate site coordinates.

Only after this pilot works should the validated extraction expand to all 26
bodies. Two examples cannot establish population-wide C ordering, boundary-label
meaning, periodicity, a physiological sign or a home vector.
