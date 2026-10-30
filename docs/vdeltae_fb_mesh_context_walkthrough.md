# Review walkthrough: cells and synapse sites with FB mesh context

Implemented 30 October 2026 in `fly_connectome/vdeltae_spatial_pilot.py`.
The subsequent [full-population walkthrough](vdeltae_population_walkthrough.md)
extends the verified pipeline to all 26 audited bodies and adds population comparisons.
Continue after the [saved coordinate and figure workflow](vdeltae_coordinate_run_walkthrough.md).
This is an offline export change. It reuses the verified coordinate run and
downloads no new data. Review the source in the stages below.

## 1. Coordinate basis and diagnostic checks

The mesh was retrieved from the same configured neuPrint dataset as the sites
and copied byte-for-byte into the verified coordinate run. Its native XYZ bounds
are compatible with the two site clouds. For the saved 30 October pilot:

| Source | Count | Within mesh's native XYZ bounding ranges |
| --- | ---: | ---: |
| hDeltaG 19798 FB sites | 4,047 | 4,047 |
| vDeltaE 54907 FB sites | 1,010 | 1,010 |

This is **not mesh containment**. A bounding box can include locations outside
the mesh, and a simplified mesh need not exactly follow the annotated FB mask.
The rendering uses the explicit native identity-transform assumption: multiply
mesh, skeleton and site XYZ by the same recorded `[0.008, 0.008, 0.008]` µm scale,
with no fitted translation, rotation, reflection or axis permutation. Source
provenance and range agreement support this visualization; they are not an
independent calibration of registration or anatomical axis orientation.

`mesh_context_ranges()` saves raw and µm bounds for the mesh vertices, each site
cloud and each full skeleton, alongside counts/fractions inside the mesh's
axis-aligned XYZ ranges. It rejects entirely disjoint mesh/site ranges as a
gross frame/unit mismatch. Passing that check does not establish alignment at
arbor or column resolution. No points are selected or discarded by this check.

`validate_context_mesh()` checks finite vertices and valid integer triangle
references, reusing the same geometry checks as `read_context_mesh()`. The
mesh remains a display resource: the [official neuprint-python documentation](https://connectome-neuprint.github.io/neuprint-python/docs/client.html)
specifies ROI meshes for visualization rather than quantitative analysis.

## 2. Project the actual mesh faces behind the cells

`body_projection_figure()` now accepts the optional keyword argument:

```python
figure = body_projection_figure(
    points, skeleton, scale_um, title,
    mesh=(vertices, faces),
)
```

For each native XY, XZ and YZ panel, the renderer uses `PolyCollection` to draw
the projected triangles, rather than only scattering vertices. The mesh is
faint gray-blue with alpha 0.12, no triangle edges, and a lower drawing order
than skeletons and sites. Filled projected triangles are visual context, not a
3D ray-traced surface or a cross-sectional contour.
The first panel's legend has a fixed upper-left position, avoiding the costly
automatic search over thousands of mesh triangles and site markers.

The top row shows mesh plus sites; the bottom adds the full coarse, unhealed
skeleton. Plot limits include mesh, all sites and all skeleton nodes. Nothing
is clipped to the mesh. Colors continue to denote dataset compartment labels;
markers distinguish pre/post polarity. Every site remains plotted, including
unannotated compartments and low-confidence sites. Anatomical axis orientation
and physical column assignment remain unresolved.

The previous function calls without `mesh=` remain supported and keep the
mesh-free view. The separate raw-coordinate mesh preview also remains available.

## 3. Request a new overlay export

`export_coordinate_pilot()` now has a keyword-only `mesh_overlay=False` option.
The default preserves the earlier export workflow. An overlay requires a
retrieved, verified and parseable mesh; missing optional mesh data are not
silently represented as a successful overlay.

From the repository root in the project environment:

```bash
python - <<'PY'
from fly_connectome.vdeltae_spatial_pilot import (
    export_coordinate_pilot, verify_coordinate_exports,
)

source = "results/vdeltae-hdeltag-coordinate-pilot-20261030"
output = "results/vdeltae-hdeltag-coordinate-fb-context-20261030"
record = export_coordinate_pilot(source, output, mesh_overlay=True)
verify_coordinate_exports(output)
print("FB context export verified:", record["status"], output)
PY
```

Use another fresh output name if that directory exists. The command does not
replace the original mesh-free figures or coordinate analyses. The new body
PNG/PDF files include mesh context; native-axis distributions and the separate
mesh preview are retained for comparison.

In addition to the previous artifacts, the export saves
`mesh_coordinate_ranges.csv`. `plots.json` records the source mesh hash, scale,
zero translation, axis order, display settings and the range-check limitation.
Offline verification checks these against the source and reconstructs the
diagnostic CSV. Previously completed mesh-free exports still verify without
requiring overlay-specific fields or artifacts.

## 4. Regression checks

Seven new cases bring the pilot test file to 39 cases. They check:

- Triangle scaling and drawing order, all points retained, full skeleton bounds
  visible, and source arrays/tables unchanged.
- A point outside a triangle but inside its box, demonstrating that the range
  diagnostic does not implement containment.
- Rejection of grossly disjoint coordinate ranges and unavailable mesh data.
- A complete overlay export/verification round trip without modifying sources.
- Detection of altered diagnostic values or a changed recorded transform.

Run:

```bash
python -m pytest -q tests/test_vdeltae_spatial_pilot.py
```

Inspect the actual body figures after tests pass. Look for gross range/scale
mismatches, obscured points, missing branches or crowded labels. Local offsets
can reflect the skeleton's coarse reconstruction. Do not fit a transform to
force agreement, infer anatomical axes from projection orientation, or infer a
column convention from two cells. This context view supports the next anatomical
inspection; it does not complete the column-mapping investigation.
