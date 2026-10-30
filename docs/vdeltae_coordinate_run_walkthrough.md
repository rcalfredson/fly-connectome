# Walkthrough: saved coordinates, descriptive analyses and figures

Prepared 29 October 2026; implementation added 30 October 2026. Continue after the helpers in the
[coordinate-pilot walkthrough](vdeltae_coordinate_pilot_walkthrough.md).
The functions below are now implemented in `fly_connectome/vdeltae_spatial_pilot.py`
with tests in `tests/test_vdeltae_spatial_pilot.py`. Use the sections as staged
review points rather than transcription instructions. The source module and
tests are authoritative; the excerpts explain the main implementation. No live
coordinate extraction was performed while implementing this stage.

Review order: descriptive tables (Section 1), run/provenance boundaries (Section 2),
offline reconstruction checks (Section 3), and plots/exports (Section 4). Section 5
explains the first 29 test cases; three additional source tests cover overlapping
layers and zero-site bodies/runs, for 32 pilot cases in total. Section 6 supplies
the commands for the live extraction and subsequent offline exports.

The two-body scope is unchanged. Save a coordinate run first, then render it
offline to a different new directory. Never replace the source probe, previous
plots, census results or presentation exports. Column mapping remains unresolved.

## 1. Descriptive tables, with explicit denominators

The implementation consolidates the following imports:

```python
from datetime import datetime, timezone
from importlib.metadata import version
import json
import platform

from neuprint import fetch_meta

from .connectivity import DATASET, SERVER, ValidationError
from .extension_analysis import markdown
from .inputs import integers, save_json, sha256
from .vdeltae_spatial_audit import (
    check_probe_metadata, spatial_probe_queries, verify_spatial_data_probe,
)
```

These constants follow the imports and existing constants:

```python
COORDINATE_KIND = "vdeltae_hdeltag_coordinate_pilot_v1"
ANALYSIS_FILES = (
    "site_counts.csv", "site_spatial_summary.csv", "layer_summary.csv", "coordinate_ranges.csv",
)
```

The pure analysis helper is:

```python
def coordinate_analysis_tables(points, probe, skeletons, scale_um):
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
```

The helper expects validated points, parsed skeletons, and the checked scale
from `native_scale_um()`. Interpret its outputs as follows:

- **Site counts:** exact body/polarity/compartment/split groups, including nulls.
- **Spatial summaries:** descriptive locations and spread, not column positions.
  A mean can fall between separated arbors; inspect the points and quantiles.
  The fraction denominator is all FB-marked sites on that body, not the whole
  population, whole neuron, contact count or an estimate of functional strength.
- **Layers:** true/false/unknown flags remain separate. The denominator is the
  FB sites in that body/polarity/compartment group. Layers can overlap; never
  sum their fractions as though they were mutually exclusive bins. A missing
  flag is not converted to a confirmed false annotation.
- **Ranges:** both raw and µm bounds for every body's site cloud and skeleton.
  Zero-site bodies remain present with count zero and missing site bounds.

The tables are descriptive. There are no fitted column angles, inferred signs,
integration time constants, statistical population claims or neuron-memory claims.

## 2. Save a bounded live coordinate run

The runner uses two small provenance helpers:

```python
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
```

Now the runner. It uses an injected client, just like the preceding probe, so
tests can substitute fake network responses. Source verification remains offline.

```python
def run_coordinate_pilot(client, source_probe, output):
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
    query = fb_site_query(probe["bodyId"])
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "kind": COORDINATE_KIND, "status": "running",
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
                raise ValidationError("Morphology pilot requires both retrieved skeletons.")
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
```

The original probe has already verified all its artifact hashes; the manifest
retains that source hash inventory as well as copied resource bytes. Both
skeletons must be parseable for this morphology pilot; the mesh remains optional.
Empty or invalid nonempty SWCs fail explicitly rather than masquerading as absent
anatomy. Queries, raw tables and partial products remain available after ordinary
failures. Initial input/path checks happen before output creation and therefore
cannot leave a failed run manifest. Disk failures or forced termination can
also prevent finalization; `running` is never treated as complete.

## 3. Verify saved coordinates and recompute their analyses offline

```python
def verify_coordinate_pilot(run):
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
    if ids != probe["bodyId"].tolist() or manifest["query"] != fb_site_query(ids):
        raise ValidationError("Coordinate selection or query differs from the source probe.")
    points = pd.read_csv(run / "fb_sites_raw.csv")
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
```

This verifies raw/derived consistency, not just file hashes. Original source
directories must remain available at their recorded paths. The manifest is a
trusted local record, not a signature; coordinated edits to it and its files
are outside an integrity check's guarantees. There is no live request during
verification, no proof that compartments are biologically correct, and no
requirement that current Python bytes equal the recorded run's code hashes.

## 4. Add native-axis distributions and export existing projections

The optional histograms use **40 fixed bins per axis**, shared between the two
bodies, with counts on the Y axis. The bin choice is an explicit display setting,
not a fitted biological parameter. Unequal counts reflect site numbers and
annotations, not differences in firing rate or coupling strength.

```python
def axis_distribution_figure(points, probe, scale_um, bins=40):
    import matplotlib.pyplot as plt

    scale = np.asarray(scale_um)
    colors = {"axon": "#d97706", "dendrite": "#0891b2", "linker": "#6b7280"}
    fig, axes = plt.subplots(len(probe), 3, figsize=(14, 7), squeeze=False, layout="constrained")
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


def export_coordinate_pilot(run, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run, output = Path(run).resolve(), Path(output).resolve()
    source_hash = sha256(run / "coordinates.json")
    manifest = verify_coordinate_pilot(run)
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
        "settings": {"formats": ["png", "pdf"], "dpi": 160, "histogram_bins": 40,
                     "subsampling": None, "mesh_overlay": False, "anatomical_axes": "unverified"},
    }
    save_json(output / "plots.json", record)
    try:
        record.update(pilot_code_provenance())
        points = pd.read_csv(run / "fb_sites_raw.csv")
        probe = pd.read_csv(run / "probe_bodies.csv")
        scale = np.asarray(manifest["native_to_um_per_axis"])

        def save_figure(figure, stem):
            try:
                for extension in record["settings"]["formats"]:
                    figure.savefig(output / f"{stem}.{extension}", dpi=record["settings"]["dpi"])
            finally:
                plt.close(figure)

        for neuron in probe.itertuples(index=False):
            body = int(neuron.bodyId)
            skeleton, _ = read_native_swc(run / f"skeleton_{body}.swc")
            save_figure(body_projection_figure(
                points.loc[points["bodyId"].eq(body)], skeleton, scale,
                f"{neuron.type} {body}: {neuron.instance}",
            ), f"body_{body}")
        save_figure(axis_distribution_figure(points, probe, scale, bins=40), "axis_distributions")
        if manifest["source_resources"]["FB_mesh"]["status"] == "retrieved":
            vertices, _ = read_context_mesh(run / "FB.obj")
            save_figure(mesh_preview_figure(vertices), "FB_context")
        (output / "summary.md").write_text("\n".join([
            "# Two-body coordinate inspection", "", markdown(probe[["bodyId", "type", "instance"]]),
            "", "## Raw site categories", "", markdown(pd.read_csv(run / "site_counts.csv")),
            "", "## Interpretation boundaries", "",
            "All FB-marked sites are shown without a confidence cutoff or spatial subsampling.",
            "Colors are dataset compartment labels; polarity is a separate marker/line style.",
            "Skeletons are coarse and unhealed. The mesh is a separate raw-coordinate vertex preview.",
            "Means/quantiles describe distributions, not column positions or physiological signals.",
            "Layer fractions have group-specific denominators and retain unknown flags; layers can overlap.",
            "Column meanings, anatomical axis orientation and compartment provenance remain unresolved.",
            "", f"Descriptive CSV tables and raw sources: {run}", "",
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
    output = Path(output).resolve()
    record = json.loads((output / "plots.json").read_text())
    if record.get("kind") != "vdeltae_hdeltag_coordinate_figures_v1" or record.get("status") != "complete":
        raise ValidationError("Verification requires complete coordinate figures.")
    run = Path(record["source"])
    manifest = verify_coordinate_pilot(run)
    if sha256(run / "coordinates.json") != record["source_manifest_sha256"]:
        raise ValidationError("Figure source manifest changed.")
    stems = [f"body_{body}" for body in manifest["selected_body_ids"]] + ["axis_distributions"]
    if manifest["source_resources"]["FB_mesh"]["status"] == "retrieved":
        stems.append("FB_context")
    required = {"summary.md"} | {f"{stem}.{extension}" for stem in stems for extension in ["png", "pdf"]}
    hashes = record.get("artifact_sha256", {})
    if set(hashes) != required:
        raise ValidationError("Figure artifact inventory differs from expected exports.")
    for name, digest in hashes.items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or not path.is_file() or sha256(path) != digest:
            raise ValidationError("Figure artifact is missing, outside the run, or changed.")
    return record
```

The export verifier checks integrity and source validity, not visual correctness.
Inspect the figures. Each six-panel body figure uses its full skeleton extent;
the histogram bins are shared, but the Y axes are counts rather than normalized
probabilities. There are no fitted rotations, mirrored views, neuronal activity
readouts or surface-distance calculations.

## 5. First wrapper and analysis tests

The implemented tests include the block below. The fake fixture creates a tiny
saved probe and replaces its verifier; it
does not test the already-tested source probe's full schema. It still supplies
real on-disk geometry hashes and source files so the new verifier can check
copy fidelity. No credentials or network are used.

```python
from copy import deepcopy
import json
from pathlib import Path

from fly_connectome import vdeltae_spatial_pilot as pilot
from fly_connectome.inputs import sha256


@pytest.fixture
def coordinate_environment(tmp_path, monkeypatch, point_example):
    points, _, availability, categories = point_example
    source = tmp_path / "source-probe"
    source.mkdir()
    probe = pd.DataFrame({
        "bodyId": [10, 1], "type": ["vDeltaE", "hDeltaG"],
        "instance": ["vDeltaE_C6", "hDeltaG_14_C8"], "annotation_column": [6, 8],
    })
    probe.to_csv(source / "probe_bodies.csv", index=False)
    availability.to_csv(source / "site_availability.csv", index=False)
    categories.to_csv(source / "fb_site_categories.csv", index=False)
    resources = {}
    for body in [10, 1]:
        name = f"skeleton_{body}.swc"
        (source / name).write_text("# synthetic coarse geometry\n1 0 0 0 0 1 -1\n2 2 16 16 0 1 1\n")
        resources[f"skeleton_{body}"] = {"status": "retrieved", "file": name}
    (source / "FB.obj").write_text("v 0 0 0\nv 16 0 0\nv 0 16 0\nf 1 2 3\n")
    resources["FB_mesh"] = {"status": "retrieved", "file": "FB.obj"}
    metadata = {
        "uuid": "test", "latestMutationId": 1, "lastDatabaseEdit": "test",
        "voxelSize": [8, 8, 8], "voxelUnits": "nanometers", "postHighAccuracyThreshold": .5,
    }
    inventory = {
        "annotation_audit_directory": str(tmp_path / "audit"),
        "sources": {"synthetic": {
            "source_manifest_file": str(tmp_path / "census" / "manifest.json"),
            "spatial_metadata": {
                "dataset_identity": {name: metadata[name] for name in ["uuid", "latestMutationId", "lastDatabaseEdit"]},
                "coordinate_metadata": {name: metadata[name] for name in ["voxelSize", "voxelUnits"]},
                "recorded_post_high_accuracy_threshold": .5,
            },
        }},
    }
    source_manifest = {
        "status": "complete", "source_inventory": inventory, "resources": resources,
        "artifact_sha256": {path.name: sha256(path) for path in source.iterdir()},
    }
    (source / "capability.json").write_text(json.dumps(source_manifest))
    monkeypatch.setattr(pilot, "verify_spatial_data_probe", lambda path: json.loads((Path(path) / "capability.json").read_text()))
    monkeypatch.setattr(pilot, "fetch_meta", lambda **kwargs: deepcopy(metadata))

    class FakeClient:
        server = f"https://{pilot.SERVER}"
        dataset = pilot.DATASET

        def fetch_custom(self, query):
            assert query == pilot.fb_site_query(probe["bodyId"])
            return points.copy(deep=True)

    return FakeClient(), source, points, metadata


def test_coordinate_analysis_retains_unknown_layer_flags(tmp_path, point_example):
    points, _, _, _ = point_example
    probe = pd.DataFrame({"bodyId": [10, 1]})
    path = tmp_path / "body.swc"
    path.write_text("1 0 0 0 0 1 -1\n")
    skeleton, _ = pilot.read_native_swc(path)
    tables = pilot.coordinate_analysis_tables(points, probe, {10: skeleton, 1: skeleton}, [.008] * 3)
    layers = tables["layer_summary.csv"]
    assert layers["sites_unknown"].eq(1).all()
    assert layers["sites_true"].eq(0).all()
    assert layers["fraction_marked_of_group_FB_sites"].eq(0).all()
    summary = tables["site_spatial_summary.csv"].set_index("bodyId")
    assert summary.loc[10, "x_mean_um"] == pytest.approx(.064)
    assert summary.loc[1, "confidence_missing"] == 1
    assert summary["fraction_of_body_FB_sites"].eq(1).all()


def test_coordinate_run_and_offline_export_round_trip(coordinate_environment, tmp_path):
    client, source, _, _ = coordinate_environment
    run, figures = tmp_path / "coordinates", tmp_path / "figures"
    assert pilot.run_coordinate_pilot(client, source, run)["status"] == "complete"
    assert pilot.verify_coordinate_pilot(run)["point_count"] == 2
    result = pilot.export_coordinate_pilot(run, figures)
    assert result["status"] == "complete"
    assert pilot.verify_coordinate_exports(figures)["status"] == "complete"
    assert (figures / "body_10.png").stat().st_size > 0
    assert (figures / "axis_distributions.pdf").stat().st_size > 0
    assert not plt.get_fignums()


def test_coordinate_runner_refuses_existing_output(coordinate_environment, tmp_path):
    client, source, _, _ = coordinate_environment
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        pilot.run_coordinate_pilot(client, source, output)
    assert list(output.iterdir()) == []


def test_coordinate_runner_saves_failed_raw_points(coordinate_environment, tmp_path):
    client, source, points, _ = coordinate_environment
    points.loc[0, "x"] = np.inf
    output = tmp_path / "invalid"
    with pytest.raises(ValidationError, match="nonfinite"):
        pilot.run_coordinate_pilot(client, source, output)
    manifest = json.loads((output / "coordinates.json").read_text())
    assert manifest["status"] == "failed"
    assert "fb_sites_raw.csv" in manifest["artifact_sha256"]
    with pytest.raises(ValidationError, match="complete coordinate pilot"):
        pilot.verify_coordinate_pilot(output)


def test_coordinate_runner_rejects_late_metadata_drift(coordinate_environment, tmp_path, monkeypatch):
    client, source, _, metadata = coordinate_environment
    calls = []

    def fetch(**kwargs):
        result = deepcopy(metadata)
        if calls:
            result["latestMutationId"] = 2
        calls.append(True)
        return result

    monkeypatch.setattr(pilot, "fetch_meta", fetch)
    output = tmp_path / "metadata-drift"
    with pytest.raises(ValidationError, match="metadata differs"):
        pilot.run_coordinate_pilot(client, source, output)
    assert len(calls) == 2
    assert json.loads((output / "coordinates.json").read_text())["status"] == "failed"


def test_coordinate_source_change_stops_completion(coordinate_environment, tmp_path, monkeypatch):
    client, source, _, _ = coordinate_environment
    original_fetch = client.fetch_custom

    def changing_fetch(query):
        result = original_fetch(query)
        path = source / "capability.json"
        path.write_text(path.read_text() + "\n")
        return result

    monkeypatch.setattr(client, "fetch_custom", changing_fetch)
    output = tmp_path / "source-drift"
    with pytest.raises(ValidationError, match="source probe changed"):
        pilot.run_coordinate_pilot(client, source, output)
    assert json.loads((output / "coordinates.json").read_text())["status"] == "failed"


def test_coordinate_runner_rejects_invalid_morphology(coordinate_environment, tmp_path):
    client, source, _, _ = coordinate_environment
    (source / "skeleton_10.swc").write_text("1 0 0 0 0 1 99\n")
    with pytest.raises(ValidationError, match="absent parent"):
        pilot.run_coordinate_pilot(client, source, tmp_path / "bad-geometry")


@pytest.mark.parametrize("filename", ["fb_sites_raw.csv", "site_spatial_summary.csv", "skeleton_10.swc"])
def test_coordinate_saved_artifact_changes_are_detected(coordinate_environment, tmp_path, filename):
    client, source, _, _ = coordinate_environment
    output = tmp_path / "coordinates"
    pilot.run_coordinate_pilot(client, source, output)
    path = output / filename
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValidationError, match="artifact is missing, outside the run, or changed"):
        pilot.verify_coordinate_pilot(output)


def test_derived_tables_are_recomputed_not_just_hashed(coordinate_environment, tmp_path):
    client, source, _, _ = coordinate_environment
    run = tmp_path / "coordinates"
    pilot.run_coordinate_pilot(client, source, run)
    path = run / "site_spatial_summary.csv"
    table = pd.read_csv(path)
    table.loc[0, "x_mean_um"] += 1
    table.to_csv(path, index=False)
    manifest_path = run / "coordinates.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_sha256"][path.name] = sha256(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        pilot.verify_coordinate_pilot(run)


def test_export_refuses_existing_output(coordinate_environment, tmp_path):
    client, source, _, _ = coordinate_environment
    run, output = tmp_path / "coordinates", tmp_path / "figures"
    pilot.run_coordinate_pilot(client, source, run)
    output.mkdir()
    with pytest.raises(FileExistsError):
        pilot.export_coordinate_pilot(run, output)
    assert list(output.iterdir()) == []


def test_export_verification_detects_changed_figure(coordinate_environment, tmp_path):
    client, source, _, _ = coordinate_environment
    run, output = tmp_path / "coordinates", tmp_path / "figures"
    pilot.run_coordinate_pilot(client, source, run)
    pilot.export_coordinate_pilot(run, output)
    (output / "body_10.png").write_bytes(b"changed")
    with pytest.raises(ValidationError, match="Figure artifact is missing, outside the run, or changed"):
        pilot.verify_coordinate_exports(output)


def test_export_failure_is_recorded_and_figures_closed(coordinate_environment, tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    client, source, _, _ = coordinate_environment
    run, output = tmp_path / "coordinates", tmp_path / "figures"
    pilot.run_coordinate_pilot(client, source, run)

    def fail_save(*args, **kwargs):
        raise OSError("Synthetic save failure")

    monkeypatch.setattr(Figure, "savefig", fail_save)
    with pytest.raises(OSError):
        pilot.export_coordinate_pilot(run, output)
    assert json.loads((output / "plots.json").read_text())["status"] == "failed"
    assert not plt.get_fignums()
```

These excerpts explain 14 cases added to the previous 15. With the three
additional overlap/zero-site cases in the source test file, there are now
**32 pilot tests**. Run:

```bash
python -m pytest -q tests/test_vdeltae_spatial_pilot.py
```

The invalid-morphology test deliberately substitutes the source verifier, so it
can reach the parser after changing its synthetic SWC. The production source
verifier can reject that alteration earlier; either route prevents completion.
The fake-client tests do not validate server-side Cypher. Review all functions
and run these tests before making the first real coordinate request.

## 6. Live extraction, then offline analysis and figures

After review, use fresh output names and the existing local
authentication pattern. This is the only block here that contacts neuPrint:

```bash
python - <<'PY'
import os
from pathlib import Path
from dotenv import load_dotenv
from neuprint import Client
from fly_connectome.connectivity import DATASET, SERVER, ValidationError
from fly_connectome.vdeltae_spatial_pilot import run_coordinate_pilot

load_dotenv(Path.cwd() / ".env", override=False)
token = os.environ.get("NEUPRINT_TOKEN", "").strip()
if not token:
    raise ValidationError("Set NEUPRINT_TOKEN locally as described in the README.")
try:
    result = run_coordinate_pilot(
        Client(SERVER, dataset=DATASET, token=token),
        "results/vdeltae-hdeltag-spatial-capability-20261029",
        "results/vdeltae-hdeltag-coordinate-pilot-20261030",
    )
except Exception as error:
    print(f"Coordinate pilot stopped ({type(error).__name__}); inspect saved manifest and raw diagnostics.")
    raise SystemExit(1) from None
print("Coordinate pilot:", result["status"], "FB sites:", result["point_count"])
PY
```

Then run locally without a client or credentials:

```bash
python - <<'PY'
from fly_connectome.vdeltae_spatial_pilot import (
    verify_coordinate_pilot, export_coordinate_pilot, verify_coordinate_exports,
)

run = "results/vdeltae-hdeltag-coordinate-pilot-20261030"
figures = "results/vdeltae-hdeltag-coordinate-figures-20261030"
verify_coordinate_pilot(run)
export_coordinate_pilot(run, figures)
verify_coordinate_exports(figures)
print("Coordinate tables and figures verified:", figures)
PY
```

Expected current site totals, if the same source snapshot is available, remain
1,010 for vDeltaE 54907 and 4,047 for hDeltaG 19798. These are observations to
reconcile, not targets to enforce by deleting points. Stop on drift or failures;
inspect their saved artifacts and use a new directory for any rerun.

## 7. Read the results before expanding

Use the following questions to guide the first inspection:

1. Are site/skeleton ranges compatible in the documented native frame? A gross
   scale mismatch calls for provenance review, not fitted registration. Coarse
   skeletons can omit fine branches; exact nearest-node coincidence is not a
   biological requirement and no distance cutoff is used here.
2. Do separate arbor regions appear in the point projections and native-axis
   distributions? Compare means with quantiles and the raw cloud to avoid
   mistaking an average between arbors for a physical column.
3. How do pre/post polarity and dataset compartment labels differ? In particular,
   input sites on an arbor labeled axonal are retained. The labels' generation
   and biological interpretation still require independent evidence.
4. Which layers contain sites, and how much annotation is unknown? Refer to the
   exact group denominators; do not add overlapping layer counts as independent
   contributions or interpret the counts as activity/coupling strength.
5. What evidence, if any, connects a C token to a particular arbor region? Two
   examples cannot establish a complete column convention or boundary/wrap rule.

This pilot succeeds when raw points reconcile, geometry parses, figures verify,
and mismatches/uncertainties remain visible. If those checks pass, expand the
same validated extraction to the audited 26 bodies. Do not introduce an inferred
eight-slot ring, effective signs, home-vector decoding or new dynamics here.
