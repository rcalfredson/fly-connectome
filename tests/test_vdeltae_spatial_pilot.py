from copy import deepcopy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from fly_connectome.connectivity import ValidationError
from fly_connectome import vdeltae_spatial_pilot as pilot
from fly_connectome.inputs import sha256
from fly_connectome.vdeltae_spatial_pilot import (
    XYZ,
    LAYERS,
    fb_site_query,
    native_scale_um,
    read_native_swc,
    skeleton_segments,
    read_context_mesh,
    validate_fb_points,
    category_counts,
    body_projection_figure,
    mesh_preview_figure,
)


def test_native_units_require_explicit_matching_metadata():
    scale = native_scale_um({"voxelSize": [8, 8, 8], "voxelUnits": "nanometers"})
    np.testing.assert_allclose(np.array([[1000, 2000, 3000]]) * scale, [[8, 16, 24]])
    with pytest.raises(ValidationError):
        native_scale_um({"voxelSize": [8, 8, 8], "voxelUnits": "micrometers"})


def test_swc_retains_types_comments_and_multiple_components(tmp_path):
    path = tmp_path / "geometry.swc"
    path.write_text(
        "# coarse geometry\n1 0 0 0 0 1 -1\n2 2 8 0 0 1 1\n3 3 0 8 0 1 -1\n"
    )
    table, comments = read_native_swc(path)
    assert comments == ["# coarse geometry"]
    assert table["node_type"].tolist() == [0, 2, 3]
    assert table["parent_id"].eq(-1).sum() == 2
    np.testing.assert_array_equal(skeleton_segments(table), [[[8, 0, 0], [0, 0, 0]]])


@pytest.mark.parametrize(
    "text",
    [
        "1 0 0 0 0 1 2\n2 0 1 0 0 1 1\n",  # cycle
        "1 0 0 0 0 1 5\n",  # absent parent
        "1 0 0 0 0 1 -1\n1 0 1 0 0 1 -1\n",  # duplicate ID
        "1 0 nan 0 0 1 -1\n",  # nonfinite position
        "1 0 0 0 0 -1 -1\n",  # negative radius
    ],
)
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
    points = pd.DataFrame(
        {
            "bodyId": [10, 1],
            "synapse_node_id": [100, 200],
            "synapse_type": ["pre", "post"],
            "x": [8.0, 16.0],
            "y": [0.0, 8.0],
            "z": [0.0, 0.0],
            "confidence": [0.9, np.nan],
            "compartment": ["axon", None],
            "split": [None, None],
            "FB": [True, True],
            **{layer: [None, None] for layer in LAYERS},
        }
    )
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
        body_projection_figure(points, skeleton, [0.008] * 3, "Synthetic body"),
        mesh_preview_figure(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        ),
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


def test_coordinate_layers_can_overlap_and_keep_explicit_false(point_example):
    points, probe, _, _ = point_example
    skeleton = pd.DataFrame({"x": [0.], "y": [0.], "z": [0.]})
    points["FBl1"] = True
    points["FBl2"] = True
    points["FBl3"] = False
    tables = pilot.coordinate_analysis_tables(
        points, probe, {10: skeleton, 1: skeleton}, [.008] * 3
    )
    layers = tables["layer_summary.csv"]
    for layer in ["FBl1", "FBl2"]:
        assert layers.loc[layers["layer"].eq(layer), "fraction_marked_of_group_FB_sites"].eq(1).all()
    assert layers.loc[layers["layer"].eq("FBl3"), "sites_false"].eq(1).all()
    assert layers.loc[layers["layer"].eq("FBl3"), "sites_unknown"].eq(0).all()
    assert layers.loc[layers["layer"].eq("FBl4"), "sites_unknown"].eq(1).all()


def test_coordinate_analysis_keeps_zero_site_body_in_ranges(point_example):
    points, probe, _, _ = point_example
    points = points.loc[points["bodyId"].eq(10)].copy()
    skeleton = pd.DataFrame({"x": [0.], "y": [0.], "z": [0.]})
    tables = pilot.coordinate_analysis_tables(
        points, probe, {10: skeleton, 1: skeleton}, [.008] * 3
    )
    ranges = tables["coordinate_ranges.csv"]
    missing_body = ranges.loc[ranges["bodyId"].eq(1) & ranges["source"].eq("FB_sites")]
    assert len(missing_body) == 1
    assert missing_body["count"].iloc[0] == 0
    assert missing_body["x_min_um"].isna().all()


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


def test_coordinate_zero_site_run_and_export_round_trip(coordinate_environment, tmp_path):
    client, source, points, _ = coordinate_environment
    points.drop(index=points.index, inplace=True)
    availability = pd.read_csv(source / "site_availability.csv")
    availability["sites_marked_FB"] = 0
    availability.to_csv(source / "site_availability.csv", index=False)
    pilot.category_counts(points).to_csv(source / "fb_site_categories.csv", index=False)
    source_manifest_path = source / "capability.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    for name in ["site_availability.csv", "fb_site_categories.csv"]:
        source_manifest["artifact_sha256"][name] = sha256(source / name)
    source_manifest_path.write_text(json.dumps(source_manifest))
    run, figures = tmp_path / "empty-coordinates", tmp_path / "empty-figures"
    pilot.run_coordinate_pilot(client, source, run)
    assert pilot.verify_coordinate_pilot(run)["point_count"] == 0
    pilot.export_coordinate_pilot(run, figures)
    assert pilot.verify_coordinate_exports(figures)["status"] == "complete"
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
