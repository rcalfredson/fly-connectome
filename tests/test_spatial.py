"""Scientific counting and validation invariants, without network access."""

import json

import numpy as np
import pandas as pd
import pytest

from fly_connectome import spatial as s
from fly_connectome.connectivity import ValidationError


def contact_rows():
    # One polyadic presynaptic site, two postsynaptic sites; one pre outside FB.
    return pd.DataFrame({
        "bodyId_pre": [1, 1, 1], "bodyId_post": [2, 2, 2],
        "x_pre": [10, 10, 20], "y_pre": [0, 0, 0], "z_pre": [0, 0, 0],
        "x_post": [11, 12, 21], "y_post": [0, 0, 0], "z_post": [0, 0, 0],
        "roi_pre": [["FB", "FBl3"], ["FB", "FBl3"], ["CentralBrain-unspecified"]],
        "roi_post": [["FB", "FBl3"], ["FB", "FBl4"], ["FB", "FBl3"]],
        "compartment_post": ["axon", "axon", "dendrite"],
    })


def test_contacts_preserve_polyadic_sites_and_endpoint_roi_distinction():
    contacts = s.prepare_contacts(contact_rows(), 1, 2)
    summary = s.contact_summary(contacts).set_index("scope")
    assert summary.loc["total", "contacts"] == 3
    assert summary.loc["total", "unique_pre_sites"] == 2
    assert summary.loc["total", "unique_post_sites"] == 3
    assert summary.loc["post_in_FB", "contacts"] == 3
    assert summary.loc["both_in_FB", "contacts"] == 2
    assert contacts.FBl3_post.sum() == 2
    assert contacts.compartment_post.tolist() == ["axon", "axon", "dendrite"]


@pytest.mark.parametrize("change", ["duplicate", "wrong_body", "nonfinite"])
def test_invalid_contacts_stop(change):
    rows = contact_rows()
    if change == "duplicate":
        rows = pd.concat([rows, rows.iloc[:1]])
    elif change == "wrong_body":
        rows.loc[0, "bodyId_post"] = 99
    else:
        rows["x_post"] = rows.x_post.astype(float)
        rows.loc[0, "x_post"] = np.inf
    with pytest.raises(ValidationError):
        s.prepare_contacts(rows, 1, 2)


def synapse_rows():
    points = pd.DataFrame({
        "bodyId": [1, 2, 2, 2], "type": ["pre", "post", "post", "post"],
        "x": [10, 11, 12, 21], "y": [0, 0, 0, 0], "z": [0, 0, 0, 0],
        "compartment": ["axon", "axon", "axon", "dendrite"],
    })
    for layer in s.LAYERS:
        points[layer] = False
    points.loc[[0, 1, 3], "FBl3"] = True
    points.loc[2, "FBl4"] = True
    return points


def test_roi_validation_and_endpoint_membership():
    points = synapse_rows()
    rois = pd.DataFrame({"bodyId": [1, 2, 2, 2], "roi": ["FB", "FB", "FBl3", "FBl4"],
                         "pre": [1, 0, 0, 0], "post": [0, 3, 2, 1]})
    contacts = s.prepare_contacts(contact_rows(), 1, 2)
    s.validate_points(points, rois, contacts)
    wrong = rois.copy()
    wrong.loc[1, "post"] = 2
    with pytest.raises(ValidationError, match="count mismatch"):
        s.validate_points(points, wrong, contacts)
    missing = points.copy()
    missing.loc[3, "x"] = 100
    with pytest.raises(ValidationError, match="endpoint missing"):
        s.validate_points(missing, rois, contacts)


def test_summary_separates_polarity_compartment_and_parent_layer():
    points = synapse_rows()
    summary = s.spatial_summary(points)
    rows = summary[(summary.bodyId == 2) & (summary.roi == "FB")].set_index("compartment")
    assert rows.loc["axon", "count"] == 2
    assert rows.loc["dendrite", "count"] == 1
    assert rows.loc["axon", "x_mean_um"] == pytest.approx(11.5 * .008)
    assert rows.loc["axon", "x_std_um"] == pytest.approx(.5 * .008)
    assert rows.loc["dendrite", "x_std_um"] == 0
    assert summary[(summary.bodyId == 2) & (summary.roi == "FBl3")]["count"].sum() == 2


def test_bounded_query_keeps_explicit_layer_memberships():
    class Client:
        def fetch_custom(self, query):
            assert "WITH DISTINCT n, s" in query
            assert "n.bodyId IN [1, 2]" in query
            assert "s.FB AND s.confidence > 0.5" in query
            assert all(f"coalesce(s.{layer}, false) AS {layer}" in query for layer in s.LAYERS)
            return synapse_rows()
    assert len(s.fetch_fb_synapses(Client(), [1, 2], .5)) == 4


def test_population_can_retain_neurons_with_no_fb_synapses():
    class Client:
        def fetch_custom(self, query):
            return synapse_rows().iloc[:0]

    with pytest.raises(ValidationError, match="Missing or duplicate"):
        s.fetch_fb_synapses(Client(), [1], .5)
    assert s.fetch_fb_synapses(Client(), [1], .5, allow_empty=True).empty


def test_obj_polygon_and_slash_indices():
    vertices, faces = s.parse_obj(b"v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1/1 2/2 3/3 4/4\n")
    assert vertices.shape == (4, 3)
    assert faces.tolist() == [[0, 1, 2], [0, 2, 3]]


@pytest.mark.parametrize("bad_count", [False, True])
def test_run_manifest_and_validation_failure(tmp_path, monkeypatch, bad_count):
    points = synapse_rows().replace({"bodyId": {1: 26913, 2: 515192}})
    contacts = contact_rows().replace({"bodyId_pre": {1: 26913}, "bodyId_post": {2: 515192}})
    annotations = pd.DataFrame({"bodyId": [26913, 515192], "type": ["PFNd", "hDeltaB"],
                                "instance": ["PFNd(PB04)_R2_C8", "hDeltaB_07_C6"], "fb_column": [8, 6]})
    rois = pd.DataFrame({"bodyId": [26913, 515192], "roi": ["FB", "FB"],
                         "pre": [1, 0], "post": [0, 3]})
    matrix = pd.DataFrame([[4 if bad_count else 3]], index=[26913], columns=[515192])
    monkeypatch.setattr(s, "PAIRS", ((26913, 515192),))
    monkeypatch.setattr(s, "REFERENCE_COUNTS", {"total": 3, "post_in_FB": 3, "both_in_FB": 2})
    monkeypatch.setattr(s, "load_source", lambda source: (annotations, matrix, {}))
    monkeypatch.setattr(s, "fetch_meta", lambda **kw: {"voxelSize": [8, 8, 8], "voxelUnits": "nanometers", "postHighAccuracyThreshold": .5})
    monkeypatch.setattr(s, "fetch_neurons", lambda *args, **kw: (annotations, rois))
    monkeypatch.setattr(s, "fetch_synapse_connections", lambda *args, **kw: contacts)
    monkeypatch.setattr(s, "plot_pilot", lambda run: (run / "plot_checked.txt").write_text("mock plot"))

    class Client:
        all_rois = ["FB", *s.LAYERS]
        primary_rois = ["FB"]
        meta = {"postHighAccuracyThreshold": .5}

        def fetch_custom(self, query):
            return points

        def fetch_skeleton(self, body, heal):
            assert heal is False
            return pd.DataFrame({"rowId": [1], "link": [-1], "x": [1], "y": [1], "z": [1]})

        def fetch_roi_mesh(self, roi):
            return b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"

    output = tmp_path / "pilot"
    if bad_count:
        with pytest.raises(ValidationError, match="saved weight"):
            s.run_pilot(Client(), tmp_path, output, {})
        manifest = json.loads((output / "spatial_pilot.json").read_text())
        assert manifest["status"] == "validation_failed"
        assert not (output / "plot_checked.txt").exists()
    else:
        manifest = s.run_pilot(Client(), tmp_path, output, {})
        assert manifest["status"] == "complete"
        assert manifest["validation"]["reference_counts"] == s.REFERENCE_COUNTS
        assert len(pd.read_csv(output / "contacts.csv")) == 3
        assert len(list((output / "skeletons").glob("*.csv"))) == 2
        assert len(list((output / "meshes").glob("*.obj"))) == 3
        assert str(output / "contacts.csv") in manifest["artifact_sha256"]
        # JSON ROI lists survive CSV serialization; they are not Python reprs.
        saved = pd.read_csv(output / "contacts.csv")
        assert json.loads(saved.roi_post.iloc[0]) == ["FB", "FBl3"]
    with pytest.raises(FileExistsError):
        s.run_pilot(Client(), tmp_path, output, {})
