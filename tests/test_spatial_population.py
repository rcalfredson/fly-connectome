"""Population coverage and cache integrity, without neuPrint network requests."""

import json
from pathlib import Path
from html.parser import HTMLParser
import shutil

import pandas as pd
import pytest

from fly_connectome import spatial as s
from fly_connectome import spatial_population as p
from fly_connectome import __main__ as cli
from fly_connectome.connectivity import DATASET, SERVER, ValidationError, build_matrix


def annotations():
    return pd.DataFrame({"bodyId": [1, 2, 3, 4], "type": ["PFNd", "hDeltaB", "PFNd", "hDeltaB"],
                         "fb_column": [1, 2, 1, 2]})


def contacts():
    # Different body pairs in one column cell, including an outside-FB contact.
    return pd.DataFrame({"bodyId_pre": [3, 3, 1, 1, 1], "bodyId_post": [4, 4, 2, 2, 2],
                         "FB_post": [True, False, True, True, True],
                         "both_in_FB": [True, False, True, False, True],
                         "compartment_post": ["dendrite", "axon", "axon", "axon", "linker"],
                         "x_post": [1, 2, 3, 4, 5], "y_post": [0] * 5, "z_post": [0] * 5})


def test_all_column_cells_include_all_positive_pairs_and_scope_counts():
    data = p.annotate_contacts(contacts(), annotations())
    coverage = p.column_coverage(data, annotations())
    assert len(coverage) == 96
    assert (coverage.total_contacts > 0).sum() == 1
    cell = coverage[(coverage.column_pre == 1) & (coverage.column_post == 2)].iloc[0]
    assert cell.possible_pairs == 4
    assert cell.connected_pairs == 2
    assert cell.total_contacts == 5
    assert cell.fb_post_contacts == 4
    assert cell.both_fb_contacts == 3
    assert cell.fb_post_axon_contacts == 2
    assert cell.fb_post_dendrite_contacts == 1
    assert cell.fb_post_linker_contacts == 1
    assert cell.representative_pre == 1
    assert cell.representative_post == 2
    assert cell.representative_weight == 3
    assert coverage.total_contacts.sum() == 5
    assert coverage.connected_pairs.sum() == 2
    assert coverage[coverage.total_contacts == 0].representative_pre.isna().all()


def test_representative_ties_break_by_body_id_without_dropping_any_contacts():
    data = contacts().iloc[[0, 1, 2, 3]]
    coverage = p.column_coverage(p.annotate_contacts(data, annotations()), annotations())
    cell = coverage[coverage.total_contacts > 0].iloc[0]
    assert cell.representative_pre == 1
    assert cell.total_contacts == 4


def test_unknown_body_is_rejected():
    data = contacts()
    data.loc[0, "bodyId_pre"] = 99
    with pytest.raises(ValidationError, match="Unannotated"):
        p.annotate_contacts(data, annotations())


def test_arbor_positions_keep_all_roi_and_fb_scopes_separate():
    data = p.annotate_contacts(contacts(), annotations())
    summary = p.arbor_contacts(data)
    total = summary[(summary.scope == "total") & (summary.bodyId_pre == 3)]
    fb = summary[(summary.scope == "post_in_FB") & (summary.bodyId_pre == 3)]
    assert total.contacts.sum() == 2
    assert set(total.compartment_post) == {"axon", "dendrite"}
    assert fb.contacts.sum() == 1
    assert fb.iloc[0].x_mean_um == .008


def test_pfnd_landmark_pools_sites_across_compartments_once_per_body():
    points = pd.DataFrame({"bodyId": [1, 1, 1, 1, 3, 2],
                           "type": ["pre", "pre", "pre", "post", "pre", "pre"],
                           "compartment": ["axon", "axon", "dendrite", "axon", "axon", "axon"],
                           "x": [0, 0, 9, 1000, 6, 1000], "y": [0] * 6, "z": [0] * 6})
    summary = pd.DataFrame(columns=["neuron_type", "type", "compartment", "roi"])
    groups = p.landmark_tables(points, summary, annotations())
    source = groups[0][1].set_index("bodyId")
    assert source.index.tolist() == [1, 3]
    # Average individual sites, not equally weighted compartment centroids.
    assert source.loc[1, "x_mean_um"] == pytest.approx(3 * .008)
    assert source.loc[3, "x_mean_um"] == pytest.approx(6 * .008)


def test_population_rejects_missing_or_new_pair_before_roi_validation(monkeypatch):
    monkeypatch.setattr(p, "validate_points", lambda *args: None)
    data = p.annotate_contacts(contacts(), annotations())
    matrix = pd.DataFrame([[3, 0], [0, 2]], index=[1, 3], columns=[2, 4])
    matrix.loc[1, 4] = 1  # Missing low-weight contact must not be silently ignored.
    with pytest.raises(ValidationError, match="saved positive pairs"):
        p.validate_population(data, None, None, matrix)
    matrix.loc[1, 4] = 0
    matrix.loc[3, 4] = 0  # A newly reported pair is also a mismatch.
    with pytest.raises(ValidationError, match="saved positive pairs"):
        p.validate_population(data, None, None, matrix)


def test_cache_requires_matching_metadata_and_verified_contents(tmp_path):
    meta = {"uuid": "same", "lastDatabaseEdit": "date1"}
    artifact = tmp_path / "fb_synapses.csv"
    artifact.write_text("bodyId\n1\n")
    manifest = {"status": "complete", "server": SERVER, "dataset": DATASET,
                "dataset_metadata": meta, "artifact_sha256": s.hashes([artifact])}
    (tmp_path / "spatial_pilot.json").write_text(json.dumps(manifest))
    cache = p.compatible_cache(tmp_path, meta)
    assert p.cache_file(cache, "fb_synapses.csv") == artifact
    with pytest.raises(ValidationError, match="metadata differs"):
        p.compatible_cache(tmp_path, {**meta, "lastDatabaseEdit": "date2"})
    artifact.write_text("bodyId\n99\n")
    with pytest.raises(ValidationError, match="not verified"):
        p.cache_file(cache, "fb_synapses.csv")


def test_load_all_connected_source_keeps_one_contact_pair_and_disconnected_body(tmp_path):
    pd.DataFrame({"bodyId": [26913, 25982], "type": ["PFNd"] * 2,
                  "instance": ["PFNd(PB04)_R2_C8", "PFNd(PB04)_L2_C1"]}).to_csv(tmp_path / "PFNd_neurons.csv", index=False)
    pd.DataFrame({"bodyId": [515192, 11512], "type": ["hDeltaB"] * 2,
                  "instance": ["hDeltaB_07_C6", "hDeltaB_08_C7"]}).to_csv(tmp_path / "hDeltaB_neurons.csv", index=False)
    edges = pd.DataFrame([(26913, 515192, 137), (25982, 515192, 1)], columns=["bodyId_pre", "bodyId_post", "weight"])
    edges.to_csv(tmp_path / "PFNd_to_hDeltaB_connections.csv", index=False)
    build_matrix(edges, [26913, 25982], [515192, 11512]).to_csv(tmp_path / "PFNd_to_hDeltaB_matrix.csv")
    (tmp_path / "metadata.json").write_text(json.dumps({"status": "complete", "server": SERVER, "dataset": DATASET}))
    selected, matrix, _ = s.load_source(tmp_path, all_connected=True)
    assert set(selected.bodyId) == {26913, 25982, 515192, 11512}
    assert matrix.loc[25982, 515192] == 1
    assert (matrix[11512] == 0).all()


@pytest.mark.parametrize("tampered", [False, True])
def test_offline_population_replot_checks_data_without_client(tmp_path, monkeypatch, tampered):
    source = tmp_path / "source"
    source.mkdir()
    data = source / "contacts.csv"
    data.write_text("bodyId_pre,bodyId_post\n1,2\n")
    manifest = {"status": "data_complete", "validation": {"all_pair_counts_match_saved_weights": True},
                "data_sha256": s.hashes([data])}
    (source / "spatial_population.json").write_text(json.dumps(manifest))
    if tampered:
        data.write_text("bodyId_pre,bodyId_post\n9,9\n")

    def no_client(*args, **kwargs):
        raise AssertionError("Offline replot must not construct a neuPrint client")

    monkeypatch.setattr(cli, "Client", no_client)
    monkeypatch.setattr(cli, "render_population", lambda run, output: (output / "index.html").write_text("rendered"))
    output = tmp_path / "plots"
    result = cli.main(["spatial-all-plot", str(source), "--output", str(output)])
    assert result == (1 if tampered else 0)
    if tampered:
        assert not output.exists()
    else:
        assert (output / "index.html").read_text() == "rendered"
        assert json.loads((output / "replot.json").read_text())["status"] == "complete"


@pytest.mark.parametrize("style", ["legacy_absolute", "legacy_relative", "run_relative"])
@pytest.mark.parametrize("original_present", [False, True])
@pytest.mark.parametrize("tampered", [False, True])
def test_replot_checks_copied_run_instead_of_original(tmp_path, monkeypatch, style, original_present, tampered):
    original = tmp_path / "original"
    original.mkdir()
    data = original / "contacts.csv"
    data.write_text("validated original\n")
    digest = s.hashes([data])[str(data)]
    recorded = {"legacy_absolute": str(data), "legacy_relative": "results/original/contacts.csv",
                "run_relative": "contacts.csv"}[style]
    manifest = {"status": "data_complete", "validation": {}, "data_sha256": {recorded: digest}}
    if style == "run_relative":
        manifest["data_sha256_base"] = "."
    (original / "spatial_population.json").write_text(json.dumps(manifest))
    copied = tmp_path / "copied"
    shutil.copytree(original, copied)
    if tampered:
        (copied / "contacts.csv").write_text("modified copied input\n")
    if not original_present:
        data.unlink()
    # Resolution of legacy relative paths must not depend on the original cwd.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    rendered = []
    monkeypatch.setattr(cli, "render_population", lambda run, output: rendered.append(run.resolve()))
    output = tmp_path / "replot"
    assert cli.main(["spatial-all-plot", str(copied), "--output", str(output)]) == (1 if tampered else 0)
    assert rendered == ([] if tampered else [copied.resolve()])
    if not tampered:
        saved = json.loads((output / "replot.json").read_text())
        assert saved["data_sha256"] == {str(copied / "contacts.csv"): digest}
    else:
        assert not output.exists()


def test_run_data_rejects_escaping_relative_paths(tmp_path):
    outside = tmp_path / "outside.csv"
    outside.write_text("data\n")
    source = tmp_path / "source"
    source.mkdir()
    manifest = {"validation": {}, "data_sha256_base": ".",
                "data_sha256": {"../outside.csv": s.hashes([outside])[str(outside)]}}
    with pytest.raises(ValidationError, match="Invalid population data paths"):
        p.verify_run_data(source, manifest)


@pytest.mark.parametrize("new_connection", [False, True])
def test_population_queries_disconnected_neurons_even_without_fb_sites(tmp_path, monkeypatch, new_connection):
    neurons = pd.DataFrame({
        "bodyId": [25982, 26913, 11512, 515192], "type": ["PFNd", "PFNd", "hDeltaB", "hDeltaB"],
        "instance": ["PFNd(PB04)_L2_C1", "PFNd(PB04)_R2_C8", "hDeltaB_08_C7", "hDeltaB_07_C6"],
        "fb_column": [1, 8, 7, 6]})
    matrix = pd.DataFrame([[0, 0], [0, 3]], index=[25982, 26913], columns=[11512, 515192])
    raw = pd.DataFrame({"bodyId_pre": [26913] * 3, "bodyId_post": [515192] * 3,
                        "x_pre": [1, 2, 3], "x_post": [11, 12, 13],
                        "y_pre": [0] * 3, "y_post": [0] * 3, "z_pre": [0] * 3, "z_post": [0] * 3,
                        "roi_pre": [["FB", "FBl3"]] * 3, "roi_post": [["FB", "FBl3"]] * 3,
                        "compartment_post": ["axon"] * 3})
    points = pd.DataFrame({"bodyId": [26913] * 3 + [515192] * 3,
                           "type": ["pre"] * 3 + ["post"] * 3, "x": [1, 2, 3, 11, 12, 13],
                           "y": [0] * 6, "z": [0] * 6, "compartment": ["axon"] * 6})
    for layer in s.LAYERS:
        points[layer] = layer == "FBl3"
    roi_counts = pd.DataFrame({"bodyId": [26913, 515192], "roi": ["FB"] * 2,
                               "pre": [3, 0], "post": [0, 3]})
    meta = {"voxelSize": [8, 8, 8], "voxelUnits": "nanometers", "postHighAccuracyThreshold": .5}
    monkeypatch.setattr(p, "load_source", lambda *args, **kwargs: (neurons, matrix, {}))
    monkeypatch.setattr(p, "fetch_meta", lambda **kwargs: meta)

    def fetch_neurons(ids, **kwargs):
        assert set(ids) == set(neurons.bodyId)
        return neurons, roi_counts

    def fetch_sites(client, ids, confidence, *, allow_empty):
        assert set(ids) == set(neurons.bodyId)
        assert allow_empty
        return points

    queried = []

    def fetch_connections(pre, targets, *args, **kwargs):
        assert set(targets) == {11512, 515192}
        queried.append(pre)
        if pre == 25982:
            return raw.iloc[:0].copy()
        if new_connection:
            added = raw.iloc[:1].copy()
            added["bodyId_post"] = 11512
            return pd.concat([raw, added], ignore_index=True)
        return raw

    class Client:
        all_rois = ["FB", *s.LAYERS]
        primary_rois = ["FB"]

        def fetch_skeleton(self, body, heal):
            return pd.DataFrame({"rowId": [1], "link": [-1], "x": [0], "y": [0], "z": [0]})

        def fetch_roi_mesh(self, roi):
            return b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"

    monkeypatch.setattr(p, "fetch_neurons", fetch_neurons)
    monkeypatch.setattr(p, "fetch_fb_synapses", fetch_sites)
    monkeypatch.setattr(p, "fetch_synapse_connections", fetch_connections)
    monkeypatch.setattr(p, "REFERENCE_COUNTS", {"total": 3, "post_in_FB": 3, "both_in_FB": 3})
    monkeypatch.setattr(p, "render_population", lambda run: None)
    output = tmp_path / "population"
    if new_connection:
        with pytest.raises(ValidationError, match="saved positive pairs"):
            p.run_population(Client(), tmp_path, output, {})
        assert json.loads((output / "spatial_population.json").read_text())["status"] == "validation_failed"
    else:
        manifest = p.run_population(Client(), tmp_path, output, {})
        assert manifest["status"] == "complete"
        assert manifest["validation"]["neurons"] == 4
        assert manifest["validation"]["possible_neuron_pairs"] == 4
        assert set(pd.read_csv(output / "neurons.csv").bodyId) == set(neurons.bodyId)
        coverage = pd.read_csv(output / "column_coverage.csv")
        assert coverage.possible_pairs.sum() == 4
        assert coverage.connected_pairs.sum() == 1
        assert (coverage[coverage.column_post == 7].target_neurons == 1).all()
        assert manifest["data_sha256_base"] == "."
        assert "contacts.csv" in manifest["data_sha256"]
        assert p.verify_run_data(output, manifest)
    assert set(queried) == {25982, 26913}


def test_replot_gallery_includes_linked_tables_and_figures(tmp_path):
    source = tmp_path / "source"
    (source / "meshes").mkdir(parents=True)
    (source / "skeletons").mkdir()
    neurons = annotations().iloc[:2].copy()
    neurons["instance"] = ["PFNd_example_C1", "hDeltaB_example_C2"]
    neurons.to_csv(source / "neurons.csv", index=False)
    selected = p.annotate_contacts(contacts().iloc[2:].copy(), neurons)
    selected.to_csv(source / "contacts.csv", index=False)
    coverage = p.column_coverage(selected, neurons)
    coverage.to_csv(source / "column_coverage.csv", index=False)
    p.arbor_contacts(selected).to_csv(source / "contact_arbor_summary.csv", index=False)
    points = pd.DataFrame({"bodyId": [1, 2], "type": ["pre", "post"], "compartment": ["axon", "axon"],
                           "x": [10, 11], "y": [0, 0], "z": [0, 0]})
    for layer in s.LAYERS:
        points[layer] = layer == "FBl3"
    points.to_csv(source / "fb_synapses.csv", index=False)
    s.spatial_summary(points).merge(neurons.rename(columns={"type": "neuron_type"}), on="bodyId").to_csv(source / "arbor_spatial_summary.csv", index=False)
    (source / "meshes" / "FB.obj").write_text("v 0 0 0\nv 20 0 0\nv 0 20 20\nf 1 2 3\n")
    for body in [1, 2]:
        pd.DataFrame({"rowId": [1, 2], "link": [-1, 1], "x": [10, 11], "y": [0, 0], "z": [0, 0]}).to_csv(source / "skeletons" / f"{body}.csv", index=False)
    output = tmp_path / "gallery"
    p.render_population(source, output)

    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links = []

        def handle_starttag(self, tag, attrs):
            if tag == "a":
                self.links.extend(value for key, value in attrs if key == "href")

    parser = Links()
    parser.feed((output / "index.html").read_text())
    assert parser.links
    assert all((output / link).is_file() for link in parser.links)
    for name in p.GALLERY_TABLES:
        assert (output / name).read_bytes() == (source / name).read_bytes()
