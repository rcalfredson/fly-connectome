from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection

from fly_connectome import vdeltae_contact_audit as audit
from fly_connectome import vdeltae_contact_figures as figures
from fly_connectome.connectivity import ValidationError
from fly_connectome.inputs import sha256


@pytest.fixture
def contact_example():
    bodies = pd.DataFrame({
        "bodyId": range(1, 27), "type": ["vDeltaE"] * 18 + ["hDeltaG"] * 8,
        "instance": [f"synthetic_{i}" for i in range(26)],
        "annotation_column": [0, 0, *range(1, 17), *range(1, 9)],
    })
    rows = []
    for i, (pre, post, pre_site, compartment) in enumerate([
        (1, 19, 101, "dendrite"), (1, 20, 101, "axon"),
        (2, 19, 201, "axon"), (3, 21, 301, None), (4, 22, 401, "linker"),
    ]):
        row = {"bodyId_pre": pre, "bodyId_post": post, "contact_relationship_id": i}
        for side, site in [("pre", pre_site), ("post", 1900 + i)]:
            row.update({
                f"synapse_node_id_{side}": site, f"synapse_type_{side}": side,
                f"x_{side}": float(site % 100), f"y_{side}": 1., f"z_{side}": 2.,
                f"confidence_{side}": .9, f"FB_{side}": True,
                f"compartment_{side}": "axon" if side == "pre" else compartment,
                f"split_{side}": None, **{f"{layer}_{side}": None for layer in audit.LAYERS},
            })
        rows.append(row)
    raw = pd.DataFrame(rows)
    raw["FB_pre"] = raw.FB_pre.astype(object)
    raw.loc[2, "confidence_post"] = .5  # Strictly above .5, not >= .5.
    raw.loc[3, "FB_pre"] = None
    raw.loc[4, "FB_post"] = False
    points = pd.concat([
        raw.loc[raw[f"FB_{side}"].eq(True), [f"{c}_{side}" for c in ["bodyId", *audit.SITE_FIELDS]]]
        .rename(columns={f"{c}_{side}": c for c in ["bodyId", *audit.SITE_FIELDS]})
        for side in ["pre", "post"]
    ], ignore_index=True).drop_duplicates()
    standard = raw.loc[audit.standard_mask(raw, .5)]
    edges = standard.groupby(audit.PAIR).size().rename("weight").reset_index()
    edges["roiInfo"] = [json.dumps({"FB": {"post": int(
        standard.loc[standard.bodyId_pre.eq(r.bodyId_pre) & standard.bodyId_post.eq(r.bodyId_post), "FB_post"].eq(True).sum()
    )}}) for r in edges.itertuples()]
    return bodies, points, edges, raw


def test_query_is_directed_bounded_and_unfiltered(contact_example):
    bodies = contact_example[0]
    query = audit.contact_query(bodies)
    assert "-[contact:SynapsesTo]->" in query
    assert "id(contact) AS contact_relationship_id" in query
    assert "WITH DISTINCT n, m, a, b, contact" in query
    assert "confidence >" not in query and "FB = true" not in query
    assert "a.compartment AS compartment_pre" in query and "b.compartment AS compartment_post" in query
    assert "n.bodyId IN [1, 2," in query and "m.bodyId IN [19, 20," in query
    with pytest.raises(ValidationError):
        audit.contact_query(bodies.iloc[:2])


def test_polyadic_sites_threshold_roi_and_unknown_compartments_are_separate(contact_example):
    bodies, points, edges, raw = contact_example
    original = raw.copy(deep=True)
    audit.validate_contacts(raw, bodies, points, edges, .5)
    tables = audit.contact_tables(raw, bodies, .5)
    scopes = tables["scope_counts.csv"].set_index("scope")
    assert scopes.loc["all_reported__all_ROIs", "contacts"] == 5
    assert scopes.loc["all_reported__all_ROIs", "unique_pre_sites"] == 4
    assert scopes.loc["standard__all_ROIs", "contacts"] == 4
    assert scopes.loc["standard__post_FB", "contacts"] == 3
    assert scopes.loc["standard__both_FB", "contacts"] == 2
    assert scopes.loc["all_reported__all_ROIs", "missing_FB_pre_contacts"] == 1
    assert len(tables["body_pair_counts.csv"]) == 6 * 18 * 8
    matrix = tables["standard__post_FB_matrix.csv"].set_index("bodyId_pre")
    assert matrix.shape == (18, 8) and int(matrix.to_numpy().sum()) == 3
    assert matrix.loc[18].eq(0).all()
    annotated = tables["contacts_annotated.csv"]
    assert annotated.loc[annotated.bodyId_pre.isin([1, 2]), "annotation_column_pre"].eq(0).all()
    compartments = tables["compartment_counts.csv"]
    missing = compartments.loc[compartments.scope.eq("standard__post_FB") & compartments.compartment_post.isna()]
    assert missing.contacts.iloc[0] == 1
    assert missing.fraction_of_scope_contacts.iloc[0] == pytest.approx(1 / 3)
    assert pd.isna(raw.loc[3, "FB_pre"])
    pd.testing.assert_frame_equal(raw, original)


@pytest.mark.parametrize("problem", ["duplicate", "shared_site", "coordinate", "polarity", "membership", "site_property", "weight", "roi", "flag", "confidence", "missing_column"])
def test_contact_validation_rejects_inconsistent_data(contact_example, problem):
    bodies, points, edges, raw = contact_example
    raw, edges = raw.copy(), edges.copy()
    if problem == "duplicate":
        raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    elif problem == "shared_site":
        raw.loc[1, "compartment_pre"] = "dendrite"
    elif problem == "coordinate":
        raw.loc[0, "x_post"] = np.nan
    elif problem == "polarity":
        raw.loc[0, "synapse_type_post"] = "pre"
    elif problem == "membership":
        raw.loc[0, "bodyId_post"] = 1
    elif problem == "site_property":
        raw.loc[0, "x_post"] += 1
    elif problem == "weight":
        edges.loc[0, "weight"] += 1
    elif problem == "roi":
        edges.loc[0, "roiInfo"] = '{}'
    elif problem == "flag":
        raw["FB_post"] = raw.FB_post.astype(object)
        raw.loc[0, "FB_post"] = "true"
    elif problem == "confidence":
        raw.loc[0, "confidence_post"] = 1.1
    else:
        raw = raw.drop(columns="compartment_post")
    with pytest.raises(ValidationError):
        audit.validate_contacts(raw, bodies, points, edges, .5)


def test_missing_confidence_is_retained_but_not_standard(contact_example):
    bodies, points, edges, raw = contact_example
    raw.loc[2, "confidence_post"] = np.nan
    points.loc[points.synapse_node_id.eq(raw.loc[2, "synapse_node_id_post"]), "confidence"] = np.nan
    audit.validate_contacts(raw, bodies, points, edges, .5)
    scopes = audit.contact_tables(raw, bodies, .5)["scope_counts.csv"].set_index("scope")
    assert scopes.loc["all_reported__all_ROIs", "missing_confidence_post_contacts"] == 1
    assert scopes.loc["standard__all_ROIs", "contacts"] == 4


@pytest.fixture
def contact_environment(contact_example, tmp_path, monkeypatch):
    bodies, points, edges, raw = contact_example
    coords, census = tmp_path / "coordinates", tmp_path / "census"
    coords.mkdir()
    census.mkdir()
    metadata = {"uuid": "synthetic", "latestMutationId": 1, "lastDatabaseEdit": "synthetic",
                "voxelSize": [8, 8, 8], "voxelUnits": "nanometers", "postHighAccuracyThreshold": .5}
    (coords / "metadata_before.json").write_text(json.dumps(metadata))
    geometry = {"source_probe": str(tmp_path / "spatial"), "native_to_um_per_axis": [.008] * 3,
                "source_resources": {"FB_mesh": {"status": "retrieved"}},
                "source_inventory": {"annotation_audit_directory": str(tmp_path / "annotations"),
                    "sources": {"synthetic": {"source_manifest_file": str(census / "manifest.json")}}}}
    provenance = {"coordinates": str(coords), "census": str(census), "manifest_sha256": {"synthetic": "hash"}}
    monkeypatch.setattr(audit, "load_sources", lambda *args: (bodies.copy(), points.copy(), edges.copy(), deepcopy(geometry), deepcopy(provenance)))
    monkeypatch.setattr(figures, "load_sources", audit.load_sources)
    monkeypatch.setattr(audit, "fetch_meta", lambda **kwargs: deepcopy(metadata))
    monkeypatch.setattr(audit, "check_probe_metadata", lambda m, inventory: None if m == metadata else (_ for _ in ()).throw(ValidationError("Metadata changed")))
    (coords / "FB.obj").write_text("v -10 -10 0\nv 100 -10 0\nv 0 100 0\nf 1 2 3\n")
    for body in bodies.loc[bodies.type.eq("hDeltaG"), "bodyId"]:
        (coords / f"skeleton_{body}.swc").write_text("1 0 0 0 0 1 -1\n2 0 8 8 8 1 1\n")

    class Client:
        server = f"https://{audit.SERVER}"
        dataset = audit.DATASET

        def fetch_custom(self, query):
            if query == audit.contact_query(bodies):
                return raw.copy()
            return bodies[["bodyId", "type", "instance"]].copy()

    return Client(), coords, census, raw


def test_audit_roundtrip_and_recomputed_counts_detect_tampering(contact_environment, tmp_path):
    client, coords, census, _ = contact_environment
    output = tmp_path / "contacts"
    audit.run_contact_audit(client, coords, census, output)
    record = audit.verify_contact_audit(output)
    assert record["status"] == "complete"
    with pytest.raises(FileExistsError):
        audit.run_contact_audit(client, coords, census, output)
    path = output / "body_pair_counts.csv"
    table = pd.read_csv(path)
    table.loc[0, "contacts"] += 1
    table.to_csv(path, index=False)
    record["artifact_sha256"][path.name] = sha256(path)
    (output / "contacts.json").write_text(json.dumps(record))
    with pytest.raises(AssertionError):
        audit.verify_contact_audit(output)


def test_failure_preserves_raw_contacts_and_safe_error_type(contact_environment, tmp_path):
    client, coords, census, raw = contact_environment
    raw.loc[0, "bodyId_post"] = 999
    output = tmp_path / "failed"
    with pytest.raises(ValidationError):
        audit.run_contact_audit(client, coords, census, output)
    record = json.loads((output / "contacts.json").read_text())
    assert record["status"] == "failed" and record["error_type"] == "ValidationError"
    assert (output / "contacts_raw.csv").is_file()
    assert "error_message" not in record
    with pytest.raises(ValidationError, match="complete"):
        audit.verify_contact_audit(output)


def test_outputs_cannot_overlap_sources(contact_environment):
    client, coords, census, _ = contact_environment
    with pytest.raises(ValidationError, match="separate"):
        audit.run_contact_audit(client, coords, census, coords / "contacts")


def test_contact_figures_render_actual_counts_without_mutating_inputs(contact_example, tmp_path):
    bodies, points, _, raw = contact_example
    before = raw.copy(deep=True)
    table = audit.contact_tables(raw, bodies, .5)["contacts_annotated.csv"]
    receiver = next(bodies.loc[bodies.bodyId.eq(19)].itertuples())
    selected = table.loc[table.bodyId_post.eq(19) & audit.standard_mask(table, .5)]
    mesh = (np.array([[-10., -10., 0.], [100., -10., 0.], [0., 100., 0.]]), np.array([[0, 1, 2]]))
    skeleton = pd.DataFrame({"node_id": [1, 2], "parent_id": [-1, 1], "x": [0., 8.], "y": [0., 8.], "z": [0., 8.]})
    context = points.loc[points.bodyId.eq(19)]
    figure = figures.receiver_contact_figure(selected, receiver, bodies, context, skeleton, mesh, [.008] * 3)
    try:
        for ax in figure.axes:
            plotted = sum(len(artist.get_offsets()) for artist in ax.collections if isinstance(artist, PathCollection))
            assert plotted == len(context) + len(selected)
        figure.savefig(tmp_path / "receiver.png")
    finally:
        plt.close(figure)
    matrix = figures.coupling_figure(raw, bodies, .5)
    try:
        assert np.asarray(matrix.axes[0].images[0].get_array()).sum() == 5
        assert np.asarray(matrix.axes[1].images[0].get_array()).sum() == 4
        matrix.savefig(tmp_path / "matrix.png")
    finally:
        plt.close(matrix)
    pd.testing.assert_frame_equal(raw, before)


def test_contact_figure_export_and_changed_transform(contact_environment, tmp_path, monkeypatch):
    client, coords, census, _ = contact_environment
    run, output = tmp_path / "contacts", tmp_path / "figures"
    audit.run_contact_audit(client, coords, census, run)
    before = {p.name: sha256(p) for p in run.iterdir()}
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", lambda self, path, **kwargs: Path(path).write_bytes(b"synthetic image"))
    figures.export_contact_figures(run, output, source_compartment_panels=True, receiving_compartment_panels=True)
    record = figures.verify_contact_figures(output)
    assert len(list(output.glob("receiver_*.png"))) == 8
    assert record["subsampling"] is None
    assert (output / "source_compartment_coupling.png").is_file()
    assert (output / "source_dendrite_matrix.csv").is_file()
    assert (output / "receiving_compartment_coupling.png").is_file()
    assert (output / "receiving_dendrite_matrix.csv").is_file()
    assert {p.name: sha256(p) for p in run.iterdir()} == before
    assert not plt.get_fignums()
    record["translation_um"] = [1., 0., 0.]
    (output / "plots.json").write_text(json.dumps(record))
    with pytest.raises(ValidationError, match="transform"):
        figures.verify_contact_figures(output)


def test_cli_verification_does_not_create_a_client(contact_environment, tmp_path, capsys):
    client, coords, census, _ = contact_environment
    run = tmp_path / "contacts"
    audit.run_contact_audit(client, coords, census, run)
    assert audit.main(["verify", "--run", str(run)]) == 0
    assert "complete" in capsys.readouterr().out


@pytest.mark.parametrize("has_dendritic_source", [True, False])
def test_source_panels_use_source_compartment_and_one_color_scale(contact_example, tmp_path, has_dendritic_source):
    bodies, _, _, raw = contact_example
    before = raw.copy(deep=True)
    selected = raw.copy()
    if has_dendritic_source:
        selected.loc[[0, 1], "compartment_pre"] = "dendrite"
    tables = figures.source_compartment_matrices(selected, bodies, .5)
    axon_count = int(tables["source_axon_matrix.csv"].drop(columns="bodyId_pre").to_numpy().sum())
    dendrite_count = int(tables["source_dendrite_matrix.csv"].drop(columns="bodyId_pre").to_numpy().sum())
    assert (axon_count, dendrite_count) == ((1, 2) if has_dendritic_source else (3, 0))
    figure = figures.source_compartment_figure(selected, bodies, .5)
    try:
        assert len(figure.axes) == 3  # Two heatmaps, one colorbar.
        left, right = figure.axes[:2]
        assert left.images[0].norm is right.images[0].norm
        assert left.images[0].get_array().shape == (18, 8)
        assert int(left.images[0].get_array().sum()) == axon_count
        assert int(right.images[0].get_array().sum()) == dendrite_count
        figure.savefig(tmp_path / "source_compartments.png")
    finally:
        plt.close(figure)
    pd.testing.assert_frame_equal(raw, before)


def test_receiving_panels_match_original_lower_panels_exactly(contact_example, tmp_path):
    bodies, _, _, raw = contact_example
    raw = raw.copy()
    raw.loc[[0, 1], "compartment_pre"] = "dendrite"  # Must not change receiving classification.
    original = figures.coupling_figure(raw, bodies, .5)
    paired = figures.receiving_compartment_figure(raw, bodies, .5)
    try:
        for old_ax, new_ax in zip(original.axes[2:4], paired.axes[:2]):
            np.testing.assert_array_equal(old_ax.images[0].get_array(), new_ax.images[0].get_array())
        assert len(paired.axes) == 3
        assert paired.axes[0].images[0].norm is paired.axes[1].images[0].norm
        assert int(paired.axes[0].images[0].get_array().sum()) == 1
        assert int(paired.axes[1].images[0].get_array().sum()) == 1
        tables = figures.receiving_compartment_matrices(raw, bodies, .5)
        assert set(tables) == {"receiving_axon_matrix.csv", "receiving_dendrite_matrix.csv"}
        paired.savefig(tmp_path / "receiving.png")
    finally:
        plt.close(original)
        plt.close(paired)
