from copy import deepcopy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import numpy as np
import pandas as pd
import pytest

from fly_connectome import vdeltae_spatial_audit as spatial
from fly_connectome import vdeltae_spatial_pilot as pilot
from fly_connectome.connectivity import ValidationError
from fly_connectome.inputs import sha256
from fly_connectome.vdeltae_population import (
    population_figure_specs, population_projection_figure, population_spatial_summary,
)


@pytest.fixture
def bodies():
    columns = [0, 0, 1, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 7, 7, 7, 8, 9] + list(range(1, 9))
    return pd.DataFrame({
        "bodyId": range(1, 27), "type": ["vDeltaE"] * 18 + ["hDeltaG"] * 8,
        "instance": [f"vDeltaE_C{c}" for c in columns[:18]] + [f"hDeltaG_{i:02d}_C{c}" for i, c in enumerate(columns[18:], 1)],
        "annotation_column": columns,
    })


def test_population_selection_keeps_all_bodies_repeated_labels_and_edges(bodies):
    original = bodies.copy(deep=True)
    pd.testing.assert_frame_equal(spatial.select_spatial_population(bodies.sample(frac=1)), original)
    assert spatial.select_spatial_population(bodies).annotation_column.eq(0).sum() == 2
    pd.testing.assert_frame_equal(bodies, original)
    with pytest.raises(ValidationError, match="18 vDeltaE"):
        spatial.select_spatial_population(bodies.iloc[1:])


@pytest.mark.parametrize("ids", [[], list(range(1, 26)), list(range(1, 28)), [1] * 26, [0] + list(range(1, 26))])
def test_full_query_bounds_are_explicit(ids):
    with pytest.raises(ValidationError):
        spatial.spatial_probe_queries(ids, full_population=True)


def test_full_queries_do_not_relax_pilot_scope(bodies):
    with pytest.raises(ValidationError):
        pilot.fb_site_query(bodies.bodyId)
    query = pilot.fb_site_query(bodies.bodyId, full_population=True)
    assert "WHERE s.FB = true" in query and "SynapsesTo" not in query
    assert "s.confidence >" not in query
    with pytest.raises(ValidationError, match="boolean"):
        spatial.spatial_probe_queries(bodies.bodyId, full_population="yes")


@pytest.fixture
def points(bodies):
    rows = []
    for body in bodies.bodyId:
        for offset, compartment, polarity in [(0, "axon", "pre"), (1, "dendrite", "post")]:
            rows.append({
                "bodyId": body, "synapse_node_id": body * 10 + offset,
                "synapse_type": polarity, "compartment": compartment, "split": None,
                "x": body, "y": offset, "z": 0., "confidence": .1, "FB": True,
                **{layer: None for layer in pilot.LAYERS},
            })
    return pd.DataFrame(rows)


def test_spatial_summary_preserves_body_denominators_missing_groups_and_untyped(points, bodies):
    points = points.loc[points.bodyId.ne(26)].copy()
    points.loc[0, "compartment"] = None
    original = points.copy(deep=True)
    result = population_spatial_summary(points, bodies, [.008] * 3)
    first = result.loc[result.bodyId.eq(1)].set_index("compartment")
    assert first.loc["axon", "site_count"] == 0
    assert first.loc["unannotated", "fraction_of_body_FB_sites"] == .5
    assert first.loc["dendrite", "x_median_um"] == pytest.approx(.008)
    empty = result.loc[result.bodyId.eq(26)]
    assert empty.site_count.eq(0).all() and empty.x_median_um.isna().all()
    assert empty.fraction_of_body_FB_sites.isna().all()
    totals = result.groupby("bodyId").site_count.sum()
    assert totals.loc[1] == 2 and totals.loc[26] == 0
    assert set(result.loc[result.annotation_column.eq(0), "bodyId"]) == {1, 2}
    pd.testing.assert_frame_equal(points, original)


def test_overview_retains_every_site_and_pages_cover_every_body(points, bodies, tmp_path):
    selected = bodies.loc[bodies.type.eq("vDeltaE")]
    sites = points.loc[points.bodyId.isin(selected.bodyId)]
    mesh = (np.array([[-100., -100., 0.], [100., -100., 0.], [0., 100., 0.]]), np.array([[0, 1, 2]]))
    figure = population_projection_figure(sites, selected, [.008] * 3, mesh, "dendrite")
    try:
        for ax in figure.axes:
            assert sum(len(c.get_offsets()) for c in ax.collections if isinstance(c, PathCollection)) == 18
        figure.savefig(tmp_path / "overview.png")
    finally:
        plt.close(figure)
    pages = [spec[2] for spec in population_figure_specs(points, bodies) if spec[1] == "histogram"]
    assert all(len(page) <= 6 for page in pages)
    assert sorted(pd.concat(pages).bodyId) == sorted(bodies.bodyId)


@pytest.fixture
def population_environment(tmp_path, monkeypatch, bodies, points):
    metadata = {"uuid": "synthetic", "latestMutationId": 1, "lastDatabaseEdit": "synthetic",
                "voxelSize": [8, 8, 8], "voxelUnits": "nanometers", "postHighAccuracyThreshold": .5}
    inventory = {
        "annotation_audit_directory": str(tmp_path / "audit"),
        "sources": {"synthetic": {
            "source_manifest_file": str(tmp_path / "census" / "manifest.json"),
            "spatial_metadata": spatial.spatial_metadata_inventory(metadata),
        }},
    }
    monkeypatch.setattr(spatial, "inspect_spatial_sources", lambda path: (bodies.copy(), deepcopy(inventory)))
    monkeypatch.setattr(spatial, "fetch_meta", lambda **kwargs: deepcopy(metadata))
    monkeypatch.setattr(pilot, "fetch_meta", lambda **kwargs: deepcopy(metadata))
    tables = {
        "body_annotations": bodies[["bodyId", "type", "instance"]],
        "site_availability": pd.DataFrame({"bodyId": bodies.bodyId, **{col: 2 for col in spatial.PROBE_COUNT_COLUMNS}}),
        "site_property_inventory": pd.DataFrame([
            {"bodyId": row.bodyId, "synapse_type": row.synapse_type, "property": name, "sites_with_property": 1}
            for row in points.itertuples() for name in ["location", "confidence"]
        ]),
        "fb_site_categories": pilot.category_counts(points),
    }
    queries = spatial.spatial_probe_queries(bodies.bodyId, full_population=True)

    class Client:
        server = f"https://{spatial.SERVER}"
        dataset = spatial.DATASET

        def fetch_custom(self, query):
            if query == pilot.fb_site_query(bodies.bodyId, full_population=True):
                return points.copy()
            name = next(name for name, text in queries.items() if text == query)
            return tables[name].copy()

        def fetch_skeleton(self, body, **kwargs):
            assert kwargs == {"heal": False, "format": "swc"}
            return "# synthetic\n1 0 0 0 0 1 -1\n2 0 16 0 0 1 1\n"

        def fetch_roi_mesh(self, roi):
            assert roi == "FB"
            return b"v -100 -100 0\nv 100 -100 0\nv 0 100 0\nf 1 2 3\n"

    return Client(), tmp_path / "audit"


def test_full_population_pipeline_roundtrip_and_semantic_tampering(population_environment, tmp_path, monkeypatch):
    client, audit = population_environment
    source, coordinates, figures = [tmp_path / name for name in ["source", "coordinates", "figures"]]
    spatial.run_spatial_data_probe(client, audit, source, full_population=True)
    assert len(spatial.verify_spatial_data_probe(source)["selected_body_ids"]) == 26
    pilot.run_coordinate_pilot(client, source, coordinates)
    assert pilot.verify_coordinate_pilot(coordinates)["point_count"] == 52
    before = {p.name: sha256(p) for p in coordinates.iterdir()}
    # Exercise rendering and export inventories without repeatedly encoding dozens of images.
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", lambda self, path, **kwargs: Path(path).write_bytes(b"synthetic image"))
    pilot.export_coordinate_pilot(coordinates, figures, mesh_overlay=True)
    record = pilot.verify_coordinate_exports(figures)
    assert len(list(figures.glob("body_*.png"))) == 26
    assert len(list(figures.glob("*_axis_distributions_*.png"))) == 5
    assert {p.name: sha256(p) for p in coordinates.iterdir()} == before
    assert not plt.get_fignums()
    table = pd.read_csv(figures / "column_spatial_summary.csv")
    table.loc[0, "site_count"] += 1
    table.to_csv(figures / "column_spatial_summary.csv", index=False)
    record["artifact_sha256"]["column_spatial_summary.csv"] = sha256(figures / "column_spatial_summary.csv")
    (figures / "plots.json").write_text(json.dumps(record))
    with pytest.raises(AssertionError):
        pilot.verify_coordinate_exports(figures)


def test_population_verifier_rejects_changed_membership(population_environment, tmp_path):
    client, audit = population_environment
    output = tmp_path / "source"
    spatial.run_spatial_data_probe(client, audit, output, full_population=True)
    record = json.loads((output / "capability.json").read_text())
    record["selected_body_ids"][0] = 999
    (output / "capability.json").write_text(json.dumps(record))
    with pytest.raises(ValidationError, match="selection"):
        spatial.verify_spatial_data_probe(output)
