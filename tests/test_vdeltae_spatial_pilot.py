import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from fly_connectome.connectivity import ValidationError
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
