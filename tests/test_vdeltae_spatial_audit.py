import json

import pandas as pd
import pytest

from fly_connectome.connectivity import ValidationError
from fly_connectome.vdeltae_spatial_audit import (
    select_spatial_probe,
    spatial_metadata_inventory,
    spatial_probe_queries,
)


@pytest.mark.parametrize("encoding", ["dictionary", "json_string"])
def test_metadata_inventory_does_not_infer_column_geometry(encoding):
    properties = {"instance": "string", "somaLocation": "point"}
    if encoding == "json_string":
        properties = json.dumps(properties)
    result = spatial_metadata_inventory(
        {
            "neuronProperties": properties,
            "voxelSize": [8, 8, 8],
            "voxelUnits": "nanometers",
            "roiInfo": {"FB": {}, "FBl6": {}, "PB": {}},
        }
    )
    assert result["fb_roi_names_in_metadata"] == ["FB", "FBl6"]
    assert result["coordinate_metadata"]["voxelSize"] == [8, 8, 8]
    assert result["declared_neuron_properties"]["somaLocation"] == "point"
    assert result["column_annotation_semantics"] == "unresolved"
    assert result["anatomical_axis_orientation"] == "unverified"
    assert result["recorded_post_high_accuracy_threshold"] is None


def test_missing_coordinate_metadata_is_not_filled_with_defaults():
    result = spatial_metadata_inventory({})
    assert result["coordinate_metadata"] == {"voxelSize": None, "voxelUnits": None}
    assert result["fb_roi_names_in_metadata"] == []


@pytest.mark.parametrize("properties", ["invalid JSON", [], "[]"])
def test_unrecognized_property_schema_is_rejected(properties):
    with pytest.raises(ValidationError):
        spatial_metadata_inventory({"neuronProperties": properties})


def test_probe_selection_is_independent_of_row_order_and_retains_annotations():
    bodies = pd.DataFrame(
        {
            "bodyId": [20, 3, 10, 1],
            "type": ["vDeltaE", "hDeltaG", "vDeltaE", "hDeltaG"],
            "instance": ["vDeltaE_C0", "hDeltaG_01_C1", "vDeltaE_C9", "hDeltaG_14_C8"],
            "annotation_column": [0, 1, 9, 8],
        }
    )
    original = bodies.copy(deep=True)
    selected = select_spatial_probe(bodies)
    assert selected["bodyId"].tolist() == [10, 1]
    assert selected["annotation_column"].tolist() == [9, 8]
    pd.testing.assert_frame_equal(bodies, original)
    pd.testing.assert_frame_equal(selected, select_spatial_probe(bodies.iloc[::-1]))


def test_probe_selection_requires_both_populations():
    bodies = pd.DataFrame(
        {
            "bodyId": [1],
            "type": ["vDeltaE"],
            "instance": ["vDeltaE_C0"],
            "annotation_column": [0],
        }
    )
    with pytest.raises(ValidationError, match="both audited populations"):
        select_spatial_probe(bodies)


@pytest.mark.parametrize("ids", [[1], [1, 1], [1, 2, 3], [0, 2], [None, 2], [1.5, 2]])
def test_probe_queries_reject_invalid_or_unbounded_selection(ids):
    with pytest.raises(ValidationError):
        spatial_probe_queries(ids)


def test_probe_queries_preserve_site_denominators_and_unknown_categories():
    queries = spatial_probe_queries([20, 10])
    assert set(queries) == {
        "body_annotations",
        "site_availability",
        "site_property_inventory",
        "fb_site_categories",
    }
    assert all("n.bodyId IN [10, 20]" in query for query in queries.values())
    assert all(
        "ConnectsTo" not in query and "SynapsesTo" not in query
        for query in queries.values()
    )
    assert "OPTIONAL MATCH" in queries["site_availability"]
    assert "WITH DISTINCT n, s" in queries["site_property_inventory"]
    assert "keys(s)" in queries["site_property_inventory"]
    assert "s.confidence >" not in "\n".join(queries.values())
    assert "s.compartment AS compartment" in queries["fb_site_categories"]
    assert "coalesce" not in queries["fb_site_categories"]
