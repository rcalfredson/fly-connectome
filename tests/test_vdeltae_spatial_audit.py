import json

import pytest

from fly_connectome.connectivity import ValidationError
from fly_connectome.vdeltae_spatial_audit import spatial_metadata_inventory


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
    assert result['coordinate_metadata'] == {'voxelSize': None, 'voxelUnits': None}
    assert result['fb_roi_names_in_metadata'] == []

@pytest.mark.parametrize('properties', ['invalid JSON', [], '[]'])
def test_unrecognized_property_schema_is_rejected(properties):
    with pytest.raises(ValidationError):
        spatial_metadata_inventory({'neuronProperties': properties})
