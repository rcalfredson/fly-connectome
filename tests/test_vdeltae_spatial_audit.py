from copy import deepcopy
import json

import pandas as pd
import pytest
from requests import Response
from requests.exceptions import HTTPError, Timeout

from fly_connectome.connectivity import ValidationError
from fly_connectome import vdeltae_spatial_audit as spatial
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


@pytest.fixture
def spatial_probe_environment(tmp_path, monkeypatch):
    probe = pd.DataFrame(
        {
            "bodyId": [10, 1],
            "type": ["vDeltaE", "hDeltaG"],
            "instance": ["vDeltaE_C6", "hDeltaG_14_C8"],
            "annotation_column": [6, 8],
        }
    )
    metadata = {
        "dataset": "male-cns",
        "uuid": "test",
        "latestMutationId": 1,
        "lastDatabaseEdit": "test",
        "voxelSize": [8, 8, 8],
        "voxelUnits": "nanometers",
        "postHighAccuracyThreshold": 0.5,
    }
    source = {
        "source_manifest_file": str(tmp_path / "source" / "manifest.json"),
        "source_manifest_sha256": "synthetic-hash",
        "spatial_metadata": spatial.spatial_metadata_inventory(metadata),
    }
    inventory = {
        "annotation_audit_directory": str(tmp_path / "audit"),
        "annotation_audit_manifest_sha256": "synthetic-audit-hash",
        "sources": {
            "vdeltae_census": deepcopy(source),
            "hdeltab_census": deepcopy(source),
        },
    }
    tables = {
        "body_annotations": probe[["bodyId", "type", "instance"]].copy(),
        "site_availability": pd.DataFrame(
            {
                "bodyId": [10, 1],
                "all_contained_sites": [1, 1],
                "sites_marked_FB": [1, 1],
                "sites_with_location": [1, 1],
                "sites_with_confidence": [1, 1],
            }
        ),
        "site_property_inventory": pd.DataFrame(
            [
                {
                    "bodyId": body,
                    "synapse_type": polarity,
                    "property": prop,
                    "sites_with_property": 1,
                }
                for body, polarity in [(10, "pre"), (1, "post")]
                for prop in ["location", "confidence"]
            ]
        ),
        "fb_site_categories": pd.DataFrame(
            {
                "bodyId": [10, 1],
                "synapse_type": ["pre", "post"],
                "compartment": [None, None],
                "split": [None, None],
                "site_count": [1, 1],
            }
        ),
    }

    class FakeClient:
        server = f"https://{spatial.SERVER}"
        dataset = spatial.DATASET

        def fetch_custom(self, query):
            name = next(
                name
                for name, text in spatial.spatial_probe_queries(probe["bodyId"]).items()
                if text == query
            )
            return tables[name].copy(deep=True)

        def fetch_skeleton(self, body, heal, format):
            assert body in [10, 1] and heal is False and format == "swc"
            return "1 0 0 0 0 1 -1\n"

        def fetch_roi_mesh(self, roi):
            assert roi == "FB"
            return b"v 0 0 0\n"

    monkeypatch.setattr(
        spatial,
        "inspect_spatial_sources",
        lambda path: (probe.copy(), deepcopy(inventory)),
    )
    monkeypatch.setattr(spatial, "fetch_meta", lambda **kwargs: deepcopy(metadata))
    return FakeClient(), probe, inventory, metadata, tables


def test_spatial_probe_run_and_offline_verification(
    spatial_probe_environment, tmp_path
):
    client, probe, inventory, _, _ = spatial_probe_environment
    output = tmp_path / "probe"
    result = spatial.run_spatial_data_probe(
        client, inventory["annotation_audit_directory"], output
    )
    assert result["status"] == "complete"
    assert result["selected_body_ids"] == [10, 1]
    assert set(result["resources"]) == {"skeleton_10", "skeleton_1", "FB_mesh"}
    assert spatial.verify_spatial_data_probe(output)["status"] == "complete"


def test_spatial_probe_refuses_existing_output(spatial_probe_environment, tmp_path):
    client, _, inventory, _, _ = spatial_probe_environment
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        spatial.run_spatial_data_probe(
            client, inventory["annotation_audit_directory"], output
        )
    assert list(output.iterdir()) == []


def test_spatial_probe_records_metadata_drift(spatial_probe_environment, tmp_path):
    client, _, inventory, metadata, _ = spatial_probe_environment
    metadata["latestMutationId"] = 2
    output = tmp_path / "drift"
    with pytest.raises(ValidationError, match="metadata differs"):
        spatial.run_spatial_data_probe(
            client, inventory["annotation_audit_directory"], output
        )
    result = json.loads((output / "capability.json").read_text())
    assert result["status"] == "failed"
    assert "metadata_before.json" in result["artifact_sha256"]
    assert not (output / "body_annotations.csv").exists()


def test_spatial_probe_saves_raw_tables_before_reconciliation_failure(
    spatial_probe_environment, tmp_path
):
    client, _, inventory, _, tables = spatial_probe_environment
    tables["fb_site_categories"].loc[0, "site_count"] = 2
    output = tmp_path / "bad-counts"
    with pytest.raises(ValidationError, match="FB categories do not reconcile"):
        spatial.run_spatial_data_probe(
            client, inventory["annotation_audit_directory"], output
        )
    result = json.loads((output / "capability.json").read_text())
    assert result["status"] == "failed"
    assert "fb_site_categories.csv" in result["artifact_sha256"]


@pytest.mark.parametrize(
    "code, expected_status",
    [
        (404, "http_not_found"),
        (401, "authentication_or_access_error"),
        (500, "http_error"),
    ],
)
def test_resource_http_failures_remain_distinct(tmp_path, code, expected_status):
    def fetch():
        response = Response()
        response.status_code = code
        raise HTTPError("not recorded", response=response)

    result = spatial.retrieve_probe_resource(fetch, tmp_path / "resource.obj")
    assert result["status"] == expected_status
    assert result["http_status"] == code
    assert "not recorded" not in json.dumps(result)


def test_resource_timeout_is_not_an_absence(tmp_path):
    def fetch():
        raise Timeout("not recorded")

    assert (
        spatial.retrieve_probe_resource(fetch, tmp_path / "resource.obj")["status"]
        == "timeout"
    )


def test_empty_resource_payload_is_preserved(tmp_path):
    path = tmp_path / "resource.obj"
    record = spatial.retrieve_probe_resource(lambda: b"", path)
    assert record["status"] == "empty_response"
    assert path.is_file() and record["bytes"] == 0


def test_resource_access_error_fails_the_run(
    spatial_probe_environment, tmp_path, monkeypatch
):
    client, _, inventory, _, _ = spatial_probe_environment

    def denied(*args, **kwargs):
        response = Response()
        response.status_code = 403
        raise HTTPError("not recorded", response=response)

    monkeypatch.setattr(client, "fetch_skeleton", denied)
    output = tmp_path / "denied"
    with pytest.raises(ValidationError, match="Optional resource request failed"):
        spatial.run_spatial_data_probe(
            client, inventory["annotation_audit_directory"], output
        )
    result = json.loads((output / "capability.json").read_text())
    assert result["status"] == "failed"
    assert result["resources"]["skeleton_10"]["http_status"] == 403
    assert "FB_mesh" not in result["resources"]


def test_spatial_probe_detects_changed_saved_output(
    spatial_probe_environment, tmp_path
):
    client, _, inventory, _, _ = spatial_probe_environment
    output = tmp_path / "probe"
    spatial.run_spatial_data_probe(
        client, inventory["annotation_audit_directory"], output
    )
    (output / "site_availability.csv").write_text("changed")
    with pytest.raises(
        ValidationError, match="artifact is missing, outside the run, or changed"
    ):
        spatial.verify_spatial_data_probe(output)


@pytest.mark.parametrize("problem", ["missing_body", "changed_instance"])
def test_spatial_probe_rejects_live_identity_changes(
    spatial_probe_environment, tmp_path, problem
):
    client, _, inventory, _, tables = spatial_probe_environment
    if problem == "missing_body":
        tables["body_annotations"] = tables["body_annotations"].iloc[:1].copy()
    else:
        tables["body_annotations"].loc[0, "instance"] = "changed"
    output = tmp_path / "identity-change"
    with pytest.raises(ValidationError):
        spatial.run_spatial_data_probe(
            client, inventory["annotation_audit_directory"], output
        )
    assert json.loads((output / "capability.json").read_text())["status"] == "failed"


def test_spatial_probe_rechecks_live_metadata_at_end(
    spatial_probe_environment, tmp_path, monkeypatch
):
    client, _, inventory, metadata, _ = spatial_probe_environment
    calls = []

    def metadata_fetcher(**kwargs):
        result = deepcopy(metadata)
        if calls:
            result["latestMutationId"] = 2
        calls.append(True)
        return result

    monkeypatch.setattr(spatial, "fetch_meta", metadata_fetcher)
    output = tmp_path / "late-metadata-drift"
    with pytest.raises(ValidationError, match="metadata differs"):
        spatial.run_spatial_data_probe(
            client, inventory["annotation_audit_directory"], output
        )
    assert len(calls) == 2
    result = json.loads((output / "capability.json").read_text())
    assert result["status"] == "failed"
    assert "metadata_after.json" in result["artifact_sha256"]


def test_spatial_probe_rechecks_saved_source_inventory(
    spatial_probe_environment, tmp_path, monkeypatch
):
    client, probe, inventory, _, _ = spatial_probe_environment
    calls = []

    def inspection(path):
        result = deepcopy(inventory)
        if calls:
            result["annotation_audit_manifest_sha256"] = "changed"
        calls.append(True)
        return probe.copy(), result

    monkeypatch.setattr(spatial, "inspect_spatial_sources", inspection)
    output = tmp_path / "late-source-drift"
    with pytest.raises(ValidationError, match="sources changed"):
        spatial.run_spatial_data_probe(
            client, inventory["annotation_audit_directory"], output
        )
    assert len(calls) == 2
    assert json.loads((output / "capability.json").read_text())["status"] == "failed"


def test_spatial_probe_preserves_optional_not_found_response(
    spatial_probe_environment, tmp_path, monkeypatch
):
    client, _, inventory, _, _ = spatial_probe_environment

    def missing_mesh(roi):
        response = Response()
        response.status_code = 404
        raise HTTPError("not recorded", response=response)

    monkeypatch.setattr(client, "fetch_roi_mesh", missing_mesh)
    output = tmp_path / "missing-mesh"
    result = spatial.run_spatial_data_probe(
        client, inventory["annotation_audit_directory"], output
    )
    assert result["status"] == "complete"
    assert result["resources"]["FB_mesh"]["status"] == "http_not_found"
    assert not (output / "FB.obj").exists()
    assert spatial.verify_spatial_data_probe(output)["status"] == "complete"


def test_zero_site_rows_are_retained_without_inventing_categories(
    spatial_probe_environment, tmp_path
):
    client, _, inventory, _, tables = spatial_probe_environment
    tables["site_availability"].loc[:, list(spatial.PROBE_COUNT_COLUMNS)] = 0
    tables["site_property_inventory"] = (
        tables["site_property_inventory"].iloc[:0].copy()
    )
    tables["fb_site_categories"] = tables["fb_site_categories"].iloc[:0].copy()
    output = tmp_path / "zero-sites"
    spatial.run_spatial_data_probe(
        client, inventory["annotation_audit_directory"], output
    )
    assert spatial.verify_spatial_data_probe(output)["status"] == "complete"
    assert len(pd.read_csv(output / "site_availability.csv")) == 2
