import json
import pandas as pd
import pytest

from fly_connectome import vdeltae_column_audit as audit
from fly_connectome.connectivity import ValidationError
from fly_connectome.vdeltae_column_audit import (
    assemble_body_table,
    load_population,
    parse_annotations,
)


def test_parser_retains_body_rows_and_distinguishes_index_from_column():
    original = pd.DataFrame(
        {
            "bodyId": [1, 2, 3, 4, 5],
            "type": ["vDeltaE"] * 4 + ["hDeltaG"],
            "instance": [
                "vDeltaE_C0",
                "vDeltaE_C9",
                "vDeltaE_C3",
                "vDeltaE_C3",
                "hDeltaG_14_C8",
            ],
        },
        index=[10, 20, 30, 40, 50],
    )
    before = original.copy(deep=True)
    result = parse_annotations(original)
    pd.testing.assert_frame_equal(original, before)
    pd.testing.assert_frame_equal(result[original.columns], original)
    assert result["annotation_column"].tolist() == [0, 9, 3, 3, 8]
    assert result["instance_index"].iloc[:4].isna().all()
    assert result["instance_index"].iloc[4] == 14
    assert str(result["annotation_column"].dtype) == "Int64"
    assert result["annotation_parse_status"].eq("matched").all()


@pytest.mark.parametrize(
    "name, status",
    [
        (None, "missing_instance"),
        ("", "missing_instance"),
        ("   ", "missing_instance"),
        ("vDeltaE_C3_extra", "unrecognized_instance"),
        ("prefix_vDeltaE_C3", "unrecognized_instance"),
        ("hDeltaG_01_C1", "unrecognized_instance"),
        (17, "unrecognized_instance"),
    ],
)
def test_unresolved_names_are_preserved_without_inventing_columns(name, status):
    original = pd.DataFrame({"bodyId": [1], "type": ["vDeltaE"], "instance": [name]})
    result = parse_annotations(original)
    pd.testing.assert_frame_equal(result[original.columns], original)
    assert len(result) == 1
    assert result["annotation_column"].isna().all()
    assert result["instance_index"].isna().all()
    assert result["annotation_parse_status"].iloc[0] == status


def test_assembly_preserves_annotations_and_rejects_conflicting_ids(tmp_path):

    v_path = tmp_path / "v.csv"
    h_path = tmp_path / "h.csv"
    pd.DataFrame(
        {
            "bodyId": [1, 2],
            "type": ["vDeltaE", "other"],
            "instance": ["unexpected_name", "other_C1"],
            "v_only_note": ["keep this", "not selected"],
        }
    ).to_csv(v_path, index=False)
    hdeltag = pd.DataFrame(
        {
            "bodyId": [3],
            "type": ["hDeltaG"],
            "instance": ["hDeltaG_14_C8"],
            "h_only_note": ["keep this too"],
        }
    )
    hdeltag.to_csv(h_path, index=False)

    result = assemble_body_table(v_path, h_path).set_index("bodyId")
    assert set(result.index) == {1, 3}
    assert result.loc[1, "v_only_note"] == "keep this"
    assert result.loc[3, "h_only_note"] == "keep this too"
    assert pd.isna(result.loc[3, "v_only_note"])
    assert result.loc[1, "annotation_parse_status"] == "unrecognized_instance"
    assert pd.isna(result.loc[1, "annotation_column"])
    assert result.loc[3, "annotation_column"] == 8
    assert result.loc[1, "source_file"] == str(v_path.resolve())
    assert result.loc[3, "source_file"] == str(h_path.resolve())
    assert result["anatomical_mapping_status"].eq("unreviewed").all()

    hdeltag["bodyId"] = 1
    hdeltag.to_csv(h_path, index=False)
    with pytest.raises(ValidationError, match="both selected populations"):
        assemble_body_table(v_path, h_path)


@pytest.mark.parametrize("missing_column", ["bodyId", "type", "instance"])
def test_loader_rejects_missing_required_columns(tmp_path, missing_column):
    path = tmp_path / "neurons.csv"
    table = pd.DataFrame(
        {"bodyId": [1], "type": ["vDeltaE"], "instance": ["vDeltaE_C1"]}
    )
    table.drop(columns=[missing_column]).to_csv(path, index=False)

    with pytest.raises(ValidationError, match="Neuron table needs"):
        load_population(path, "vDeltaE")


def test_loader_rejects_absent_exact_type(tmp_path):
    path = tmp_path / "neurons.csv"
    pd.DataFrame(
        {
            "bodyId": [1],
            "type": ["vDeltaE_other"],
            "instance": ["vDeltaE_C1"],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValidationError, match="No exact-type vDeltaE"):
        load_population(path, "vDeltaE")


@pytest.mark.parametrize("bad_id", [None, 0, -1])
def test_loader_rejects_missing_or_nonpositive_ids(tmp_path, bad_id):
    path = tmp_path / "neurons.csv"
    pd.DataFrame(
        {
            "bodyId": [bad_id],
            "type": ["vDeltaE"],
            "instance": ["vDeltaE_C1"],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValidationError, match="present and positive"):
        load_population(path, "vDeltaE")


def test_loader_rejects_duplicate_selected_ids(tmp_path):
    path = tmp_path / "neurons.csv"
    pd.DataFrame(
        {
            "bodyId": [1, 1],
            "type": ["vDeltaE", "vDeltaE"],
            "instance": ["vDeltaE_C1", "vDeltaE_C2"],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValidationError, match="Duplicate body IDs"):
        load_population(path, "vDeltaE")


def test_loader_rejects_fractional_ids(tmp_path):
    path = tmp_path / "neurons.csv"
    pd.DataFrame(
        {
            "bodyId": [1.5],
            "type": ["vDeltaE"],
            "instance": ["vDeltaE_C1"],
        }
    ).to_csv(path, index=False)

    with pytest.raises((TypeError, ValueError)):
        load_population(path, "vDeltaE")


@pytest.fixture
def provenance_sources(tmp_path, monkeypatch):
    directories = {}
    calls = []
    for role, filename, csv_text in [
        (
            "vdeltae_census",
            "vDeltaE_neurons.csv",
            "bodyId,type,instance\n1,vDeltaE,vDeltaE_C1\n",
        ),
        (
            "hdeltab_census",
            "candidate_neurons.csv",
            "bodyId,type,instance\n2,hDeltaG,hDeltaG_01_C1\n",
        ),
    ]:
        directory = tmp_path / role
        directory.mkdir()
        annotation = directory / filename
        annotation.write_text(csv_text)
        manifest = {
            "status": "complete",
            "server": audit.SERVER,
            "dataset": audit.DATASET,
            "dataset_metadata": {
                "uuid": "synthetic-dataset",
                "latestMutationId": 1,
                "lastDatabaseEdit": "synthetic-edit-marker",
            },
            "data_sha256": {filename: audit.sha256(annotation)},
        }
        (directory / "manifest.json").write_text(json.dumps(manifest))
        directories[role] = directory

    def verify_vdeltae(directory):
        calls.append(("vdeltae_census", directory))
        return json.loads((directory / "manifest.json").read_text())

    def verify_hdeltab(directory):
        calls.append(("hdeltab_census", directory))
        manifest = json.loads((directory / "manifest.json").read_text())
        return manifest, None, None, None

    monkeypatch.setattr(audit, "verify_vdeltae_census", verify_vdeltae)
    monkeypatch.setattr(audit, "validate_saved_census", verify_hdeltab)
    return directories, calls


def test_provenance_records_each_files_own_hash(provenance_sources):
    directories, calls = provenance_sources
    sources = audit.collect_source_provenance(
        directories["vdeltae_census"], directories["hdeltab_census"]
    )
    assert calls == list(directories.items())
    assert set(sources) == set(directories)
    for role, filename in [
        ("vdeltae_census", "vDeltaE_neurons.csv"),
        ("hdeltab_census", "candidate_neurons.csv"),
    ]:
        directory = directories[role].resolve()
        record = sources[role]
        assert record["directory"] == str(directory)
        assert record["manifest_file"] == str(directory / "manifest.json")
        assert record["annotation_file"] == str(directory / filename)
        assert record["manifest_sha256"] == audit.sha256(directory / "manifest.json")
        assert record["annotation_sha256"] == audit.sha256(directory / filename)
        assert record["manifest_sha256"] != record["annotation_sha256"]
        assert record["server"] == audit.SERVER
        assert record["dataset"] == audit.DATASET
    assert sources["vdeltae_census"]["dataset_identity"] == {
        "uuid": "synthetic-dataset",
        "latestMutationId": 1,
        "lastDatabaseEdit": "synthetic-edit-marker",
    }
    assert (
        sources["vdeltae_census"]["dataset_identity"]
        == sources["hdeltab_census"]["dataset_identity"]
    )


@pytest.mark.parametrize("field", ["uuid", "latestMutationId", "lastDatabaseEdit"])
def test_provenance_rejects_different_dataset_states(provenance_sources, field):
    directories, _ = provenance_sources
    path = directories["hdeltab_census"] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["dataset_metadata"][field] = "different-value"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValidationError, match="different dataset states"):
        audit.collect_source_provenance(
            directories["vdeltae_census"], directories["hdeltab_census"]
        )


@pytest.mark.parametrize("field", ["uuid", "latestMutationId", "lastDatabaseEdit"])
def test_provenance_rejects_missing_identity(provenance_sources, field):
    directories, _ = provenance_sources
    path = directories["vdeltae_census"] / "manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["dataset_metadata"][field]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValidationError, match="identity fields are missing"):
        audit.collect_source_provenance(
            directories["vdeltae_census"], directories["hdeltab_census"]
        )


@pytest.mark.parametrize("field", ["server", "dataset"])
def test_provenance_rejects_wrong_dataset_location(provenance_sources, field):
    directories, _ = provenance_sources
    path = directories["vdeltae_census"] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = "different-value"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValidationError, match="Source server or dataset differs"):
        audit.collect_source_provenance(
            directories["vdeltae_census"], directories["hdeltab_census"]
        )


@pytest.mark.parametrize("problem", ["missing", "incorrect"])
def test_provenance_requires_recorded_annotation_hash(provenance_sources, problem):
    directories, _ = provenance_sources
    path = directories["vdeltae_census"] / "manifest.json"
    manifest = json.loads(path.read_text())
    if problem == "missing":
        del manifest["data_sha256"]["vDeltaE_neurons.csv"]
    else:
        manifest["data_sha256"]["vDeltaE_neurons.csv"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(
        ValidationError, match="not covered by its source manifest hash"
    ):
        audit.collect_source_provenance(
            directories["vdeltae_census"], directories["hdeltab_census"]
        )


@pytest.mark.parametrize(
    "role, filename, verifier_name",
    [
        ("vdeltae_census", "manifest.json", "verify_vdeltae_census"),
        ("vdeltae_census", "vDeltaE_neurons.csv", "verify_vdeltae_census"),
        ("hdeltab_census", "manifest.json", "validate_saved_census"),
        ("hdeltab_census", "candidate_neurons.csv", "validate_saved_census"),
    ],
)
def test_provenance_rejects_changes_during_verification(
    provenance_sources, monkeypatch, role, filename, verifier_name
):
    directories, _ = provenance_sources
    original_verifier = getattr(audit, verifier_name)

    def changing_verifier(directory):
        result = original_verifier(directory)
        path = directory / filename
        path.write_text(path.read_text() + "\n")
        return result

    monkeypatch.setattr(audit, verifier_name, changing_verifier)
    with pytest.raises(ValidationError, match="changed during verification"):
        audit.collect_source_provenance(
            directories["vdeltae_census"], directories["hdeltab_census"]
        )
    assert (role, directories[role]) in provenance_sources[1]


@pytest.mark.parametrize(
    "verifier_name", ["verify_vdeltae_census", "validate_saved_census"]
)
def test_provenance_propagates_verifier_failure(
    provenance_sources, monkeypatch, verifier_name
):
    directories, _ = provenance_sources

    def failing_verifier(directory):
        raise ValidationError("Synthetic census verification failed.")

    monkeypatch.setattr(audit, verifier_name, failing_verifier)
    with pytest.raises(ValidationError, match="Synthetic census verification failed"):
        audit.collect_source_provenance(
            directories["vdeltae_census"], directories["hdeltab_census"]
        )


def test_audit_command_route_trip(provenance_sources, tmp_path, capsys):
    directories, _ = provenance_sources
    output = tmp_path / "audit"
    audit.main(
        [
            "run",
            "--vdeltae-census",
            str(directories["vdeltae_census"]),
            "--hdeltab-census",
            str(directories["hdeltab_census"]),
            "--output",
            str(output),
        ]
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["population_counts"] == {"vDeltaE": 1, "hDeltaG": 1}
    assert manifest["parse_status_counts"] == {"matched": 2}
    assert "vdeltae_column_audit.py" in manifest["code_sha256"]
    bodies = pd.read_csv(output / "body_annotations.csv")
    assert set(bodies["bodyId"]) == {1, 2}
    assert bodies["anatomical_mapping_status"].eq("unreviewed").all()
    audit.main(["verify", str(output)])
    assert "Annotation audit verified:" in capsys.readouterr().out


def test_audit_refuses_existing_output(provenance_sources, tmp_path):
    directories, _ = provenance_sources
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        audit.run_audit(
            directories["vdeltae_census"], directories["hdeltab_census"], output
        )
    assert marker.read_text() == "preserve"
    assert not (output / "manifest.json").exists()


def test_audit_refuses_output_inside_source(provenance_sources):
    directories, _ = provenance_sources
    output = directories["vdeltae_census"] / "audit"
    with pytest.raises(ValidationError, match="separate from source"):
        audit.run_audit(
            directories["vdeltae_census"], directories["hdeltab_census"], output
        )
    assert not output.exists()


def test_audit_records_verifier_failure(provenance_sources, tmp_path, monkeypatch):
    directories, _ = provenance_sources
    output = tmp_path / "failed"

    def failing_verifier(directory):
        raise ValidationError("Synthetic source failure.")

    monkeypatch.setattr(audit, "verify_vdeltae_census", failing_verifier)
    with pytest.raises(ValidationError, match="Synthetic source failure"):
        audit.run_audit(
            directories["vdeltae_census"], directories["hdeltab_census"], output
        )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error_type"] == "ValidationError"
    assert manifest["finished_at_utc"]
    with pytest.raises(ValidationError, match="complete annotation audit"):
        audit.verify_audit(output)


def test_audit_rechecks_sources_after_assembly(
    provenance_sources, tmp_path, monkeypatch
):
    directories, _ = provenance_sources
    output = tmp_path / "changed-source"
    original_assemble = audit.assemble_body_table

    def changing_assemble(v_path, h_path):
        bodies = original_assemble(v_path, h_path)
        path = directories["vdeltae_census"] / "vDeltaE_neurons.csv"
        path.write_text(path.read_text() + "\n")
        return bodies

    monkeypatch.setattr(audit, "assemble_body_table", changing_assemble)
    with pytest.raises(ValidationError, match="changed after verification"):
        audit.run_audit(
            directories["vdeltae_census"], directories["hdeltab_census"], output
        )
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("filename", audit.AUDIT_FILES)
@pytest.mark.parametrize("problem", ["changed", "missing"])
def test_audit_detects_changed_or_missing_outputs(
    provenance_sources, tmp_path, filename, problem
):
    directories, _ = provenance_sources
    output = tmp_path / "audit"
    audit.run_audit(
        directories["vdeltae_census"], directories["hdeltab_census"], output
    )
    path = output / filename
    if problem == "changed":
        path.write_text(path.read_text() + "\n")
    else:
        path.unlink()
    with pytest.raises(
        ValidationError, match="artifact is missing, outside the run, or changed"
    ):
        audit.verify_audit(output)


def test_audit_verification_rechecks_saved_sources(provenance_sources, tmp_path):
    directories, _ = provenance_sources
    output = tmp_path / "audit"
    audit.run_audit(
        directories["vdeltae_census"], directories["hdeltab_census"], output
    )
    path = directories["hdeltab_census"] / "manifest.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValidationError, match="changed after verification"):
        audit.verify_audit(output)
