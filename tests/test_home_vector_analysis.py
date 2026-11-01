import argparse
import json
import shutil

import numpy as np
import pytest

from fly_connectome.home_vector.__main__ import begin, finish
from fly_connectome.home_vector.analysis.__main__ import main
from fly_connectome.home_vector.analysis.common import Audit
from fly_connectome.home_vector.analysis.statistics import direction_statistics
from fly_connectome.home_vector.data import digest, sample_protocol, verify_bundle, write_json
from test_home_vector_run import make_bundle


def audit_inputs(tmp_path):
    source = make_bundle(tmp_path / "source")
    groups = json.loads((source / "trajectories.json").read_text())
    for row in groups["test"]:
        for segment in row["protocol"]:
            segment[2] *= 10
            segment[3] *= 10
    write_json(source / "trajectories.json", groups)
    finish(source, json.loads((source / "manifest.json").read_text()))
    run, record = begin(tmp_path / "run", "home_vector_training_v1",
                        source=str(source), source_manifest_sha256=digest(source / "manifest.json"),
                        seed=0, recurrence=True)
    shutil.copyfile(source / "circuit.json", run / "circuit.json")
    rows = groups["test"]
    arrays = {}
    for i, row in enumerate(rows):
        _, target = sample_protocol(row["protocol"], .0025)
        arrays[f"prediction_{i}"] = target
        arrays[f"target_{i}"] = target
    np.savez(run / "test_predictions.npz", **arrays)
    finish(run, record)
    args = argparse.Namespace(command="statistics", run=str(run), source=str(source),
                              output=str(tmp_path / "audit"))
    return args, record


def test_statistics_cli_accepts_relocated_bundles_and_keeps_inputs_unchanged(tmp_path):
    args, _ = audit_inputs(tmp_path)
    old_source = args.source
    relocated = tmp_path / "relocated-data"
    shutil.move(old_source, relocated)
    args.source = str(relocated)
    audit = Audit(args)
    # The run still records the old path; source identity is verified by manifest hash.
    assert audit.manifest["settings"]["source"] == old_source
    main(["statistics", args.run, "--source", args.source, "--output", args.output])
    report = json.loads((audit.out / "statistics.json").read_text())
    assert report["groups"]["all"]["fraction_angle_within15deg"] == 1
    assert report["groups"]["all"]["fraction_vector_error_over1mm"] == 0
    assert report["groups"]["all"]["parallel_fraction_squared_error"] is None
    manifest = verify_bundle(audit.out)
    assert manifest["source_sha256"] == audit.hashes
    assert manifest["code_sha256"]
    with pytest.raises(FileExistsError):
        Audit(args)


def test_wrong_source_rejected_before_creating_output(tmp_path):
    args, _ = audit_inputs(tmp_path)
    other = make_bundle(tmp_path / "other-source")
    args.source = str(other)
    with pytest.raises(ValueError, match="source manifest"):
        Audit(args)
    assert not (tmp_path / "audit").exists()


def test_failed_and_mutating_audits_cannot_claim_completion(tmp_path):
    args, _ = audit_inputs(tmp_path)
    audit = Audit(args)
    with pytest.raises(ValueError, match="deliberate failure"):
        with audit:
            raise ValueError("deliberate failure")
    assert json.loads((audit.out / "manifest.json").read_text())["status"] == "failed"
    args.output = str(tmp_path / "changed-input-audit")
    with pytest.raises(ValueError, match="changed during audit"):
        with Audit(args):
            # Alter an unlisted file too; the integrity snapshot covers all input files.
            (audit.run / "extra.txt").write_text("changed")
    with pytest.raises(ValueError, match="complete"):
        verify_bundle(args.output)


def test_output_cannot_overlap_inputs_and_prediction_schema_is_checked(tmp_path):
    args, record = audit_inputs(tmp_path)
    original_output = args.output
    args.output = str(tmp_path / "source" / "nested")
    with pytest.raises(ValueError, match="separate"):
        Audit(args)
    args.output = original_output
    run = tmp_path / "run"
    np.savez(run / "test_predictions.npz", prediction_0=np.zeros((1, 2)))
    finish(run, record)
    with pytest.raises(ValueError, match="trajectory count"):
        Audit(args).predictions("test")


def test_direction_statistics_known_errors_and_undefined_subsets():
    truth = [np.array([[2., 0.], [2., 0.], [2., 0.]])]
    prediction = [np.array([[1., 0.], [2., 2.], [0., 0.]])]
    result = direction_statistics(prediction, truth)
    assert result["non_near_home_samples"] == 3
    assert result["fraction_angle_within15deg"] == pytest.approx(1 / 3)
    assert result["parallel_rmse_mm"] == pytest.approx(np.sqrt(5 / 3))
    assert result["perpendicular_rmse_mm"] == pytest.approx(np.sqrt(4 / 3))
    assert result["parallel_fraction_squared_error"] == pytest.approx(5 / 9)
    assert result["fraction_angle_within15deg_among_over1mm_errors"] == 0
    empty = direction_statistics([np.zeros((2, 2))], [np.zeros((2, 2))])
    assert empty["median_angle_error_deg"] is None


def test_independent_geometry_checks_turns_and_closure():
    from fly_connectome.home_vector.analysis.predictions import independent_target
    protocol = [[0, 0, 1, 0], [1, 90, 1, 0], [2, 180, 1, 0],
                [3, 270, 1, 0], [4, 270, 0, 0]]
    target = independent_target(protocol, 17, .25)
    np.testing.assert_allclose(target[4], [-1, 0], atol=1e-12)
    np.testing.assert_allclose(target[-1], 0, atol=1e-12)


def test_pause_selection_uses_metadata_instead_of_fixed_indices():
    pytest.importorskip("torch")
    from fly_connectome.home_vector.analysis.perturbations import pause_pair
    rows = [dict(family="other")]
    for speed in (-1, 1):
        rows.append(dict(family="pause", kind="matched_pause", suffix_start_s=3,
                         protocol=[[0, 0, speed, 0], [3, 90, 0, 0], [5, 90, 0, 0]]))
    indices, records = pause_pair(rows, "pause")
    assert indices == [1, 2]
    assert records[0]["suffix_start_s"] == 3
    with pytest.raises(ValueError, match="exactly two"):
        pause_pair(rows, "missing")
    rows[2]["protocol"][1][2] = 1
    with pytest.raises(AssertionError):
        pause_pair(rows, "pause")
