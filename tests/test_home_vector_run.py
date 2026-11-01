import argparse
import json

import numpy as np
import pytest

from fly_connectome.home_vector.__main__ import (baseline, begin, fail, finish, metrics,
                                                prepare, load)
from fly_connectome.home_vector.data import datasets, write_json, verify_bundle
from test_home_vector_data import circuit_fixture


def make_bundle(path):
    output, record = begin(path, "synthetic_test_bundle")
    c = circuit_fixture()
    write_json(output / "circuit.json", c.as_dict())
    write_json(output / "task.json", dict(dt_s=.0025, length_scale_mm=1.))
    write_json(output / "trajectories.json", datasets(4, dict(train=4, validation=2, test=2)))
    np.save(output / "legacy_weights.npy", np.ones((2, 3)) * .001)
    finish(output, record)
    return output


def test_ridge_end_to_end_and_fresh_output_enforcement(tmp_path):
    source = make_bundle(tmp_path / "source")
    args = argparse.Namespace(source=str(source), output=str(tmp_path / "ridge"))
    result = baseline(args)
    assert set(result) == {"current_inputs", "legacy_frozen", "initialized_frozen"}
    assert verify_bundle(args.output)["status"] == "complete"
    assert load(source)[0].dynamic_ids == [10, 11, 12]
    with pytest.raises(FileExistsError):
        baseline(args)


def test_failed_run_is_not_verifiable_and_source_tampering_is_detected(tmp_path):
    output, record = begin(tmp_path / "failed", "test")
    fail(output, record, ValueError("example"))
    with pytest.raises(ValueError, match="complete"):
        verify_bundle(output)
    source = make_bundle(tmp_path / "source")
    (source / "task.json").write_text('{}')
    with pytest.raises(ValueError, match="hash mismatch"):
        load(source)


def test_metrics_do_not_hide_undefined_predicted_direction():
    target = [np.array([[0., 0.], [3., 4.]])]
    prediction = [np.zeros((2, 2))]
    result = metrics(prediction, target)
    assert result["normalized_rmse"] == 1
    assert result["near_home_samples"] == 1
    assert result["direction_failures"] == 1
    assert result["median_angle_error_deg"] == 180
    assert metrics([np.zeros((1, 2))], [np.zeros((1, 2))])["normalized_rmse"] is None


def test_invalid_dt_rejected_before_reading_sources():
    with pytest.raises(ValueError, match="timestep"):
        prepare(argparse.Namespace(dt=.1))
