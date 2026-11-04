import argparse
import copy
import json

import numpy as np
import pytest

from fly_connectome.home_vector.__main__ import begin, finish
from fly_connectome.home_vector.data import digest, verify_bundle, write_json
from fly_connectome.home_vector.profile_training import batch_description, profile_training
from test_home_vector_data import circuit_fixture


def prepared(path):
    output, record = begin(path, "home_vector_prepared_v1")
    write_json(output / "circuit.json", circuit_fixture().as_dict())
    write_json(output / "task.json", dict(dt_s=.0025, length_scale_mm=2.))
    rows = [dict(family=f"short-{i}", kind="random",
                 protocol=[[0, i * 90, .5, .2], [duration, i * 90, .5, .2]])
            for i, duration in enumerate([.025, .05, .075])]
    write_json(output / "trajectories.json", {split: rows for split in ["train", "validation", "test", "challenge"]})
    finish(output, record)
    return output


def arguments(source, output):
    return argparse.Namespace(source=str(source), output=str(output), seed=0, updates=3,
                              warmup=0, batch_size=2, threads=1, device="cpu", trace_seconds=0.)


@pytest.mark.parametrize("batching", ["padded", "length-groups"])
def test_profile_matches_canonical_trainer_loss_gradients_and_optimizer(tmp_path, batching):
    torch = pytest.importorskip("torch")
    from fly_connectome.home_vector.__main__ import train
    source = prepared(tmp_path / "source")
    hashes = {p.name: digest(p) for p in source.iterdir()}
    args = arguments(source, tmp_path / "profile")
    args.batching, args.length_ratio = batching, 1.
    result = profile_training(args)
    assert all(v["passed"] for v in result["agreement"].values())
    assert result["baseline_mean_update_s"] > 0
    assert result["mean_phases_s"]["backward"] > 0
    assert verify_bundle(args.output)["status"] == "complete"
    assert hashes == {p.name: digest(p) for p in source.iterdir()}
    canonical = copy.copy(args)
    canonical.output = str(tmp_path / "canonical")
    canonical.rewire_seed, canonical.no_recurrence = None, False
    canonical.benchmark, canonical.max_seconds, canonical.validate_every = True, 60., 50
    train(canonical)
    actual = torch.load(tmp_path / "profile" / "baseline_final.pt", weights_only=False)
    expected = torch.load(tmp_path / "canonical" / "last.pt", weights_only=False)
    for name, value in expected["model"].items():
        torch.testing.assert_close(actual["model"][name], value, atol=0, rtol=0)
    assert actual["optimizer"]["param_groups"] == expected["optimizer"]["param_groups"]
    for key, fields in expected["optimizer"]["state"].items():
        for name, value in fields.items():
            torch.testing.assert_close(actual["optimizer"]["state"][key][name], value, atol=0, rtol=0)
    a = json.loads((tmp_path / "profile" / "baseline_updates.json").read_text())
    b = json.loads((tmp_path / "canonical" / "history.json").read_text())
    assert [r["loss"] for r in a] == [r["loss"] for r in b]
    assert [r["gradient_norm"] for r in a] == [r["gradient_norm"] for r in b]
    with pytest.raises(FileExistsError):
        profile_training(args)


def test_padding_counts_include_endpoint_and_indices_are_serializable():
    rows = [dict(protocol=[[0, 0, 1, 0], [t, 0, 1, 0]]) for t in [1, 2]]
    result = batch_description(np.array([4, 8]), rows, .5)
    assert result["valid_steps"] == 8
    assert result["padded_steps"] == 10
    assert result["padding_fraction"] == pytest.approx(.2)
    json.dumps(result)


def test_warmup_is_recorded_but_excluded_from_summary_and_short_trace_is_optional(tmp_path):
    pytest.importorskip("torch")
    args = arguments(prepared(tmp_path / "source"), tmp_path / "profile")
    args.warmup, args.updates, args.trace_seconds = 1, 1, .025
    result = profile_training(args)
    rows = json.loads((tmp_path / "profile" / "baseline_updates.json").read_text())
    assert [r["measured"] for r in rows] == [False, True]
    assert result["baseline_mean_update_s"] == rows[1]["wall_s"]
    assert (tmp_path / "profile" / "operator_trace.json").is_file()
    assert json.loads((tmp_path / "profile" / "operators.json").read_text())
    assert verify_bundle(args.output)["status"] == "complete"


def test_source_change_marks_profile_failed(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import fly_connectome.home_vector.profile_training as module
    args = arguments(prepared(tmp_path / "source"), tmp_path / "profile")
    args.updates = 1
    original = module.compare_passes

    def change_source(a, b):
        result = original(a, b)
        (tmp_path / "source" / "unexpected.txt").write_text("source changed during profiling")
        return result

    monkeypatch.setattr(module, "compare_passes", change_source)
    with pytest.raises(ValueError, match="source files changed"):
        profile_training(args)
    manifest = json.loads((tmp_path / "profile" / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    with pytest.raises(ValueError, match="complete"):
        verify_bundle(args.output)


def test_keyboard_interrupt_is_not_reported_as_complete(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import fly_connectome.home_vector.profile_training as module
    args = arguments(prepared(tmp_path / "source"), tmp_path / "profile")

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("user interrupted the disposable profile")

    monkeypatch.setattr(module, "run_pass", interrupt)
    with pytest.raises(KeyboardInterrupt):
        profile_training(args)
    manifest = json.loads((tmp_path / "profile" / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "KeyboardInterrupt" in manifest["error"]
