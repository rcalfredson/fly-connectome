import copy
import json

import numpy as np
import pytest

from fly_connectome.home_vector.__main__ import begin, finish, load
from fly_connectome.home_vector.data import datasets, digest, sample_protocol, verify_bundle, write_json
from fly_connectome.home_vector.long_training import augment, prepare, split_controls
from test_home_vector_data import circuit_fixture


def test_grouped_split_coverage_and_historical_challenge_holdout():
    controls = split_controls(20261104)
    assert controls == split_controls(20261104)
    assert controls != split_controls(12)
    sets = {s: set(rows) for s, rows in controls.items()}
    assert [len(sets[s]) for s in ("train", "validation", "test")] == [256, 32, 32]
    assert not (sets["train"] & sets["validation"] or sets["train"] & sets["test"]
                or sets["validation"] & sets["test"])
    assert (1, 45, 0) in sets["test"]
    assert len(set.union(*sets.values())) == 320
    for speed in [.25, .5, 1, 2]:
        for heading in range(0, 360, 45):
            assert sum(c[:2] == (speed, heading) for c in controls["train"]) == 8


def test_augmentation_preserves_originals_and_holds_out_longer_durations():
    groups = datasets()
    before = copy.deepcopy(groups)
    result, controls = augment(groups, 20261104)
    assert groups == before
    assert {s: len(rows) for s, rows in result.items()} == dict(
        train=2048, validation=192, test=320, challenge=12, extrapolation=128)
    for split in groups:
        assert result[split][:len(groups[split])] == groups[split]
    assert max(r["protocol"][-1][0] for r in result["train"]) == 64
    assert {r["protocol"][-1][0] for r in result["extrapolation"]} == {96, 128}
    assert {r["control_status"] for r in result["extrapolation"]} == {"seen", "held_out"}
    # Real task integration, not the metadata, establishes the physical units.
    row = next(r for r in result["train"] if r.get("speed_mm_s") == 2 and r["protocol"][-1][0] == 64)
    _, target = sample_protocol(row["protocol"], .0025)
    assert np.linalg.norm(target[-1]) == pytest.approx(128)


def test_preparation_preserves_circuit_loss_scale_and_provenance(tmp_path):
    source, record = begin(tmp_path / "base", "home_vector_prepared_v1", circuit_variant="core")
    write_json(source / "circuit.json", circuit_fixture().as_dict())
    write_json(source / "task.json", dict(dt_s=.0025, length_scale_mm=2.0450918869))
    write_json(source / "trajectories.json", datasets())
    write_json(source / "sources.json", {"example": "synthetic"})
    np.save(source / "legacy_weights.npy", np.zeros((2, 3)))
    finish(source, record)
    before = {p.name: digest(p) for p in source.iterdir()}
    output = tmp_path / "long"
    prepare(source, output)
    assert verify_bundle(output)["status"] == "complete"
    assert load(output)[2]["length_scale_mm"] == 2.0450918869
    for name in ["circuit.json", "sources.json", "legacy_weights.npy"]:
        assert digest(output / name) == before[name]
    assert before == {p.name: digest(p) for p in source.iterdir()}
    with pytest.raises(FileExistsError):
        prepare(source, output)
    design = json.loads((output / "long_training_design.json").read_text())
    assert design["long_training_sampling_probability"] == .5


def test_cross_distribution_sweep_requires_explicit_opt_in_same_circuit_and_dt(tmp_path):
    from types import SimpleNamespace
    from fly_connectome.home_vector.analysis.straight import validate_comparison
    audits = []
    for name in ["old", "new"]:
        folder = tmp_path / name
        folder.mkdir()
        (folder / "circuit.json").write_text('{"same": "anatomy and encoding"}')
        (folder / "task.json").write_text(name)
        (folder / "trajectories.json").write_text(name)
        audits.append(SimpleNamespace(run=folder, source=folder, task={"dt_s": .0025}))
    with pytest.raises(ValueError, match="matched task"):
        validate_comparison(*audits, False)
    validate_comparison(*audits, True)
    audits[1].task["dt_s"] = .00125
    with pytest.raises(ValueError, match="identical circuits and time steps"):
        validate_comparison(*audits, True)
    audits[1].task["dt_s"] = .0025
    (audits[1].run / "circuit.json").write_text('{"different": "circuit"}')
    with pytest.raises(ValueError, match="identical circuits and time steps"):
        validate_comparison(*audits, True)
