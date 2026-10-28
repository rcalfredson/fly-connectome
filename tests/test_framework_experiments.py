from dataclasses import asdict
import json

import numpy as np
import pandas as pd
import pytest

from fly_connectome import framework_experiments as f
from fly_connectome.connectivity import ValidationError
from fly_connectome.movement import MovementParameters, movement_weights
from fly_connectome.trajectory import replay


def save_manifest(directory, name):
    f.save_json(directory / name, dict(status="complete", data_validated=True,
        data_sha256={p.name: f.sha256(p) for p in directory.iterdir() if p.is_file()}))


def test_saved_pfn_adapter_preserves_sources_and_rejects_tampering(tmp_path):
    movement, trajectory = tmp_path / "movement", tmp_path / "trajectory"
    movement.mkdir()
    trajectory.mkdir()
    p = MovementParameters(normalize_incoming_by_group=True, pfnd_weight_multiplier=6.)
    mapping = pd.DataFrame(dict(bodyId=[1, 2, 3, 4], type=["PFNd", "PFNd", "PFNv", "PFNv"],
        pb_side=["L", "R", "L", "R"], preferred_heading_deg=[0, 90, 180, -90],
        preferred_translation_deg=[-31, 31, 137, -137]))
    counts = pd.DataFrame([[2, 0], [1, 3], [0, 4], [2, 1]], index=[1, 2, 3, 4], columns=[10, 20])
    weights = movement_weights(counts, mapping, p)
    config = dict(parameters=asdict(p), source_ids=[1, 2, 3, 4], target_ids=[10, 20])
    f.save_json(movement / "movement_config.json", dict(config, baseline_levels=[1.]))
    mapping.to_csv(movement / "PFN_encoding_map.csv", index=False)
    counts.to_csv(movement / "raw_contact_matrix.csv")
    for directory in [movement, trajectory]:
        pd.DataFrame(weights, index=counts.index, columns=counts.columns).to_csv(directory / "effective_weight_matrix.csv")
    (movement / "baseline_1").mkdir()
    for representation in ["raw", "movement_evoked"]:
        pd.DataFrame(dict(relation=["same_world_different_heading"], centered_cosine=[.7])).to_csv(
            movement / "baseline_1" / f"{representation}_pair_comparisons.csv", index=False)
    # Hash nested files explicitly, as the real movement manifest does.
    f.save_json(movement / "movement_model.json", dict(status="complete", data_validated=True,
        data_sha256={str(p.relative_to(movement)): f.sha256(p) for p in movement.rglob("*") if p.is_file()}))
    protocol = pd.DataFrame(dict(time_s=[0, .1, .2, .3, .4, .5], heading_deg=[0, 0, 90, 180, 270, 270],
        forward_mm_s=[0, 1, 1, -1, 0, 0], right_mm_s=[0, 0, 1, 0, 0, 0]))
    inputs, prescribed, drive, activity, exact = replay(protocol, mapping, weights, p)
    protocol.to_csv(trajectory / "protocol.csv", index=False)
    f.save_json(trajectory / "trajectory_config.json", dict(config, baseline=1.))
    np.savez_compressed(trajectory / "activity.npz", pfn_activity=prescribed, hdeltab_activity=activity)
    save_manifest(trajectory, "trajectory_model.json")
    output = tmp_path / "comparison"
    result = f.run_pfn(movement, trajectory, output)
    assert result["status"] == "complete" and all(result["validation"].values())
    assert f.verify(output)["status"] == "complete"
    assert (output / "pfn_comparison.png").stat().st_size > 1000
    inherited = pd.read_csv(output / "inherited_heading_differences.csv")
    np.testing.assert_allclose(inherited.mean_centered_cosine, .7)
    assert pd.read_csv(output / "coupling.csv").contacts.sum() == counts.to_numpy().sum()
    with pytest.raises(FileExistsError):
        f.run_pfn(movement, trajectory, output)
    (movement / "PFN_encoding_map.csv").write_text("tampered")
    with pytest.raises(ValidationError, match="changed"):
        f.run_pfn(movement, trajectory, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("mode", ["centered", "absolute"])
def test_janke_demo_exports_verifies_and_refuses_overwrite(tmp_path, mode):
    config = f.default_janke_config()
    config.update(input_mode=mode, duration_s=4, reset_times_s=[0, 3], dt_s=.01,
        experiments=[dict(name="test_pulse", events=[dict(start_s=1, stop_s=2, amplitude=.05, phase_deg=0)])])
    output = tmp_path / "janke"
    result = f.run_janke(output, config)
    assert result["status"] == "complete" and all(result["validation"].values())
    f.verify(output)
    assert (output / "test_pulse.pdf").stat().st_size > 1000
    readout = pd.read_csv(output / "test_pulse_candidate_rate_readout.csv")
    assert readout.phase_rad.isna().any()
    with np.load(output / "test_pulse_activity.npz") as data:
        assert (data["candidate_rate"] >= 0).all()
        assert (data["reference_fluorescence"] < 0).any()
        if mode == "centered":
            np.testing.assert_allclose(data["fluorescence_proxy"], data["reference_fluorescence"], atol=1e-12)
        else:
            assert not np.allclose(data["fluorescence_proxy"], data["reference_fluorescence"])
    with np.load(output / "test_pulse_latent_control.npz") as control:
        np.testing.assert_allclose(control["candidate_latent_proxy"], control["mathematical_control"], atol=1e-12)
    metrics = pd.read_csv(output / "numerical_comparisons.csv")
    assert metrics.input_mode.eq(mode).all()
    with pytest.raises(FileExistsError):
        f.run_janke(output, config)
    (output / "candidate_assumptions.json").write_text("changed")
    with pytest.raises(ValidationError, match="changed"):
        f.verify(output)


def test_failed_candidate_has_failed_manifest_not_success(tmp_path):
    config = f.default_janke_config()
    config["dt_s"] = .03  # Does not divide its native 1-second reference interval.
    with pytest.raises(ValidationError, match="divide"):
        f.run_janke(tmp_path / "failed", config)
    manifest = json.loads((tmp_path / "failed" / "framework_run.json").read_text())
    assert manifest["status"] == "failed" and manifest["error_type"] == "ValidationError"
    with pytest.raises(ValidationError, match="complete"):
        f.verify(tmp_path / "failed")
