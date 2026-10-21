import json

import matplotlib.figure
import numpy as np
import pandas as pd
import pytest

from fly_connectome.connectivity import ValidationError
from fly_connectome import movement_decoder as d


def order_fixture():
    rows = []
    for i, column, count in d.POSITIONS:
        for _ in range(count):
            rows.append({"bodyId": len(rows) + 100, "instance": f"hDeltaB_{i:02d}_C{column}"})
    return pd.DataFrame(rows)


def test_exact_published_inventory_and_duplicate_averaging():
    mapping = d.decoder_mapping(order_fixture().sample(frac=1, random_state=8))
    profiles = pd.DataFrame(0., index=["one", "duplicate"], columns=[str(i) for i in mapping.bodyId])
    first = str(mapping.loc[mapping.position.eq(0), "bodyId"].iloc[0])
    duplicate = mapping.loc[mapping.position.eq(9), "bodyId"].astype(str)
    profiles.loc["one", first] = 1.
    profiles.loc["duplicate", duplicate.iloc[0]] = 3.
    result, bins = d.decode_profiles(profiles, mapping)
    assert bins.loc["duplicate", "position_09"] == pytest.approx(1.)
    assert result.loc["one", "phase_deg"] == pytest.approx(-180.)
    assert result.loc["one", "published_decoded_deg"] == pytest.approx(-180 - d.PUBLISHED_OFFSET_DEG)
    shuffled, _ = d.decode_profiles(profiles.sample(frac=1, axis=1, random_state=9), mapping)
    pd.testing.assert_frame_equal(result, shuffled)
    invalid = order_fixture()
    invalid.loc[0, "instance"] = "hDeltaB_99_C7"
    with pytest.raises(ValidationError, match="inventory"):
        d.decoder_mapping(invalid)


def test_independent_cosine_oracle_and_undefined_vectors():
    mapping = d.decoder_mapping(order_fixture())
    angles = np.deg2rad(mapping.phase_angle_deg.to_numpy())
    desired = np.array([-133., 0., 79.])
    activity = 2 + np.cos(angles[None, :] - np.deg2rad(desired[:, None]))
    activity = np.vstack([activity, np.zeros(19), np.ones(19)])
    frame = pd.DataFrame(activity, columns=mapping.bodyId.astype(str))
    result, _ = d.decode_profiles(frame, mapping)
    np.testing.assert_allclose(result.phase_deg[:3], desired, atol=1e-12)
    np.testing.assert_allclose(result.resultant_length[:3], .25)
    assert not result.valid[3:].any()
    assert result.published_decoded_deg[3:].isna().all()
    with pytest.raises(ValidationError, match="identities"):
        d.decode_profiles(frame.iloc[:, :-1], mapping)
    frame.iloc[0, 0] = -1
    with pytest.raises(ValidationError, match="nonnegative"):
        d.decode_profiles(frame, mapping)


@pytest.mark.parametrize("silent", [False, True])
def test_saved_run_decoder_handles_fractional_baselines_and_undefined_angles(tmp_path, monkeypatch, silent):
    source = tmp_path / "source"
    source.mkdir()
    order = order_fixture()
    order.to_csv(source / "hDeltaB_anatomical_order.csv", index=False)
    (source / "movement_config.json").write_text(json.dumps({"baseline_levels": [.5], "parameters": {}}))
    pd.DataFrame({"trial": ["moving", "rest"], "heading_deg": [0, 0],
                  "world_direction_deg": [0., np.nan], "speed_mm_s": [1., 0.],
                  "primary_grid": [True, False]}).to_csv(source / "conditions.csv", index=False)
    mapping = d.decoder_mapping(order)
    activity = 2 + np.cos(np.deg2rad(mapping.phase_angle_deg.to_numpy()))
    if silent:
        activity[:] = 0
    profiles = pd.DataFrame([activity, np.zeros(19)], index=["moving", "rest"],
                            columns=mapping.bodyId.astype(str))
    folder = source / "baseline_0p5"
    folder.mkdir()
    for filename in ["steady_activity.csv", "movement_evoked_activity.csv"]:
        profiles.to_csv(folder / filename)
    hashes = {str(p.relative_to(source)): d.sha256(p) for p in source.rglob("*") if p.is_file()}
    (source / "movement_model.json").write_text(json.dumps({"data_validated": True, "data_sha256": hashes}))
    # Exercise real decoding and saved-table handling; raster rendering is already checked on experiment outputs.
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", lambda *a, **k: None)
    output = tmp_path / "decoded"
    manifest = d.run_decoder(source, output, {})
    assert manifest["status"] == "complete"
    summary = pd.read_csv(output / "decoder_summary.csv")
    assert summary.valid_primary_trials.eq(0 if silent else 1).all()
    if silent:
        assert summary.mean_absolute_error_published_convention_deg.isna().all()
    else:
        np.testing.assert_allclose(summary.mean_absolute_error_published_convention_deg, abs(d.PUBLISHED_OFFSET_DEG))
    saved = pd.read_csv(output / "baseline_0p5/movement_evoked_decoded_trials.csv", index_col=0)
    assert pd.isna(saved.loc["rest", "angular_error_deg"])
    for name, digest in hashes.items():
        assert d.sha256(source / name) == digest
