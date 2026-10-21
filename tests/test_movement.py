from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from fly_connectome import __main__ as cli
from fly_connectome import movement as m
from fly_connectome.anatomy import annotation_order
from fly_connectome.connectivity import ValidationError


def mapping_fixture():
    orders, next_id = {}, 1
    for pop, prefix in [("PFNd", "PB04"), ("PFNv", "PB05")]:
        rows = []
        for side in ["L", "R"]:
            for glomerulus in range(2, 10):
                column = glomerulus - 1 if side == "L" else 10 - glomerulus
                rows.append({"bodyId": next_id, "type": pop, "instance": f"{pop}({prefix})_{side}{glomerulus}_C{column}"})
                next_id += 1
        orders[pop] = annotation_order(pd.DataFrame(rows), pop)
    mapping = m.heading_mapping(orders, list(range(1, next_id)))
    return mapping, orders


def test_heading_map_uses_pb_annotations_and_wraps_outer_glomeruli():
    mapping, orders = mapping_fixture()
    expected_left = {2: -135, 3: -90, 4: -45, 5: 0, 6: 45, 7: 90, 8: 135, 9: -180}
    expected_right = {2: 135, 3: 90, 4: 45, 5: 0, 6: -45, 7: -90, 8: -135, 9: -180}
    for row in mapping.itertuples():
        assert row.preferred_heading_deg == (expected_left if row.pb_side == "L" else expected_right)[row.pb_glomerulus]
    # Column labels are not the source of assigned functional heading.
    changed = {pop: order.assign(fb_column=99) for pop, order in orders.items()}
    np.testing.assert_array_equal(m.heading_mapping(changed, mapping.bodyId).preferred_heading_deg, mapping.preferred_heading_deg)


def test_encoding_baseline_heading_tuning_and_body_velocity_sign():
    mapping, _ = mapping_fixture()
    p = m.MovementParameters()
    rates = m.encode_pfn(mapping, 0., 0., 0., 0., p)
    assert not rates.any()
    background = m.encode_pfn(mapping, 0., 0., 0., 1., p)
    assert background.max() == pytest.approx(1.)
    assert (background >= 0).all()
    left = mapping[(mapping.type == "PFNd") & (mapping.pb_side == "L") & (mapping.pb_glomerulus == 5)].index[0]
    right = mapping[(mapping.type == "PFNd") & (mapping.pb_side == "R") & (mapping.pb_glomerulus == 5)].index[0]
    lateral = m.encode_pfn(mapping, 0., 0., 1., 0., p)
    assert lateral[right] > 0 and lateral[left] == 0
    opposite = m.encode_pfn(mapping, 180., 0., 1., 0., p)
    assert opposite[right] == pytest.approx(0., abs=1e-14)
    one = m.encode_pfn(mapping, 45., 1., .5, 1., p)
    two = m.encode_pfn(mapping, 45., 2., 1., 1., p)
    rest = m.encode_pfn(mapping, 45., 0., 0., 1., p)
    np.testing.assert_allclose(two - rest, 2 * (one - rest), atol=1e-14)


def test_condition_geometry_and_balanced_primary_grid():
    conditions = m.movement_conditions()
    assert len(conditions) == 74 and conditions.primary_grid.sum() == 64
    assert conditions.loc[conditions.speed_mm_s == 0, "world_direction_deg"].isna().all()
    moving = conditions[conditions.speed_mm_s > 0]
    body = np.rad2deg(np.arctan2(moving.right_mm_s, moving.forward_mm_s))
    np.testing.assert_allclose(m.wrap_degrees(body + moving.heading_deg - moving.world_direction_deg), 0., atol=1e-12)
    np.testing.assert_allclose(np.hypot(moving.forward_mm_s, moving.right_mm_s), moving.speed_mm_s)


def test_type_weight_scaling_preserves_identity_and_pfn_encoding():
    mapping, _ = mapping_fixture()
    counts = pd.DataFrame(np.arange(len(mapping) * 3).reshape(-1, 3), index=mapping.bodyId)
    p = m.MovementParameters()
    scaled = replace(p, pfnd_weight_multiplier=6.)
    original = m.movement_weights(counts, mapping, p)
    weighted = m.movement_weights(counts, mapping.sample(frac=1, random_state=8), scaled)
    is_d = mapping.type.eq("PFNd").to_numpy()
    np.testing.assert_array_equal(weighted[is_d], 6 * original[is_d])
    np.testing.assert_array_equal(weighted[~is_d], original[~is_d])
    np.testing.assert_array_equal(m.encode_pfn(mapping, 45., 1., .5, 1., p),
                                  m.encode_pfn(mapping, 45., 1., .5, 1., scaled))
    for changes in [{"pfnd_weight_multiplier": -1.}, {"pfnv_weight_multiplier": np.nan}]:
        with pytest.raises(ValidationError):
            replace(p, **changes).validate()


def test_single_stage_dynamics_baseline_and_half_open_stimulus():
    p = m.MovementParameters()
    times, output, exact = m.solve_output(np.array([0., 1.]), np.array([2., 0.]), p)
    assert output[200, 0] == 0 and output[201, 0] > 0
    assert exact[300, 0] == pytest.approx(2 * (1 - np.exp(-1)))
    assert output[-1, 1] == pytest.approx(1., abs=1e-8)
    assert output[-1, 0] < .001
    assert m.relative_error(output, exact) < .01
    _, refined, _ = m.solve_output(np.array([0., 1.]), np.array([2., 0.]), replace(p, dt_s=.0005))
    assert m.relative_error(refined[::2], exact) < m.relative_error(output, exact)


def test_incoming_normalization_by_target_type_and_side_matches_published_axes():
    mapping = pd.DataFrame({"bodyId": range(8), "type": ["PFNd"] * 4 + ["PFNv"] * 4,
                            "pb_side": ["L", "L", "R", "R"] * 2})
    counts = pd.DataFrame([[1, 0, 2], [3, 0, 6], [2, 1, 0], [2, 3, 0],
                           [1, 2, 1], [1, 6, 3], [3, 0, 9], [1, 0, 3]], dtype=float)
    p = m.MovementParameters(normalize_incoming_by_group=True, pfnd_weight_multiplier=6.)
    actual = m.movement_weights(counts, mapping.sample(frac=1, random_state=2), p)
    for positions, scale in [([0, 1], 6), ([2, 3], 6), ([4, 5], 1), ([6, 7], 1)]:
        # Authors store targets in rows and PFNs in columns (transpose of ours).
        published = counts.iloc[positions].to_numpy().T
        totals = published.sum(axis=1)[:, None]
        expected = np.divide(published, totals, out=np.zeros_like(published), where=totals > 0)
        np.testing.assert_allclose(actual[positions], expected.T * scale * p.gain)
        np.testing.assert_allclose(actual[positions].sum(axis=0), (totals[:, 0] > 0) * scale * p.gain)
    assert np.isfinite(actual).all()
    # Group/target total strength is removed, while each group's contact ratios remain.
    changed = counts.copy()
    changed.iloc[:2, 0] *= 17
    np.testing.assert_allclose(actual, m.movement_weights(changed, mapping, p))
    np.testing.assert_array_equal(counts.iloc[:2, 0], [1, 3])


def test_pattern_metric_has_independent_ideal_oracle_and_rejects_constant_output():
    grid = m.movement_conditions().query("primary_grid").reset_index(drop=True)
    world = np.deg2rad(grid.world_direction_deg.to_numpy())
    target_angles = np.deg2rad(np.arange(0, 360, 45))
    profiles = 2 + np.cos(world[:, None] - target_angles)
    summary, pairs, similarity = m.pattern_comparison(profiles, grid)
    assert summary["same_world_mean_centered_cosine"] == pytest.approx(1.)
    assert summary["different_world_mean_centered_cosine"] == pytest.approx(-1 / 7)
    assert summary["contrast"] == pytest.approx(8 / 7)
    assert len(pairs) == 448 and np.diag(similarity).min() == pytest.approx(1.)
    invalid, _, _ = m.pattern_comparison(np.ones_like(profiles), grid)
    assert invalid["contrast"] is None and invalid["valid_directional_patterns"] == 0
    scores = [m.pattern_comparison(profiles, m.shuffled_labels(grid, np.random.default_rng(seed)), include_pairs=False)[0]["contrast"] for seed in range(10)]
    assert max(scores) < summary["contrast"]


def test_shuffle_preserves_group_totals_and_is_fixed_across_conditions():
    mapping, _ = mapping_fixture()
    rng = np.random.default_rng(123)
    permutation = m.shuffled_sources(mapping, rng)
    np.testing.assert_array_equal(permutation, m.shuffled_sources(mapping, np.random.default_rng(123)))
    assert not np.array_equal(permutation, np.arange(len(mapping)))
    weights = np.arange(len(mapping) * 3).reshape(-1, 3)
    for _, group in mapping.groupby(["type", "pb_side"]):
        ids = group.index.to_numpy()
        assert set(permutation[ids]) == set(ids)
        np.testing.assert_array_equal(weights[permutation[ids]].sum(axis=0), weights[ids].sum(axis=0))


@pytest.mark.parametrize("changes", [{"tau_s": 0}, {"dt_s": .1}, {"seed": -1}, {"seed": 1.5},
                                     {"null_repeats": 0}, {"gain": np.nan}, {"analysis_stop_s": 1.5}])
def test_invalid_parameters_fail(changes):
    with pytest.raises(ValidationError):
        replace(m.MovementParameters(), **changes).validate()


@pytest.mark.parametrize("normalize", [False, True])
def test_offline_run_preserves_inputs_validates_traces_and_replots_after_move(tmp_path, monkeypatch, normalize):
    mapping, orders = mapping_fixture()
    target = pd.DataFrame({"bodyId": [1001, 1002, 1003], "type": "hDeltaB", "instance": ["hDeltaB_01_C1", "hDeltaB_02_C2", "hDeltaB_03_C3"]})
    orders["hDeltaB"] = annotation_order(target, "hDeltaB")
    populations = {pop: table[["bodyId", "type", "instance"]] for pop, table in orders.items()}
    counts = pd.DataFrame(np.random.default_rng(5).integers(0, 20, (len(mapping), 3)), index=mapping.bodyId, columns=target.bodyId)
    source = tmp_path / "source"
    source.mkdir()
    (source / "data.csv").write_text("untouched\n")
    digest = m.sha256(source / "data.csv")
    monkeypatch.setattr(m, "load_pfn_network", lambda path: (counts, populations, orders, {"dataset_metadata": {"uuid": "test"}}, {"data.csv": digest}))
    monkeypatch.setattr(cli, "Client", lambda *a, **kw: pytest.fail("Movement simulation opened a client"))
    rendered = []
    monkeypatch.setattr(m, "render_movement", lambda source, output: rendered.append(str(source)))
    p = m.MovementParameters(tau_s=.03, dt_s=.001, duration_s=.4, onset_s=.05, offset_s=.25,
                             analysis_start_s=.2, analysis_stop_s=.25, null_repeats=3,
                             normalize_incoming_by_group=normalize, pfnd_weight_multiplier=6. if normalize else 1.)
    output = tmp_path / "run"
    manifest = m.run_movement(source, output, {}, p)
    assert manifest["status"] == "complete" and manifest["validation"]["trials"] == 148
    assert m.sha256(source / "data.csv") == digest
    assert pd.read_csv(output / "numerical_validation.csv").passed.all()
    assert (output / "experiment_notes.md").stat().st_size > 1000
    raw = pd.read_csv(output / "baseline_1/steady_activity.csv", index_col=0)
    evoked = pd.read_csv(output / "baseline_1/movement_evoked_activity.csv", index_col=0)
    zero_baseline = pd.read_csv(output / "baseline_0/steady_activity.csv", index_col=0)
    np.testing.assert_allclose(evoked, zero_baseline, atol=1e-12)
    assert raw.loc["rest_heading000"].max() > 0
    assert not evoked.loc["rest_heading000"].any()
    trace_path = output / "baseline_1/trials/world000_heading090_speed1.npz"
    with np.load(trace_path) as trace:
        assert trace["pfn_activity"].shape == (401, len(mapping))
        assert trace["hdeltab_activity"].shape == (401, 3)
        assert trace["forward_mm_s"][50] == pytest.approx(0., abs=1e-14)
        assert trace["right_mm_s"][50] == pytest.approx(-1.)
        assert trace["right_mm_s"][250] == 0
    assert len(pd.read_csv(output / "null_comparisons.csv")) == 12
    m.verify_model_data(output, manifest)
    moved = tmp_path / "moved"
    output.rename(moved)
    source.rename(tmp_path / "moved-source")
    m.replot_movement(moved, tmp_path / "replot", {})
    assert json.loads((tmp_path / "replot/replot.json").read_text())["status"] == "complete"
    with (moved / "conditions.csv").open("a") as stream:
        stream.write("tampered\n")
    with pytest.raises(ValidationError, match="changed"):
        m.replot_movement(moved, tmp_path / "rejected", {})
    assert not (tmp_path / "rejected").exists()
