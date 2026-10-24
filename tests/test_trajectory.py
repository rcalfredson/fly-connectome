from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from fly_connectome import trajectory as t
from fly_connectome import movement as m
from fly_connectome.connectivity import ValidationError
from test_movement import mapping_fixture


def test_world_axes_and_analytic_paths():
    np.testing.assert_allclose(t.world_velocity(np.array([0, 90, 180, 270]), np.ones(4), np.zeros(4)),
                               [[1, 0], [0, 1], [-1, 0], [0, -1]], atol=1e-14)
    np.testing.assert_allclose(t.world_velocity(90, 2, 3), [-3, 2], atol=1e-14)
    protocol = t.default_protocol()
    sampled = t.sample_inputs(protocol, .001)
    geometry = t.geometric_reference(sampled)
    # Analytic endpoints of every segment, independently specified.
    expected = [[0, 0], [0, 0], [1, 0], [1, 1], [0, 1], [0, 0],
                [0, 0], [1, 0], [1, 1], [0, 1], [0, 0], [0, 0]]
    ids = [m.grid_index(v, .001) for v in protocol.time_s]
    np.testing.assert_allclose(geometry.iloc[ids][["displacement_x_mm", "displacement_y_mm"]], expected, atol=1e-12)
    np.testing.assert_array_equal(geometry.return_to_origin_x_mm, -geometry.displacement_x_mm)
    np.testing.assert_array_equal(geometry.return_to_origin_y_mm, -geometry.displacement_y_mm)
    # Terminal sample never adds displacement, even with nonzero terminal velocity.
    protocol.loc[len(protocol) - 1, "forward_mm_s"] = 999
    np.testing.assert_allclose(t.geometric_reference(t.sample_inputs(protocol, .001)).displacement_x_mm,
                               geometry.displacement_x_mm)


def test_hold_boundaries_and_angle_wrap():
    protocol = pd.DataFrame(dict(time_s=[0, .1, .2], heading_deg=[359, 1, 20],
                                 forward_mm_s=[1, -1, 8], right_mm_s=[0, 2, 0]))
    samples = t.sample_inputs(protocol, .001)
    assert samples.heading_deg[99] == 359
    assert samples.heading_deg[100] == 1
    assert samples.forward_mm_s[100] == -1
    assert len(samples) == 201
    for change in [dict(time_s=[0, .1005, .2]), dict(time_s=[0, .1, .1]),
                   dict(time_s=[.01, .1, .2]), dict(heading_deg=[0, np.nan, 0])]:
        with pytest.raises(ValidationError):
            t.sample_inputs(protocol.assign(**change), .001)


def test_existing_solver_and_continuation_with_nonzero_initial_state():
    p = m.MovementParameters()
    times, old, old_exact = m.solve_output(np.array([1., 2.]), np.array([3., 4.]), p)
    drive = np.tile([1., 2.], (len(times), 1))
    drive[200:1200] += [3., 4.]
    activity, exact = t.solve_continuous(drive, p.dt_s, p.tau_s)
    np.testing.assert_array_equal(activity, old)
    np.testing.assert_allclose(exact, old_exact, atol=1e-14)
    left, left_exact = t.solve_continuous(drive[:1201], p.dt_s, p.tau_s, [7., 8.])
    right, _ = t.solve_continuous(drive[1200:], p.dt_s, p.tau_s, left[-1])
    full, _ = t.solve_continuous(drive, p.dt_s, p.tau_s, [7., 8.])
    np.testing.assert_array_equal(np.vstack([left[:-1], right]), full)
    assert np.all(right[0] > 0)
    np.testing.assert_allclose(full[1201], full[1200] + .01 * (drive[1200] - full[1200]))


def test_stationary_equilibrium_and_timestep_convergence():
    mapping, _ = mapping_fixture()
    p = m.MovementParameters(normalize_incoming_by_group=True, pfnd_weight_multiplier=6.)
    counts = pd.DataFrame(np.ones((len(mapping), 3)), index=mapping.bodyId)
    weights = m.movement_weights(counts, mapping, p)
    protocol = t.default_protocol().copy()
    protocol[["forward_mm_s", "right_mm_s"]] = 0.
    protocol["heading_deg"] = 45.
    rest = m.encode_pfn(mapping, 45, 0, 0, 1, p) @ weights
    inputs, _, _, activity, exact = t.replay(protocol, mapping, weights, p, initial=rest)
    np.testing.assert_allclose(activity, np.broadcast_to(rest, activity.shape), atol=1e-15)
    geometry = t.geometric_reference(inputs)
    assert not geometry[["displacement_x_mm", "displacement_y_mm"]].to_numpy().any()
    # Changing heading at rest changes prescribed baseline; the state still carries through.
    protocol = t.default_protocol()
    _, pfn, drive, activity, exact = t.replay(protocol, mapping, weights, p)
    _, _, _, fine, fine_exact = t.replay(protocol, mapping, weights, replace(p, dt_s=.0005))
    assert m.relative_error(fine[::2], exact) < m.relative_error(activity, exact) < .01
    np.testing.assert_allclose(fine_exact[::2], exact, atol=1e-14)
    assert pfn[-1].max() > 0  # b=1 remains active at the final stop.
    expected = drive[-1] + (exact[9000] - drive[-1]) * np.exp(-1 / p.tau_s)
    np.testing.assert_allclose(exact[-1], expected, atol=1e-14)


@pytest.mark.parametrize("id_column", ["bodyId", "bodyId_pre"])
def test_saved_run_exports_and_rejects_changes(tmp_path, id_column):
    mapping, orders = mapping_fixture()
    # Minimal independently hashed selected source; no network calls needed.
    source = tmp_path / "source"
    source.mkdir()
    p = m.MovementParameters(normalize_incoming_by_group=True, pfnd_weight_multiplier=6.)
    config = dict(parameters=m.asdict(p), baseline_levels=[1., 0.], source_ids=mapping.bodyId.tolist(),
                  target_ids=[101, 102], encoding="test", heading_mapping="test")
    m.save_json(source / "movement_config.json", config)
    mapping.rename(columns={"bodyId": id_column}).to_csv(source / "PFN_encoding_map.csv", index=False)
    counts = pd.DataFrame(np.ones((len(mapping), 2)), index=mapping.bodyId, columns=[101, 102])
    counts.to_csv(source / "raw_contact_matrix.csv")
    pd.DataFrame(m.movement_weights(counts, mapping, p), index=counts.index, columns=counts.columns).to_csv(source / "effective_weight_matrix.csv")
    for pop in ["PFNd", "PFNv"]:
        orders[pop].to_csv(source / f"{pop}_anatomical_order.csv", index=False)
    pd.DataFrame(dict(bodyId=[101, 102], label=["a", "b"])).to_csv(source / "hDeltaB_anatomical_order.csv", index=False)
    manifest = dict(status="complete", data_validated=True,
                    data_sha256={f.name: m.sha256(f) for f in source.iterdir()})
    m.save_json(source / "movement_model.json", manifest)
    output = tmp_path / "output"
    result = t.run_trajectory(source, output, {})
    assert result["status"] == "complete" and result["validation"]["passed"]
    m.verify_model_data(output, result)
    assert (output / "raw_activity.png").stat().st_size > 1000
    assert (output / "trajectory_inputs.pdf").stat().st_size > 1000
    with np.load(output / "activity.npz") as data:
        assert data["hdeltab_activity"].shape == (10001, 2)
    assert json.loads((output / "trajectory_config.json").read_text())["baseline"] == 1.
    with pytest.raises(FileExistsError):
        t.run_trajectory(source, output, {})
    (source / "PFN_encoding_map.csv").write_text("tampered")
    with pytest.raises(ValidationError, match="changed"):
        t.run_trajectory(source, tmp_path / "rejected", {})
    assert not (tmp_path / "rejected").exists()
