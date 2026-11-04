import numpy as np
import pytest

from fly_connectome.home_vector.analysis.straight import (
    conditions, constant_rollout, controls_and_velocity, summary,
)


def test_world_matched_paths_share_targets_despite_different_body_controls():
    rows = [r for r in conditions([1], [0, 90], [0, 90, 127, 180, 270])
            if r["family"] == "same_world"]
    controls, velocity = controls_and_velocity(rows)
    np.testing.assert_allclose(velocity, np.tile([2 ** -.5, 2 ** -.5], (5, 1)), atol=1e-14)
    assert len(np.unique(controls, axis=0)) == 5


def test_vector_threshold_detects_wrong_direction_when_magnitude_is_correct():
    dt = .01
    time = np.arange(1001) * dt
    row = dict(speed_mm_s=1.)
    correct = np.column_stack((-time, np.zeros_like(time)))
    good = summary(row, correct, np.array([1., 0.]), dt)
    assert good["vector_error_10pct_distance_mm"] is None
    assert good["underestimate_10pct_distance_mm"] is None
    assert good["late_along_track_gain"] == pytest.approx(1)
    wrong = summary(row, -correct, np.array([1., 0.]), dt)
    assert wrong["underestimate_10pct_distance_mm"] is None
    assert wrong["vector_error_10pct_distance_mm"] == 2
    assert wrong["final_direction_error_deg"] == pytest.approx(180)


def test_threshold_time_and_distance_separate_speed_and_censoring():
    dt = .01
    time = np.arange(1001) * dt
    row = dict(speed_mm_s=2.)
    prediction = np.column_stack((-np.minimum(2 * time, 6.), np.zeros_like(time)))
    result = summary(row, prediction, np.array([2., 0.]), dt)
    assert result["underestimate_10pct_distance_mm"] == pytest.approx(6.68)
    assert result["underestimate_10pct_time_s"] == pytest.approx(3.34)


def test_batched_constant_replay_matches_torch_and_independent_numpy_reference():
    torch = pytest.importorskip("torch")
    from fly_connectome.home_vector.data import Circuit
    from fly_connectome.home_vector.dynamics import numpy_rollout
    from fly_connectome.home_vector.model import HomeVectorNetwork
    circuit = Circuit([1], ["PFNd"], [2, 3], ["hDeltaB", "vDeltaE"], [0.], [0.], [
        dict(pre=0, post=0, contacts=1, denominator=1., source_type="PFNd", target_type="hDeltaB"),
        dict(pre=1, post=1, contacts=1, denominator=1., source_type="hDeltaB", target_type="vDeltaE"),
        dict(pre=2, post=0, contacts=1, denominator=1., source_type="vDeltaE", target_type="hDeltaB")])
    model = HomeVectorNetwork(circuit)
    with torch.no_grad():
        model.a.copy_(torch.tensor([.6, -.2, .4]))
        model.D.copy_(torch.tensor([[1., -.5], [.2, .7]]))
        model.d.copy_(torch.tensor([.1, -.1]))
    pfn, dt, duration = np.array([[.5], [2.]], dtype=np.float32), .0025, .25
    prediction, norms, _ = constant_rollout(model, pfn, dt, duration)
    traces = np.repeat(pfn[:, None], 101, axis=1)
    with torch.no_grad():
        expected, rates = model(torch.tensor(traces), dt)
    np.testing.assert_allclose(prediction, expected.numpy(), atol=2e-7)
    np.testing.assert_allclose(norms, torch.linalg.vector_norm(rates, dim=-1).numpy(), atol=2e-7)
    for i in range(2):
        reference = numpy_rollout(circuit, traces[i], dt, model.a.detach().numpy(),
                                  model.bias.detach().numpy(), model.tau.detach().numpy())
        np.testing.assert_allclose(prediction[i], reference @ model.D.detach().numpy().T
                                   + model.d.detach().numpy(), atol=2e-7)
