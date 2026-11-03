import numpy as np

from fly_connectome.home_vector.analysis.compare import sustained_underestimate, trajectory_summary


def test_distance_threshold_requires_a_sustained_breach_and_excludes_overestimation():
    distance = np.arange(0, 11, dtype=float)
    prediction = distance.copy()
    prediction[3] = 0  # One-sample dip must not count as a two-second breach.
    prediction[6:] *= .7
    assert sustained_underestimate(distance, prediction, 1., .2, hold=2.) == 6
    assert sustained_underestimate(distance, distance * 1.5, 1., .2) is None
    assert sustained_underestimate(distance[:3], prediction[:3], 1., .2, hold=4.) is None


def test_pause_drift_and_accuracy_are_distinct():
    row = dict(family="pause", kind="matched_pause", suffix_start_s=1.,
               protocol=[[0, 0, -1, 0], [1, 90, 0, 0], [3, 90, 0, 0]])
    target = np.array([[0., 0.], [1., 0.], [1., 0.], [1., 0.]])
    prediction = np.array([[0., 0.], [2., 0.], [1.5, 0.], [1., 0.]])
    result = trajectory_summary(row, prediction, target, 1.)
    assert result["pause"]["movement"] == "backward"
    assert result["pause"]["onset_error_mm"] == 1
    assert result["pause"]["endpoint_drift_mm"] == 1
    assert result["terminal_error_mm"] == 0
    assert result["pause"]["decoded_path_length_during_pause_mm"] == 1


def test_square_component_errors_include_transverse_error():
    row = dict(family="challenge-square", kind="closed_loop")
    target = np.array([[0., 0.], [-2., 0.], [-2., -2.], [0., -2.], [0., 0.]])
    prediction = target + [0., 1.]
    result = trajectory_summary(row, prediction, target, 1.)
    assert result["component_rmse_mm"] == [0., 1.]
    assert result["terminal_error_mm"] == 1


def test_reversal_lag_measures_turnaround_separately_from_late_closure():
    row = dict(family="challenge-reversal", kind="reversal",
               protocol=[[0, 0, 1, 0], [2, 0, -1, 0], [4, 0, 0, 0], [5, 0, 0, 0]])
    target = np.column_stack(([0., -1., -2., -1., 0., 0.], np.zeros(6)))
    prediction = np.column_stack(([0., -1., -2., -2.5, -.5, 0.], np.zeros(6)))
    result = trajectory_summary(row, prediction, target, 1.)
    assert result["reversal"]["turnaround_lag_s"] == 1
    assert result["reversal"]["additional_negative_x_excursion_mm"] == .5
    assert result["reversal"]["error_at_true_return_mm"] == .5
    assert result["reversal"]["final_pause_error_mm"] == 0
