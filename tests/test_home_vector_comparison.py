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


def test_cross_task_challenge_comparison_requires_explicit_opt_in_and_exact_protocols(tmp_path):
    import copy
    from types import SimpleNamespace
    import pytest
    from fly_connectome.home_vector.analysis.compare import validate_challenge_comparison
    inputs = []
    for name in ('reference', 'current'):
        folder = tmp_path / name
        folder.mkdir()
        (folder / 'circuit.json').write_text('same circuit')
        (folder / 'task.json').write_text(name)
        (folder / 'trajectories.json').write_text(name)
        inputs.append(SimpleNamespace(run=folder, source=folder,
            task=dict(dt_s=.0025, length_scale_mm=2.),
            groups=dict(challenge=[dict(family='example', protocol=[[0, 0, 1, 0], [2, 0, 1, 0]])])))
    reference, current = inputs
    with pytest.raises(ValueError, match='identical task'):
        validate_challenge_comparison(reference, current, False)
    validate_challenge_comparison(reference, current, True)
    old = copy.deepcopy(current.groups)
    current.groups['challenge'][0]['protocol'][1][0] = 3
    with pytest.raises(ValueError, match='protocols'):
        validate_challenge_comparison(reference, current, True)
    current.groups = old
    for key in ('dt_s', 'length_scale_mm'):
        value = current.task[key]
        current.task[key] = value * 2
        with pytest.raises(ValueError, match='circuits, time steps and loss scales'):
            validate_challenge_comparison(reference, current, True)
        current.task[key] = value
    (current.run / 'circuit.json').write_text('different circuit')
    with pytest.raises(ValueError, match='circuits'):
        validate_challenge_comparison(reference, current, True)


def test_validation_plot_warns_about_actual_membership_not_permission_flag(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import matplotlib.pyplot as plt
    from fly_connectome.home_vector.analysis.compare import plot_comparison
    from fly_connectome.home_vector.data import datasets, sample_protocol
    groups = datasets(4, dict(train=2, validation=2, test=2))
    audit = SimpleNamespace(groups=groups, task=dict(dt_s=.25), out=tmp_path)
    truth = [sample_protocol(row['protocol'], .25)[1] for row in groups['challenge']]
    calls = []
    original_subplots = plt.subplots

    def capture(*args, **kwargs):
        result = original_subplots(*args, **kwargs)
        calls.append(result[0])
        return result

    monkeypatch.setattr(plt, 'subplots', capture)
    history = [dict(update=1, scaled_mse=1.)]
    report = dict(model_labels=dict(reference='Earlier fit', current='Later fit'),
                  reference=dict(validation_history=history), current=dict(validation_history=history),
                  task_differences_allowed=True)
    for matching in (True, False):
        report['validation_sets_match'] = matching
        plot_comparison((audit, audit), ((truth, truth), (truth, truth)), report)
        title = calls[-1].axes[0].get_title()
        assert ('Different validation sets' in title) == (not matching)


def test_joint_long_path_view_distinguishes_equal_magnitudes_with_wrong_direction():
    import matplotlib.pyplot as plt
    from fly_connectome.home_vector.analysis.compare import draw_long_paths
    target = np.column_stack((np.arange(5.), np.zeros(5)))
    rotated = np.column_stack((np.zeros(5), np.arange(5.)))
    fig, axes = plt.subplots(1, 3)
    draw_long_paths(axes, [('Correct', target, target, 1.), ('Wrong direction', rotated, target, 1.)])
    magnitude = {line.get_gid(): line for line in axes[1].lines}
    np.testing.assert_array_equal(magnitude['magnitude-0'].get_ydata(), magnitude['magnitude-1'].get_ydata())
    error = {line.get_gid(): line for line in axes[2].lines}
    np.testing.assert_array_equal(error['vector-error-0'].get_ydata(), np.zeros(5))
    np.testing.assert_allclose(error['vector-error-1'].get_ydata(), np.arange(5.) * np.sqrt(2))
    tips = {line.get_gid(): line for line in axes[0].lines if line.get_gid() != 'matched-time-error'}
    np.testing.assert_array_equal(tips['vector-tips-1'].get_xydata(), rotated)
    connectors = [line for line in axes[0].lines if line.get_gid() == 'matched-time-error']
    np.testing.assert_array_equal(connectors[-1].get_xydata(), [target[-1], rotated[-1]])
    assert np.linalg.norm(np.diff(connectors[-1].get_xydata(), axis=0)) == np.sqrt(32)
    assert axes[0].get_aspect() == 1.
    assert axes[0].yaxis_inverted()
    assert {text.get_text() for text in axes[0].texts} >= {'0 s', '1 s', '2 s', '3 s', '4 s'}
    plt.close(fig)


def test_joint_long_path_view_handles_zero_vectors_without_inventing_direction():
    import matplotlib.pyplot as plt
    from fly_connectome.home_vector.analysis.compare import draw_long_paths
    target = np.zeros((5, 2))
    fig, axes = plt.subplots(1, 3)
    draw_long_paths(axes, [('At home', target, target, .5)])
    for axis in axes:
        for line in axis.lines:
            assert np.isfinite(line.get_xydata()).all()
    plt.close(fig)
