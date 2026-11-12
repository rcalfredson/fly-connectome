"""Independent arithmetic examples for fixed-decoder attribution."""

import numpy as np
import pytest

from fly_connectome.home_vector.analysis.decoder_contributions import (
    agreement, contributions, duration_rms, ranking,
)


def test_signed_addends_cancel_and_intercept_is_separate():
    decoder = [[2., -1., 0.], [0., 0., 3.]]
    values = contributions([[2., 4., 1.], [3., 6., 2.]], decoder,
                           [10., -5.], [[10., -2.], [10., 1.]])
    np.testing.assert_array_equal(values['absolute'][0], [[4., 0.], [-4., 0.], [0., 3.]])
    np.testing.assert_array_equal(values['from_initial'][1], [[2., 0.], [-2., 0.], [0., 3.]])
    # Neurons 1/2 have large opposing terms despite zero combined x output.
    assert np.linalg.norm(values['absolute'][0], axis=-1).sum() == 11.
    np.testing.assert_array_equal(values['absolute'][0].sum(axis=0), [0., 3.])


def test_persistent_offsets_change_absolute_ranking_but_not_initial_referenced_change():
    decoder = [[1., 0.], [0., 1.]]
    a = contributions([[100., 0.], [100., 4.], [100., 4.]], decoder,
                      [0., 0.], [[100., 0.], [100., 4.], [100., 4.]])
    assert ranking(duration_rms(a['absolute'], [0., 1., 2.]), [1, 2]).tolist() == [0, 1]
    assert ranking(duration_rms(a['from_initial'], [0., 1., 2.]), [1, 2]).tolist() == [1, 0]


def test_initial_pair_difference_is_not_removed_by_centering_each_path():
    decoder = [[1., -1.], [0., 1.]]
    a = contributions([[1., 2.], [2., 3.]], decoder, [7., 8.], [[6., 10.], [6., 11.]])
    b = contributions([[3., 2.], [4., 3.]], decoder, [7., 8.], [[8., 10.], [8., 11.]])
    paired = b['absolute'] - a['absolute']
    np.testing.assert_array_equal(paired.sum(axis=1), [[2., 0.], [2., 0.]])
    np.testing.assert_array_equal(b['from_initial'] - a['from_initial'], 0.)


def test_rms_uses_durations_and_excludes_arbitrary_terminal_state():
    vectors = np.array([[[3., 4.]], [[0., 2.]], [[1e9, 1e9]]])
    # First interval lasts one second, second lasts three seconds.
    np.testing.assert_allclose(duration_rms(vectors, [0., 1., 4.]), [np.sqrt(37./4.)])
    for time in ([0., 0., 4.], [0., np.nan, 4.], [0., 4., 1.]):
        with pytest.raises(ValueError):
            duration_rms(vectors, time)


def test_rank_comparison_uses_average_ties_and_flags_arbitrary_cutoffs():
    assert ranking([1., 1., 2.], [30, 10, 20]).tolist() == [2, 1, 0]
    same = agreement([3., 2., 1.], [30., 20., 10.], [1, 2, 3])
    assert same['spearman_all_neurons'] == 1.
    opposite = agreement([3., 2., 1.], [1., 2., 3.], [1, 2, 3])
    assert opposite['spearman_all_neurons'] == -1.
    tied = agreement(np.zeros(12), np.zeros(12), list(range(12)))
    assert tied['spearman_all_neurons'] is None
    assert tied['cutoff_tied_a'] and tied['cutoff_tied_b']


def test_misaligned_nonfinite_and_incorrect_prediction_are_rejected():
    with pytest.raises(ValueError):
        contributions([[1., np.nan]], np.eye(2), [0., 0.], [[1., 2.]])
    with pytest.raises(ValueError):
        contributions([[1., 2.]], [[1., 2.]], [0., 0.], [[1., 2.]])
    with pytest.raises(AssertionError):
        contributions([[1., 2.]], np.eye(2), [0., 0.], [[1., 3.]])
