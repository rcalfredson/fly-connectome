"""Separate states can share a decoder-visible projection without sharing rates."""

import numpy as np
import pytest

from fly_connectome.home_vector.analysis.state_patterns import project_states, select_examples


def test_equal_predictions_share_row_pattern_despite_different_actual_neurons():
    decoder = np.array([[1., 1., 0.], [0., 0., 1.]])
    a = np.array([[8., 2., 3.], [6., 4., 5.]])
    b = np.array([[2., 8., 3.], [4., 6., 5.]])
    ra, na = project_states(a, decoder)
    rb, nb = project_states(b, decoder)
    np.testing.assert_allclose(ra, [[5., 5., 3.], [5., 5., 5.]])
    np.testing.assert_allclose(ra, rb)
    np.testing.assert_allclose(na, [[3., -3., 0.], [1., -1., 0.]], atol=1e-14)
    np.testing.assert_allclose(nb, -na, atol=1e-14)
    np.testing.assert_allclose(na @ decoder.T, 0., atol=1e-14)
    assert not np.array_equal(a, b)


def test_correlated_decoder_rows_project_onto_whole_plane_not_each_row_separately():
    decoder = np.array([[1., 1., 0.], [1., 1., 1.]])
    rates = np.array([[8., 2., 3.], [1., 3., 4.]])
    row, null = project_states(rates, decoder)
    expected = rates @ decoder.T @ np.linalg.inv(decoder @ decoder.T) @ decoder
    np.testing.assert_allclose(row, expected, atol=1e-14)
    np.testing.assert_allclose(row + null, rates)


def test_row_selection_rewards_magnitude_in_both_not_agreement():
    # Entry 0 reverses sign between trajectories yet outranks the agreeing entry 1.
    ra = np.array([[10., 5., 100.], [10., 5., 100.], [9999., 9999., 9999.]])
    rb = np.array([[-10., 5., 0.], [-10., 5., 0.], [9999., 9999., 9999.]])
    na = np.zeros_like(ra)
    nb = np.array([[3., 0., 0.], [3., 0., 0.], [9999., 9999., 9999.]])
    picked = select_examples(ra, rb, na, nb, [0., 1., 3.], [1, 2, 3], 1)
    np.testing.assert_allclose(picked['row_shared_score'], [10., 5., 0.])
    assert picked['row_selected'].tolist() == [0]
    assert picked['null_selected'].tolist() == [0]  # Overlap is allowed.


def test_separate_projection_differences_equal_projection_of_difference():
    decoder = np.array([[1., 2., 0., 0.], [2., 1., 3., 0.]])
    a = np.array([[4., 2., 3., 8.], [5., 9., 7., 2.]])
    b = np.array([[2., 4., 5., 1.], [6., 7., 9., 5.]])
    ra, na = project_states(a, decoder)
    rb, nb = project_states(b, decoder)
    rd, nd = project_states(b - a, decoder)
    np.testing.assert_allclose(rb - ra, rd, atol=1e-14)
    np.testing.assert_allclose(nb - na, nd, atol=1e-14)


def test_degenerate_decoder_and_invalid_inputs():
    row, null = project_states([[1., 2.]], np.zeros((2, 2)))
    np.testing.assert_array_equal(row, [[0., 0.]])
    np.testing.assert_array_equal(null, [[1., 2.]])
    for rates, decoder in [([[np.nan, 1.]], np.eye(2)), ([[1., 2.]], [[1., 2.]])]:
        with pytest.raises(ValueError):
            project_states(rates, decoder)
