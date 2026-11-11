"""Known decoder geometry and paired-state interpretation checks."""

import numpy as np
import pytest

pytest.importorskip('torch')

from fly_connectome.home_vector.analysis.state_spaces import compare_pair, split_difference


def test_sum_decoder_splits_known_example_without_removing_initial_difference():
    parts = split_difference([[4., -2.], [3., -3.]], [[1., 1.]])
    np.testing.assert_allclose(parts['row_component'], [[1., 1.], [0., 0.]], atol=1e-14)
    np.testing.assert_allclose(parts['null_component'], [[3., -3.], [3., -3.]], atol=1e-14)
    np.testing.assert_allclose(parts['decoded_difference'], [[2.], [0.]])
    assert parts['rank'] == 1


def test_nonorthogonal_decoder_preserves_output_and_euclidean_geometry():
    decoder = np.array([[1., 2., 0., 0.], [2., 1., 3., 0.]])
    delta = np.array([[4., -2., 1., 5.], [-1., 2., 3., -8.]])
    parts = split_difference(delta, decoder)
    # Independent full-rank projection formula, including correlated decoder rows.
    expected = delta @ decoder.T @ np.linalg.inv(decoder @ decoder.T) @ decoder
    np.testing.assert_allclose(parts['row_component'], expected, atol=1e-14)
    np.testing.assert_allclose(parts['null_component'] @ decoder.T, 0, atol=1e-14)
    np.testing.assert_allclose(np.sum(delta ** 2, axis=1),
        np.sum(parts['row_component'] ** 2 + parts['null_component'] ** 2, axis=1))
    assert parts['rank'] == 2


def test_zero_and_rank_deficient_decoders_do_not_invent_visible_dimensions():
    for decoder, rank in [(np.zeros((2, 3)), 0), (np.array([[1., 1., 0.], [2., 2., 0.]]), 1)]:
        parts = split_difference([[1., 2., 4.]], decoder)
        assert parts['rank'] == rank
        np.testing.assert_allclose(parts['row_component'] @ decoder.T, [[1., 2., 4.]] @ decoder.T)


def pair():
    a = dict(time_s=np.array([0., .0025]), rates=np.array([[1., 1., 1.], [2., 2., 2.]]),
             controls=np.array([[0., 1., 0.], [0., 0., 0.]]),
             pfn=np.ones((2, 4)), target=np.zeros((2, 2)))
    b = {k: v.copy() for k, v in a.items()}
    b['rates'] += [2., -2., 0.]
    b['controls'][:, 0] = 270.
    b['controls'][0, 1:] = [0., 1.]
    b['pfn'] *= 2
    decoder = np.array([[1., 1., 0.], [0., 0., 1.]])
    # A nonzero intercept must cancel from the paired decoded difference.
    for arrays in (a, b):
        arrays['prediction'] = arrays['rates'] @ decoder.T + [10., -5.]
    return a, b, decoder


def test_equal_decoded_vectors_can_be_wrong_and_pfn_inputs_can_differ_at_equal_speed():
    a, b, decoder = pair()
    _, metrics = compare_pair(a, b, decoder)
    np.testing.assert_allclose(metrics['decoded_difference_norm_mm'], 0.)
    assert np.all(metrics['error_a_mm'] > 0.)
    np.testing.assert_allclose(metrics['error_a_mm'], metrics['error_b_mm'])
    assert np.all(metrics['null_component_norm'] > 0.)
    np.testing.assert_allclose(metrics['speed_a_mm_s'], metrics['speed_b_mm_s'])
    assert not metrics['identical_pfn'].any()
    assert not metrics['pfn_equal_within_tolerance'].any()


def test_identical_states_have_undefined_fraction_and_unmatched_targets_are_rejected():
    a, b, decoder = pair()
    _, metrics = compare_pair(a, a, decoder)
    assert np.isnan(metrics['null_squared_norm_fraction']).all()
    b['target'][1, 0] = .01
    with pytest.raises(AssertionError):
        compare_pair(a, b, decoder)


def test_nonfinite_or_misaligned_projection_inputs_are_rejected():
    for delta, decoder in [([[np.nan, 1.]], [[1., 1.]]), ([[1., 2.]], [[1., 2., 3.]])]:
        with pytest.raises(ValueError):
            split_difference(delta, decoder)
