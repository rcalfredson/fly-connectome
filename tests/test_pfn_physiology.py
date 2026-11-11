"""Scientific checks for digitized response fits and calibrated coordinates."""

import numpy as np
import pytest

from fly_connectome.home_vector.analysis.pfn_physiology import (
    affine, digitize, fit_curve, predict,
)


def test_truncated_and_downward_image_axes_keep_physical_offsets():
    # A panel whose lower tick is 10 spikes/s must not be treated as zero.
    np.testing.assert_allclose(affine([820.5, 726.5, 632.5],
        [820.5, 632.5], [10, 30]), [10, 20, 30])
    np.testing.assert_allclose(affine([100, 150, 200], [100, 200], [-1, 5]), [-1, 2, 5])


def test_absent_trace_stays_missing_instead_of_becoming_zero():
    calibration = dict(pixel_tolerance=2, panels=[dict(cell=1,
        x_pixels=[5, 65], x_values=[-1, 5], y_pixels=[65, 5], y_values=[0, 10], crop_y=[0, 70])])
    points = digitize(np.full((70, 70, 3), 255, dtype=np.uint8), calibration)
    assert all(p['status'] == 'missing_or_occluded' for p in points)
    assert all(p['rate_spikes_s'] is None for p in points)


def test_antialiased_axis_is_not_misread_as_a_low_firing_curve():
    calibration = dict(pixel_tolerance=2, panels=[dict(cell=1,
        x_pixels=[5, 65], x_values=[-1, 5], y_pixels=[65, 5], y_values=[0, 10], crop_y=[0, 70])])
    rgb = np.full((70, 70, 3), 255, dtype=np.uint8)
    rgb[64:67, 5:66] = 155
    rgb[5:66, 5:7] = 155
    points = digitize(rgb, calibration)
    assert all(p['rate_spikes_s'] is None for p in points if p['condition'] == 'antipreferred')


def test_signed_and_residual_floor_fit_distinct_known_negative_responses():
    x = np.linspace(-2, 5, 29)
    for model, parameters in [('signed', [2., 3.]), ('residual_floor', [4., 2., 3.])]:
        # Known units: baseline 2 or 6, positive slope 3, with/without a floor of 4.
        y = parameters[-1] * x + (2. if model == 'signed' else 6.)
        y = np.maximum(0. if model == 'signed' else 4., y)
        fitted = fit_curve(model, x, y)
        np.testing.assert_allclose(fitted, parameters, atol=1e-5)
        np.testing.assert_allclose(predict(model, fitted, x), y, atol=1e-5)


def test_fixed_suppression_shape_cannot_explain_nonzero_negative_firing_by_rescaling():
    x = np.array([-1., -.5, 0., 1., 2., 3.])
    y = 4. + 2.*np.maximum(0, x)
    q = fit_curve('positive_only', x, y)
    np.testing.assert_allclose(q, [4., 2.], atol=1e-10)
    proposed = predict('proposed_shape', fit_curve('proposed_shape', x, y), x)
    np.testing.assert_array_equal(proposed[:2], [0., 0.])
    assert np.sum((proposed-y)**2) >= 32.


def test_nonfinite_measurements_are_rejected():
    with pytest.raises(ValueError):
        fit_curve('signed', [0, 1, 2, 3], [1, 2, np.nan, 4])
