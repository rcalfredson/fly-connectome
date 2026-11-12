"""Unit scaling cannot change normalized response shape or invent missing data."""

import numpy as np
import pytest

from fly_connectome.home_vector.analysis.hdeltag_reset_assessment import affine_normalize, extract_mean_curve


def test_fixed_positive_gain_and_offset_preserve_anchored_response():
    values = np.array([8., 1., 3., 6.])
    original = affine_normalize(values, 8., 1.)
    np.testing.assert_allclose(original, affine_normalize(5.*values + 11., 51., 16.))
    with pytest.raises(ValueError):
        affine_normalize(values, 1., 1.)


def test_digitization_preserves_physical_offsets_and_missing_trace():
    c = dict(x_pixels=[5, 15], x_seconds=[0, 100], y_pixels=[25, 5], y_hz=[5, 15],
             crop_y=[0, 30], half_strip_width_pixels=1, dark_threshold=85,
             envelope_padding_pixels=1, sample_seconds=[0, 100])
    image = np.full((30, 20, 3), 255, dtype=np.uint8)
    image[15, 4:7] = 0
    rows = extract_mean_curve(image, c)
    assert rows[0]['rate_hz'] == 10.
    assert rows[0]['envelope_low_hz'] == 9.5
    assert rows[0]['envelope_high_hz'] == 10.5
    assert rows[1]['status'] == 'missing'
    assert rows[1]['rate_hz'] is None
