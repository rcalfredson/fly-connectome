"""Loop construction preserves movement and closure without an intermediate stop."""

import numpy as np
import pytest

pytest.importorskip('torch')

from fly_connectome.home_vector.data import sample_protocol
from fly_connectome.home_vector.analysis.repeated_loops import repeat_protocol


def test_repeat_preserves_first_cycle_and_removes_only_unused_stop():
    circle = [[0., 0., 1., 0.], [1., 90., 1., 0.], [2., 180., 1., 0.],
              [3., 270., 1., 0.], [4., 270., 0., 0.]]
    repeated, n, period = repeat_protocol(circle, .0025)
    c, y = sample_protocol(repeated, .0025)
    original, _ = sample_protocol(circle, .0025)
    assert period == 4. and n == 1600
    np.testing.assert_array_equal(c[:n], original[:-1])
    np.testing.assert_array_equal(c[n:2*n], c[:n])
    np.testing.assert_array_equal(c[n], [0., 1., 0.])
    np.testing.assert_allclose(y[[0, n, 2*n]], 0., atol=1e-12)
    np.testing.assert_allclose(y[n:2*n+1], y[:n+1], atol=1e-12)


def test_open_path_is_not_misrepresented_as_a_closed_loop():
    with pytest.raises(ValueError):
        repeat_protocol([[0., 0., 1., 0.], [1., 0., 0., 0.]], .0025)
