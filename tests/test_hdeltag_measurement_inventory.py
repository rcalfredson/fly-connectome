"""Mean-dot coordinate calibration and absence checks."""

import numpy as np
import pytest

from fly_connectome.home_vector.analysis.hdeltag_measurement_inventory import extract_dot


def test_dot_center_preserves_offset_and_reversed_axis():
    rgb=np.full((20,20,3),255,dtype=np.uint8)
    rgb[9:12,9:12]=[20,100,20]
    dot=extract_dot(rgb,[5,5,15,15],'green',[15,5],[2,6],1)
    assert dot['value']==4.
    assert dot['x_pixel']==10.
    assert dot['pixel_envelope_low']==pytest.approx(3.2)
    assert dot['pixel_envelope_high']==pytest.approx(4.8)


def test_missing_pixels_are_not_invented_and_purple_uses_its_own_mask():
    rgb=np.full((20,20,3),255,dtype=np.uint8)
    with pytest.raises(ValueError,match='No dark'):
        extract_dot(rgb,[5,5,15,15],'green',[15,5],[0,1],1)
    rgb[9:12,9:12]=[110,30,140]
    dot=extract_dot(rgb,[5,5,15,15],'purple',[15,5],[0,1],1)
    assert dot['value']==.5
    with pytest.raises(ValueError,match='No dark'):
        extract_dot(rgb,[5,5,15,15],'green',[15,5],[0,1],1)
