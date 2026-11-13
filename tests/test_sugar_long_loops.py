"""Long-loop geometry agrees with the archived two-loop probe."""

import numpy as np
import pytest

pytest.importorskip('torch')

from fly_connectome.home_vector.analysis.sugar_long_loops import extend_protocol
from fly_connectome.home_vector.analysis.repeated_loops import repeat_protocol


def test_many_loop_extension_matches_original_two_loop_protocol():
    protocol=[[0,0,1,0],[1,90,1,0],[2,180,1,0],[3,270,1,0],[4,270,0,0]]
    a,n,period=extend_protocol(protocol,.0025,2)
    b,m,p=repeat_protocol(protocol,.0025)
    np.testing.assert_array_equal(a,b)
    assert n==m and period==p
    _,n,period=extend_protocol(protocol,.0025,16)
    assert n==1600 and period==4.
