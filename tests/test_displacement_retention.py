import numpy as np
import pytest

pytest.importorskip('torch')
from fly_connectome.home_vector.analysis.displacement_retention import waveform_error,pair_metrics,prefixes
from fly_connectome.home_vector.data import sample_protocol


def test_error_budget_separates_offset_gain_timing_and_higher_harmonics():
    dt=.01;t=np.arange(800)*dt;angle=2*np.pi*t/8
    target=np.column_stack((np.cos(angle),np.sin(angle)))
    prediction=np.column_stack((2*np.cos(angle-.2),.5*np.sin(angle)))+[1.,-2.]
    prediction[:,0]+=.3*np.cos(2*angle)
    score,_,_=waveform_error(prediction,target,dt)
    np.testing.assert_allclose(score['offset_mse_mm2'],5.)
    np.testing.assert_allclose(score['residual_mse_mm2'],.3**2/2)
    np.testing.assert_allclose([score['x_amplitude_ratio'],score['y_amplitude_ratio']],[2.,.5])
    np.testing.assert_allclose(score['x_peak_lag_s'],.2*8/(2*np.pi))
    np.testing.assert_allclose(score['y_peak_lag_s'],0.,atol=1e-12)


def test_norm_alone_does_not_establish_retained_displacement_direction():
    a=np.zeros((3,2));b=np.array([[2.,0.],[0.,2.],[-2.,0.]])
    values=pair_metrics(a,b,a,b,np.array([2.,0.]))
    np.testing.assert_array_equal(values['prediction_separation_mm'],[2.,2.,2.])
    np.testing.assert_array_equal(values['signed_target_gain'],[1.,0.,-1.])
    assert 'signed_target_gain' not in pair_metrics(a,b,a,b,np.zeros(2))


def test_prefixes_are_equal_duration_reachable_and_include_equal_home_control():
    ends={}
    for name,protocol in prefixes().items():
        controls,target=sample_protocol(protocol,.0025)
        assert len(controls)==1601
        np.testing.assert_array_equal(controls[-1],[45.,0.,0.])
        ends[name]=target[-1]
    np.testing.assert_allclose(ends['closed_square'],ends['rest'],atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(ends['forward_4mm']),4.)
    np.testing.assert_allclose(np.dot(ends['forward_4mm'],ends['right_4mm']),0.,atol=1e-12)
