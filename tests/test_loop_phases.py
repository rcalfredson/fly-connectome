"""Phase alignment must distinguish repeatability, accuracy and hidden drift."""

import numpy as np
import pytest

pytest.importorskip('torch')
from fly_connectome.home_vector.analysis.loop_phases import aligned_metrics


def test_constant_population_mean_can_hide_individual_drift():
    # Both loops have population mean 2, but opposite neurons change by 1.
    rates=np.array([[1.,3.],[1.,3.],[2.,2.],[2.,2.],[2.,2.]])
    prediction=np.ones((5,2));target=np.zeros((5,2))
    rows,populations,neurons=aligned_metrics(rates,prediction,target,2,2,['A','A'],[10,11])
    assert populations[0]['mean_activity']==populations[1]['mean_activity']==2.
    np.testing.assert_allclose(rows[1]['adjacent_state_rms'],np.sqrt(2))
    assert rows[1]['adjacent_prediction_rms_mm']==0.
    np.testing.assert_allclose(rows[1]['vector_rmse_mm'],np.sqrt(2))
    assert [r['adjacent_change_rms'] for r in neurons[2:]]==[1.,1.]


def test_terminal_state_is_not_included_in_phase_average():
    rates=np.array([[1.],[2.],[1.],[2.],[100.]])
    prediction=np.zeros((5,2));prediction[-1]=[3.,4.]
    rows,_,_=aligned_metrics(rates,prediction,np.zeros((5,2)),2,2,['A'],[10])
    assert rows[1]['adjacent_state_rms']==0.
    assert rows[1]['max_activity']==2.
    assert rows[1]['closure_state_norm']==100.
    assert rows[1]['vector_rmse_mm']==0.
    assert rows[1]['closure_error_mm']==5.


def test_comparison_requires_matching_world_targets_and_complete_loops():
    rates=np.zeros((5,1));p=np.zeros((5,2));y=p.copy();y[2]=[1.,0.]
    with pytest.raises(AssertionError):aligned_metrics(rates,p,y,2,2,['A'],[10])
    with pytest.raises(ValueError):aligned_metrics(rates[:-1],p,y,2,2,['A'],[10])
