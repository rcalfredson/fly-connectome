"""Independent scoring conventions for the frozen reliable evaluation."""

import numpy as np
import pytest

pytest.importorskip('torch')

from fly_connectome.home_vector.analysis.reliable_evaluation import (
    CONDITIONS,describe,aggregate,csv_rows,load_metric_rows,validate_completed_scores)
from fly_connectome.home_vector.data import sample_protocol


def test_zero_closure_is_not_zero_tracking_error_and_entry_is_not_center():
    target=np.array([[0.,0.],[1.,2.],[0.,0.]])
    controls=np.array([[0.,1.,0.]]*3);time=np.array([0.,1.,2.])
    row=dict(sugar_events_s=[1.],return_times_s=[2.])
    score,events,excursions=describe(np.zeros_like(target),target,None,controls,time,row,2.5,['hDeltaG'])
    assert score['mse_mm2']==2.5 and score['normalized_mse']==1. and score['endpoint_sq_mm2']==0.
    assert score['max_activity'] is None and score['dwell_mse_mm2'] is None
    assert events[0]['event_kind']=='patch_entry' and events[0]['error_mm']==np.sqrt(5.)
    assert events[1]['event_kind']=='center_return' and events[1]['error_mm']==0.
    assert len(excursions)==1 and excursions[0]['mse_mm2']==2.5


def test_phase_windows_are_unions_and_do_not_double_count_overlaps():
    time=np.arange(9)*.5;target=np.zeros((9,2));prediction=np.column_stack((time,np.zeros(9)))
    controls=np.array([[0.,1.,0.]]*9);row=dict(sugar_events_s=[.5,1.5])
    score,_,_=describe(prediction,target,None,controls,time,row,1.,['hDeltaG'])
    select=((time[:-1]>.5)&(time[:-1]<2.5))|((time[:-1]>1.5)&(time[:-1]<3.5))
    np.testing.assert_allclose(score['early_post_contact_mse_mm2'],np.mean(time[:-1][select]**2))


def test_aggregation_uses_fixed_weights_and_renormalizes_missing_phases():
    rows=[]
    for i,(weight,mse,normalized,moving) in enumerate(((.25,1.,1.,None),(.75,9.,3.,9.))):
        rows.append(dict(split='test',condition='original',id=str(i),family=str(i),stratum='one',weight=weight,
            mse_mm2=mse,normalized_mse=normalized,endpoint_sq_mm2=mse,max_activity=None,moving_mse_mm2=moving))
    result=next(r for r in aggregate(rows) if r['subset']=='overall')
    np.testing.assert_allclose(result['vector_rmse_mm'],np.sqrt(7.))
    assert result['normalized_mse']==2.5 and result['moving_rms_mm']==3.


def test_completion_preserves_fresh_subset_labels_and_missing_metadata(tmp_path):
    row=dict(split='test',condition='original',id='one',family='one',stratum='returns',
        route_class=None,traversal='forward',return_count=4,end_away=False,contacts=4,
        weight=1.,mse_mm2=2.,normalized_mse=1.,endpoint_sq_mm2=0.,max_activity=None)
    path=tmp_path/'metrics.csv'
    csv_rows(path,[row])
    restored=load_metric_rows(path)
    assert restored==[row]
    assert type(restored[0]['return_count']) is int
    assert type(restored[0]['end_away']) is bool
    assert aggregate(restored)==aggregate([row])


def test_completion_rejects_missing_event_or_excursion_scores():
    record=dict(id='one',protocol=[[0.,0.,1.,0.],[1.,180.,1.,0.],[2.,180.,0.,0.]],
        sugar_events_s=[1.5],return_times_s=[2.])
    controls,target=sample_protocol(record['protocol'],.5)
    rows,events,excursions=[],[],[]
    for condition in CONDITIONS:
        score,er,xr=describe(np.zeros_like(target),target,None,controls,np.arange(5)*.5,record,1.,['A'])
        meta=dict(split='test',id='one',condition=condition)
        rows.append(dict(meta,**score))
        events.extend(dict(meta,**r) for r in er)
        excursions.extend(dict(meta,**r) for r in xr)
    sets={'test':[record]}
    validate_completed_scores(sets,.5,rows,events,excursions)
    with pytest.raises(ValueError,match='incomplete event'):
        validate_completed_scores(sets,.5,rows,events[:-1],excursions)
    with pytest.raises(ValueError,match='incomplete excursion'):
        validate_completed_scores(sets,.5,rows,events,excursions[:-1])
