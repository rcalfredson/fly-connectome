"""Environmental events, sealed family splits, and readout-only fitting checks."""

import numpy as np
import pytest

torch=pytest.importorskip('torch')

from fly_connectome.home_vector.analysis.sugar_decoder_data import patch_events,return_family,validate_splits
from fly_connectome.home_vector.analysis.sugar_decoder_fit import (
    family_weights,moments,merge,mse,fit_ridge,batched_sugar_rates,add_record,
)
from fly_connectome.home_vector.analysis.sugar_response import SugarResponse,event_gate,rollout
from fly_connectome.home_vector.data import Circuit,sample_protocol
from fly_connectome.home_vector.model import HomeVectorNetwork


def test_patch_requires_departure_and_consumes_refractory_contacts():
    target=np.column_stack(([0.,.2,.05,.05,.2,.05,.05,.2,.05],np.zeros(9)))
    events=patch_events(np.zeros((9,3)),target,10.,minimum_interval=30.)
    assert events==[20.,50.]  # Last contact at 80 s has no following state.
    events=patch_events(np.zeros((9,3)),target,1.,minimum_interval=30.)
    assert events==[2.]


def test_return_variants_share_geometry_and_stay_closed_without_target_reset():
    rows=return_family(np.random.default_rng(12),'test-family',.0025)
    assert len(rows)==8
    assert {r['traversal'] for r in rows}=={'forward','rightward'}
    for traversal in ('forward','rightward'):
        a,b=[r for r in rows if r['traversal']==traversal and r['arm'].startswith('continuous')]
        assert a['protocol']==b['protocol']
        assert a['sugar_events_s']==[] and len(b['sugar_events_s'])>=1
        _,target=sample_protocol(b['protocol'],.0025)
        np.testing.assert_allclose(target[-1],0.,atol=1e-12)
        for event in b['sugar_events_s']:
            assert np.linalg.norm(target[round(event/.0025)])<=.1
    a=[r for r in rows if r['traversal']=='forward' and r['arm']=='continuous_sugar'][0]
    b=[r for r in rows if r['traversal']=='rightward' and r['arm']=='continuous_sugar'][0]
    np.testing.assert_allclose(sample_protocol(a['protocol'],.0025)[1],sample_protocol(b['protocol'],.0025)[1],atol=1e-12)


def test_family_leakage_and_exact_protocol_leakage_are_rejected():
    groups={s:[dict(family=s,protocol=[[0,0,1,0],[i+1,0,0,0]])]
            for i,s in enumerate(('train','validation','test'))}
    validate_splits(groups)
    groups['test'][0]['family']='train'
    with pytest.raises(ValueError,match='families'):validate_splits(groups)
    groups['test'][0]['family']='test'
    groups['test'][0]['protocol']=groups['train'][0]['protocol']
    with pytest.raises(ValueError,match='protocol'):validate_splits(groups)


def test_domain_family_and_sibling_weights_do_not_reward_duplicate_variants():
    rows=[dict(group='existing',family='a') for _ in range(3)]+[dict(group='existing',family='b')]
    rows += [dict(group='returns',family='c') for _ in range(8)]
    w=family_weights(rows)
    np.testing.assert_allclose([w[:3].sum(),w[3],w[4:].sum()],[.25,.25,.5])


def test_sufficient_statistic_score_matches_direct_prediction_and_fits_known_readout():
    rng=np.random.default_rng(9)
    r=rng.normal(size=(100,3));true=np.array([[1.,2.],[-.5,.3],[.2,-.8],[3.,-1.]])
    y=np.column_stack((r,np.ones(100)))@true
    s=moments(r,y)
    wrong=true+1.
    np.testing.assert_allclose(mse(s,wrong),np.mean(np.sum((np.column_stack((r,np.ones(100)))@wrong-y)**2,axis=1)))
    store={};merge(store,'train',s,1.);merge(store,'validation',s,1.)
    fit=fit_ridge(store['train'],store['validation'])
    np.testing.assert_allclose(fit['weights'],true,atol=5e-6)


def test_batch_gate_is_native_when_disabled_and_matches_individual_probe():
    c=Circuit([1],['PFNd'],[2,3],['hDeltaG','hDeltaB'],[0.],[0.],[
        dict(pre=0,post=0,contacts=1,denominator=1.,source_type='PFNd',target_type='hDeltaG'),
        dict(pre=1,post=1,contacts=1,denominator=1.,source_type='hDeltaG',target_type='hDeltaB')])
    net=HomeVectorNetwork(c);pfn=torch.ones(2,101,1);mask=torch.tensor([True,False])
    response=SugarResponse(.1,300.,1.,0.)
    gates=np.stack((np.zeros(101),event_gate(np.arange(101)*.0025,[.05],response)))
    with torch.no_grad():
        _,native=net(pfn,.0025)
        zero=batched_sugar_rates(net,pfn,.0025,np.zeros_like(gates),response,mask)
        observed=batched_sugar_rates(net,pfn,.0025,gates,response,mask)
        _,individual=rollout(net,pfn[1:2],.0025,gates[1],response,mask)
    torch.testing.assert_close(native,zero,atol=0,rtol=0)
    torch.testing.assert_close(observed[0],native[0],atol=0,rtol=0)
    torch.testing.assert_close(observed[1:],individual,atol=1e-7,rtol=1e-6)


def test_closure_states_are_scored_separately_from_time_averaged_samples():
    rates=np.array([[0.],[2.],[10.]])
    target=np.zeros((3,2));store={}
    row=dict(group='returns',family='f',arm='continuous_sugar',sugar_events_s=[.5],loop_period_s=1.)
    add_record(store,'test','sugar',row,rates,target,np.array([0.,1.,2.]),1.)
    weights=np.array([[1.,0.],[0.,0.]])
    assert mse(store['test','sugar','overall'],weights)==2.
    assert mse(store['test','sugar','closure:first_return'],weights)==4.
    assert mse(store['test','sugar','closure:final_return'],weights)==100.
