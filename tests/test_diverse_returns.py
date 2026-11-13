"""World-path matching, explicit contacts, family leakage and fixed task scales."""

import numpy as np
import pytest

torch=pytest.importorskip('torch')

from fly_connectome.home_vector.diverse_returns import (
    ROUTES,STRATA,return_family,diagnostic_controls,sampling_weights,loss_plan,validate,
)
from fly_connectome.home_vector.data import sample_protocol
from fly_connectome.home_vector.sugar_training import evaluate,checkpoint_score
from fly_connectome.home_vector.analysis.sugar_response import SugarResponse
from test_sugar_training import circuit
from test_sugar_training import derived_task


@pytest.mark.parametrize('route',ROUTES)
@pytest.mark.parametrize('end_away',[False,True])
def test_route_siblings_share_world_path_and_only_deliver_on_eligible_contact(route,end_away):
    records=return_family(np.random.default_rng(71),route,route,.0025,end_away=end_away)
    assert len(records)==12 and len({r['parent_signature'] for r in records})==1
    for count in (4,8):
        variants=[r for r in records if r['return_count']==count]
        truth=[]
        for row in variants:
            controls,target=sample_protocol(row['protocol'],.0025)
            truth.append(target)
            np.testing.assert_allclose(target[np.rint(np.array(row['return_times_s'])/.0025).astype(int)],0.,atol=5e-6)
            moving=np.linalg.norm(controls[:,1:],axis=1)
            assert np.all(moving[moving>1e-8]>=.5-1e-10) and moving.max()<=2.+1e-10
            if end_away:assert np.linalg.norm(target[-1])>.15
            else:np.testing.assert_allclose(target[-1],0.,atol=5e-6)
            if row['arm']=='no_sugar':assert row['sugar_events_s']==[]
            elif row['arm']=='single_sugar':assert row['sugar_events_s']==row['eligible_contact_times_s'][:1]
            else:assert len(row['sugar_events_s'])==(count if end_away else count-1)
            for event in row['sugar_events_s']:
                assert np.linalg.norm(target[round(event/.0025)])<=.1+1e-10
        for target in truth[1:]:np.testing.assert_allclose(target,truth[0],atol=5e-6)


def toy_rows():
    return [dict(family=g,stratum=g,group='existing',kind='toy',arm='no_sugar',sugar_events_s=[],
        protocol=[[0.,0.,float(i+1),0.],[.1,0.,0.,0.]]) for i,g in enumerate(STRATA)]


def test_training_scale_normalization_equals_one_for_zero_predictor_and_is_frozen():
    rows=toy_rows();plan=loss_plan(rows,.0025)
    energies=plan['target_energy_mm2']
    np.testing.assert_allclose(energies[STRATA[-1]]/energies[STRATA[0]],16.)
    class Zero(torch.nn.Module):
        def forward(self,pfn,dt,gates):
            return pfn.new_zeros(*pfn.shape[:2],2),pfn.new_zeros(*pfn.shape[:2],2)
    score=evaluate(Zero(),circuit(),rows,.0025,'cpu',SugarResponse(.1,300.,1.,0.),False,normalization=plan)
    np.testing.assert_allclose(score['selection_score'],1.,rtol=1e-6)
    for subset in score['by_stratum'].values():np.testing.assert_allclose(subset['normalized_mse'],1.,rtol=1e-6)
    import copy
    held_out=copy.deepcopy(rows)
    for row in held_out:row['protocol'][0][2]*=2.
    shifted=evaluate(Zero(),circuit(),held_out,.0025,'cpu',SugarResponse(.1,300.,1.,0.),False,normalization=plan)
    np.testing.assert_allclose(shifted['selection_score'],4.,rtol=1e-6)
    assert plan['target_energy_mm2']==energies


def test_sampling_and_selection_use_strata_not_counts_or_raw_rmse():
    rows=toy_rows();rows.extend([dict(rows[0]) for _ in range(7)])
    p=sampling_weights(rows)
    for group in STRATA:np.testing.assert_allclose(sum(v for r,v in zip(rows,p) if r['stratum']==group),.25)
    assert checkpoint_score(dict(vector_rmse_mm=1.,selection_score=4.))>checkpoint_score(dict(vector_rmse_mm=2.,selection_score=1.))
    assert checkpoint_score(dict(vector_rmse_mm=3.))==3.


def test_cross_split_parent_variants_and_bad_extrapolation_are_rejected():
    import copy
    groups={}
    for i,split in enumerate(('train','validation','test')):
        groups[split]=copy.deepcopy(toy_rows())
        for row in groups[split]:
            row['family']=split+'-'+row['family'];row['protocol'][-1][0]+=.025*i
            row['parent_signature']=row['family']
    validate(groups)
    groups['test'][0]['parent_signature']=groups['train'][0]['parent_signature']
    with pytest.raises(ValueError,match='parent'):validate(groups)
    groups['test'][0]['parent_signature']='fresh'
    groups['extrapolation']=[dict(family='unknown')]
    with pytest.raises(ValueError,match='Extrapolation'):validate(groups)


def test_near_misses_and_away_pauses_never_generate_sugar():
    rows=diagnostic_controls(np.random.default_rng(19),.0025)
    assert len(rows)==4 and {r['route_class'] for r in rows}=={'near_miss','pause_away'}
    assert all(not r['sugar_events_s'] for r in rows)


def test_expanded_trainer_uses_frozen_normalized_selection_end_to_end(derived_task,tmp_path):
    import json
    from fly_connectome.home_vector import sugar_training
    from fly_connectome.home_vector.data import write_json,verify_bundle
    from fly_connectome.home_vector.__main__ import finish
    from fly_connectome.home_vector.diverse_returns import KIND
    groups={}
    for index,split in enumerate(('train','validation','test')):
        groups[split]=toy_rows()
        for row in groups[split]:
            row['family']=split+'-'+row['family'];row['id']=row['family']
            row['protocol'][-1][0]+=.025*index
    plan=loss_plan(groups['train'],.0025)
    write_json(derived_task/'trajectories.json',groups);write_json(derived_task/'loss_plan.json',plan)
    manifest=json.loads((derived_task/'manifest.json').read_text());manifest['kind']=KIND
    finish(derived_task,manifest)
    output=tmp_path/'expanded-fit'
    result=sugar_training.main(['--data',str(derived_task),'--output',str(output),
        '--updates','2','--batch-size','4','--validate-every','1'])
    verify_bundle(output)
    config=json.loads((output/'training_config.json').read_text())
    assert config['normalization']==plan and config['selection_score_key']=='selection_score'
    history=json.loads((output/'history.json').read_text())
    for name in ('original','sugar'):
        chosen=min(history,key=lambda r:r[name+'_validation']['selection_score'])
        assert result[name]['selected_update']==chosen['update']
        assert set(result[name]['test']['by_stratum'])==set(STRATA)
