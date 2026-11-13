"""Selected-checkpoint identity and honest loop/error scoring for frozen replay."""

import json

import numpy as np
import pytest

torch=pytest.importorskip('torch')

from fly_connectome.home_vector.__main__ import begin,finish
from fly_connectome.home_vector.data import write_json
from fly_connectome.home_vector.analysis.sugar_response import SugarResponse
from fly_connectome.home_vector.analysis.sugar_retrained_loops import load_selected_models,loop_metrics
from fly_connectome.home_vector.sugar_training import SugarTrainingNetwork
from test_sugar_training import circuit


@pytest.fixture
def selected_bundle(tmp_path):
    path,manifest=begin(tmp_path/'training','home_vector_sugar_paired_training_v1',seed=0,smoke=False,benchmark=False)
    c=circuit();response=SugarResponse(.1,300.,1.,0.)
    write_json(path/'circuit.json',c.as_dict())
    write_json(path/'training_config.json',dict(response=response.__dict__,dt_s=.0025))
    history=[dict(update=1,original_validation=dict(vector_rmse_mm=1.),sugar_validation=dict(vector_rmse_mm=1.1)),
             dict(update=2,original_validation=dict(vector_rmse_mm=1.2),sugar_validation=dict(vector_rmse_mm=.8))]
    comparison={}
    for name,update,value in (('original',1,.2),('sugar',2,.3)):
        net=SugarTrainingNetwork(c,response)
        with torch.no_grad():net.D.fill_(value)
        validation=history[update-1][name+'_validation']
        torch.save(dict(model=net.state_dict(),update=update,validation=validation),path/(name+'_best.pt'))
        with torch.no_grad():net.D.fill_(9.)
        torch.save(dict(model=net.state_dict(),update=2),path/(name+'_last.pt'))
        comparison[name]=dict(selected_update=update,validation=validation)
    write_json(path/'history.json',history);write_json(path/'comparison.json',comparison)
    finish(path,manifest)
    return path


def test_each_arm_loads_its_own_validation_selection_not_last(selected_bundle):
    models,c,response,config,selection=load_selected_models(selected_bundle)
    assert selection['original']['update']==1 and selection['sugar']['update']==2
    torch.testing.assert_close(models['original'].D,torch.full_like(models['original'].D,.2))
    torch.testing.assert_close(models['sugar'].D,torch.full_like(models['sugar'].D,.3))
    assert not any(net.training for net in models.values())


def test_inconsistent_selection_is_rejected_even_when_artifact_hashes_match(selected_bundle):
    path=selected_bundle/'original_best.pt'
    checkpoint=torch.load(path,weights_only=False,map_location='cpu');checkpoint['update']=2
    torch.save(checkpoint,path)
    manifest=json.loads((selected_bundle/'manifest.json').read_text());finish(selected_bundle,manifest)
    with pytest.raises(ValueError,match='validation history'):load_selected_models(selected_bundle)


def test_zero_output_can_have_perfect_closure_but_nonzero_tracking_error():
    target=np.array([[0.,0.],[1.,2.],[0.,0.]])
    rows=loop_metrics('toy','original_trained',np.zeros_like(target),np.ones((3,2)),target,
                      n=2,loops=1,dynamic_types=['hDeltaG','hDeltaB'])
    row=rows[0]
    assert row['closure_error_mm']==0.
    assert row['vector_rmse_mm']==row['zero_prediction_rmse_mm']==np.sqrt(2.5)
    assert row['mean_x_error_mm']==-.5 and row['mean_y_error_mm']==-1.
    np.testing.assert_allclose(row['centered_error_rms_mm'],np.sqrt(1.25))
