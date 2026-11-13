"""Gradient and native-dynamics checks for the opt-in sugar trainer."""

import pytest

torch=pytest.importorskip('torch')

from fly_connectome.home_vector.data import Circuit
from fly_connectome.home_vector.model import HomeVectorNetwork
from fly_connectome.home_vector.sugar_training import SugarTrainingNetwork,prepared_batch
from fly_connectome.home_vector.analysis.sugar_response import SugarResponse,rollout


def circuit():
    return Circuit([1],['PFNd'],[2,3],['hDeltaG','hDeltaB'],[0.],[0.],[
        dict(pre=0,post=0,contacts=1,denominator=1.,source_type='PFNd',target_type='hDeltaG'),
        dict(pre=1,post=1,contacts=1,denominator=1.,source_type='hDeltaG',target_type='hDeltaB')])


def test_disabled_sugar_reproduces_native_states_predictions_and_parameter_gradients():
    c=circuit();response=SugarResponse(.1,300.,1.,0.)
    native=HomeVectorNetwork(c);net=SugarTrainingNetwork(c,response)
    with torch.no_grad():native.D.fill_(.2)
    net.load_state_dict(native.state_dict())
    x=torch.ones(2,71,1);g=torch.zeros(2,71)
    p,r=net(x,.0025,g);q,s=native(x,.0025)
    torch.testing.assert_close(p,q,atol=0,rtol=0);torch.testing.assert_close(r,s,atol=0,rtol=0)
    (p.square().mean()+r.square().mean()).backward()
    (q.square().mean()+s.square().mean()).backward()
    for a,b in zip(net.parameters(),native.parameters()):torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=2e-5)


def test_treated_states_match_frozen_probe_and_gradients_reach_recurrent_weights():
    c=circuit();response=SugarResponse(.1,300.,1.,0.);net=SugarTrainingNetwork(c,response)
    with torch.no_grad():net.a.fill_(1.);net.D.fill_(.2)
    x=torch.ones(1,121,1);g=torch.linspace(0.,1.,121)[None]
    p,r=net(x,.0025,g)
    with torch.no_grad():
        q,s=rollout(net,x,.0025,g[0].numpy(),response,torch.tensor([True,False]))
    torch.testing.assert_close(p,q,atol=1e-7,rtol=1e-6);torch.testing.assert_close(r,s,atol=1e-7,rtol=1e-6)
    p.square().mean().backward()
    assert torch.isfinite(net.a.grad).all() and abs(net.a.grad[1])>0
    assert torch.isfinite(net.tau.grad).all() and net.tau.grad.abs().sum()>0


def test_compiled_sugar_block_preserves_full_history_gradients():
    torch._dynamo.reset()
    c=circuit();response=SugarResponse(.1,300.,1.,0.)
    eager=SugarTrainingNetwork(c,response);compiled=SugarTrainingNetwork(c,response).compile_sugar(16,'aot_eager')
    with torch.no_grad():eager.a.fill_(1.);eager.D.fill_(.2)
    compiled.load_state_dict(eager.state_dict())
    x=torch.ones(1,66,1);g=torch.linspace(0.,1.,66)[None]
    p,r=eager(x,.0025,g);q,s=compiled(x,.0025,g)
    torch.testing.assert_close(p,q);torch.testing.assert_close(r,s)
    p.square().mean().backward();q.square().mean().backward()
    for a,b in zip(eager.parameters(),compiled.parameters()):torch.testing.assert_close(a.grad,b.grad,atol=1e-7,rtol=1e-5)


def test_variable_batch_sizes_reuse_graphs_and_match_eager_gradients():
    torch._dynamo.reset()
    graphs=[]
    def backend(graph,inputs):
        graphs.append(graph)
        return graph.forward
    c=circuit();response=SugarResponse(.1,300.,1.,0.)
    eager=SugarTrainingNetwork(c,response)
    with torch.no_grad():eager.a.fill_(1.);eager.D.fill_(.2)
    models=[SugarTrainingNetwork(c,response).compile_sugar(8,backend) for _ in range(2)]
    for net in models:net.load_state_dict(eager.state_dict())
    for size in (1,2,3,4,5,6,7,8,2,1):
        steps=27+size
        x=torch.ones(size,steps,1)
        for treated,net in enumerate(models):
            gates=torch.linspace(0.,float(treated),steps)[None].expand(size,-1).contiguous()
            net.zero_grad(set_to_none=True);eager.zero_grad(set_to_none=True)
            p,r=net(x,.0025,gates);q,s=eager(x,.0025,gates)
            torch.testing.assert_close(p,q);torch.testing.assert_close(r,s)
            p.square().mean().backward();q.square().mean().backward()
            for a,b in zip(net.parameters(),eager.parameters()):
                torch.testing.assert_close(a.grad,b.grad,atol=1e-7,rtol=1e-5)
    assert 1<=len(graphs)<=2
    count=len(graphs)
    with torch.no_grad():models[0](torch.ones(3,29,1),.0025,torch.zeros(3,29))
    assert len(graphs)==count


def test_prepared_mask_excludes_endpoint_and_gates_are_independent_per_member():
    rows=[dict(protocol=[[0.,0.,1.,0.],[.1,0.,0.,0.]],sugar_events_s=e) for e in ([],[.05])]
    x,y,mask,g=prepared_batch(circuit(),rows,.0025,'cpu',SugarResponse(.1,300.,1.,0.))
    assert mask[:,-1].sum()==0 and mask[:,:-1].all()
    assert not g[0].any() and g[1,-1]>0
    torch.testing.assert_close(x[0],x[1]);torch.testing.assert_close(y[0],y[1])


@pytest.fixture
def derived_task(tmp_path):
    from fly_connectome.home_vector.__main__ import begin,finish
    from fly_connectome.home_vector.data import digest,write_json
    source,manifest=begin(tmp_path/'source','home_vector_prepared_v1')
    write_json(source/'circuit.json',circuit().as_dict())
    write_json(source/'task.json',dict(length_scale_mm=1.))
    finish(source,manifest)
    data,manifest=begin(tmp_path/'data','home_vector_sugar_decoder_data_v1')
    manifest['immutable_input_sha256']={str(p):digest(p) for p in source.iterdir()}
    rows={}
    for index,split in enumerate(('train','validation','test')):
        heading=index*30.
        protocol=[[j*.025,(heading+90*(j%4))%360,1.,0.] for j in range(8)]
        protocol.append([.2,heading+270.,0.,0.])
        rows[split]=[
            dict(id=split+'-straight',family=split+'-straight',group='existing',arm='no_sugar',
                 protocol=[[0.,heading,1.,0.],[.1+index*.025,heading,0.,0.]],sugar_events_s=[]),
            dict(id=split+'-return',family=split+'-return',group='returns',arm='continuous_sugar',
                 traversal='forward',protocol=protocol,sugar_events_s=[.1])]
    write_json(data/'trajectories.json',rows)
    write_json(data/'task.json',dict(source=str(source),activity=str(source),diagnostics=str(source),dt_s=.0025))
    write_json(data/'response_config.json',dict(response=SugarResponse(.1,300.,1.,0.).__dict__))
    finish(data,manifest)
    return data


def test_paired_training_selects_on_validation_and_preserves_recovery_state(derived_task,tmp_path,monkeypatch):
    import json
    import fly_connectome.home_vector.sugar_training as training
    from fly_connectome.home_vector.data import verify_bundle
    calls=[]
    actual_evaluate=training.evaluate
    def observed(*args,**kwargs):
        calls.append(args[2][0]['id'].split('-')[0])
        return actual_evaluate(*args,**kwargs)
    monkeypatch.setattr(training,'evaluate',observed)
    output=tmp_path/'fit'
    result=training.main(['--data',str(derived_task),'--output',str(output),
                         '--updates','2','--batch-size','2','--validate-every','1'])
    verify_bundle(output)
    assert calls==['validation']*4+['test']*2
    history=json.loads((output/'history.json').read_text())
    paired=torch.load(output/'paired_last.pt',weights_only=False,map_location='cpu')
    assert paired['update']==2
    initial=[]
    for name in ('original','sugar'):
        initial.append(torch.load(output/(name+'_initial.pt'),weights_only=True))
        last=torch.load(output/(name+'_last.pt'),weights_only=False,map_location='cpu')
        for key,value in last['model'].items():
            torch.testing.assert_close(value,paired['arms'][name]['model'][key],atol=0,rtol=0)
        assert last['numpy_rng']==paired['numpy_rng']
        for key,slots in last['optimizer']['state'].items():
            for slot,value in slots.items():
                torch.testing.assert_close(value,paired['arms'][name]['optimizer']['state'][key][slot],atol=0,rtol=0)
        chosen=min(history,key=lambda r:r[name+'_validation']['vector_rmse_mm'])
        assert result[name]['selected_update']==chosen['update']
    for key,value in initial[0].items():torch.testing.assert_close(value,initial[1][key],atol=0,rtol=0)


def test_failed_second_arm_keeps_last_complete_paired_checkpoint(derived_task,tmp_path,monkeypatch):
    import json
    import fly_connectome.home_vector.sugar_training as training
    actual_step=torch.optim.Adam.step
    calls=0
    def failing_step(self,*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==4:raise RuntimeError('injected second-arm optimizer failure')
        return actual_step(self,*args,**kwargs)
    monkeypatch.setattr(torch.optim.Adam,'step',failing_step)
    output=tmp_path/'failed'
    with pytest.raises(RuntimeError,match='injected'):
        training.main(['--data',str(derived_task),'--output',str(output),
                       '--updates','2','--batch-size','2','--validate-every','1'])
    assert json.loads((output/'manifest.json').read_text())['status']=='failed'
    paired=torch.load(output/'paired_last.pt',weights_only=False,map_location='cpu')
    assert paired['update']==1
    assert len(json.loads((output/'history.json').read_text()))==1
    for name in ('original','sugar'):
        best=torch.load(output/(name+'_best.pt'),weights_only=False,map_location='cpu')
        for key,value in best['model'].items():
            torch.testing.assert_close(value,paired['arms'][name]['model'][key],atol=0,rtol=0)
