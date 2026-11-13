"""Reliable delivery, duplicate removal, preservation and paired-arm semantics."""

import copy
import json
from argparse import Namespace

import numpy as np
import pytest

torch=pytest.importorskip('torch')

from fly_connectome.home_vector import reliable_sugar,diverse_returns,sugar_training
from fly_connectome.home_vector.data import digest,write_json,verify_bundle,sample_protocol
from fly_connectome.home_vector.__main__ import finish
from test_sugar_training import derived_task
from test_diverse_returns import toy_rows


def two_returns(heading=0.):
    return [[0.,heading,1.,0.],[.5,(heading+180)%360,1.,0.],
            [1.,heading,1.,0.],[1.5,(heading+180)%360,1.,0.],[2.,heading,0.,0.]]


def test_every_reentry_delivers_including_final_approach_with_no_refractory_interval():
    row=dict(id='one',family='f',group='returns',protocol=two_returns(),sugar_events_s=[])
    events=reliable_sugar.contact_events(row,.0025)
    assert len(events)==2 and .8<events[0]<1. and 1.8<events[1]<2.
    converted=reliable_sugar.convert_rows([row],.0025)[0]
    assert converted['sugar_events_s']==events
    assert converted['arm']=='home_reentry_sugar'
    _,target=sample_protocol(row['protocol'],.0025)
    # Entry at the patch edge is not a geometric reset to a zero target.
    assert np.linalg.norm(target[round(events[0]/.0025)])>0.


def test_initial_occupancy_dwell_and_away_pause_do_not_generate_extra_delivery():
    stationary=dict(protocol=[[0.,0.,0.,0.],[2.,0.,0.,0.]])
    assert reliable_sugar.contact_events(stationary,.0025)==[]
    away=dict(protocol=[[0.,0.,1.,0.],[1.,0.,0.,0.],[2.,0.,0.,0.]])
    assert reliable_sugar.contact_events(away,.0025)==[]
    dwell=dict(protocol=[[0.,0.,1.,0.],[.5,180.,1.,0.],[1.,0.,0.,0.],[2.,0.,0.,0.]])
    assert len(reliable_sugar.contact_events(dwell,.0025))==1


def test_schedule_siblings_collapse_without_changing_movements_or_source_records():
    rows=[dict(id=f'old-{i}',family='f',group='diverse_returns',protocol=two_returns(),
               sugar_events_s=e,arm=s) for i,(e,s) in enumerate((([], 'no_sugar'),([.9], 'single_sugar'),([.9,1.9],'repeated_sugar')))]
    before=copy.deepcopy(rows);result=reliable_sugar.convert_rows(rows,.0025)
    assert rows==before and len(result)==1
    assert result[0]['protocol']==rows[0]['protocol']
    assert result[0]['source_variant_ids']==['old-0','old-1','old-2']
    assert len(result[0]['sugar_events_s'])==2


@pytest.fixture
def reliable_task(derived_task,tmp_path):
    groups={}
    for i,split in enumerate(('train','validation','test')):
        groups[split]=[]
        for base in toy_rows():
            base.update(family=split+'-'+base['stratum'])
            base['protocol'][-1][0]+=.025*i
            if base['stratum'] in ('two_return','diverse_return'):
                base['group']='returns' if base['stratum']=='two_return' else 'diverse_returns'
                base['protocol']=two_returns(i*30.)
                for j in range(2 if base['group']=='returns' else 3):
                    groups[split].append(dict(base,id=base['family']+f'-{j}',arm='source-schedule'))
            else:groups[split].append(dict(base,id=base['family']))
    write_json(derived_task/'trajectories.json',groups)
    torch.save(dict(reference_only=True),derived_task/'current_checkpoint.pt')
    write_json(derived_task/'current_metadata.json',dict(test_reference=True))
    (derived_task/'neurons.csv').write_text('body_id\n2\n3\n')
    manifest=json.loads((derived_task/'manifest.json').read_text());manifest['kind']=diverse_returns.KIND
    finish(derived_task,manifest)
    hashes={str(p):digest(p) for p in derived_task.iterdir() if p.is_file()}
    output=tmp_path/'reliable'
    reliable_sugar.prepare(Namespace(source=derived_task,output=output))
    assert all(digest(p)==h for p,h in hashes.items())
    return output


def test_builder_recomputes_all_contacts_and_rejects_missing_delivery(reliable_task):
    verify_bundle(reliable_task)
    task=json.loads((reliable_task/'task.json').read_text())
    assert task['delivery_policy']['mode']==reliable_sugar.POLICY
    groups=json.loads((reliable_task/'trajectories.json').read_text())
    assert all(len(rows)==4 for rows in groups.values())
    row=next(r for r in groups['train'] if r['stratum']=='two_return')
    assert len(row['sugar_events_s'])==2
    row['sugar_events_s']=row['sugar_events_s'][:1]
    with pytest.raises(ValueError,match='Missing or spurious'):reliable_sugar.validate(groups,.0025)


def test_paired_training_reliably_gates_only_the_sugar_arm(reliable_task,tmp_path,monkeypatch):
    seen={};actual=sugar_training.SugarTrainingNetwork.forward
    def observed(self,pfn,dt,gates):
        seen.setdefault(id(self),[]).append(bool(gates.any()))
        return actual(self,pfn,dt,gates)
    monkeypatch.setattr(sugar_training.SugarTrainingNetwork,'forward',observed)
    output=tmp_path/'fit'
    result=sugar_training.main(['--data',str(reliable_task),'--output',str(output),
                              '--updates','2','--batch-size','4','--validate-every','1'])
    verify_bundle(output)
    config=json.loads((output/'training_config.json').read_text())
    assert config['delivery_policy']['mode']==reliable_sugar.POLICY
    assert config['original_network_gate']=='Always zero'
    calls=list(seen.values());assert len(calls)==2 and not any(calls[0]) and any(calls[1])
    assert result['sugar']['test']['selection_metric']=='mean_stratum_normalized_mse'
