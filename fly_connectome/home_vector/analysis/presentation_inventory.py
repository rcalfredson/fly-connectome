"""Audit current model flexibility and numerical sensitivity for presentation claims."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from ..__main__ import begin,fail,finish
from ..data import digest,encode,sample_protocol,verify_bundle,write_json
from ..model import HomeVectorNetwork
from .reliable_evaluation import csv_rows
from .sugar_retrained_loops import load_selected_models


def parameter_inventory(net,circuit):
    a=net.a.detach().numpy();initial=net.a_initial.numpy();pre=net.pre.numpy();counts=net.counts.numpy()
    internal=pre>=net.input_size;input_edges=~internal
    result=dict(dynamic_neurons=len(circuit.dynamic_ids),dynamic_types=dict(Counter(circuit.dynamic_types)),
        prescribed_inputs=len(circuit.input_ids),input_types=dict(Counter(circuit.input_types)),
        anatomical_edges=len(a),internal_edges=int(internal.sum()),input_edges=int(input_edges.sum()),
        tau_s=net.tau.detach().tolist(),population_types=net.population_types,
        internal_initial_sign_changes=int(((a*initial<0)&internal).sum()),
        internal_trained_to_initial_effective_weight_norm=float(np.linalg.norm(a[internal]*counts[internal])/np.linalg.norm(initial[internal]*counts[internal])))
    sources=[]
    for source in np.unique(pre[internal]):
        weights=a[pre==source]*counts[pre==source]
        sources.append(dict(neuron_id=circuit.dynamic_ids[source-net.input_size],population=circuit.dynamic_types[source-net.input_size],
            positive_edges=int((weights>0).sum()),negative_edges=int((weights<0).sum()),
            mixed_effective_outgoing_signs=bool((weights>0).any() and (weights<0).any())))
    result['mixed_effective_outgoing_sources']=sum(r['mixed_effective_outgoing_signs'] for r in sources)
    return result,sources


def analyze(args):
    training,retention,output=[Path(getattr(args,k)).resolve() for k in ('training','retention','output')]
    tm=verify_bundle(training);rm=verify_bundle(retention)
    models,circuit,response,config,selection=load_selected_models(training)
    frozen=dict(tm['immutable_input_sha256']);frozen.update(rm['immutable_input_sha256'])
    for folder in (training,retention):frozen.update({str(p):digest(p) for p in folder.iterdir() if p.is_file()})
    for source in frozen:
        folder=Path(source).parent
        if output==folder or output.is_relative_to(folder) or folder.is_relative_to(output):raise ValueError('Output overlaps source.')
    if any(digest(p)!=h for p,h in frozen.items()):raise ValueError('Source changed.')
    output,manifest=begin(output,'home_vector_presentation_inventory_v1',training=str(training),retention=str(retention))
    manifest.update(immutable_input_sha256=frozen,selection=selection,training=False,refitting=False)
    write_json(output/'manifest.json',manifest)
    try:
        shutil.copyfile(__file__,output/'presentation_inventory.py')
        inventory={};sources=[]
        for name,net in models.items():
            inventory[name],rows=parameter_inventory(net,circuit)
            sources.extend(dict(condition=name,**r) for r in rows)
        write_json(output/'parameters.json',inventory);csv_rows(output/'source_signs.csv',sources)
        design=json.loads((retention/'design.json').read_text());names=list(design['prefix_protocols']);fine_dt=config['dt_s']/2
        controls=[sample_protocol(v,fine_dt)[0] for v in design['prefix_protocols'].values()]
        net=HomeVectorNetwork(circuit);net.load_state_dict(models['original'].state_dict());net.eval();torch.set_num_threads(1)
        with torch.no_grad():p,r=net(torch.tensor(np.array([encode(circuit,c) for c in controls]),dtype=torch.float32),fine_dt)
        initial=r[:,-1].clone();checks=[]
        for branch in ('pause_32s','path_09','path_15'):
            controls,target=sample_protocol(design['continuations'][branch],fine_dt)
            x=torch.tensor(np.repeat(encode(circuit,controls)[None],len(names),axis=0),dtype=torch.float32)
            with torch.no_grad():pf,rf=net(x,fine_dt,initial=initial)
            pf,rf=pf[:,::2].numpy(),rf[:,::2].numpy()
            with np.load(retention/(branch+'.npz')) as coarse:
                for j,name in enumerate(names):
                    checks.append(dict(branch=branch,prefix=name,prediction_step_difference_rms_mm=float(np.sqrt(np.mean(np.sum((pf[j].astype(float)-coarse['prediction'][j])**2,axis=1)))),
                        state_entry_step_difference_rms=float(np.sqrt(np.mean((rf[j].astype(float)-coarse['rates'][j])**2)))))
                for j in (names.index('forward_4mm'),names.index('right_4mm')):
                    fine_separation=float(np.linalg.norm(pf[j,-1]-pf[0,-1]));coarse_separation=float(np.linalg.norm(coarse['prediction'][j,-1]-coarse['prediction'][0,-1]))
                    checks.append(dict(branch=branch,prefix='rest versus '+names[j],coarse_final_separation_mm=coarse_separation,fine_final_separation_mm=fine_separation))
            print('Half-step retention check '+branch,flush=True)
        csv_rows(output/'step_checks.csv',checks)
        if any(digest(p)!=h for p,h in frozen.items()):raise ValueError('Source changed.')
        finish(output,manifest)
    except BaseException as error:fail(output,manifest,error);raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',default='results/cx-sugar-reliable-return-training-seed0-20261113')
    p.add_argument('--retention',default='results/cx-standard-displacement-retention-20261117')
    p.add_argument('--output',required=True)
    return analyze(p.parse_args(argv))


if __name__=='__main__':main()
