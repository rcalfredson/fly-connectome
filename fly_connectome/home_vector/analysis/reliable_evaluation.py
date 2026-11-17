"""Frozen evaluation of reliably cued training under the saved evaluation plan."""

import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..__main__ import begin,fail,finish
from ..data import digest,encode,sample_protocol,verify_bundle,write_json
from ..model import HomeVectorNetwork
from ..diverse_returns import sampling_weights,stratum
from ..reliable_sugar import contact_events,validate as validate_reliable
from .sugar_retrained_loops import load_selected_models
from .sugar_response import event_gate


CONDITIONS=('original','sugar','sugar_gate_disabled','zero')
COLORS=('tab:blue','tab:green','tab:purple','gray')


def csv_rows(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def load_metric_rows(path):
    """Restore scoring metadata so completion groups cases exactly like a fresh run."""
    text_keys={'split','id','family','stratum','route_class','traversal','condition','event_kind'}
    integer_keys={'return_count','contacts','event_number','excursion'}
    with Path(path).open(newline='') as f:
        rows=list(csv.DictReader(f))
    for row in rows:
        for key,value in row.items():
            if value=='':row[key]=None
            elif key in text_keys:continue
            elif key=='end_away':
                if value not in ('True','False'):raise ValueError('Invalid end-away metadata.')
                row[key]=value=='True'
            elif key in integer_keys:
                number=float(value)
                if not np.isfinite(number) or not number.is_integer():
                    raise ValueError('Invalid integer scoring metadata.')
                row[key]=int(number)
            else:row[key]=float(value)
    return rows


def centers(row,time,target):
    values=row.get('return_times_s',[])
    if not values and row.get('group')=='returns':values=[row['loop_period_s'],float(time[-1])]
    if not values and row.get('group')=='diagnostic':values=[8.,float(time[-1])]
    indices=np.rint(np.asarray(values)/(time[1]-time[0])).astype(int)
    if len(indices):np.testing.assert_allclose(target[indices],0.,atol=5e-6,rtol=0)
    return indices


def validate_completed_scores(sets,dt,rows,events,excursions):
    """Reject partial CSV snapshots after interruption between file writes."""
    expected_rows,expected_events,expected_excursions=set(),set(),set()
    for split,records in sets.items():
        for record in records:
            controls,target=sample_protocol(record['protocol'],dt)
            time=np.arange(len(controls))*dt
            center_indices=centers(record,time,target)
            boundaries=list(dict.fromkeys([0,*center_indices.tolist(),len(time)-1]))
            count=len(boundaries)-1 if len(center_indices) else 0
            for condition in CONDITIONS:
                key=(split,record['id'],condition)
                expected_rows.add(key)
                for kind,number in (('patch_entry',len(record['sugar_events_s'])),
                                    ('center_return',len(center_indices))):
                    expected_events.update((*key,kind,i) for i in range(1,number+1))
                expected_excursions.update((*key,i) for i in range(1,count+1))
    for name,actual,expected,extra in (
            ('trajectory',rows,expected_rows,()),
            ('event',events,expected_events,('event_kind','event_number')),
            ('excursion',excursions,expected_excursions,('excursion',))):
        keys=[tuple(row[k] for k in ('split','id','condition',*extra)) for row in actual]
        if len(keys)!=len(expected) or set(keys)!=expected:
            raise ValueError(f'Interrupted evaluation has incomplete {name} scores.')


def describe(prediction,target,rates,controls,time,row,energy,types):
    """Uniform physical-time scores; endpoint and event states are separate."""
    error=np.asarray(prediction,float)-target;e=error[:-1];squared=np.sum(e*e,axis=1)
    t=time[:-1];moving=np.linalg.norm(controls[:-1,1:],axis=1)>1e-8
    events=row['sugar_events_s'];early=np.zeros(len(t),bool)
    for event in events:early|=(t>event)&(t<event+2.)
    masks=dict(moving=moving,dwell=~moving,early_post_contact=early,
        later_post_contact=(t>=events[0])&~early if events else np.zeros(len(t),bool),
        before_contact=t<=events[0] if events else np.ones(len(t),bool))
    mean=e.mean(0)
    result=dict(mse_mm2=float(squared.mean()),normalized_mse=float(squared.mean()/energy),
        target_mse_mm2=float(np.mean(np.sum(target[:-1]**2,axis=1))),
        endpoint_sq_mm2=float(np.sum(error[-1]**2)),mean_x_error_mm=float(mean[0]),mean_y_error_mm=float(mean[1]),
        centered_mse_mm2=float(np.mean(np.sum((e-mean)**2,axis=1))),
        max_activity=None if rates is None else float(rates.max()),
        endpoint_state_norm=None if rates is None else float(np.linalg.norm(rates[-1])))
    for key,select in masks.items():result[key+'_mse_mm2']=float(squared[select].mean()) if select.any() else None
    for population in dict.fromkeys(types):
        select=np.array(types)==population
        result[population+'_mean_activity']=None if rates is None else float(rates[:-1,select].mean())
    event_rows=[]
    for kind,indices in (('patch_entry',np.rint(np.array(events)/(time[1]-time[0])).astype(int)),('center_return',centers(row,time,target))):
        for j,index in enumerate(indices,1):
            item=dict(event_kind=kind,event_number=j,time_s=float(time[index]),
                error_mm=float(np.linalg.norm(error[index])),target_norm_mm=float(np.linalg.norm(target[index])),
                state_norm=None if rates is None else float(np.linalg.norm(rates[index])),
                hDeltaG_mean_activity=None if rates is None else float(rates[index,np.array(types)=='hDeltaG'].mean()))
            event_rows.append(item)
    excursion_rows=[]
    center_indices=centers(row,time,target)
    boundaries=list(dict.fromkeys([0,*center_indices.tolist(),len(time)-1]))
    if len(center_indices):
        for j,(a,b) in enumerate(zip(boundaries,boundaries[1:]),1):
            if b<=a:continue
            x=error[a:b]
            excursion_rows.append(dict(excursion=j,start_s=float(time[a]),end_s=float(time[b]),
                mse_mm2=float(np.mean(np.sum(x*x,axis=1))),end_error_mm=float(np.linalg.norm(error[b])),
                target_mse_mm2=float(np.mean(np.sum(target[a:b]**2,axis=1))),
                max_activity=None if rates is None else float(rates[a:b+1].max()),
                hDeltaG_mean_activity=None if rates is None else float(rates[a:b,np.array(types)=='hDeltaG'].mean())))
    return result,event_rows,excursion_rows


def aggregate(rows):
    """Renormalize fixed base weights among members of each subset/phase."""
    output=[]
    for split in dict.fromkeys(r['split'] for r in rows):
        for condition in CONDITIONS:
            members=[r for r in rows if r['split']==split and r['condition']==condition]
            if not members:continue
            labels={'overall':members}
            for key in ('stratum','route_class','return_count','traversal','end_away'):
                for value in dict.fromkeys(r.get(key) for r in members if r.get(key) is not None):
                    labels[key+':'+str(value)]=[r for r in members if r.get(key)==value]
            for label,selected in labels.items():
                item=dict(split=split,condition=condition,subset=label,count=len(selected),
                    families=len({r['family'] for r in selected}))
                keys=[k for k in selected[0] if k.endswith('_mse_mm2') or k in ('mse_mm2','normalized_mse','endpoint_sq_mm2','mean_x_error_mm','mean_y_error_mm','centered_mse_mm2','endpoint_state_norm') or k.endswith('_mean_activity')]
                for key in keys:
                    valid=[r for r in selected if r.get(key) is not None]
                    value=sum(r['weight']*r[key] for r in valid)/sum(r['weight'] for r in valid) if valid else None
                    if key=='mse_mm2':dest='vector_rmse_mm'
                    elif key=='endpoint_sq_mm2':dest='endpoint_rms_mm'
                    elif key.endswith('_mse_mm2'):dest=key.replace('_mse_mm2','_rms_mm')
                    else:dest=key
                    item[dest]=None if value is None else (float(np.sqrt(value)) if dest!=key else float(value))
                peaks=[r['max_activity'] for r in selected if r['max_activity'] is not None]
                item['max_activity']=max(peaks) if peaks else None;output.append(item)
    return output


def replay(models,pfn,gates,dt):
    with torch.no_grad():
        po,ro=models['original'](pfn,dt,torch.zeros_like(gates))
        p,r=models['sugar'](pfn.repeat(2,1,1),dt,torch.cat((gates,torch.zeros_like(gates))))
    batch=len(pfn)
    output={'original':(po,ro),'sugar':(p[:batch],r[:batch]),'sugar_gate_disabled':(p[batch:],r[batch:])}
    for prediction,rates in output.values():
        if not torch.isfinite(prediction).all() or not torch.isfinite(rates).all() or (rates<0).any() or rates.max()>1e6:
            raise ValueError('Replay exceeded numerical guard; do not silently drop divergent cases.')
    return {key:(p.numpy(),r.numpy()) for key,(p,r) in output.items()}


def plot_case(output,stem,cache,types):
    time,target=cache['time_s'],cache['target'];predictions=cache['predictions'];rates=cache['rates']
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for condition,color in zip(CONDITIONS,COLORS):
        p=predictions[condition];error=np.linalg.norm(p-target,axis=1)
        style=dict(ls='--',alpha=.65) if condition in ('zero','sugar_gate_disabled') else {}
        axes[0,0].plot(time,p[:,0],color=color,label=condition,**style)
        axes[0,1].plot(time,p[:,1],color=color,label=condition,**style)
        axes[0,2].plot(time,error,color=color,label=condition,**style)
        if condition!='zero':
            r=rates[condition]
            axes[1,1].plot(time,r[:,np.array(types)=='hDeltaG'].mean(1),color=color,label=condition,**style)
            axes[1,2].plot(time,r.max(1),color=color,label=condition,**style)
            c=cache['centers'];axes[1,0].plot(np.arange(1,len(c)+1),error[c],marker='.',color=color,label=condition,**style)
    for j in (0,1):axes[0,j].plot(time,target[:,j],'k:',lw=2,label='Correct')
    titles=('Home x (mm)','Home y (mm)','Vector error (mm)','Geometric-return error (mm)','hDeltaG mean activity','Largest neuron activity')
    for ax,title in zip(axes.flat,titles):ax.set(title=title);ax.grid(alpha=.2);ax.legend(fontsize=7)
    for ax in axes.flat:ax.set_xlabel('Return number' if ax is axes[1,0] else 'Time (s)')
    fig.suptitle(stem+': selected fixed models, raw predictions and reliable contact schedule')
    fig.savefig(output/(stem+'.png'),dpi=140);plt.close(fig)
    start=cache['centers'][-2] if len(cache['centers'])>1 else 0
    fig,axes=plt.subplots(4,3,figsize=(14,11),sharex=True,layout='constrained')
    for col,(condition,color) in enumerate(zip(CONDITIONS[:3],COLORS[:3])):
        p=predictions[condition][start:];y=target[start:];t=time[start:]-time[start]
        for j in (0,1):
            axes[j,col].plot(t,p[:,j],color=color);axes[j,col].plot(t,y[:,j],'k--')
            axes[j+2,col].plot(t,p[:,j]-y[:,j],color=color);axes[j+2,col].axhline(0,color='k',ls='--',lw=.7)
            axes[j,col].set_ylabel(('x' if j==0 else 'y')+' (mm)')
            axes[j+2,col].set_ylabel(('x' if j==0 else 'y')+' error (mm)')
        axes[0,col].set_title(condition);axes[-1,col].set_xlabel('Time since penultimate center return (s)')
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle(stem+': final excursion and suffix; predicted minus correct; panel scales vary')
    fig.savefig(output/(stem+'_components.png'),dpi=140);plt.close(fig)


def analyze(args):
    plan_path=Path(args.plan).resolve();plan=json.loads(plan_path.read_text())
    training,data,output=Path(plan['training']).resolve(),Path(plan['data']).resolve(),Path(args.output).resolve()
    dm=verify_bundle(data)
    if digest(data/'manifest.json')!=plan['data_manifest_sha256']:raise ValueError('Dataset differs from fixed evaluation plan.')
    models,circuit,response,config,selection=load_selected_models(training)
    tm=verify_bundle(training)
    if Path(tm['settings']['data']).resolve()!=data:raise ValueError('Training dataset differs from evaluation data.')
    groups=json.loads((data/'trajectories.json').read_text());dt=config['dt_s'];validate_reliable(groups,dt)
    normalization=json.loads((data/'loss_plan.json').read_text())
    if config['normalization']!=normalization:raise ValueError('Training normalization differs from sealed scales.')
    frozen=dict(dm['immutable_input_sha256']);frozen.update(tm['immutable_input_sha256'])
    for folder in (data,training):frozen.update({str(p):digest(p) for p in folder.iterdir() if p.is_file()})
    frozen[str(plan_path)]=digest(plan_path)
    reference=Path('results/cx-sugar-long-loops-20261112').resolve();verify_bundle(reference)
    frozen.update({str(p):digest(p) for p in reference.iterdir() if p.is_file()})
    for folder in {Path(p).parent for p in frozen}:
        if output==folder or output.is_relative_to(folder) or folder.is_relative_to(output):raise ValueError('Output overlaps an immutable source.')
    if output.exists() and not args.complete_existing:raise FileExistsError(output)
    if args.batch_size<1:raise ValueError('Require a positive batch size.')
    if any(digest(p)!=h for p,h in frozen.items()):raise ValueError('Archived input hash mismatch.')
    torch.set_num_threads(1)
    before={name:{k:v.clone() for k,v in net.state_dict().items()} for name,net in models.items()}
    sets={s:groups[s] for s in plan['sets']}
    # Additional training-fit diagnostic explicitly separated from held-out results.
    sets['training_fit_diverse']=[r for r in groups['train'] if r['stratum']=='diverse_return']
    circles=json.loads((reference/'protocols.json').read_text());sets['historical_circles']=[]
    for name,record in circles.items():
        row=dict(record,id=name,family=name,stratum='diverse_return',route_class='historical_circle',
            traversal='forward' if name in ('path_08','path_14') else 'rightward',return_count=16,
            return_times_s=[8.*j for j in range(1,17)],end_away=False,sugar_events_s=[])
        row['sugar_events_s']=contact_events(row,dt);sets['historical_circles'].append(row)
    wanted={x[k] for x in plan['representatives'] for k in ('test_id','extrapolation_id')}
    wanted.update(r['id'] for r in sets['historical_circles'])
    if args.complete_existing:
        manifest=json.loads((output/'manifest.json').read_text())
        if manifest['status'] not in ('running','failed') or manifest['immutable_input_sha256']!=frozen:
            raise ValueError('Completion requires an interrupted matching evaluation.')
        manifest.setdefault('completion_attempts',[]).append(dict(status=manifest['status'],error=manifest.pop('error',None)))
        manifest['status']='running'
        manifest['completion_source_sha256']=digest(__file__)
    else:
        output,manifest=begin(output,'home_vector_reliable_evaluation_v1',training=str(training),data=str(data),plan_sha256=digest(plan_path),batch_size=args.batch_size)
    manifest.update(immutable_input_sha256=frozen,selection=selection,refitting=False,recurrent_training=False,device='cpu',torch=torch.__version__)
    write_json(output/'manifest.json',manifest)
    try:
        shutil.copyfile(__file__,output/('completion_evaluator.py' if args.complete_existing else 'reliable_evaluation.py'));shutil.copyfile(plan_path,output/'evaluation_plan.json')
        for name in models:shutil.copyfile(training/(name+'_best.pt'),output/(name+'_selected.pt'))
        write_json(output/'additional_diagnostics.json',dict(training_fit_diverse='All 96 diverse training paths; post-hoc fit diagnostic, no selection or fitting.',
            historical_circles='Archived motion with current reliable contact detection; event schedules differ from historical plots.'))
        rows,events,excursions,cache,verification=[],[],[],{},[]
        if args.complete_existing:
            rows=load_metric_rows(output/'trajectory_metrics.csv')
            events=load_metric_rows(output/'event_metrics.csv')
            excursions=load_metric_rows(output/'excursion_metrics.csv')
            validate_completed_scores(sets,dt,rows,events,excursions)
            for records in sets.values():
                for record in records:
                    if record['id'] not in wanted:continue
                    controls,target=sample_protocol(record['protocol'],dt);time=np.arange(len(controls))*dt
                    gate=event_gate(time,record['sugar_events_s'],response)
                    arrays=replay(models,torch.tensor(encode(circuit,controls)[None],dtype=torch.float32),torch.tensor(gate[None],dtype=torch.float32),dt)
                    cache[record['id']]=dict(time_s=time,target=target,controls=controls,gate=gate,record=record,
                        centers=centers(record,time,target),predictions={**{c:v[0][0] for c,v in arrays.items()},'zero':np.zeros_like(target)},
                        rates={c:v[1][0] for c,v in arrays.items()})
            verification.append(dict(completion='All trajectory scores present; representative replays regenerated after interruption.'))
        for split,records in ([] if args.complete_existing else sets.items()):
            weights=sampling_weights(records);order=sorted(range(len(records)),key=lambda i:records[i]['protocol'][-1][0])
            for start in range(0,len(order),args.batch_size):
                indices=order[start:start+args.batch_size];selected=[records[i] for i in indices]
                samples=[sample_protocol(r['protocol'],dt) for r in selected];length=max(len(c) for c,y in samples)
                x=np.zeros((len(selected),length,len(circuit.input_ids)),np.float32);gates=np.zeros((len(selected),length),np.float32)
                for j,(record,(controls,target)) in enumerate(zip(selected,samples)):
                    n=len(controls);x[j,:n]=encode(circuit,controls);gates[j,:n]=event_gate(np.arange(n)*dt,record['sugar_events_s'],response)
                output_arrays=replay(models,torch.tensor(x),torch.tensor(gates),dt)
                if start==0:
                    with torch.no_grad():
                        for name,condition in (('original','original'),('sugar','sugar_gate_disabled')):
                            native=HomeVectorNetwork(circuit);native.load_state_dict(models[name].state_dict())
                            p,r=native(torch.tensor(x[:,:128]),dt)
                            np.testing.assert_allclose(p.numpy(),output_arrays[condition][0][:,:128],atol=1e-4,rtol=2e-5)
                            np.testing.assert_allclose(r.numpy(),output_arrays[condition][1][:,:128],atol=1e-4,rtol=2e-5)
                    verification.append(dict(split=split,disabled_gate_native_prefix=True))
                for j,(record,(controls,target),index) in enumerate(zip(selected,samples,indices)):
                    n=len(controls);time=np.arange(n)*dt;energy=normalization['target_energy_mm2'][stratum(record)]
                    meta=dict(split=split,id=record['id'],family=record['family'],stratum=stratum(record),
                        route_class=record.get('route_class'),return_count=record.get('return_count',2 if record.get('group')=='returns' else None),
                        traversal=record.get('traversal'),end_away=record.get('end_away'),weight=float(weights[index]),
                        duration_s=float(time[-1]),contacts=len(record['sugar_events_s']))
                    item=dict(time_s=time,target=target,controls=controls,gate=gates[j,:n],record=record,
                              predictions={},rates={},centers=centers(record,time,target))
                    for condition in CONDITIONS:
                        p,r=(np.zeros_like(target),None) if condition=='zero' else (output_arrays[condition][0][j,:n],output_arrays[condition][1][j,:n])
                        scores,er,xr=describe(p,target,r,controls,time,record,energy,circuit.dynamic_types)
                        rows.append(dict(meta,condition=condition,**scores))
                        events.extend(dict(meta,condition=condition,**v) for v in er)
                        excursions.extend(dict(meta,condition=condition,**v) for v in xr)
                        if record['id'] in wanted:
                            item['predictions'][condition]=p.copy()
                            if r is not None:item['rates'][condition]=r.copy()
                    if record['id'] in wanted:cache[record['id']]=item
                del output_arrays
                if start%(args.batch_size*4)==0:print(f'{split}: {min(start+args.batch_size,len(records))}/{len(records)}',flush=True)
            csv_rows(output/'trajectory_metrics.csv',rows);csv_rows(output/'event_metrics.csv',events);csv_rows(output/'excursion_metrics.csv',excursions)
        summaries=aggregate(rows);csv_rows(output/'summary.csv',summaries)
        # Paired family distributions; every family stays a single analysis unit.
        family_rows=[]
        for split in sets:
            for family in dict.fromkeys(r['family'] for r in rows if r['split']==split):
                members=[r for r in rows if r['split']==split and r['family']==family]
                result=dict(split=split,family=family)
                for condition in CONDITIONS:
                    chosen=[r for r in members if r['condition']==condition]
                    result[condition+'_rmse_mm']=float(np.sqrt(sum(r['weight']*r['mse_mm2'] for r in chosen)/sum(r['weight'] for r in chosen)))
                result['sugar_minus_original_rmse_mm']=result['sugar_rmse_mm']-result['original_rmse_mm']
                family_rows.append(result)
        csv_rows(output/'family_metrics.csv',family_rows)
        fine_checks=[]
        for representative in plan['representatives']:
            for split,key in (('test','test_id'),('extrapolation','extrapolation_id')):
                name=representative[key];item=cache[name];stem=representative['route_class']+'_'+split
                plot_case(output,stem,item,circuit.dynamic_types)
                saved=dict(time_s=item['time_s'],target=item['target'],controls=item['controls'],gate=item['gate'])
                saved.update({c+'_prediction':p for c,p in item['predictions'].items()})
                saved.update({c+'_rates':r for c,r in item['rates'].items()})
                np.savez_compressed(output/(stem+'.npz'),**saved)
                if split=='extrapolation':
                    controls,target=sample_protocol(item['record']['protocol'],dt/2);time=np.arange(len(controls))*dt/2
                    x=torch.tensor(encode(circuit,controls)[None],dtype=torch.float32)
                    g=torch.tensor(event_gate(time,item['record']['sugar_events_s'],response)[None],dtype=torch.float32)
                    fine=replay(models,x,g,dt/2)
                    for c,(p,r) in fine.items():
                        coarse=item['predictions'][c];difference=p[0,::2].astype(float)-coarse
                        fine_checks.append(dict(id=name,route_class=representative['route_class'],condition=c,
                            state_step_difference_rms=float(np.sqrt(np.mean((r[0,::2][:-1].astype(float)-item['rates'][c][:-1])**2))),
                            prediction_step_difference_rms_mm=float(np.sqrt(np.mean(np.sum(difference[:-1]**2,axis=1)))),
                            coarse_rmse_mm=float(np.sqrt(np.mean(np.sum((coarse[:-1]-item['target'][:-1])**2,axis=1)))),
                            fine_rmse_mm=float(np.sqrt(np.mean(np.sum((p[0,::2][:-1]-item['target'][:-1])**2,axis=1))))))
                    print('Half-step checked '+representative['route_class'],flush=True)
        for record in sets['historical_circles']:
            item=cache[record['id']];plot_case(output,'historical_'+record['id'],item,circuit.dynamic_types)
        csv_rows(output/'step_checks.csv',fine_checks);write_json(output/'verification.json',verification)
        report=dict(selection=selection,summaries=summaries,step_checks=fine_checks)
        write_json(output/'evaluation.json',report)
        lines=['# Reliable home-sugar evaluation','',
            'Frozen validation-selected models with their trained decoders. No fitting or parameter changes. '
            'Normalized scores use sealed training energies; raw RMSE uses the same fixed family weights. '
            'The zero baseline is scored on identical samples.','',
            '| Set | Subset | Condition | Normalized MSE | Vector RMSE (mm) | Endpoint RMS (mm) |',
            '| --- | --- | --- | ---: | ---: | ---: |']
        for r in summaries:
            if r['subset']=='overall' or (r['split']=='test' and r['subset'].startswith('stratum:')):
                lines.append(f"| {r['split']} | {r['subset']} | {r['condition']} | {r['normalized_mse']:.5f} | {r['vector_rmse_mm']:.5f} | {r['endpoint_rms_mm']:.5f} |")
        lines+=['','Trajectory, event, excursion and family CSVs retain all cases. '
            'Patch entry is distinct from geometric center return. Gate-disabled replay is a same-weights '
            'out-of-training intervention. Family relatives are not independent replicates. '
            'Training-fit probes are additional diagnostics, not held-out evidence. '
            'The historical circles use the current contact policy, not old scheduled pulses. '
            'One seed and finite training budget do not separate architecture, optimization, '
            'objective and physiological-model limitations.','']
        for representative in plan['representatives']:
            for split in ('test','extrapolation'):
                stem=representative['route_class']+'_'+split
                lines.extend([f'## {stem}','',f'![Full trajectory]({stem}.png)','',f'![Components]({stem}_components.png)',''])
        for record in sets['historical_circles']:lines.extend([f'![Historical circle](historical_{record["id"]}.png)',''])
        (output/'report.md').write_text('\n'.join(lines))
        for name,net in models.items():
            for k,v in net.state_dict().items():torch.testing.assert_close(v,before[name][k],atol=0,rtol=0)
        if any(digest(p)!=h for p,h in frozen.items()):raise ValueError('An archived input changed during evaluation.')
        finish(output,manifest)
    except BaseException as error:fail(output,manifest,error);raise
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',default='docs/data/cx_home_vector_reliable_evaluation_plan.json')
    p.add_argument('--output',required=True);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--complete-existing',action='store_true',help='Finish interrupted plotting/checks only after all trajectory scores were saved.')
    return analyze(p.parse_args(argv))


if __name__=='__main__':main()
