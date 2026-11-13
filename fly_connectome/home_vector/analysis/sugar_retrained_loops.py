"""Replay archived long-loop protocols with the selected jointly trained models."""

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

from ..__main__ import begin, fail, finish
from ..data import Circuit, digest, encode, sample_protocol, verify_bundle, write_json
from ..model import HomeVectorNetwork
from ..sugar_training import SugarTrainingNetwork
from .decoder_contributions import write_csv
from .sugar_response import SugarResponse, event_gate


PRIMARY=('original_trained','sugar_trained_single','sugar_trained_repeated')
COLORS=('tab:blue','tab:orange','tab:green')


def load_selected_models(training):
    """Use independently selected best checkpoints, never final-update weights."""
    training=Path(training)
    manifest=verify_bundle(training)
    if manifest['kind']!='home_vector_sugar_paired_training_v1':
        raise ValueError('Require a complete paired recurrent-training bundle.')
    if manifest['settings'].get('smoke') or manifest['settings'].get('benchmark'):
        raise ValueError('Require scientific training, not an engineering check.')
    config=json.loads((training/'training_config.json').read_text())
    comparison=json.loads((training/'comparison.json').read_text())
    history=json.loads((training/'history.json').read_text())
    circuit=Circuit(**json.loads((training/'circuit.json').read_text()))
    response=SugarResponse(**config['response'])
    selection_key=config.get('selection_score_key','vector_rmse_mm')
    if selection_key not in ('vector_rmse_mm','selection_score'):raise ValueError('Unknown checkpoint-selection score.')
    models,selection={},{}
    for name in ('original','sugar'):
        path=training/(name+'_best.pt')
        checkpoint=torch.load(path,weights_only=False,map_location='cpu')
        chosen=min((r for r in history if name+'_validation' in r),
                   key=lambda r:r[name+'_validation'][selection_key])
        if (checkpoint['update']!=comparison[name]['selected_update']
                or checkpoint['update']!=chosen['update']
                or checkpoint['validation']!=chosen[name+'_validation']
                or checkpoint['validation']!=comparison[name]['validation']):
            raise ValueError('Selected checkpoint disagrees with validation history.')
        net=SugarTrainingNetwork(circuit,response,seed=manifest['settings']['seed'])
        net.load_state_dict(checkpoint['model'],strict=True);net.eval()
        models[name]=net
        selection[name]=dict(update=checkpoint['update'],checkpoint_sha256=digest(path))
    return models,circuit,response,config,selection


def loop_metrics(path,condition,prediction,rates,target,n,loops,dynamic_types):
    rows=[]
    for loop in range(1,loops+1):
        start,end=(loop-1)*n,loop*n
        e=prediction[start:end]-target[start:end];mean=e.mean(0)
        closure=prediction[end]-target[end]
        row=dict(path=path,condition=condition,loop=loop,
            vector_rmse_mm=float(np.sqrt(np.mean(np.sum(e*e,axis=1)))),
            zero_prediction_rmse_mm=float(np.sqrt(np.mean(np.sum(target[start:end]**2,axis=1)))),
            closure_error_mm=float(np.linalg.norm(closure)),
            closure_x_error_mm=float(closure[0]),closure_y_error_mm=float(closure[1]),
            mean_x_error_mm=float(mean[0]),mean_y_error_mm=float(mean[1]),
            centered_error_rms_mm=float(np.sqrt(np.mean(np.sum((e-mean)**2,axis=1)))),
            max_activity=float(rates[start:end+1].max()),
            endpoint_activity_norm=float(np.linalg.norm(rates[end])))
        for population in dict.fromkeys(dynamic_types):
            select=np.array(dynamic_types)==population
            row[population+'_mean_activity']=float(rates[start:end][:,select].mean())
        rows.append(row)
    return rows


def component_plot(output,name,time,target,predictions,n,loops):
    a,b=(loops-1)*n,loops*n;phase=time[a:b+1]-time[a]
    fig,axes=plt.subplots(4,3,figsize=(15,12),sharex=True,layout='constrained')
    for col,(condition,color) in enumerate(zip(PRIMARY,COLORS)):
        p=predictions[condition][a:b+1];y=target[a:b+1]
        for component in (0,1):
            axes[component,col].plot(phase,p[:,component],color=color,label='Predicted')
            axes[component,col].plot(phase,y[:,component],'k--',label='Correct')
            axes[component,col].set_ylabel(('Home x' if component==0 else 'Home y')+' (mm)')
            axes[component+2,col].plot(phase,p[:,component]-y[:,component],color=color)
            axes[component+2,col].axhline(0,color='k',ls='--',lw=.7)
            axes[component+2,col].set_ylabel(('x' if component==0 else 'y')+' error (mm)')
        axes[0,col].set_title(condition);axes[0,col].legend(fontsize=8)
        axes[-1,col].set_xlabel('Time within final loop (s)')
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle(f'{name}, loop {loops}, {time[a]:g}–{time[b]:g} s: jointly trained selected models\n'
        'Raw predictions; signed error = predicted - correct; panel scales vary')
    for ext in ('png','pdf'):fig.savefig(output/f'{name}_loop_{loops:02d}_components.{ext}',dpi=140)
    plt.close(fig)


def analyze(args):
    training,reference,output=[Path(getattr(args,k)).resolve() for k in ('training','reference','output')]
    models,circuit,response,config,selection=load_selected_models(training)
    rm=verify_bundle(reference)
    if rm['kind']!='home_vector_sugar_long_loops_v1':raise ValueError('Require archived frozen long-loop comparison.')
    data=Path(rm['settings']['data']).resolve();dm=verify_bundle(data)
    tm=json.loads((training/'manifest.json').read_text())
    if Path(tm['settings']['data']).resolve()!=data:
        raise ValueError('Training and reference must use the same derived dataset.')
    source=Path(json.loads((data/'task.json').read_text())['source'])
    if circuit.as_dict()!=json.loads((source/'circuit.json').read_text()):
        raise ValueError('Training changed the archived anatomy.')
    if response.__dict__!=json.loads((reference/'response.json').read_text()):
        raise ValueError('Response constants differ from the reference experiment.')
    originals=dict(dm['immutable_input_sha256']);originals.update(tm['immutable_input_sha256'])
    folders={training,reference,data,*[Path(k).parent for k in originals]}
    for folder in folders:
        if output==folder or output.is_relative_to(folder) or folder.is_relative_to(output):
            raise ValueError('Output must be separate from every original archive.')
    if output.exists():raise FileExistsError(output)
    frozen={str(p):digest(p) for folder in (training,reference,data) for p in folder.iterdir() if p.is_file()}
    frozen.update(originals)
    if any(digest(p)!=h for p,h in frozen.items()):raise ValueError('An archived input changed.')
    protocols=json.loads((reference/'protocols.json').read_text())
    old_rows=list(csv.DictReader((reference/'loop_metrics.csv').open()))
    loops=rm['settings']['loops'];dt=config['dt_s'];torch.set_num_threads(1)
    before={name:{k:v.clone() for k,v in net.state_dict().items()} for name,net in models.items()}
    output,manifest=begin(output,'home_vector_sugar_retrained_loops_v1',training=str(training),reference=str(reference),loops=loops)
    manifest.update(immutable_input_sha256=frozen,selection=selection,refitting=False,recurrent_training=False,
                    response_parameter_fitting=False,torch=torch.__version__,device='cpu')
    write_json(output/'manifest.json',manifest)
    try:
        for path in (Path(__file__),Path(__file__).parents[1]/'sugar_training.py',
                     Path(__file__).parents[1]/'model.py',Path(__file__).with_name('sugar_response.py')):
            shutil.copyfile(path,output/path.name)
        shutil.copyfile(training/'circuit.json',output/'circuit.json')
        for name in models:shutil.copyfile(training/(name+'_best.pt'),output/(name+'_selected.pt'))
        write_json(output/'response.json',response.__dict__);write_json(output/'protocols.json',protocols)
        rows,comparisons,checks=[],[],[]
        comparison_fig,comparison_axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
        for number,comparison_ax in zip((8,9,14,15),comparison_axes.flat):
            name=f'path_{number:02d}';record=protocols[name]
            controls,target=sample_protocol(record['protocol'],dt)
            time=np.arange(len(controls))*dt;n=round(record['period_s']/dt)
            if len(controls)!=loops*n+1:raise ValueError('Archived duration does not match loop count.')
            np.testing.assert_allclose(target[np.arange(1,loops+1)*n],0.,atol=5e-6,rtol=0)
            with np.load(reference/(name+'.npz')) as old:
                np.testing.assert_array_equal(time,old['time_s']);np.testing.assert_array_equal(target,old['target'])
            pfn=torch.tensor(encode(circuit,controls),dtype=torch.float32)[None]
            events=record['events_s']
            gates=np.stack([event_gate(time,events[k],response) for k in ('original','sugar_single','sugar_repeated')])
            saved=dict(time_s=time,target=target,gates=gates);predictions={};rates={}
            with torch.no_grad():
                p,r=models['original'](pfn,dt,torch.zeros_like(torch.tensor(gates[:1],dtype=pfn.dtype)))
                predictions['original_trained']=p[0].numpy();rates['original_trained']=r[0].numpy()
                p,r=models['sugar'](pfn.repeat(3,1,1),dt,torch.tensor(gates,dtype=pfn.dtype))
                for i,condition in enumerate(('sugar_trained_no_sugar','sugar_trained_single','sugar_trained_repeated')):
                    predictions[condition]=p[i].numpy();rates[condition]=r[i].numpy()
                # The two selected networks need not agree before sugar. Each
                # must independently match native zero-gate dynamics there.
                for model,conditions in (('original',('original_trained',)),
                        ('sugar',('sugar_trained_no_sugar','sugar_trained_single','sugar_trained_repeated'))):
                    native=HomeVectorNetwork(circuit);native.load_state_dict(models[model].state_dict())
                    expected_p,expected_r=native(pfn[:,:n+1],dt)
                    for condition in conditions:
                        np.testing.assert_allclose(rates[condition][:n+1],expected_r[0].numpy(),atol=1e-4,rtol=2e-5)
                        np.testing.assert_allclose(predictions[condition][:n+1],expected_p[0].numpy(),atol=1e-4,rtol=2e-5)
            checks.append(dict(path=name,first_loop_native_match=True,targets_match_reference=True))
            fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
            for condition,p in predictions.items():
                r=rates[condition]
                if not np.isfinite(r).all() or not np.isfinite(p).all() or (r<0).any() or r.max()>1e6:
                    raise ValueError('Replay exceeded numerical guard.')
                saved[condition+'_prediction']=p;saved[condition+'_rates']=r
                current=loop_metrics(name,condition,p,r,target,n,loops,circuit.dynamic_types);rows.extend(current)
                error=np.linalg.norm(p-target,axis=1)
                style=dict(ls='--',alpha=.65) if condition=='sugar_trained_no_sugar' else {}
                color=dict(zip(PRIMARY,COLORS)).get(condition,'tab:purple')
                axes[0,0].plot(time,error,label=condition,color=color,**style)
                axes[0,1].plot(np.arange(1,loops+1),[x['closure_error_mm'] for x in current],label=condition,color=color,**style)
                axes[1,0].plot(time,r.max(1),label=condition,color=color,**style)
                axes[1,1].plot(time,r[:,np.array(circuit.dynamic_types)=='hDeltaG'].mean(1),label=condition,color=color,**style)
                if condition in PRIMARY:
                    previous={'original_trained':'original','sugar_trained_single':'sugar_single','sugar_trained_repeated':'sugar_repeated'}[condition]
                    old=[x for x in old_rows if x['path']==name and x['condition']==previous and x['readout']=='refit']
                    comparison_ax.plot(np.arange(1,loops+1),[float(x['closure_error_mm']) for x in old],color=color,ls='--',alpha=.7,label=previous+' / old circuit + refit')
                    comparison_ax.plot(np.arange(1,loops+1),[x['closure_error_mm'] for x in current],color=color,label=condition+' / joint training')
                    for x,o in zip(current,old):comparisons.append(dict(path=name,condition=condition,loop=x['loop'],
                        previous_vector_rmse_mm=float(o['vector_rmse_mm']),new_vector_rmse_mm=x['vector_rmse_mm'],
                        previous_closure_error_mm=float(o['closure_error_mm']),new_closure_error_mm=x['closure_error_mm']))
            titles=('Vector error (mm)','Closure error (mm)','Largest single-neuron activity','hDeltaG mean activity')
            for ax,title in zip(axes.flat,titles):
                ax.set(title=title,xlabel='Loop number' if ax is axes[0,1] else 'Time (s)');ax.grid(alpha=.2);ax.legend(fontsize=7)
            fig.suptitle(name+': validation-selected jointly trained models; no refit or state reset')
            for ext in ('png','pdf'):fig.savefig(output/(name+'.'+ext),dpi=140)
            plt.close(fig)
            component_plot(output,name,time,target,predictions,n,loops)
            np.savez_compressed(output/(name+'.npz'),**saved)
            comparison_ax.set(title=name,xlabel='Loop number',ylabel='Closure error (mm)',yscale='log')
            comparison_ax.grid(alpha=.2);comparison_ax.legend(fontsize=6)
            print(f'Completed {name}: selected jointly trained models, {loops} loops',flush=True)
        comparison_fig.suptitle('Original frozen circuit + refitted readouts (dashed) versus jointly trained selected models (solid)')
        comparison_fig.savefig(output/'previous_comparison.png',dpi=150);plt.close(comparison_fig)
        write_csv(output/'loop_metrics.csv',rows);write_csv(output/'previous_comparison.csv',comparisons)
        write_json(output/'verification.json',checks)
        lines=['# Extended loops after joint recurrent/readout training','',
            f"Selected updates: original {selection['original']['update']}, sugar {selection['sugar']['update']}. "
            'Every condition uses its own network’s selected decoder. No new fitting or parameter selection occurs. '
            'Inputs, target anchor and explicit delivery schedules match the archived 16-loop experiment. '
            'Sugar is delivered once at 8 s or every 8 s after nonfinal returns. '
            'The sugar-trained network with its gate disabled is an additional control.','',
            '| Path | Condition | Loop | Loop vector RMSE (mm) | Closure error (mm) | Zero-output RMSE (mm) | Max activity |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
        for r in rows:
            if r['loop'] in (2,4,8,16):lines.append(f"| {r['path']} | {r['condition']} | {r['loop']} | {r['vector_rmse_mm']:.5f} | {r['closure_error_mm']:.5f} | {r['zero_prediction_rmse_mm']:.5f} | {r['max_activity']:.3f} |")
        lines+=['','These familiar paths are diagnostic extrapolation. The new networks were trained with '
            'at most two return loops and a single sugar event; long straight examples were also present. '
            'Within-loop error excludes its final zero-duration state; closure error scores that state separately. '
            'Near-zero closure does not establish useful tracking: compare the zero-vector baseline. '
            'Old versus new comparisons change recurrent training, readouts and training objective; they do not '
            'isolate the effect of one parameter or establish a physiological reset mechanism. '
            'Current forward activity units are not calibrated firing rates.','',
            '![Previous comparison](previous_comparison.png)','']
        for number in (8,9,14,15):lines.extend([f'## Path {number}','',f'![Full duration](path_{number:02d}.png)','',f'![Components](path_{number:02d}_loop_{loops:02d}_components.png)',''])
        (output/'report.md').write_text('\n'.join(lines))
        (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Jointly trained loop evaluation</title>'
            '<style>body{font:16px system-ui;max-width:1400px;margin:2em auto}img{max-width:100%}</style>'
            '<h1>Jointly trained models: extended-loop evaluation</h1><a href="report.md">Methods and metrics</a>'
            '<img src="previous_comparison.png">'+''.join(f'<h2>Path {n}</h2><img src="path_{n:02d}.png">'
                f'<img src="path_{n:02d}_loop_{loops:02d}_components.png">' for n in (8,9,14,15)))
        for name,net in models.items():
            for key,value in net.state_dict().items():torch.testing.assert_close(value,before[name][key],atol=0,rtol=0)
        if any(digest(p)!=h for p,h in frozen.items()):raise ValueError('An archived source changed during replay.')
        finish(output,manifest)
    except BaseException as error:fail(output,manifest,error);raise
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('training','reference','output'):p.add_argument('--'+name,required=True)
    return analyze(p.parse_args(argv))


if __name__=='__main__':main()
