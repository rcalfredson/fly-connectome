"""Compare matching phases of repeated circles in a frozen standard network."""

import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..__main__ import begin, fail, finish
from ..data import digest, encode, sample_protocol, verify_bundle, write_json
from ..model import HomeVectorNetwork
from .reliable_evaluation import csv_rows
from .sugar_retrained_loops import load_selected_models


PATHS = {'path_08': 'counterclockwise, forward', 'path_09': 'counterclockwise, rightward',
         'path_14': 'clockwise, forward', 'path_15': 'clockwise, rightward'}


def aligned_metrics(rates, prediction, target, steps, loops, types, ids):
    """Use half-open loop intervals; score each closing state separately.

    State difference RMS = sqrt(phase_mean(sum_neuron(delta**2))).
    Per-neuron and population summaries retain actual activity, not projections.
    """
    if (rates.shape != (steps * loops + 1, len(types))
            or prediction.shape != (steps * loops + 1, 2)
            or target.shape != prediction.shape or len(ids) != len(types)):
        raise ValueError('Arrays must contain complete loops and one terminal state.')
    if not all(np.isfinite(x).all() for x in (rates, prediction, target)):
        raise ValueError('Require finite saved states and outputs.')
    r = rates[:-1].astype(float).reshape(loops, steps, -1)
    p = prediction[:-1].astype(float).reshape(loops, steps, 2)
    y = target[:-1].astype(float).reshape(loops, steps, 2)
    np.testing.assert_allclose(y, np.broadcast_to(y[0], y.shape), atol=5e-6, rtol=0)
    vector_rms = lambda x: float(np.sqrt(np.mean(np.sum(x*x, axis=-1))))
    rows, populations, neurons = [], [], []
    type_array = np.asarray(types)
    for k in range(loops):
        dr = None if k == 0 else r[k] - r[k-1]
        dp = None if k == 0 else p[k] - p[k-1]
        row = dict(loop=k+1, state_rms=vector_rms(r[k]),
            state_vs_first_rms=vector_rms(r[k]-r[0]),
            state_vs_final_rms=vector_rms(r[k]-r[-1]),
            adjacent_state_rms=None if dr is None else vector_rms(dr),
            adjacent_state_max_phase_norm=None if dr is None else float(np.linalg.norm(dr, axis=1).max()),
            adjacent_prediction_rms_mm=None if dp is None else vector_rms(dp),
            vector_rmse_mm=vector_rms(p[k]-y[k]),
            max_activity=float(r[k].max()),
            closure_state_norm=float(np.linalg.norm(rates[(k+1)*steps])),
            closure_error_mm=float(np.linalg.norm(prediction[(k+1)*steps]-target[(k+1)*steps])))
        rows.append(row)
        for population in dict.fromkeys(types):
            mask = type_array == population
            populations.append(dict(loop=k+1, population=population,
                mean_activity=float(r[k,:,mask].mean()), max_activity=float(r[k,:,mask].max()),
                adjacent_state_rms=None if dr is None else vector_rms(dr[:,mask]),
                adjacent_entry_rms=None if dr is None else float(np.sqrt(np.mean(dr[:,mask]**2)))))
        for i, (neuron, population) in enumerate(zip(ids, types)):
            neurons.append(dict(loop=k+1, neuron_id=neuron, population=population,
                mean_activity=float(r[k,:,i].mean()), max_activity=float(r[k,:,i].max()),
                adjacent_change_rms=None if dr is None else float(np.sqrt(np.mean(dr[:,i]**2))),
                adjacent_max_abs_change=None if dr is None else float(np.abs(dr[:,i]).max())))
    return rows, populations, neurons


def plots(output, name, label, rates, prediction, target, steps, loops, dt, types, rows):
    phase = np.arange(steps)*dt
    r = rates[:-1].reshape(loops, steps, -1)
    p = prediction[:-1].reshape(loops, steps, 2)
    y = target[:steps]
    colors = plt.colormaps['viridis'](np.linspace(0,1,loops))
    fig, axes = plt.subplots(3,2,figsize=(13,11),layout='constrained')
    for k, color in enumerate(colors):
        for j in (0,1):
            axes[0,j].plot(phase,p[k,:,j],color=color,lw=.8,alpha=.8)
        axes[1,0].plot(phase,np.linalg.norm(r[k],axis=1),color=color,lw=.8)
    for j in (0,1):
        axes[0,j].plot(phase,y[:,j],'k--',lw=1.7,label='Correct')
        axes[0,j].set(title='Home '+('x' if j==0 else 'y'),ylabel='mm')
        axes[0,j].legend()
    axes[1,0].set(title='Whole-population activity norm',ylabel='Model activity units')
    for k in (1,3,7,15):
        if k < loops:
            difference=np.linalg.norm(r[k].astype(float)-r[k-1],axis=1)
            axes[1,1].plot(phase,difference,label=f'{k+1} minus {k}')
    axes[1,1].set(title='State difference at matching phases',ylabel='Euclidean norm, activity units')
    axes[1,1].legend()
    indices=np.arange(1,loops+1)
    adjacent=np.array([np.nan if x['adjacent_state_rms'] is None else x['adjacent_state_rms'] for x in rows])
    axes[2,0].plot(indices,adjacent,'o-',label='Successive-loop state difference')
    axes[2,0].set(title='Phase RMS of full-state difference',ylabel='Model activity units',xlabel='Later loop')
    axes[2,1].plot(indices,[x['vector_rmse_mm'] for x in rows],'o-',label='Tracking RMSE')
    axes[2,1].plot(indices,[x['closure_error_mm'] for x in rows],'s--',label='Closure error')
    axes[2,1].plot(indices,[np.nan if x['adjacent_prediction_rms_mm'] is None else x['adjacent_prediction_rms_mm'] for x in rows],'.-',label='Successive-loop prediction difference')
    axes[2,1].set(title='Accuracy versus repeatability',ylabel='mm',xlabel='Loop')
    axes[2,1].legend(fontsize=8)
    for ax in axes[:2].flat:ax.set_xlabel('Movement phase within loop (s)')
    for ax in axes.flat:ax.grid(alpha=.2)
    colorbar=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(1,loops),cmap='viridis'),ax=list(axes[0])+[axes[1,0]],shrink=.6)
    colorbar.set_label('Loop number')
    fig.suptitle(f'{name}: {label}; frozen standard network, continuous state\nRaw states and predictions; no recentering or per-loop reset')
    fig.savefig(output/f'{name}_phase_comparison.png',dpi=150);plt.close(fig)
    populations=list(dict.fromkeys(types));chosen=(0,7,15)
    fig,axes=plt.subplots(len(populations),3,figsize=(14,12),layout='constrained',squeeze=False)
    for a,population in enumerate(populations):
        mask=np.asarray(types)==population;vmax=float(r[:,:,mask].max())
        for b,k in enumerate(chosen):
            im=axes[a,b].imshow(r[k][:,mask].T,aspect='auto',origin='lower',extent=(0,steps*dt,0,mask.sum()),vmin=0,vmax=vmax,cmap='magma')
            axes[a,b].set(title=f'{population}, loop {k+1}',ylabel='Neuron index within type',xlabel='Movement phase (s)')
        fig.colorbar(im,ax=list(axes[a]),shrink=.7,label='Actual activity (model units)')
    fig.suptitle(f'{name}: actual activity of every neuron; fixed anatomical order\nCommon scale across loops within each population; scales differ between populations')
    fig.savefig(output/f'{name}_phase_activity.png',dpi=150);plt.close(fig)


def analyze(args):
    training=Path(args.training).resolve();reference=Path(args.reference).resolve();output=Path(args.output).resolve()
    tm=verify_bundle(training);rm=verify_bundle(reference)
    models,circuit,response,config,selection=load_selected_models(training)
    frozen=dict(tm['immutable_input_sha256']);frozen.update(rm['immutable_input_sha256'])
    for directory in (training,reference):
        frozen.update({str(p):digest(p) for p in directory.iterdir() if p.is_file()})
    for source in frozen:
        folder=Path(source).parent
        if output==folder or output.is_relative_to(folder) or folder.is_relative_to(output):
            raise ValueError('Output overlaps an immutable source.')
    if any(digest(p)!=h for p,h in frozen.items()):raise ValueError('Source changed.')
    dt=config['dt_s'];protocols=json.loads((reference/'protocols.json').read_text());loops=16
    samples=[sample_protocol(protocols[name]['protocol'],dt) for name in PATHS]
    steps=round(8/dt);period=steps*dt
    for controls,target in samples:
        if len(controls)!=loops*steps+1:raise ValueError('Require sixteen exact 8-second loops.')
        tiled=np.broadcast_to(controls[:steps],(loops,steps,3))
        np.testing.assert_allclose(controls[:-1].reshape(loops,steps,3),tiled,atol=1e-10,rtol=0)
        np.testing.assert_allclose(target[::steps],0.,atol=5e-6,rtol=0)
    x=np.array([encode(circuit,c) for c,y in samples],dtype=np.float32)
    np.testing.assert_array_equal(x[:,:-1].reshape(4,loops,steps,-1),np.broadcast_to(x[:,:steps,None,:].transpose(0,2,1,3),(4,loops,steps,x.shape[-1])))
    net=HomeVectorNetwork(circuit);net.load_state_dict(models['original'].state_dict());net.eval()
    before={k:v.clone() for k,v in net.state_dict().items()};torch.set_num_threads(1)
    output,manifest=begin(output,'home_vector_standard_loop_phases_v1',training=str(training),reference=str(reference),source_manifest_sha256=digest(reference/'manifest.json'),source=str(reference))
    manifest.update(immutable_input_sha256=frozen,selection=selection['original'],condition='standard_original',refitting=False,training=False)
    write_json(output/'manifest.json',manifest)
    try:
        shutil.copyfile(__file__,output/'loop_phases.py');shutil.copyfile(training/'original_best.pt',output/'original_selected.pt')
        with torch.no_grad():predictions,states=net(torch.tensor(x),dt)
        predictions=predictions.numpy();states=states.numpy()
        rows,populations,neurons,summary=[],[],[],[]
        for i,(name,label) in enumerate(PATHS.items()):
            controls,target=samples[i];r=states[i];p=predictions[i]
            records,pr,nr=aligned_metrics(r,p,target,steps,loops,circuit.dynamic_types,circuit.dynamic_ids)
            rows.extend(dict(path=name,**v) for v in records)
            populations.extend(dict(path=name,**v) for v in pr);neurons.extend(dict(path=name,**v) for v in nr)
            np.savez_compressed(output/f'{name}.npz',time_s=np.arange(len(r))*dt,phase_s=np.arange(steps)*dt,controls=controls,pfn=x[i],target=target,prediction=p,rates=r,neuron_ids=np.array(circuit.dynamic_ids),neuron_types=np.array(circuit.dynamic_types))
            plots(output,name,label,r,p,target,steps,loops,dt,circuit.dynamic_types,records)
            summary.append(dict(path=name,configuration=label,early_state_difference_rms=records[1]['adjacent_state_rms'],late_state_difference_rms=records[-1]['adjacent_state_rms'],late_prediction_difference_rms_mm=records[-1]['adjacent_prediction_rms_mm'],late_tracking_rmse_mm=records[-1]['vector_rmse_mm'],late_closure_error_mm=records[-1]['closure_error_mm']))
            print(name,summary[-1],flush=True)
        csv_rows(output/'loop_metrics.csv',rows);csv_rows(output/'population_metrics.csv',populations);csv_rows(output/'neuron_metrics.csv',neurons);csv_rows(output/'summary.csv',summary)
        write_json(output/'verification.json',dict(identical_phase_controls=True,identical_phase_float32_pfn=True,matching_world_targets=True,dt_s=dt,period_s=period,loops=loops,half_open_phase_interval=True,terminal_states_scored_separately=True))
        for k,v in net.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)
        if any(digest(p)!=h for p,h in frozen.items()):raise ValueError('An archived source changed.')
        lines=['# Matching movement phases across loops: frozen standard network','','Every trial starts from zero once. No sugar dynamics or per-loop state reset. All 16 loops have identical sampled movement and float32 PFN input at matching phases. Differences therefore reflect evolving inherited neural state.','','State difference RMS = sqrt(phase_mean(sum_neuron((r_loop - r_previous)**2))). This is a whole-population Euclidean difference, not a per-neuron RMS. Each neuron is also reported separately. Phase averages use [0, 8 s); closing states are scored separately.','','| Path | Loop 2 minus 1 state RMS | Loop 16 minus 15 state RMS | Final prediction difference, mm RMS | Final tracking RMSE, mm |','| --- | ---: | ---: | ---: | ---: |']
        for v in summary:lines.append(f"| {v['path']} | {v['early_state_difference_rms']:.6g} | {v['late_state_difference_rms']:.6g} | {v['late_prediction_difference_rms_mm']:.6g} | {v['late_tracking_rmse_mm']:.6g} |")
        lines+=['','Small differences establish repeatability only over this finite replay at float32 precision. They do not prove asymptotic stability, an attractor, robustness, accurate integration, or a causal mechanism. Actual activity is not a decoder projection. No row/null split, drive attribution, perturbation, fitting or new training is performed here. Familiar circles are diagnostic motions, not fresh held-out evidence.']
        for name in PATHS:lines+=['',f'## {name}', '',f'![Phase comparison]({name}_phase_comparison.png)','',f'![All neuronal activities]({name}_phase_activity.png)']
        (output/'report.md').write_text('\n'.join(lines)+'\n');finish(output,manifest)
    except BaseException as error:fail(output,manifest,error);raise
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training',default='results/cx-sugar-reliable-return-training-seed0-20261113')
    parser.add_argument('--reference',default='results/cx-sugar-long-loops-20261112')
    parser.add_argument('--output',required=True)
    return analyze(parser.parse_args(argv))


if __name__=='__main__':main()
