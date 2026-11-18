"""Describe persistent circle error and probe memory with common continuations."""

import argparse
import itertools
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
from .loop_phases import PATHS
from .reliable_evaluation import csv_rows
from .sugar_retrained_loops import load_selected_models


def waveform_error(prediction,target,dt):
    """Descriptive orthogonal decomposition, never a corrected prediction."""
    p,y=np.asarray(prediction,float),np.asarray(target,float)
    if p.shape!=y.shape or p.ndim!=2 or p.shape[1]!=2 or len(p)<4:
        raise ValueError('Require matching complete-cycle x/y samples, excluding endpoint.')
    theta=2*np.pi*np.arange(len(p))/len(p)
    basis=np.column_stack((np.ones(len(p)),np.cos(theta),np.sin(theta)))
    cp=np.linalg.lstsq(basis,p,rcond=None)[0];cy=np.linalg.lstsq(basis,y,rcond=None)[0]
    error=p-y;mean=error.mean(0);harmonic=basis[:,1:]@(cp-cy)[1:]
    residual=error-mean-harmonic
    total=float(np.mean(np.sum(error**2,axis=1)))
    result=dict(vector_rmse_mm=np.sqrt(total),mean_x_error_mm=mean[0],mean_y_error_mm=mean[1],
        offset_mse_mm2=float(mean@mean),harmonic_mse_mm2=float(np.mean(np.sum(harmonic**2,axis=1))),
        residual_mse_mm2=float(np.mean(np.sum(residual**2,axis=1))))
    np.testing.assert_allclose(sum(result[k] for k in ('offset_mse_mm2','harmonic_mse_mm2','residual_mse_mm2')),total,atol=1e-10,rtol=1e-10)
    for j,axis in enumerate(('x','y')):
        ap=float(np.linalg.norm(cp[1:,j]));ay=float(np.linalg.norm(cy[1:,j]))
        lag=(np.arctan2(cp[2,j],cp[1,j])-np.arctan2(cy[2,j],cy[1,j])+np.pi)%(2*np.pi)-np.pi
        result[axis+'_amplitude_ratio']=ap/ay if ay>1e-10 else None
        result[axis+'_peak_lag_s']=float(lag*len(p)*dt/(2*np.pi)) if ap>1e-10 and ay>1e-10 else None
    return result,basis@cp,basis@cy


def pair_metrics(prediction_a,prediction_b,rates_a,rates_b,target_difference):
    """Signed separation gain avoids mistaking a rotated vector for retained memory."""
    dp=np.asarray(prediction_b,float)-prediction_a;dr=np.asarray(rates_b,float)-rates_a
    dy=np.asarray(target_difference,float);length=float(np.linalg.norm(dy))
    output=dict(prediction_separation_mm=np.linalg.norm(dp,axis=-1),
        separation_error_mm=np.linalg.norm(dp-dy,axis=-1),state_separation=np.linalg.norm(dr,axis=-1))
    if length>1e-8:
        output['signed_target_gain']=dp@dy/(length**2)
        output['orthogonal_separation_mm']=np.linalg.norm(dp-output['signed_target_gain'][...,None]*dy,axis=-1)
    return output


def prefixes():
    rows={}
    for name,vf,vr in [('rest',0.,0.),('forward_2mm',.5,0.),('forward_4mm',1.,0.),('right_2mm',0.,.5),('right_4mm',0.,1.)]:
        rows[name]=[[0.,45.,vf,vr],[4.,45.,0.,0.]]
    rows['closed_square']=[[0.,45.,1.,0.],[1.,135.,1.,0.],[2.,225.,1.,0.],[3.,315.,1.,0.],[4.,45.,0.,0.]]
    return rows


def analyze(args):
    training,phases,reference,output=[Path(getattr(args,k)).resolve() for k in ('training','phases','reference','output')]
    manifests=[verify_bundle(p) for p in (training,phases,reference)]
    models,circuit,response,config,selection=load_selected_models(training)
    if manifests[1]['selection']!=selection['original']:raise ValueError('Phase checkpoint differs.')
    frozen={}
    for m in manifests:frozen.update(m['immutable_input_sha256'])
    for folder in (training,phases,reference):frozen.update({str(p):digest(p) for p in folder.iterdir() if p.is_file()})
    for source in frozen:
        folder=Path(source).parent
        if output==folder or output.is_relative_to(folder) or folder.is_relative_to(output):raise ValueError('Output overlaps archive.')
    if any(digest(p)!=h for p,h in frozen.items()):raise ValueError('Source changed.')
    dt=config['dt_s'];steps=round(8/dt);loops=16
    protocols=prefixes();circles=json.loads((reference/'protocols.json').read_text())
    branches={'pause_32s':[[0.,45.,0.,0.],[32.,45.,0.,0.]],
              'straight_32s':[[0.,45.,1.,0.],[32.,45.,0.,0.]]}
    branches.update({name:circles[name]['protocol'] for name in PATHS})
    net=HomeVectorNetwork(circuit);net.load_state_dict(models['original'].state_dict());net.eval();torch.set_num_threads(1)
    before={k:v.clone() for k,v in net.state_dict().items()}
    output,manifest=begin(output,'home_vector_displacement_retention_v1',training=str(training),phases=str(phases),reference=str(reference))
    manifest.update(immutable_input_sha256=frozen,selection=selection['original'],training=False,refitting=False)
    write_json(output/'manifest.json',manifest)
    try:
        shutil.copyfile(__file__,output/'displacement_retention.py')
        write_json(output/'design.json',dict(prefix_protocols=protocols,continuations=branches,dt_s=dt,
            design='Equal-duration four-second physically reachable prefixes; six common continuations; fixed original home anchor.',
            selected_pairs=[['rest','forward_4mm'],['rest','right_4mm'],['rest','closed_square']],
            interpretation='Exploratory probes, not new independent held-out training families. No biological time constant is fitted.'))
        errors=[];fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        for ax,(name,label) in zip(axes.flat,PATHS.items()):
            with np.load(phases/(name+'.npz')) as a:
                p,y=a['prediction'],a['target'];time=np.arange(steps)*dt
                for k in range(loops):
                    score,fp,fy=waveform_error(p[k*steps:(k+1)*steps],y[k*steps:(k+1)*steps],dt)
                    errors.append(dict(path=name,loop=k+1,**score))
                last=errors[-1];e=p[-steps-1:-1]-y[-steps-1:-1]
                ax.plot(time,e[:,0],label='x error');ax.plot(time,e[:,1],label='y error')
                ax.axhline(last['mean_x_error_mm'],ls='--',color='C0',alpha=.7)
                ax.axhline(last['mean_y_error_mm'],ls='--',color='C1',alpha=.7)
                ax.axhline(0,color='k',lw=.7);ax.set(title=name+': '+label,xlabel='Phase within loop 16 (s)',ylabel='Predicted minus correct (mm)');ax.legend();ax.grid(alpha=.2)
        fig.suptitle('Persistent error: raw frozen predictions; dashed lines show mean component errors')
        fig.savefig(output/'circle_error_components.png',dpi=150);plt.close(fig);csv_rows(output/'waveform_metrics.csv',errors)
        last=[r for r in errors if r['loop']==16]
        fig,ax=plt.subplots(figsize=(9,5),layout='constrained');bottom=np.zeros(4)
        for key,label in [('offset_mse_mm2','Mean vector offset'),('harmonic_mse_mm2','Cyclic amplitude/timing/mixing'),('residual_mse_mm2','Remaining waveform error')]:
            values=np.array([r[key] for r in last]);ax.bar(list(PATHS),values,bottom=bottom,label=label);bottom+=values
        ax.set(ylabel='Vector mean squared error (mm²)',title='Loop 16 error budget; descriptive decomposition, no prediction correction');ax.legend();fig.savefig(output/'circle_error_budget.png',dpi=150);plt.close(fig)
        names=list(protocols);samples=[sample_protocol(v,dt) for v in protocols.values()]
        xp=torch.tensor(np.array([encode(circuit,c) for c,y in samples]),dtype=torch.float32)
        with torch.no_grad():pp,rp=net(xp,dt)
        initial=rp[:,-1].clone();prefix_target=np.array([y[-1] for c,y in samples]);prefix_prediction=pp[:,-1].numpy()
        np.testing.assert_allclose(prefix_target[[0,5]],0,atol=1e-10)
        np.savez_compressed(output/'prefixes.npz',rates=rp.numpy(),prediction=pp.numpy(),target=np.array([y for c,y in samples]),names=np.array(names),controls=np.array([c for c,y in samples]))
        summaries=[];traces=[];prefix_rows=[]
        for i,name in enumerate(names):prefix_rows.append(dict(prefix=name,true_x_mm=prefix_target[i,0],true_y_mm=prefix_target[i,1],predicted_x_mm=prefix_prediction[i,0],predicted_y_mm=prefix_prediction[i,1],initial_error_mm=float(np.linalg.norm(prefix_prediction[i]-prefix_target[i]))))
        csv_rows(output/'prefix_metrics.csv',prefix_rows)
        for branch,protocol in branches.items():
            controls,displacement=sample_protocol(protocol,dt);time=np.arange(len(controls))*dt
            encoded=encode(circuit,controls).astype(np.float32)
            x=torch.tensor(np.repeat(encoded[None],len(names),axis=0))
            with torch.no_grad():p,r=net(x,dt,initial=initial)
            p,r=p.numpy(),r.numpy();target=prefix_target[:,None,:]+displacement[None]
            np.testing.assert_allclose(r[:,0],initial.numpy(),atol=0,rtol=0)
            np.testing.assert_allclose(p[:,0],prefix_prediction,atol=2e-6,rtol=2e-6)
            np.savez_compressed(output/(branch+'.npz'),time_s=time,controls=controls,pfn=encoded,prediction=p,rates=r,target=target,prefix_names=np.array(names))
            fig,axes=plt.subplots(3,3,figsize=(14,10),layout='constrained')
            for i,j in itertools.combinations(range(len(names)),2):
                dy=prefix_target[j]-prefix_target[i];metrics=pair_metrics(p[i],p[j],r[i],r[j],dy)
                np.testing.assert_allclose(target[j]-target[i],np.broadcast_to(dy,target[i].shape),atol=1e-10,rtol=0)
                for t in (0.,1.,2.,4.,8.,16.,32.,64.,128.):
                    k=round(t/dt)
                    if k>=len(time):continue
                    record=dict(branch=branch,prefix_a=names[i],prefix_b=names[j],time_s=t,true_separation_mm=float(np.linalg.norm(dy)))
                    record.update({key:float(v[k]) for key,v in metrics.items()});summaries.append(record)
                for k in np.unique(np.r_[np.arange(0,len(time),round(.1/dt)),len(time)-1]):
                    traces.append(dict(branch=branch,prefix_a=names[i],prefix_b=names[j],time_s=float(time[k]),**{key:float(v[k]) for key,v in metrics.items()}))
            for col,other in enumerate(('forward_4mm','right_4mm','closed_square')):
                j=names.index(other);dy=prefix_target[j]-prefix_target[0];metrics=pair_metrics(p[0],p[j],r[0],r[j],dy)
                axes[0,col].plot(time,metrics['prediction_separation_mm'],label='Decoded separation')
                axes[0,col].axhline(np.linalg.norm(dy),ls='--',color='k',label='Required separation')
                axes[0,col].set(title='rest versus '+other,ylabel='mm');axes[0,col].legend(fontsize=8)
                axes[1,col].plot(time,metrics['state_separation']);axes[1,col].set(ylabel='Complete-state separation (model units)')
                for j0 in (0,j):axes[2,col].plot(time,np.linalg.norm(p[j0]-target[j0],axis=1),label=names[j0])
                axes[2,col].set(ylabel='Tracking error (mm)',xlabel='Time since common input began (s)');axes[2,col].legend(fontsize=8)
            for ax in axes.flat:ax.grid(alpha=.2)
            fig.suptitle(branch+': same input after different reachable histories; original home anchor retained')
            fig.savefig(output/(branch+'_retention.png'),dpi=150);plt.close(fig)
            print('Completed common continuation '+branch,flush=True)
        csv_rows(output/'pair_metrics.csv',summaries);csv_rows(output/'pair_traces.csv',traces)
        write_json(output/'verification.json',dict(physically_reachable_prefixes=True,equal_prefix_duration_s=4.,identical_suffix_inputs=True,constant_correct_pair_separation=True,original_home_anchor=True,no_state_or_decoder_fit=True))
        for k,v in net.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)
        if any(digest(p)!=h for p,h in frozen.items()):raise ValueError('Source changed.')
        lines=['# Persistent error and common-input retention','','Frozen standard checkpoint, update 2000. Descriptive waveform fitting changes no network prediction. Retention starts from physically reached states; home stays at the original origin. A loop returns to the continuation start, not necessarily to home. No sugar dynamics or fitted new decoder.','','## Final-loop error','', '| Path | x offset, mm | y offset, mm | Offset share of squared error | x amplitude ratio | y amplitude ratio | x peak lag, s | y peak lag, s |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for row in last:lines.append(f"| {row['path']} | {row['mean_x_error_mm']:.4f} | {row['mean_y_error_mm']:.4f} | {row['offset_mse_mm2']/row['vector_rmse_mm']**2:.1%} | {row['x_amplitude_ratio']:.3f} | {row['y_amplitude_ratio']:.3f} | {row['x_peak_lag_s']:.3f} | {row['y_peak_lag_s']:.3f} |")
        lines+=['','![Error budget](circle_error_budget.png)','','![Signed errors](circle_error_components.png)','','## Common-input probes','','Pair metrics include all 15 prefix pairs. Signed target gain is decoded separation projected onto the required vector, divided by its squared length; ideal gain=1. For equal-home-vector controls this gain is undefined and omitted. A small initial gain must not be mistaken for forgetting that occurs only later. All results are exploratory within one selected checkpoint; no asymptotic, biological or architecture-wide impossibility is established.']
        for branch in branches:lines+=['',f'![{branch}]({branch}_retention.png)']
        (output/'report.md').write_text('\n'.join(lines)+'\n');finish(output,manifest)
    except BaseException as error:fail(output,manifest,error);raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',default='results/cx-sugar-reliable-return-training-seed0-20261113')
    p.add_argument('--phases',default='results/cx-standard-loop-phases-20261116')
    p.add_argument('--reference',default='results/cx-sugar-long-loops-20261112')
    p.add_argument('--output',required=True)
    return analyze(p.parse_args(argv))


if __name__=='__main__':main()
