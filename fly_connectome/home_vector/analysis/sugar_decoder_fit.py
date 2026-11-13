"""Fit readouts only, on a sealed derived sugar task; frozen recurrent parameters."""

import argparse
from collections import Counter
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
from .checkpoint import capture
from .decoder_contributions import CHECKPOINT_SHA256, write_csv
from .sugar_response import SugarResponse, event_gate, rollout
from .sugar_decoder_data import validate_splits


RIDGES=(1e-6,1e-4,1e-2,1.)


def family_weights(rows):
    """Half mass per existing/return domain, then equal families and siblings."""
    domains={r['group'] for r in rows}
    families={g:{r['family'] for r in rows if r['group']==g} for g in domains}
    counts=Counter((r['group'],r['family']) for r in rows)
    return np.array([1./len(domains)/len(families[r['group']])/counts[r['group'],r['family']] for r in rows])


def moments(rates,targets):
    x=np.column_stack((np.asarray(rates,dtype=float),np.ones(len(rates))))
    y=np.asarray(targets,dtype=float)
    return dict(G=x.T@x/len(x),C=x.T@y/len(x),Y=float(np.mean(np.sum(y*y,axis=1))))


def merge(store,key,s,w):
    if key not in store:store[key]=dict(G=np.zeros_like(s['G']),C=np.zeros_like(s['C']),Y=0.,weight=0.)
    a=store[key]
    a['G']+=w*s['G'];a['C']+=w*s['C'];a['Y']+=w*s['Y'];a['weight']+=w


def normalized(s):
    return {k:s[k]/s['weight'] for k in ('G','C','Y')}


def mse(s,weights):
    v=normalized(s) if 'weight' in s else s
    value=float(np.sum(weights*(v['G']@weights))-2*np.sum(weights*v['C'])+v['Y'])
    if value < -1e-7*max(1.,v['Y']):raise ValueError('Negative MSE beyond numerical cancellation tolerance.')
    return max(0.,value)


def fit_ridge(training,validation):
    """Training-only centering/scaling; validation-only regularization choice."""
    tr=normalized(training)
    mean=tr['G'][:-1,-1]
    scale=np.sqrt(np.maximum(np.diag(tr['G'])[:-1]-mean*mean,1e-12))
    width=len(mean)+1
    transform=np.eye(width)
    transform[:-1,:-1]=np.diag(1./scale)
    transform[-1,:-1]=-mean/scale
    gram=transform.T@tr['G']@transform
    cross=transform.T@tr['C']
    candidates=[]
    for ridge in RIDGES:
        penalty=np.diag([ridge]*(width-1)+[0.])
        raw=transform@np.linalg.solve(gram+penalty,cross)
        candidates.append((mse(validation,raw),ridge,raw))
    score,ridge,raw=min(candidates,key=lambda x:(x[0],x[1]))
    return dict(weights=raw,mean=mean,scale=scale,ridge=ridge,validation_mse=score,
                candidates=[dict(ridge=r,validation_mse=e) for e,r,_ in candidates])


def batched_sugar_rates(network,pfn,dt,gates,response,mask):
    """Native operation order at zero gate; independent gates per batch member."""
    if torch.is_grad_enabled():raise ValueError('Require no_grad for frozen replay.')
    weight=network.weights();drive=(pfn@weight[:network.input_size]).unbind(1)
    recurrent=weight[network.input_size:];tau=network.tau[network.type_index]
    alpha=-torch.expm1(-dt/tau)
    state=pfn.new_zeros(pfn.shape[0],network.size);states=[]
    gate=torch.as_tensor(gates,device=pfn.device,dtype=pfn.dtype)
    for tick in range(pfn.shape[1]):
        states.append(state)
        if tick+1<len(drive):
            target=torch.relu(network.bias[network.type_index]+drive[tick]+state@recurrent)
            native=state+alpha*(target-state)
            extra=gate[:,tick,None]*response.relaxation_rate_per_s
            if torch.any(extra>0):
                decay=1./tau+extra
                equilibrium=(target/tau+extra*response.reference_activity)/decay
                modified=state+(-torch.expm1(-dt*decay))*(equilibrium-state)
                state=torch.where(mask & (extra>0),modified,native)
            else:state=native
    rates=torch.stack(states,dim=1)
    if not torch.isfinite(rates).all() or (rates<0).any() or rates.max()>1e6:
        raise ValueError('Replay exceeded numerical guard.')
    return rates


def add_record(store,split,model,record,rates,target,time,w):
    # All durations are uniform. Exclude the final, zero-duration state.
    r,y,t=rates[:-1],target[:-1],time[:-1]
    whole=moments(r,y)
    labels=['overall','group:'+record['group'],'arm:'+record['arm']]
    if 'traversal' in record:labels.append('traversal:'+record['traversal'])
    for label in labels:merge(store,(split,model,label),whole,w)
    # Closure is a separate state measurement, not a zero-duration contribution
    # to a time average. Return families retain the original anchor throughout.
    merge(store,(split,model,'endpoint:all'),moments(rates[-1:],target[-1:]),w)
    if record['group']=='returns':
        merge(store,(split,model,'closure:final_return'),moments(rates[-1:],target[-1:]),w)
        first=int(np.argmin(abs(time-record['loop_period_s'])))
        np.testing.assert_allclose(target[first],0.,atol=5e-6,rtol=0)
        merge(store,(split,model,'closure:first_return'),moments(rates[first:first+1],target[first:first+1]),w)
    events=record['sugar_events_s']
    if not events:
        merge(store,(split,model,'phase:no_sugar'),whole,w)
    else:
        first=events[0]
        masks={'before_first_sugar':t<=first,'first_2s':(t>first)&(t<first+2),
               'later':t>=first+2}
        for name,select in masks.items():
            if select.any():merge(store,(split,model,'phase:'+name),moments(r[select],y[select]),w)


def plot_diagnostics(output,cache,readouts):
    summaries=[]
    for number in (8,9,14,15):
        name=f'path_{number:02d}'
        fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        for ax,arm in zip(axes.flat,('continuous_control','continuous_sugar','pause_control','pause_sugar')):
            item=cache[name+'_'+arm];time=item['time_s'];target=item['target']
            saved=dict(time_s=time,target=target,gate=item['gate'])
            for label,(model,weights) in readouts.items():
                prediction=np.column_stack((item[model],np.ones(len(time))))@weights
                error=np.linalg.norm(prediction-target,axis=1)
                ax.plot(time,error,label=label)
                summaries.append(dict(path=name,arm=arm,readout=label,
                    vector_rmse_mm=float(np.sqrt(np.mean(error[:-1]**2))),terminal_error_mm=float(error[-1])))
                saved[label]=prediction
            np.savez_compressed(output/(name+'_'+arm+'_predictions.npz'),**saved)
            ax.axvline(8.,color='gray',ls=':',lw=.7)
            ax.set(title=arm,xlabel='Time (s)',ylabel='Vector error (mm)');ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.suptitle(name+': diagnostic only; not used for fitting or ridge selection')
        for extension in ('png','pdf'):fig.savefig(output/(name+'.'+extension),dpi=140)
        plt.close(fig)
    return summaries


def analyze(args):
    data,output=Path(args.data).resolve(),Path(args.output).resolve()
    if output==data or output.is_relative_to(data) or data.is_relative_to(output):
        raise ValueError('Output must be separate from the derived dataset.')
    if output.exists():raise FileExistsError(output)
    if args.batch_size<1:raise ValueError('Require positive batch size.')
    data_manifest=verify_bundle(data)
    if data_manifest['kind']!='home_vector_sugar_decoder_data_v1':raise ValueError('Require the derived decoder dataset.')
    groups=json.loads((data/'trajectories.json').read_text());validate_splits(groups)
    task=json.loads((data/'task.json').read_text());dt=task['dt_s']
    for name in ('source','activity','diagnostics'):
        folder=Path(task[name]).resolve()
        if output==folder or output.is_relative_to(folder) or folder.is_relative_to(output):
            raise ValueError('Output must be separate from original source archives.')
    # Capture the unchanged original activity snapshot; verify copied identity too.
    snap=capture(Path(task['activity']),Path(task['source']))
    if snap['checkpoint_sha256']!=CHECKPOINT_SHA256 or digest(data/'current_checkpoint.pt')!=CHECKPOINT_SHA256:
        raise ValueError('Require the exact frozen checkpoint.')
    torch.set_num_threads(1)
    network,circuit=snap['network'],Circuit(**snap['circuit'])
    response=SugarResponse(**json.loads((data/'response_config.json').read_text())['response'])
    mask=torch.tensor(np.asarray(circuit.dynamic_types)=='hDeltaG')
    if mask.sum()!=8:raise ValueError('Require eight hDeltaG cells.')
    frozen=dict(data_manifest['immutable_input_sha256'])
    if any(digest(p)!=h for p,h in frozen.items()):raise ValueError('An original dataset source changed.')
    frozen.update({str(p):digest(p) for p in data.iterdir() if p.is_file()})
    state_before={k:v.clone() for k,v in network.state_dict().items()}
    output,manifest=begin(output,'home_vector_sugar_decoder_fit_v1',data=str(data),data_manifest_sha256=digest(data/'manifest.json'))
    manifest.update(immutable_input_sha256=frozen,checkpoint_sha256=CHECKPOINT_SHA256,
        recurrent_training=False,response_parameter_fitting=False,decoder_fitting=True,
        batch_size=args.batch_size,torch=torch.__version__,threads=1,
        analysis_code_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')})
    write_json(output/'manifest.json',manifest)
    try:
        for name in ('current_checkpoint.pt','current_metadata.json','neurons.csv','response_config.json','task.json'):
            shutil.copyfile(data/name,output/name)
        for name in ('sugar_decoder_fit.py','sugar_decoder_data.py','sugar_response.py'):
            shutil.copyfile(Path(__file__).with_name(name),output/name)
        store,cache,replay_checks={}, {}, []
        old_weights=np.vstack((network.D.detach().numpy().T,network.d.detach().numpy()))
        for split,records in groups.items():
            weights=family_weights(records)
            # Sorting only for padded CPU replay efficiency, never changes weights/splits.
            order=sorted(range(len(records)),key=lambda j:records[j]['protocol'][-1][0])
            for batch_index,start in enumerate(range(0,len(order),args.batch_size)):
                indices=order[start:start+args.batch_size]
                rows=[records[j] for j in indices]
                samples=[sample_protocol(r['protocol'],dt) for r in rows]
                lengths=[len(c) for c,_ in samples];length=max(lengths)
                inputs=np.zeros((len(rows),length,network.input_size),dtype=np.float32)
                gates=np.zeros((len(rows),length))
                for i,(r,(controls,_)) in enumerate(zip(rows,samples)):
                    inputs[i,:lengths[i]]=encode(circuit,controls)
                    gates[i,:lengths[i]]=event_gate(np.arange(lengths[i])*dt,r['sugar_events_s'],response)
                with torch.no_grad():
                    pfn=torch.as_tensor(inputs)
                    _,native=network(pfn,dt)
                    if not torch.isfinite(native).all() or (native<0).any() or native.max()>1e6:
                        raise ValueError('Native replay exceeded numerical guard.')
                    modified=batched_sugar_rates(network,pfn,dt,gates,response,mask) if gates.any() else native
                    # Batch kernel versus existing isolated probe; check one treated
                    # and one untreated example per split, with the same padded batch inputs.
                    if batch_index==0:
                        disabled=batched_sugar_rates(network,pfn,dt,np.zeros_like(gates),response,mask)
                        torch.testing.assert_close(disabled,native,atol=0,rtol=0)
                for i,(record,(_,target),n,j) in enumerate(zip(rows,samples,lengths,indices)):
                    arrays={'original':native[i,:n].numpy(),'sugar':modified[i,:n].numpy()}
                    time=np.arange(n)*dt
                    for model,rate in arrays.items():add_record(store,split,model,record,rate,target,time,weights[j])
                    if record['sugar_events_s'] and not any(c['split']==split and c['treated'] for c in replay_checks):
                        with torch.no_grad():
                            _,single=rollout(network,torch.as_tensor(inputs[i:i+1,:n]),dt,gates[i,:n],response,mask)
                        error=float(np.max(np.abs(single[0].numpy()-arrays['sugar'])))
                        np.testing.assert_allclose(single[0].numpy(),arrays['sugar'],atol=2e-3,rtol=2e-5)
                        replay_checks.append(dict(split=split,treated=True,batch_vs_single_max_abs_activity=error))
                    if split=='diagnostic':cache[record['id']]=dict(**arrays,time_s=time,target=target,gate=gates[i,:n])
                del native,modified,pfn
                if batch_index%16==0:print(f'Replay {split}: {min(start+args.batch_size,len(order))}/{len(order)}',flush=True)
        fits={model:fit_ridge(store['train',model,'overall'],store['validation',model,'overall']) for model in ('original','sugar')}
        readouts={'original_frozen':('original',old_weights),
                  'original_refit':('original',fits['original']['weights']),
                  'sugar_frozen':('sugar',old_weights),'sugar_refit':('sugar',fits['sugar']['weights'])}
        metrics=[]
        for (split,model,label),s in store.items():
            for name,(which,weight) in readouts.items():
                if which==model:metrics.append(dict(split=split,readout=name,subset=label,
                    family_weight_mass=float(s['weight']),vector_rmse_mm=float(np.sqrt(mse(s,weight)))))
        for model,fit in fits.items():
            np.savez_compressed(output/(model+'_decoder.npz'),decoder=fit['weights'][:-1].T,
                intercept=fit['weights'][-1],training_mean=fit['mean'],training_scale=fit['scale'],ridge=fit['ridge'],
                body_ids=circuit.dynamic_ids)
        # Preserve every aggregate moment used for fitting/scoring, not just rounded metrics.
        keys=list(store)
        np.savez_compressed(output/'sufficient_statistics.npz',G=np.stack([store[k]['G'] for k in keys]),
            C=np.stack([store[k]['C'] for k in keys]),Y=np.array([store[k]['Y'] for k in keys]),
            weight=np.array([store[k]['weight'] for k in keys]))
        write_json(output/'statistics_keys.json',[list(k) for k in keys])
        write_csv(output/'metrics.csv',metrics)
        diagnostic=plot_diagnostics(output,cache,readouts)
        write_csv(output/'diagnostic_metrics.csv',diagnostic)
        report=dict(checkpoint_sha256=CHECKPOINT_SHA256,
            fit_parameters={k:{key:v for key,v in fit.items() if key in ('ridge','validation_mse','candidates')} for k,fit in fits.items()},
            replay_checks=replay_checks,
            selection='Train-only feature scaling and ridge fit; validation-only ridge choice. Test/challenge/diagnostic never select parameters.',
            weights='50/50 existing and return domains; equal family then equal sibling; uniform time within each trajectory excluding endpoint.',
            phase_weights='Same family weights renormalized among eligible trajectories; uniform time within each phase.',
            phases='Before first delivered sugar; first 2 s after; later. Descriptive windows, not physiological stages.',
            response='Fixed uncalibrated working approximation; no joint training.',
            metrics=metrics)
        write_json(output/'comparison.json',report)
        lines=['# Frozen dynamics, separately fitted decoders','',report['selection'],'',report['weights'],'',
            'Four readouts are retained: original frozen/refitted and sugar frozen/refitted. '
            'Only the two new linear decoders/intercepts are fitted. The response rule is fixed; '
            'this is not physiological validation. All rates enter in the original 65-cell order.','',
            '| Split | Readout | Balanced vector RMSE (mm) |','| --- | --- | ---: |']
        for r in metrics:
            if r['subset']=='overall':lines.append(f"| {r['split']} | {r['readout']} | {r['vector_rmse_mm']:.5f} |")
        lines+=['','## Fresh test return-family closure','',
                '| Readout | First return RMS error (mm) | Final return RMS error (mm) |',
                '| --- | ---: | ---: |']
        for label in readouts:
            a=next(r['vector_rmse_mm'] for r in metrics if r['split']=='test' and r['readout']==label and r['subset']=='closure:first_return')
            b=next(r['vector_rmse_mm'] for r in metrics if r['split']=='test' and r['readout']==label and r['subset']=='closure:final_return')
            lines.append(f'| {label} | {a:.5f} | {b:.5f} |')
        lines+=['','metrics.csv includes existing/return domains, arms, traversal and phases. '
            'No decoder sees sugar, time, target or condition labels as features. '
            'Targets remain relative to the original anchor. Different fitting weights and '
            'derived examples mean these aggregate scores are not directly interchangeable '
            'with the earlier original-training report.','',
            'Familiar circular paths are diagnostics, not untouched generalization evidence. '
            'Fresh test families supply the main held-out assessment. No long recovery '
            'calibration or original extrapolation evaluation is claimed.','']
        for number in (8,9,14,15):lines += [f'![Diagnostic path {number}](path_{number:02d}.png)','']
        (output/'report.md').write_text('\n'.join(lines))
        (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Sugar decoder comparison</title>'
            '<style>body{font:16px system-ui;max-width:1200px;margin:2em auto}img{max-width:100%}</style>'
            '<h1>Frozen dynamics: decoder-only comparison</h1><p>Fixed working sugar response; no physiological fit.</p>'
            '<p><a href="report.md">Results and methods</a> · <a href="metrics.csv">All subset metrics</a> · '
            '<a href="diagnostic_metrics.csv">Familiar-path diagnostics</a></p>'
            +''.join(f'<h2>Diagnostic path {n}</h2><img src="path_{n:02d}.png" alt="Decoder comparison">' for n in (8,9,14,15)))
        for k,v in network.state_dict().items():torch.testing.assert_close(v,state_before[k],atol=0,rtol=0)
        if any(digest(p)!=h for p,h in frozen.items()):raise ValueError('The derived data changed during fitting.')
        finish(output,manifest)
    except BaseException as error:
        fail(output,manifest,error);raise
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--batch-size',type=int,default=16)
    return analyze(p.parse_args(argv))


if __name__=='__main__':main()
