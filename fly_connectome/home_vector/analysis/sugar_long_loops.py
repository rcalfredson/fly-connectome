"""Long circular replay with frozen dynamics/readouts and explicit sugar schedules."""

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
from ..data import Circuit, digest, encode, sample_protocol, verify_bundle, write_json
from .checkpoint import capture
from .decoder_contributions import CHECKPOINT_SHA256, write_csv
from .sugar_decoder_fit import batched_sugar_rates
from .sugar_response import SugarResponse, event_gate


def extend_protocol(protocol, dt, loops):
    if not isinstance(loops,int) or loops<2:
        raise ValueError('Require at least two complete loops.')
    a=np.asarray(protocol,float);controls,target=sample_protocol(a,dt)
    if np.any(a[-1,2:]!=0) or np.linalg.norm(target[-1])>5e-6:
        raise ValueError('Require a closed movement with terminal stop.')
    period=float(a[-1,0]);n=len(controls)-1
    p=np.vstack([a[:-1]+[j*period,0,0,0] for j in range(loops)]+[a[-1]+[(loops-1)*period,0,0,0]])
    repeated,truth=sample_protocol(p,dt)
    for j in range(loops):
        np.testing.assert_array_equal(repeated[j*n:(j+1)*n],controls[:-1])
    np.testing.assert_allclose(truth[np.arange(1,loops+1)*n],0.,atol=5e-6,rtol=0)
    return p.tolist(),n,period


def analyze(args):
    data,fit,output=[Path(getattr(args,k)).resolve() for k in ('data','fit','output')]
    dm,fm=verify_bundle(data),verify_bundle(fit)
    if dm['kind']!='home_vector_sugar_decoder_data_v1' or fm['kind']!='home_vector_sugar_decoder_fit_v1':
        raise ValueError('Require the derived task and completed decoder comparison.')
    if fm['settings']['data_manifest_sha256']!=digest(data/'manifest.json'):
        raise ValueError('Decoder fit does not match derived data.')
    task=json.loads((data/'task.json').read_text());activity=Path(task['activity']);source=Path(task['source'])
    for folder in (data,fit,activity,source):
        if output==folder or output.is_relative_to(folder) or folder.is_relative_to(output):
            raise ValueError('Output must be separate from source archives.')
    if output.exists():raise FileExistsError(output)
    snap=capture(activity,source)
    if snap['checkpoint_sha256']!=CHECKPOINT_SHA256:raise ValueError('Wrong frozen checkpoint.')
    network,circuit,dt=snap['network'],Circuit(**snap['circuit']),task['dt_s']
    torch.set_num_threads(1)
    response=SugarResponse(**json.loads((data/'response_config.json').read_text())['response'])
    mask=torch.tensor(np.array(circuit.dynamic_types)=='hDeltaG')
    records={r['path']:r['record'] for r in json.loads((activity/'raw_paths.json').read_text())}
    old=np.vstack((network.D.detach().numpy().T,network.d.detach().numpy()))
    readouts={'original_frozen':old,'sugar_frozen':old}
    for model in ('original','sugar'):
        with np.load(fit/(model+'_decoder.npz')) as z:
            np.testing.assert_array_equal(z['body_ids'],circuit.dynamic_ids)
            readouts[model+'_refit']=np.vstack((z['decoder'].T,z['intercept']))
    frozen={str(p):digest(p) for folder in (data,fit,activity,source) for p in folder.iterdir() if p.is_file()}
    before={k:v.clone() for k,v in network.state_dict().items()}
    output,manifest=begin(output,'home_vector_sugar_long_loops_v1',data=str(data),fit=str(fit),loops=args.loops)
    manifest.update(immutable_input_sha256=frozen,checkpoint_sha256=CHECKPOINT_SHA256,refitting=False)
    write_json(output/'manifest.json',manifest)
    try:
        shutil.copyfile(__file__,output/'sugar_long_loops.py')
        for name in ('sugar_decoder_fit.py','sugar_response.py'):
            shutil.copyfile(Path(__file__).with_name(name),output/name)
        write_json(output/'response.json',response.__dict__)
        rows,protocols=[],{}
        for number in (8,9,14,15):
            name=f'path_{number:02d}'
            protocol,n,period=extend_protocol(records[name]['protocol'],dt,args.loops)
            controls,target=sample_protocol(protocol,dt);time=np.arange(len(controls))*dt
            events={'original':[],'sugar_single':[period],
                    'sugar_repeated':[j*period for j in range(1,args.loops)]}
            protocols[name]=dict(protocol=protocol,events_s=events,period_s=period)
            gates=np.stack([event_gate(time,e,response) for e in events.values()])
            inputs=torch.tensor(np.repeat(encode(circuit,controls)[None],3,axis=0),dtype=network.a.dtype)
            with torch.no_grad():
                rates=batched_sugar_rates(network,inputs,dt,gates,response,mask).numpy()
                _,native=network(inputs[:1,:n+1],dt)
            np.testing.assert_allclose(rates[:,:n+1],np.repeat(native.numpy(),3,axis=0),atol=3e-5,rtol=1e-6)
            saved=dict(time_s=time,target=target,gates=gates)
            fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
            for j,condition in enumerate(events):
                rate=rates[j];saved[condition+'_rates']=rate
                model='original' if j==0 else 'sugar'
                for status in ('frozen','refit'):
                    label=condition+'_'+status;weight=readouts[model+'_'+status]
                    prediction=np.column_stack((rate,np.ones(len(rate))))@weight
                    saved[label+'_prediction']=prediction
                    error=np.linalg.norm(prediction-target,axis=1)
                    for loop in range(1,args.loops+1):
                        start,end=(loop-1)*n,loop*n
                        rows.append(dict(path=name,condition=condition,readout=status,loop=loop,
                            vector_rmse_mm=float(np.sqrt(np.mean(error[start:end]**2))),
                            closure_error_mm=float(error[end]),
                            closure_drift_mm=float(np.linalg.norm(prediction[end]-prediction[0])),
                            max_activity=float(rate[start:end+1].max()),
                            endpoint_activity_norm=float(np.linalg.norm(rate[end]))))
                    if status=='refit':
                        axes[0,0].plot(time,error,label=label)
                        axes[0,1].plot(np.arange(1,args.loops+1),error[np.arange(1,args.loops+1)*n],marker='.',label=label)
                axes[1,0].plot(time,rate.max(1),label=condition)
                hg=rate[:,np.array(circuit.dynamic_types)=='hDeltaG'].mean(1)
                axes[1,1].plot(time,hg,label=condition)
            for ax,title in zip(axes.flat,('Vector error: refitted readouts (mm)','Closure error: refitted readouts (mm)','Largest single-neuron activity','hDeltaG mean activity')):
                ax.set_title(title);ax.grid(alpha=.2);ax.legend(fontsize=7)
                ax.set_xlabel('Loop number' if ax is axes[0,1] else 'Time (s)')
            fig.suptitle(name+': frozen circuit and readouts; scheduled sugar; no pauses or state resets')
            fig.savefig(output/(name+'.png'),dpi=140);plt.close(fig)
            np.savez_compressed(output/(name+'.npz'),**saved)
            print(f'Completed {name}: {args.loops} loops',flush=True)
        write_csv(output/'loop_metrics.csv',rows);write_json(output/'protocols.json',protocols)
        lines=['# Longer circular paths: frozen sugar comparison','',
            'Original, one sugar pulse at 8 s, and pulses every 8 s after each nonfinal return. '
            'Repeated pulses are explicit scheduled delivery assumptions, with no 30 s patch refractory rule. '
            'Gate pulses combine by maximum, not addition. No pauses; one zero-state initialization per trial. '
            'The original anchor, recurrent weights, sugar parameters, and all readouts remain fixed.','',
            '| Path | Condition | Loops | Loop RMSE (mm) | Closure error (mm) | Max activity |',
            '| --- | --- | ---: | ---: | ---: | ---: |']
        for r in rows:
            if r['readout']=='refit' and r['loop'] in (2,4,8,16):
                lines.append(f"| {r['path']} | {r['condition']} | {r['loop']} | {r['vector_rmse_mm']:.4f} | {r['closure_error_mm']:.4f} | {r['max_activity']:.3f} |")
        lines+=['','These familiar paths are diagnostic extrapolation beyond decoder training durations. '
            'The table gives the individual numbered loop, not cumulative RMSE. '
            'loop_metrics.csv retains both frozen and refitted readouts and every loop. '
            'Activity units are arbitrary; the model has no upper firing-rate cap. '
            'Finite-horizon behavior does not establish asymptotic stability. '
            'No weights or parameters are selected using these results.','']
        for number in (8,9,14,15):lines.extend([f'![Path {number}](path_{number:02d}.png)',''])
        (output/'report.md').write_text('\n'.join(lines))
        for k,v in network.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)
        if any(digest(p)!=h for p,h in frozen.items()):raise ValueError('An archived input changed.')
        finish(output,manifest)
    except BaseException as error:fail(output,manifest,error);raise
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','fit','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--loops',type=int,default=16)
    return analyze(p.parse_args(argv))


if __name__=='__main__':main()
