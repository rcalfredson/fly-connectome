"""Component-wise close-ups from archived long-loop predictions; no replay or fit."""

import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ..__main__ import begin, fail, finish
from ..data import digest, verify_bundle, write_json
from .decoder_contributions import write_csv


CONDITIONS=('original','sugar_single','sugar_repeated')
COLORS=('tab:blue','tab:orange','tab:green')


def analyze(args):
    source,output=Path(args.source).resolve(),Path(args.output).resolve()
    if output==source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Require separate source and output directories.')
    original=verify_bundle(source)
    if original['kind']!='home_vector_sugar_long_loops_v1':raise ValueError('Require completed long-loop predictions.')
    if type(args.loop) is not int or not 1<=args.loop<=original['settings']['loops']:
        raise ValueError('Requested loop is outside the archived simulation.')
    if output.exists():raise FileExistsError(output)
    frozen={str(p):digest(p) for p in source.iterdir() if p.is_file()}
    protocols=json.loads((source/'protocols.json').read_text())
    reference=list(csv.DictReader((source/'loop_metrics.csv').open()))
    output,manifest=begin(output,'home_vector_sugar_loop_components_v1',source=str(source),
        source_manifest_sha256=digest(source/'manifest.json'),loop=args.loop,readout='refit')
    manifest.update(immutable_input_sha256=frozen,refitting=False,replay=False)
    write_json(output/'manifest.json',manifest)
    try:
        shutil.copyfile(__file__,output/'sugar_loop_components.py')
        rows=[]
        for number in (8,9,14,15):
            path=f'path_{number:02d}';stem=f'{path}_loop_{args.loop:02d}'
            with np.load(source/(path+'.npz')) as z:
                time=z['time_s'];period=protocols[path]['period_s'];dt=time[1]-time[0]
                n=round(period/dt);a,b=(args.loop-1)*n,args.loop*n
                time=time[a:b+1];phase=time-time[0];target=z['target'][a:b+1]
                predictions={c:z[c+'_refit_prediction'][a:b+1] for c in CONDITIONS}
            fig,axes=plt.subplots(4,3,figsize=(15,12),sharex=True,layout='constrained')
            saved=dict(time_s=time,loop_time_s=phase,target=target)
            for col,(condition,color) in enumerate(zip(CONDITIONS,COLORS)):
                p=predictions[condition];error=p-target;saved[condition+'_prediction']=p
                saved[condition+'_error']=error
                e=error[:-1];bias=e.mean(0)
                row=dict(path=path,loop=args.loop,condition=condition,
                    vector_rmse_mm=float(np.sqrt(np.mean(np.sum(e*e,axis=1)))),
                    x_rmse_mm=float(np.sqrt(np.mean(e[:,0]**2))),y_rmse_mm=float(np.sqrt(np.mean(e[:,1]**2))),
                    mean_x_error_mm=float(bias[0]),mean_y_error_mm=float(bias[1]),
                    mean_error_norm_mm=float(np.linalg.norm(bias)),
                    centered_vector_rms_mm=float(np.sqrt(np.mean(np.sum((e-bias)**2,axis=1)))),
                    closure_x_error_mm=float(error[-1,0]),closure_y_error_mm=float(error[-1,1]),
                    closure_error_mm=float(np.linalg.norm(error[-1])))
                old=next(r for r in reference if r['path']==path and r['condition']==condition
                         and int(r['loop'])==args.loop and r['readout']=='refit')
                np.testing.assert_allclose([row['vector_rmse_mm'],row['closure_error_mm']],
                    [float(old['vector_rmse_mm']),float(old['closure_error_mm'])],atol=1e-12,rtol=1e-12)
                np.testing.assert_allclose(row['vector_rmse_mm']**2,
                    row['mean_error_norm_mm']**2+row['centered_vector_rms_mm']**2,atol=1e-10)
                rows.append(row)
                for component in (0,1):
                    ax=axes[component,col]
                    ax.plot(phase,p[:,component],color=color,label='Predicted')
                    ax.plot(phase,target[:,component],'k--',label='Correct')
                    ax.set_ylabel(('Home x' if component==0 else 'Home y')+' (mm)')
                    ax=axes[component+2,col]
                    ax.plot(phase,error[:,component],color=color)
                    ax.axhline(0,color='k',ls='--',lw=.8)
                    ax.set_ylabel(('x' if component==0 else 'y')+' error: predicted - correct (mm)')
                axes[0,col].set_title(condition+'_refit')
                axes[0,col].legend(fontsize=8)
                axes[-1,col].set_xlabel('Time within loop (s)')
            for ax in axes.flat:ax.grid(alpha=.2);ax.set_xlim(0,period)
            fig.suptitle(f'{path}: loop {args.loop}, {time[0]:g}–{time[-1]:g} s; fixed refitted readouts\n'
                'Raw predictions and signed errors; vertical scales vary across panels')
            for ext in ('png','pdf'):fig.savefig(output/(stem+'_components.'+ext),dpi=150)
            plt.close(fig)
            # Common axes make smaller differences between the two sugar schedules visible.
            fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,layout='constrained')
            for component in (0,1):
                axes[component,0].plot(phase,target[:,component],'k--',lw=1.5,label='Correct')
                for condition,color in zip(CONDITIONS[1:],COLORS[1:]):
                    p=predictions[condition]
                    axes[component,0].plot(phase,p[:,component],color=color,label=condition+'_refit')
                    axes[component,1].plot(phase,p[:,component]-target[:,component],color=color,label=condition+'_refit')
                axes[component,1].axhline(0,color='k',ls='--',lw=.8)
                for col in (0,1):axes[component,col].set_ylabel(('x' if component==0 else 'y')+' (mm)')
            axes[0,0].set_title('Predicted and correct home-vector components')
            axes[0,1].set_title('Signed component errors: predicted - correct')
            for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_xlim(0,period)
            for ax in axes[-1]:ax.set_xlabel('Time within loop (s)')
            fig.suptitle(f'{path}, loop {args.loop}: sugar schedules on shared axes; no baseline subtraction')
            for ext in ('png','pdf'):fig.savefig(output/(stem+'_sugar_zoom.'+ext),dpi=150)
            plt.close(fig)
            np.savez_compressed(output/(stem+'.npz'),**saved)
        write_csv(output/'component_metrics.csv',rows)
        lines=[f'# Loop {args.loop}: home-vector component close-ups','',
            'All three conditions use the previously refitted decoders. There is no neural replay, '
            'new fitting, baseline correction, origin reset or parameter selection. '
            'Single sugar means one delivery at 8 s; repeated sugar means deliveries every 8 s '
            'after nonfinal returns. The same sugar decoder serves both schedules.','',
            'Component errors are predicted minus correct, in world-centered coordinates. '
            'The main panels have independent vertical scales so the much larger original errors '
            'do not hide sugar-condition waveforms. The sugar zoom overlays both schedules on shared axes.','',
            '| Path | Condition | Vector RMSE | Mean x error | Mean y error | Centered error RMS | Closure error |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
        for r in rows:lines.append(f"| {r['path']} | {r['condition']} | {r['vector_rmse_mm']:.4f} | {r['mean_x_error_mm']:.4f} | {r['mean_y_error_mm']:.4f} | {r['centered_vector_rms_mm']:.4f} | {r['closure_error_mm']:.4f} |")
        lines+=['','All numerical columns use mm. Mean error measures average signed bias over this loop. '
            'Centered error RMS measures variation of the error around that mean; it still includes '
            'tracking distortions and drift within the loop. This descriptive split does not establish '
            'their cause or correct the predictions. Vector RMSE squared equals mean-error-vector norm '
            'squared plus centered error RMS squared. Time averages exclude the final zero-duration '
            'state, which is shown in plots and scored separately as closure error.','']
        for number in (8,9,14,15):
            stem=f'path_{number:02d}_loop_{args.loop:02d}'
            lines.extend([f'## Path {number}','',f'![Components]({stem}_components.png)','',
                f'![Sugar close-up]({stem}_sugar_zoom.png)',''])
        (output/'report.md').write_text('\n'.join(lines))
        (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Loop component close-ups</title>'
            '<style>body{font:16px system-ui;max-width:1400px;margin:2em auto}img{max-width:100%}</style>'
            '<h1>Home-vector component close-ups</h1><p><a href="report.md">Methods and metrics</a> · '
            '<a href="component_metrics.csv">Component metrics CSV</a></p>'
            +''.join(f'<h2>Path {n}, loop {args.loop}</h2><img src="path_{n:02d}_loop_{args.loop:02d}_components.png">'
                f'<img src="path_{n:02d}_loop_{args.loop:02d}_sugar_zoom.png">' for n in (8,9,14,15)))
        if any(digest(p)!=h for p,h in frozen.items()):raise ValueError('Source archive changed.')
        finish(output,manifest)
    except BaseException as error:fail(output,manifest,error);raise
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--loop',type=int,default=16)
    return analyze(p.parse_args(argv))


if __name__=='__main__':main()
