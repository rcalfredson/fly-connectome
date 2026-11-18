"""Export a focused, frozen-data retention figure for presentation slide 7."""

import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ..__main__ import begin,fail,finish
from ..data import digest,verify_bundle,write_json


STEM='cx_home_vector_standard_retention_cw_rightward_16loops_slide07'


def basic_panels(output,time,decoded,state,*,stem=STEM,
                 xlabel='Time under identical circular input (s)',
                 xticks=(0,32,64,96,128),xlim=(-2,131)):
    """Independent panels with slide text and annotations deliberately omitted."""
    specifications=(
        ('decoded_basic',decoded,'#1765aa','Home-vector separation\n(mm)',(-.12,4.55),[0,1,2,3,4]),
        ('state_basic',state,'#7351a4','State distance\n(model units)',(-1.1,48),[0,10,20,30,40]))
    for suffix,values,color,ylabel,limits,ticks in specifications:
        fig,ax=plt.subplots(figsize=(8,3.6))
        fig.subplots_adjust(left=.17,right=.975,bottom=.235,top=.96)
        if suffix=='decoded_basic':
            ax.axhline(4.,color='#303740',ls=(0,(6,4)),lw=1.8)
        ax.plot(time,values,color=color,lw=2.3)
        ax.set(xlim=xlim,ylim=limits,xticks=xticks,yticks=ticks)
        ax.set_xlabel(xlabel,labelpad=8)
        ax.set_ylabel(ylabel,labelpad=8)
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color('#aab1b9')
        ax.grid(axis='y',color='#e3e7ec',lw=.8);ax.set_axisbelow(True)
        for extension in ('png','svg','pdf'):
            fig.savefig(output/(stem+'_'+suffix+'.'+extension),dpi=300,facecolor='white')
        plt.close(fig)


def export(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    parent=verify_bundle(source)
    if parent['kind']!='home_vector_displacement_retention_v1':
        raise ValueError('Require the completed displacement-retention experiment.')
    if output==source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Output must not overlap the source archive.')
    paths=[source/'manifest.json',source/'path_15.npz',source/'design.json',source/'pair_metrics.csv']
    hashes={str(p):digest(p) for p in paths}
    with np.load(source/'path_15.npz') as data:
        names=data['prefix_names'].tolist();a=names.index('rest');b=names.index('forward_4mm')
        time=data['time_s'].copy()
        predictions=data['prediction'][[a,b]].astype(float)
        states=data['rates'][[a,b]].astype(float)
        targets=data['target'][[a,b]].astype(float)
    decoded=np.linalg.norm(predictions[1]-predictions[0],axis=1)
    state=np.linalg.norm(states[1]-states[0],axis=1)
    target_delta=targets[1]-targets[0]
    np.testing.assert_allclose(target_delta,np.broadcast_to(target_delta[0],target_delta.shape),atol=1e-10,rtol=0)
    np.testing.assert_allclose(np.linalg.norm(target_delta,axis=1),4.,atol=1e-10,rtol=0)
    np.testing.assert_allclose([time[0],time[-1]],[0.,128.],atol=1e-10,rtol=0)
    # Independently check the plotted endpoints against the archived pair table.
    with (source/'pair_metrics.csv').open() as f:
        rows=[r for r in csv.DictReader(f) if r['branch']=='path_15' and r['prefix_a']=='rest' and r['prefix_b']=='forward_4mm']
    for t,k in ((0.,0),(128.,-1)):
        row=next(r for r in rows if float(r['time_s'])==t)
        np.testing.assert_allclose(decoded[k],float(row['prediction_separation_mm']),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(state[k],float(row['state_separation']),rtol=1e-12,atol=1e-12)
    output,manifest=begin(output,'home_vector_slide07_retention_v1',source=str(source),source_manifest_sha256=hashes[str(source/'manifest.json')],branch='path_15',prefix_a='rest',prefix_b='forward_4mm')
    manifest.update(immutable_input_sha256=hashes,simulation=False,refitting=False)
    write_json(output/'manifest.json',manifest)
    try:
        shutil.copyfile(__file__,output/'slide_retention.py')
        with plt.rc_context({'font.family':'DejaVu Sans','font.size':14,'axes.titlesize':17,
                             'axes.labelsize':15,'xtick.labelsize':13,'ytick.labelsize':13,
                             'svg.fonttype':'none','pdf.fonttype':42}):
            fig,axes=plt.subplots(2,1,figsize=(12.8,7.2),sharex=True)
            fig.subplots_adjust(left=.115,right=.965,bottom=.115,top=.745,hspace=.38)
            fig.text(.115,.945,'Different histories, identical subsequent movement',fontsize=23,weight='bold',ha='left')
            fig.text(.115,.886,'A: stationary for 4 s     B: walks forward 4 mm in 4 s',fontsize=15,ha='left')
            fig.text(.115,.843,'Then both: 16 clockwise circles with rightward traversal (128 s)',fontsize=15,ha='left')
            fig.text(.115,.798,'Frozen standard network  •  Original home anchor retained',fontsize=12,color='#525b66',ha='left')
            blue,purple='#1765aa','#7351a4'
            ax=axes[0]
            ax.axhline(4.,color='#303740',ls=(0,(6,4)),lw=1.8,label='Required separation: 4 mm')
            ax.plot(time,decoded,color=blue,lw=2.3,label='Decoded separation')
            ax.scatter(time[[0,-1]],decoded[[0,-1]],color=blue,s=28,zorder=4)
            ax.annotate(f'{decoded[0]:.3f} mm',(time[0],decoded[0]),xytext=(10,8),textcoords='offset points',color=blue,fontsize=13,weight='bold')
            ax.annotate(f'{decoded[-1]:.3f} mm',(time[-1],decoded[-1]),xytext=(-8,11),ha='right',textcoords='offset points',color=blue,fontsize=13,weight='bold')
            ax.set_ylabel('Home-vector separation\n(mm)');ax.set_ylim(-.12,4.55)
            ax.set_yticks([0,1,2,3,4])
            ax.legend(loc='upper right',frameon=True,facecolor='white',edgecolor='white',framealpha=1.,fontsize=13)
            ax=axes[1]
            ax.plot(time,state,color=purple,lw=2.3)
            ax.scatter(time[[0,-1]],state[[0,-1]],color=purple,s=28,zorder=4)
            ax.annotate(f'{state[0]:.3f}',(time[0],state[0]),xytext=(10,-2),textcoords='offset points',color=purple,fontsize=13,weight='bold')
            ax.annotate(f'{state[-1]:.3f}',(time[-1],state[-1]),xytext=(-8,11),ha='right',textcoords='offset points',color=purple,fontsize=13,weight='bold')
            ax.set_ylabel('Neural-state distance\n(model units)');ax.set_ylim(-1.1,48)
            ax.set_yticks([0,10,20,30,40]);ax.set_xlabel('Time under identical circular input (s)',labelpad=9)
            ax.set_xticks([0,32,64,96,128]);ax.set_xlim(-2,131)
            for ax in axes:
                ax.spines[['top','right']].set_visible(False)
                ax.spines[['left','bottom']].set_color('#aab1b9')
                ax.grid(axis='y',color='#e3e7ec',lw=.8);ax.set_axisbelow(True)
            for extension in ('png','svg','pdf'):
                fig.savefig(output/(STEM+'.'+extension),dpi=300,facecolor='white')
            plt.close(fig)
            basic_panels(output,time,decoded,state)
        with (output/'plotted_values.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['time_s','required_separation_mm','decoded_separation_mm','neural_state_distance_model_units'])
            writer.writerows(zip(time,np.full(len(time),4.),decoded,state))
        summary=dict(required_separation_mm=4.,initial_decoded_separation_mm=float(decoded[0]),final_decoded_separation_mm=float(decoded[-1]),initial_state_distance=float(state[0]),final_state_distance=float(state[-1]),time_s=128.,loops=16,phase='clockwise_rightward',samples=len(time),resampling=False)
        write_json(output/'figure_values.json',summary)
        (output/'caption.md').write_text('Two equal-duration histories precede identical circular movement in the frozen standard network. A remains stationary; B walks 4 mm forward. With the original home anchor retained, their correct home vectors must remain 4 mm apart. Decoded separation falls from 3.424 to 0.020 mm, while the Euclidean distance between their complete 65-neuron activity vectors falls from 43.413 to 0.040 model units. Full-resolution raw data are shown without smoothing, refitting or state resets. The example demonstrates attenuation of relevant history in this learned solution; it does not establish an architectural impossibility or identical neural states.\n')
        if any(digest(p)!=h for p,h in hashes.items()):raise ValueError('Source archive changed.')
        finish(output,manifest)
    except BaseException as error:fail(output,manifest,error);raise
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='results/cx-standard-displacement-retention-20261117')
    parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    print(json.dumps(export(args.source,args.output),indent=2))


if __name__=='__main__':main()
