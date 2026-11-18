"""Minimal pause-retention panels for slide 8; no model replay or fitting."""

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
from .slide_retention import basic_panels

STEM='cx_home_vector_standard_retention_pause32s_forward4mm_slide08'


def export(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    parent=verify_bundle(source)
    if parent['kind']!='home_vector_displacement_retention_v1':raise ValueError('Require displacement-retention results.')
    if output==source or output.is_relative_to(source) or source.is_relative_to(output):raise ValueError('Output overlaps archive.')
    paths=[source/name for name in ('manifest.json','pause_32s.npz','path_15.npz','pair_metrics.csv','design.json')]
    hashes={str(p):digest(p) for p in paths}
    with np.load(source/'pause_32s.npz') as a:
        names=a['prefix_names'].tolist();i,j=names.index('rest'),names.index('forward_4mm')
        time=a['time_s'].copy();p=a['prediction'][[i,j]].astype(float);r=a['rates'][[i,j]].astype(float)
        target=a['target'][[i,j]].astype(float);controls=a['controls'].copy()
    np.testing.assert_allclose(controls,np.broadcast_to([45.,0.,0.],controls.shape),atol=0,rtol=0)
    np.testing.assert_allclose(target,np.broadcast_to(target[:,0:1,:],target.shape),atol=0,rtol=0)
    target_delta=target[1]-target[0];required=np.linalg.norm(target_delta,axis=1)
    np.testing.assert_allclose(required,4.,atol=1e-10,rtol=0)
    np.testing.assert_allclose([time[0],time[-1]],[0.,32.],atol=1e-10,rtol=0)
    dp=p[1]-p[0];decoded=np.linalg.norm(dp,axis=1);state=np.linalg.norm(r[1]-r[0],axis=1)
    signed_gain=dp@target_delta[0]/16.
    orthogonal=np.linalg.norm(dp-signed_gain[:,None]*target_delta[0],axis=1)
    with (source/'pair_metrics.csv').open() as f:
        rows=[x for x in csv.DictReader(f) if x['branch']=='pause_32s' and x['prefix_a']=='rest' and x['prefix_b']=='forward_4mm']
    for t,k in ((0.,0),(32.,-1)):
        row=next(x for x in rows if float(x['time_s'])==t)
        np.testing.assert_allclose([decoded[k],state[k]],[float(row['prediction_separation_mm']),float(row['state_separation'])],atol=1e-12,rtol=1e-12)
    # Same elapsed time as the pause: do not attribute 32-vs-128-second differences to input.
    with np.load(source/'path_15.npz') as a:
        n=int(np.argmin(abs(a['time_s']-32.)))
        np.testing.assert_allclose(a['time_s'][n],32.,atol=1e-10,rtol=0)
        moving_decoded=float(np.linalg.norm(a['prediction'][j,n].astype(float)-a['prediction'][i,n]))
        moving_state=float(np.linalg.norm(a['rates'][j,n].astype(float)-a['rates'][i,n]))
    minimum=int(np.argmin(decoded))
    summary=dict(initial_decoded_separation_mm=float(decoded[0]),final_decoded_separation_mm=float(decoded[-1]),
        required_separation_mm=4.,initial_state_distance=float(state[0]),final_state_distance=float(state[-1]),
        minimum_decoded_separation_mm=float(decoded[minimum]),minimum_time_s=float(time[minimum]),
        initial_signed_target_gain=float(signed_gain[0]),final_signed_target_gain=float(signed_gain[-1]),
        final_orthogonal_separation_mm=float(orthogonal[-1]),moving_path15_at_32s_decoded_separation_mm=moving_decoded,
        moving_path15_at_32s_state_distance=moving_state,duration_s=32.,prefix_a='rest',prefix_b='forward_4mm',samples=len(time))
    output,manifest=begin(output,'home_vector_slide08_pause_v1',source=str(source),source_manifest_sha256=hashes[str(source/'manifest.json')])
    manifest.update(immutable_input_sha256=hashes,simulation=False,refitting=False)
    write_json(output/'manifest.json',manifest)
    try:
        shutil.copyfile(__file__,output/'slide_pause.py');shutil.copyfile(Path(__file__).with_name('slide_retention.py'),output/'slide_retention.py')
        with plt.rc_context({'font.family':'DejaVu Sans','font.size':14,'axes.labelsize':15,'xtick.labelsize':13,'ytick.labelsize':13,'svg.fonttype':'none','pdf.fonttype':42}):
            basic_panels(output,time,decoded,state,stem=STEM,xlabel='Time under identical stationary input (s)',xticks=(0,8,16,24,32),xlim=(-.5,32.75))
        with (output/'plotted_values.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['time_s','required_separation_mm','decoded_separation_mm','state_distance_model_units','signed_target_gain','orthogonal_separation_mm'])
            writer.writerows(zip(time,required,decoded,state,signed_gain,orthogonal))
        write_json(output/'figure_values.json',summary)
        (output/'caption.md').write_text('Two simulations of the same frozen standard network have equal-duration prefixes: A is stationary for four seconds; B walks forward four millimeters in four seconds. Both then receive identical stationary inputs at heading 45 degrees for 32 seconds. The original home anchor is retained. Their correct home-vector separation remains 4 mm, whereas decoded separation changes from 3.424 to 2.130 mm and full-state distance from 43.413 to 4.216 model units. The response is nonmonotonic and retains an imperfect distinction between histories. This illustrative pair does not establish a universal pause leak; other displacement directions behave differently. The main slide-7 example lasts 128 seconds, so compare equal elapsed times before attributing differences to movement versus pausing. No simulation, smoothing, fitting or correction is performed for these exports.\n')
        if any(digest(p)!=h for p,h in hashes.items()):raise ValueError('Source changed.')
        finish(output,manifest)
    except BaseException as error:fail(output,manifest,error);raise
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='results/cx-standard-displacement-retention-20261117')
    p.add_argument('--output',required=True)
    args=p.parse_args(argv);print(json.dumps(export(args.source,args.output),indent=2))


if __name__=='__main__':main()
