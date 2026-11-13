"""Derived decoder task: existing no-cue paths plus independent return families."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from ..__main__ import begin, fail, finish
from ..data import digest, sample_protocol, verify_bundle, write_json
from .sugar_probe import insert_pause
from .sugar_response import SugarResponse


SPLITS = ('train', 'validation', 'test')
PATCH_RADIUS_MM = .1
EXIT_RADIUS_MM = .15
MIN_DELIVERY_INTERVAL_S = 30.


def patch_events(controls, target, dt, radius=PATCH_RADIUS_MM,
                 exit_radius=EXIT_RADIUS_MM, minimum_interval=MIN_DELIVERY_INTERVAL_S):
    """Environment-only contact detector; initial occupancy does not deliver sugar.

    Arm after leaving the outer radius; consume on reentry into the inner patch.
    A refractory contact is consumed, not delayed until eligibility while inside.
    """
    controls, target = np.asarray(controls), np.asarray(target)
    if (target.ndim != 2 or target.shape[1] != 2 or len(controls) != len(target)
            or not np.isfinite(target).all() or not np.isfinite([dt,radius,exit_radius,minimum_interval]).all()
            or not 0 < radius < exit_radius or dt <= 0 or minimum_interval < 0):
        raise ValueError('Require finite aligned geometry and positive nested patch radii.')
    events, armed, last = [], False, -np.inf
    for tick, distance in enumerate(np.linalg.norm(target, axis=1)):
        if distance > exit_radius:
            armed = True
        elif armed and distance <= radius:
            armed = False
            seconds = tick*dt
            # Last state has no following interval: do not deliver an unused event.
            if tick+1 < len(target) and seconds-last >= minimum_interval:
                events.append(float(seconds));last=seconds
    return events


def return_family(rng, family, dt):
    """Two repeats of a regular polygon, matched forward/rightward and pause arms."""
    sides = int(rng.choice([4,8,24]))
    choices = {4: np.arange(1.5,6.01,.25), 8: np.arange(.75,2.51,.25),
               24: np.arange(.25,.751,.25)}
    segment_s = float(rng.choice(choices[sides]))
    speed = float(rng.uniform(.5,2.))
    heading = float(rng.uniform(0,360))
    turn = int(rng.choice([-1,1]))
    pause_s = float(rng.choice([.5,1.,2.,4.]))
    period = sides*segment_s
    rows = []
    for alpha in (0.,90.):
        protocol=[]
        for step in range(2*sides):
            world_angle=heading+turn*360*(step % sides)/sides
            protocol.append([step*segment_s, (world_angle-alpha) % 360,
                             speed*np.cos(np.deg2rad(alpha)),speed*np.sin(np.deg2rad(alpha))])
        protocol.append([2*period, protocol[-1][1],0.,0.])
        for paused in (False,True):
            p = insert_pause(protocol,period,pause_s,dt) if paused else protocol
            controls,target=sample_protocol(p,dt)
            if np.linalg.norm(target[-1]) > 5e-6:
                raise ValueError('Generated return family is not closed.')
            contacts=patch_events(controls,target,dt)
            if not contacts:
                raise ValueError('Generated path has no sugar-patch reentry.')
            for sugar in (False,True):
                arm=('pause' if paused else 'continuous')+('_sugar' if sugar else '_control')
                rows.append(dict(family=family,kind='return_polygon',group='returns',
                    traversal='forward' if alpha==0 else 'rightward',arm=arm,
                    protocol=p,sugar_events_s=contacts if sugar else [],potential_contact_times_s=contacts,
                    sides=sides,speed_mm_s=speed,loop_period_s=period,
                    pause_s=pause_s if paused else 0.,turn_sign=turn))
    return rows


def validate_splits(groups):
    families=[{r['family'] for r in groups[s]} for s in SPLITS]
    if any(families[i]&families[j] for i in range(3) for j in range(i)):
        raise ValueError('Related families cross train/validation/test boundaries.')
    protocols=[]
    for s in SPLITS:
        protocols.append({json.dumps(r['protocol'],separators=(',',':')) for r in groups[s]})
    if any(protocols[i]&protocols[j] for i in range(3) for j in range(i)):
        raise ValueError('An exact movement protocol crosses fit/evaluation splits.')


def prepare(args):
    source,activity,config,diagnostics,output=[Path(getattr(args,k)).resolve()
        for k in ('source','activity','config','diagnostics','output')]
    for folder in (source,activity,diagnostics):
        if output==folder or output.is_relative_to(folder) or folder.is_relative_to(output):
            raise ValueError('Output must be separate from every source bundle.')
        verify_bundle(folder)
    if output.exists():raise FileExistsError(output)
    counts=[args.train_families,args.validation_families,args.test_families]
    if min(counts)<2:raise ValueError('Require at least two new families per split.')
    settings=json.loads(config.read_text());SugarResponse(**settings['response'])
    original=json.loads((source/'trajectories.json').read_text())
    dt=json.loads((source/'task.json').read_text())['dt_s']
    frozen={str(p):digest(p) for folder in (source,activity,diagnostics) for p in folder.iterdir() if p.is_file()}
    frozen[str(config)]=digest(config)
    output,manifest=begin(output,'home_vector_sugar_decoder_data_v1',source=str(source),
        source_manifest_sha256=digest(source/'manifest.json'),seed=args.seed)
    manifest.update(immutable_input_sha256=frozen,
        analysis_code_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')})
    write_json(output/'manifest.json',manifest)
    try:
        groups={}
        for split,seed,count in zip(SPLITS,np.random.SeedSequence(args.seed).spawn(3),counts):
            groups[split]=[dict(r,group='existing',arm='no_sugar',sugar_events_s=[]) for r in original[split]]
            rng=np.random.default_rng(seed)
            for i in range(count):
                groups[split].extend(return_family(rng,f'{split}-sugar-return-{i:04d}',dt))
            for i,r in enumerate(groups[split]):r['id']=f'{split}-{i:05d}'
        validate_splits(groups)
        groups['challenge']=[dict(r,group='existing',arm='no_sugar',sugar_events_s=[],id=f'challenge-{i:04d}')
                             for i,r in enumerate(original['challenge'])]
        old_protocols=json.loads((diagnostics/'protocols.json').read_text())
        old_config=json.loads((diagnostics/'config.json').read_text())
        groups['diagnostic']=[]
        for number in (8,9,14,15):
            path=f'path_{number:02d}'
            for pause in (False,True):
                key=path+('_pause' if pause else '_continuous')
                for sugar in (False,True):
                    arm=('pause' if pause else 'continuous')+('_sugar' if sugar else '_control')
                    groups['diagnostic'].append(dict(old_protocols[key],group='diagnostic',family=f'diagnostic-{path}',
                        id=path+'_'+arm,arm=arm,sugar_events_s=old_config['event_times_s'] if sugar else [],
                        event_policy='Historical scheduled pulse; not the new patch detector.'))
        for name in ('current_checkpoint.pt','current_metadata.json','neurons.csv'):
            shutil.copyfile(activity/name,output/name)
        write_json(output/'trajectories.json',groups)
        write_json(output/'response_config.json',settings)
        write_json(output/'task.json',dict(dt_s=dt,seed=args.seed,
            target='Fixed original anchor minus position; no origin reset.',
            event_policy=dict(radius_mm=PATCH_RADIUS_MM,exit_radius_mm=EXIT_RADIUS_MM,
                minimum_delivery_interval_s=MIN_DELIVERY_INTERVAL_S,initial_occupancy_delivers=False,
                note='Environmental reentry approximates sugar consumption; no visual/taste/feeding mechanics.'),
            source=str(source),activity=str(activity),diagnostics=str(diagnostics),
            group_weights=dict(existing=.5,returns=.5),family_weighting='Equal family, then equal variants within each group.',
            counts={s:len(rs) for s,rs in groups.items()},
            families={s:len({r['family'] for r in rs}) for s,rs in groups.items()},
            response_status='Fixed working approximation; not physiologically calibrated.',
            excluded='Original extrapolation split preserved at source, not included in this first comparison.'))
        for name in ('sugar_decoder_data.py','sugar_probe.py','sugar_response.py'):
            shutil.copyfile(Path(__file__).with_name(name),output/name)
        if any(digest(p)!=h for p,h in frozen.items()):raise ValueError('A source file changed.')
        finish(output,manifest)
    except BaseException as error:
        fail(output,manifest,error);raise
    return groups


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','activity','config','diagnostics','output'):p.add_argument(f'--{name}',required=True)
    p.add_argument('--seed',type=int,default=20261112)
    p.add_argument('--train-families',type=int,default=64)
    p.add_argument('--validation-families',type=int,default=16)
    p.add_argument('--test-families',type=int,default=24)
    return prepare(p.parse_args(argv))


if __name__=='__main__':main()
