"""Separate, sealed diverse-return task and training-only stratum normalization."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from .__main__ import begin, fail, finish
from .data import digest, sample_protocol, verify_bundle, write_json
from .analysis.sugar_decoder_data import patch_events, validate_splits


KIND='home_vector_sugar_diverse_data_v1'
STRATA=('existing_other','long_straight','two_return','diverse_return')
ROUTES=('regular_loop','out_and_back','irregular_return','mixed_excursions')
ENERGY_FLOOR_MM2=1e-8


def stratum(row):
    if 'stratum' in row:return row['stratum']
    if row['group']=='diverse_returns':return 'diverse_return'
    if row['group'] in ('returns','diagnostic'):return 'two_return'
    return 'long_straight' if row['kind'] in ('long_training','long_holdout') else 'existing_other'


def sampling_weights(rows):
    """Equal stratum, then equal family, then equal sibling probability."""
    from collections import Counter
    families={g:{r['family'] for r in rows if stratum(r)==g} for g in {stratum(r) for r in rows}}
    counts=Counter((stratum(r),r['family']) for r in rows)
    if not families:raise ValueError('Require nonempty trajectory strata.')
    return np.array([1./len(families)/len(families[stratum(r)])/counts[stratum(r),r['family']] for r in rows])


def loss_plan(training,dt):
    """Compute scales once, without consulting validation/test targets."""
    groups={stratum(r) for r in training}
    if groups!=set(STRATA):raise ValueError('Training requires all four declared strata.')
    probability=sampling_weights(training)
    energy={g:0. for g in STRATA};mass={g:0. for g in STRATA}
    for row,p in zip(training,probability):
        _,y=sample_protocol(row['protocol'],dt)
        g=stratum(row);energy[g]+=p*float(np.mean(np.sum(y[:-1]**2,axis=1)));mass[g]+=p
    raw={g:energy[g]/mass[g] for g in STRATA}
    if not all(np.isfinite(v) and v>0 for v in raw.values()):raise ValueError('Require positive finite training energies.')
    return dict(kind='training_stratum_target_energy_v1',source_split='train',dt_s=dt,
        energy_floor_mm2=ENERGY_FLOOR_MM2,raw_target_energy_mm2=raw,
        target_energy_mm2={g:max(v,ENERGY_FLOOR_MM2) for g,v in raw.items()},
        stratum_sampling_mass=mass,
        aggregation='Equal stratum/family/sibling; uniform time excluding final zero-duration state.',
        selection_metric='mean_stratum_normalized_mse',
        counts={g:sum(stratum(r)==g for r in training) for g in STRATA})


def _episode(rng,route,steps,home_steps,dt):
    """Return canonical world-space velocity segments with exact closure."""
    if route=='out_and_back':vertices=np.array([[0.,0.],[1.,0.],[0.,0.]])
    else:
        sides=int(rng.choice([4,8,24])) if route=='regular_loop' else int(rng.integers(3,8))
        if route=='regular_loop':angles=np.arange(sides)*2*np.pi/sides
        else:
            increments=rng.uniform(.7,1.3,sides);angles=np.r_[0.,np.cumsum(increments[:-1])]/increments.sum()*2*np.pi
        points=np.column_stack((np.cos(angles),np.sin(angles)))
        if route!='regular_loop':points*=rng.uniform(.85,1.15,(sides,1))
        points-=points[0];vertices=np.vstack((points,[0.,0.]))
    edges=np.diff(vertices,axis=0);lengths=np.linalg.norm(edges,axis=1)
    travel=steps-home_steps
    if travel<len(edges):raise ValueError('Insufficient movement steps.')
    durations=lengths/lengths.sum()*travel
    ticks=np.floor(durations).astype(int)
    ticks[np.argsort(-(durations-ticks))[:travel-int(ticks.sum())]]+=1
    if (ticks<1).any():raise ValueError('Empty generated segment.')
    speed=float(rng.uniform(.65,1.7))
    scale=speed*travel*dt/lengths.sum()
    velocities=edges*scale/(ticks[:,None]*dt)
    speeds=np.linalg.norm(velocities,axis=1)
    if speeds.min()<.5 or speeds.max()>2.:raise ValueError('Generated speed outside declared envelope.')
    theta=float(rng.uniform(0,2*np.pi))
    rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    velocities=velocities@rotation.T
    segments=[(int(t),v.tolist()) for t,v in zip(ticks,velocities)]
    if home_steps:segments.append((home_steps,[0.,0.]))
    return segments,travel


def _protocol(segments,heading_offset_deg,dt,orientation_deg):
    theta=np.deg2rad(orientation_deg)
    rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    alpha=np.deg2rad(heading_offset_deg);ticks=0;heading=orientation_deg-heading_offset_deg;rows=[]
    for count,world in segments:
        velocity=np.asarray(world)@rotation.T;speed=float(np.linalg.norm(velocity))
        if speed>0:heading=float(np.rad2deg(np.arctan2(velocity[1],velocity[0]))-heading_offset_deg)%360
        rows.append([ticks*dt,heading,speed*np.cos(alpha),speed*np.sin(alpha)])
        ticks+=count
    rows.append([ticks*dt,heading,0.,0.])
    return rows


def return_family(rng,family,route,dt,return_counts=(4,8),end_away=False):
    """Geometric siblings share complete route lineage, cue and body variants."""
    if route not in ROUTES or min(return_counts)<2:raise ValueError('Invalid route or return count.')
    for attempt in range(100):
        orientation=float(rng.uniform(0,360));budget=int(rng.choice([round(6/dt),round(12/dt)]))
        home_steps=round(float(rng.choice([.5,1.] if end_away else [0.,.25,.5,1.]))/dt)
        episodes=[];arrivals=[];elapsed=0
        template=None
        try:
            for j in range(max(return_counts)):
                selected=str(rng.choice(ROUTES[:3])) if route=='mixed_excursions' else route
                steps=budget if route=='regular_loop' else int(rng.choice([round(6/dt),round(12/dt)]))
                if route=='regular_loop' and template is not None:segments,travel=template
                else:segments,travel=_episode(rng,selected,steps,home_steps,dt)
                if route=='regular_loop':template=segments,travel
                episodes.append(segments);arrivals.append((elapsed+travel)*dt);elapsed+=steps
            # Same signature for 4/8/16-return variants and rotated/body copies.
            signature=hashlib.sha256(json.dumps(episodes[:4],separators=(',',':')).encode()).hexdigest()
            tail_direction=float(rng.uniform(0,2*np.pi));tail_speed=float(rng.uniform(.5,1.5))
            tail=[tail_speed*np.cos(tail_direction),tail_speed*np.sin(tail_direction)]
            records=[]
            for count in return_counts:
                segments=[s for episode in episodes[:count] for s in episode]
                if end_away:segments[-1]=(home_steps,tail)
                for alpha in (0.,90.):
                    protocol=_protocol(segments,alpha,dt,orientation)
                    controls,target=sample_protocol(protocol,dt)
                    returns=arrivals[:count]
                    indices=np.rint(np.array(returns)/dt).astype(int)
                    np.testing.assert_allclose(target[indices],0.,atol=5e-6,rtol=0)
                    contacts=patch_events(controls,target,dt,minimum_interval=0.)
                    # Reject incidental contacts en route: one arrival per excursion.
                    if len(contacts)!=count or not np.all((np.array(returns)-np.array(contacts)>=-dt)&
                            (np.array(returns)-np.array(contacts)<=.25)):
                        raise ValueError('Unexpected home-patch contact geometry.')
                    eligible=contacts if end_away else contacts[:-1]
                    if not eligible:raise ValueError('No delivery followed by continuation.')
                    if end_away and np.linalg.norm(target[-1])<=.15:raise ValueError('Open suffix did not leave home.')
                    for schedule in ('no_sugar','single_sugar','repeated_sugar'):
                        events=[] if schedule=='no_sugar' else (eligible[:1] if schedule=='single_sugar' else eligible)
                        records.append(dict(family=family,parent_signature=signature,kind='diverse_return',
                            group='diverse_returns',stratum='diverse_return',route_class=route,
                            traversal='forward' if alpha==0 else 'rightward',arm=schedule,
                            return_count=count,return_times_s=returns,end_away=end_away,
                            home_dwell_s=home_steps*dt,protocol=protocol,sugar_events_s=events,
                            potential_contact_times_s=contacts,eligible_contact_times_s=eligible))
            return records
        except (ValueError,AssertionError):continue
    raise ValueError(f'Could not generate valid {route} family after 100 attempts.')


def diagnostic_controls(rng,dt):
    rows=[]
    for kind in ('near_miss','pause_away'):
        vertices=([[0.,0.],[1.,0.],[1.,1.],[0.,.2],[-1.,.2]] if kind=='near_miss'
                  else [[0.,0.],[1.,0.],[1.,0.],[1.,1.]])
        segments=[(round(2/dt),((np.array(b)-a)/2).tolist()) for a,b in zip(vertices,vertices[1:])]
        orientation=float(rng.uniform(0,360))
        for alpha in (0.,90.):
            p=_protocol(segments,alpha,dt,orientation);controls,y=sample_protocol(p,dt)
            if patch_events(controls,y,dt,minimum_interval=0.):raise ValueError('Control unexpectedly contacted home.')
            rows.append(dict(id=f'diverse-control-{kind}-{alpha:g}',family='diverse-control-'+kind,
                kind=kind,group='diverse_returns',stratum='diverse_return',route_class=kind,
                traversal='forward' if alpha==0 else 'rightward',arm='no_sugar',protocol=p,sugar_events_s=[]))
    return rows


def validate(groups):
    validate_splits(groups)
    owners={}
    for split in ('train','validation','test'):
        for row in groups[split]:
            signature=row.get('parent_signature')
            if signature:
                if signature in owners and owners[signature]!=split:raise ValueError('Geometric parent crosses fit/evaluation splits.')
                owners[signature]=split
        if {stratum(r) for r in groups[split]}!=set(STRATA):raise ValueError('Missing declared stratum.')
    if 'extrapolation' in groups:
        test={r['family'] for r in groups['test'] if r['group']=='diverse_returns'}
        if {r['family'] for r in groups['extrapolation']}!=test:
            raise ValueError('Extrapolation must extend exactly the held-out test families.')


def prepare(args):
    source,output=Path(args.source).resolve(),Path(args.output).resolve()
    manifest=verify_bundle(source)
    if manifest['kind']!='home_vector_sugar_decoder_data_v1':raise ValueError('Require the sealed shorter-return task.')
    frozen=dict(manifest['immutable_input_sha256'])
    frozen.update({str(p):digest(p) for p in source.iterdir() if p.is_file()})
    for folder in {source,*[Path(p).parent for p in frozen]}:
        if output==folder or output.is_relative_to(folder) or folder.is_relative_to(output):raise ValueError('Output overlaps source archives.')
    counts=(args.train_families,args.validation_families,args.test_families)
    if any(type(c) is not int or c<4 or c%4 for c in counts):raise ValueError('Use positive multiples of four families per split.')
    if any(digest(p)!=h for p,h in frozen.items()):raise ValueError('Original source changed.')
    task=json.loads((source/'task.json').read_text());dt=task['dt_s']
    groups=json.loads((source/'trajectories.json').read_text())
    for rows in groups.values():
        for row in rows:row['stratum']=stratum(row)
    output,record=begin(output,KIND,source=str(source),source_manifest_sha256=digest(source/'manifest.json'),seed=args.seed)
    record['immutable_input_sha256']=frozen;write_json(output/'manifest.json',record)
    try:
        streams=np.random.SeedSequence(args.seed).spawn(4);extrapolation=[]
        for split,seed,count in zip(('train','validation','test'),streams[:3],counts):
            rng=np.random.default_rng(seed)
            for i in range(count):
                family=f'{split}-diverse-return-{i:04d}'
                variants=return_family(rng,family,ROUTES[i%4],dt,
                    return_counts=(4,8,16) if split=='test' else (4,8),end_away=bool((i//4)%2))
                for j,row in enumerate(variants):
                    row['id']=family+f'-{j:02d}'
                    (extrapolation if row['return_count']==16 else groups[split]).append(row)
        groups['extrapolation']=extrapolation
        groups['diverse_controls']=diagnostic_controls(np.random.default_rng(streams[3]),dt)
        validate(groups)
        plan=loss_plan(groups['train'],dt)
        for name in ('current_checkpoint.pt','current_metadata.json','neurons.csv','response_config.json'):
            shutil.copyfile(source/name,output/name)
        shutil.copyfile(__file__,output/'diverse_returns.py')
        write_json(output/'trajectories.json',groups);write_json(output/'loss_plan.json',plan)
        task.pop('group_weights',None)
        task.update(derived_parent=str(source),seed=args.seed,dataset_variant='diverse_returns_v1',
            family_weighting='Equal four strata, equal family, equal sibling within stratum.',
            counts={s:len(rows) for s,rows in groups.items()},
            families={s:len({r['family'] for r in rows}) for s,rows in groups.items()},
            new_event_policy=dict(entry_radius_mm=.1,exit_radius_mm=.15,minimum_interval_s=0.,
                final_return='Delivery eligible only when followed by another excursion or an open departure.',
                initial_occupancy_delivers=False),
            inherited_event_policy=task.pop('event_policy'),
            target='Fixed original anchor minus position; no reset or target reanchoring.')
        write_json(output/'task.json',task)
        statistics={}
        for split in ('train','validation','test','extrapolation'):
            new=[r for r in groups[split] if r['group']=='diverse_returns']
            statistics[split]=dict(trajectories=len(new),duration_range_s=[min(r['protocol'][-1][0] for r in new),max(r['protocol'][-1][0] for r in new)],
                route_counts={g:sum(r['route_class']==g for r in new) for g in ROUTES},
                deliveries_range=[min(len(r['sugar_events_s']) for r in new),max(len(r['sugar_events_s']) for r in new)])
        write_json(output/'dataset_summary.json',statistics)
        (output/'report.md').write_text('# Diverse return task\n\n'
            'All original examples retained with their split identities; new siblings remain grouped. '
            'The original anchor, PFN encoder, anatomy and response constants are unchanged. '
            'No network training is performed by this builder.\n\n'
            '## Counts\n\n'+ '\n'.join(f'- {s}: {len(rows)} trajectories' for s,rows in groups.items())+
            '\n\n## Frozen training-only normalization\n\n```json\n'+json.dumps(plan,indent=2)+'\n```\n')
        if any(digest(p)!=h for p,h in frozen.items()):raise ValueError('An archived input changed.')
        finish(output,record)
    except BaseException as error:fail(output,record,error);raise
    return groups


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=20261113)
    p.add_argument('--train-families',type=int,default=24)
    p.add_argument('--validation-families',type=int,default=12)
    p.add_argument('--test-families',type=int,default=24)
    return prepare(p.parse_args(argv))


if __name__=='__main__':main()
