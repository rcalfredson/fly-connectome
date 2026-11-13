"""Seal a task with reliable delivery at every home-patch reentry.

The sugar arm receives these events; the paired original arm always disables
the gate. Movement, target anchor, encoder, anatomy and response law stay fixed.
"""

import argparse
import json
from pathlib import Path
import shutil

from .__main__ import begin,fail,finish
from .data import digest,sample_protocol,verify_bundle,write_json
from . import diverse_returns
from .analysis.sugar_decoder_data import patch_events


KIND='home_vector_sugar_reliable_data_v1'
POLICY='all_home_patch_reentries'


def contact_events(row,dt):
    """No refractory interval, loop-count rule, or schedule-based withholding.

    Initial occupancy is not an event. An entry at the final zero-duration
    state has no following update and is excluded by the common detector.
    Earlier entries on the final approach are ordinary delivered events.
    """
    controls,target=sample_protocol(row['protocol'],dt)
    return patch_events(controls,target,dt,minimum_interval=0.)


def convert_rows(rows,dt):
    """Retain each distinct family/movement once, replacing all cue schedules."""
    unique={};output=[]
    for source in rows:
        key=(source['family'],json.dumps(source['protocol'],separators=(',',':')))
        # Only cue-expanded families have deliberate duplicate movements.
        collapse=source['group'] in ('returns','diverse_returns','diagnostic')
        if collapse and key in unique:
            unique[key]['source_variant_ids'].append(source['id'])
            continue
        row=dict(source)
        events=contact_events(row,dt)
        row.update(arm='home_reentry_sugar',delivery_policy=POLICY,sugar_events_s=events,
            potential_contact_times_s=events,eligible_contact_times_s=events,
            source_variant_ids=[source['id']])
        output.append(row)
        if collapse:unique[key]=row
    return output


def validate(groups,dt):
    diverse_returns.validate(groups)
    for split,rows in groups.items():
        movements=set()
        for row in rows:
            if row.get('delivery_policy')!=POLICY or row['arm']!='home_reentry_sugar':
                raise ValueError(f'Inconsistent reliable-delivery policy in {split}.')
            expected=contact_events(row,dt)
            if row['sugar_events_s']!=expected:
                raise ValueError(f'Missing or spurious home delivery in {split}: {row["id"]}')
            if row['group'] in ('returns','diverse_returns','diagnostic'):
                key=(row['family'],json.dumps(row['protocol'],separators=(',',':')))
                if key in movements:raise ValueError('Duplicate cue-expanded movement remains.')
                movements.add(key)


def prepare(args):
    source,output=Path(args.source).resolve(),Path(args.output).resolve()
    parent=verify_bundle(source)
    if parent['kind']!=diverse_returns.KIND:raise ValueError('Require the sealed diverse-return task.')
    task=json.loads((source/'task.json').read_text());dt=task['dt_s']
    frozen=dict(parent['immutable_input_sha256'])
    frozen.update({str(p):digest(p) for p in source.iterdir() if p.is_file()})
    for folder in {source,*[Path(p).parent for p in frozen]}:
        if output==folder or output.is_relative_to(folder) or folder.is_relative_to(output):
            raise ValueError('Output overlaps an original archive.')
    if output.exists():raise FileExistsError(output)
    if any(digest(p)!=h for p,h in frozen.items()):raise ValueError('An archived source changed.')
    originals=json.loads((source/'trajectories.json').read_text())
    output,manifest=begin(output,KIND,source=str(source),source_manifest_sha256=digest(source/'manifest.json'))
    manifest['immutable_input_sha256']=frozen;write_json(output/'manifest.json',manifest)
    try:
        groups={split:convert_rows(rows,dt) for split,rows in originals.items()}
        for split,rows in groups.items():
            for index,row in enumerate(rows):row['id']=f'{split}-home-sugar-{index:05d}'
        validate(groups,dt)
        plan=diverse_returns.loss_plan(groups['train'],dt)
        for name in ('current_checkpoint.pt','current_metadata.json','neurons.csv','response_config.json'):
            shutil.copyfile(source/name,output/name)
        for name in ('reliable_sugar.py','diverse_returns.py'):
            shutil.copyfile(Path(__file__).with_name(name),output/name)
        write_json(output/'trajectories.json',groups);write_json(output/'loss_plan.json',plan)
        task.update(dataset_variant='reliable_home_sugar_v1',policy_parent=str(source),
            counts={s:len(rows) for s,rows in groups.items()},
            families={s:len({r['family'] for r in rows}) for s,rows in groups.items()},
            delivery_policy=dict(mode=POLICY,entry_radius_mm=.1,departure_radius_mm=.15,
                minimum_interval_s=0.,initial_occupancy_delivers=False,
                eligibility='Every entry after departure with at least one following update; includes final approaches.',
                original_network_gate='Always zero',sugar_network_gate='All recorded home-patch events',
                target='Original anchor throughout; no reset or reanchoring.'))
        # These fields describe schedules that this new dataset replaces.
        for name in ('new_event_policy','inherited_event_policy'):task.pop(name,None)
        write_json(output/'task.json',task)
        summary={}
        for split,rows in groups.items():
            durations=[r['protocol'][-1][0] for r in rows]
            summary[split]=dict(trajectories=len(rows),with_delivery=sum(bool(r['sugar_events_s']) for r in rows),
                without_home_reentry=sum(not r['sugar_events_s'] for r in rows),
                duration_range_s=[min(durations),max(durations)],
                delivery_count_range=[min(len(r['sugar_events_s']) for r in rows),max(len(r['sugar_events_s']) for r in rows)],
                strata={g:sum(diverse_returns.stratum(r)==g for r in rows) for g in diverse_returns.STRATA})
        write_json(output/'dataset_summary.json',summary)
        lines=['# Reliable home-patch sugar task','',
            'Sugar-response training receives every eligible home-patch reentry after departure. '
            'The paired original network sees identical movement and targets with its gate always zero. '
            'All paths retain the original anchor; there are no geometric state resets. '
            'A final approach contact is delivered when there is a following update. '
            'Paths without reentry naturally have no event.','',
            '| Split | Trajectories | With home deliveries | No reentry |',
            '| --- | ---: | ---: | ---: |']
        for split,s in summary.items():lines.append(f'| {split} | {s["trajectories"]} | {s["with_delivery"]} | {s["without_home_reentry"]} |')
        lines+=['','Frozen normalization is derived from training only. Source variant IDs preserve '
            'provenance while redundant movement copies are removed. This builder performs no network training.','']
        (output/'report.md').write_text('\n'.join(lines))
        if any(digest(p)!=h for p,h in frozen.items()):raise ValueError('An archived input changed.')
        finish(output,manifest)
    except BaseException as error:fail(output,manifest,error);raise
    return groups


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    return prepare(p.parse_args(argv))


if __name__=='__main__':main()
