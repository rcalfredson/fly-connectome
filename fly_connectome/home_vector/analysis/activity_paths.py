"""Raw-rate leaders on varied paths and neural-state norms after movement resumes."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..__main__ import batch, begin, fail, finish
from ..data import Circuit, digest, sample_protocol, verify_bundle, write_json
from .activity import record_trajectory, write_csv
from .activity_checks import assess, sweep_records
from .checkpoint import capture


def rate_leaders(rates, types):
    """Share ties; ignore silent states and the zero-duration final sample."""
    rates = np.asarray(rates)
    if rates.ndim != 2 or len(rates) < 2 or rates.shape[1] != len(types) or not np.isfinite(rates).all():
        raise ValueError('Require finite time/neuron rates and matching types.')

    def fractions(values):
        peak = values[:-1].max(axis=1)
        active = peak > 1e-8
        tied = np.isclose(values[:-1], peak[:, None], rtol=1e-6, atol=1e-8) & active[:, None]
        weights = tied / np.maximum(tied.sum(axis=1, keepdims=True), 1)
        return weights.sum(axis=0) / max(int(active.sum()), 1)

    global_fraction = fractions(rates)
    within = np.zeros(rates.shape[1])
    for kind in set(types):
        mask = np.asarray(types) == kind
        within[mask] = fractions(rates[:, mask])
    return global_fraction, within


def norm_traces(rates, decoder, reference):
    """Absolute norms and changes relative to one fixed pause-onset state."""
    r = np.asarray(rates, dtype=float)
    _, singular, vh = np.linalg.svd(np.asarray(decoder, dtype=float), full_matrices=False)
    tolerance = (singular[0] if len(singular) else 0.) * max(decoder.shape) * np.finfo(float).eps
    basis = vh[singular > tolerance]
    visible = (r @ basis.T) @ basis
    hidden = r - visible
    delta = r - reference
    delta_visible = (delta @ basis.T) @ basis
    return dict(full_norm=np.linalg.norm(r, axis=1), hidden_norm=np.linalg.norm(hidden, axis=1),
        visible_norm=np.linalg.norm(visible, axis=1),
        change_from_pause_onset_norm=np.linalg.norm(delta, axis=1),
        hidden_change_from_pause_onset_norm=np.linalg.norm(delta - delta_visible, axis=1))


def varied_paths(seed):
    """Matched world paths with distinct body motions; held-control curved approximations."""
    rng = np.random.default_rng(seed)
    rows = []
    for path in range(4):
        segments = 3 + path % 2
        world = rng.choice(np.arange(8) * 45., size=segments)
        speeds = rng.choice([.5, 1., 2.], size=segments)
        body = rng.choice(np.arange(8) * 45., size=segments)
        for mode in ('forward', 'mixed'):
            protocol = []
            for j in range(segments):
                direction = 0. if mode == 'forward' else body[j]
                v = speeds[j] * np.array([np.cos(np.deg2rad(direction)), np.sin(np.deg2rad(direction))])
                protocol.append([2. * j, float((world[j] - direction) % 360), float(v[0]), float(v[1])])
            protocol.append([2. * segments, protocol[-1][1], 0., 0.])
            rows.append(dict(family=f'random-{path}', kind=f'{segments}-segment-{mode}', protocol=protocol))
    for omega in (-45., -22.5, 22.5, 45.):
        for direction in (0., 90.):
            protocol = []
            for tick in range(80):
                time = tick / 10
                v = np.array([np.cos(np.deg2rad(direction)), np.sin(np.deg2rad(direction))])
                protocol.append([time, float((45. + omega * time - direction) % 360), float(v[0]), float(v[1])])
            protocol.append([8., protocol[-1][1], 0., 0.])
            rows.append(dict(family=f'curve-{omega:g}', kind='forward' if direction == 0 else 'rightward',
                             protocol=protocol))
    return rows


def summarize_raw(rates, circuit, condition, focus):
    global_fraction, within = rate_leaders(rates, circuit.dynamic_types)
    mean, peak = rates[:-1].mean(axis=0), rates.max(axis=0)
    focus_index = circuit.dynamic_ids.index(focus) if focus in circuit.dynamic_ids else None
    leader = int(np.argmax(global_fraction)) if global_fraction.sum() else None
    summary = dict(**condition, most_often_highest_body_id=None if leader is None else circuit.dynamic_ids[leader],
        largest_leading_time_fraction=float(global_fraction.max()),
        focus_global_leading_fraction=None if focus_index is None else float(global_fraction[focus_index]),
        focus_within_population_leading_fraction=None if focus_index is None else float(within[focus_index]))
    neurons = [dict(**condition, body_id=body_id, cell_type=circuit.dynamic_types[j],
        mean_rate=float(mean[j]), peak_rate=float(peak[j]),
        fraction_active_time_highest_global=float(global_fraction[j]),
        fraction_active_time_highest_within_population=float(within[j]))
        for j, body_id in enumerate(circuit.dynamic_ids)]
    return summary, neurons


def raw_checks(snap, output, args):
    circuit, network, dt = Circuit(**snap['circuit']), snap['network'], snap['task']['dt_s']
    conditions, neurons = [], []
    records = sweep_records(dt, 8., [.5, 1., 2.], args.angles)
    for start in range(0, len(records), args.chunk_size):
        selected = records[start:start + args.chunk_size]
        with torch.no_grad():
            x, y, mask = batch(circuit, selected, dt, 'cpu', network.a.dtype)
            prediction, rates = network(x, dt)
        for k, row in enumerate(selected):
            length = int(mask[k].sum())
            r = rates[k, :length].numpy()
            if assess(prediction[k, :length].numpy(), y[k, :length].numpy(), r, dt)['status'] != 'finite':
                raise ValueError('Raw sweep exceeded numerical guard.')
            condition = dict(condition=len(conditions), speed_mm_s=row['speed_mm_s'],
                heading_deg=row['heading_deg'], body_direction_deg=row['body_direction_deg'])
            summary, per_neuron = summarize_raw(r, circuit, condition, args.focus_body_id)
            conditions.append(summary)
            neurons.extend(per_neuron)
    write_csv(output / 'raw_sweep_conditions.csv', conditions)
    write_csv(output / 'raw_sweep_neurons.csv', neurons)
    write_json(output / 'raw_sweep_protocols.json', records)
    path_reports, path_neurons = [], []
    for i, row in enumerate(varied_paths(args.seed)):
        arrays, _ = record_trajectory(network, circuit, row, dt)
        stem = f'path_{i:02d}'
        np.savez_compressed(output / f'{stem}.npz', **arrays)
        summary, per_neuron = summarize_raw(arrays['rates'], circuit,
            dict(path=stem, family=row['family'], kind=row['kind']), args.focus_body_id)
        path_reports.append(dict(**summary, record=row))
        path_neurons.extend(per_neuron)
    write_csv(output / 'raw_path_neurons.csv', path_neurons)
    write_json(output / 'raw_paths.json', path_reports)
    print(f'Raw-rate checks: {len(conditions)} straight conditions and {len(path_reports)} varied paths complete', flush=True)
    return conditions, path_reports


def resume_records(duration, angles, speeds, pause_heading):
    headings = [float(v) for v in np.arange(angles) * 360 / angles]
    rows = [dict(kind='stationary', heading_deg=pause_heading, body_direction_deg=None, speed_mm_s=0.,
                 protocol=[[0., pause_heading, 0., 0.], [duration, pause_heading, 0., 0.]])]
    for heading in headings:
        if not np.isclose(heading, pause_heading):
            rows.append(dict(kind='stationary', heading_deg=heading, body_direction_deg=None, speed_mm_s=0.,
                protocol=[[0., heading, 0., 0.], [duration, heading, 0., 0.]]))
    for speed in speeds:
        for heading in headings:
            for direction in np.arange(angles) * 360 / angles:
                v = speed * np.array([np.cos(np.deg2rad(direction)), np.sin(np.deg2rad(direction))])
                rows.append(dict(kind='moving', heading_deg=heading, body_direction_deg=float(direction), speed_mm_s=speed,
                    protocol=[[0., heading, float(v[0]), float(v[1])], [duration, heading, 0., 0.]]))
    return rows


def add_varied_restarts(records, duration, dt, seed):
    """Matched heading-only controls for entire mixed/curved restart protocols."""
    final_tick = round(duration / dt)
    for original in varied_paths(seed):
        # Quantize stretched controls onto the actual simulation grid. Last
        # control wins if several boundaries coincide on a very short probe.
        controls = {}
        for time, heading, forward, right in original['protocol']:
            tick = round(time / original['protocol'][-1][0] * final_tick)
            controls[tick] = [tick * dt, heading, forward, right]
        protocol = [controls[tick] for tick in sorted(controls)]
        common = dict(heading_deg=protocol[0][1], body_direction_deg=None, speed_mm_s=None,
                      path_family=original['family'], path_kind=original['kind'])
        control_index = len(records)
        records.append(dict(**common, kind='path_stationary', matched_control_index=control_index,
            protocol=[[t, h, 0., 0.] for t, h, _, _ in protocol]))
        records.append(dict(**common, kind='path_moving', matched_control_index=control_index, protocol=protocol))
    return records


def resume_checks(snap, atlas, output, args):
    circuit, network, dt = Circuit(**snap['circuit']), snap['network'], snap['task']['dt_s']
    reports, table = [], []
    decoder = network.D.detach().numpy()
    histories = [row for row in atlas['trajectories'] if row['record']['family'] == 'challenge-pause-5']
    if len(histories) != 2:
        raise ValueError('Require the two standard pause histories.')
    for trajectory in histories:
        row = trajectory['record']
        history = 'backward' if row['protocol'][0][2] < 0 else 'forward'
        with np.load(Path(args.activity) / f'{trajectory["stem"]}.npz') as saved:
            onset = round(row['suffix_start_s'] / dt)
            reference = saved['rates'][onset].copy()
            home = saved['target'][onset].copy()
            old_pause = saved['prediction'][onset:].copy()
        heading = row['protocol'][1][1]
        for pause in args.pauses:
            stationary = dict(protocol=[[0., heading, 0., 0.], [pause, heading, 0., 0.]])
            with torch.no_grad():
                x, _, _ = batch(circuit, [stationary], dt, 'cpu', network.a.dtype)
                pause_prediction, pause_rates = network(x, dt, initial=torch.as_tensor(reference[None]))
            overlap = min(len(old_pause), pause_prediction.shape[1])
            np.testing.assert_allclose(pause_prediction[0, :overlap].numpy(), old_pause[:overlap], atol=.002, rtol=0)
            initial = pause_rates[0, -1].numpy().copy()
            records = resume_records(args.resume_seconds, args.angles, [.5, 1., 2.], heading)
            records = add_varied_restarts(records, args.resume_seconds, dt, args.seed)
            curves = {key: [] for key in norm_traces(initial[None], decoder, reference)}
            predictions, targets, case_rows = [], [], []
            for start in range(0, len(records), args.chunk_size):
                selected = records[start:start + args.chunk_size]
                with torch.no_grad():
                    x, y, _ = batch(circuit, selected, dt, 'cpu', network.a.dtype)
                    p, r = network(x, dt, initial=torch.as_tensor(initial[None]).expand(len(selected), -1))
                for k, control in enumerate(selected):
                    rates, prediction, target = r[k].numpy(), p[k].numpy(), y[k].numpy() + home
                    score = assess(prediction, target, rates, dt)
                    if score['status'] != 'finite':
                        raise ValueError('Unperturbed resume probe exceeded numerical guard.')
                    norms = norm_traces(rates, decoder, reference)
                    for key, values in norms.items():
                        curves[key].append(values)
                    predictions.append(prediction.copy())
                    targets.append(target.copy())
                    item = dict(case=len(reports), variant=len(case_rows), history=history, pause_s=pause,
                        kind=control['kind'], heading_deg=control['heading_deg'],
                        body_direction_deg=control['body_direction_deg'], speed_mm_s=control['speed_mm_s'],
                        path_family=control.get('path_family'), path_kind=control.get('path_kind'), **score)
                    for key in norms:
                        values = norms[key]
                        item.update({f'{key}_initial': float(values[0]), f'{key}_end': float(values[-1]),
                            f'{key}_minimum_change': float(values.min() - values[0]),
                            f'{key}_end_change': float(values[-1] - values[0]),
                            f'{key}_minimum_time_s': float(np.argmin(values) * dt),
                            f'{key}_maximum_drawdown': float(np.max(np.maximum.accumulate(values) - values))})
                    case_rows.append(item)
            curves = {key: np.stack(values) for key, values in curves.items()}
            # Heading-only controls separate heading changes from walking effects.
            for i, item in enumerate(case_rows):
                matched = records[i].get('matched_control_index')
                if matched is None:
                    matched = next(j for j, c in enumerate(records) if c['kind'] == 'stationary'
                                   and np.isclose(c['heading_deg'], item['heading_deg']))
                for key in curves:
                    item[f'{key}_end_difference_from_same_heading_stationary'] = float(curves[key][i, -1] - curves[key][matched, -1])
                    item[f'{key}_end_difference_from_continued_pause'] = float(curves[key][i, -1] - curves[key][0, -1])
            stem = f'resume_{len(reports):02d}'
            # Persist full-resolution norm/readout curves for every branch; raw
            # activity for these branches is reproducible from initial+protocol.
            np.savez_compressed(output / f'{stem}.npz', **curves,
                prediction=np.stack(predictions), target=np.stack(targets),
                time_s=np.arange(len(predictions[0])) * dt,
                pause_onset_state=reference, restart_state=initial)
            write_json(output / f'{stem}_protocols.json', records)
            reports.append(dict(stem=stem, case=len(reports), history=history, pause_s=pause,
                rows=case_rows, full_norm_at_pause_onset=float(np.linalg.norm(reference)),
                full_norm_at_restart=float(np.linalg.norm(initial))))
            table.extend(case_rows)
            print(f'Restart: {history}, {pause:g} s pause, {len(records)} controls complete', flush=True)
    write_csv(output / 'resume_conditions.csv', table)
    write_json(output / 'resume_cases.json', reports)
    return reports


def analyze(args):
    if (args.angles < 2 or args.chunk_size < 1 or args.threads < 1 or args.seed < 0
            or min(args.pauses) <= 0 or args.resume_seconds <= 0
            or not np.isfinite([*args.pauses, args.resume_seconds]).all()):
        raise ValueError('Require valid positive finite probe settings.')
    activity, source, output = [Path(getattr(args, key)).resolve() for key in ('activity', 'source', 'output')]
    for path in (activity, source):
        if output == path or output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError('Output must be separate from input bundles.')
    if output.exists():
        raise FileExistsError(output)
    if verify_bundle(activity)['kind'] != 'home_vector_activity_v1':
        raise ValueError('Require a completed activity atlas.')
    args.activity = str(activity)
    snap = capture(activity, source)
    atlas = json.loads((activity / 'activity.json').read_text())
    if atlas['checkpoint_sha256'] != snap['checkpoint_sha256']:
        raise ValueError('Activity and checkpoint identities differ.')
    dt = snap['task']['dt_s']
    for duration in [*args.pauses, args.resume_seconds]:
        if not np.isclose(duration / dt, round(duration / dt), rtol=0, atol=1e-7):
            raise ValueError('Durations must fall on the simulation grid.')
    torch.set_num_threads(args.threads)
    frozen = {str(p): digest(p) for folder in (activity, source) for p in folder.iterdir() if p.is_file()}
    output, manifest = begin(output, 'home_vector_activity_paths_v1', source=str(source),
        source_manifest_sha256=digest(source / 'manifest.json'), activity=str(activity))
    manifest.update(device='cpu', threads=args.threads, torch=torch.__version__, refitting=False,
        immutable_input_sha256=frozen,
        analysis_code_sha256={p.name: digest(p) for p in Path(__file__).parent.glob('*.py')})
    write_json(output / 'manifest.json', manifest)
    try:
        for name in ('current_checkpoint.pt', 'current_metadata.json', 'neurons.csv'):
            (output / name).write_bytes((activity / name).read_bytes())
        conditions, paths = raw_checks(snap, output, args)
        restarts = resume_checks(snap, atlas, output, args)
        report = dict(checkpoint_update=snap['checkpoint']['update'], checkpoint_sha256=snap['checkpoint_sha256'],
            settings=vars(args), straight_conditions=len(conditions), varied_paths=len(paths), restart_cases=len(restarts),
            definitions=dict(raw='Raw model rates, no decoder weighting. Ties share credit; silent samples and zero-duration final sample excluded.',
                full_norm='L2 norm of current rate vector, without subtraction.',
                hidden_norm='L2 norm of current rate vector projected into decoder null space.',
                hidden_change_from_pause_onset_norm='L2 norm of null-space component of r(t)-r(pause onset), with the SAME reference after restarting.',
                minimum='Minimum over tested restart horizon, including t=0; a negative value means a decrease below the restart value.',
                drawdown='Largest decrease from any earlier peak after restart; can be positive even when every sample exceeds the restart norm.',
                curved='Piecewise-constant heading updated every 0.1 s, not a continuously varying control.',
                scope='One frozen seed; exploratory probes, no model selection or physiological validation.'))
        write_json(output / 'paths_report.json', report)
        from .activity_paths_plots import make_plots
        make_plots(output, snap, conditions, paths, restarts, report)
        if any(digest(path) != value for path, value in frozen.items()):
            raise ValueError('An immutable input bundle changed during analysis.')
        finish(output, manifest)
    except BaseException as error:
        fail(output, manifest, error)
        raise
    print(json.dumps(dict(output=str(output), status='complete', checkpoint_update=report['checkpoint_update']), indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('activity', 'source', 'output'):
        parser.add_argument(f'--{name}', required=True)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--chunk-size', type=int, default=8)
    parser.add_argument('--angles', type=int, default=8)
    parser.add_argument('--pauses', type=float, nargs='+', default=[5., 30., 60.])
    parser.add_argument('--resume-seconds', type=float, default=8.)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--focus-body-id', type=int, default=142911)
    return analyze(parser.parse_args(argv))


if __name__ == '__main__':
    main()
