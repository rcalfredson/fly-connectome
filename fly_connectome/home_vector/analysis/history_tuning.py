"""Identical movement probes from states established by different trajectories."""

import argparse
import html
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from ..__main__ import batch, begin, fail, finish
from ..data import Circuit, digest, encode, write_json
from .activity import write_csv
from .activity_paths import resume_records
from .checkpoint import capture


VIEWS = ('raw', 'starting_delta', 'movement_delta')
LABELS = dict(raw='Actual firing rate', starting_delta='Change from saved starting rate',
              movement_delta='Moving minus same-heading stationary continuation')


def histories(duration, heading, speed):
    return [dict(name=name, protocol=[[0., heading, forward, 0.], [duration, heading, 0., 0.]])
            for name, forward in [('stationary', 0.), ('forward', speed), ('backward', -speed)]]


def probe_from_state(network, circuit, records, state, dt, chunk_size):
    """Every trial gets its own state copy; input at t changes the state at t+dt."""
    predictions, rates = [], []
    for start in range(0, len(records), chunk_size):
        selected = records[start:start + chunk_size]
        with torch.no_grad():
            x, _, _ = batch(circuit, selected, dt, 'cpu', network.a.dtype)
            initial = torch.as_tensor(state, dtype=network.a.dtype).repeat(len(selected), 1)
            p, r = network(x, dt, initial=initial)
        np.testing.assert_array_equal(r[:, 0].numpy(), np.broadcast_to(state, (len(selected), network.size)))
        predictions.extend(p.numpy())
        rates.extend(r.numpy())
    rates, predictions = np.stack(rates), np.stack(predictions)
    if (not np.isfinite(rates).all() or not np.isfinite(predictions).all()
            or rates.min() < 0 or rates.max() > 1e6):
        raise ValueError('History probe exceeded numerical guard.')
    return predictions, rates


def response_views(rates, state, records):
    """Return moving-condition/time/neuron arrays, with a matched control per heading."""
    moving = [i for i, row in enumerate(records) if row['kind'] == 'moving']
    controls = [next(i for i, row in enumerate(records) if row['kind'] == 'stationary'
                     and row['heading_deg'] == records[j]['heading_deg']) for j in moving]
    raw = rates[moving]
    return dict(raw=raw, starting_delta=raw - state,
                movement_delta=raw - rates[controls]), moving, controls


def map_stats(values, conditions):
    values = np.asarray(values, dtype=float)
    span = float(np.ptp(values))
    flat = span <= 1e-6
    maximum = conditions[int(np.argmax(values))]
    return dict(minimum=float(values.min()), maximum=float(values.max()), modulation_range=span,
                flat=flat, peak_heading_deg=None if flat else maximum['heading_deg'],
                peak_body_direction_deg=None if flat else maximum['body_direction_deg'],
                peak_tied_points=None if flat else int(np.isclose(values, values.max(), rtol=1e-6, atol=1e-8).sum()))


def compare_maps(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ac, bc = a - a.mean(), b - b.mean()
    norm = np.linalg.norm(ac) * np.linalg.norm(bc)
    return dict(mean_shift=float(b.mean() - a.mean()), rms_difference=float(np.sqrt(np.mean((b - a) ** 2))),
                centered_rms_difference=float(np.sqrt(np.mean((bc - ac) ** 2))),
                pattern_correlation=float(np.clip(ac @ bc / norm, -1., 1.))
                if min(np.ptp(a), np.ptp(b)) > 1e-6 and norm > 0 else None)


def plot_neuron(output, body_id, cell_type, values, records, history_names, sample_times, view, pattern=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .activity_plots import save

    # values: history, moving-condition, sampled-time
    angles = sorted({r['heading_deg'] for r in records})
    directions = sorted({r['body_direction_deg'] for r in records})
    display = values.copy()
    if pattern:
        low = display.min(axis=1, keepdims=True)
        span = np.ptp(display, axis=1, keepdims=True)
        display = np.divide(display - low, span, out=np.full_like(display, np.nan), where=span > 1e-6)
    signed = view != 'raw' and not pattern
    limit = 1. if pattern else max(float(np.max(np.abs(display))), 1e-12)
    fig, axes = plt.subplots(len(sample_times), len(history_names), squeeze=False,
        figsize=(4 * len(history_names), 3.35 * len(sample_times)), layout='constrained')
    cmap = plt.get_cmap('coolwarm' if signed else 'viridis').copy()
    cmap.set_bad('lightgray')
    for h, name in enumerate(history_names):
        for t, seconds in enumerate(sample_times):
            grid = np.empty((len(angles), len(directions)))
            for k, row in enumerate(records):
                grid[angles.index(row['heading_deg']), directions.index(row['body_direction_deg'])] = display[h, k, t]
            ax = axes[t, h]
            image = ax.imshow(grid, origin='lower', aspect='auto', cmap=cmap,
                              vmin=-limit if signed else 0., vmax=limit)
            ax.set(title=f'{name} history; probe {seconds * 1000:g} ms',
                   xlabel='Body movement direction (deg)', ylabel='Global heading (deg)')
            ax.set_xticks(range(len(directions)), [f'{v:g}' for v in directions], rotation=45, fontsize=7)
            ax.set_yticks(range(len(angles)), [f'{v:g}' for v in angles], fontsize=7)
    note = ('Each map separately spans 0–1; amplitude and offsets removed; flat maps gray'
            if pattern else 'One shared scale across all histories and probe times; rates in model units')
    fig.suptitle(f'{cell_type} {body_id}: {LABELS[view]}\n{note}')
    fig.colorbar(image, ax=list(axes.flat), shrink=.65,
                 label='Per-map normalized pattern' if pattern else 'Rate (model units)' if view == 'raw' else 'Rate difference (model units)')
    stem = f'neuron_{body_id}_{view}' + ('_pattern' if pattern else '')
    save(fig, output, stem)
    return stem


def analyze(args):
    run, source, output = [Path(getattr(args, key)).resolve() for key in ('run', 'source', 'output')]
    if args.threads < 1 or args.chunk_size < 1 or args.angles < 2:
        raise ValueError('Require positive thread/chunk counts and at least two angles.')
    if not all(np.isfinite(v) and v > 0 for v in (args.history_seconds, args.history_speed, args.speed, *args.times)):
        raise ValueError('Durations, sample times and speeds must be finite and positive.')
    if not args.times or not np.isfinite(args.heading):
        raise ValueError('Require sample times and a finite history heading.')
    for folder in (run, source):
        if output == folder or output.is_relative_to(folder) or folder.is_relative_to(output):
            raise ValueError('Output must be separate from every input bundle.')
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(args.threads)
    snap = capture(run, source, args.snapshot_key)
    circuit, network, dt = Circuit(**snap['circuit']), snap['network'], snap['task']['dt_s']
    for seconds in [args.history_seconds, *args.times]:
        if not np.isclose(seconds / dt, round(seconds / dt), atol=1e-8, rtol=0):
            raise ValueError('History duration and probe times must lie exactly on the simulation time grid.')
    if len(set(args.focus)) != len(args.focus) or any(body not in circuit.dynamic_ids for body in args.focus):
        raise ValueError('Focus body IDs must be distinct and present in the circuit.')
    ticks = sorted({round(t / dt) for t in args.times})
    times = np.array(ticks) * dt
    history_rows = histories(args.history_seconds, args.heading % 360, args.history_speed)
    records = resume_records(float(times[-1]), args.angles, [args.speed], args.heading % 360)
    frozen_folders = {source}
    if snap['input_kind'] != 'training_run' or snap['manifest']['status'] == 'complete':
        frozen_folders.add(run)
    frozen = {str(p): digest(p) for folder in frozen_folders for p in folder.iterdir() if p.is_file()}
    output, manifest = begin(output, 'home_vector_history_tuning_v1', run=str(run), source=str(source),
        source_manifest_sha256=digest(source / 'manifest.json'), history_seconds=args.history_seconds,
        history_heading_deg=args.heading % 360, history_speed_mm_s=args.history_speed,
        probe_speed_mm_s=args.speed, angles=args.angles, sample_times_s=times.tolist(), focus=args.focus)
    manifest.update(device='cpu', refitting=False, threads=args.threads, torch=torch.__version__,
        immutable_input_sha256=frozen,
        analysis_code_sha256={p.name: digest(p) for p in Path(__file__).parent.glob('*.py')})
    write_json(output / 'manifest.json', manifest)
    try:
        (output / 'current_checkpoint.pt').write_bytes(snap['blob'])
        write_json(output / 'current_metadata.json', dict(run=str(run), source=str(source),
            run_manifest=snap['manifest'], circuit=snap['circuit'], history=snap['history'],
            training_config=snap['training_config'], checkpoint_update=snap['checkpoint']['update'],
            checkpoint_sha256=snap['checkpoint_sha256'], input_kind=snap['input_kind']))
        write_json(output / 'histories.json', history_rows)
        write_json(output / 'probe_conditions.json', records)
        write_csv(output / 'neurons.csv', [dict(neuron_index=j, body_id=body, cell_type=circuit.dynamic_types[j])
                  for j, body in enumerate(circuit.dynamic_ids)])
        view_arrays = {key: [] for key in VIEWS}
        endpoint_rows, states, probe_rates, probe_predictions = [], [], [], []
        for history in history_rows:
            with torch.no_grad():
                x, y, _ = batch(circuit, [history], dt, 'cpu', network.a.dtype)
                p, r = network(x, dt)
            state = r[0, -1].numpy().copy()
            if not np.isfinite(r.numpy()).all() or r.min() < 0 or r.max() > 1e6:
                raise ValueError('History trajectory exceeded numerical guard.')
            states.append(state)
            np.savez_compressed(output / f'history_{history["name"]}.npz',
                time_s=np.arange(r.shape[1]) * dt, pfn=x[0].numpy(), rates=r[0].numpy(),
                prediction=p[0].numpy(), target=y[0].numpy())
            endpoint_rows.append(dict(history=history['name'], state_norm=float(np.linalg.norm(state)),
                target_home_x_mm=float(y[0, -1, 0]), target_home_y_mm=float(y[0, -1, 1]),
                predicted_home_x_mm=float(p[0, -1, 0]), predicted_home_y_mm=float(p[0, -1, 1])))
            prediction, rates = probe_from_state(network, circuit, records, state, dt, args.chunk_size)
            views, moving, controls = response_views(rates, state, records)
            for view in VIEWS:
                view_arrays[view].append(views[view][:, ticks])
            probe_rates.append(rates)
            probe_predictions.append(prediction)
            print(f'History tuning: {history["name"]}, {len(records)} independent probes complete', flush=True)
        views = {key: np.stack(value) for key, value in view_arrays.items()}
        selected = [records[i] for i in moving]
        names = [h['name'] for h in history_rows]
        np.savez_compressed(output / 'probe_responses.npz', rates=np.stack(probe_rates),
            prediction=np.stack(probe_predictions), initial_states=np.stack(states),
            time_s=np.arange(probe_rates[0].shape[1]) * dt, sample_ticks=ticks,
            moving_indices=moving, stationary_indices=controls,
            pfn_at_onset=encode(circuit, np.array([row['protocol'][0][1:] for row in records])), **views)
        write_csv(output / 'history_endpoints.csv', endpoint_rows)
        statistics, comparisons = [], []
        for view in VIEWS:
            for t, seconds in enumerate(times):
                for j, body in enumerate(circuit.dynamic_ids):
                    common = dict(body_id=body, cell_type=circuit.dynamic_types[j], view=view, time_s=float(seconds))
                    for h, name in enumerate(names):
                        statistics.append(dict(**common, history=name, **map_stats(views[view][h, :, t, j], selected)))
                    for a, b in combinations(range(len(names)), 2):
                        comparisons.append(dict(**common, history_a=names[a], history_b=names[b],
                            **compare_maps(views[view][a, :, t, j], views[view][b, :, t, j])))
        write_csv(output / 'tuning_metrics.csv', statistics)
        write_csv(output / 'history_comparisons.csv', comparisons)
        plots = []
        for body in args.focus:
            j = circuit.dynamic_ids.index(body)
            for view in VIEWS:
                for pattern in ([False, True] if view == 'movement_delta' else [False]):
                    stem = plot_neuron(output, body, circuit.dynamic_types[j], views[view][:, :, :, j],
                                       selected, names, times, view, pattern)
                    plots.append(dict(body_id=body, view=view, pattern=pattern, stem=stem))
        report = dict(checkpoint_update=snap['checkpoint']['update'], checkpoint_sha256=snap['checkpoint_sha256'],
            dt_s=dt, histories=history_rows, endpoints=endpoint_rows, focus_body_ids=args.focus,
            moving_conditions=len(moving), stationary_controls=len(records) - len(moving),
            sample_times_s=times.tolist(), plots=plots,
            definitions=dict(timing='State at t precedes the update driven by input at t; terminal history state seeds every probe independently.',
                raw=LABELS['raw'], starting_delta='r_probe(t) - r_history_endpoint.',
                movement_delta='r_moving(t) - r_stationary_same_heading(t), both seeded from the same full history state.',
                array_order='rates/prediction: history, probe_condition, time, neuron/xy; views: history, moving_condition, sampled_time, neuron.',
                pattern='Separate min–max normalization for each history/time map; flat maps gray; removes amplitudes and offsets.',
                correlation='Pearson map correlation across heading/body pairs; null for a flat map; ignores offsets and positive scale.',
                peaks='Sampled grid maxima with tied-point counts; omitted for flat maps. Peak changes alone do not establish remapping.',
                interpretation='History affects initial state, not weights. Shape changes can reflect recurrence and rectification. This does not isolate home-vector memory from other history effects.',
                starting_delta_caveat='Subtracting one starting rate per neuron shifts every grid point equally, so raw and starting-delta maps have identical shape at any fixed history/time.'))
        write_json(output / 'history_tuning.json', report)
        parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>History-dependent tuning probes</title>',
            '<style>body{font:16px sans-serif;max-width:1250px;margin:2em auto;padding:1em}img{max-width:100%}'
            'table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:.4em}</style>',
            f'<h1>History-dependent tuning: update {report["checkpoint_update"]}</h1>',
            f'<p>Each history lasts {args.history_seconds:g} s at heading {args.heading % 360:g}°. '
            f'Walking histories use {args.history_speed:g} mm/s; probes use {args.speed:g} mm/s. '
            'Every probe starts from an independent copy of its complete history endpoint state. No refitting.</p>',
            '<p>Columns compare histories; rows compare time since the probe began. '
            'Raw and difference figures retain shared scales across histories/times within each figure. '
            'Normalized movement-response figures remove amplitudes to display shape. '
            'Rates are model units, not Hz. This is a history comparison, not a test of physiological realism or task-specific emergence.</p>',
            '<p>Change-from-start subtracts a constant across each map; it cannot by itself change tuning shape. '
            'Movement-minus-stationary additionally removes the matched heading-only continuation from the same saved state.</p>',
            '<p><a href="history_endpoints.csv">History endpoints</a> · <a href="tuning_metrics.csv">All-neuron tuning metrics</a> · '
            '<a href="history_comparisons.csv">Pairwise map comparisons</a> · '
            '<a href="history_tuning.json">Definitions and metadata</a> · '
            '<a href="probe_responses.npz">Full probe traces and sampled response views</a></p>']
        for body in args.focus:
            parts.append(f'<h2>Body {body}</h2>')
            for plot in plots:
                if plot['body_id'] == body:
                    stem = plot['stem']
                    parts.append(f'<h3>{html.escape(LABELS[plot["view"]])}' + (' — normalized pattern' if plot['pattern'] else '') + '</h3>'
                                 f'<a href="{stem}.pdf">PDF</a><img loading="lazy" src="{stem}.png" alt="History tuning maps">')
        (output / 'index.html').write_text('\n'.join(parts) + '</html>\n')
        if any(digest(path) != sha for path, sha in frozen.items()):
            raise ValueError('An immutable input bundle changed during analysis.')
        finish(output, manifest)
        return report
    except BaseException as error:
        fail(output, manifest, error)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--snapshot-key', choices=['current', 'reference'], default='current')
    parser.add_argument('--history-seconds', type=float, default=4.)
    parser.add_argument('--heading', type=float, default=0.)
    parser.add_argument('--history-speed', type=float, default=1.)
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--angles', type=int, default=8)
    parser.add_argument('--times', type=float, nargs='+', default=[.0025, .05, .25])
    parser.add_argument('--focus', type=int, nargs='+', default=[142911, 17816, 20231])
    parser.add_argument('--chunk-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=1)
    return analyze(parser.parse_args(argv))


if __name__ == '__main__':
    main()
