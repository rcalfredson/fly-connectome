"""Frozen-model contribution sweeps, neuron interventions and pause robustness."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..__main__ import batch, begin, fail, finish
from ..data import Circuit, digest, write_json, verify_bundle
from .activity import write_csv
from .checkpoint import capture


def decoder_spaces(rates, decoder):
    """Project changes from the first state into the fixed D row/null spaces."""
    decoder = np.asarray(decoder, dtype=np.float64)
    _, singular, vh = np.linalg.svd(decoder, full_matrices=False)
    tolerance = (singular[0] if len(singular) else 0.) * max(decoder.shape) * np.finfo(float).eps
    rank = int(np.sum(singular > tolerance))
    delta = np.asarray(rates, dtype=np.float64) - rates[0]
    visible = (delta @ vh[:rank].T) @ vh[:rank]
    hidden = delta - visible
    np.testing.assert_allclose(visible + hidden, delta, atol=1e-10)
    np.testing.assert_allclose(hidden @ decoder.T, 0., atol=1e-9)
    np.testing.assert_allclose(np.sum(visible * hidden, axis=1), 0., atol=1e-8)
    return dict(state_change_norm=np.linalg.norm(delta, axis=1),
                visible_change_norm=np.linalg.norm(visible, axis=1),
                hidden_change_norm=np.linalg.norm(hidden, axis=1),
                singular_values=singular, rank=rank, rank_tolerance=tolerance)


def assess(prediction, target, rates, dt, baseline=None):
    bad = (~np.isfinite(prediction).all(axis=1) | ~np.isfinite(rates).all(axis=1)
           | (np.max(np.abs(rates), axis=1) > 1e6))
    if bad.any():
        return dict(status='numerical_guard_failed', first_invalid_s=float(np.flatnonzero(bad)[0] * dt),
                    vector_rmse_mm=None, endpoint_error_mm=None, prediction_change_rms_mm=None)
    return dict(status='finite', first_invalid_s=None,
                vector_rmse_mm=float(np.sqrt(np.mean(np.sum((prediction - target) ** 2, axis=1)))),
                endpoint_error_mm=float(np.linalg.norm(prediction[-1] - target[-1])),
                prediction_change_rms_mm=(None if baseline is None else
                    float(np.sqrt(np.mean(np.sum((prediction - baseline) ** 2, axis=1))))))


def sweep_records(dt, distance, speeds, angles):
    rows = []
    for speed in speeds:
        duration = distance / speed
        if not np.isclose(duration / dt, round(duration / dt), rtol=0, atol=1e-7):
            raise ValueError('Straight-path duration must fall on the simulation grid.')
        for heading in np.arange(angles) * 360. / angles:
            for direction in np.arange(angles) * 360. / angles:
                forward, right = speed * np.array([np.cos(np.deg2rad(direction)), np.sin(np.deg2rad(direction))])
                rows.append(dict(family='activity-straight-sweep', kind='straight',
                    speed_mm_s=float(speed), heading_deg=float(heading), body_direction_deg=float(direction),
                    distance_mm=float(distance), protocol=[[0., float(heading), float(forward), float(right)],
                        [float(duration), float(heading), 0., 0.]]))
    return rows


def contribution_sweep(snap, output, args):
    circuit, network, dt = Circuit(**snap['circuit']), snap['network'], snap['task']['dt_s']
    rows = sweep_records(dt, args.distance_mm, args.speeds, args.angles)
    write_json(output / 'sweep_protocols.json', rows)
    decoder = network.D.detach().numpy()
    decoder_norm = np.linalg.norm(decoder.astype(float), axis=0)
    neuron_rows, conditions = [], []
    # Each speed has equal-duration trajectories; chunk without cross-speed padding.
    for speed in args.speeds:
        selected = [row for row in rows if row['speed_mm_s'] == speed]
        for start in range(0, len(selected), args.chunk_size):
            records = selected[start:start + args.chunk_size]
            with torch.no_grad():
                x, y, _ = batch(circuit, records, dt, 'cpu', network.a.dtype)
                prediction, rates = network(x, dt)
            for k, row in enumerate(records):
                r, p, target = rates[k].numpy(), prediction[k].numpy(), y[k].numpy()
                score = assess(p, target, r, dt)
                if score['status'] != 'finite':
                    raise ValueError('Unperturbed sweep exceeded numerical guard.')
                # ||D_j r_j(t)|| RMS; no contribution can be a fraction of net output
                # when opposing contributions cancel. The denominator is explicitly
                # the sum of individual RMS magnitudes, not explained variance.
                rms = np.sqrt(np.mean(r.astype(float) ** 2, axis=0)) * decoder_norm
                shares = rms / rms.sum() if rms.sum() else np.zeros_like(rms)
                order = np.argsort(-rms, kind='stable')
                ranks = np.empty(len(order), dtype=int)
                ranks[order] = np.arange(1, len(order) + 1)
                condition = dict(condition=len(conditions), speed_mm_s=row['speed_mm_s'],
                    heading_deg=row['heading_deg'], body_direction_deg=row['body_direction_deg'])
                conditions.append(dict(**condition, **score, largest_rms_share=float(shares.max()),
                    top_body_id=circuit.dynamic_ids[order[0]] if rms.sum() else None))
                # Slope with accumulated distance, over the last quarter of the path.
                distance = np.arange(len(r)) * dt * speed
                last = max(2, int(np.ceil(len(r) / 4)))
                centered = distance[-last:] - distance[-last:].mean()
                slopes = centered @ r[-last:].astype(float) / np.sum(centered ** 2)
                for j, body_id in enumerate(circuit.dynamic_ids):
                    neuron_rows.append(dict(**condition, body_id=body_id, cell_type=circuit.dynamic_types[j],
                        direct_contribution_rms_mm=float(rms[j]), share_of_summed_rms=float(shares[j]),
                        rank=int(ranks[j]) if rms.sum() else None, rate_max=float(r[:, j].max()),
                        late_rate_slope_per_mm=float(slopes[j])))
        print(f'Contribution sweep: speed {speed} mm/s complete', flush=True)
    write_csv(output / 'sweep_conditions.csv', conditions)
    write_csv(output / 'sweep_neurons.csv', neuron_rows)
    return conditions, neuron_rows


def neuron_interventions(snap, activity, output, args):
    circuit, network, dt = Circuit(**snap['circuit']), snap['network'], snap['task']['dt_s']
    rows = []
    decoder = network.D.detach().numpy()
    for trajectory in activity['trajectories']:
        stem, record = trajectory['stem'], trajectory['record']
        with np.load(Path(args.activity) / f'{stem}.npz') as data:
            base, target, base_rates = data['prediction'].copy(), data['target'].copy(), data['rates'].copy()
        baseline = assess(base, target, base_rates, dt)
        saved_predictions = [base]
        with torch.no_grad():
            x, _, _ = batch(circuit, [record], dt, 'cpu', network.a.dtype)
            # Sham matches the intervention batch shape, catching numerical changes
            # due to batched GEMM without conflating them with a causal effect.
            for start in range(0, network.size, args.chunk_size):
                indices = list(range(start, min(start + args.chunk_size, network.size)))
                masks = torch.zeros(len(indices) + 1, network.size, dtype=torch.bool)
                masks[torch.arange(1, len(indices) + 1), indices] = True
                prediction, rates = network(x.expand(len(masks), -1, -1), dt, silence_mask=masks)
                np.testing.assert_allclose(prediction[0].numpy(), base, rtol=0, atol=.002)
                for k, j in enumerate(indices, 1):
                    direct = base - base_rates[:, j, None] * decoder[:, j]
                    for condition, p, r in [('readout_removed', direct, base_rates),
                                            ('silenced_throughout', prediction[k].numpy(), rates[k].numpy())]:
                        metrics = assess(p, target, r, dt, baseline=base)
                        rows.append(dict(trajectory=stem, family=record['family'], kind=record['kind'],
                            body_id=circuit.dynamic_ids[j], cell_type=circuit.dynamic_types[j],
                            condition=condition, baseline_vector_rmse_mm=baseline['vector_rmse_mm'],
                            **metrics))
                    saved_predictions.append(prediction[k].numpy().copy())
        np.savez_compressed(output / f'{stem}_silencing.npz',
            prediction=np.stack(saved_predictions), target=target,
            time_s=np.arange(len(base)) * dt, body_ids=np.asarray(circuit.dynamic_ids))
        print(f'Neuron interventions: {stem} complete', flush=True)
    write_csv(output / 'neuron_interventions.csv', rows)
    return rows


def perturb_states(state, fractions, repeats, seed):
    """Seeded multiplicative jitter rescaled to a requested state-norm fraction."""
    rng = np.random.default_rng(seed)
    states, metadata = [state.copy()], [dict(condition='sham', fraction=0., replicate=-1)]
    for fraction in fractions:
        for replicate in range(repeats):
            delta = state * rng.normal(size=state.shape)
            norm = np.linalg.norm(delta)
            if norm:
                delta *= fraction * np.linalg.norm(state) / norm
            perturbed = np.maximum(state + delta, 0.)
            states.append(perturbed)
            metadata.append(dict(condition='state_jitter', fraction=float(fraction), replicate=replicate))
    states.append(np.zeros_like(state))
    metadata.append(dict(condition='reset_at_pause', fraction=1., replicate=-1))
    for initial, row in zip(states, metadata):
        row['actual_relative_state_change'] = float(np.linalg.norm(initial - state) / np.linalg.norm(state)) if np.linalg.norm(state) else 0.
    return np.stack(states), metadata


def pause_checks(snap, activity, output, args):
    network, circuit, dt = snap['network'], Circuit(**snap['circuit']), snap['task']['dt_s']
    if not np.isclose(args.pause_seconds / dt, round(args.pause_seconds / dt), rtol=0, atol=1e-7):
        raise ValueError('Pause duration must fall on the simulation grid.')
    selected = [r for r in activity['trajectories'] if r['record']['family'] == 'challenge-pause-5']
    if len(selected) != 2:
        raise ValueError('Require both standard five-second pause histories.')
    table, reports = [], []
    for history_index, trajectory in enumerate(selected):
        record, stem = trajectory['record'], trajectory['stem']
        onset = round(record['suffix_start_s'] / dt)
        with np.load(Path(args.activity) / f'{stem}.npz') as data:
            initial = data['rates'][onset].copy()
            true_home = data['target'][onset].copy()
            old_prediction = data['prediction'][onset:].copy()
        heading = record['protocol'][1][1]
        protocol = dict(protocol=[[0., heading, 0., 0.], [args.pause_seconds, heading, 0., 0.]])
        states, metadata = perturb_states(initial, args.jitter_fractions, args.repeats, args.seed + history_index)
        predictions, paired_differences, state_norms, variant_metrics = [], [], [], []
        sham_predictions = []
        with torch.no_grad():
            x, _, _ = batch(circuit, [protocol], dt, 'cpu', network.a.dtype)
            truth = np.broadcast_to(true_home, (x.shape[1], 2))
            for start in range(0, len(states), args.chunk_size):
                selected_states = states[start:start + args.chunk_size]
                # A matched sham in every chunk separates state-perturbation
                # effects from changes in floating-point batch execution.
                initials = torch.as_tensor(np.concatenate([states[:1], selected_states]), dtype=network.a.dtype)
                p, r = network(x.expand(len(initials), -1, -1), dt, initial=initials)
                sham_predictions.append(p[0].numpy().copy())
                for k in range(len(selected_states)):
                    predictions.append(p[k + 1].numpy().copy())
                    paired_differences.append((p[k + 1] - p[0]).numpy().copy())
                    state_norms.append(np.linalg.norm(r[k + 1].numpy().astype(float), axis=1))
                    variant_metrics.append(assess(predictions[-1], truth, r[k + 1].numpy(), dt))
                    if start + k == 0:
                        baseline_rates = r[k + 1].numpy().copy()
        baseline = predictions[0]
        sham_difference = max(float(np.max(np.abs(p - baseline))) for p in sham_predictions)
        if sham_difference >= .002:
            raise ValueError('Pause shams differ materially across batch shapes.')
        overlap = min(len(baseline), len(old_prediction))
        np.testing.assert_allclose(baseline[:overlap], old_prediction[:overlap], rtol=0, atol=.002)
        if variant_metrics[0]['status'] != 'finite':
            raise ValueError('Unperturbed long pause exceeded numerical guard; shorten --pause-seconds.')
        spaces = decoder_spaces(baseline_rates, network.D.detach().numpy())
        np.testing.assert_allclose((baseline_rates - baseline_rates[0]).astype(float) @ network.D.detach().numpy().T,
                                   baseline - baseline[0], atol=2e-5, rtol=1e-5)
        for k, (p, row) in enumerate(zip(predictions, metadata)):
            score = variant_metrics[k]
            finite = score['status'] == 'finite'
            table.append(dict(trajectory=stem, history='backward' if record['protocol'][0][2] < 0 else 'forward',
                **row, **score, initial_output_shift_mm=float(np.linalg.norm(paired_differences[k][0])),
                endpoint_difference_from_sham_mm=float(np.linalg.norm(paired_differences[k][-1])) if finite else None,
                max_difference_from_sham_mm=float(np.max(np.linalg.norm(paired_differences[k], axis=1))) if finite else None))
        time = np.arange(len(baseline)) * dt
        np.savez_compressed(output / f'{stem}_pause.npz', time_s=time, rates=baseline_rates,
            prediction=baseline, target=truth, variant_prediction=np.stack(predictions),
            variant_difference_from_sham=np.stack(paired_differences),
            variant_initial_state=states, variant_state_norm=np.stack(state_norms),
            **{key: value for key, value in spaces.items() if key.endswith('_norm')})
        checkpoints = []
        for seconds in [0., 1., 5., 10., 30., 60.]:
            if seconds > args.pause_seconds:
                continue
            t = round(seconds / dt)
            checkpoints.append(dict(pause_time_s=seconds,
                target_error_mm=float(np.linalg.norm(baseline[t] - truth[t])),
                output_drift_mm=float(np.linalg.norm(baseline[t] - baseline[0])),
                **{key: float(value[t]) for key, value in spaces.items() if key.endswith('_norm')}))
        reports.append(dict(trajectory=stem, history=table[-1]['history'], variants=metadata,
            sham_max_abs_prediction_difference_mm=sham_difference,
            checkpoints=checkpoints, decoder_rank=spaces['rank'],
            decoder_singular_values=spaces['singular_values'].tolist(),
            rank_tolerance=float(spaces['rank_tolerance'])))
        print(f'Pause checks: {stem}, {args.pause_seconds} seconds complete', flush=True)
    write_csv(output / 'pause_interventions.csv', table)
    write_json(output / 'pause_details.json', reports)
    return table, reports


def analyze(args):
    if (args.threads < 1 or args.chunk_size < 1 or args.angles < 2 or args.repeats < 1
            or not np.isfinite([args.distance_mm, args.pause_seconds, *args.speeds, *args.jitter_fractions]).all()
            or args.distance_mm <= 0 or args.pause_seconds <= 0 or min(args.speeds) <= 0
            or min(args.jitter_fractions) <= 0 or len(set(args.speeds)) != len(args.speeds)):
        raise ValueError('Require positive finite probe settings and distinct speeds.')
    source, activity, output = [Path(getattr(args, k)).resolve() for k in ('source', 'activity', 'output')]
    for path in (source, activity):
        if output == path or output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError('Output must be separate from input bundles.')
    if output.exists():
        raise FileExistsError(output)
    if verify_bundle(activity)['kind'] != 'home_vector_activity_v1':
        raise ValueError('Require a completed activity recorder bundle.')
    args.activity = str(activity)
    torch.set_num_threads(args.threads)
    snap = capture(activity, source)
    atlas = json.loads((activity / 'activity.json').read_text())
    if atlas['checkpoint_sha256'] != snap['checkpoint_sha256']:
        raise ValueError('Activity traces do not match the archived checkpoint.')
    frozen = {str(p): digest(p) for folder in (source, activity) for p in folder.iterdir() if p.is_file()}
    output, manifest = begin(output, 'home_vector_activity_checks_v1', source=str(source),
        source_manifest_sha256=digest(source / 'manifest.json'), activity=str(activity),
        activity_manifest_sha256=digest(activity / 'manifest.json'))
    manifest.update(device='cpu', torch=torch.__version__, refitting=False, threads=args.threads,
        immutable_input_sha256=frozen,
        analysis_code_sha256={p.name: digest(p) for p in Path(__file__).parent.glob('*.py')})
    write_json(output / 'manifest.json', manifest)
    try:
        for name in ('current_checkpoint.pt', 'current_metadata.json', 'neurons.csv'):
            (output / name).write_bytes((activity / name).read_bytes())
        conditions, neurons = contribution_sweep(snap, output, args)
        interventions = neuron_interventions(snap, atlas, output, args)
        pauses, pause_reports = pause_checks(snap, atlas, output, args)
        report = dict(checkpoint_update=snap['checkpoint']['update'], checkpoint_sha256=snap['checkpoint_sha256'],
            settings=vars(args), sweep_conditions=len(conditions), neuron_interventions=len(interventions),
            pause_variants=len(pauses), numerical_guard_failures=sum(r['status'] != 'finite' for r in interventions + pauses),
            definitions=dict(contribution='RMS norm of D[:,j] * rate_j; shares divide by summed individual RMS, not net output or explained variance.',
                readout_removed='Subtract one observed neuron contribution; neural trajectory is untouched.',
                silenced_throughout='Clamp one unit to zero at every recorded state for the whole trajectory, including before recurrent propagation.',
                causal_scope='Ablations alter encoding and retention together; they are not pause-specific necessity tests.',
                jitter='One-time multiplicative Gaussian jitter at pause onset, rescaled to requested fraction of initial state norm; nonnegative clipping recorded in actual fraction.',
                pause='Pauses continue with original heading-dependent stationary PFN input; no ongoing injected noise.',
                generalization='Developmental probes on one frozen seed, not physiological validation.'))
        write_json(output / 'diagnostics.json', report)
        from .activity_checks_plots import make_plots
        make_plots(output, conditions, neurons, interventions, pauses, pause_reports, report)
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
    parser.add_argument('--distance-mm', type=float, default=8.)
    parser.add_argument('--speeds', type=float, nargs='+', default=[.5, 1., 2.])
    parser.add_argument('--pause-seconds', type=float, default=60.)
    parser.add_argument('--jitter-fractions', type=float, nargs='+', default=[.001, .01, .05])
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--focus-body-id', default='142911')
    return analyze(parser.parse_args(argv))


if __name__ == '__main__':
    main()
