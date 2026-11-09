"""Replay identical probes from selected moments of archived trajectories."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ..__main__ import begin, fail, finish
from ..data import Circuit, digest, encode, verify_bundle, write_json
from .activity import record_trajectory, write_csv
from .activity_paths import resume_records
from .checkpoint import capture
from .history_tuning import VIEWS, compare_maps, map_stats, probe_from_state, response_views


def default_cases():
    """Five preselected questions, including early and late versions of two pairs."""
    return [
        dict(case='case_1', title='Same world path, identical final inputs',
             pairs=[[[6, 6.], [7, 6.]], [[6, 7.5], [7, 7.5]]], focus=[142911, 17816, 20231]),
        dict(case='case_2', title='Matched curved world path, forward versus lateral',
             pairs=[[[12, 2.], [13, 2.]], [[12, 7.5], [13, 7.5]]], focus=[142911, 17816]),
        dict(case='case_3', title='Repeated input within one trajectory',
             pairs=[[[6, 1.5], [6, 7.5]]], focus=[17816, 142911, 20231]),
        dict(case='case_4', title='Matched mixed walk with hDeltaC activity contrast',
             pairs=[[[2, 4.], [3, 4.]]], focus=[20231, 142911]),
        dict(case='case_5', title='Active versus nearly silent states',
             pairs=[[[12, 4.], [13, 4.]]], focus=[20951, 59176]),
    ]


def state_key(path, seconds):
    return f'path_{path:02d}_at_{round(seconds * 1e6):08d}us'


def effect_metrics(a, b, conditions):
    """Descriptive offset, amplitude and shape; flat maps have no fitted gain."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    sa, sb = map_stats(a, conditions), map_stats(b, conditions)
    ac, bc = a - a.mean(), b - b.mean()
    fit = not sa['flat']
    gain = float(ac @ bc / (ac @ ac)) if fit else None
    offset = float(b.mean() - gain * a.mean()) if fit else None
    residual = float(np.sqrt(np.mean((bc - gain * ac) ** 2))) if fit else None
    result = dict(**compare_maps(a, b), mean_a=float(a.mean()), mean_b=float(b.mean()),
        range_a=sa['modulation_range'], range_b=sb['modulation_range'],
        range_ratio_b_over_a=None if sa['flat'] else sb['modulation_range'] / sa['modulation_range'],
        fitted_gain_b_from_a=gain, fitted_offset_b_from_a=offset, affine_residual_rms=residual,
        residual_fraction_of_b_range=None if not fit or sb['flat'] else residual / sb['modulation_range'],
        peak_heading_a=sa['peak_heading_deg'], peak_body_direction_a=sa['peak_body_direction_deg'],
        peak_ties_a=sa['peak_tied_points'], peak_heading_b=sb['peak_heading_deg'],
        peak_body_direction_b=sb['peak_body_direction_deg'], peak_ties_b=sb['peak_tied_points'])
    # A concrete preference contrast at one common heading. A sign change
    # smaller than 1% of the larger modulation range is left indeterminate.
    right = next((i for i, row in enumerate(conditions)
                  if row['heading_deg'] == 90. and row['body_direction_deg'] == 90.), None)
    left = next((i for i, row in enumerate(conditions)
                 if row['heading_deg'] == 90. and row['body_direction_deg'] == 270.), None)
    da, db = (float(a[right] - a[left]), float(b[right] - b[left])) if right is not None and left is not None else (None, None)
    tolerance = max(1e-6, .01 * max(sa['modulation_range'], sb['modulation_range']))
    result.update(right_minus_left_at_heading90_a=da, right_minus_left_at_heading90_b=db,
        right_left_order_reversed=None if da is None or min(abs(da), abs(db)) <= tolerance else bool(da * db < 0),
        right_left_contrast_tolerance=tolerance)
    return result


def validate_cases(cases, circuit, path_rows, folder, dt):
    """Validate every requested state before creating outputs or probing anything."""
    if not cases or len({row['case'] for row in cases}) != len(cases):
        raise ValueError('Require distinct, nonempty cases.')
    states, loaded = {}, {}
    for case in cases:
        if not case['pairs'] or not case['focus'] or len(set(case['focus'])) != len(case['focus']):
            raise ValueError('Each case needs state pairs and distinct focus neurons.')
        if any(body not in circuit.dynamic_ids for body in case['focus']):
            raise ValueError('Focus neuron missing from archived circuit.')
        for pair in case['pairs']:
            if len(pair) != 2:
                raise ValueError('Each comparison must have exactly two saved states.')
            for path, seconds in pair:
                if not isinstance(path, int) or path < 0 or not np.isfinite(seconds) or seconds < 0:
                    raise ValueError('Invalid path number or saved-state time.')
                stem = f'path_{path:02d}'
                if stem not in path_rows:
                    raise ValueError(f'Unknown path {stem}.')
                if stem not in loaded:
                    with np.load(folder / f'{stem}.npz') as saved:
                        loaded[stem] = {key: saved[key].copy() for key in saved.files}
                    arrays = loaded[stem]
                    if (arrays['rates'].shape != (len(arrays['time_s']), len(circuit.dynamic_ids))
                            or not np.isfinite(arrays['rates']).all()
                            or np.min(arrays['rates']) < 0 or np.max(arrays['rates']) > 1e6):
                        raise ValueError('Invalid archived rates.')
                    np.testing.assert_allclose(arrays['time_s'], np.arange(len(arrays['time_s'])) * dt, atol=1e-10, rtol=0)
                tick = round(seconds / dt)
                if not np.isclose(seconds / dt, tick, atol=1e-8, rtol=0) or tick >= len(loaded[stem]['time_s']):
                    raise ValueError('State time must lie within the archived simulation time grid.')
                key = state_key(path, seconds)
                states[key] = dict(key=key, path=stem, time_s=float(seconds), tick=tick)
    return states, loaded


def analyze(args):
    activity, source, output = [Path(getattr(args, key)).resolve() for key in ('activity', 'source', 'output')]
    if args.threads < 1 or args.chunk_size < 1 or args.angles < 2:
        raise ValueError('Require positive thread/chunk counts and at least two angles.')
    if not args.times or not all(np.isfinite(t) and t > 0 for t in args.times) or not np.isfinite(args.speed) or args.speed <= 0:
        raise ValueError('Probe times and speed must be positive and finite.')
    for folder in (activity, source):
        if output == folder or output.is_relative_to(folder) or folder.is_relative_to(output):
            raise ValueError('Output must be separate from input bundles.')
    if output.exists():
        raise FileExistsError(output)
    if verify_bundle(activity)['kind'] != 'home_vector_activity_paths_v1':
        raise ValueError('Require an archived per-trajectory activity bundle.')
    torch.set_num_threads(args.threads)
    snap = capture(activity, source)
    circuit, network, dt = Circuit(**snap['circuit']), snap['network'], snap['task']['dt_s']
    if any(not np.isclose(t / dt, round(t / dt), atol=1e-8, rtol=0) for t in args.times):
        raise ValueError('Probe times must lie on the simulation time grid.')
    ticks = sorted({round(t / dt) for t in args.times})
    times = np.array(ticks) * dt
    cases = json.loads(Path(args.cases).read_text()) if args.cases else default_cases()
    path_rows = {row['path']: row for row in json.loads((activity / 'raw_paths.json').read_text())}
    states, loaded = validate_cases(cases, circuit, path_rows, activity, dt)
    with (activity / 'neurons.csv').open() as stream:
        identities = list(csv.DictReader(stream))
    if ([int(row['body_id']) for row in identities] != circuit.dynamic_ids
            or [row['cell_type'] for row in identities] != circuit.dynamic_types):
        raise ValueError('Archived neuron order differs from checkpoint.')
    # Replay each used path once to bind all selected states to the exact
    # archived checkpoint and input protocol, not merely to matching dimensions.
    replay_checks = []
    for stem, arrays in loaded.items():
        replay, _ = record_trajectory(network, circuit, path_rows[stem]['record'], dt)
        for key in ('controls', 'pfn', 'rates', 'prediction', 'target'):
            np.testing.assert_allclose(replay[key], arrays[key], atol=3e-5, rtol=1e-6)
        replay_checks.append(dict(path=stem, rate_replay_max_abs_error=float(np.max(abs(replay['rates'] - arrays['rates'])))))
    frozen = {str(p): digest(p) for folder in (activity, source) for p in folder.iterdir() if p.is_file()}
    output, manifest = begin(output, 'home_vector_trajectory_tuning_v1', activity=str(activity),
        activity_manifest_sha256=digest(activity / 'manifest.json'), source=str(source),
        source_manifest_sha256=digest(source / 'manifest.json'), speed_mm_s=args.speed,
        angles=args.angles, sample_times_s=times.tolist(), cases=cases)
    manifest.update(device='cpu', threads=args.threads, refitting=False, torch=torch.__version__,
        immutable_input_sha256=frozen,
        analysis_code_sha256={p.name: digest(p) for p in Path(__file__).parent.glob('*.py')})
    write_json(output / 'manifest.json', manifest)
    try:
        for name in ('current_checkpoint.pt', 'current_metadata.json', 'neurons.csv', 'raw_paths.json'):
            (output / name).write_bytes((activity / name).read_bytes())
        write_json(output / 'cases.json', cases)
        records = resume_records(float(times[-1]), args.angles, [args.speed], 0.)
        write_json(output / 'probe_conditions.json', records)
        endpoints, responses = [], {}
        for key, state_info in states.items():
            arrays, tick = loaded[state_info['path']], state_info['tick']
            state = arrays['rates'][tick].copy()
            prediction, rates = probe_from_state(network, circuit, records, state, dt, args.chunk_size)
            views, moving, controls = response_views(rates, state, records)
            responses[key] = {view: value[:, ticks] for view, value in views.items()}
            np.savez_compressed(output / f'{key}.npz', rates=rates, prediction=prediction,
                initial_state=state, time_s=np.arange(rates.shape[1]) * dt, sample_ticks=ticks,
                moving_indices=moving, stationary_indices=controls,
                pfn_at_onset=encode(circuit, np.array([r['protocol'][0][1:] for r in records])),
                **responses[key])
            endpoints.append(dict(**state_info, state_norm=float(np.linalg.norm(state)),
                heading_deg=float(arrays['controls'][tick, 0]), forward_mm_s=float(arrays['controls'][tick, 1]),
                rightward_mm_s=float(arrays['controls'][tick, 2]),
                home_x_mm=float(arrays['target'][tick, 0]), home_y_mm=float(arrays['target'][tick, 1])))
            print(f'Trajectory probes: {len(endpoints)}/{len(states)} {key}', flush=True)
        selected = [records[k] for k in moving]
        comparisons, summaries, pair_checks, plots = [], [], [], []
        from .trajectory_tuning_plots import plot_pair, write_report
        for case in cases:
            for pair_number, pair in enumerate(case['pairs']):
                a, b = [state_key(*item) for item in pair]
                aa, bb = [loaded[states[key]['path']] for key in (a, b)]
                ta, tb = states[a]['tick'], states[b]['tick']
                pair_id = f'{case["case"]}_pair_{pair_number + 1}'
                pair_checks.append(dict(case=case['case'], pair_id=pair_id, state_a=a, state_b=b,
                    identical_current_pfn=bool(np.array_equal(aa['pfn'][ta], bb['pfn'][tb])),
                    target_vector_difference_mm=float(np.linalg.norm(aa['target'][ta] - bb['target'][tb])),
                    initial_state_difference_norm=float(np.linalg.norm(aa['rates'][ta] - bb['rates'][tb]))))
                for t, seconds in enumerate(times):
                    for j, body in enumerate(circuit.dynamic_ids):
                        row = dict(case=case['case'], pair_id=pair_id, state_a=a, state_b=b,
                            body_id=body, cell_type=circuit.dynamic_types[j], time_s=float(seconds),
                            initial_rate_a=float(aa['rates'][ta, j]), initial_rate_b=float(bb['rates'][tb, j]))
                        effects = {}
                        for view in VIEWS:
                            effects[view] = effect_metrics(responses[a][view][:, t, j], responses[b][view][:, t, j], selected)
                            comparisons.append(dict(**row, view=view, **effects[view]))
                        if body in case['focus']:
                            summaries.append(dict(**row,
                                raw_mean_shift=effects['raw']['mean_shift'],
                                movement_range_a=effects['movement_delta']['range_a'],
                                movement_range_b=effects['movement_delta']['range_b'],
                                movement_range_ratio=effects['movement_delta']['range_ratio_b_over_a'],
                                movement_pattern_correlation=effects['movement_delta']['pattern_correlation'],
                                movement_fitted_gain=effects['movement_delta']['fitted_gain_b_from_a'],
                                movement_affine_residual_rms=effects['movement_delta']['affine_residual_rms'],
                                movement_residual_fraction_of_b_range=effects['movement_delta']['residual_fraction_of_b_range'],
                                right_minus_left_a=effects['movement_delta']['right_minus_left_at_heading90_a'],
                                right_minus_left_b=effects['movement_delta']['right_minus_left_at_heading90_b'],
                                right_left_order_reversed=effects['movement_delta']['right_left_order_reversed']))
                for body in case['focus']:
                    j = circuit.dynamic_ids.index(body)
                    for view in ('raw', 'movement_delta'):
                        stem = plot_pair(output, pair_id, body, circuit.dynamic_types[j],
                            responses[a][view][:, :, j], responses[b][view][:, :, j],
                            selected, times, a, b, view)
                        plots.append(dict(case=case['case'], pair_id=pair_id, body_id=body, view=view, stem=stem))
        write_csv(output / 'all_neuron_comparisons.csv', comparisons)
        write_csv(output / 'focus_summary.csv', summaries)
        write_csv(output / 'state_endpoints.csv', endpoints)
        write_csv(output / 'pair_checks.csv', pair_checks)
        report = dict(checkpoint_update=snap['checkpoint']['update'], checkpoint_sha256=snap['checkpoint_sha256'],
            dt_s=dt, cases=cases, states=endpoints, pairs=pair_checks, plots=plots,
            replay_verification=replay_checks, moving_conditions=len(moving), stationary_controls=len(records) - len(moving),
            speed_mm_s=args.speed, sample_times_s=times.tolist(),
            definitions=dict(raw='Actual activity; starting_delta subtracts the saved starting rate; movement_delta subtracts same-heading stationary continuation from the same state.',
                gain='Least-squares B = offset + gain*A across all heading/body grid points. Negative gain indicates inversion, not unchanged positive tuning.',
                residual='RMS error after fitted offset and gain; fractional residual divides by the B modulation range. Undefined if A or B is flat as appropriate.',
                correlation='Pearson correlation ignores additive offset and positive gain. Undefined if either map range <= 1e-6.',
                right_left='Rightward-minus-leftward movement response at heading 90 degrees. Reversal reported only if both contrasts exceed max(1e-6,1% of the larger map range); descriptive threshold, not significance.',
                peaks='Discrete grid maxima, omitted for flat maps; tied counts retained. Peak switches alone do not establish remapping.',
                timing='Saved state at t precedes input at t. Independent copies seed every movement and stationary probe. Abrupt heading changes are matched in controls.',
                limits='One trained checkpoint, one probe speed and selected histories; no causal attribution to training, physiology, or task-specific memory. Nonlinear gating can reshape responses without changing weights.',
                array_order='Each state NPZ: rates/prediction [probe_condition,time,neuron/xy]; views [moving_condition,sampled_time,neuron]; neuron order from neurons.csv.'))
        write_json(output / 'trajectory_tuning.json', report)
        write_report(output, report, summaries)
        if any(digest(path) != sha for path, sha in frozen.items()):
            raise ValueError('An input bundle changed during analysis.')
        finish(output, manifest)
        return report
    except BaseException as error:
        fail(output, manifest, error)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activity', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cases', help='JSON list in the default_cases() format; otherwise use the five planned cases.')
    parser.add_argument('--times', nargs='+', type=float, default=[.0025, .05, .25, 1.])
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--angles', type=int, default=8)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--chunk-size', type=int, default=32)
    return analyze(parser.parse_args(argv))


if __name__ == '__main__':
    main()
