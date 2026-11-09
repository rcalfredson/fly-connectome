"""Signed presynaptic drive and rate-update accounting for selected histories."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ..__main__ import begin, fail, finish
from ..data import Circuit, digest, verify_bundle, write_json
from .activity import record_trajectory, write_csv
from .checkpoint import capture
from .trajectory_tuning import state_key, validate_cases


def decompose(network, pfn, rates, targets, dt):
    """Exact model algebra in double precision, with signed drives before ReLU.

    pfn and rates may be time/neuron or batch/time/neuron; their leading
    dimensions must broadcast. No weights or activity are changed.
    """
    pfn, rates = np.asarray(pfn, dtype=float), np.asarray(rates, dtype=float)
    with torch.no_grad():
        weight = network.weights().numpy().astype(float)
        types = network.type_index.numpy()
        bias = network.bias.numpy().astype(float)[types][targets]
        tau = network.tau.numpy().astype(float)[types][targets]
    wi, wr = weight[:network.input_size, targets], weight[network.input_size:, targets]
    selected = rates[..., targets]
    input_drive = np.broadcast_to(pfn @ wi, selected.shape)
    recurrent_drive = rates @ wr
    z = bias + input_drive + recurrent_drive
    target = np.maximum(z, 0.)
    alpha = -np.expm1(-dt / tau)
    populations = [rates[..., types == k] @ wr[types == k]
                   for k in range(len(network.population_types))]
    return dict(rate=selected, input_drive=input_drive, recurrent_drive=recurrent_drive,
        recurrent_by_population=np.stack(populations, axis=-2), preactivation=z,
        rectified_target=target, target_increment=alpha * target,
        own_state_increment=-alpha * selected, increment=alpha * (target - selected)), wr


def verify_updates(arrays, parts, targets):
    expected = np.asarray(arrays['rates'], dtype=float)[..., 1:, targets]
    reconstructed = parts['rate'][..., :-1, :] + parts['increment'][..., :-1, :]
    np.testing.assert_allclose(reconstructed, expected, rtol=2e-6, atol=4e-5)
    return float(np.max(abs(reconstructed - expected)))


def analyze(args):
    activity, probes, source, output = [Path(getattr(args, k)).resolve()
                                        for k in ('activity', 'probes', 'source', 'output')]
    if args.threads < 1:
        raise ValueError('Require positive CPU thread count.')
    for folder in (activity, probes, source):
        if output == folder or output.is_relative_to(folder) or folder.is_relative_to(output):
            raise ValueError('Output must be separate from every input bundle.')
    if output.exists():
        raise FileExistsError(output)
    if verify_bundle(activity)['kind'] != 'home_vector_activity_paths_v1':
        raise ValueError('Require archived per-trajectory activity.')
    if verify_bundle(probes)['kind'] != 'home_vector_trajectory_tuning_v1':
        raise ValueError('Require completed selected-trajectory probes.')
    report = json.loads((probes / 'trajectory_tuning.json').read_text())
    snap = capture(activity, source)
    if report['checkpoint_sha256'] != snap['checkpoint_sha256'] or digest(probes / 'current_checkpoint.pt') != snap['checkpoint_sha256']:
        raise ValueError('Trajectory and probe checkpoint identities differ.')
    torch.set_num_threads(args.threads)
    network, circuit, dt = snap['network'], Circuit(**snap['circuit']), snap['task']['dt_s']
    if report['dt_s'] != dt:
        raise ValueError('Probe and trajectory timesteps differ.')
    for folder in (activity, probes):
        with (folder / 'neurons.csv').open() as stream:
            neurons = list(csv.DictReader(stream))
        if [int(row['body_id']) for row in neurons] != circuit.dynamic_ids:
            raise ValueError('Neuron order differs from checkpoint.')
    cases = report['cases']
    path_rows = {row['path']: row for row in json.loads((activity / 'raw_paths.json').read_text())}
    states, paths = validate_cases(cases, circuit, path_rows, activity, dt)
    path_focus, state_focus = {}, {}
    for case in cases:
        for pair in case['pairs']:
            for path, time in pair:
                key = state_key(path, time)
                path_focus.setdefault(states[key]['path'], set()).update(case['focus'])
                state_focus.setdefault(key, set()).update(case['focus'])
    replay_checks = []
    for stem, arrays in paths.items():
        replay, _ = record_trajectory(network, circuit, path_rows[stem]['record'], dt)
        for key in ('controls', 'pfn', 'rates', 'prediction', 'target'):
            np.testing.assert_allclose(replay[key], arrays[key], atol=3e-5, rtol=1e-6)
        replay_checks.append(dict(path=stem, maximum_rate_replay_error=float(np.max(abs(replay['rates'] - arrays['rates'])))))
    frozen = {str(p): digest(p) for folder in (activity, probes, source) for p in folder.iterdir() if p.is_file()}
    output, manifest = begin(output, 'home_vector_incoming_drive_v1', activity=str(activity), probes=str(probes),
        source=str(source), source_manifest_sha256=digest(source / 'manifest.json'),
        activity_manifest_sha256=digest(activity / 'manifest.json'), probes_manifest_sha256=digest(probes / 'manifest.json'))
    manifest.update(refitting=False, device='cpu', threads=args.threads, immutable_input_sha256=frozen,
        analysis_code_sha256={p.name: digest(p) for p in Path(__file__).parent.glob('*.py')})
    write_json(output / 'manifest.json', manifest)
    try:
        for name in ('current_checkpoint.pt', 'current_metadata.json', 'neurons.csv', 'raw_paths.json'):
            (output / name).write_bytes((activity / name).read_bytes())
        for name in ('cases.json', 'probe_conditions.json'):
            (output / name).write_bytes((probes / name).read_bytes())
        from .incoming_drive_plots import plot_path, plot_pair, write_report
        natural, columns, path_checks, plots = {}, {}, [], []
        for stem, arrays in paths.items():
            targets = sorted(circuit.dynamic_ids.index(body) for body in path_focus[stem])
            parts, wr = decompose(network, arrays['pfn'], arrays['rates'], targets, dt)
            error = verify_updates(arrays, parts, targets)
            natural[stem], columns[stem] = parts, targets
            # All signed recurrent partners are retained, including self edges.
            contributions = arrays['rates'].astype(float)[..., :, None] * wr
            np.testing.assert_allclose(contributions.sum(axis=-2), parts['recurrent_drive'], atol=1e-10, rtol=1e-10)
            np.savez_compressed(output / f'{stem}_drives.npz', time_s=arrays['time_s'],
                target_indices=targets, recurrent_partner_contributions=contributions, **parts)
            path_checks.append(dict(path=stem, maximum_next_state_reconstruction_error=error))
            for col, j in enumerate(targets):
                plot = plot_path(output, stem, circuit, arrays['time_s'], parts, contributions[:, :, col], col, j,
                                 path_rows[stem]['record'])
                plots.append(dict(kind='path', path=stem, body_id=circuit.dynamic_ids[j], stem=plot))
            print(f'Incoming drive: {stem}, {len(targets)} target neurons complete', flush=True)
        endpoints, partners, common_updates, gates = [], [], [], []
        probe_parts = {}
        conditions = json.loads((probes / 'probe_conditions.json').read_text())
        for key, info in states.items():
            with np.load(probes / f'{key}.npz') as saved:
                rates, initial, pfn, ticks, moving, stationary = [saved[k].copy() for k in
                    ('rates', 'initial_state', 'pfn_at_onset', 'sample_ticks', 'moving_indices', 'stationary_indices')]
            np.testing.assert_array_equal(initial, paths[info['path']]['rates'][info['tick']])
            np.testing.assert_array_equal(rates[:, 0], np.broadcast_to(initial, rates[:, 0].shape))
            targets = sorted(circuit.dynamic_ids.index(body) for body in state_focus[key])
            # Each probe input is constant through the final update. The final
            # sample's drive is a hypothetical continuation, excluded from all
            # interval/gate summaries and from the reconstruction check.
            parts, _ = decompose(network, pfn[:, None, :], rates, targets, dt)
            error = verify_updates(dict(rates=rates), parts, targets)
            probe_parts[key] = (parts, targets)
            np.savez_compressed(output / f'{key}_probe_drives.npz',
                target_indices=targets, update_time_s=np.arange(rates.shape[1] - 1) * dt,
                **{name: value[:, :-1] for name, value in parts.items()})
            path_checks.append(dict(path=key + '_probes', maximum_next_state_reconstruction_error=error))
            for col, j in enumerate(targets):
                for tick in ticks:
                    for kind, indices in [('moving', moving), ('stationary', np.unique(stationary))]:
                        z = parts['preactivation'][indices, :tick, col]
                        gates.append(dict(state=key, body_id=circuit.dynamic_ids[j], kind=kind,
                            interval_end_s=float(tick * dt), rectified_update_fraction=float(np.mean(z <= 0.)),
                            minimum_preactivation=float(z.min()), maximum_preactivation=float(z.max())))
        for case in cases:
            for pair_number, pair in enumerate(case['pairs'], 1):
                a, b = [state_key(*item) for item in pair]
                sa, sb = states[a], states[b]
                aa, bb = paths[sa['path']], paths[sb['path']]
                ta, tb = sa['tick'], sb['tick']
                pair_id = f'{case["case"]}_pair_{pair_number}'
                for body in case['focus']:
                    j = circuit.dynamic_ids.index(body)
                    ca, cb = columns[sa['path']].index(j), columns[sb['path']].index(j)
                    pa, pb = natural[sa['path']], natural[sb['path']]
                    row = dict(case=case['case'], pair_id=pair_id, body_id=body, cell_type=circuit.dynamic_types[j],
                        state_a=a, state_b=b)
                    for name in ('rate', 'input_drive', 'recurrent_drive', 'preactivation', 'rectified_target',
                                 'target_increment', 'own_state_increment', 'increment'):
                        va, vb = float(pa[name][ta, ca]), float(pb[name][tb, cb])
                        row.update({name + '_a': va, name + '_b': vb, name + '_difference': vb - va})
                    for k, population in enumerate(circuit.population_types):
                        row[f'{population}_recurrent_difference'] = float(pb['recurrent_by_population'][tb, k, cb] - pa['recurrent_by_population'][ta, k, ca])
                    endpoints.append(row)
                    with torch.no_grad():
                        weights = network.weights()[network.input_size:, j].numpy().astype(float)
                    partner_rows = []
                    for k, weight in enumerate(weights):
                        if weight == 0:
                            continue
                        ra, rb = float(aa['rates'][ta, k]), float(bb['rates'][tb, k])
                        partner_rows.append(dict(**{n: row[n] for n in ('case', 'pair_id', 'body_id', 'cell_type')},
                            presynaptic_body_id=circuit.dynamic_ids[k], presynaptic_type=circuit.dynamic_types[k],
                            self_edge=bool(k == j), effective_weight=float(weight),
                            presynaptic_rate_a=ra, presynaptic_rate_b=rb, drive_a=weight * ra,
                            drive_b=weight * rb, drive_difference=weight * (rb - ra)))
                    partners.extend(partner_rows)
                    np.testing.assert_allclose(sum(r['drive_difference'] for r in partner_rows), row['recurrent_drive_difference'], atol=1e-10)
                    ppa, ia = probe_parts[a]
                    ppb, ib = probe_parts[b]
                    ia, ib = ia.index(j), ib.index(j)
                    for q, condition in enumerate(conditions):
                        item = dict(case=case['case'], pair_id=pair_id, body_id=body, condition=q,
                            kind=condition['kind'], heading_deg=condition['heading_deg'], body_direction_deg=condition['body_direction_deg'])
                        for name in ('rate', 'input_drive', 'recurrent_drive', 'preactivation', 'rectified_target',
                                     'target_increment', 'own_state_increment', 'increment'):
                            va, vb = float(ppa[name][q, 0, ia]), float(ppb[name][q, 0, ib])
                            item.update({name + '_a': va, name + '_b': vb, name + '_difference': vb - va})
                        np.testing.assert_allclose(item['input_drive_difference'], 0., atol=0.)
                        np.testing.assert_allclose(item['increment_difference'],
                            item['target_increment_difference'] + item['own_state_increment_difference'], atol=1e-12)
                        common_updates.append(item)
                    stem = plot_pair(output, row, partner_rows, circuit.population_types)
                    plots.append(dict(kind='pair', case=case['case'], pair_id=pair_id, body_id=body, stem=stem))
        write_csv(output / 'paired_snapshot_drives.csv', endpoints)
        # A recurrence-disabled checkpoint legitimately has no partners.
        with (output / 'presynaptic_partner_differences.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['case', 'pair_id', 'body_id', 'cell_type',
                'presynaptic_body_id', 'presynaptic_type', 'self_edge', 'effective_weight',
                'presynaptic_rate_a', 'presynaptic_rate_b', 'drive_a', 'drive_b', 'drive_difference'])
            writer.writeheader()
            writer.writerows(partners)
        write_csv(output / 'identical_probe_first_updates.csv', common_updates)
        write_csv(output / 'probe_rectification.csv', gates)
        result = dict(checkpoint_update=snap['checkpoint']['update'], checkpoint_sha256=snap['checkpoint_sha256'],
            dt_s=dt, cases=cases, plots=plots, replay_checks=replay_checks, reconstruction_checks=path_checks,
            definitions=dict(drive='Signed weighted presynaptic activity before rectification. Not a firing rate or physical current.',
                update='r_next = r + alpha*(relu(bias + PFN_drive + recurrent_drive) - r); alpha=1-exp(-dt/tau).',
                own_state='The -alpha*r term is separate from incoming drive; recurrent self-edges, if any, remain in the recurrent sum.',
                recurrence='Population totals and individual signed w*r contributions sum exactly to recurrent drive. Opposite signs can cancel.',
                probes='Identical probe inputs across states. Gate fractions count preactivation<=0 across conditions and update times, not across trials or animals.',
                arrays='Natural path arrays: time,target; populations: time,population,target; partners: time,presynaptic_neuron,target. Probe arrays: condition,update_time,target (populations before target). Target indices refer to neurons.csv.',
                inference='Exact local accounting under fixed weights, not intervention-based causal necessity or proof that training created the effect.'))
        write_json(output / 'incoming_drive.json', result)
        write_report(output, result, endpoints, partners, gates)
        if any(digest(path) != sha for path, sha in frozen.items()):
            raise ValueError('An input bundle changed during analysis.')
        finish(output, manifest)
        return result
    except BaseException as error:
        fail(output, manifest, error)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activity', required=True)
    parser.add_argument('--probes', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--threads', type=int, default=1)
    return analyze(parser.parse_args(argv))


if __name__ == '__main__':
    main()
