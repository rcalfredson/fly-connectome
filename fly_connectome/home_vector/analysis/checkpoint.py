"""Replay standard challenges from independent snapshots of active/completed runs."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..__main__ import begin, evaluate, fail, finish, load
from ..data import Circuit, digest, verify_bundle, write_json
from ..model import HomeVectorNetwork
from .compare import plot_comparison, summarize_challenges
from .predictions import independent_target
from .validation_subsets import snapshot, stable_read


def capture(run, source, snapshot_key="current"):
    if snapshot_key not in ('current', 'reference'):
        raise ValueError('Require current or reference snapshot key.')
    kind = json.loads(stable_read(run / 'manifest.json')).get('kind')
    if kind == 'home_vector_challenge_snapshot_v1':
        verify_bundle(run)
        if not (run / f'{snapshot_key}_metadata.json').exists():
            raise ValueError(f'Snapshot bundle has no saved {snapshot_key} checkpoint.')
        metadata = json.loads((run / f'{snapshot_key}_metadata.json').read_text())
        blob = (run / f'{snapshot_key}_checkpoint.pt').read_bytes()
        sha = hashlib.sha256(blob).hexdigest()
        checkpoint = torch.load(io.BytesIO(blob), weights_only=False, map_location='cpu')
        if sha != metadata['checkpoint_sha256'] or checkpoint['update'] != metadata['checkpoint_update']:
            raise ValueError('Saved checkpoint does not match its snapshot metadata.')
        snap = dict(manifest=metadata['run_manifest'], circuit=metadata['circuit'],
                    checkpoint=checkpoint, blob=blob, history=metadata['history'],
                    checkpoint_sha256=sha, training_config=metadata['training_config'],
                    input_kind='saved_challenge_snapshot', snapshot_key=snapshot_key)
    else:
        if snapshot_key != 'current':
            raise ValueError('Snapshot key applies only to a saved challenge evaluation bundle.')
        snap = snapshot(run)
        if snap['manifest']['status'] == 'complete':
            verify_bundle(run)
        snap['training_config'] = json.loads(stable_read(run / 'training_config.json'))
        snap['input_kind'] = 'training_run'
    _, groups, task = load(source)
    if snap['manifest']['settings']['source_manifest_sha256'] != digest(source / 'manifest.json'):
        raise ValueError('Checkpoint run does not match its supplied prepared source.')
    snap.update(run=run, source=source, groups=groups, task=task)
    families = {row['family'] for row in groups['challenge']}
    required = {'challenge-long-64', 'challenge-pause-5', 'challenge-square',
                'challenge-reversal', 'challenge-world-equivalent'}
    if not required <= families:
        raise ValueError('Require the standard long, pause, square, reversal and world-equivalent challenges.')
    network = HomeVectorNetwork(Circuit(**snap['circuit']), seed=snap['manifest']['settings']['seed'],
                               recurrence=snap['manifest']['settings'].get('recurrence', True))
    for key in ('pre', 'post', 'counts', 'type_index', 'a_initial'):
        torch.testing.assert_close(network.state_dict()[key], snap['checkpoint']['model'][key], atol=0, rtol=0)
    network.load_state_dict(snap['checkpoint']['model'], strict=True)
    network.eval()
    snap['network'] = network
    return snap


def validate_pair(reference, current, allow_task_differences):
    if reference['groups']['challenge'] != current['groups']['challenge']:
        raise ValueError('Challenge protocols must match exactly.')
    if any(reference['task'][key] != current['task'][key] for key in ('dt_s', 'length_scale_mm')):
        raise ValueError('Require identical time steps and loss scales.')
    if allow_task_differences:
        if reference['circuit'] != current['circuit']:
            raise ValueError('Different-task comparison requires identical circuits.')
    else:
        for name in ('trajectories.json', 'task.json'):
            if digest(reference['source'] / name) != digest(current['source'] / name):
                raise ValueError('Comparison requires identical tasks; use --allow-task-differences explicitly.')


def replay(snap, output, key):
    rows, task = snap['groups']['challenge'], snap['task']
    predictions, targets = [], []
    for index, row in enumerate(rows, 1):
        print(f'{key}: update {snap["checkpoint"]["update"]}, challenge {index}/{len(rows)}: {row["family"]}', flush=True)
        _, p, y, _ = evaluate(snap['network'], Circuit(**snap['circuit']), [row],
                             task['dt_s'], 'cpu', task['length_scale_mm'])
        expected = independent_target(row['protocol'], len(y[0]), task['dt_s'])
        np.testing.assert_allclose(y[0], expected, rtol=1e-6, atol=5e-6)
        predictions.extend(p)
        targets.extend(y)
    from ..__main__ import metrics
    result = summarize_challenges(rows, predictions, targets, task['dt_s'])
    result.update(checkpoint_update=snap['checkpoint']['update'], checkpoint_sha256=snap['checkpoint_sha256'],
                  run_status_at_snapshot=snap['manifest']['status'], aggregate=metrics(predictions, targets),
                  validation_history=[dict(update=row['update'], scaled_mse=row['validation_scaled_mse'])
                      for row in snap['history'] if 'validation_scaled_mse' in row
                      and row['update'] <= snap['checkpoint']['update']])
    result['aggregate']['equal_trajectory_scaled_mse'] = float(np.mean([
        float(np.mean(np.sum((p - y) ** 2, axis=1))) for p, y in zip(predictions, targets)]) / task['length_scale_mm'] ** 2)
    np.savez_compressed(output / f'{key}_predictions.npz', **{
        f'{kind}_{i}': value for kind, arrays in [('prediction', predictions), ('target', targets)]
        for i, value in enumerate(arrays)})
    return result, (predictions, targets)


def write_table(output, report):
    table = []
    for key, label in report['model_labels'].items():
        for item in report[key]['challenge']:
            pause, long, reversal = [item.get(k, {}) for k in ('pause', 'long_path', 'reversal')]
            table.append(dict(model=label, checkpoint_update=report[key]['checkpoint_update'],
                family=item['family'], kind=item['kind'], vector_rmse_mm=item['vector_rmse_mm'],
                x_rmse_mm=item['component_rmse_mm'][0], y_rmse_mm=item['component_rmse_mm'][1],
                terminal_error_mm=item['terminal_error_mm'],
                terminal_prediction_x_mm=item['terminal_prediction_mm'][0],
                terminal_prediction_y_mm=item['terminal_prediction_mm'][1],
                pause_onset_error_mm=pause.get('onset_error_mm'),
                pause_endpoint_drift_mm=pause.get('endpoint_drift_mm'),
                pause_final_second_rmse_mm=pause.get('final_second_vector_rmse_mm'),
                late_magnitude_growth_mm_s=long.get('late_10s_predicted_magnitude_slope_mm_s'),
                reversal_turnaround_lag_s=reversal.get('turnaround_lag_s')))
    with (output / 'challenge_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


def plot_reversal(snaps, arrays, labels, output):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, layout='constrained')
    for model_index, (key, snap) in enumerate(snaps.items()):
        i = next(i for i, row in enumerate(snap['groups']['challenge']) if row['kind'] == 'reversal')
        p, y = arrays[key][0][i], arrays[key][1][i]
        row = snap['groups']['challenge'][i]
        times = np.arange(len(p)) * snap['task']['dt_s']
        for j, component in enumerate(('x', 'y')):
            axes[j].plot(times, p[:, j], label=labels[key], color=f'C{model_index}')
            if model_index == 0:
                axes[j].plot(times, y[:, j], 'k--', label='True')
                for boundary in row['protocol'][1:3]:
                    axes[j].axvline(boundary[0], color='gray', linewidth=.7)
            axes[j].set(ylabel=f'Home-vector {component} (mm)')
            axes[j].grid(alpha=.2)
            axes[j].legend(fontsize=8)
    axes[0].set_title('Reversal and return-home pause')
    axes[-1].set_xlabel('Time (s)')
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'reversal.{extension}', dpi=180)
    plt.close(fig)


def plot_home_vectors(snaps, arrays, labels, output):
    """Actual decoded/target components; preserve pauses and distinct body motions."""
    families = [('challenge-long-64', 'Long straight path'),
                ('challenge-pause-5', 'Opposite histories, then five-second pause'),
                ('challenge-square', 'Square-loop traversal'),
                ('challenge-world-equivalent', 'Same world path, different body motions')]
    fig, axes = plt.subplots(4, 2, figsize=(13, 13), layout='constrained')
    for model_index, (key, snap) in enumerate(snaps.items()):
        predictions, targets = arrays[key]
        for row_index, (family, title) in enumerate(families):
            indices = [i for i, row in enumerate(snap['groups']['challenge']) if row['family'] == family]
            for condition_index, i in enumerate(indices):
                row = snap['groups']['challenge'][i]
                condition = ('backward' if row['protocol'][0][2] < 0 else 'forward') if family == 'challenge-pause-5' else (
                    row['kind'] if family == 'challenge-world-equivalent' else '')
                variant = condition in ('backward', 'sideways')
                times = np.arange(len(predictions[i])) * snap['task']['dt_s']
                suffix = f', {condition}' if condition else ''
                for j, component in enumerate(('x', 'y')):
                    ax = axes[row_index, j]
                    ax.plot(times, predictions[i][:, j], color=f'C{model_index}',
                            linestyle='--' if variant else '-', label=labels[key] + suffix)
                    if model_index == 0:
                        # World-equivalent targets coincide; draw their shared truth once.
                        if family != 'challenge-world-equivalent' or condition_index == 0:
                            ax.plot(times, targets[i][:, j], color='black',
                                    linestyle='-.' if variant else ':', linewidth=1.5,
                                    label='True' + (suffix if family == 'challenge-pause-5' else ''))
                        if condition_index == 0:
                            for boundary in row['protocol'][1:-1]:
                                ax.axvline(boundary[0], color='gray', linewidth=.6)
                    ax.set(title=f'{title}: {component}', xlabel='Time (s)', ylabel=f'Home-vector {component} (mm)')
                    ax.grid(alpha=.2)
                    ax.legend(fontsize=7)
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'home_vectors.{extension}', dpi=180)
    plt.close(fig)


def analyze(args):
    if args.threads < 1:
        raise ValueError('Require a positive CPU thread count.')
    if bool(args.reference_run) != bool(args.reference_source):
        raise ValueError('Supply both --reference-run and --reference-source.')
    paths = dict(current=(Path(args.run).resolve(), Path(args.source).resolve()))
    if args.reference_run:
        paths = dict(reference=(Path(args.reference_run).resolve(), Path(args.reference_source).resolve()), **paths)
    output = Path(args.output).resolve()
    for pair in paths.values():
        for path in pair:
            if output == path or output.is_relative_to(path) or path.is_relative_to(output):
                raise ValueError('Output must be separate from every input bundle.')
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(args.threads)
    snaps = {key: capture(*pair, snapshot_key=getattr(args,
                 'reference_snapshot_key' if key == 'reference' else 'snapshot_key', 'current'))
             for key, pair in paths.items()}
    if 'reference' in snaps:
        validate_pair(snaps['reference'], snaps['current'], args.allow_task_differences)
    frozen_paths = {source for _, source in paths.values()}
    frozen_paths.update(snap['run'] for snap in snaps.values() if snap['input_kind'] == 'saved_challenge_snapshot')
    frozen = {str(p): digest(p) for source in frozen_paths for p in source.iterdir() if p.is_file()}
    output, record = begin(output, 'home_vector_challenge_snapshot_v1',
        source=str(paths['current'][1]), source_manifest_sha256=digest(paths['current'][1] / 'manifest.json'),
        run=str(paths['current'][0]), reference_run=str(paths['reference'][0]) if 'reference' in paths else None,
        allow_task_differences=args.allow_task_differences)
    record.update(refitting=False, active_runs_may_change=True, torch=torch.__version__, device='cpu',
        threads=args.threads, immutable_input_sha256=frozen,
        analysis_code_sha256={str(p.relative_to(Path(__file__).parents[1])): digest(p)
                             for p in Path(__file__).parent.glob('*.py')})
    write_json(output / 'manifest.json', record)
    try:
        report, arrays, audits, labels = {}, {}, [], {}
        for key, snap in snaps.items():
            (output / f'{key}_checkpoint.pt').write_bytes(snap['blob'])
            write_json(output / f'{key}_metadata.json', dict(
                run=str(snap['run']), source=str(snap['source']), run_manifest=snap['manifest'],
                circuit=snap['circuit'], training_config=snap['training_config'], history=snap['history'],
                checkpoint_update=snap['checkpoint']['update'], checkpoint_sha256=snap['checkpoint_sha256'],
                input_kind=snap['input_kind'], snapshot_key=snap.get('snapshot_key')))
            report[key], arrays[key] = replay(snap, output, key)
            label = args.reference_label if key == 'reference' else args.label
            labels[key] = f'{label} (update {snap["checkpoint"]["update"]})'
            audits.append(SimpleNamespace(groups=snap['groups'], task=snap['task'], out=output))
        report.update(model_labels=labels, task_differences_allowed=args.allow_task_differences,
            validation_sets_match=('reference' not in snaps or
                snaps['reference']['groups']['validation'] == snaps['current']['groups']['validation']),
            definitions=dict(checkpoint='Independent best.pt snapshot; may precede the latest optimizer update.',
                metrics='Shared with the completed-run comparison; pause drift and target error are distinct.',
                distance='Sustained magnitude underestimation beyond 2 mm for at least 0.5 s; not a general accuracy limit.',
                reversal='Delay to minimum predicted x during the backward-walking interval.',
                validation_history='Historical unweighted scores through the snapshotted update; not necessarily the selection criterion.',
                numerical_gate='No new half-step convergence gate is performed by this exploratory replay.',
                interpretation='Repeated developmental challenge inspection; no refitting or checkpoint selection by challenges.'))
        write_json(output / 'challenge_metrics.json', report)
        write_table(output, report)
        plot_comparison(tuple(audits), tuple(arrays.values()), report)
        plot_reversal(snaps, arrays, labels, output)
        plot_home_vectors(snaps, arrays, labels, output)
        if any(digest(path) != value for path, value in frozen.items()):
            raise ValueError('An immutable input bundle changed during evaluation.')
        finish(output, record)
    except BaseException as error:
        fail(output, record, error)
        raise
    result = dict(output=str(output), status='complete',
                  checkpoint_updates={key: snap['checkpoint']['update'] for key, snap in snaps.items()})
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'source', 'output'):
        parser.add_argument(f'--{name}', required=True)
    parser.add_argument('--reference-run')
    parser.add_argument('--reference-source')
    parser.add_argument('--snapshot-key', choices=['current', 'reference'], default='current',
                        help='Checkpoint to use when --run is a saved evaluation bundle')
    parser.add_argument('--reference-snapshot-key', choices=['current', 'reference'], default='current',
                        help='Checkpoint to use when --reference-run is a saved evaluation bundle')
    parser.add_argument('--allow-task-differences', action='store_true')
    parser.add_argument('--label', default='Current')
    parser.add_argument('--reference-label', default='Reference')
    parser.add_argument('--threads', type=int, default=1)
    return analyze(parser.parse_args(argv))


if __name__ == '__main__':
    main()
