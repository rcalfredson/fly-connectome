"""Two uninterrupted repetitions of the archived closed circles; frozen model."""

import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..__main__ import begin, fail, finish
from ..data import Circuit, digest, sample_protocol, verify_bundle, write_json
from .activity import record_trajectory
from .checkpoint import capture
from .decoder_contributions import CHECKPOINT_SHA256, write_csv
from .state_patterns import project_states


def repeat_protocol(protocol, dt):
    """Repeat held movement twice, removing only the unused first-loop stop input."""
    original = np.asarray(protocol, dtype=float)
    controls, target = sample_protocol(original, dt)
    if np.any(original[-1, 2:] != 0) or np.linalg.norm(target[-1]) > 5e-6:
        raise ValueError('Require a closed path with a terminal stationary sample.')
    period = float(original[-1, 0])
    movement = original[:-1]
    twice = np.vstack((movement, movement + [period, 0., 0., 0.],
                       original[-1] + [period, 0., 0., 0.]))
    repeated, truth = sample_protocol(twice, dt)
    n = len(controls) - 1
    np.testing.assert_array_equal(repeated[:n], controls[:-1])
    np.testing.assert_array_equal(repeated[n:2*n], controls[:-1])
    np.testing.assert_allclose(truth[n], 0., atol=5e-6, rtol=0)
    np.testing.assert_allclose(truth[-1], 0., atol=5e-6, rtol=0)
    return twice.tolist(), n, period


def plot_pair(output, stem, members):
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True, layout='constrained')
    for label, data in members:
        t, p, y = data['time_s'], data['prediction'], data['target']
        for j in range(2):
            axes[0, j].plot(t, p[:, j], label=label)
            if label == 'forward':
                axes[0, j].plot(t, y[:, j], 'k:', lw=2, label='Correct')
        axes[1, 0].plot(t, np.linalg.norm(p - y, axis=1), label=label)
        axes[1, 1].plot(t, np.max(data['rates'], axis=1), label=label)
        axes[2, 0].plot(t, np.linalg.norm(data['row'], axis=1), label=label)
        axes[2, 1].plot(t, np.linalg.norm(data['null'], axis=1), label=label)
    settings = [('Home x', 'mm'), ('Home y', 'mm'), ('Vector error', 'mm'),
                ('Largest single-neuron activity', 'Model activity units'),
                ('Row-space state norm', 'Model activity units'),
                ('Null-space state norm', 'Model activity units')]
    for ax, (title, units) in zip(axes.flat, settings):
        ax.set(title=title, ylabel=units)
        ax.axvline(8., color='gray', ls='--', lw=.8)
        ax.axvline(16., color='gray', ls='--', lw=.8)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('Time (s); uninterrupted state across 8 s')
    fig.suptitle(stem + ': two identical movement loops; no state reset')
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'{stem}.{extension}', dpi=150)
    plt.close(fig)


def write_report(output, report, rows):
    lines = ['# Two uninterrupted circular loops: frozen checkpoint', '',
        'Repeat the exact held movement of the archived closed-circle paths 8/9 and '
        '14/15 twice, for 16 seconds. The checkpoint, PFN encoder and decoder remain '
        'fixed. Each trial starts from zero activity once, with no reset at 8 seconds. '
        'The first-loop stationary terminal control is replaced by the next loop’s '
        'first movement control; this cannot affect the already-recorded boundary state.', '',
        'Both loop endpoints have essentially zero true displacement. The initial '
        'decoder intercept is retained. Compare absolute endpoint error and endpoint '
        'drift relative to the initial prediction; these differ because the initial '
        'prediction is not exactly zero.', '',
        '| Path | Loop | Vector RMSE (mm) | Closure error (mm) | Max activity | End row norm | End null norm |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['path']} | {r['loop']} | {r['vector_rmse_mm']:.4f} | "
            f"{r['closure_error_mm']:.4f} | {r['max_activity']:.3f} | {r['end_row_norm']:.3f} | {r['end_null_norm']:.3f} |")
    lines += ['', '## Interpretation limits', '',
        'The model rectifies activity below zero but has no upper firing-rate cap. '
        'The linear decoder has no hard output range either. Therefore there is no '
        'literal upper saturation threshold tested here. Finite activity growth, '
        'closure error and changes in error across loops are the relevant measurements. '
        'Activity units are not calibrated firing rates. The replay guard of 1e6 '
        'activity units is a numerical guard, not a physiological ceiling.', '',
        'Returning home requires a correct readout, not recovery of every initial '
        'neural activity. Additional null activity need not be harmful immediately, '
        'but may affect later dynamics. Two loops cannot establish asymptotic '
        'boundedness, unlimited stability, or a specific causal mechanism. Comparing '
        'loop 2 with loop 1 keeps movement inputs fixed and changes the inherited '
        'state; it does not isolate which component of that state causes any change.', '',
        'Loop RMSE uses states preceding the following update and excludes each '
        'zero-duration loop endpoint from that loop’s time average. Endpoint '
        'measurements are reported separately. All corresponding-phase arrays '
        'remain in the NPZ files, including both closure states.', '',
        '## Verification', '',
        'The complete original one-loop protocols are replayed first. Repeated '
        'trials reproduce their first-loop states, predictions and targets, '
        'including the closure state. Controls and PFNs repeat exactly across '
        'the two movement loops. Targets are independently integrated. Source '
        'bundles and checkpoint identity are verified and their hashes remain unchanged.', '',
        'The manifest records code/input hashes. protocols.json preserves every '
        'timestamped input, loop_metrics.csv gives endpoint and phase comparisons, '
        'and each path NPZ preserves full-resolution activities, PFNs, targets, '
        'predictions, and separate row/null projections.', '']
    for stem in ('paths_08_09', 'paths_14_15'):
        lines += [f'## {stem}', '', f'![Two loops]({stem}.png)', '', f'[PDF]({stem}.pdf)', '']
    (output / 'report.md').write_text('\n'.join(lines))
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8">'
        '<title>Two-loop frozen-model probe</title><style>body{font:16px system-ui;'
        'max-width:1200px;margin:2em auto}img{max-width:100%}</style>'
        '<h1>Two uninterrupted circular loops</h1><p>Fixed encoder, network and decoder; no reset at 8 s.</p>'
        '<p><a href="report.md">Methods, results and limitations</a> · '
        '<a href="loop_metrics.csv">Numerical comparisons</a></p>'
        + ''.join(f'<h2>{stem}</h2><img src="{stem}.png" alt="Two-loop readout and activity">'
                  for stem in ('paths_08_09', 'paths_14_15')))


def analyze(args):
    activity, source, output = [Path(getattr(args, k)).resolve() for k in ('activity', 'source', 'output')]
    for folder in (activity, source):
        if output == folder or output.is_relative_to(folder) or folder.is_relative_to(output):
            raise ValueError('Output must be separate from immutable input bundles.')
    if output.exists():
        raise FileExistsError(output)
    if verify_bundle(activity)['kind'] != 'home_vector_activity_paths_v1':
        raise ValueError('Require the original archived activity-paths bundle.')
    verify_bundle(source)
    torch.set_num_threads(1)
    snap = capture(activity, source)
    if snap['checkpoint_sha256'] != CHECKPOINT_SHA256 or snap['checkpoint']['update'] != 1975:
        raise ValueError('Require the specified frozen checkpoint.')
    network, circuit, dt = snap['network'], Circuit(**snap['circuit']), snap['task']['dt_s']
    records = {r['path']: r for r in json.loads((activity / 'raw_paths.json').read_text())}
    frozen = {str(p): digest(p) for folder in (activity, source) for p in folder.iterdir() if p.is_file()}
    output, manifest = begin(output, 'home_vector_repeated_loops_v1', source=str(source),
        source_manifest_sha256=digest(source / 'manifest.json'), activity=str(activity),
        activity_manifest_sha256=digest(activity / 'manifest.json'))
    manifest.update(immutable_input_sha256=frozen, analysis_script_sha256=digest(__file__),
                    checkpoint_sha256=CHECKPOINT_SHA256, refitting=False, threads=1, torch=torch.__version__)
    write_json(output / 'manifest.json', manifest)
    try:
        for name in ('current_checkpoint.pt', 'current_metadata.json', 'neurons.csv'):
            shutil.copyfile(activity / name, output / name)
        shutil.copyfile(__file__, output / 'analysis_script.py')
        decoder = network.D.detach().numpy().astype(float)
        np.savez_compressed(output / 'decoder.npz', decoder=decoder,
                            intercept=network.d.detach().numpy(), body_ids=circuit.dynamic_ids)
        report, rows, protocols, members = dict(checkpoint_sha256=CHECKPOINT_SHA256, dt_s=dt, trials=[]), [], {}, {}
        for number in (8, 9, 14, 15):
            name = f'path_{number:02d}'
            record = records[name]['record']
            baseline, _ = record_trajectory(network, circuit, record, dt)
            with np.load(activity / f'{name}.npz', allow_pickle=False) as saved:
                for key in ('controls', 'pfn', 'rates', 'prediction', 'target'):
                    np.testing.assert_allclose(baseline[key], saved[key], atol=3e-5, rtol=1e-6)
            protocol, n, period = repeat_protocol(record['protocol'], dt)
            if period != 8.:
                raise ValueError('Require the archived eight-second circle.')
            trial = dict(record, protocol=protocol)
            data, _ = record_trajectory(network, circuit, trial, dt)
            replay_checks = {}
            for key in ('rates', 'prediction', 'target'):
                np.testing.assert_allclose(data[key][:n+1], baseline[key], atol=3e-5, rtol=1e-6)
                replay_checks[key] = float(np.max(np.abs(data[key][:n+1] - baseline[key])))
            for key in ('controls', 'pfn'):
                np.testing.assert_array_equal(data[key][:n], baseline[key][:-1])
                np.testing.assert_array_equal(data[key][:n], data[key][n:2*n])
            data['row'], data['null'] = project_states(data['rates'], decoder)
            np.testing.assert_allclose(data['target'][[n, 2*n]], 0., atol=5e-6, rtol=0)
            protocols[name] = trial
            for loop in (1, 2):
                start, end = (loop-1)*n, loop*n
                error = data['prediction'][start:end] - data['target'][start:end]
                row = dict(path=name, traversal=records[name]['kind'], loop=loop,
                    vector_rmse_mm=float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
                    closure_error_mm=float(np.linalg.norm(data['prediction'][end] - data['target'][end])),
                    closure_drift_from_initial_mm=float(np.linalg.norm(data['prediction'][end] - data['prediction'][0])),
                    closure_prediction_x_mm=float(data['prediction'][end, 0]), closure_prediction_y_mm=float(data['prediction'][end, 1]),
                    max_activity=float(np.max(data['rates'][start:end+1])),
                    end_rate_norm=float(np.linalg.norm(data['rates'][end])),
                    end_row_norm=float(np.linalg.norm(data['row'][end])),
                    end_null_norm=float(np.linalg.norm(data['null'][end])))
                rows.append(row)
            report['trials'].append(dict(path=name, first_loop_replay_max_abs=replay_checks,
                phase_matched_rate_difference_rms=float(np.sqrt(np.mean(np.sum((data['rates'][n:2*n] - data['rates'][:n])**2, axis=1)))),
                phase_matched_prediction_difference_rms_mm=float(np.sqrt(np.mean(np.sum((data['prediction'][n:2*n] - data['prediction'][:n])**2, axis=1))))))
            np.savez_compressed(output / f'{name}.npz', **data)
            members[number] = data
            print(f'Replayed {name}: two loops without reset', flush=True)
        for a, b in ((8, 9), (14, 15)):
            np.testing.assert_allclose(members[a]['target'], members[b]['target'], atol=5e-6, rtol=0)
            plot_pair(output, f'paths_{a:02d}_{b:02d}', [('forward', members[a]), ('rightward', members[b])])
        write_json(output / 'protocols.json', protocols)
        write_json(output / 'repeated_loops.json', report)
        write_csv(output / 'loop_metrics.csv', rows)
        write_report(output, report, rows)
        if any(digest(path) != sha for path, sha in frozen.items()):
            raise ValueError('An immutable input changed during analysis.')
        finish(output, manifest)
    except BaseException as error:
        fail(output, manifest, error)
        raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('activity', 'source', 'output'):
        parser.add_argument(f'--{name}', required=True)
    return analyze(parser.parse_args(argv))


if __name__ == '__main__':
    main()
