"""Attribute the frozen linear decoder to neurons in saved matched trajectories.

Reads a verified paired-state bundle; never simulates, refits or perturbs neurons.
Vector magnitudes describe arithmetic contributions, not unique information.
"""

import argparse
import csv
import html
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from ..__main__ import begin, fail, finish
from ..data import digest, verify_bundle, write_json


CHECKPOINT_SHA256 = '2aab91f13258883819f66f12897919b2bf509617e7927120ca7f9db069d9a31f'
SNAPSHOTS = (0., 2., 4., 6., 7.5, 8.)
MODES = ('absolute', 'from_initial')


def contributions(rates, decoder, intercept, prediction):
    """Return T x neuron x (x,y) addends; the intercept stays separate."""
    rates, decoder, intercept, prediction = [np.asarray(x, dtype=float)
                                           for x in (rates, decoder, intercept, prediction)]
    if (rates.ndim != 2 or len(rates) == 0 or decoder.shape != (2, rates.shape[1])
            or intercept.shape != (2,) or prediction.shape != (len(rates), 2)
            or not all(np.isfinite(x).all() for x in (rates, decoder, intercept, prediction))):
        raise ValueError('Require finite aligned rates, two-output decoder and predictions.')
    absolute = rates[:, :, None] * decoder.T[None, :, :]
    np.testing.assert_allclose(absolute.sum(axis=1) + intercept, prediction,
                               atol=3e-5, rtol=1e-5)
    changed = (rates - rates[0])[:, :, None] * decoder.T[None, :, :]
    np.testing.assert_allclose(changed.sum(axis=1), prediction - prediction[0],
                               atol=6e-5, rtol=1e-5)
    return dict(absolute=absolute, from_initial=changed)


def duration_rms(vectors, time):
    """Left-held duration-weighted vector RMS; last state has zero duration."""
    vectors, time = np.asarray(vectors, dtype=float), np.asarray(time, dtype=float)
    if (time.ndim != 1 or len(time) < 2 or vectors.ndim < 2
            or len(vectors) != len(time) or not np.isfinite(vectors).all()
            or not np.isfinite(time).all() or np.any(np.diff(time) <= 0)):
        raise ValueError('Require finite vectors and strictly increasing matched times.')
    squared = np.sum(vectors[:-1] ** 2, axis=-1)
    return np.sqrt(np.tensordot(np.diff(time), squared, axes=(0, 0)) / (time[-1] - time[0]))


def ranking(scores, body_ids):
    """Descending magnitude; body ID breaks ties for display, not correlation."""
    return np.lexsort((np.asarray(body_ids), -np.asarray(scores)))


def agreement(a, b, body_ids):
    ra, rb = rankdata(-np.asarray(a)), rankdata(-np.asarray(b))
    correlation = (float(np.corrcoef(ra, rb)[0, 1])
                   if np.ptp(ra) > 0 and np.ptp(rb) > 0 else None)
    count = min(10, len(a))
    top_a, top_b = ranking(a, body_ids)[:count], ranking(b, body_ids)[:count]
    return dict(spearman_all_neurons=correlation, top_k=count,
                top_k_overlap=len(set(top_a) & set(top_b)),
                cutoff_tied_a=bool(np.sum(np.asarray(a) == a[top_a[-1]]) > 1),
                cutoff_tied_b=bool(np.sum(np.asarray(b) == b[top_b[-1]]) > 1))


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_pair(output, stem, time, delta, body_ids, cell_types, scores):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    for column, mode in enumerate(MODES):
        sa, sb = scores[mode]
        order = ranking(np.maximum(sa, sb), body_ids)[:10]
        y = np.arange(len(order))
        ax = axes[0, column]
        ax.barh(y - .18, sa[order], height=.35, label='A: forward')
        ax.barh(y + .18, sb[order], height=.35, label='B: rightward')
        ax.set_yticks(y, [f'{body_ids[j]} {cell_types[j]}' for j in order], fontsize=8)
        ax.invert_yaxis()
        ax.set(title=mode.replace('_', ' '), xlabel='Duration-weighted contribution RMS (mm)')
        ax.legend(fontsize=8)
    ax = axes[1, 0]
    for k, name in enumerate(('x', 'y')):
        ax.plot(time, delta.sum(axis=1)[:, k], label=f'Net B−A {name}')
    ax.plot(time, np.linalg.norm(delta, axis=-1).sum(axis=1), '--',
            label='Sum of individual B−A vector magnitudes')
    ax.set(xlabel='Time (s)', ylabel='mm', title='Signed net difference and gross contributions')
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    tick = int(np.argmin(np.abs(time - 4.)))
    vectors = delta[tick]
    order = ranking(np.linalg.norm(vectors, axis=-1), body_ids)[:10]
    y = np.arange(len(order))
    ax.barh(y - .18, vectors[order, 0], height=.35, label='B−A x')
    ax.barh(y + .18, vectors[order, 1], height=.35, label='B−A y')
    ax.set_yticks(y, [f'{body_ids[j]} {cell_types[j]}' for j in order], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0., color='black', lw=.5)
    ax.set(xlabel='Signed contribution difference (mm)', title=f'Largest differences at {time[tick]:g} s')
    ax.legend(fontsize=8)
    fig.suptitle(stem + ' — same fixed decoder in both trajectories')
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'{stem}.{extension}', dpi=150)
    plt.close(fig)


def write_report(output, report):
    lines = ['# Frozen decoder contributions on matched paths', '',
        'No training, new simulation, neuron intervention or decoder refit. '
        'The verified paired-state bundle supplies the exact archived rates and decoder.', '',
        '```text', 'absolute_i(t) = rate_i(t) * (D_xi, D_yi)',
        'prediction(t) = intercept + sum_i absolute_i(t)',
        'from_initial_i(t) = (rate_i(t) - rate_i(0)) * (D_xi, D_yi)',
        'paired_difference_i(t) = absolute_B_i(t) - absolute_A_i(t)', '```', '',
        'The intercept is separate. Changes from initial state sum to the change in '
        'prediction, not its absolute value. B−A is uncentered and includes any initial '
        'difference. Neuron identity and decoder weights are fixed across trajectories.', '',
        '## Rankings and overlap', '',
        'Rank scores are duration-weighted RMS Euclidean magnitudes in mm. Each saved '
        'state represents its following interval; the terminal state has zero duration. '
        'Spearman correlation uses all neurons with average ranks for ties. Top-ten '
        'overlap is descriptive; ties are broken by body ID and flagged in JSON. '
        'Each plot displays the ten largest scores across either trajectory, not necessarily '
        'both individual top-ten sets. Complete ranks and scores are in neuron_rankings.csv.', '',
        '| Pair | View | Rank correlation | Top-ten overlap |',
        '| --- | --- | ---: | ---: |']
    for pair in report['pairs']:
        for mode in MODES:
            a = pair['agreement'][mode]
            rho = a['spearman_all_neurons']
            value = f'{rho:.4f}' if rho is not None else 'undefined'
            lines.append(f"| {pair['pair']} | {mode} | {value} | {a['top_k_overlap']}/{a['top_k']} |")
    lines += ['', 'Initial rate maxima are recorded per trajectory in contributions.json. '
        'When all initial rates are zero, absolute and initial-referenced addends coincide '
        'exactly; agreement of those views is then an initialization consequence.', '']
    for pair in report['pairs']:
        lines += [f"### Largest absolute contributors: {pair['pair']}", '',
            '| Traversal | Rank | Body ID | Type | Contribution RMS (mm) |',
            '| --- | ---: | ---: | --- | ---: |']
        for label in ('a', 'b'):
            for rank, row in enumerate(pair['top_absolute'][label], 1):
                lines.append(f"| {label.upper()} | {rank} | {row['body_id']} | {row['cell_type']} | {row['rms_mm']:.4f} |")
    worked = next(p for p in report['pairs'] if p['pair'] == 'path_12_vs_path_13')['worked_snapshot']
    lines += ['', '## Worked example: paths 12/13 at 4 seconds', '',
        'The ten largest individual B−A magnitudes are listed below. Signs are retained; '
        'these are contributions to prediction disagreement, not individually assigned errors.', '',
        '| Body ID | Type | B−A x (mm) | B−A y (mm) |',
        '| ---: | --- | ---: | ---: |']
    for row in worked['largest_differences']:
        lines.append(f"| {row['body_id']} | {row['cell_type']} | {row['x_mm']:.4f} | {row['y_mm']:.4f} |")
    lines += ['', f"Sum across all 65 neurons: ({worked['net_x_mm']:.4f}, {worked['net_y_mm']:.4f}) mm. "
        f"Sum of individual vector magnitudes: {worked['sum_individual_norms_mm']:.4f} mm. "
        'These quantities differ because the signed vectors cancel. The table lists only '
        'ten neurons; the other 55 remain included in the reported net sum.', '']
    lines += ['', '## Interpretation', '',
        'A large contribution can be canceled by other neurons or the intercept. '
        'Magnitude is not a share of unique information, predictive accuracy, or biological '
        'necessity. These are addends of a fixed linear readout. Removing a term from this '
        'sum would not simulate removing a neuron from the recurrent circuit.', '',
        'Large paired differences may cancel even when they dominate the rankings. '
        'Keep signed x/y components beside magnitudes. Population summaries give both '
        'RMS of the signed population sum and sum of individual RMS scores; the latter '
        'does not measure the population’s net output. No additive allocation of vector '
        'error or cross-neuron covariance is claimed.', '',
        'Absolute and initial-referenced views answer different questions. Rankings of '
        'B−A identify changes responsible for output disagreement, not necessarily neurons '
        'that dominate either prediction. Matching rankings also need not imply matching '
        'activities or signed contributions.', '',
        '## Files', '',
        '- neuron_rankings.csv: all 65 neurons, weights, scores and ranks for both views and B−A.',
        '- snapshot_neurons.csv: signed x/y addends at 0, 2, 4, 6, 7.5 and 8 seconds.',
        '- population_summary.csv: net and individual magnitude summaries by cell type.',
        '- timecourses.csv: signed sums, intercept, predictions, errors and gross magnitudes.',
        '- Pair NPZ files: full-resolution contributions, times, predictions and targets.',
        '- contributions.json: reconstruction residuals and ranking comparisons.',
        '- decoder.npz, neurons.csv, analysis_script.py and manifest.json: identities and provenance.', '']
    for pair in report['pairs']:
        stem = pair['pair']
        lines += [f'## {stem}', '', f'![Contributions]({stem}.png)', '',
                  f'[PDF]({stem}.pdf) · [Full arrays]({stem}.npz)', '']
    (output / 'report.md').write_text('\n'.join(lines))
    overlaps = ', '.join(str(p['agreement']['absolute']['top_k_overlap']) for p in report['pairs'])
    initial_note = ('Absolute and initial-referenced views coincide because all saved activities start at zero.'
                    if all(v == 0 for p in report['pairs'] for v in p['initial_rate_max_abs'].values())
                    else 'Initial activities are nonzero; inspect both contribution views.')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8">'
        '<title>Decoder contributions</title><style>body{font:16px system-ui;max-width:1100px;'
        'margin:2em auto;line-height:1.5}img{max-width:100%}</style>'
        '<h1>Frozen decoder contributions</h1><p>A = forward; B = rightward. '
        'Contribution magnitudes are arithmetic addends, not unique information.</p>'
        '<p><a href="report.md">Methods and full report</a> · '
        '<a href="neuron_rankings.csv">All neuron rankings</a> · '
        '<a href="snapshot_neurons.csv">Signed snapshots</a> · '
        '<a href="population_summary.csv">Population summaries</a></p>'
        f'<p>Top-ten overlap counts in pair order: {overlaps}. '
        f'Read the report for full-rank comparisons and signed cancellation. {initial_note}</p>'
        + ''.join(f'<h2>{html.escape(p["pair"])}</h2><img src="{p["pair"]}.png" '
                  f'alt="Neuron contributions for {p["pair"]}">' for p in report['pairs']))


def analyze(args):
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Output must be separate from the immutable source bundle.')
    if output.exists():
        raise FileExistsError(output)
    if verify_bundle(source)['kind'] != 'home_vector_paired_state_spaces_v1':
        raise ValueError('Require the verified paired-state analysis bundle.')
    parent = json.loads((source / 'state_spaces.json').read_text())
    if (parent['checkpoint_sha256'] != CHECKPOINT_SHA256 or parent['checkpoint_update'] != 1975
            or digest(source / 'current_checkpoint.pt') != CHECKPOINT_SHA256):
        raise ValueError('Require frozen best checkpoint at update 1975.')
    expected = {f'path_{a:02d}_vs_path_{a+1:02d}' for a in (8, 10, 12, 14)}
    if len(parent['pairs']) != 4 or {p['pair'] for p in parent['pairs']} != expected:
        raise ValueError('Require all four specified archived matched pairs.')
    with np.load(source / 'decoder.npz', allow_pickle=False) as saved:
        decoder, intercept, body_ids = [saved[k].copy() for k in ('decoder', 'intercept', 'body_ids')]
    with (source / 'neurons.csv').open() as stream:
        neurons = list(csv.DictReader(stream))
    if (len(body_ids) != 65 or len(set(body_ids)) != 65
            or [int(n['body_id']) for n in neurons] != body_ids.tolist()):
        raise ValueError('Require the aligned 65-neuron identity order.')
    cell_types = [n['cell_type'] for n in neurons]
    frozen = {str(p): digest(p) for p in source.iterdir() if p.is_file()}
    output, manifest = begin(output, 'home_vector_decoder_contributions_v1', source=str(source),
                             source_manifest_sha256=digest(source / 'manifest.json'))
    manifest.update(refitting=False, simulation=False, immutable_input_sha256=frozen,
                    analysis_script_sha256=digest(__file__))
    write_json(output / 'manifest.json', manifest)
    try:
        for name in ('decoder.npz', 'neurons.csv'):
            shutil.copyfile(source / name, output / name)
        shutil.copyfile(__file__, output / 'analysis_script.py')
        report = dict(checkpoint_update=1975, checkpoint_sha256=CHECKPOINT_SHA256,
                      source=str(source), units='mm', intercept=intercept.tolist(), pairs=[])
        ranking_rows, snapshots, populations, timecourses = [], [], [], []
        for pair in parent['pairs']:
            stem = pair['pair']
            with np.load(source / f'{stem}.npz', allow_pickle=False) as saved:
                data = {k: saved[k].copy() for k in saved.files}
            time = data['time_s']
            ca, cb = [contributions(data[f'rates_{label}'], decoder, intercept,
                                    data[f'prediction_{label}']) for label in ('a', 'b')]
            if len(time) != len(ca['absolute']):
                raise ValueError('Time and activity lengths differ.')
            np.testing.assert_allclose(data['target_a'], data['target_b'], atol=5e-6, rtol=0)
            # B minus A, without subtracting its initial value.
            delta = cb['absolute'] - ca['absolute']
            reconstructed = delta.sum(axis=1)
            archived = data['prediction_b'].astype(float) - data['prediction_a'].astype(float)
            np.testing.assert_allclose(reconstructed, archived, atol=6e-5, rtol=1e-5)
            np.testing.assert_allclose(reconstructed, data['decoded_difference'], atol=1e-10, rtol=1e-10)
            scores = {mode: (duration_rms(ca[mode], time), duration_rms(cb[mode], time)) for mode in MODES}
            scores['paired_difference'] = (duration_rms(delta, time),)
            ranks = {}
            for mode, members in scores.items():
                for label, score in zip(('a', 'b'), members):
                    order = ranking(score, body_ids)
                    rank = np.empty(len(order), dtype=int)
                    rank[order] = np.arange(1, len(order) + 1)
                    ranks[mode, label] = rank
            for j, body in enumerate(body_ids):
                row = dict(pair=stem, body_id=int(body), cell_type=cell_types[j],
                           decoder_x=float(decoder[0, j]), decoder_y=float(decoder[1, j]))
                for mode, members in scores.items():
                    for label, score in zip(('a', 'b'), members):
                        suffix = '' if mode == 'paired_difference' else '_' + label
                        row[mode + suffix + '_rms_mm'] = float(score[j])
                        row[mode + suffix + '_rank'] = int(ranks[mode, label][j])
                ranking_rows.append(row)
            views = {f'{mode}_{label}': values[mode] for mode in MODES for label, values in (('a', ca), ('b', cb))}
            views['paired_difference'] = delta
            for seconds in SNAPSHOTS:
                tick = int(np.argmin(np.abs(time - seconds)))
                if abs(time[tick] - seconds) > 1e-8:
                    continue
                for j, body in enumerate(body_ids):
                    row = dict(pair=stem, time_s=float(time[tick]), body_id=int(body), cell_type=cell_types[j])
                    for name, vectors in views.items():
                        row[name + '_x_mm'], row[name + '_y_mm'] = map(float, vectors[tick, j])
                        row[name + '_norm_mm'] = float(np.linalg.norm(vectors[tick, j]))
                    snapshots.append(row)
            for cell_type in sorted(set(cell_types)):
                select = np.array(cell_types) == cell_type
                row = dict(pair=stem, cell_type=cell_type, neuron_count=int(select.sum()))
                for name, vectors in views.items():
                    row[name + '_net_rms_mm'] = float(duration_rms(vectors[:, select].sum(axis=1), time))
                    row[name + '_sum_individual_rms_mm'] = float(duration_rms(vectors[:, select], time).sum())
                populations.append(row)
            sums = {name: vectors.sum(axis=1) for name, vectors in views.items()}
            gross = {name: np.linalg.norm(vectors, axis=-1).sum(axis=1) for name, vectors in views.items()}
            for tick, seconds in enumerate(time):
                row = dict(pair=stem, time_s=float(seconds), duration_s=float(time[tick+1]-seconds) if tick+1 < len(time) else 0.)
                for name in views:
                    row[name + '_net_x_mm'], row[name + '_net_y_mm'] = map(float, sums[name][tick])
                    row[name + '_sum_individual_norms_mm'] = float(gross[name][tick])
                for label in ('a', 'b'):
                    for k, coord in enumerate(('x', 'y')):
                        row[f'prediction_{label}_{coord}_mm'] = float(data[f'prediction_{label}'][tick, k])
                        row[f'target_{label}_{coord}_mm'] = float(data[f'target_{label}'][tick, k])
                        row[f'intercept_{coord}_mm'] = float(intercept[k])
                    row[f'error_{label}_mm'] = float(np.linalg.norm(data[f'prediction_{label}'][tick] - data[f'target_{label}'][tick]))
                timecourses.append(row)
            tick = int(np.argmin(np.abs(time - 4.)))
            snapshot_delta = delta[tick]
            snapshot_order = ranking(np.linalg.norm(snapshot_delta, axis=-1), body_ids)[:10]
            report['pairs'].append(dict(pair=stem,
                agreement={mode: agreement(*scores[mode], body_ids) for mode in MODES},
                initial_rate_max_abs={label: float(np.max(np.abs(data[f'rates_{label}'][0]))) for label in ('a', 'b')},
                top_absolute={label: [dict(body_id=int(body_ids[j]), cell_type=cell_types[j], rms_mm=float(score[j]))
                    for j in ranking(score, body_ids)[:5]] for label, score in zip(('a', 'b'), scores['absolute'])},
                worked_snapshot=dict(time_s=float(time[tick]),
                    net_x_mm=float(snapshot_delta.sum(axis=0)[0]), net_y_mm=float(snapshot_delta.sum(axis=0)[1]),
                    sum_individual_norms_mm=float(np.linalg.norm(snapshot_delta, axis=-1).sum()),
                    largest_differences=[dict(body_id=int(body_ids[j]), cell_type=cell_types[j],
                                             x_mm=float(snapshot_delta[j, 0]), y_mm=float(snapshot_delta[j, 1]))
                                         for j in snapshot_order]),
                prediction_reconstruction_max_abs_mm={label: float(np.max(np.abs(
                    values['absolute'].sum(axis=1) + intercept - data[f'prediction_{label}'])))
                    for label, values in (('a', ca), ('b', cb))},
                paired_reconstruction_max_abs_mm=float(np.max(np.abs(reconstructed - archived)))))
            np.savez_compressed(output / f'{stem}.npz', time_s=time, body_ids=body_ids, **views,
                **{k: data[k] for k in ('prediction_a', 'prediction_b', 'target_a', 'target_b')})
            plot_pair(output, stem, time, delta, body_ids, cell_types, scores)
            print(f'Analyzed {stem}', flush=True)
        write_csv(output / 'neuron_rankings.csv', ranking_rows)
        write_csv(output / 'snapshot_neurons.csv', snapshots)
        write_csv(output / 'population_summary.csv', populations)
        write_csv(output / 'timecourses.csv', timecourses)
        write_json(output / 'contributions.json', report)
        write_report(output, report)
        if any(digest(path) != sha for path, sha in frozen.items()):
            raise ValueError('An immutable source file changed during analysis.')
        finish(output, manifest)
    except BaseException as error:
        fail(output, manifest, error)
        raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    return analyze(parser.parse_args(argv))


if __name__ == '__main__':
    main()
