"""Visualize separate decoder-visible and null projections of matched states."""

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

from ..__main__ import begin, fail, finish
from ..data import digest, verify_bundle, write_json
from .decoder_contributions import CHECKPOINT_SHA256, duration_rms, ranking, write_csv


def project_states(rates, decoder):
    """Project actual states in their original neuron coordinates, without centering."""
    rates, decoder = np.asarray(rates, dtype=float), np.asarray(decoder, dtype=float)
    if (rates.ndim != 2 or not len(rates) or decoder.ndim != 2
            or decoder.shape[0] != 2 or decoder.shape[1] != rates.shape[1]
            or not np.isfinite(rates).all() or not np.isfinite(decoder).all()):
        raise ValueError('Require finite aligned rates and a two-output decoder.')
    _, singular, vh = np.linalg.svd(decoder, full_matrices=False)
    tolerance = singular[0] * max(decoder.shape) * np.finfo(float).eps
    basis = vh[singular > tolerance]
    row = (rates @ basis.T) @ basis
    null = rates - row
    np.testing.assert_allclose(row + null, rates, atol=1e-10, rtol=1e-12)
    np.testing.assert_allclose(null @ decoder.T, 0., atol=1e-9, rtol=0)
    np.testing.assert_allclose(row @ decoder.T, rates @ decoder.T, atol=1e-9, rtol=1e-12)
    np.testing.assert_allclose(np.sum(row * null, axis=1), 0., atol=1e-8, rtol=0)
    return row, null


def select_examples(row_a, row_b, null_a, null_b, time, body_ids, count):
    """Select large-in-both row entries, not entries chosen to resemble each other."""
    if not 1 <= count <= len(body_ids):
        raise ValueError('Selection count must be between one and the neuron count.')
    def scalar_rms(values):
        return duration_rms(np.asarray(values)[:, :, None], time)
    ra, rb = scalar_rms(row_a), scalar_rms(row_b)
    shared = np.minimum(ra, rb)
    differing = scalar_rms(null_b - null_a)
    return dict(row_a_rms=ra, row_b_rms=rb, row_shared_score=shared,
                null_difference_rms=differing,
                row_selected=ranking(shared, body_ids)[:count],
                null_selected=ranking(differing, body_ids)[:count])


def correlation(a, b):
    """Descriptive temporal correlation; constant traces have no defined value."""
    return float(np.corrcoef(a, b)[0, 1]) if np.ptp(a) > 0 and np.ptp(b) > 0 else None


def plot_patterns(output, stem, time, data, patterns, selected, ids, types):
    count = len(selected['row_selected'])
    fig = plt.figure(figsize=(19, 3 + 2.1 * count), layout='constrained')
    grid = fig.add_gridspec(count + 1, 4)
    overlap = set(selected['row_selected']) & set(selected['null_selected'])
    for k, component in enumerate(('x', 'y')):
        ax = fig.add_subplot(grid[0, 2*k:2*k+2])
        ax.plot(time, data['target_a'][:, k], color='black', lw=2, label='Shared correct vector')
        ax.plot(time, data['prediction_a'][:, k], color='tab:blue', label='A: forward')
        ax.plot(time, data['prediction_b'][:, k], color='tab:orange', ls='--', label='B: rightward')
        ax.set(title=f'Decoded home vector: {component}', ylabel='mm')
        ax.legend(fontsize=8)
    for rank in range(count):
        for column, (group, projection) in enumerate((('row', 'row'), ('null', 'null'))):
            j = selected[group + '_selected'][rank]
            label = f'{ids[j]} {types[j]}' + (' *' if j in overlap else '')
            for offset, view in enumerate((projection, 'rates')):
                ax = fig.add_subplot(grid[rank + 1, 2*column + offset])
                values = patterns if view != 'rates' else data
                for member, color, style in [('a', 'tab:blue', '-'), ('b', 'tab:orange', '--')]:
                    ax.plot(time, values[f'{view}_{member}'][:, j], color=color, ls=style, lw=1.2)
                ax.axhline(0, color='black', lw=.4, alpha=.4)
                name = f'{projection}-space projection' if offset == 0 else 'actual activity'
                ax.set(title=f'{label} — {name}', ylabel='Model activity units')
                ax.grid(alpha=.2)
                if rank == count - 1:
                    ax.set_xlabel('Time (s)')
    fig.suptitle(stem + '\nLeft: large row-space entries in both paths; right: large null-space differences. '
                 '* = selected in both sets. Projections are signed components, not firing rates.', fontsize=12)
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'{stem}.{extension}', dpi=140)
    plt.close(fig)


def write_report(output, report):
    lines = ['# Separate state projections on matched world paths', '',
        'A = forward traversal; B = rightward traversal. The exact same frozen decoder '
        'projects each trajectory separately. Actual activities are shown beside projected '
        'components in the original neuron coordinates. No training, simulation or centering.', '',
        '```text', 'row_A = activity_A @ Q.T @ Q', 'null_A = activity_A - row_A',
        'row_B = activity_B @ Q.T @ Q', 'null_B = activity_B - row_B',
        'prediction_A = intercept + row_A @ D.T',
        'prediction_B = intercept + row_B @ D.T', '```', '',
        'Q is an orthonormal basis for the fixed decoder row space. Both null projections '
        'decode to zero. Equal predictions imply equal row projections for this fixed decoder; '
        'agreement with the correct vector must still be checked separately.', '',
        '## Example selection', '',
        'For each neuron, the row score is min(RMS(row_A_i), RMS(row_B_i)): large projected '
        'activity in both paths. This rule never rewards matching signs, temporal correlation '
        'or a small difference. The null score is RMS(null_B_i - null_A_i). Select the top '
        f"{report['examples_per_set']} of each independently. Asterisks mark neurons selected in both sets. "
        'RMS uses interval durations; the terminal sample has zero duration. Body ID breaks ties. '
        'Correlations and agreement metrics also exclude that endpoint. All 65 scores are saved.', '',
        'These are mathematical population components, not separate cell populations or '
        'separately measured firing rates. A neuron can appear in both sets, and its actual '
        'activity need not resemble its row projection. Neither set establishes unique '
        'information or causal necessity. Null activity may affect future recurrent dynamics.', '',
        '## Full-population comparisons', '',
        '| Pair | RMS row A / B | RMS row B−A | RMS null B−A |',
        '| --- | ---: | ---: | ---: |']
    for p in report['pairs']:
        lines.append(f"| {p['pair']} | {p['row_a_rms']:.3f} / {p['row_b_rms']:.3f} | "
                     f"{p['row_difference_rms']:.3f} | {p['null_difference_rms']:.3f} |")
    lines += ['', 'All norms above use model activity units, not mm or fractions of information. '
        'Sums and projections were checked against the original paired-state decomposition '
        'and archived predictions. The figures show actual prediction disagreement rather '
        'than presuming that the row patterns coincide.', '',
        'Files: neuron_scores.csv contains complete selection scores, within-path projected '
        'RMS and descriptive temporal correlations; timecourses.csv contains whole-population '
        'norms and outputs; pair NPZ files preserve full arrays; patterns.json records '
        'selection, validation and initial-state checks. The decoder, identities and exact '
        'script are copied for provenance.', '']
    for p in report['pairs']:
        stem = p['pair']
        lines += [f'## {stem}', '',
            'Row examples: ' + ', '.join(p['row_labels']) + '.', '',
            'Null-difference examples: ' + ', '.join(p['null_labels']) + '.', '',
            f'![Separate projections and actual activities]({stem}.png)', '',
            f'[PDF]({stem}.pdf) · [Full arrays]({stem}.npz)', '']
    (output / 'report.md').write_text('\n'.join(lines))
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8">'
        '<title>Matched activity patterns</title><style>body{font:16px system-ui;max-width:1600px;'
        'margin:2em auto;line-height:1.5}img{max-width:100%}</style>'
        '<h1>Separate decoder-visible and null activity patterns</h1>'
        '<p>Blue: A, forward. Dashed orange: B, rightward. Black: shared correct home vector. '
        'Projected traces are signed mathematical components, not separate firing rates.</p>'
        '<p>Row examples have large projected activity in both paths; selection does not require '
        'agreement. Null examples have large null-component differences. * marks overlapping selections.</p>'
        '<p><a href="report.md">Full methods and report</a> · '
        '<a href="neuron_scores.csv">All neuron scores</a> · '
        '<a href="timecourses.csv">Whole-population timecourses</a></p>'
        + ''.join(f'<h2>{html.escape(p["pair"])}</h2><a href="{p["pair"]}.pdf">PDF</a>'
                  f'<img src="{p["pair"]}.png" alt="Separate projected and actual activity traces for {p["pair"]}">'
                  for p in report['pairs']))


def analyze(args):
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Output must be separate from the source bundle.')
    if output.exists():
        raise FileExistsError(output)
    if not 1 <= args.examples <= 65:
        raise ValueError('Require between one and 65 examples per set.')
    if verify_bundle(source)['kind'] != 'home_vector_paired_state_spaces_v1':
        raise ValueError('Require the completed paired-state bundle.')
    parent = json.loads((source / 'state_spaces.json').read_text())
    if (parent['checkpoint_sha256'] != CHECKPOINT_SHA256 or parent['checkpoint_update'] != 1975
            or digest(source / 'current_checkpoint.pt') != CHECKPOINT_SHA256):
        raise ValueError('Require the specified frozen checkpoint at update 1975.')
    expected = {f'path_{a:02d}_vs_path_{a+1:02d}' for a in (8, 10, 12, 14)}
    if len(parent['pairs']) != 4 or {p['pair'] for p in parent['pairs']} != expected:
        raise ValueError('Require all four matched pairs.')
    with np.load(source / 'decoder.npz', allow_pickle=False) as saved:
        decoder, intercept, ids = [saved[k].copy() for k in ('decoder', 'intercept', 'body_ids')]
    with (source / 'neurons.csv').open() as stream:
        neurons = list(csv.DictReader(stream))
    if len(ids) != 65 or len(set(ids)) != 65 or [int(r['body_id']) for r in neurons] != ids.tolist():
        raise ValueError('Neuron identities must match the 65-cell decoder order.')
    types = [r['cell_type'] for r in neurons]
    frozen = {str(p): digest(p) for p in source.iterdir() if p.is_file()}
    output, manifest = begin(output, 'home_vector_separate_state_patterns_v1', source=str(source),
        source_manifest_sha256=digest(source / 'manifest.json'), examples_per_set=args.examples)
    manifest.update(immutable_input_sha256=frozen, analysis_script_sha256=digest(__file__),
                    refitting=False, simulation=False)
    write_json(output / 'manifest.json', manifest)
    try:
        for name in ('decoder.npz', 'neurons.csv'):
            shutil.copyfile(source / name, output / name)
        shutil.copyfile(__file__, output / 'analysis_script.py')
        report = dict(checkpoint_sha256=CHECKPOINT_SHA256, examples_per_set=args.examples, pairs=[])
        score_rows, time_rows = [], []
        for pair in parent['pairs']:
            stem = pair['pair']
            with np.load(source / f'{stem}.npz', allow_pickle=False) as saved:
                data = {k: saved[k].copy() for k in saved.files}
            time = data['time_s']
            patterns, checks = {}, {}
            for label in ('a', 'b'):
                row, null = project_states(data[f'rates_{label}'], decoder)
                patterns[f'row_{label}'], patterns[f'null_{label}'] = row, null
                residual = row @ decoder.T + intercept - data[f'prediction_{label}']
                np.testing.assert_allclose(residual, 0., atol=3e-5, rtol=0)
                checks[label + '_prediction_max_abs_mm'] = float(np.max(np.abs(residual)))
                checks[label + '_null_output_max_abs_mm'] = float(np.max(np.abs(null @ decoder.T)))
            np.testing.assert_allclose(data['target_a'], data['target_b'], atol=5e-6, rtol=0)
            row_delta = patterns['row_b'] - patterns['row_a']
            null_delta = patterns['null_b'] - patterns['null_a']
            np.testing.assert_allclose(row_delta, data['row_component'], atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(null_delta, data['null_component'], atol=1e-10, rtol=1e-10)
            selected = select_examples(patterns['row_a'], patterns['row_b'], patterns['null_a'],
                                       patterns['null_b'], time, ids, args.examples)
            for j, body in enumerate(ids):
                score_rows.append(dict(pair=stem, body_id=int(body), cell_type=types[j],
                    **{name: float(selected[name][j]) for name in ('row_a_rms', 'row_b_rms', 'row_shared_score', 'null_difference_rms')},
                    selected_row=bool(j in selected['row_selected']), selected_null=bool(j in selected['null_selected']),
                    row_temporal_correlation=correlation(patterns['row_a'][:-1, j], patterns['row_b'][:-1, j]),
                    actual_temporal_correlation=correlation(data['rates_a'][:-1, j], data['rates_b'][:-1, j])))
            norms = {key: np.linalg.norm(values, axis=1) for key, values in patterns.items()}
            norms.update(row_difference=np.linalg.norm(row_delta, axis=1), null_difference=np.linalg.norm(null_delta, axis=1))
            for tick, seconds in enumerate(time):
                record = dict(pair=stem, time_s=float(seconds), **{name + '_norm': float(v[tick]) for name, v in norms.items()})
                for label in ('a', 'b'):
                    for k, coord in enumerate(('x', 'y')):
                        record[f'prediction_{label}_{coord}_mm'] = float(data[f'prediction_{label}'][tick, k])
                        record[f'target_{label}_{coord}_mm'] = float(data[f'target_{label}'][tick, k])
                time_rows.append(record)
            summary = dict(pair=stem, checks=checks,
                row_labels=[f'{ids[j]} {types[j]}' for j in selected['row_selected']],
                null_labels=[f'{ids[j]} {types[j]}' for j in selected['null_selected']],
                overlap_body_ids=[int(ids[j]) for j in selected['row_selected'] if j in selected['null_selected']],
                initial_rate_max_abs={label: float(np.max(np.abs(data[f'rates_{label}'][0]))) for label in ('a', 'b')},
                row_a_rms=float(duration_rms(patterns['row_a'], time)),
                row_b_rms=float(duration_rms(patterns['row_b'], time)),
                row_difference_rms=float(duration_rms(row_delta, time)),
                null_difference_rms=float(duration_rms(null_delta, time)))
            report['pairs'].append(summary)
            np.savez_compressed(output / f'{stem}.npz', time_s=time, body_ids=ids, **patterns,
                **{name: data[name] for name in ('rates_a', 'rates_b', 'prediction_a', 'prediction_b', 'target_a', 'target_b')},
                selected_row_indices=selected['row_selected'], selected_null_indices=selected['null_selected'])
            plot_patterns(output, stem, time, data, patterns, selected, ids, types)
            print(f'Plotted {stem}', flush=True)
        write_csv(output / 'neuron_scores.csv', score_rows)
        write_csv(output / 'timecourses.csv', time_rows)
        write_json(output / 'patterns.json', report)
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
    parser.add_argument('--examples', type=int, default=5)
    return analyze(parser.parse_args(argv))


if __name__ == '__main__':
    main()
