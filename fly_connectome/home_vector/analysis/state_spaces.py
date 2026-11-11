"""Split paired curved-path activity differences through the frozen decoder."""

import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..__main__ import begin, fail, finish
from ..data import Circuit, digest, verify_bundle, write_json
from .activity import record_trajectory, write_csv
from .checkpoint import capture


CHECKPOINT_SHA256 = '2aab91f13258883819f66f12897919b2bf509617e7927120ca7f9db069d9a31f'
TARGET_ATOL_MM = 5e-6
PFN_ATOL = 1e-6


def split_difference(difference, decoder):
    """Orthogonal Euclidean split of B-A; never subtract its first time sample.

    D has output-by-neuron order. SVD supplies an orthonormal row-space basis;
    projection uses that basis, not the generally nonorthogonal rows of D.
    Components are signed activity differences, not realizable rate states.
    """
    delta, decoder = np.asarray(difference, dtype=float), np.asarray(decoder, dtype=float)
    if (decoder.ndim != 2 or min(decoder.shape) < 1 or delta.ndim < 1
            or delta.shape[-1] != decoder.shape[1]
            or not np.isfinite(decoder).all() or not np.isfinite(delta).all()):
        raise ValueError('Require finite activity differences and an aligned output-by-neuron decoder.')
    _, singular, vh = np.linalg.svd(decoder, full_matrices=False)
    tolerance = singular[0] * max(decoder.shape) * np.finfo(float).eps
    rank = int(np.sum(singular > tolerance))
    basis = vh[:rank]
    row = (delta @ basis.T) @ basis
    null = delta - row
    decoded = delta @ decoder.T
    checks = dict(
        reconstruction_max_abs=float(np.max(np.abs(delta - row - null))),
        null_output_max_abs=float(np.max(np.abs(null @ decoder.T))),
        row_output_max_abs_error=float(np.max(np.abs(row @ decoder.T - decoded))),
        orthogonality_max_abs=float(np.max(np.abs(np.sum(row * null, axis=-1)))))
    np.testing.assert_allclose(row + null, delta, atol=1e-10, rtol=1e-12)
    np.testing.assert_allclose(null @ decoder.T, 0., atol=1e-9, rtol=0)
    np.testing.assert_allclose(row @ decoder.T, decoded, atol=1e-9, rtol=1e-12)
    np.testing.assert_allclose(np.sum(row * null, axis=-1), 0., atol=1e-8, rtol=0)
    return dict(difference=delta, row_component=row, null_component=null,
                decoded_difference=decoded, row_basis=basis, singular_values=singular,
                rank=rank, rank_tolerance=float(tolerance), checks=checks)


def curved_pairs(rows):
    """Select all four archived curve families, without selecting on outcomes."""
    families = {}
    for row in rows:
        if row['family'].startswith('curve-'):
            family = families.setdefault(row['family'], {})
            if row['kind'] in family:
                raise ValueError('Duplicate curve traversal kind.')
            family[row['kind']] = row
    if not families:
        raise ValueError('No archived curved paths found.')
    pairs = []
    for family, variants in families.items():
        if set(variants) != {'forward', 'rightward'}:
            raise ValueError('Each curve family must have forward and rightward traversals.')
        pairs.append((family, variants['forward'], variants['rightward']))
    return pairs


def compare_pair(a, b, decoder):
    """Require matched time/displacement, and preserve input mismatch diagnostics."""
    for arrays in (a, b):
        n = len(arrays['time_s'])
        expected = dict(rates=(n, decoder.shape[1]), controls=(n, 3),
                        prediction=(n, 2), target=(n, 2))
        if any(arrays[k].shape != shape for k, shape in expected.items()):
            raise ValueError('Saved trajectory shapes are not aligned.')
        if any(not np.isfinite(arrays[k]).all() for k in (*expected, 'time_s', 'pfn')):
            raise ValueError('Nonfinite trajectory data.')
    np.testing.assert_allclose(a['time_s'], b['time_s'], atol=1e-10, rtol=0)
    np.testing.assert_allclose(a['target'], b['target'], atol=TARGET_ATOL_MM, rtol=0)
    parts = split_difference(b['rates'].astype(float) - a['rates'].astype(float), decoder)
    archived_delta = b['prediction'].astype(float) - a['prediction'].astype(float)
    np.testing.assert_allclose(parts['decoded_difference'], archived_delta, atol=3e-5, rtol=1e-5)
    parts['checks']['archived_output_max_abs_error_mm'] = float(np.max(np.abs(
        parts['decoded_difference'] - archived_delta)))
    ca, cb = a['controls'], b['controls']
    pfn_delta = b['pfn'].astype(float) - a['pfn'].astype(float)
    target_delta = b['target'].astype(float) - a['target'].astype(float)
    row_norm = np.linalg.norm(parts['row_component'], axis=1)
    null_norm = np.linalg.norm(parts['null_component'], axis=1)
    total_squared = row_norm ** 2 + null_norm ** 2
    # Undefined at identical states; serialize as blank in CSV, NaN in NPZ only.
    fraction = np.divide(null_norm ** 2, total_squared,
                         out=np.full_like(null_norm, np.nan), where=total_squared > 0)
    return parts, dict(
        time_s=a['time_s'], state_difference_norm=np.linalg.norm(parts['difference'], axis=1),
        row_component_norm=row_norm, null_component_norm=null_norm,
        null_squared_norm_fraction=fraction,
        decoded_difference_x_mm=parts['decoded_difference'][:, 0],
        decoded_difference_y_mm=parts['decoded_difference'][:, 1],
        decoded_difference_norm_mm=np.linalg.norm(parts['decoded_difference'], axis=1),
        target_difference_norm_mm=np.linalg.norm(target_delta, axis=1),
        error_a_mm=np.linalg.norm(a['prediction'].astype(float) - a['target'], axis=1),
        error_b_mm=np.linalg.norm(b['prediction'].astype(float) - b['target'], axis=1),
        heading_a_deg=ca[:, 0], heading_b_deg=cb[:, 0],
        heading_difference_deg=(cb[:, 0] - ca[:, 0] + 180) % 360 - 180,
        forward_a_mm_s=ca[:, 1], forward_b_mm_s=cb[:, 1],
        rightward_a_mm_s=ca[:, 2], rightward_b_mm_s=cb[:, 2],
        speed_a_mm_s=np.linalg.norm(ca[:, 1:], axis=1),
        speed_b_mm_s=np.linalg.norm(cb[:, 1:], axis=1),
        pfn_difference_norm=np.linalg.norm(pfn_delta, axis=1),
        pfn_difference_max_abs=np.max(np.abs(pfn_delta), axis=1),
        identical_pfn=np.all(a['pfn'] == b['pfn'], axis=1),
        pfn_equal_within_tolerance=np.max(np.abs(pfn_delta), axis=1) <= PFN_ATOL)


def scalar(value):
    value = value.item() if isinstance(value, np.generic) else value
    return None if isinstance(value, float) and not np.isfinite(value) else value


def plot_pair(output, stem, label, a, b, metrics):
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True, layout='constrained')
    t = metrics['time_s']
    for key, name in [('state_difference_norm', 'Total difference'),
                      ('row_component_norm', 'Row space: decoder-visible'),
                      ('null_component_norm', 'Null space: cancels now')]:
        axes[0].plot(t, metrics[key], label=name)
    axes[0].set(ylabel='Activity difference norm\n(model units)', title=label + ' — B minus A')
    for k, component in enumerate(('x', 'y')):
        axes[1].plot(t, metrics[f'decoded_difference_{component}_mm'], label=f'Decoded {component} difference')
        axes[1].plot(t, b['target'][:, k] - a['target'][:, k], ':', label=f'Correct {component} difference')
    axes[1].set(ylabel='Home-vector difference (mm)')
    axes[2].plot(t, metrics['error_a_mm'], label='A: forward walking')
    axes[2].plot(t, metrics['error_b_mm'], label='B: lateral walking')
    axes[2].set(ylabel='Vector error (mm)')
    axes[3].plot(t, metrics['pfn_difference_norm'], label='Full PFN input difference norm')
    axes[3].set(ylabel='PFN difference\n(encoder units)', xlabel='Time (s)')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'{stem}.{extension}', dpi=160)
    plt.close(fig)


def write_report(output, report, snapshots):
    lines = ['# Paired curved-path state differences', '',
        'Frozen best checkpoint: update 1975. No training, refitting or new continuations.', '',
        '## Reading the split', '',
        'At each matched time, subtract A (forward walking) from B (lateral walking). '
        'Project that 65-neuron difference onto the fixed decoder’s two-dimensional row space. '
        'The remainder is in its 63-dimensional null space. The row component reproduces '
        'the entire current decoded difference; the null component decodes to zero. '
        'These are signed differences, not separate populations or nonnegative neural states.', '',
        'Activity norms use model units; decoded differences and errors use mm. '
        'A small row-space norm is not itself a vector-error measurement. '
        'The null squared-norm fraction describes Euclidean geometry, not a fraction of information '
        'or task performance. It is undefined when both states are identical.', '',
        '## Comparisons', '',
        'All saved time samples are compared, including the terminal state. '
        'The terminal sample has stationary controls but no subsequent update in this archive. '
        'Summary RMS values below exclude that zero-duration endpoint.', '',
        '| Pair (A / B) | RMS activity difference | RMS row component | RMS null component | RMS decoded difference (mm) | Vector RMSE A / B (mm) |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in report['pairs']:
        lines.append(f"| {row['path_a']} / {row['path_b']} | {row['rms_state_difference']:.4f} | "
            f"{row['rms_row_component']:.4f} | {row['rms_null_component']:.4f} | "
            f"{row['rms_decoded_difference_mm']:.4f} | {row['vector_rmse_a_mm']:.4f} / {row['vector_rmse_b_mm']:.4f} |")
    worked = next((r for r in snapshots if r['pair'] == 'path_12_vs_path_13' and r['time_s'] == 4.), None)
    worked_text = ''
    if worked is not None:
        worked_text = (
            f"For paths 12/13 at 4 s, the total activity-difference norm is {worked['state_difference_norm']:.3f}; "
            f"the row component has norm {worked['row_component_norm']:.3f} and the null component "
            f"{worked['null_component_norm']:.3f}. These norms combine by squared sum, not addition. "
            f"The null component accounts for {100 * worked['null_squared_norm_fraction']:.3f}% of the "
            f"squared difference norm, but the decoded vectors still differ by "
            f"{worked['decoded_difference_norm_mm']:.3f} mm despite identical correct vectors. "
            f"Individual errors are {worked['error_a_mm']:.3f} mm (forward) and "
            f"{worked['error_b_mm']:.3f} mm (lateral). Both speeds are 1 mm/s; headings are "
            f"{worked['heading_a_deg']:.0f} and {worked['heading_b_deg']:.0f} degrees. PFN inputs differ. "
            'This is strong cancellation in activity space, accompanied by residual output disagreement; '
            'null-space dominance alone does not establish accurate or history-invariant computation.')
        lines += ['', '## Worked example: paths 12/13 at 4 s', '', worked_text]
    lines += ['', '## Interpretation limits', '',
        'The paired displacements match throughout, but headings and body velocities differ. '
        'Input equality is checked from all 60 actual PFN traces, with exact equality and '
        f'a separate absolute tolerance of {PFN_ATOL:g}. These original trajectories do not isolate '
        'history from current-input effects. Identical-input continuations belong to the next experiment.', '',
        'Large neural differences with small decoded differences indicate cancellation at the current '
        'readout. Both individual errors must also be examined: similar predictions can both be wrong. '
        'No accuracy cutoff or causal role for recurrence is inferred here. '
        'Null-space differences may affect future outputs, including through unequal passive decay rates.', '',
        '## Files and worked snapshots', '',
        '`snapshots.csv` gives matched states at 0, 2, 4, 6, 7.5 and 8 s when available. '
        '`snapshot_neurons.csv` shows the signed total, row and null differences for every neuron '
        'at these times. `timecourses.csv` retains all times and control/input diagnostics. '
        'Each pair NPZ retains the complete component vectors, original rates, controls, PFNs, '
        'targets and predictions. `decoder.npz` preserves D, intercept, SVD basis and neuron IDs. '
        '`state_spaces.json` records projection checks, replay checks and tolerances.', '']
    for row in report['pairs']:
        lines += [f"### {row['family']}: {row['path_a']} / {row['path_b']}", '',
                  f"![State and output differences]({row['pair']}.png)", '',
                  f"[PDF]({row['pair']}.pdf) · [Full arrays]({row['pair']}.npz)", '']
    (output / 'report.md').write_text('\n'.join(lines))
    table_rows = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in (
        r['pair'], f"{r['rms_state_difference']:.3f}", f"{r['rms_row_component']:.3f}",
        f"{r['rms_null_component']:.3f}", f"{r['rms_decoded_difference_mm']:.3f}",
        f"{r['vector_rmse_a_mm']:.3f} / {r['vector_rmse_b_mm']:.3f}")) + '</tr>' for r in report['pairs'])
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8">'
        '<title>Paired state differences</title><style>body{font:16px sans-serif;max-width:1050px;'
        'margin:2em auto;line-height:1.5}img{max-width:100%}td,th{padding:.5em;border:1px solid #ddd}'
        'table{border-collapse:collapse}</style><h1>Paired curved-path state differences</h1>'
        '<p>Frozen best checkpoint at update 1975. A = forward walking; B = lateral walking.</p>'
        '<p>At each matched time, split B minus A into a row-space component that reproduces the '
        'decoded difference and a null-space component that cancels at the current readout. '
        'Activity norms are in model units; decoded differences and errors are in mm.</p>'
        '<table><tr><th>Pair</th><th>RMS total norm</th><th>RMS row norm</th><th>RMS null norm</th>'
        '<th>RMS decoded difference (mm)</th><th>Vector RMSE A / B (mm)</th></tr>' + table_rows + '</table>'
        '<p>RMS summaries exclude the terminal sample. Matched displacements do not imply matched PFNs; '
        'these comparisons do not isolate history effects or establish recurrent necessity.</p>'
        + ('<h2>Worked example: paths 12/13 at 4 s</h2><p>' + html.escape(worked_text) + '</p>' if worked_text else '')
        + '<p><a href="report.md">Full written report and interpretation limits</a> · '
        '<a href="snapshots.csv">Snapshots</a> · <a href="snapshot_neurons.csv">Neuron components</a> · '
        '<a href="timecourses.csv">Full timecourses</a></p>'
        + ''.join(f'<h2>{html.escape(r["family"])}</h2><img alt="Paired state and output differences" '
                  f'src="{r["pair"]}.png"><p><a href="{r["pair"]}.pdf">PDF</a> · '
                  f'<a href="{r["pair"]}.npz">Full arrays</a></p>' for r in report['pairs']))


def analyze(args):
    activity, source, output = [Path(getattr(args, k)).resolve() for k in ('activity', 'source', 'output')]
    if args.threads < 1:
        raise ValueError('Require a positive CPU thread count.')
    for folder in (activity, source):
        if output == folder or output.is_relative_to(folder) or folder.is_relative_to(output):
            raise ValueError('Output must be separate from input bundles.')
    if output.exists():
        raise FileExistsError(output)
    if verify_bundle(activity)['kind'] != 'home_vector_activity_paths_v1':
        raise ValueError('Require the archived per-trajectory activity bundle.')
    torch.set_num_threads(args.threads)
    snap = capture(activity, source)
    if snap['checkpoint_sha256'] != CHECKPOINT_SHA256 or snap['checkpoint']['update'] != 1975:
        raise ValueError('This investigation requires the specified frozen best checkpoint at update 1975.')
    circuit, network, dt = Circuit(**snap['circuit']), snap['network'], snap['task']['dt_s']
    with (activity / 'neurons.csv').open() as stream:
        neurons = list(csv.DictReader(stream))
    if ([int(r['body_id']) for r in neurons] != circuit.dynamic_ids
            or [r['cell_type'] for r in neurons] != circuit.dynamic_types):
        raise ValueError('Archived neuron identities differ from checkpoint.')
    pairs = curved_pairs(json.loads((activity / 'raw_paths.json').read_text()))
    frozen = {str(p): digest(p) for folder in (activity, source) for p in folder.iterdir() if p.is_file()}
    loaded, replays = {}, []
    for _, *members in pairs:
        for member in members:
            stem = member['path']
            with np.load(activity / f'{stem}.npz') as saved:
                arrays = {k: saved[k].copy() for k in saved.files}
            np.testing.assert_allclose(arrays['time_s'], np.arange(len(arrays['rates'])) * dt, atol=1e-10, rtol=0)
            replay, _ = record_trajectory(network, circuit, member['record'], dt)
            checks = {}
            for key in ('controls', 'pfn', 'rates', 'prediction', 'target'):
                np.testing.assert_allclose(replay[key], arrays[key], atol=3e-5, rtol=1e-6)
                checks[key + '_max_abs_error'] = float(np.max(np.abs(replay[key] - arrays[key])))
            replays.append(dict(path=stem, **checks))
            loaded[stem] = arrays
            print(f'Verified {stem}', flush=True)
    output, manifest = begin(output, 'home_vector_paired_state_spaces_v1',
        activity=str(activity), source=str(source), source_manifest_sha256=digest(source / 'manifest.json'),
        activity_manifest_sha256=digest(activity / 'manifest.json'))
    manifest.update(refitting=False, device='cpu', torch=torch.__version__, threads=args.threads,
        immutable_input_sha256=frozen,
        analysis_code_sha256={p.name: digest(p) for p in Path(__file__).parent.glob('*.py')})
    write_json(output / 'manifest.json', manifest)
    try:
        for name in ('current_checkpoint.pt', 'current_metadata.json', 'neurons.csv', 'raw_paths.json'):
            (output / name).write_bytes((activity / name).read_bytes())
        decoder, intercept = network.D.detach().numpy().astype(float), network.d.detach().numpy().astype(float)
        report = dict(checkpoint_update=1975, checkpoint_sha256=snap['checkpoint_sha256'], dt_s=dt,
            difference_direction='B minus A; A forward, B rightward',
            target_match_atol_mm=TARGET_ATOL_MM, pfn_match_atol=PFN_ATOL,
            state_timing='State at t precedes the update driven by PFN(t).',
            replay_verification=replays, pairs=[])
        timecourses, snapshots, neuron_rows = [], [], []
        for family, aa, bb in pairs:
            a, b = loaded[aa['path']], loaded[bb['path']]
            parts, metrics = compare_pair(a, b, decoder)
            stem = aa['path'] + '_vs_' + bb['path']
            prefix = dict(pair=stem, family=family, path_a=aa['path'], path_b=bb['path'])
            rows = [dict(**prefix, **{k: scalar(v[t]) for k, v in metrics.items()})
                    for t in range(len(a['time_s']))]
            timecourses.extend(rows)
            for seconds in (0., 2., 4., 6., 7.5, 8.):
                tick = round(seconds / dt)
                if tick >= len(rows):
                    continue
                snapshots.append(rows[tick])
                for j, body in enumerate(circuit.dynamic_ids):
                    neuron_rows.append(dict(**prefix, time_s=float(a['time_s'][tick]), body_id=body,
                        cell_type=circuit.dynamic_types[j], rate_a=float(a['rates'][tick, j]),
                        rate_b=float(b['rates'][tick, j]), difference=float(parts['difference'][tick, j]),
                        row_component=float(parts['row_component'][tick, j]),
                        null_component=float(parts['null_component'][tick, j])))
            summary = dict(**prefix, samples=len(rows), projection_checks=parts['checks'],
                max_target_difference_mm=float(metrics['target_difference_norm_mm'].max()),
                identical_pfn_samples=int(metrics['identical_pfn'].sum()),
                pfn_equal_within_tolerance_samples=int(metrics['pfn_equal_within_tolerance'].sum()))
            for name, key in [('rms_state_difference', 'state_difference_norm'),
                              ('rms_row_component', 'row_component_norm'),
                              ('rms_null_component', 'null_component_norm'),
                              ('rms_decoded_difference_mm', 'decoded_difference_norm_mm'),
                              ('vector_rmse_a_mm', 'error_a_mm'), ('vector_rmse_b_mm', 'error_b_mm')]:
                summary[name] = float(np.sqrt(np.mean(metrics[key][:-1] ** 2)))
            report['pairs'].append(summary)
            np.savez_compressed(output / f'{stem}.npz', **metrics,
                **{k: parts[k] for k in ('difference', 'row_component', 'null_component', 'decoded_difference')},
                **{f'{k}_{label}': arrays[k] for label, arrays in (('a', a), ('b', b))
                   for k in ('rates', 'pfn', 'controls', 'target', 'prediction')})
            plot_pair(output, stem, family, a, b, metrics)
        report.update(decoder_rank=parts['rank'], decoder_nullity=len(circuit.dynamic_ids) - parts['rank'],
            decoder_singular_values=parts['singular_values'].tolist(), rank_tolerance=parts['rank_tolerance'])
        np.savez_compressed(output / 'decoder.npz', decoder=decoder, intercept=intercept,
            body_ids=circuit.dynamic_ids, row_basis=parts['row_basis'], singular_values=parts['singular_values'])
        write_csv(output / 'timecourses.csv', timecourses)
        write_csv(output / 'snapshots.csv', snapshots)
        write_csv(output / 'snapshot_neurons.csv', neuron_rows)
        write_json(output / 'state_spaces.json', report)
        write_report(output, report, snapshots)
        if any(digest(path) != sha for path, sha in frozen.items()):
            raise ValueError('An immutable input file changed during analysis.')
        finish(output, manifest)
    except BaseException as error:
        fail(output, manifest, error)
        raise
    print(json.dumps(dict(output=str(output), status='complete', pairs=len(pairs)), indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('activity', 'source', 'output'):
        parser.add_argument(f'--{name}', required=True)
    parser.add_argument('--threads', type=int, default=1)
    return analyze(parser.parse_args(argv))


if __name__ == '__main__':
    main()
