"""Matched-state tuning figures and a portable report with explicit effect sizes."""

import html

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .activity_plots import save


def plot_pair(output, pair_id, body, cell_type, a, b, conditions, times, state_a, state_b, view):
    headings = sorted({row['heading_deg'] for row in conditions})
    directions = sorted({row['body_direction_deg'] for row in conditions})
    signed = view != 'raw'
    values = [a, b, b - a]
    limit = max(float(max(abs(a).max(), abs(b).max())), 1e-12)
    difference_limit = max(float(abs(b - a).max()), 1e-12)
    fig, axes = plt.subplots(len(times), 3, squeeze=False,
        figsize=(12, 3.2 * len(times) + .7), layout='constrained')
    for column, data in enumerate(values):
        for t, seconds in enumerate(times):
            grid = np.empty((len(headings), len(directions)))
            for k, row in enumerate(conditions):
                grid[headings.index(row['heading_deg']), directions.index(row['body_direction_deg'])] = data[k, t]
            ax = axes[t, column]
            vmax = difference_limit if column == 2 else limit
            image = ax.imshow(grid, origin='lower', aspect='auto',
                cmap='coolwarm' if signed or column == 2 else 'viridis',
                vmin=-vmax if signed or column == 2 else 0., vmax=vmax)
            name = ['State A', 'State B', 'B minus A'][column]
            ax.set(title=f'{name}; {seconds * 1000:g} ms', xlabel='Body movement (deg)', ylabel='Heading (deg)')
            ax.set_xticks(range(len(directions)), [f'{x:g}' for x in directions], rotation=45, fontsize=7)
            ax.set_yticks(range(len(headings)), [f'{x:g}' for x in headings], fontsize=7)
        fig.colorbar(image, ax=list(axes[:, column]), shrink=.6,
                     label='Rate difference (model units)' if signed or column == 2 else 'Rate (model units)')
    label = 'Actual rate' if view == 'raw' else 'Movement minus matched stationary continuation'
    fig.suptitle(f'{pair_id}: {cell_type} {body} — {label}\nA: {state_a}; B: {state_b}\n'
                 'A/B share one scale across times; B−A has its own symmetric scale. No per-map normalization.')
    stem = f'{pair_id}_neuron_{body}_{view}'
    save(fig, output, stem)
    return stem


def number(value):
    return '—' if value is None else f'{value:.4g}'


def write_report(output, report, summaries):
    intro = (
        'Identical heading/body-movement probes are replayed from independent copies of saved full-network states. '
        f'This uses checkpoint update {report["checkpoint_update"]}, at {report["speed_mm_s"]:g} mm/s, '
        f'{report["moving_conditions"]} movement conditions and {report["stationary_controls"]} stationary controls per state. '
        'Every archived path was replayed against its saved rates, inputs, targets and predictions before probing. '
        'No weights or decoder were refitted. Rates are model units, not Hz.')
    explanations = [
        'Raw mean shift measures the average B−A activity difference across the probe grid. It includes carried-over state.',
        'Movement-response range is the strongest minus weakest response after subtracting the same-heading stationary continuation from the same state. A/B ranges retain absolute model-unit amplitudes.',
        'Shape correlation ignores offsets and positive gain. A fitted B = offset + gain×A separates these from the remaining RMS residual. A negative gain is inversion. Fractional residual divides by the B range; it can be unstable when B is very weak.',
        'Flat maps (range ≤ 1e-6) have undefined shape correlation; fits from a flat A are undefined. Inspect both amplitudes before interpreting correlations or gains.',
        'Right-minus-left is an additional contrast at heading 90°. Order reversal requires both contrasts to exceed max(1e-6,1% of the larger map range). This is a descriptive threshold, not statistical significance.',
        'These are selected states from one model at one probe speed. History can change activity, responsiveness, or tuning without changing weights. This does not establish task-specific learning or physiological realism.',
    ]
    headers = ['Pair', 'Neuron', 'Probe ms', 'Raw mean B−A', 'Movement range A', 'Movement range B',
               'Shape r', 'Fitted gain', 'Residual / B range']
    md = ['# Matched probes from trajectory states', '', intro, '', '## Measurements', '']
    md.extend(f'- {line}' for line in explanations)
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Trajectory-state tuning comparisons</title>',
        '<style>body{font:16px sans-serif;max-width:1280px;margin:2em auto;padding:1em}img{max-width:100%}'
        'table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.35em}code{overflow-wrap:anywhere}</style>',
        '<h1>Matched probes from trajectory states</h1>', f'<p>{html.escape(intro)}</p>',
        '<p><a href="report.md">Portable written report</a> · <a href="focus_summary.csv">Focus summary, all probe times</a> · '
        '<a href="all_neuron_comparisons.csv">All neurons and response views</a> · '
        '<a href="pair_checks.csv">Matched-input and displacement checks</a> · '
        '<a href="trajectory_tuning.json">Definitions, checkpoint identity and replay checks</a></p>',
        '<ul>' + ''.join(f'<li>{html.escape(line)}</li>' for line in explanations) + '</ul>']
    for case in report['cases']:
        title = f'{case["case"]}: {case["title"]}'
        md.extend(['', f'## {title}', ''])
        parts.append(f'<h2>{html.escape(title)}</h2>')
        for pair in [p for p in report['pairs'] if p['case'] == case['case']]:
            text = (f'{pair["pair_id"]}: A = {pair["state_a"]}; B = {pair["state_b"]}. '
                    f'Identical current PFNs: {pair["identical_current_pfn"]}; '
                    f'target-vector difference {pair["target_vector_difference_mm"]:.4g} mm; '
                    f'initial neural-state difference norm {pair["initial_state_difference_norm"]:.4g}.')
            md.extend([text, ''])
            parts.append(f'<p>{html.escape(text)}</p>')
        # Two late samples for the written summary; CSV retains all four.
        available = sorted({row['time_s'] for row in summaries})[-2:]
        rows = [row for row in summaries if row['case'] == case['case'] and row['time_s'] in available]
        md.extend(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'])
        parts.append('<table><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr>')
        for row in rows:
            values = [row['pair_id'], str(row['body_id']), f'{row["time_s"] * 1000:g}',
                number(row['raw_mean_shift']), number(row['movement_range_a']), number(row['movement_range_b']),
                number(row['movement_pattern_correlation']), number(row['movement_fitted_gain']),
                number(row['movement_residual_fraction_of_b_range'])]
            md.append('| ' + ' | '.join(values) + ' |')
            parts.append('<tr>' + ''.join(f'<td>{html.escape(v)}</td>' for v in values) + '</tr>')
        md.append('')
        parts.append('</table>')
        for plot in [p for p in report['plots'] if p['case'] == case['case']]:
            stem = plot['stem']
            parts.append(f'<h3>{html.escape(stem)}</h3><a href="{stem}.pdf">PDF</a>'
                         f'<img loading="lazy" src="{stem}.png" alt="Matched-state tuning maps and paired differences">')
            md.append(f'- [{stem}: PNG]({stem}.png) · [PDF]({stem}.pdf)')
    md.extend(['', '## Reproduce', '',
        'The immutable manifest records exact input paths and hashes, cases, probe times, speed and angle grid. '
        'Use `python -m fly_connectome.home_vector.analysis.trajectory_tuning --help` for the CLI. '
        'A saved-state NPZ retains full probe rates and predictions, starting state, PFN inputs at onset, '
        'moving/control indices and sampled response maps. `neurons.csv` defines the neuron ordering.', ''])
    (output / 'report.md').write_text('\n'.join(md))
    (output / 'index.html').write_text('\n'.join(parts) + '</html>\n')
