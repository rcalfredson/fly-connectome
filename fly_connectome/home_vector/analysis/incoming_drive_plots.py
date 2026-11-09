"""Incoming-drive traces, signed partner contrasts and portable explanations."""

import html

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .activity_plots import save


def plot_path(output, path, circuit, time, parts, contributions, col, j, record):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, layout='constrained')
    fig.suptitle(f'{path}: incoming drive to {circuit.dynamic_types[j]} {circuit.dynamic_ids[j]}\n'
                 'Original trajectory, fixed trained weights; signed preactivation contributions')
    for key, label in [('rate', 'Actual rate'), ('rectified_target', 'Rectified target rate')]:
        axes[0].plot(time, parts[key][:, col], label=label)
    axes[0].set_ylabel('Rate (model units)')
    for key, label in [('input_drive', 'PFN drive'), ('recurrent_drive', 'Recurrent drive'),
                       ('preactivation', 'Total preactivation incl. bias')]:
        axes[1].plot(time, parts[key][:, col], label=label)
    axes[1].axhline(0., color='black', linewidth=.6)
    axes[1].set_ylabel('Signed drive')
    for k, kind in enumerate(circuit.population_types):
        axes[2].plot(time, parts['recurrent_by_population'][:, k, col], label=kind)
    axes[2].axhline(0., color='black', linewidth=.6)
    axes[2].set_ylabel('Recurrent drive\nby population')
    top = np.argsort(-np.mean(abs(contributions), axis=0), kind='stable')[:6]
    for k in top:
        if np.any(contributions[:, k] != 0):
            axes[3].plot(time, contributions[:, k], label=f'{circuit.dynamic_ids[k]} ({circuit.dynamic_types[k]})')
    axes[3].axhline(0., color='black', linewidth=.6)
    axes[3].set(ylabel='Signed partner drive', xlabel='Time (s)',
                title='Six largest mean absolute partner contributions; full arrays retain all partners')
    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, ncol=3)
        ax.grid(alpha=.15)
        if record.get('family', '').startswith('random'):
            for boundary in record['protocol'][1:-1]:
                ax.axvline(boundary[0], color='gray', linewidth=.6)
    stem = f'{path}_neuron_{circuit.dynamic_ids[j]}_incoming_drive'
    save(fig, output, stem)
    return stem


def plot_pair(output, row, partners, populations):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
    fig.suptitle(f'{row["pair_id"]}: {row["cell_type"]} {row["body_id"]}\n'
                 f'A: {row["state_a"]}; B: {row["state_b"]}\nOriginal trajectory inputs at each saved state; these are not standardized probes')
    fields = ['rate', 'input_drive', 'recurrent_drive', 'rectified_target']
    x = np.arange(len(fields))
    for shift, suffix, label in [(-.18, 'a', 'State A'), (.18, 'b', 'State B')]:
        axes[0].bar(x + shift, [row[f'{key}_{suffix}'] for key in fields], width=.36, label=label)
    axes[0].set_xticks(x, ['Rate', 'PFN drive', 'Recurrent\ndrive', 'ReLU target'], rotation=20)
    axes[0].set_ylabel('Model units')
    axes[0].legend()
    diffs = [row[f'{p}_recurrent_difference'] for p in populations]
    axes[1].barh(populations, diffs, color=['C0' if v >= 0 else 'C1' for v in diffs])
    axes[1].set(title='Recurrent-drive difference by population', xlabel='B minus A (signed drive)')
    top = sorted(partners, key=lambda item: abs(item['drive_difference']), reverse=True)[:8]
    axes[2].barh([f'{r["presynaptic_body_id"]} ({r["presynaptic_type"]})' for r in top][::-1],
                 [r['drive_difference'] for r in top][::-1])
    axes[2].set(title='Largest individual partner differences', xlabel='B minus A (signed drive)')
    for ax in axes:
        if ax is axes[0]:
            ax.axhline(0., color='gray', linewidth=.5)
        else:
            ax.axvline(0., color='gray', linewidth=.5)
        ax.grid(alpha=.15)
    stem = f'{row["pair_id"]}_neuron_{row["body_id"]}_presynaptic_difference'
    save(fig, output, stem)
    return stem


def write_report(output, report, snapshots, partners, gates):
    explanations = [
        'PFN drive is the sum of fixed input weights times current prescribed PFN rates. Recurrent drive is the sum of recurrent weights times dynamic-neuron rates.',
        'The rectified target is max(0, bias + PFN drive + recurrent drive). The next change is alpha times that target minus alpha times the current rate. Thus incoming drive and carried-over target activity are separate terms.',
        'Signed partner contributions sum exactly to recurrent drive. They can cancel; a large contribution is not an ablation result. Self-edges, if present, remain in recurrence and are flagged in the partner CSV.',
        'Snapshot figures use ORIGINAL trajectory inputs at the saved time. Cases with different headings or body motions need the separate identical-probe accounting before attributing differences solely to recurrent state.',
        'The identical-probe first-update table holds PFN input fixed across A/B. Its preactivation difference then comes entirely from recurrent drive; its rate-change difference also includes the target neuron’s own carried-over activity.',
        'Probe rectification fractions count nonpositive preactivation over conditions and update times. They describe gate occupancy, not causality, trials, or biological firing probabilities.',
        'This is exact local accounting under the trained weights. It does not establish which neurons are necessary, why training selected these weights, or whether the mechanism matches physiology.',
    ]
    md = ['# Incoming-drive accounting', '',
          f'Frozen checkpoint update {report["checkpoint_update"]}; no refitting. Rates and signed drives use model units.', '',
          '## Interpretation', '', *[f'- {text}' for text in explanations], '',
          '## Snapshot comparisons', '',
          '| Pair | Target | Rate A | Rate B | PFN drive B−A | Recurrent drive B−A | Largest signed recurrent-drive differences |',
          '| --- | --- | ---: | ---: | ---: | ---: | --- |']
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Incoming-drive accounting</title>',
        '<style>body{font:16px sans-serif;max-width:1400px;margin:2em auto;padding:1em}img{max-width:100%}'
        'table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.35em}</style>',
        f'<h1>Incoming-drive accounting: update {report["checkpoint_update"]}</h1>',
        '<p><a href="report.md">Portable report</a> · <a href="paired_snapshot_drives.csv">Snapshot table</a> · '
        '<a href="presynaptic_partner_differences.csv">Every recurrent partner</a> · '
        '<a href="identical_probe_first_updates.csv">Identical-input update accounting</a> · '
        '<a href="probe_rectification.csv">Probe rectification</a> · '
        '<a href="incoming_drive.json">Definitions and reconstruction checks</a></p>',
        '<ul>' + ''.join(f'<li>{html.escape(t)}</li>' for t in explanations) + '</ul>',
        '<h2>Saved-state comparisons under their original inputs</h2>',
        '<table><tr><th>Pair</th><th>Target</th><th>Rate A</th><th>Rate B</th><th>PFN Δdrive</th>'
        '<th>Recurrent Δdrive</th><th>Largest signed recurrent partner Δdrive</th></tr>']
    for row in snapshots:
        top = sorted([r for r in partners if r['pair_id'] == row['pair_id'] and r['body_id'] == row['body_id']],
                     key=lambda r: abs(r['drive_difference']), reverse=True)[:3]
        drivers = '; '.join(f'{r["presynaptic_body_id"]} ({r["presynaptic_type"]}): {r["drive_difference"]:+.3g}' for r in top)
        cells = [row['pair_id'], str(row['body_id']), f'{row["rate_a"]:.4g}', f'{row["rate_b"]:.4g}',
                 f'{row["input_drive_difference"]:+.4g}', f'{row["recurrent_drive_difference"]:+.4g}', drivers]
        md.append('| ' + ' | '.join(cells) + ' |')
        parts.append('<tr>' + ''.join(f'<td>{html.escape(t)}</td>' for t in cells) + '</tr>')
    parts.append('</table>')
    md.extend(['', '## Figures', ''])
    for plot in report['plots']:
        stem = plot['stem']
        parts.append(f'<h3>{html.escape(stem)}</h3><a href="{stem}.pdf">PDF</a>'
                     f'<img loading="lazy" src="{stem}.png" alt="Incoming-drive accounting">')
        md.append(f'- [{stem}: PNG]({stem}.png) · [PDF]({stem}.pdf)')
    (output / 'report.md').write_text('\n'.join(md) + '\n')
    (output / 'index.html').write_text('\n'.join(parts) + '</html>\n')
