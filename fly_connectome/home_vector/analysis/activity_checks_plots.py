"""Figures for descriptive, causal and pause-robustness checks."""

import html

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .activity_plots import save


def make_plots(output, conditions, neurons, interventions, pauses, pause_reports, report):
    focus = str(report['settings']['focus_body_id'])
    ids = sorted({str(r['body_id']) for r in neurons}, key=int)
    if focus not in ids:
        focus = ids[0]
    speeds = report['settings']['speeds']
    headings = sorted({r['heading_deg'] for r in conditions})
    directions = sorted({r['body_direction_deg'] for r in conditions})
    fig, axes = plt.subplots(len(speeds), 3, figsize=(14, 3.8 * len(speeds)),
                             squeeze=False, layout='constrained')
    fig.suptitle(f'Fixed {report["settings"]["distance_mm"]:g} mm paths; neuron {focus}; '
                 f'checkpoint {report["checkpoint_update"]}')
    for i, speed in enumerate(speeds):
        for col, (field, title, rows, limit) in enumerate([
                ('largest_rms_share', 'Largest individual share of summed RMS', conditions, 1.),
                ('rank', f'{focus}: direct contribution rank', [r for r in neurons if str(r['body_id']) == focus], len(ids)),
                ('direct_contribution_rms_mm', f'{focus}: contribution RMS (mm)', [r for r in neurons if str(r['body_id']) == focus],
                 max(r['direct_contribution_rms_mm'] for r in neurons if str(r['body_id']) == focus))]):
            grid = np.full((len(headings), len(directions)), np.nan)
            for row in rows:
                if row['speed_mm_s'] == speed:
                    grid[headings.index(row['heading_deg']), directions.index(row['body_direction_deg'])] = row[field] if row[field] is not None else np.nan
            ax = axes[i, col]
            im = ax.imshow(grid, origin='lower', aspect='auto', vmin=1 if field == 'rank' else 0,
                           vmax=max(limit, 1e-12), cmap='viridis_r' if field == 'rank' else 'viridis')
            ax.set(title=f'{title}\n{speed:g} mm/s', xlabel='Body-centered movement direction (deg)', ylabel='Global heading (deg)')
            ax.set_xticks(range(len(directions)), [f'{v:g}' for v in directions], rotation=45)
            ax.set_yticks(range(len(headings)), [f'{v:g}' for v in headings])
            fig.colorbar(im, ax=ax)
    save(fig, output, 'contribution_sweep')

    fig, ax = plt.subplots(figsize=(9, 7), layout='constrained')
    excluded = []
    plotted = []
    for body_id in ids:
        direct = [r for r in interventions if str(r['body_id']) == body_id and r['condition'] == 'readout_removed']
        causal = [r for r in interventions if str(r['body_id']) == body_id and r['condition'] == 'silenced_throughout']
        if any(r['status'] != 'finite' for r in causal):
            excluded.append(body_id)
            continue
        x, y = [float(np.sqrt(np.mean([r['prediction_change_rms_mm'] ** 2 for r in rows]))) for rows in (direct, causal)]
        ax.scatter(max(x, 1e-8), max(y, 1e-8), color='C3' if body_id == focus else 'C0', alpha=.7)
        plotted.append((body_id, x, y))
    for body_id, x, y in plotted:
        if body_id == focus or y >= sorted([v[2] for v in plotted])[-min(5, len(plotted))]:
            ax.annotate(body_id, (max(x, 1e-8), max(y, 1e-8)), fontsize=8)
    ax.set(xscale='log', yscale='log', xlabel='Remove direct readout: prediction-change RMS (mm)',
           ylabel='Silence neuron throughout: prediction-change RMS (mm)',
           title='Readout influence versus recurrent intervention\nEqual-trajectory RMS across 12 challenges; fixed decoder')
    ax.text(.02, .98, f'{len(excluded)} units omitted: numerical guard failed in ≥1 silenced challenge.\n'
            'All failures and task errors are retained in neuron_interventions.csv.',
            transform=ax.transAxes, va='top', fontsize=8)
    ax.grid(alpha=.2)
    save(fig, output, 'neuron_interventions')

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), squeeze=False, layout='constrained')
    fig.suptitle('Extended pauses: internal state, decoded drift and one-time state jitter')
    for i, history in enumerate(pause_reports):
        with np.load(output / f'{history["trajectory"]}_pause.npz') as a:
            time = a['time_s']
            for key, label in [('state_change_norm', 'Total'), ('visible_change_norm', 'Decoder-visible'),
                               ('hidden_change_norm', 'Decoder-hidden')]:
                axes[i, 0].plot(time, a[key], label=label)
            axes[i, 0].set(title=history['history'].capitalize() + ': neural state change', ylabel='State norm (model units)')
            axes[i, 1].plot(time, np.linalg.norm(a['prediction'] - a['prediction'][0], axis=1), label='Drift from pause onset')
            axes[i, 1].plot(time, np.linalg.norm(a['prediction'] - a['target'], axis=1), label='Error relative to true home')
            axes[i, 1].set(title='Unperturbed output', ylabel='Vector distance (mm)')
            for j, fraction in enumerate(report['settings']['jitter_fractions']):
                indices = [k for k, r in enumerate(history['variants']) if r['condition'] == 'state_jitter' and r['fraction'] == fraction]
                valid = [k for k in indices if all(r['status'] == 'finite' for r in pauses
                    if r['trajectory'] == history['trajectory'] and r['condition'] == 'state_jitter'
                    and r['fraction'] == fraction and r['replicate'] == history['variants'][k]['replicate'])]
                if valid:
                    curves = np.linalg.norm(a['variant_difference_from_sham'][valid], axis=2)
                    axes[i, 2].plot(time, np.median(curves, axis=0), color=f'C{j}', label=f'{100 * fraction:g}% initial state jitter')
                    axes[i, 2].fill_between(time, curves.min(axis=0), curves.max(axis=0), color=f'C{j}', alpha=.15)
            axes[i, 2].set(title='Difference from unperturbed pause\nMedian and min–max across seeded perturbations', ylabel='Output difference (mm)')
            for ax in axes[i]:
                ax.set_xlabel('Time since pause onset (s)')
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
    save(fig, output, 'pause_stability')
    (output / 'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>Frozen network activity checks</title><style>body{font:16px sans-serif;max-width:1200px;margin:2em auto;padding:1em}'
        'img{max-width:100%}</style>'
        f'<h1>Activity checks: update {report["checkpoint_update"]}</h1>'
        '<p>One frozen model, fixed decoder; developmental probes, not physiological validation. '
        'Contribution shares are not fractions of explained behavior. Whole-trial silencing affects '
        'encoding and retention. Pause perturbations are one-time state changes, not continuous noise.</p>'
        f'<p>Units with silencing guard failures: {html.escape(", ".join(excluded)) or "none"}.</p>'
        '<p><a href="diagnostics.json">Settings and definitions</a> · '
        '<a href="sweep_neurons.csv">Neuron contributions</a> · '
        '<a href="sweep_conditions.csv">Sweep task errors</a> · '
        '<a href="neuron_interventions.csv">All neuron interventions</a> · '
        '<a href="pause_interventions.csv">Pause perturbations</a> · '
        '<a href="pause_details.json">Pause checkpoints and variant ordering</a></p>' +
        ''.join(f'<h2>{label}</h2><p><a href="{stem}.pdf">PDF</a></p><img src="{stem}.png" alt="{label}">'
                for stem, label in [('contribution_sweep', 'Movement-condition contributions'),
                                    ('neuron_interventions', 'Readout and recurrent interventions'),
                                    ('pause_stability', 'Pause-state changes and sensitivity')]) + '</html>\n')
