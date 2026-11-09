"""Raw activity leaders and movement-after-pause figures."""

import html
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ..data import Circuit
from .activity_plots import save
from .activity_paths import norm_traces


def make_plots(output, snap, conditions, paths, restarts, report):
    circuit = Circuit(**snap['circuit'])
    focus = report['settings']['focus_body_id']
    leader_ids = sorted({r['most_often_highest_body_id'] for r in conditions if r['most_often_highest_body_id'] is not None})
    headings = sorted({r['heading_deg'] for r in conditions})
    directions = sorted({r['body_direction_deg'] for r in conditions})
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), layout='constrained')
    fig.suptitle(f'Raw-rate leaders over 8 mm paths; checkpoint {report["checkpoint_update"]}\n'
                 'No decoder weighting; fractions of active sampled time; ties share credit')
    for row, speed in enumerate([.5, 1., 2.]):
        for col, (field, title) in enumerate([
                ('most_often_highest_body_id', 'Most often highest: whole network'),
                ('focus_global_leading_fraction', f'{focus}: highest in whole network'),
                ('focus_within_population_leading_fraction', f'{focus}: highest within its population')]):
            grid = np.full((len(headings), len(directions)), np.nan)
            for condition in conditions:
                if condition['speed_mm_s'] != speed:
                    continue
                value = condition[field]
                if col == 0:
                    value = leader_ids.index(value) if value is not None else np.nan
                grid[headings.index(condition['heading_deg']), directions.index(condition['body_direction_deg'])] = np.nan if value is None else value
            ax = axes[row, col]
            if col == 0:
                colors = plt.get_cmap('tab20', max(len(leader_ids), 1))
                im = ax.imshow(grid, origin='lower', aspect='auto', cmap=colors,
                               vmin=-.5, vmax=max(len(leader_ids) - .5, .5))
                bar = fig.colorbar(im, ax=ax, ticks=range(len(leader_ids)))
                bar.ax.set_yticklabels([str(v) for v in leader_ids], fontsize=7)
                bar.set_label('Body ID (categorical)')
            else:
                im = ax.imshow(grid, origin='lower', aspect='auto', vmin=0, vmax=1, cmap='viridis')
                fig.colorbar(im, ax=ax, label='Fraction of active time')
            ax.set(title=f'{title}\n{speed:g} mm/s', xlabel='Body-centered direction (deg)', ylabel='Global heading (deg)')
            ax.set_xticks(range(len(directions)), [f'{v:g}' for v in directions], rotation=45)
            ax.set_yticks(range(len(headings)), [f'{v:g}' for v in headings])
    save(fig, output, 'raw_rate_sweep')

    decoder = snap['network'].D.detach().numpy()
    for row in paths:
        with np.load(output / f'{row["path"]}.npz') as a:
            time, rates = a['time_s'], a['rates']
            selected = list(np.argsort(-rates.mean(axis=0), kind='stable')[:6])
            if focus in circuit.dynamic_ids and circuit.dynamic_ids.index(focus) not in selected:
                selected.append(circuit.dynamic_ids.index(focus))
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
            fig.suptitle(f'{row["path"]}: {row["family"]}, {row["kind"]}; raw activity')
            position = -a['target']
            axes[0, 0].plot(position[:, 0], position[:, 1], color='black')
            axes[0, 0].scatter(position[[0, -1], 0], position[[0, -1], 1], c=['green', 'red'])
            axes[0, 0].set(xlabel='World x (mm)', ylabel='World y (mm)', title='World path: start green, end red', aspect='equal')
            leader = np.argmax(rates, axis=1)
            valid = rates.max(axis=1) > 1e-8
            categories = sorted(set(leader[valid]), key=lambda j: circuit.dynamic_ids[j])
            series = np.full(len(leader), np.nan)
            for k, j in enumerate(categories):
                series[(leader == j) & valid] = k
            axes[0, 1].step(time, series, where='post')
            axes[0, 1].set_yticks(range(len(categories)), [str(circuit.dynamic_ids[j]) for j in categories], fontsize=8)
            axes[0, 1].set(title='Instantaneous highest-rate unit\nTies shown by first circuit index; tables share tie credit',
                           xlabel='Time (s)', ylabel='Body ID (categorical)')
            for j in selected:
                axes[1, 0].plot(time, rates[:, j], label=f'{circuit.dynamic_ids[j]} ({circuit.dynamic_types[j]})')
            axes[1, 0].set(title='Six largest mean rates, plus focus unit', xlabel='Time (s)', ylabel='Rate (model units)')
            axes[1, 0].legend(fontsize=7, ncol=2)
            norms = norm_traces(rates, decoder, np.zeros(rates.shape[1]))
            for key, label in [('full_norm', 'Full state norm'), ('hidden_norm', 'Absolute decoder-hidden norm')]:
                axes[1, 1].plot(time, norms[key], label=label)
            axes[1, 1].set(xlabel='Time (s)', ylabel='Norm (model units)', title='Absolute state norms; no baseline subtraction')
            axes[1, 1].legend(fontsize=8)
            for boundary in row['record']['protocol'][1:-1]:
                # Curves have many closely spaced control changes; only segment
                # boundaries on random walks need visible event markers.
                if row['family'].startswith('random'):
                    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
                        ax.axvline(boundary[0], color='gray', alpha=.4, linewidth=.6)
            save(fig, output, row['path'] + '_activity')

    fig, axes = plt.subplots(len(restarts), 3, figsize=(16, 3.4 * len(restarts)), squeeze=False, layout='constrained')
    fig.suptitle('Movement after pauses: absolute norm versus fixed-reference hidden change\n'
                 'Highlighted branches selected for hidden-change reduction and full-norm drawdown')
    for i, case in enumerate(restarts):
        rows = case['rows']
        chosen = [0]
        chosen.extend(j for j, row in enumerate(rows) if row['kind'] == 'stationary' and row['heading_deg'] == 0. and j != 0)
        chosen.extend(j for j, row in enumerate(rows) if row['kind'] == 'moving' and row['heading_deg'] == 0.
                      and row['speed_mm_s'] == 1. and row['body_direction_deg'] in (0., 90., 180.))
        moving = [j for j, row in enumerate(rows) if row['kind'] in ('moving', 'path_moving')]
        best = min(moving, key=lambda j: rows[j]['hidden_change_from_pause_onset_norm_minimum_change'])
        has_drop = rows[best]['hidden_change_from_pause_onset_norm_minimum_change'] < -1e-4
        if not has_drop:
            best = min(moving, key=lambda j: rows[j]['full_norm_end'])
        if best not in chosen:
            chosen.append(best)
        protocols = json.loads((output / f'{case["stem"]}_protocols.json').read_text())
        matched = protocols[best].get('matched_control_index')
        if matched is None:
            matched = next(j for j, row in enumerate(rows) if row['kind'] == 'stationary'
                           and np.isclose(row['heading_deg'], rows[best]['heading_deg']))
        if matched not in chosen:
            chosen.append(matched)
        drawdown = max(moving, key=lambda j: rows[j]['full_norm_maximum_drawdown'])
        if rows[drawdown]['full_norm_maximum_drawdown'] > 1e-4:
            if drawdown not in chosen:
                chosen.append(drawdown)
            control = protocols[drawdown].get('matched_control_index')
            if control is not None and control not in chosen:
                chosen.append(control)
        with np.load(output / f'{case["stem"]}.npz') as a:
            for j in chosen:
                row = rows[j]
                if row['kind'] in ('path_moving', 'path_stationary'):
                    label = f'{row["path_family"]} {row["path_kind"]}' + (' heading-only' if row['kind'] == 'path_stationary' else '')
                elif row['kind'] == 'stationary':
                    label = 'Continued pause' if j == 0 else f'Stationary h={row["heading_deg"]:g}'
                else:
                    label = f'h={row["heading_deg"]:g}, body={row["body_direction_deg"]:g}, speed={row["speed_mm_s"]:g}'
                if j == best:
                    label += ' [largest hidden drop]' if has_drop else ' [smallest final norm]'
                if j == drawdown and rows[drawdown]['full_norm_maximum_drawdown'] > 1e-4:
                    label += ' [largest norm drawdown]'
                axes[i, 0].plot(a['time_s'], a['full_norm'][j], label=label)
                axes[i, 1].plot(a['time_s'], a['hidden_change_from_pause_onset_norm'][j], label=label)
                axes[i, 2].plot(a['time_s'], np.linalg.norm(a['prediction'][j] - a['target'][j], axis=1), label=label)
            axes[i, 0].axhline(case['full_norm_at_pause_onset'], color='gray', linestyle=':', label='Norm before pause')
        for col, title in enumerate(['Absolute ||r||', 'Hidden ||Pnull(r − r_pause_onset)||', 'Home-vector error']):
            axes[i, col].set(title=f'{case["history"].capitalize()}, pause {case["pause_s"]:g} s: {title}',
                xlabel='Time after restarting (s)', ylabel='mm' if col == 2 else 'Norm (model units)')
            axes[i, col].grid(alpha=.2)
            axes[i, col].legend(fontsize=6)
    save(fig, output, 'movement_after_pause')
    (output / 'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>Raw activity and movement after pauses</title><style>body{font:16px sans-serif;max-width:1200px;margin:2em auto;padding:1em}'
        'img{max-width:100%}</style>'
        f'<h1>Raw activity and movement after pauses: update {report["checkpoint_update"]}</h1>'
        '<p>Raw model activity, without decoder weighting. State norms use model units, not mm. '
        'Absolute state norm and change from pause onset are different measurements; the latter retains '
        'the same reference after walking resumes. Heading-only stationary controls accompany the movement grid.</p>'
        '<p><a href="paths_report.json">Settings and definitions</a> · '
        '<a href="raw_sweep_conditions.csv">Straight-path leaders</a> · '
        '<a href="raw_sweep_neurons.csv">All straight-path raw rates</a> · '
        '<a href="raw_paths.json">Varied-path index</a> · '
        '<a href="raw_path_neurons.csv">Varied-path neuron summaries</a> · '
        '<a href="resume_conditions.csv">All restart comparisons</a> · '
        '<a href="resume_cases.json">Restart case metadata</a></p>'
        '<h2>Raw-rate sweep</h2><a href="raw_rate_sweep.pdf">PDF</a><img src="raw_rate_sweep.png" alt="Raw-rate sweep">'
        '<h2>Movement after pauses</h2><a href="movement_after_pause.pdf">PDF</a>'
        '<img src="movement_after_pause.png" alt="Norms after movement resumes">' +
        ''.join(f'<h2>{html.escape(row["path"] + ": " + row["family"] + " / " + row["kind"])}</h2>'
                f'<a href="{row["path"]}_activity.pdf">PDF</a> · <a href="{row["path"]}.npz">Full activity arrays</a>'
                f'<img loading="lazy" src="{row["path"]}_activity.png" alt="Raw path activity">' for row in paths) + '</html>\n')
