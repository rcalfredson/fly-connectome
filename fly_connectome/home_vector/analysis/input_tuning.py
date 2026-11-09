"""Map PFN input mixtures to all neurons' standardized zero-state onset tuning."""

import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..__main__ import begin, fail, finish
from ..data import Circuit, digest, encode, verify_bundle, write_json
from ..model import HomeVectorNetwork
from .activity import write_csv
from .activity_plots import save


def first_step(pfn, weight, bias, tau, dt):
    """Zero-state input/bias preactivation, rectification and exact rate increment."""
    drive = np.asarray(pfn, dtype=float) @ np.asarray(weight, dtype=float)
    z = drive + bias
    return drive, z, np.maximum(z, 0.) * (-np.expm1(-dt / tau))


def plot_population(output, circuit, indices, conditions, rates, stationary, tick, time, kind, population, columns=5):
    if columns < 1:
        raise ValueError('Require a positive subplot column count.')
    headings = sorted({row['heading_deg'] for row in conditions})
    directions = sorted({row['body_direction_deg'] for row in conditions})
    data = rates[indices, tick].copy()
    if kind == 'movement_delta':
        data -= rates[stationary, tick]
    if kind == 'pattern':
        low, span = data.min(axis=0), np.ptp(data, axis=0)
        data = np.divide(data - low, span, out=np.full_like(data, np.nan), where=span > 1e-6)
    units = sorted([j for j, value in enumerate(circuit.dynamic_types) if value == population],
                   key=lambda j: circuit.dynamic_ids[j])
    columns = min(columns, len(units))
    fig, axes = plt.subplots(int(np.ceil(len(units) / columns)), columns, squeeze=False,
        figsize=(3 * columns, 2.8 * int(np.ceil(len(units) / columns)) + 1), layout='constrained')
    signed = kind == 'movement_delta'
    finite = data[:, units][np.isfinite(data[:, units])]
    limit = float(np.max(np.abs(finite))) if len(finite) else 0.
    label = ('Per-neuron normalized pattern; amplitude removed' if kind == 'pattern' else
             'Movement minus same-heading stationary' if signed else 'Raw rate')
    fig.suptitle(f'{population}: {label}, {time * 1000:g} ms, 1 mm/s\n' +
                 ('Identical zero states; each unit spans 0=min to 1=max; flat responses gray' if kind == 'pattern' else
                  'Identical zero starting states; shared model-unit scale within this figure'))
    for ax, j in zip(axes.flat, units):
        grid = np.zeros((len(headings), len(directions)))
        for k, condition in enumerate(conditions):
            grid[headings.index(condition['heading_deg']), directions.index(condition['body_direction_deg'])] = data[k, j]
        cmap = plt.get_cmap('coolwarm' if signed else 'viridis').copy()
        cmap.set_bad('lightgray')
        image = ax.imshow(grid, origin='lower', aspect='auto', cmap=cmap,
                          vmin=-max(limit, 1e-12) if signed else 0., vmax=max(limit, 1e-12))
        ax.set(title=f'Body {circuit.dynamic_ids[j]}', xlabel='Body movement (deg)', ylabel='Heading (deg)')
        ax.set_xticks(range(len(directions)), [f'{v:g}' for v in directions], rotation=45, fontsize=6)
        ax.set_yticks(range(len(headings)), [f'{v:g}' for v in headings], fontsize=6)
    for ax in list(axes.flat)[len(units):]:
        ax.set_visible(False)
    if limit > 0:
        fig.colorbar(image, ax=list(axes.flat)[:len(units)],
            label='Per-neuron min–max pattern' if kind == 'pattern' else 'Rate difference (model units)' if signed else 'Rate (model units)', shrink=.7)
    else:
        fig.suptitle(fig.get_suptitle() + ('\nAll tuning patterns are flat.' if kind == 'pattern' else '\nAll responses in this view are zero.'))
    stem = f'{population}_{kind}_{round(time * 1e6):07d}us'
    save(fig, output, stem)
    return stem


def analyze(args):
    if args.columns < 1:
        raise ValueError('Require a positive subplot column count.')
    onset, output = Path(args.onset).resolve(), Path(args.output).resolve()
    if output == onset or output.is_relative_to(onset) or onset.is_relative_to(output):
        raise ValueError('Output must be separate from the input bundle.')
    if output.exists():
        raise FileExistsError(output)
    if verify_bundle(onset)['kind'] != 'home_vector_onset_probe_v1':
        raise ValueError('Require a completed standardized onset probe.')
    metadata = json.loads((onset / 'current_metadata.json').read_text())
    source_report = json.loads((onset / 'report.json').read_text())
    sha = digest(onset / 'current_checkpoint.pt')
    if sha != source_report['checkpoint_sha256'] or sha != metadata['checkpoint_sha256']:
        raise ValueError('Onset responses and checkpoint identities differ.')
    circuit = Circuit(**metadata['circuit'])
    checkpoint = torch.load(onset / 'current_checkpoint.pt', map_location='cpu', weights_only=False)
    if checkpoint['update'] != source_report['checkpoint_update']:
        raise ValueError('Onset checkpoint update differs.')
    settings = metadata['run_manifest']['settings']
    network = HomeVectorNetwork(circuit, seed=settings['seed'], recurrence=settings.get('recurrence', True))
    for key in ('pre', 'post', 'counts', 'type_index', 'a_initial'):
        torch.testing.assert_close(network.state_dict()[key], checkpoint['model'][key], rtol=0, atol=0)
    network.load_state_dict(checkpoint['model'])
    with torch.no_grad():
        weight = network.weights()[:network.input_size].numpy()
        bias = network.bias[network.type_index].numpy()
        tau = network.tau[network.type_index].numpy()
    conditions = json.loads((onset / 'conditions.json').read_text())
    with np.load(onset / 'onset_rates.npz') as arrays:
        rates, times, pfn = [arrays[key].copy() for key in ('rates', 'time_s', 'input_at_onset')]
    if (rates.shape != (len(conditions), len(times), network.size)
            or pfn.shape != (len(conditions), network.input_size)
            or not all(np.isfinite(a).all() for a in (rates, times, pfn))
            or not np.allclose(np.diff(times), source_report['dt_s'], atol=1e-12, rtol=0)):
        raise ValueError('Invalid onset array dimensions, times or values.')
    np.testing.assert_array_equal(rates[:, 0], 0.)
    np.testing.assert_allclose(pfn, encode(circuit, np.array([r['protocol'][0][1:] for r in conditions])), rtol=1e-6, atol=1e-6)
    _, _, reconstructed = first_step(pfn, weight, bias, tau, source_report['dt_s'])
    np.testing.assert_allclose(reconstructed, rates[:, 1], rtol=1e-5, atol=1e-6)
    indices = [k for k, row in enumerate(conditions) if row['kind'] == 'moving' and row['speed_mm_s'] == 1.]
    if not indices:
        raise ValueError('Require 1 mm/s movement conditions.')
    selected = [conditions[k] for k in indices]
    stationary = [next(k for k, row in enumerate(conditions) if row['kind'] == 'stationary'
                       and row['heading_deg'] == condition['heading_deg']) for condition in selected]
    frozen = {str(p): digest(p) for p in onset.iterdir() if p.is_file()}
    output, manifest = begin(output, 'home_vector_input_tuning_v1', onset=str(onset), onset_manifest_sha256=digest(onset / 'manifest.json'))
    manifest.update(refitting=False, immutable_input_sha256=frozen,
        analysis_code_sha256={p.name: digest(p) for p in Path(__file__).parent.glob('*.py')})
    write_json(output / 'manifest.json', manifest)
    try:
        edge_rows, neuron_rows = [], []
        sample_ticks = sorted({1, int(np.argmin(abs(times - .05))), int(np.argmin(abs(times - 1.)))})
        for j, body_id in enumerate(circuit.dynamic_ids):
            edges = [e for e in circuit.edges if e['pre'] < network.input_size and e['post'] == j]
            scores = []
            for edge in edges:
                i = edge['pre']
                raw_drive = pfn[indices, i].astype(float) * weight[i, j]
                movement_drive = (pfn[indices, i].astype(float) - pfn[stationary, i]) * weight[i, j]
                score = float(np.sqrt(np.mean(movement_drive ** 2)))
                scores.append(score)
                edge_rows.append(dict(target_body_id=body_id, target_type=circuit.dynamic_types[j],
                    input_body_id=circuit.input_ids[i], input_type=circuit.input_types[i],
                    input_heading_deg=circuit.preferred_heading_deg[i], input_body_direction_deg=circuit.preferred_translation_deg[i],
                    contacts=edge['contacts'], denominator=edge['denominator'], effective_weight=float(weight[i, j]),
                    raw_preactivation_rms=float(np.sqrt(np.mean(raw_drive ** 2))), movement_preactivation_rms=score))
            top = edges[int(np.argmax(scores))] if scores and max(scores) > 0 else None
            first = rates[indices, 1, j]
            modulation = float(np.ptp(first))
            peak = selected[int(np.argmax(first))] if modulation > 1e-6 else None
            row = dict(body_id=body_id, cell_type=circuit.dynamic_types[j], direct_pfn_pairs=len(edges),
                positive_input_weights=sum(weight[e['pre'], j] > 0 for e in edges),
                negative_input_weights=sum(weight[e['pre'], j] < 0 for e in edges),
                bias=float(bias[j]), tau_s=float(tau[j]),
                top_movement_input_body_id=circuit.input_ids[top['pre']] if top else None,
                top_movement_input_type=circuit.input_types[top['pre']] if top else None,
                top_input_share_of_summed_movement_rms=max(scores) / sum(scores) if scores and sum(scores) else None,
                first_step_modulation_range=modulation,
                first_step_peak_tied_grid_points=int(np.isclose(first, first.max(), rtol=1e-6, atol=1e-8).sum()),
                first_step_peak_heading_deg=peak['heading_deg'] if peak else None,
                first_step_peak_body_direction_deg=peak['body_direction_deg'] if peak else None)
            for tick in sample_ticks:
                row[f'modulation_range_at_{round(times[tick] * 1e6):07d}us'] = float(np.ptp(rates[indices, tick, j]))
            neuron_rows.append(row)
        write_csv(output / 'input_connections.csv', edge_rows)
        write_csv(output / 'neuron_input_summary.csv', neuron_rows)
        report = dict(checkpoint_update=checkpoint['update'], checkpoint_sha256=sha, input_bundle=str(onset),
            first_step_reconstruction_max_abs_error=float(np.max(abs(reconstructed - rates[:, 1]))),
            populations={}, plots=[], definitions=dict(
                first_step='r(dt) = (1-exp(-dt/tau))*relu(bias + PFN(0) @ W_input); recurrent drive is zero at zero initial state.',
                driver='Largest RMS signed preactivation contribution from movement-minus-stationary PFN input over the 1 mm/s grid.',
                shares='Each edge movement-drive RMS divided by summed edge RMS. Not explained variance or causal necessity; cancellation and ReLU matter.',
                later_maps='Same zero initial state, but subsequent updates include recurrence.',
                peaks='First discrete grid maximum, with tied-point count in CSV; omitted for flat first-step responses. Not physiological preferred directions.'))
        for population in circuit.population_types:
            rows = [r for r in neuron_rows if r['cell_type'] == population]
            report['populations'][population] = dict(neurons=len(rows), with_direct_pfn=sum(r['direct_pfn_pairs'] > 0 for r in rows),
                initially_modulated=sum(r['first_step_modulation_range'] > 1e-6 for r in rows))
            for tick in sample_ticks:
                for kind in ('pattern', 'raw', 'movement_delta'):
                    stem = plot_population(output, circuit, indices, selected, rates, stationary,
                                           tick, times[tick], kind, population, columns=args.columns)
                    report['plots'].append(dict(population=population, time_s=float(times[tick]), kind=kind, stem=stem))
            print(f'Input tuning atlas: {population} complete', flush=True)
        write_json(output / 'input_tuning.json', report)
        parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>PFN input tuning atlas</title>',
            '<style>body{font:16px sans-serif;max-width:1300px;margin:2em auto;padding:1em}img{max-width:100%}'
            'table{border-collapse:collapse}td,th{padding:.35em;border:1px solid #ddd}</style>',
            f'<h1>PFN input tuning atlas: update {checkpoint["update"]}</h1>',
            '<p>First update is purely input/bias driven from zero state; later maps include recurrence. '
            'Raw/delta maps share a model-unit scale across neurons within each figure. '
            'Pattern maps rescale each neuron separately to show tuning shape and REMOVE amplitude differences; flat patterns are gray. '
            'Movement-delta maps subtract a same-heading stationary trial. Input preferences are prescribed by the PFN encoder.</p>',
            '<p><a href="neuron_input_summary.csv">All neuron summaries</a> · '
            '<a href="input_connections.csv">Every direct PFN connection</a> · '
            '<a href="input_tuning.json">Definitions and verification</a></p>']
        for population in circuit.population_types:
            parts.append(f'<h2>{population}</h2><table><tr><th>Body ID</th><th>PFN pairs</th><th>Largest movement input</th>'
                         '<th>Discrete initial peak: heading / body direction</th></tr>')
            for row in neuron_rows:
                if row['cell_type'] != population:
                    continue
                driver = str(row['top_movement_input_body_id']) + ' ' + str(row['top_movement_input_type']) if row['top_movement_input_body_id'] else 'No movement drive'
                peak = f'{row["first_step_peak_heading_deg"]} / {row["first_step_peak_body_direction_deg"]}' if row['first_step_peak_heading_deg'] is not None else 'Flat first response'
                parts.append(f'<tr><td>{row["body_id"]}</td><td>{row["direct_pfn_pairs"]}</td><td>{html.escape(driver)}</td><td>{peak}</td></tr>')
            parts.append('</table>')
            for plot in report['plots']:
                if plot['population'] == population:
                    stem = plot['stem']
                    parts.append(f'<h3>{plot["kind"]}, {1000 * plot["time_s"]:g} ms</h3><a href="{stem}.pdf">PDF</a>'
                                 f'<img loading="lazy" src="{stem}.png" alt="{population} tuning atlas">')
        (output / 'index.html').write_text('\n'.join(parts) + '</html>\n')
        if any(digest(path) != value for path, value in frozen.items()):
            raise ValueError('Onset input bundle changed during analysis.')
        finish(output, manifest)
    except BaseException as error:
        fail(output, manifest, error)
        raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--columns', type=int, default=5, help='Subplot columns in each population atlas')
    return analyze(parser.parse_args(argv))


if __name__ == '__main__':
    main()
