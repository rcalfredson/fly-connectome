"""Record frozen checkpoint activity and plot the standard challenge atlas on CPU."""

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np
import torch

from ..__main__ import batch, begin, fail, finish
from ..data import Circuit, digest, sample_protocol, write_json
from .activity_plots import plot_atlas, plot_readout
from .activity_summary import neuron_summary, population_readout
from .checkpoint import capture
from .predictions import independent_target


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def record_trajectory(network, circuit, row, dt):
    """Use the production forward pass once; the input at t updates state t+dt."""
    controls, geometric_target = sample_protocol(row['protocol'], dt)
    with torch.no_grad():
        pfn, target, _ = batch(circuit, [row], dt, 'cpu', network.a.dtype)
        prediction, rates = network(pfn, dt)
    arrays = dict(time_s=np.arange(len(controls)) * dt, controls=controls,
                  pfn=pfn[0].numpy(), rates=rates[0].numpy(),
                  prediction=prediction[0].numpy(), target=target[0].numpy())
    if (not all(np.isfinite(a).all() for a in arrays.values())
            or np.max(arrays['rates']) > 1e6 or np.min(arrays['rates']) < 0):
        raise ValueError('Activity replay exceeded numerical guard.')
    np.testing.assert_allclose(arrays['target'], geometric_target, rtol=1e-6, atol=5e-6)
    np.testing.assert_allclose(arrays['target'], independent_target(row['protocol'], len(controls), dt),
                               rtol=1e-6, atol=5e-6)
    decoder, intercept = network.D.detach().numpy(), network.d.detach().numpy()
    contributions, reconstructed = population_readout(
        arrays['rates'], decoder, intercept, circuit.dynamic_types, circuit.population_types)
    np.testing.assert_allclose(reconstructed, arrays['prediction'], rtol=1e-5, atol=2e-5)
    arrays['population_contribution'] = contributions
    return arrays, float(np.max(np.abs(reconstructed - arrays['prediction'])))


def write_index(output, report):
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8">',
             '<title>Frozen CX activity atlas</title>',
             '<style>body{font:16px sans-serif;max-width:1100px;margin:2em auto;padding:0 1em}'
             'img{max-width:100%;height:auto}code{overflow-wrap:anywhere}</style>',
             '<h1>Frozen CX activity atlas</h1>',
             f'<p>Checkpoint update {report["checkpoint_update"]}; CPU replay, no refitting.</p>',
             f'<p>SHA256: <code>{report["checkpoint_sha256"]}</code></p>',
             '<p>Rates are model units, not Hz. Detail heatmaps share a raw activity scale within each trajectory; '
             'the linked global atlases share one scale across all trajectories. Neither standardizes individual neurons. '
             'body-ID ordering is not anatomical. Gray lines mark control transitions. '
             'All data and statistics retain the full simulation resolution. '
             'Signed readout contributions may cancel; they do not establish causal necessity.</p>',
             '<p><a href="activity.json">Metadata and verification</a> · '
             '<a href="neurons.csv">Neuron identities and decoder</a> · '
             '<a href="neuron_summary.csv">Neuron summaries</a> · '
             '<a href="population_summary.csv">Population summaries</a></p>']
    for row in report['trajectories']:
        stem = row['stem']
        parts.append(f'<h2>{html.escape(row["title"])}</h2>'
                     f'<p><a href="{stem}.npz">Full-resolution arrays</a> · '
                     f'<a href="{stem}_atlas_detail.pdf">Detail atlas PDF</a> · '
                     f'<a href="{stem}_atlas.png">Global-scale atlas PNG</a> · '
                     f'<a href="{stem}_atlas.pdf">Global-scale atlas PDF</a> · '
                     f'<a href="{stem}_readout.pdf">Readout PDF</a></p>'
                     f'<img loading="lazy" src="{stem}_atlas_detail.png" alt="Neural activity detail atlas">'
                     f'<img loading="lazy" src="{stem}_readout.png" alt="Signed population readout">')
    (output / 'index.html').write_text('\n'.join(parts) + '</html>\n')


def analyze(args):
    if args.threads < 1:
        raise ValueError('Require a positive CPU thread count.')
    run, source, output = [Path(getattr(args, k)).resolve() for k in ('run', 'source', 'output')]
    for path in (run, source):
        if output == path or output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError('Output must be separate from every input bundle.')
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(args.threads)
    snap = capture(run, source, getattr(args, 'snapshot_key', 'current'))
    circuit, network, dt = Circuit(**snap['circuit']), snap['network'], snap['task']['dt_s']
    frozen_paths = {source}
    if snap['input_kind'] != 'training_run' or snap['manifest']['status'] == 'complete':
        frozen_paths.add(run)
    frozen = {str(p): digest(p) for folder in frozen_paths for p in folder.iterdir() if p.is_file()}
    output, manifest = begin(output, 'home_vector_activity_v1', source=str(source),
        source_manifest_sha256=digest(source / 'manifest.json'), run=str(run))
    manifest.update(device='cpu', threads=args.threads, torch=torch.__version__, refitting=False,
        immutable_input_sha256=frozen,
        analysis_code_sha256={p.name: digest(p) for p in Path(__file__).parent.glob('*.py')})
    write_json(output / 'manifest.json', manifest)
    try:
        (output / 'current_checkpoint.pt').write_bytes(snap['blob'])
        write_json(output / 'current_metadata.json', dict(run=str(run), source=str(source),
            run_manifest=snap['manifest'], circuit=snap['circuit'], history=snap['history'],
            training_config=snap['training_config'], checkpoint_update=snap['checkpoint']['update'],
            checkpoint_sha256=snap['checkpoint_sha256'], input_kind=snap['input_kind'],
            snapshot_key=snap.get('snapshot_key')))
        decoder, intercept = network.D.detach().numpy(), network.d.detach().numpy()
        write_csv(output / 'neurons.csv', [dict(neuron_index=j, body_id=body_id, cell_type=cell_type,
            decoder_x=float(decoder[0, j]), decoder_y=float(decoder[1, j]))
            for j, (body_id, cell_type) in enumerate(zip(circuit.dynamic_ids, circuit.dynamic_types))])
        write_csv(output / 'inputs.csv', [dict(input_index=j, body_id=body_id, cell_type=cell_type)
            for j, (body_id, cell_type) in enumerate(zip(circuit.input_ids, circuit.input_types))])
        report = dict(checkpoint_update=snap['checkpoint']['update'],
            checkpoint_sha256=snap['checkpoint_sha256'], dt_s=dt,
            population_order=list(circuit.population_types), decoder_intercept_mm=intercept.tolist(),
            axes=dict(controls=['heading_deg', 'forward_mm_s', 'rightward_mm_s'],
                      rates='time, neuron_index (neurons.csv)', pfn='time, input_index (inputs.csv)',
                      prediction=['home_x_mm', 'home_y_mm'], target=['home_x_mm', 'home_y_mm'],
                      population_contribution='time, population_order, home_xy_mm'),
            definitions=dict(rates='Nonnegative model units; not calibrated Hz.',
                timing='State at t precedes update driven by PFN input at t; initial state is zero.',
                contributions='Signed D_population @ rate_population; intercept included once in total.',
                display='Full-resolution traces rendered; raster resolution limits visible detail.',
                ordering='Population then body ID in plots; original circuit order in saved arrays.'),
            trajectories=[])
        summaries, pop_summaries, vmax = [], [], 0.
        for i, row in enumerate(snap['groups']['challenge']):
            print(f'Activity: {i + 1}/{len(snap["groups"]["challenge"])} {row["family"]}', flush=True)
            arrays, residual = record_trajectory(network, circuit, row, dt)
            stem = f'trajectory_{i:02d}'
            np.savez_compressed(output / f'{stem}.npz', **arrays)
            vmax = max(vmax, float(arrays['rates'].max()))
            title = f'{i:02d}: {row["family"]} / {row["kind"]} — update {snap["checkpoint"]["update"]}'
            events = [float(item[0]) for item in row['protocol'][1:-1]]
            report['trajectories'].append(dict(stem=stem, title=title, record=row, events_s=events,
                samples=len(arrays['time_s']), trajectory_rate_vmax=float(arrays['rates'].max()),
                readout_max_abs_residual_mm=residual))
            summaries.extend(dict(trajectory=stem, **item) for item in neuron_summary(
                arrays['rates'], decoder, circuit.dynamic_ids, circuit.dynamic_types))
            for p, population in enumerate(circuit.population_types):
                rates = arrays['rates'][:, np.asarray(circuit.dynamic_types) == population]
                item = dict(trajectory=stem, cell_type=population, rate_mean=float(rates.mean()),
                            rate_max=float(rates.max()))
                for k, axis in enumerate(('x', 'y')):
                    contribution = arrays['population_contribution'][:, p, k]
                    item.update({f'contribution_{axis}_mean_mm': float(contribution.mean()),
                                 f'contribution_{axis}_final_mm': float(contribution[-1]),
                                 f'contribution_{axis}_max_abs_mm': float(np.abs(contribution).max())})
                pop_summaries.append(item)
        report['shared_rate_vmax'] = vmax
        write_csv(output / 'neuron_summary.csv', summaries)
        write_csv(output / 'population_summary.csv', pop_summaries)
        for row in report['trajectories']:
            with np.load(output / f'{row["stem"]}.npz') as arrays:
                plot_atlas(arrays, circuit, circuit.population_types, row['events_s'], row['title'],
                           vmax, output, row['stem'])
                plot_atlas(arrays, circuit, circuit.population_types, row['events_s'], row['title'],
                           row['trajectory_rate_vmax'], output, row['stem'], detail=True)
                plot_readout(arrays, circuit.population_types, intercept, row['events_s'], row['title'],
                             output, row['stem'])
        write_json(output / 'activity.json', report)
        write_index(output, report)
        if any(digest(path) != value for path, value in frozen.items()):
            raise ValueError('An immutable input bundle changed during recording.')
        finish(output, manifest)
    except BaseException as error:
        fail(output, manifest, error)
        raise
    result = dict(output=str(output), status='complete', checkpoint_update=report['checkpoint_update'],
                  trajectories=len(report['trajectories']))
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'source', 'output'):
        parser.add_argument(f'--{name}', required=True)
    parser.add_argument('--snapshot-key', choices=['current', 'reference'], default='current')
    parser.add_argument('--threads', type=int, default=1)
    return analyze(parser.parse_args(argv))


if __name__ == '__main__':
    main()
