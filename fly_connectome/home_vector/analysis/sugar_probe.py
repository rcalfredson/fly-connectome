"""Isolated, explicit-parameter sugar experiment; never select parameters by decoding."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..__main__ import batch, begin, fail, finish
from ..data import Circuit, digest, sample_protocol, verify_bundle, write_json
from .activity import record_trajectory
from .checkpoint import capture
from .decoder_contributions import CHECKPOINT_SHA256, write_csv
from .predictions import independent_target
from .repeated_loops import repeat_protocol
from .sugar_response import SugarResponse, event_gate, rollout


def insert_pause(protocol, at_s, duration_s, dt):
    """Identical movement before/after the specified pause; no state reset."""
    p = np.asarray(protocol, dtype=float)
    sample_protocol(p, dt)
    if (not np.isfinite([at_s, duration_s]).all() or duration_s < 0 or not 0 < at_s < p[-1, 0]
            or not np.isclose(at_s/dt, round(at_s/dt), atol=1e-7, rtol=0)
            or not np.isclose(duration_s/dt, round(duration_s/dt), atol=1e-7, rtol=0)):
        raise ValueError('Require a grid-aligned nonnegative pause strictly inside the trial.')
    if duration_s == 0:
        return p.tolist()
    before = p[p[:, 0] < at_s]
    current = p[np.searchsorted(p[:, 0], at_s, side='right') - 1]
    after = p[p[:, 0] >= at_s].copy()
    after[:, 0] += duration_s
    # A movement boundary at pause onset may need to be reintroduced at its end.
    if not np.any(p[:, 0] == at_s):
        after = np.vstack(([at_s+duration_s, *current[1:]], after))
    return np.vstack((before, [at_s, current[1], 0., 0.], after)).tolist()


def read_config(path, dt, end_s):
    config = json.loads(Path(path).read_text())
    if config.get('status') != 'uncalibrated_engineering_example':
        raise ValueError('This first setup accepts only explicitly uncalibrated engineering configurations.')
    if not config.get('selection_basis'):
        raise ValueError('Record the parameter selection basis.')
    response = SugarResponse(**config['response'])
    events = np.asarray(config['event_times_s'], dtype=float)
    if (events.ndim != 1 or not len(events) or not np.isfinite(events).all()
            or (events < 0).any() or (events >= end_s).any() or (np.diff(events) <= 0).any()
            or not np.allclose(events/dt, np.round(events/dt), atol=1e-7, rtol=0)):
        raise ValueError('Require explicit unique grid-aligned events within every trial.')
    # Respect the thesis delivery schedule for this initial scaffold, without
    # treating that protocol rule as a biological refractory-period measurement.
    if len(events) > 1 and np.min(np.diff(events)) < 30.:
        raise ValueError('The initial setup requires at least 30 s between deliveries.')
    return config, response


def record_probe(network, circuit, record, dt, response, events, use_sugar):
    controls, truth = sample_protocol(record['protocol'], dt)
    time = np.arange(len(controls))*dt
    gate = event_gate(time, events, response) if use_sugar else np.zeros(len(time))
    mask = torch.tensor(np.asarray(circuit.dynamic_types) == 'hDeltaG', dtype=torch.bool)
    if mask.sum().item() != 8:
        raise ValueError('Require exactly eight hDeltaG target cells.')
    with torch.no_grad():
        pfn, target, _ = batch(circuit, [record], dt, 'cpu', network.a.dtype)
        prediction, rates = rollout(network, pfn, dt, gate, response, mask)
        # Every gate-disabled trial must match the production forward pass.
        if not use_sugar:
            native_prediction, native_rates = network(pfn, dt)
            torch.testing.assert_close(rates, native_rates, atol=0, rtol=0)
            torch.testing.assert_close(prediction, native_prediction, atol=0, rtol=0)
    arrays = dict(time_s=time, controls=controls, gate=gate, pfn=pfn[0].numpy(),
                  target=target[0].numpy(), prediction=prediction[0].numpy(), rates=rates[0].numpy())
    np.testing.assert_allclose(arrays['target'], truth, atol=5e-6, rtol=1e-6)
    np.testing.assert_allclose(arrays['target'], independent_target(record['protocol'], len(time), dt), atol=5e-6, rtol=1e-6)
    reconstructed = arrays['rates'].astype(float) @ network.D.detach().numpy().astype(float).T + network.d.detach().numpy()
    np.testing.assert_allclose(reconstructed, arrays['prediction'], atol=3e-5, rtol=1e-5)
    return arrays


def plot_path(output, name, trials, types, events):
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), layout='constrained')
    for arm, data in trials.items():
        t = data['time_s']
        axes[0, 0].plot(t, np.linalg.norm(data['prediction']-data['target'], axis=1), label=arm)
        axes[0, 1].plot(t, data['gate'], label=arm)
        for ax, pop in zip(axes.flat[2:], ('hDeltaG', 'hDeltaB', 'hDeltaC', 'vDeltaE')):
            ax.plot(t, data['rates'][:, np.asarray(types) == pop].mean(axis=1), label=arm)
            ax.set(title=pop + ' actual mean activity', ylabel='Model activity units')
    axes[0, 0].set(title='Decoded error (secondary outcome)', ylabel='mm')
    axes[0, 1].set(title='Imposed sugar gate (not a measured neural response)', ylabel='Dimensionless')
    for ax in axes.flat:
        for event in events:
            ax.axvline(event, color='gray', ls=':', lw=.7)
        ax.set_xlabel('Elapsed time (s)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle(name + ' — UNCALIBRATED ENGINEERING EXAMPLE; no physiological fit')
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'{name}.{extension}', dpi=150)
    plt.close(fig)


def analyze(args):
    activity, source, baseline, config_path, output = [Path(getattr(args, k)).resolve()
        for k in ('activity', 'source', 'baseline', 'config', 'output')]
    for folder in (activity, source, baseline):
        if output == folder or output.is_relative_to(folder) or folder.is_relative_to(output):
            raise ValueError('Output must be separate from every immutable input bundle.')
    if output.exists():
        raise FileExistsError(output)
    if verify_bundle(activity)['kind'] != 'home_vector_activity_paths_v1':
        raise ValueError('Require the archived activity paths.')
    if verify_bundle(baseline)['kind'] != 'home_vector_repeated_loops_v1':
        raise ValueError('Require the unmodified repeated-loop reference.')
    verify_bundle(source)
    torch.set_num_threads(1)
    snap = capture(activity, source)
    if snap['checkpoint_sha256'] != CHECKPOINT_SHA256 or snap['checkpoint']['update'] != 1975:
        raise ValueError('Require the specified frozen model.')
    network, circuit, dt = snap['network'], Circuit(**snap['circuit']), snap['task']['dt_s']
    config, response = read_config(config_path, dt, 16.)
    if config['pause_at_s'] != 8.:
        raise ValueError('This probe inserts its matched pause at the first closure, 8 s.')
    if config['pause_duration_s'] <= 0:
        raise ValueError('Require a positive duration to distinguish the pause controls.')
    events = config['event_times_s']
    frozen = {str(p): digest(p) for folder in (activity, source, baseline) for p in folder.iterdir() if p.is_file()}
    frozen[str(config_path)] = digest(config_path)
    original_state = {k: v.clone() for k, v in network.state_dict().items()}
    output, manifest = begin(output, 'home_vector_hdeltag_sugar_probe_v1', source=str(source),
        source_manifest_sha256=digest(source / 'manifest.json'), config=str(config_path))
    manifest.update(immutable_input_sha256=frozen, refitting=False, physiological_validation=False,
        parameter_selection='Explicit uncalibrated engineering configuration; not optimized.',
        checkpoint_sha256=CHECKPOINT_SHA256, torch=torch.__version__, threads=1,
        analysis_code_sha256={p.name: digest(p) for p in Path(__file__).parent.glob('*.py')})
    write_json(output / 'manifest.json', manifest)
    try:
        for name in ('current_checkpoint.pt', 'current_metadata.json', 'neurons.csv'):
            shutil.copyfile(activity / name, output / name)
        shutil.copyfile(config_path, output / 'config.json')
        for name in ('sugar_response.py', 'sugar_probe.py'):
            shutil.copyfile(Path(__file__).with_name(name), output / name)
        write_json(output / 'response.json', asdict(response))
        records = {r['path']: r['record'] for r in json.loads((activity / 'raw_paths.json').read_text())}
        report = dict(status='uncalibrated_engineering_example', parameter_selection=config['selection_basis'],
            checkpoint_sha256=CHECKPOINT_SHA256, events_s=events, comparisons=[])
        protocols, rows, population_rows = {}, [], []
        for number in (8, 9, 14, 15):
            name = f'path_{number:02d}'
            protocol, _, period = repeat_protocol(records[name]['protocol'], dt)
            if period != 8.:
                raise ValueError('Require the archived eight-second circle.')
            trials = {}
            for pause in (False, True):
                p = insert_pause(protocol, config['pause_at_s'], config['pause_duration_s'], dt) if pause else protocol
                record = dict(records[name], protocol=p)
                key = name + ('_pause' if pause else '_continuous')
                protocols[key] = record
                for sugar in (False, True):
                    arm = ('pause' if pause else 'continuous') + ('_sugar' if sugar else '_control')
                    data = record_probe(network, circuit, record, dt, response, events, sugar)
                    trials[arm] = data
                    if not pause and not sugar:
                        with np.load(baseline / f'{name}.npz', allow_pickle=False) as saved:
                            for field in ('controls', 'pfn', 'rates', 'prediction', 'target'):
                                np.testing.assert_allclose(data[field], saved[field], atol=3e-5, rtol=1e-6)
                    np.savez_compressed(output / f'{name}_{arm}.npz', **data)
                    duration = np.diff(data['time_s'])
                    error2 = np.sum((data['prediction']-data['target']).astype(float)**2, axis=1)
                    rows.append(dict(path=name, arm=arm, duration_s=float(data['time_s'][-1]),
                        vector_rmse_mm=float(np.sqrt(np.average(error2[:-1], weights=duration))),
                        final_error_mm=float(np.sqrt(error2[-1])),
                        final_prediction_x_mm=float(data['prediction'][-1, 0]),
                        final_prediction_y_mm=float(data['prediction'][-1, 1])))
                    for pop in circuit.population_types:
                        r = data['rates'][:, np.asarray(circuit.dynamic_types) == pop].astype(float)
                        population_rows.append(dict(path=name, arm=arm, cell_type=pop,
                            population_mean_before_first_event=float(r[int(round(events[0]/dt))].mean()),
                            population_mean_at_end=float(r[-1].mean()), maximum_activity=float(r.max()),
                            final_cross_cell_rms_about_mean=float(np.sqrt(np.mean((r[-1]-r[-1].mean())**2)))))
                control = trials[('pause' if pause else 'continuous') + '_control']
                treated = trials[('pause' if pause else 'continuous') + '_sugar']
                for field in ('controls', 'pfn', 'target'):
                    np.testing.assert_array_equal(control[field], treated[field])
                prefix = control['time_s'] <= events[0]
                np.testing.assert_array_equal(control['rates'][prefix], treated['rates'][prefix])
                report['comparisons'].append(dict(path=name, pause=pause,
                    equal_controls_pfns_targets=True, exact_pre_event_activity_match=True))
            plot_path(output, name, trials, circuit.dynamic_types, events)
            print(f'Verified four matched arms: {name}', flush=True)
        for k, value in network.state_dict().items():
            torch.testing.assert_close(value, original_state[k], atol=0, rtol=0)
        write_csv(output / 'decoder_metrics.csv', rows)
        write_csv(output / 'population_metrics.csv', population_rows)
        write_json(output / 'protocols.json', protocols)
        write_json(output / 'probe.json', report)
        (output / 'report.md').write_text(
            '# hDeltaG sugar probe: uncalibrated engineering example\n\n'
            'This run checks an isolated candidate-B implementation. Parameters are explicit '
            'engineering choices, not a physiological fit or a selected decoding improvement. '
            'The long gate-decay constant does not prescribe the resulting activity-recovery time.\n\n'
            'Four arms per path: continuous control, continuous sugar, pause control, pause sugar. '
            'Within each sugar/control comparison, movement, PFNs and targets are identical. '
            'Only the scalar scheduled sugar gate differs. The pause is inserted at 8 s; '
            'there is no state reset or reference-origin change. All targets remain referenced '
            'to the original origin. The event schedule represents experimenter delivery, '
            'not an implemented floor-contact detector. The network never receives true position.\n\n'
            'Only hDeltaG has an added relaxation term. All other populations retain their '
            'normal updates and can change through recurrence. Neither a neurotransmitter '
            'nor a specific sugar-to-hDeltaG pathway is identified.\n\n'
            'The gate is the maximum across causal rise-times-decay event kernels. A zero gate '
            'matches the production forward pass exactly. Frozen model parameters and source '
            'files are verified unchanged. decoder_metrics.csv is a secondary outcome; '
            'population_metrics.csv and NPZ traces support physiological assessment. '
            'Cross-cell contrast is not a measured bump amplitude; no phase map or calcium '
            'observation model has been assumed. This short probe cannot validate minutes-long recovery.\n')
        (output / 'index.html').write_text('<!doctype html><meta charset="utf-8">'
            '<title>Uncalibrated hDeltaG sugar setup</title><style>body{font:16px system-ui;'
            'max-width:1200px;margin:2em auto}img{max-width:100%}</style>'
            '<h1>UNCALIBRATED ENGINEERING EXAMPLE</h1><p>Setup verification, not physiological validation.</p>'
            '<p><a href="report.md">Methods and limitations</a> · '
            '<a href="config.json">Explicit parameters</a> · '
            '<a href="population_metrics.csv">Population measurements</a> · '
            '<a href="decoder_metrics.csv">Secondary decoding measurements</a></p>'
            + ''.join(f'<h2>path_{n:02d}</h2><img src="path_{n:02d}.png" alt="Matched sugar controls">' for n in (8, 9, 14, 15)))
        if any(digest(path) != sha for path, sha in frozen.items()):
            raise ValueError('An immutable input changed during the probe.')
        finish(output, manifest)
    except BaseException as error:
        fail(output, manifest, error)
        raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('activity', 'source', 'baseline', 'config', 'output'):
        parser.add_argument(f'--{name}', required=True)
    return analyze(parser.parse_args(argv))


if __name__ == '__main__':
    main()
