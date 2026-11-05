"""Audit a two-batch-size eager/compiled profiling comparison; never train."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .__main__ import begin, fail, finish
from .data import digest, verify_bundle, write_json


def tensor_tree_difference(a, b):
    """Check every parameter/Adam field, returning the largest absolute delta."""
    if isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError('Checkpoint fields differ.')
        return max([tensor_tree_difference(a[k], b[k]) for k in a] + [0.])
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise ValueError('Checkpoint lengths differ.')
        return max([tensor_tree_difference(x, y) for x, y in zip(a, b)] + [0.])
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-4)
        return float((a.double() - b.double()).abs().max())
    if a != b:
        raise ValueError('Optimizer settings differ.')
    return 0.


def matched_rows(a, b):
    """Prevent comparing different batches, warmups, or numerical outcomes."""
    if len(a) != len(b) or not a:
        raise ValueError('Require equal, nonempty update histories.')
    for x, y in zip(a, b):
        if any(x[k] != y[k] for k in ('update', 'measured', 'batch')):
            raise ValueError('Sampled batches or measured-update selections differ.')
        for key in ('loss', 'task', 'activity', 'edge_prior', 'decoder', 'gradient_norm'):
            if not np.isclose(x[key], y[key], atol=2e-5, rtol=2e-4):
                raise ValueError(f'Numerical disagreement in {key}.')
    if not any(x['measured'] for x in a):
        raise ValueError('No measured updates.')


def compare(args):
    names = ('small_eager', 'small_compiled', 'large_eager', 'large_compiled')
    paths = {name: Path(getattr(args, name)).resolve() for name in names}
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    for path in [source, *paths.values()]:
        if output == path or output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError('Output must be separate from every input bundle.')
    verify_bundle(source)
    manifests = {name: verify_bundle(path) for name, path in paths.items()}
    expected_source = digest(source / 'manifest.json')
    for manifest in manifests.values():
        if (manifest['kind'] != 'home_vector_training_profile_v1'
                or manifest['settings']['source_manifest_sha256'] != expected_source
                or manifest['settings'].get('batching', 'padded') != 'padded'):
            raise ValueError('Require padded profiles from the same prepared source.')
    settings = [manifests[name]['settings'] for name in names]
    for key in ('seed', 'updates', 'warmup', 'threads', 'device', 'trace_seconds'):
        if len({s[key] for s in settings}) != 1:
            raise ValueError(f'Profiles differ in {key}.')
    small, large = settings[0]['batch_size'], settings[2]['batch_size']
    block = settings[1].get('compile_block', 0)
    if not (0 < small < large and block > 0
            and settings[1]['batch_size'] == small and settings[3]['batch_size'] == large
            and settings[3].get('compile_block', 0) == block
            and not settings[0].get('compile_block', 0) and not settings[2].get('compile_block', 0)):
        raise ValueError('Require two distinct batch sizes, each with eager and the same compiled block size.')
    # Model and profiling code must agree across the four measurements.
    for filename in ('model.py', 'profile_training.py'):
        key = f'fly_connectome/home_vector/{filename}'
        if len({m['code_sha256'][key] for m in manifests.values()}) != 1:
            raise ValueError(f'Profiles used different {filename} implementations.')
    hashes = {str(p): digest(p) for folder in [source, *paths.values()]
              for p in folder.iterdir() if p.is_file()}
    output, record = begin(output, 'home_vector_combined_compilation_comparison_v1',
        source=str(source), source_manifest_sha256=expected_source,
        profiles={name: str(path) for name, path in paths.items()})
    record.update(input_sha256=hashes, scientific_training=False)
    write_json(output / 'manifest.json', record)
    try:
        histories = {name: json.loads((path / 'baseline_updates.json').read_text())
                     for name, path in paths.items()}
        agreement = {}
        for size in ('small', 'large'):
            a, b = [histories[f'{size}_{mode}'] for mode in ('eager', 'compiled')]
            matched_rows(a, b)
            checkpoints = [torch.load(paths[f'{size}_{mode}'] / 'baseline_final.pt',
                                     map_location='cpu', weights_only=False)
                           for mode in ('eager', 'compiled')]
            agreement[size] = dict(passed=True,
                max_model_difference=tensor_tree_difference(checkpoints[0]['model'], checkpoints[1]['model']),
                max_optimizer_difference=tensor_tree_difference(checkpoints[0]['optimizer'], checkpoints[1]['optimizer']),
                max_loss_difference=max(abs(x['loss'] - y['loss']) for x, y in zip(a, b)),
                max_gradient_norm_difference=max(abs(x['gradient_norm'] - y['gradient_norm']) for x, y in zip(a, b)))
        cases = {}
        exposure = 2000 * small
        for name, rows in histories.items():
            measured = [row for row in rows if row['measured']]
            seconds = [row['wall_s'] for row in measured]
            mean = float(np.mean(seconds))
            batch_size = manifests[name]['settings']['batch_size']
            cases[name] = dict(batch_size=batch_size, mean_update_s=mean,
                measured_update_seconds=seconds, trajectories_per_s=batch_size / mean,
                exposure_matched_update_count=exposure / batch_size,
                exposure_matched_update_seconds=exposure / batch_size * mean,
                peak_cuda_allocated_bytes=max(row.get('cuda_memory_bytes', {}).get('peak_allocated', 0) for row in rows),
                warmup_update_seconds=[row['wall_s'] for row in rows if not row['measured']])
        times = {name: case['mean_update_s'] for name, case in cases.items()}
        small_speedup = times['small_eager'] / times['small_compiled']
        large_speedup = times['large_eager'] / times['large_compiled']
        predicted = times['large_eager'] / small_speedup
        result = dict(cases=cases, agreement=agreement,
            compilation_speedup_small=small_speedup, compilation_speedup_large=large_speedup,
            speedup_ratio_large_to_small=large_speedup / small_speedup,
            independent_speedups_predicted_large_update_s=predicted,
            actual_over_independent_predicted_large_update_time=times['large_compiled'] / predicted,
            equal_exposure_speedup=cases['small_eager']['exposure_matched_update_seconds'] / cases['large_compiled']['exposure_matched_update_seconds'],
            definitions=dict(exposure=f'{exposure} sampled trajectory presentations, not guaranteed distinct examples',
                timing='Mean of measured baseline updates; full BPTT, prep/transfer, Adam and guards included. Validation, checkpoint I/O and compilation warmup excluded.',
                interpretation='A short timing comparison, not a statistical independence test or convergence comparison. Equal exposure does not imply equal learning.'))
        if any(digest(p) != value for p, value in hashes.items()):
            raise ValueError('Input bundles changed during comparison.')
        write_json(output / 'comparison.json', result)
        finish(output, record)
    except BaseException as error:
        fail(output, record, error)
        raise
    return dict(output=str(output), **result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('--output', required=True)
    for name in ('small-eager', 'small-compiled', 'large-eager', 'large-compiled'):
        parser.add_argument(f'--{name}', required=True)
    print(json.dumps(compare(parser.parse_args(argv)), indent=2))


if __name__ == '__main__':
    main()
