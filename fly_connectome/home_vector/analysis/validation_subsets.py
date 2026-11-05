"""Evaluate immutable CPU checkpoint snapshots on original and added validation paths.

Unlike completed-run audits, this explicitly supports a running current run.
Only selected files are snapshotted; the active run is never written or locked.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import pickle
import time

import numpy as np
import torch

from ..__main__ import begin, evaluate, fail, finish, load
from ..data import digest, verify_bundle, write_json
from ..model import HomeVectorNetwork


def stable_read(path, attempts=5):
    """Reject an observed concurrent rewrite; checkpoint deserialization checks completeness."""
    path = Path(path)
    for _ in range(attempts):
        before = path.stat()
        blob = path.read_bytes()
        after = path.stat()
        if (blob and before.st_size == after.st_size == len(blob)
                and before.st_mtime_ns == after.st_mtime_ns and before.st_ino == after.st_ino):
            return blob
        time.sleep(.05)
    raise ValueError(f'Could not read a stable snapshot of {path}.')


def snapshot(run):
    manifest = json.loads(stable_read(run / 'manifest.json'))
    if manifest.get('kind') != 'home_vector_training_v1' or manifest.get('status') not in ('running', 'complete'):
        raise ValueError('Require a running or completed training run.')
    for attempt in range(5):
        blob = stable_read(run / 'best.pt')
        try:
            checkpoint = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=False)
            break
        except (EOFError, RuntimeError, pickle.UnpicklingError):
            # A writer can briefly pause with a stable-looking partial zip.
            if attempt == 4:
                raise
            time.sleep(.05)
    if not isinstance(checkpoint.get('update'), int) or checkpoint['update'] < 1:
        raise ValueError('Checkpoint lacks a valid completed update.')
    # History is rewritten after checkpoint saving; a just-saved checkpoint
    # can legitimately be newer than this copied history.
    try:
        history = json.loads(stable_read(run / 'history.json'))
    except (ValueError, FileNotFoundError):
        history = []
    return dict(manifest=manifest, circuit=json.loads(stable_read(run / 'circuit.json')),
                checkpoint=checkpoint, blob=blob, history=history,
                checkpoint_sha256=hashlib.sha256(blob).hexdigest())


def subset_summary(rows, scale):
    if not rows or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Require nonempty subsets and a positive scale.')
    mse = float(np.mean([row['mse_mm2'] for row in rows]))
    return dict(trajectories=len(rows), equal_trajectory_mse_mm2=mse,
                scaled_mse=mse / scale ** 2, equal_trajectory_vector_rmse_mm=float(np.sqrt(mse)))


def evaluate_snapshot(snap, circuit, original, added, task, output, label):
    network = HomeVectorNetwork(circuit, seed=snap['manifest']['settings']['seed'],
                               recurrence=snap['manifest']['settings'].get('recurrence', True))
    expected = network.state_dict()
    for key in ('pre', 'post', 'counts', 'type_index', 'a_initial'):
        torch.testing.assert_close(snap['checkpoint']['model'][key], expected[key], atol=0, rtol=0)
    network.load_state_dict(snap['checkpoint']['model'], strict=True)
    network.eval()  # No compile flag; evaluate() also enters no_grad().
    summaries, details = {}, {}
    for name, records in [('original', original), ('added_long', added)]:
        values = []
        print(f'{label}: {name}, {len(records)} trajectories, checkpoint {snap["checkpoint"]["update"]}', flush=True)
        for index, row in enumerate(records, 1):
            result, _, _, _ = evaluate(network, circuit, [row], task['dt_s'], 'cpu', task['length_scale_mm'])
            values.extend(result['per_trajectory'])
            if index % 16 == 0 or index == len(records):
                print(f'{label}: {name} {index}/{len(records)}', flush=True)
        details[name] = values
        summaries[name] = subset_summary(values, task['length_scale_mm'])
        write_json(output / f'{label}_per_trajectory.json', details)
    all_rows = details['original'] + details['added_long']
    summaries['combined'] = subset_summary(all_rows, task['length_scale_mm'])
    weighted = sum(summaries[name]['scaled_mse'] * summaries[name]['trajectories']
                   for name in ('original', 'added_long')) / len(all_rows)
    np.testing.assert_allclose(weighted, summaries['combined']['scaled_mse'], rtol=1e-12)
    logged = next((row['validation_scaled_mse'] for row in snap['history']
                   if row['update'] == snap['checkpoint']['update'] and 'validation_scaled_mse' in row), None)
    return dict(checkpoint_update=snap['checkpoint']['update'], checkpoint_sha256=snap['checkpoint_sha256'],
                original_run_status_at_snapshot=snap['manifest']['status'], subsets=summaries,
                own_logged_validation_scaled_mse=logged)


def analyze(args):
    torch.set_num_threads(1)
    paths = {name: Path(getattr(args, name)).resolve()
             for name in ('run', 'source', 'reference_run', 'reference_source')}
    output = Path(args.output).resolve()
    for path in paths.values():
        if output == path or output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError('Output must be separate from all inputs.')
    if output.exists():
        raise FileExistsError(output)
    for key in ('source', 'reference_source'):
        verify_bundle(paths[key])
    verify_bundle(paths['reference_run'])
    snaps = {label: snapshot(paths[key]) for label, key in [('current', 'run'), ('reference', 'reference_run')]}
    circuit, groups, task = load(paths['source'])
    old_circuit, old_groups, old_task = load(paths['reference_source'])
    for label, key in [('current', 'source'), ('reference', 'reference_source')]:
        if snaps[label]['manifest']['settings']['source_manifest_sha256'] != digest(paths[key] / 'manifest.json'):
            raise ValueError('Checkpoint run does not match its supplied source.')
        if snaps[label]['circuit'] != circuit.as_dict():
            raise ValueError('Both run circuits must match the common prepared circuit.')
    if old_circuit.as_dict() != circuit.as_dict() or any(task[key] != old_task[key] for key in ('dt_s', 'length_scale_mm')):
        raise ValueError('Require identical circuits, dt and normalization scale.')
    original = old_groups['validation']
    if groups['validation'][:len(original)] != original:
        raise ValueError('Original validation trajectories are not an exact retained prefix.')
    added = groups['validation'][len(original):]
    if not added or any(row['kind'] != 'long_holdout' for row in added):
        raise ValueError('Require added long-holdout validation trajectories.')
    frozen = {str(p): digest(p) for key in ('source', 'reference_source', 'reference_run')
              for p in paths[key].iterdir() if p.is_file()}
    output, record = begin(output, 'home_vector_validation_subsets_snapshot_v1',
        source=str(paths['source']), source_manifest_sha256=digest(paths['source'] / 'manifest.json'),
        **{key: str(path) for key, path in paths.items() if key != 'source'})
    record.update(refitting=False, active_run_may_change=True, immutable_input_sha256=frozen,
        torch=torch.__version__, device='cpu', threads=1,
        analysis_code_sha256={str(p.relative_to(Path(__file__).parents[1])): digest(p)
                             for p in Path(__file__).parent.glob('*.py')})
    write_json(output / 'manifest.json', record)
    try:
        for label, snap in snaps.items():
            (output / f'{label}_checkpoint.pt').write_bytes(snap['blob'])
            write_json(output / f'{label}_metadata.json', dict(
                run_manifest=snap['manifest'], circuit=snap['circuit'], history=snap['history'],
                checkpoint_update=snap['checkpoint']['update'], checkpoint_sha256=snap['checkpoint_sha256']))
        results = {label: evaluate_snapshot(snap, circuit, original, added, task, output, label)
                   for label, snap in snaps.items()}
        # Current combined evaluation and reference ORIGINAL evaluation have
        # the same membership as their respective logged selection objectives.
        for label, subset in [('current', 'combined'), ('reference', 'original')]:
            logged = results[label]['own_logged_validation_scaled_mse']
            if logged is not None:
                actual = results[label]['subsets'][subset]['scaled_mse']
                np.testing.assert_allclose(actual, logged, atol=1e-4, rtol=1e-5)
                results[label]['matches_own_logged_validation'] = True
        results['definitions'] = dict(
            metric='Equal-trajectory, time-averaged squared Cartesian error / inherited L²; RMSE is sqrt of equal-trajectory unscaled MSE.',
            checkpoint='Independent byte copy of best.pt at the recorded update, not necessarily the latest optimizer update.',
            comparison='Both models evaluated on identical original and added-long validation trajectories; no test/challenge data.',
            uncertainty='Single checkpoints and seed; running training can improve after the snapshot.')
        if any(digest(path) != value for path, value in frozen.items()):
            raise ValueError('A supposedly immutable source/reference bundle changed.')
        write_json(output / 'validation_subsets.json', results)
        finish(output, record)
    except BaseException as error:
        fail(output, record, error)
        raise
    print(json.dumps(dict(output=str(output), **results), indent=2))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'source', 'reference-run', 'reference-source', 'output'):
        parser.add_argument(f'--{name}', required=True)
    return analyze(parser.parse_args(argv))


if __name__ == '__main__':
    main()
