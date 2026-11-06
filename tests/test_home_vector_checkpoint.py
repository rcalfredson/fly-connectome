"""Reusable CPU challenge snapshots must remain independent of a live writer."""
import argparse
import copy
import csv
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from fly_connectome.home_vector.__main__ import begin, finish, load
from fly_connectome.home_vector.analysis.checkpoint import analyze, capture, validate_pair
from fly_connectome.home_vector.analysis.compare import trajectory_summary
from fly_connectome.home_vector.data import datasets, digest, sample_protocol, verify_bundle, write_json
from fly_connectome.home_vector.model import HomeVectorNetwork
from test_home_vector_data import circuit_fixture


def fixtures(tmp_path):
    source, record = begin(tmp_path / 'source', 'home_vector_prepared_v1')
    circuit = circuit_fixture()
    groups = datasets(4, dict(train=2, validation=2, test=2))
    for records in groups.values():
        for row in records:
            for control in row['protocol']:
                control[0] /= 100
            if 'suffix_start_s' in row:
                row['suffix_start_s'] /= 100
    write_json(source / 'circuit.json', circuit.as_dict())
    write_json(source / 'task.json', dict(dt_s=.0025, length_scale_mm=2.))
    write_json(source / 'trajectories.json', groups)
    finish(source, record)
    for name, complete, recurrent in [('active', False, False), ('reference', True, True)]:
        run, manifest = begin(tmp_path / name, 'home_vector_training_v1', seed=0, recurrence=recurrent,
            source=str(source), source_manifest_sha256=digest(source / 'manifest.json'))
        shutil.copyfile(source / 'circuit.json', run / 'circuit.json')
        write_json(run / 'training_config.json', dict(loss=dict(mode='original')))
        network = HomeVectorNetwork(circuit, recurrence=recurrent)
        with torch.no_grad():
            network.D.fill_(.03)
        torch.save(dict(model=network.state_dict(), update=7), run / 'best.pt')
        write_json(run / 'history.json', [dict(update=n, validation_scaled_mse=1. / n) for n in (1, 7, 8)])
        if complete:
            finish(run, manifest)
    return argparse.Namespace(run=str(tmp_path / 'active'), source=str(source), output=str(tmp_path / 'audit'),
        reference_run=None, reference_source=None, allow_task_differences=False,
        label='Current', reference_label='Reference', threads=1)


@pytest.mark.parametrize('reference', [False, True])
def test_single_or_comparison_snapshot_matches_model_and_shared_metrics(tmp_path, reference):
    args = fixtures(tmp_path)
    if reference:
        args.reference_run, args.reference_source = str(tmp_path / 'reference'), args.source
    before = {str(p): digest(p) for p in tmp_path.rglob('*') if p.is_file()}
    result = analyze(args)
    assert result['checkpoint_updates']['current'] == 7
    output = Path(args.output)
    manifest = verify_bundle(output)
    assert manifest['device'] == 'cpu' and not manifest['refitting']
    assert all(digest(p) == value for p, value in before.items())
    report = json.loads((output / 'challenge_metrics.json').read_text())
    assert set(report['model_labels']) == ({'current', 'reference'} if reference else {'current'})
    assert 'update 7' in report['model_labels']['current']
    assert report['current']['run_status_at_snapshot'] == 'running'
    assert [r['update'] for r in report['current']['validation_history']] == [1, 7]
    circuit, groups, task = load(args.source)
    saved = torch.load(output / 'current_checkpoint.pt', weights_only=False)
    model = HomeVectorNetwork(circuit, recurrence=False)
    model.load_state_dict(saved['model'])
    from fly_connectome.home_vector.__main__ import evaluate
    expected, predictions, targets, _ = evaluate(model, circuit, groups['challenge'], .0025, 'cpu', 2.)
    with np.load(output / 'current_predictions.npz') as arrays:
        assert len(arrays.files) == 2 * len(groups['challenge'])
        for i, (row, p, y) in enumerate(zip(groups['challenge'], predictions, targets)):
            np.testing.assert_array_equal(arrays[f'prediction_{i}'], p)
            np.testing.assert_array_equal(arrays[f'target_{i}'], y)
            assert report['current']['challenge'][i] == trajectory_summary(row, p, y, .0025)
    assert report['current']['aggregate']['equal_trajectory_scaled_mse'] == expected['equal_trajectory_scaled_mse']
    assert digest(output / 'current_checkpoint.pt') == report['current']['checkpoint_sha256']
    for name in ('comparison.png', 'comparison.pdf', 'validation_history.png', 'reversal.png', 'reversal.pdf', 'long_path.png', 'long_path.pdf', 'home_vectors.png', 'home_vectors.pdf'):
        assert (output / name).stat().st_size > 0
    with (output / 'challenge_metrics.csv').open() as stream:
        table = list(csv.DictReader(stream))
    assert len(table) == len(groups['challenge']) * (2 if reference else 1)
    assert float(table[-1]['terminal_error_mm']) >= 0


def test_writer_replacing_checkpoint_after_capture_cannot_change_replay(tmp_path, monkeypatch):
    from fly_connectome.home_vector.analysis import checkpoint as module
    args = fixtures(tmp_path)
    old_hash = digest(Path(args.run) / 'best.pt')
    original = module.replay

    def replace_then_replay(snap, output, key):
        torch.save(dict(model=snap['checkpoint']['model'], update=99), Path(args.run) / 'best.pt')
        write_json(Path(args.run) / 'history.json', [dict(update=99, validation_scaled_mse=0.)])
        return original(snap, output, key)

    monkeypatch.setattr(module, 'replay', replace_then_replay)
    result = analyze(args)
    assert result['checkpoint_updates']['current'] == 7
    assert digest(Path(args.output) / 'current_checkpoint.pt') == old_hash
    assert digest(Path(args.run) / 'best.pt') != old_hash


def test_reference_requires_exact_challenges_and_explicit_task_difference_permission(tmp_path):
    args = fixtures(tmp_path)
    snap = capture(Path(args.run), Path(args.source))
    other = dict(snap, groups=copy.deepcopy(snap['groups']))
    other['groups']['challenge'][0]['protocol'][0][2] += 1
    with pytest.raises(ValueError, match='protocols'):
        validate_pair(other, snap, True)
    for key in ('dt_s', 'length_scale_mm'):
        other = dict(snap, task=dict(snap['task']))
        other['task'][key] *= 2
        with pytest.raises(ValueError, match='time steps and loss scales'):
            validate_pair(other, snap, True)
    other = dict(snap, circuit={})
    with pytest.raises(ValueError, match='circuits'):
        validate_pair(other, snap, True)
    other_source = tmp_path / 'other_source'
    shutil.copytree(args.source, other_source)
    (other_source / 'task.json').write_text('different task metadata')
    other = dict(snap, source=other_source)
    with pytest.raises(ValueError, match='allow-task-differences'):
        validate_pair(other, snap, False)
    validate_pair(other, snap, True)


def test_bad_source_and_output_paths_rejected_before_evaluation(tmp_path):
    args = fixtures(tmp_path)
    args.output = str(Path(args.run) / 'audit')
    with pytest.raises(ValueError, match='separate'):
        analyze(args)
    args.output = str(tmp_path / 'source')
    with pytest.raises(ValueError, match='separate'):
        analyze(args)
    args.output = str(tmp_path / 'audit')
    args.reference_run = str(tmp_path / 'reference')
    with pytest.raises(ValueError, match='both'):
        analyze(args)
    args.reference_run = None
    manifest = json.loads((Path(args.run) / 'manifest.json').read_text())
    manifest['settings']['source_manifest_sha256'] = 'wrong'
    write_json(Path(args.run) / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='source'):
        analyze(args)
    assert not Path(args.output).exists()


def test_evaluation_failure_marks_output_failed(tmp_path, monkeypatch):
    from fly_connectome.home_vector.analysis import checkpoint as module
    args = fixtures(tmp_path)

    def fail_replay(*args):
        raise RuntimeError('interrupted evaluation')

    monkeypatch.setattr(module, 'replay', fail_replay)
    with pytest.raises(RuntimeError, match='interrupted'):
        analyze(args)
    assert json.loads((Path(args.output) / 'manifest.json').read_text())['status'] == 'failed'


def test_saved_current_and_reference_can_replay_without_original_training_directories(tmp_path):
    args = fixtures(tmp_path)
    args.reference_run, args.reference_source = str(tmp_path / 'reference'), args.source
    analyze(args)
    saved = Path(args.output)
    before = {p.name: digest(p) for p in saved.iterdir() if p.is_file()}
    Path(args.run).rename(tmp_path / 'moved_active')
    Path(args.reference_run).rename(tmp_path / 'moved_reference')
    args.run, args.reference_run, args.reference_source = str(saved), None, None
    for key in ('current', 'reference'):
        args.snapshot_key, args.output = key, str(tmp_path / f'reloaded_{key}')
        result = analyze(args)
        assert result['checkpoint_updates']['current'] == 7
        assert digest(Path(args.output) / 'current_checkpoint.pt') == digest(saved / f'{key}_checkpoint.pt')
        with np.load(saved / f'{key}_predictions.npz') as old, np.load(Path(args.output) / 'current_predictions.npz') as new:
            for name in old.files:
                np.testing.assert_array_equal(new[name], old[name])
        meta = json.loads((Path(args.output) / 'current_metadata.json').read_text())
        assert meta['input_kind'] == 'saved_challenge_snapshot' and meta['snapshot_key'] == key
        assert verify_bundle(args.output)['status'] == 'complete'
    assert before == {p.name: digest(p) for p in saved.iterdir() if p.is_file()}


def test_saved_snapshot_rejects_missing_key_and_inconsistent_identity(tmp_path):
    args = fixtures(tmp_path)
    analyze(args)
    saved = Path(args.output)
    with pytest.raises(ValueError, match='no saved reference'):
        capture(saved, Path(args.source), snapshot_key='reference')
    metadata = json.loads((saved / 'current_metadata.json').read_text())
    metadata['checkpoint_update'] = 99
    write_json(saved / 'current_metadata.json', metadata)
    finish(saved, json.loads((saved / 'manifest.json').read_text()))
    with pytest.raises(ValueError, match='does not match its snapshot metadata'):
        capture(saved, Path(args.source))
