"""Validation subset accounting and read-only snapshots of active training."""
import argparse
import copy
import json
import shutil

import pytest

torch = pytest.importorskip('torch')
from fly_connectome.home_vector.__main__ import begin, evaluate, finish, load
from fly_connectome.home_vector.analysis.validation_subsets import analyze, stable_read, subset_summary
from fly_connectome.home_vector.data import digest, verify_bundle, write_json
from fly_connectome.home_vector.model import HomeVectorNetwork
from test_home_vector_profile import prepared


def test_subset_summary_averages_trajectories_and_keeps_rmse_units():
    result = subset_summary([dict(mse_mm2=1.), dict(mse_mm2=9.)], 2.)
    assert result['scaled_mse'] == 1.25
    assert result['equal_trajectory_vector_rmse_mm'] == pytest.approx(5 ** .5)
    with pytest.raises(ValueError):
        subset_summary([], 2.)


def fixtures(tmp_path):
    old_source = prepared(tmp_path / 'old_source')
    circuit, old_groups, task = load(old_source)
    source, record = begin(tmp_path / 'source', 'home_vector_prepared_v1')
    for name in ('circuit.json', 'task.json'):
        shutil.copyfile(old_source / name, source / name)
    groups = copy.deepcopy(old_groups)
    groups['validation'].append(dict(family='added', kind='long_holdout',
        protocol=[[0, 45, 1, 0], [.1, 45, 1, 0]]))
    write_json(source / 'trajectories.json', groups)
    finish(source, record)
    runs = []
    for name, data, rows, complete in [('reference', old_source, old_groups['validation'], True),
                                       ('active', source, groups['validation'], False)]:
        run, record = begin(tmp_path / name, 'home_vector_training_v1', seed=0, recurrence=True,
            source=str(data), source_manifest_sha256=digest(data / 'manifest.json'))
        shutil.copyfile(data / 'circuit.json', run / 'circuit.json')
        network = HomeVectorNetwork(circuit)
        value = evaluate(network, circuit, rows, task['dt_s'], 'cpu', task['length_scale_mm'])[0]
        torch.save(dict(model=network.state_dict(), update=7), run / 'best.pt')
        write_json(run / 'history.json', [dict(update=7, validation_scaled_mse=value['equal_trajectory_scaled_mse'])])
        if complete:
            finish(run, record)
        runs.append(run)
    return argparse.Namespace(run=str(runs[1]), source=str(source), reference_run=str(runs[0]),
                              reference_source=str(old_source), output=str(tmp_path / 'audit'))


def test_running_run_is_snapshotted_without_writes_and_subsets_recombine(tmp_path):
    args = fixtures(tmp_path)
    before = {str(p): digest(p) for p in tmp_path.rglob('*') if p.is_file()}
    report = analyze(args)
    assert report['current']['checkpoint_update'] == 7
    assert report['current']['original_run_status_at_snapshot'] == 'running'
    assert report['current']['matches_own_logged_validation']
    assert report['reference']['matches_own_logged_validation']
    s = report['current']['subsets']
    assert s['original']['trajectories'] == 3
    assert s['added_long']['trajectories'] == 1
    assert s['combined']['scaled_mse'] == pytest.approx((3*s['original']['scaled_mse']+s['added_long']['scaled_mse'])/4)
    assert verify_bundle(args.output)['status'] == 'complete'
    assert all(digest(path) == value for path, value in before.items())
    # A later write to the active checkpoint cannot affect the evaluated copy.
    copied = stable_read(tmp_path / 'audit' / 'current_checkpoint.pt')
    (tmp_path / 'active' / 'best.pt').write_bytes(b'new checkpoint being written')
    assert stable_read(tmp_path / 'audit' / 'current_checkpoint.pt') == copied


def test_changed_validation_prefix_is_rejected_before_creating_audit(tmp_path):
    args = fixtures(tmp_path)
    from pathlib import Path
    source = Path(args.source)
    groups = json.loads((source / 'trajectories.json').read_text())
    groups['validation'][0]['protocol'][0][2] += 1
    write_json(source / 'trajectories.json', groups)
    finish(source, json.loads((source / 'manifest.json').read_text()))
    # Reflect the revised source in the synthetic active run's provenance.
    run = Path(args.run)
    manifest = json.loads((run / 'manifest.json').read_text())
    manifest['settings']['source_manifest_sha256'] = digest(source / 'manifest.json')
    write_json(run / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='exact retained prefix'):
        analyze(args)
    assert not Path(args.output).exists()


def test_checkpoint_snapshot_retries_an_incomplete_serialization(tmp_path, monkeypatch):
    from pathlib import Path
    from fly_connectome.home_vector.analysis import validation_subsets as module
    args = fixtures(tmp_path)
    original = torch.load
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise EOFError('checkpoint writer has not finished')
        return original(*args, **kwargs)

    monkeypatch.setattr(module.torch, 'load', interrupted)
    snap = module.snapshot(Path(args.run))
    assert snap['checkpoint']['update'] == 7
    assert len(calls) == 2


def test_empty_in_progress_file_is_not_a_valid_snapshot(tmp_path):
    path = tmp_path / 'empty.pt'
    path.write_bytes(b'')
    with pytest.raises(ValueError, match='stable snapshot'):
        stable_read(path, attempts=1)
