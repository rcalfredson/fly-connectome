"""Audit guards for the combined batch-size/compilation experiment."""
import argparse
import copy

import pytest

torch = pytest.importorskip('torch')
from fly_connectome.home_vector.__main__ import begin, finish
from fly_connectome.home_vector.compare_compilation import compare, matched_rows, tensor_tree_difference
from fly_connectome.home_vector.data import digest, verify_bundle, write_json


def row():
    return dict(update=1, measured=True, batch=dict(indices=[1, 1]), loss=2., task=2.,
                activity=.1, edge_prior=0., decoder=0., gradient_norm=.5, wall_s=4.)


def test_history_guard_rejects_resampled_batches_and_different_warmups():
    a = row()
    for change in ({'batch': dict(indices=[1, 2])}, {'measured': False}, {'loss': 20.}):
        b = dict(a, **change)
        with pytest.raises(ValueError):
            matched_rows([a], [b])


def test_optimizer_moments_are_checked():
    a = dict(state={0: dict(exp_avg=torch.tensor([.1]), exp_avg_sq=torch.tensor([.01]))})
    b = copy.deepcopy(a)
    assert tensor_tree_difference(a, b) == 0
    b['state'][0]['exp_avg_sq'].add_(.1)
    with pytest.raises(AssertionError):
        tensor_tree_difference(a, b)


def test_complete_factorial_audit_and_input_protection(tmp_path):
    source, record = begin(tmp_path / 'source', 'home_vector_prepared_v1')
    write_json(source / 'fixture.json', dict(synthetic=True))
    finish(source, record)
    args = argparse.Namespace(source=str(source), output=str(tmp_path / 'comparison'))
    # Short synthetic artifacts exercise provenance and matching; these are not timing measurements.
    for label, batch_size, block, seconds in [('small_eager', 8, 0, 8.), ('small_compiled', 8, 32, 2.),
                                              ('large_eager', 128, 0, 12.), ('large_compiled', 128, 32, 4.)]:
        path, record = begin(tmp_path / label, 'home_vector_training_profile_v1',
            source=str(source), source_manifest_sha256=digest(source / 'manifest.json'),
            batching='padded', batch_size=batch_size, compile_block=block,
            seed=0, updates=1, warmup=0, threads=1, device='cuda', trace_seconds=0.)
        r = row()
        r['wall_s'] = seconds
        r['batch']['indices'] = [1] * batch_size
        write_json(path / 'baseline_updates.json', [r])
        torch.save(dict(model=dict(a=torch.tensor([1.])), optimizer=dict(state={})), path / 'baseline_final.pt')
        finish(path, record)
        setattr(args, label, str(path))
    hashes = {str(p): digest(p) for p in tmp_path.rglob('*') if p.is_file()}
    result = compare(args)
    assert result['compilation_speedup_small'] == 4.
    assert result['compilation_speedup_large'] == 3.
    assert result['speedup_ratio_large_to_small'] == .75
    assert result['cases']['large_compiled']['exposure_matched_update_count'] == 125
    assert result['cases']['large_compiled']['exposure_matched_update_seconds'] == 500
    assert verify_bundle(args.output)['status'] == 'complete'
    assert all(digest(path) == value for path, value in hashes.items())
    with pytest.raises(FileExistsError):
        compare(args)
    args.output = str(source / 'nested')
    with pytest.raises(ValueError):
        compare(args)
