"""Resumption must continue optimization, sampling and checkpoint selection."""
import json

import pytest

torch = pytest.importorskip('torch')
from fly_connectome.home_vector.__main__ import train
from fly_connectome.home_vector.data import digest, verify_bundle, write_json
from test_home_vector_balanced_loss import prepared
from test_home_vector_profile import arguments


def options(source, output):
    args = arguments(source, output)
    args.updates, args.validate_every, args.max_seconds, args.benchmark = 2, 1, 600., False
    args.rewire_seed, args.no_recurrence = None, False
    args.loss_mode = 'balanced'
    return args


def compare_state(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            compare_state(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            compare_state(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('device,block,change', [('cpu', 0, 0.), ('cpu', 4, 1.), ('cuda', 4, 0.)])
def test_split_training_matches_uninterrupted_optimizer_rng_history_and_best(tmp_path, device, block, change):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch._dynamo.reset()
    source = prepared(tmp_path / 'source')
    args = options(source, tmp_path / 'continuous')
    args.device, args.compile_block, args.change_weight, args.change_window = device, block, change, .01
    args.updates = 4
    train(args)
    args.output, args.updates = str(tmp_path / 'parent'), 2
    train(args)
    before = {p.name: digest(p) for p in (tmp_path / 'parent').iterdir()}
    parent_history = json.loads((tmp_path / 'parent' / 'history.json').read_text())
    args.output, args.resume = str(tmp_path / 'child'), str(tmp_path / 'parent')
    timing = train(args)
    assert timing['completed_updates'] == 2
    assert timing['starting_update'] == 2 and timing['final_update'] == 4
    assert timing['stopping_reason'] == 'update_limit'
    manifest = verify_bundle(args.output)
    assert manifest['settings']['starting_update'] == 2
    assert before == {p.name: digest(p) for p in (tmp_path / 'parent').iterdir()}
    for name in ('last.pt', 'best.pt'):
        actual = torch.load(tmp_path / 'child' / name, weights_only=False, map_location='cpu')
        expected = torch.load(tmp_path / 'continuous' / name, weights_only=False, map_location='cpu')
        compare_state(actual, expected)
    h = json.loads((tmp_path / 'child' / 'history.json').read_text())
    full = json.loads((tmp_path / 'continuous' / 'history.json').read_text())
    assert h[:2] == parent_history
    assert [r['update'] for r in h] == [1, 2, 3, 4]
    for a, b in zip(h, full):
        compare_state({k: v for k, v in a.items() if k != 'elapsed_s'},
                      {k: v for k, v in b.items() if k != 'elapsed_s'})
    assert all(b['elapsed_s'] > a['elapsed_s'] for a, b in zip(h, h[1:]))
    # Chain a second continuation to verify cumulative numbering and history.
    args.resume, args.output, args.updates = str(tmp_path / 'child'), str(tmp_path / 'grandchild'), 1
    result = train(args)
    assert result['starting_update'] == 4 and result['final_update'] == 5
    assert len(json.loads((tmp_path / 'grandchild' / 'history.json').read_text())) == 5


def test_prior_best_survives_worse_continuation_and_resume_uses_last(tmp_path, monkeypatch):
    import fly_connectome.home_vector.__main__ as cli
    source = prepared(tmp_path / 'source')
    args = options(source, tmp_path / 'parent')
    args.loss_mode = 'original'
    real, calls = cli.evaluate, []

    def scripted(*a, **kw):
        result, *rest = real(*a, **kw)
        if len(calls) < 2:
            score = [.1, .2][len(calls)]
            result['equal_trajectory_scaled_mse'] = score
            for row in result['per_trajectory']:
                row['mse_mm2'] = score * 4
            calls.append(True)
        return (result, *rest)

    monkeypatch.setattr(cli, 'evaluate', scripted)
    train(args)
    old_best = torch.load(tmp_path / 'parent' / 'best.pt', weights_only=False)
    assert old_best['update'] == 1

    def worse(*a, **kw):
        result, *rest = real(*a, **kw)
        result['equal_trajectory_scaled_mse'] = 100.
        for row in result['per_trajectory']:
            row['mse_mm2'] = 400.
        return (result, *rest)

    monkeypatch.setattr(cli, 'evaluate', worse)
    args.resume, args.output, args.updates = str(tmp_path / 'parent'), str(tmp_path / 'child'), 1
    train(args)
    compare_state(old_best, torch.load(tmp_path / 'child' / 'best.pt', weights_only=False))
    assert json.loads((tmp_path / 'child' / 'metrics.json').read_text())['selected_update'] == 1
    # Last must continue from the parent's last, rather than its earlier best.
    monkeypatch.setattr(cli, 'evaluate', real)
    args.resume, args.output, args.updates = None, str(tmp_path / 'continuous'), 3
    train(args)
    compare_state(torch.load(tmp_path / 'child' / 'last.pt', weights_only=False),
                  torch.load(tmp_path / 'continuous' / 'last.pt', weights_only=False))


@pytest.mark.parametrize('field,value', [('batch_size', 3), ('loss_mode', 'original'),
                                        ('seed', 7), ('compile_block', 4), ('change_weight', 1.)])
def test_incompatible_recipe_rejected_without_output(tmp_path, field, value):
    source = prepared(tmp_path / 'source')
    args = options(source, tmp_path / 'parent')
    train(args)
    args.resume, args.output = str(tmp_path / 'parent'), str(tmp_path / 'child')
    args.change_window = .01
    setattr(args, field, value)
    with pytest.raises(ValueError, match='mismatch'):
        train(args)
    assert not (tmp_path / 'child').exists()


@pytest.mark.parametrize('mode', ['original', 'balanced'])
def test_legacy_bundle_without_optional_metadata_can_resume(tmp_path, mode):
    source = prepared(tmp_path / 'source')
    args = options(source, tmp_path / 'parent')
    args.loss_mode = mode
    train(args)
    path = tmp_path / 'parent' / 'training_config.json'
    config = json.loads(path.read_text())
    del config['loss']['vector_change']
    if mode == 'original':
        for key in ('mode', 'selection_metric'):
            del config['loss'][key]
    write_json(path, config)
    path = tmp_path / 'parent' / 'history.json'
    history = json.loads(path.read_text())
    for row in history:
        del row['gradient_clipped']
        if mode == 'original':
            for key in ('validation_selection_score', 'validation_selection_metric'):
                del row[key]
    write_json(path, history)
    path = tmp_path / 'parent' / 'manifest.json'
    manifest = json.loads(path.read_text())
    for key in ('change_weight', 'change_window_s'):
        del manifest['settings'][key]
    if mode == 'original':
        del manifest['settings']['loss_mode']
    manifest['artifacts']['training_config.json'] = digest(tmp_path / 'parent' / 'training_config.json')
    manifest['artifacts']['history.json'] = digest(tmp_path / 'parent' / 'history.json')
    write_json(path, manifest)
    args.resume, args.output, args.updates = str(tmp_path / 'parent'), str(tmp_path / 'child'), 1
    assert train(args)['final_update'] == 3


def test_parent_integrity_and_dataset_identity_are_checked(tmp_path):
    source = prepared(tmp_path / 'source')
    args = options(source, tmp_path / 'parent')
    train(args)
    args.resume, args.output = str(tmp_path / 'parent'), str(tmp_path / 'child')
    other = prepared(tmp_path / 'other')
    args.source = str(other)
    with pytest.raises(ValueError, match='dataset'):
        train(args)
    args.source = str(source)
    with (tmp_path / 'parent' / 'last.pt').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        train(args)
    assert not (tmp_path / 'child').exists()


def test_cli_resume_and_additional_update_count(monkeypatch, capsys):
    import fly_connectome.home_vector.__main__ as cli
    calls = []
    monkeypatch.setattr(cli, 'train', lambda args: calls.append(args) or dict(ok=True))
    cli.main(['train', 'prepared', '--output', 'child', '--resume', 'parent', '--updates', '1500'])
    assert calls[0].resume == 'parent' and calls[0].updates == 1500
    capsys.readouterr()
