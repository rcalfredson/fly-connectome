import copy
import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from fly_connectome.home_vector.__main__ import batch, begin, finish, train
from fly_connectome.home_vector.data import verify_bundle, write_json
from fly_connectome.home_vector.loss_balance import example_weights, loss_plan, trajectory_group, validation_scores
from fly_connectome.home_vector.model import HomeVectorNetwork, LOSS_WEIGHTS, objective
from fly_connectome.home_vector.training_batches import backward_grouped
from test_home_vector_data import circuit_fixture
from test_home_vector_profile import arguments


def rows():
    return [dict(family=f'f-{i}', kind=kind,
                 protocol=[[0, i * 45, speed, 0], [duration, i * 45, speed, 0]])
            for i, (kind, speed, duration) in enumerate([
                ('random', 1., .025), ('random', 2., .05),
                ('long_training', 3., .075), ('long_training', 4., .1)])]


def prepared(path):
    output, record = begin(path, 'home_vector_prepared_v1')
    write_json(output / 'circuit.json', circuit_fixture().as_dict())
    write_json(output / 'task.json', dict(dt_s=.0025, length_scale_mm=2.))
    splits = dict(train=rows())
    for split in ('validation', 'test', 'challenge'):
        splits[split] = copy.deepcopy(rows())
        for row in splits[split][2:]:
            row['kind'] = 'long_holdout'
    # Different validation proportions must not implicitly change group selection weights.
    splits['validation'] = splits['validation'][:2] * 2 + splits['validation'][2:]
    write_json(output / 'trajectories.json', splits)
    finish(output, record)
    return output


def test_training_energy_weights_equalize_zero_predictor_and_have_mean_one():
    plan = loss_plan('balanced', rows(), .0025)
    energy, weights = plan['training_target_energy_mm2'], plan['weights']
    assert np.mean(list(weights.values())) == pytest.approx(1.)
    assert weights['original'] * energy['original'] == pytest.approx(weights['long'] * energy['long'])
    assert trajectory_group({'kind': 'long'}) == 'original'
    assert loss_plan('original', [], .0025)['mode'] == 'original'
    for bad in ([], rows()[:2], rows()[:3]):
        with pytest.raises(ValueError, match='equal nonempty'):
            loss_plan('balanced', bad, .0025)
    stationary = copy.deepcopy(rows())
    for row in stationary:
        for control in row['protocol']:
            control[2:] = [0, 0]
    with pytest.raises(ValueError, match='positive'):
        loss_plan('balanced', stationary, .0025)


def test_validation_balances_group_means_and_keeps_raw_score():
    records = rows()[:2] * 2 + rows()[2:]
    plan = loss_plan('balanced', rows(), .0025)
    result = dict(equal_trajectory_scaled_mse=12.,
                  per_trajectory=[dict(mse_mm2=v) for v in [4., 4., 4., 4., 100., 100.]])
    scores = validation_scores(result, records, 2., plan)
    assert scores['validation_selection_score'] == pytest.approx(
        (plan['weights']['original'] + 25 * plan['weights']['long']) / 2)
    assert scores['validation_scaled_mse'] == 12.
    assert scores['validation_subsets']['original']['count'] == 4
    # Evaluation errors cannot alter the frozen training normalization.
    before = copy.deepcopy(plan)
    result['per_trajectory'][0]['mse_mm2'] = 1e12
    validation_scores(result, records, 2., plan)
    assert plan == before


def test_weighted_task_reference_gradients_and_grouped_accumulation():
    model = HomeVectorNetwork(circuit_fixture(), dtype=torch.float64)
    with torch.no_grad():
        model.D.fill_(.2)
        model.a.add_(.05)
    grouped = copy.deepcopy(model)
    x, target, mask = batch(circuit_fixture(), rows(), .0025, 'cpu', torch.float64)
    weights = example_weights(rows(), loss_plan('balanced', rows(), .0025), model.a)
    prediction, rates = model(x, .0025)
    loss, terms = objective(prediction, target, rates, mask, 2., model, weights)
    _, raw = objective(prediction, target, rates, mask, 2., model)
    expected_task = ((((prediction - target) / 2).square().sum(-1) * mask).sum(1)
                     / mask.sum(1) * weights).mean()
    torch.testing.assert_close(terms['task'], expected_task)
    for key in ('activity', 'edge_prior', 'decoder'):
        torch.testing.assert_close(terms[key], raw[key], atol=0, rtol=0)
    reference = expected_task + sum(LOSS_WEIGHTS[k] * raw[k] for k in LOSS_WEIGHTS)
    reference_grad = torch.autograd.grad(reference, tuple(model.parameters()), retain_graph=True)
    loss.backward()
    for parameter, expected in zip(model.parameters(), reference_grad):
        torch.testing.assert_close(parameter.grad, expected)
    actual, other, _ = backward_grouped(grouped, circuit_fixture(), rows(), .0025, 'cpu', 2.,
                                         1., trajectory_weights=weights)
    torch.testing.assert_close(actual, loss)
    for key in terms:
        torch.testing.assert_close(other[key], terms[key])
    for a, b in zip(model.parameters(), grouped.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_balanced_compiled_training_logs_subsets_and_reloads(tmp_path, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch._dynamo.reset()
    args = arguments(prepared(tmp_path / 'source'), tmp_path / 'run')
    args.loss_mode, args.compile_block, args.device = 'balanced', 4, device
    args.updates, args.validate_every, args.benchmark = 2, 1, False
    args.max_seconds, args.rewire_seed, args.no_recurrence = 600., None, False
    train(args)
    assert verify_bundle(args.output)['settings']['loss_mode'] == 'balanced'
    output = tmp_path / 'run'
    config = json.loads((output / 'training_config.json').read_text())
    assert config['loss']['weights'] == loss_plan('balanced', rows(), .0025)['weights']
    history = json.loads((output / 'history.json').read_text())
    metrics = json.loads((output / 'metrics.json').read_text())
    chosen = min(history, key=lambda r: r['validation_selection_score'])
    assert metrics['selected_update'] == chosen['update']
    assert metrics['selection']['validation_selection_score'] == pytest.approx(chosen['validation_selection_score'])
    assert metrics['validation']['equal_trajectory_scaled_mse'] == chosen['validation_scaled_mse']
    assert all(set(r['validation_subsets']) == {'original', 'long'} for r in history)
    assert all('gradient_clipped' in r for r in history)
    model = HomeVectorNetwork(circuit_fixture())
    saved = torch.load(output / 'best.pt', weights_only=False, map_location='cpu')
    model.load_state_dict(saved['model'], strict=True)
    assert model._compiled_block is None


def test_checkpoint_uses_balanced_score_instead_of_historical_score(tmp_path, monkeypatch):
    import fly_connectome.home_vector.__main__ as cli
    real_evaluate, calls = cli.evaluate, []

    def scripted(*args, **kwargs):
        result, *rest = real_evaluate(*args, **kwargs)
        if len(calls) < 2:
            # First is better for original paths, second for long paths. Raw
            # combined score chooses second; energy-balanced score chooses first.
            original, long = [(1., 100.), (10., 1.)][len(calls)]
            for item, source in zip(result['per_trajectory'], args[2]):
                item['mse_mm2'] = original if trajectory_group(source) == 'original' else long
            result['equal_trajectory_scaled_mse'] = np.mean([r['mse_mm2'] for r in result['per_trajectory']]) / 4
            calls.append(True)
        return (result, *rest)

    monkeypatch.setattr(cli, 'evaluate', scripted)
    args = arguments(prepared(tmp_path / 'source'), tmp_path / 'run')
    args.loss_mode, args.updates, args.validate_every, args.benchmark = 'balanced', 2, 1, False
    args.max_seconds, args.rewire_seed, args.no_recurrence = 60., None, False
    train(args)
    history = json.loads((tmp_path / 'run' / 'history.json').read_text())
    assert history[0]['validation_scaled_mse'] > history[1]['validation_scaled_mse']
    assert history[0]['validation_selection_score'] < history[1]['validation_selection_score']
    saved = torch.load(tmp_path / 'run' / 'best.pt', weights_only=False)
    assert saved['update'] == 1


def test_cli_default_and_explicit_loss_modes(monkeypatch, capsys):
    import fly_connectome.home_vector.__main__ as cli
    selected = []
    monkeypatch.setattr(cli, 'train', lambda args: selected.append(args.loss_mode) or dict(ok=True))
    for option in ([], ['--loss-mode', 'original'], ['--loss-mode', 'balanced']):
        cli.main(['train', 'source', '--output', 'fresh', *option])
    assert selected == ['original', 'original', 'balanced']
    capsys.readouterr()


def test_invalid_frozen_weights_are_rejected():
    model = HomeVectorNetwork(circuit_fixture())
    x, target, mask = batch(circuit_fixture(), rows(), .0025, 'cpu', torch.float32)
    prediction, rates = model(x, .0025)
    for weights in (torch.ones(3), torch.tensor([1., 1., 0., 1.]),
                    torch.tensor([1., 1., float('nan'), 1.]), torch.ones(4, requires_grad=True)):
        with pytest.raises(ValueError, match='frozen weight'):
            objective(prediction, target, rates, mask, 2., model, weights)
