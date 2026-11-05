"""Window supervision math, backward compatibility and compiled trainer integration."""
import copy
import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from fly_connectome.home_vector.__main__ import batch, train
from fly_connectome.home_vector.data import sample_protocol, verify_bundle
from fly_connectome.home_vector.loss_balance import example_weights, loss_plan
from fly_connectome.home_vector.model import HomeVectorNetwork, objective
from fly_connectome.home_vector.training_batches import backward_grouped
from fly_connectome.home_vector.vector_change import change_plan, change_loss, change_metrics, window_steps
from test_home_vector_balanced_loss import prepared, rows
from test_home_vector_data import circuit_fixture
from test_home_vector_profile import arguments


def test_scale_is_one_equal_trajectory_training_rms_and_zero_predictor_has_unit_loss():
    records, dt = rows(), .0025
    plan = change_plan(1., .01, records, dt)
    energy = []
    predictions, targets = [], []
    for row in records:
        _, y = sample_protocol(row['protocol'], dt)
        energy.append(np.mean(np.sum((y[4:] - y[:-4]) ** 2, axis=1)))
        targets.append(y)
        predictions.append(np.zeros_like(y))
    assert plan['scale_mm'] ** 2 == pytest.approx(np.mean(energy))
    result = change_metrics(predictions, targets, records, dt, plan)
    assert result['scaled_mse'] == pytest.approx(1.)
    before = copy.deepcopy(plan)
    change_metrics([p + 1e5 for p in predictions], targets, records, dt, plan)
    assert plan == before
    assert window_steps(1., .0025) == 400
    assert window_steps(1., .00125) == 800
    assert not change_plan(0., 1., [], .003)["enabled"]  # Disabled mode adds no grid constraint.


def test_change_loss_masking_endpoint_gradients_and_explicit_window_reference():
    torch.manual_seed(7)
    prediction = torch.randn(2, 7, 2, dtype=torch.float64, requires_grad=True)
    target = torch.randn(2, 7, 2, dtype=torch.float64)
    mask = torch.tensor([[1., 1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 0., 0., 0.]])
    plan = dict(window_steps=2, scale_mm=.3)
    actual = change_loss(prediction, target, mask, plan)
    expected = torch.stack([torch.stack([
        (((prediction[i, j + 2] - prediction[i, j]) - (target[i, j + 2] - target[i, j])) / .3).square().sum()
        for j in range(length - 2)]).mean() for i, length in enumerate([7, 4])]).mean()
    torch.testing.assert_close(actual, expected)
    expected_grad = torch.autograd.grad(expected, prediction, retain_graph=True)[0]
    actual.backward()
    torch.testing.assert_close(prediction.grad, expected_grad)
    assert prediction.grad[0, 0].abs().sum() > 0
    assert prediction.grad[0, -1].abs().sum() > 0
    assert prediction.grad[1, 4:].abs().sum() == 0
    altered = prediction.detach().clone()
    altered[1, 4:] = 1e8
    torch.testing.assert_close(change_loss(altered, target, mask, plan), actual)


def test_pauses_reversals_offsets_and_plateau():
    target = torch.tensor([[[0., 0.], [-1., 0.], [-2., 0.], [-1., 0.], [0., 0.], [0., 0.]]])
    mask, plan = torch.ones(1, 6), dict(window_steps=1, scale_mm=1.)
    assert change_loss(target + 17., target, mask, plan) == 0
    assert change_loss(torch.zeros_like(target), target, mask, plan) > 0
    paused_target = torch.zeros(1, 5, 2)
    drifting = torch.arange(5.).view(1, 5, 1).expand(1, 5, 2)
    assert change_loss(drifting, paused_target, torch.ones(1, 5), plan) == 2.


@pytest.mark.parametrize('mode', ['original', 'balanced'])
def test_disabled_is_bitwise_identical_and_grouped_combined_gradients_match(mode):
    torch.manual_seed(5)
    circuit = circuit_fixture()
    model = HomeVectorNetwork(circuit, dtype=torch.float64)
    with torch.no_grad():
        model.D.uniform_(-.2, .3)
        model.a.add_(.05)
    x, target, mask = batch(circuit, rows(), .0025, 'cpu', torch.float64)
    p, rates = model(x, .0025)
    weights = example_weights(rows(), loss_plan(mode, rows(), .0025), model.a)
    a, at = objective(p, target, rates, mask, 2., model, weights)
    b, bt = objective(p, target, rates, mask, 2., model, weights, change_plan(0., 1., rows(), .0025))
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert at.keys() == bt.keys()
    for key in at:
        torch.testing.assert_close(at[key], bt[key], atol=0, rtol=0)
    plan = change_plan(.7, .01, rows(), .0025)
    loss, terms = objective(p, target, rates, mask, 2., model, weights, plan)
    torch.testing.assert_close(terms['change'], change_loss(p, target, mask, plan))
    torch.testing.assert_close(loss, a + .7 * terms['change'])
    for key in at:
        torch.testing.assert_close(terms[key], at[key], atol=0, rtol=0)
    loss.backward()
    grouped = copy.deepcopy(model)
    grouped.zero_grad(set_to_none=True)
    actual, gt, _ = backward_grouped(grouped, circuit, rows(), .0025, 'cpu', 2., 1.,
                                     trajectory_weights=weights, change=plan)
    torch.testing.assert_close(actual, loss)
    for key in terms:
        torch.testing.assert_close(gt[key], terms[key])
    for ap, bp in zip(model.parameters(), grouped.parameters()):
        torch.testing.assert_close(ap.grad, bp.grad, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_eager_and_compiled_training_change_diagnostics_and_selection(tmp_path, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch._dynamo.reset()
    source = prepared(tmp_path / 'source')
    records = json.loads((source / 'trajectories.json').read_text())
    args = arguments(source, tmp_path / 'unused')
    args.change_weight, args.change_window = 1., .01
    args.device, args.loss_mode = device, 'balanced'
    args.updates, args.validate_every, args.max_seconds, args.benchmark = 2, 1, 600., False
    args.rewire_seed, args.no_recurrence = None, False
    for name, block in [('eager', 0), ('compiled', 4)]:
        args.output, args.compile_block = str(tmp_path / name), block
        train(args)
        manifest = verify_bundle(args.output)
        assert manifest['settings']['change_weight'] == 1.
        config = json.loads((tmp_path / name / 'training_config.json').read_text())
        plan = config['loss']['vector_change']
        assert plan == change_plan(1., .01, records['train'], .0025)
        h = json.loads((tmp_path / name / 'history.json').read_text())
        metrics = json.loads((tmp_path / name / 'metrics.json').read_text())
        assert metrics['selected_update'] == min(h, key=lambda row: row['validation_selection_score'])['update']
        assert all(r['validation_selection_metric'] == 'balanced_group_scaled_mse' for r in h)
        assert all(r['change'] >= 0 and set(r['validation_change']['subsets']) == {'original', 'long'} for r in h)
        for split in ['validation', 'test', 'challenge']:
            with np.load(tmp_path / name / f'{split}_predictions.npz') as saved:
                predictions = [saved[f'prediction_{i}'] for i in range(len(records[split]))]
                targets = [saved[f'target_{i}'] for i in range(len(records[split]))]
            expected = change_metrics(predictions, targets, records[split], .0025, plan)
            assert metrics[split]['vector_change'] == expected
    a, b = [torch.load(tmp_path / name / 'last.pt', weights_only=False, map_location='cpu') for name in ['eager', 'compiled']]
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], atol=2e-7, rtol=2e-5)
    assert metrics['numerical_validation']['passed']


@pytest.mark.parametrize('weight,window', [(-1., 1.), (float('nan'), 1.), (1., 0.), (1., .003), (1., 1.)])
def test_invalid_change_config_rejected_before_output_creation(tmp_path, weight, window):
    args = arguments(prepared(tmp_path / 'source'), tmp_path / 'run')
    args.change_weight, args.change_window = weight, window
    args.max_seconds, args.validate_every = 60., 1
    with pytest.raises(ValueError, match='[Vv]ector-change'):
        train(args)
    assert not (tmp_path / 'run').exists()


def test_stationary_training_has_no_scale_and_half_step_preserves_physical_window():
    records = rows()
    for row in records:
        for control in row['protocol']:
            control[2:] = [0., 0.]
    with pytest.raises(ValueError, match='target energy'):
        change_plan(1., .01, records, .0025)
    plan = dict(window_s=.5, scale_mm=1.)
    results = []
    for dt in [.25, .125]:
        time = np.arange(0, 2 + dt, dt)
        p = np.stack([time, time], axis=1)
        results.append(change_metrics([p], [np.zeros_like(p)], [rows()[0]], dt, plan)['scaled_mse'])
    assert results == pytest.approx([.5, .5])


def test_cli_change_flags_default_off(monkeypatch, capsys):
    import fly_connectome.home_vector.__main__ as cli
    calls = []
    monkeypatch.setattr(cli, 'train', lambda args: calls.append(args) or dict(ok=True))
    cli.main(['train', 'source', '--output', 'fresh'])
    cli.main(['train', 'source', '--output', 'fresh', '--change-weight', '1', '--change-window', '.5'])
    assert (calls[0].change_weight, calls[0].change_window) == (0., 1.)
    assert (calls[1].change_weight, calls[1].change_window) == (1., .5)
    capsys.readouterr()
