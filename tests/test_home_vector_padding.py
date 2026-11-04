import copy

import pytest

torch = pytest.importorskip("torch")
from fly_connectome.home_vector.__main__ import batch
from fly_connectome.home_vector.model import HomeVectorNetwork, objective
from fly_connectome.home_vector.training_batches import backward_grouped, length_groups
from test_home_vector_data import circuit_fixture
from test_home_vector_gradient_assembly import four_population_circuit


def records():
    rows = [dict(protocol=[[0, i * 45, .5 + i * .1, -.2], [duration, i * 45, .5 + i * .1, -.2]])
            for i, duration in enumerate([.025, .05, .1])]
    return [rows[0], rows[1], rows[2], rows[0]]  # Sampling with replacement preserves multiplicity.


def test_group_plan_preserves_every_position_and_accounts_for_saved_slots():
    groups, stats = length_groups(records(), .0025, 2.)
    assert sorted(i for group in groups for i in group) == list(range(4))
    assert list(map(len, groups)) == [3, 1]
    assert stats["original_padded_steps"] == 4 * 41
    assert stats["executed_steps"] == 3 * 21 + 41
    assert stats["valid_steps"] == 11 + 21 + 41 + 11
    assert 0 < stats["remaining_padding_fraction"] < stats["original_padding_fraction"]
    _, exact = length_groups(records(), .0025, 1.)
    assert exact["executed_steps"] == exact["valid_steps"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("ratio", [1., 2., 100.])
@pytest.mark.parametrize("expanded", [False, True])
def test_grouped_loss_every_gradient_and_adam_updates_match_padded_batch(dtype, ratio, expanded):
    torch.manual_seed(4)
    circuit = four_population_circuit() if expanded else circuit_fixture()
    padded = HomeVectorNetwork(circuit, dtype=dtype)
    with torch.no_grad():
        padded.D.uniform_(-.2, .3)
        padded.a.add_(.05)  # Nonzero edge prior, so duplicated penalties would be detected.
    grouped = copy.deepcopy(padded)
    optimizers = [torch.optim.Adam(model.parameters(), lr=.001) for model in [padded, grouped]]
    tolerance = dict(atol=2e-7, rtol=2e-5) if dtype == torch.float32 else dict(atol=1e-12, rtol=1e-10)
    for _ in range(3):
        optimizers[0].zero_grad(set_to_none=True)
        x, target, mask = batch(circuit, records(), .0025, "cpu", dtype)
        prediction, rates = padded(x, .0025)
        expected, expected_terms = objective(prediction, target, rates, mask, 2., padded)
        expected.backward()
        optimizers[1].zero_grad(set_to_none=True)
        actual, actual_terms, _ = backward_grouped(grouped, circuit, records(), .0025, "cpu", 2., ratio)
        torch.testing.assert_close(actual, expected, **tolerance)
        for key in expected_terms:
            torch.testing.assert_close(actual_terms[key], expected_terms[key], **tolerance)
        for (name, a), (_, b) in zip(padded.named_parameters(), grouped.named_parameters()):
            torch.testing.assert_close(a.grad, b.grad, **tolerance, msg=lambda msg: name + ": " + msg)
        for model, optimizer in zip([padded, grouped], optimizers):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            model.project_parameters()
        for name, value in padded.state_dict().items():
            torch.testing.assert_close(value, grouped.state_dict()[name], **tolerance)


def test_invalid_length_ratio_and_empty_batch_are_rejected():
    for ratio in [0., .9, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            length_groups(records(), .0025, ratio)
    with pytest.raises(ValueError):
        length_groups([], .0025)
