"""Compare the optimized loop with the pre-optimization indexed implementation."""

import copy
from collections import Counter

import pytest

torch = pytest.importorskip("torch")
from fly_connectome.home_vector.model import HomeVectorNetwork, objective
from fly_connectome.home_vector.data import Circuit
from test_home_vector_data import circuit_fixture


def indexed_forward(network, pfn, dt, initial=None, reset_at=None):
    """Frozen reference: original recurrence, including per-step time selection."""
    state = pfn.new_zeros(pfn.shape[0], network.size) if initial is None else initial
    weight = network.weights()
    input_drive = pfn @ weight[:network.input_size]
    recurrent_weight = weight[network.input_size:]
    alpha = -torch.expm1(-dt / network.tau[network.type_index])
    states = []
    for n in range(pfn.shape[1]):
        if n == reset_at:
            state = torch.zeros_like(state)
        states.append(state)
        if n + 1 < pfn.shape[1]:
            target = torch.relu(network.bias[network.type_index] + input_drive[:, n] + state @ recurrent_weight)
            state = state + alpha * (target - state)
    rates = torch.stack(states, dim=1)
    return rates @ network.D.T + network.d, rates


def four_population_circuit():
    circuit = circuit_fixture().as_dict()
    circuit["dynamic_ids"].append(13)
    circuit["dynamic_types"].append("hDeltaC")
    circuit["edges"].extend([
        dict(pre=2, post=3, contacts=2, denominator=2., source_type="hDeltaB", target_type="hDeltaC"),
        dict(pre=5, post=1, contacts=2, denominator=2., source_type="hDeltaC", target_type="vDeltaE")])
    return Circuit(**circuit)


@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["normal", "continuation", "reset"])
def test_predictions_losses_all_gradients_and_adam_updates_match_indexed_reference(expanded, dtype, mode):
    torch.manual_seed(31)
    circuit = four_population_circuit() if expanded else circuit_fixture()
    model = HomeVectorNetwork(circuit, seed=12, dtype=dtype)
    with torch.no_grad():
        model.D.uniform_(-.3, .4)  # Nonzero decoder exercises edge/intrinsic gradients immediately.
    old = copy.deepcopy(model)
    optimizers = [torch.optim.Adam(m.parameters(), lr=.001) for m in [old, model]]
    stimulus = torch.rand(3, 81, 2, dtype=dtype)
    target = torch.randn(3, 81, 2, dtype=dtype)
    mask = torch.ones(3, 81, dtype=dtype)
    mask[0, -20:] = 0
    tolerance = dict(atol=2e-7, rtol=2e-5) if dtype == torch.float32 else dict(atol=1e-12, rtol=1e-10)
    for _ in range(3):
        outputs, gradients, losses, input_gradients, initial_gradients = [], [], [], [], []
        for network, optimizer in zip([old, model], optimizers):
            optimizer.zero_grad(set_to_none=True)
            x = stimulus.clone().requires_grad_()
            initial = None if mode == "normal" else torch.full((3, network.size), .2, dtype=dtype, requires_grad=True)
            kwargs = dict(initial=initial, reset_at=40 if mode == "reset" else None)
            prediction, rates = indexed_forward(network, x, .0025, **kwargs) if network is old else network(x, .0025, **kwargs)
            loss, terms = objective(prediction, target, rates, mask, 2., network)
            loss.backward()
            outputs.append((prediction.detach(), rates.detach()))
            losses.append([loss.detach(), *[value.detach() for value in terms.values()]])
            gradients.append({n: p.grad.clone() for n, p in network.named_parameters()})
            input_gradients.append(x.grad.clone())
            initial_gradients.append(initial.grad.clone() if initial is not None else None)
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            network.project_parameters()
        for a, b in zip(outputs[0], outputs[1]):
            torch.testing.assert_close(a, b, **tolerance)
        for a, b in zip(losses[0], losses[1]):
            torch.testing.assert_close(a, b, **tolerance)
        for name in gradients[0]:
            torch.testing.assert_close(gradients[0][name], gradients[1][name], **tolerance)
        torch.testing.assert_close(input_gradients[0], input_gradients[1], **tolerance)
        if mode != "normal":
            torch.testing.assert_close(initial_gradients[0], initial_gradients[1], **tolerance)
        for name, value in old.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], **tolerance)


def backward_node_counts(output):
    pending, seen, counts = [output.grad_fn], set(), Counter()
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        counts[type(node).__name__] += 1
        pending.extend(child for child, _ in node.next_functions if child is not None)
    return counts


def test_time_selection_gradients_are_assembled_by_one_unbind_node():
    model = HomeVectorNetwork(circuit_fixture())
    x = torch.ones(2, 9, 2)
    original, _ = indexed_forward(model, x, .0025)
    optimized, _ = model(x, .0025)
    before, after = backward_node_counts(original), backward_node_counts(optimized)
    assert before["SelectBackward0"] == 8
    assert after["SelectBackward0"] == 0
    assert after["UnbindBackward0"] == 1
