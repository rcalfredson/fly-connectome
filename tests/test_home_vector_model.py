import numpy as np
import pytest

torch = pytest.importorskip("torch")
from fly_connectome.home_vector.model import HomeVectorNetwork, objective, numpy_rollout
from test_home_vector_data import circuit_fixture


def test_forward_matches_numpy_and_common_framework():
    circuit = circuit_fixture()
    model = HomeVectorNetwork(circuit, seed=12, dtype=torch.float64)
    pfn = np.tile([1., 2.], (101, 1))
    pfn[30:60] = [0., .5]
    _, rates = model(torch.tensor(pfn)[None], .0025)
    expected = numpy_rollout(circuit, pfn, .0025, model.a.detach().numpy(),
                             model.bias.detach().numpy(), model.tau.detach().numpy())
    np.testing.assert_allclose(rates[0].detach(), expected, atol=1e-14)


def test_common_framework_is_independent_forward_reference():
    pytest.importorskip("neuprint")
    import pandas as pd
    from fly_connectome.rate_framework import RateNetwork
    circuit = circuit_fixture()
    model = HomeVectorNetwork(circuit, seed=12, dtype=torch.float64)
    ids = circuit.input_ids + circuit.dynamic_ids
    nodes = pd.DataFrame(dict(neuron_id=[str(i) for i in ids],
        cell_type=circuit.input_types + circuit.dynamic_types, tau_s=.1,
        bias=[0.] * 2 + [.1] * 3))
    weights = model.weights().detach().numpy()
    edges = pd.DataFrame([dict(source_id=str(ids[e["pre"]]),
        target_id=str(circuit.dynamic_ids[e["post"]]), effective_weight=weights[e["pre"], e["post"]],
        channel="rate", sign_evidence="assumed", sign_detail="Capacity parameter") for e in circuit.edges])
    reference = RateNetwork(nodes, edges, rate_unit="arbitrary_rate_units")
    pfn = np.tile([1., 2.], (101, 1))
    expected = reference.solve(np.zeros((101, 5)), .0025,
        prescribed={str(i): pfn[:, k] for k, i in enumerate(circuit.input_ids)}, method="exponential_euler")
    _, rates = model(torch.tensor(pfn)[None], .0025)
    np.testing.assert_allclose(rates[0].detach(), expected.rates[:, 2:], atol=1e-14)


def test_gradients_match_finite_difference_for_recurrence_tau_and_decoder():
    model = HomeVectorNetwork(circuit_fixture(), seed=1, dtype=torch.float64)
    with torch.no_grad():
        model.a.fill_(.2)
        model.D.fill_(.3)
    pfn = torch.ones(1, 12, 2, dtype=torch.float64)

    def value():
        prediction, _ = model(pfn, .0025)
        return prediction.square().sum()

    value().backward()
    for parameter, index in [(model.a, 2), (model.tau, 1), (model.D, (0, 1)), (model.bias, 0)]:
        analytical = parameter.grad[index].item()
        original = parameter[index].item()
        results = []
        for delta in [1e-6, -1e-6]:
            with torch.no_grad():
                parameter[index] = original + delta
            results.append(value().item())
        with torch.no_grad():
            parameter[index] = original
        assert analytical == pytest.approx((results[0] - results[1]) / 2e-6, rel=1e-5, abs=1e-9)


def test_loss_trajectory_weighting_and_padding_are_explicit():
    model = HomeVectorNetwork(circuit_fixture(), dtype=torch.float64)
    mask = torch.tensor([[1., 0., 0.], [1., 1., 1.]], dtype=torch.float64)
    prediction = torch.tensor([[[2., 0.], [999., 999.], [999., 999.]],
                               [[4., 0.], [4., 0.], [4., 0.]]], dtype=torch.float64)
    target = torch.zeros_like(prediction)
    rates = torch.ones(2, 3, 3, dtype=torch.float64)
    rates[0, 1:] = 999
    total, terms = objective(prediction, target, rates, mask, 2., model)
    assert terms["task"].item() == pytest.approx((1 + 4) / 2)
    assert terms["activity"].item() == 1
    assert terms["edge_prior"].item() == 0
    assert total.item() == pytest.approx(2.5 + 1e-5)


def test_decoder_has_no_temporal_state_and_reset_removes_history():
    model = HomeVectorNetwork(circuit_fixture(), dtype=torch.float64)
    with torch.no_grad():
        model.D.fill_(1)
    pfn = torch.zeros(2, 101, 2, dtype=torch.float64)
    pfn[0, :20, 0] = 1
    pfn[1, :20, 0] = 2
    prediction, _ = model(pfn, .0025)
    assert not torch.equal(prediction[0, 20:], prediction[1, 20:])
    reset, _ = model(pfn, .0025, reset_at=20)
    torch.testing.assert_close(reset[0, 20:], reset[1, 20:])


def test_continuation_and_parameter_projection():
    model = HomeVectorNetwork(circuit_fixture(), dtype=torch.float64)
    pfn = torch.ones(1, 81, 2, dtype=torch.float64)
    _, full = model(pfn, .0025)
    _, first = model(pfn[:, :31], .0025)
    _, second = model(pfn[:, 30:], .0025, initial=first[:, -1])
    torch.testing.assert_close(torch.cat((first[:, :-1], second), dim=1), full)
    with torch.no_grad():
        model.a.fill_(10)
        model.tau.fill_(.001)
        model.bias.fill_(-10)
    model.project_parameters()
    assert model.a.max() == 5
    assert model.tau.min().item() == pytest.approx(.05)
    assert model.bias.min() == -2


def test_optimizer_updates_input_edges_as_well_as_recurrence():
    model = HomeVectorNetwork(circuit_fixture(), dtype=torch.float64)
    with torch.no_grad():
        model.D.fill_(.1)
    prediction, rates = model(torch.ones(1, 100, 2, dtype=torch.float64), .0025)
    loss, _ = objective(prediction, torch.ones_like(prediction), rates,
                        torch.ones(1, 100), 1., model)
    loss.backward()
    assert model.a.grad[0].abs() > 0  # PFN input edge.
    assert model.a.grad[2:].abs().max() > 0  # Internal edge.


def test_memory_ablation_retains_input_weights_and_zeroes_recurrence():
    model = HomeVectorNetwork(circuit_fixture(), recurrence=False)
    weights = model.weights()
    assert weights[:2].abs().sum() > 0
    assert weights[2:].abs().sum() == 0
