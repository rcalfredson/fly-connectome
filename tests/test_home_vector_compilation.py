"""Bounded compilation preserves full-history differentiation and checkpoints."""
import copy

import pytest

torch = pytest.importorskip('torch')
from fly_connectome.home_vector.model import HomeVectorNetwork, objective
from test_home_vector_gradient_assembly import four_population_circuit


@pytest.mark.parametrize('reset', [None, 0, 7, 12, 16])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_compiled_blocks_match_eager(reset, dtype, backend='aot_eager', device='cpu'):
    # Independent dtype/reset cases should not share Dynamo's small global
    # per-code-object specialization cache.
    torch._dynamo.reset()
    torch.manual_seed(71)
    eager = HomeVectorNetwork(four_population_circuit(), dtype=dtype).to(device)
    with torch.no_grad():
        eager.D.uniform_(-.3, .4)
    compiled = copy.deepcopy(eager).compile_recurrence(4, backend=backend)
    optimizers = [torch.optim.Adam(m.parameters()) for m in (eager, compiled)]
    stimulus, target = torch.rand(2, 17, 2, dtype=dtype, device=device), torch.randn(2, 17, 2, dtype=dtype, device=device)
    mask = torch.ones(2, 17, dtype=dtype, device=device)
    mask[0, -5:] = 0
    for _ in range(2):
        records = []
        for model, optimizer in zip((eager, compiled), optimizers):
            optimizer.zero_grad(set_to_none=True)
            x = stimulus.clone().requires_grad_()
            initial = torch.full((2, model.size), .2, dtype=dtype, device=device, requires_grad=True)
            pred, rates = model(x, .0025, initial=initial, reset_at=reset)
            loss, _ = objective(pred, target, rates, mask, 2., model)
            loss.backward()
            records.append([pred.detach(), rates.detach(), loss.detach(), x.grad,
                            initial.grad if initial.grad is not None else torch.zeros_like(initial),
                            *[p.grad.clone() for p in model.parameters()]])
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            model.project_parameters()
        for a, b in zip(*records):
            torch.testing.assert_close(a, b, atol=2e-7 if dtype == torch.float32 else 1e-12, rtol=2e-5)
        assert eager.state_dict().keys() == compiled.state_dict().keys()
        for key in eager.state_dict():
            torch.testing.assert_close(eager.state_dict()[key], compiled.state_dict()[key])
        for a, b in zip(optimizers[0].state.values(), optimizers[1].state.values()):
            for key in a:
                torch.testing.assert_close(a[key], b[key])


def test_block_size_validation():
    for value in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            HomeVectorNetwork(four_population_circuit()).compile_recurrence(value)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_inductor_outputs_all_gradients_and_adam(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    test_compiled_blocks_match_eager(7, torch.float32, backend='inductor', device=device)
