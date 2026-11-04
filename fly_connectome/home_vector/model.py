"""Optional differentiable counterpart of the common rate equation."""

import numpy as np
import torch
from torch import nn

from .dynamics import initial_coefficients, numpy_rollout


class HomeVectorNetwork(nn.Module):
    """Only PFN traces enter forward(); geometric targets enter the loss outside."""

    def __init__(self, circuit, seed=0, recurrence=True, dtype=torch.float32):
        super().__init__()
        self.input_size, self.size = len(circuit.input_ids), len(circuit.dynamic_ids)
        self.recurrence = recurrence
        self.population_types = circuit.population_types
        self.register_buffer("type_index", torch.tensor([
            self.population_types.index(t) for t in circuit.dynamic_types]))
        self.register_buffer("pre", torch.tensor([e["pre"] for e in circuit.edges], dtype=torch.long))
        self.register_buffer("post", torch.tensor([e["post"] for e in circuit.edges], dtype=torch.long))
        self.register_buffer("counts", torch.tensor([e["contacts"] / e["denominator"] for e in circuit.edges], dtype=dtype))
        self.register_buffer("a_initial", torch.tensor(initial_coefficients(circuit, seed), dtype=dtype))
        self.a = nn.Parameter(self.a_initial.clone())
        self.bias = nn.Parameter(torch.full((len(self.population_types),), .1, dtype=dtype))
        self.tau = nn.Parameter(torch.full((len(self.population_types),), .1, dtype=dtype))
        self.D = nn.Parameter(torch.zeros(2, self.size, dtype=dtype))
        self.d = nn.Parameter(torch.zeros(2, dtype=dtype))

    def weights(self):
        effective = self.a * self.counts
        if not self.recurrence:
            effective = effective * (self.pre < self.input_size)
        weight = effective.new_zeros(self.input_size + self.size, self.size)
        return weight.index_put((self.pre, self.post), effective)

    def forward(self, pfn, dt, initial=None, reset_at=None):
        if (pfn.ndim != 3 or pfn.shape[2] != self.input_size or pfn.shape[1] < 2
                or not np.isfinite(dt) or dt <= 0 or dt > .05 / 20 + 1e-12):
            raise ValueError("Require batch/time/PFN input and dt <= 0.0025 s.")
        if not torch.isfinite(pfn).all() or (pfn < 0).any():
            raise ValueError("PFN activity must be finite and nonnegative.")
        state = pfn.new_zeros(pfn.shape[0], self.size) if initial is None else initial
        if state.shape != (pfn.shape[0], self.size):
            raise ValueError("Initial state has wrong shape.")
        weight = self.weights()
        input_drive = pfn @ weight[:self.input_size]
        # Expose time slices together: unbind's backward assembles their
        # gradients once. Repeated input_drive[:, n] selections each construct
        # a full batch/time/neuron gradient buffer during backpropagation.
        input_steps = input_drive.unbind(dim=1)
        recurrent_weight = weight[self.input_size:]
        alpha = -torch.expm1(-dt / self.tau[self.type_index])
        states = []
        for n in range(pfn.shape[1]):
            if n == reset_at:
                state = torch.zeros_like(state)
            states.append(state)
            if n + 1 < pfn.shape[1]:
                target = torch.relu(self.bias[self.type_index] + input_steps[n] + state @ recurrent_weight)
                state = state + alpha * (target - state)
        rates = torch.stack(states, dim=1)
        return rates @ self.D.T + self.d, rates

    def project_parameters(self):
        with torch.no_grad():
            self.a.clamp_(-5, 5)
            self.bias.clamp_(-2, 2)
            self.tau.clamp_(.05, .5)


def objective(prediction, target, rates, mask, scale, network):
    """Equal trajectory weighting, including penalties on valid rate samples."""
    if not np.isfinite(scale) or scale <= 0 or mask.ndim != 2 or not (mask.sum(1) > 0).all():
        raise ValueError("Require positive training-only scale and nonempty trajectory masks.")
    error = ((prediction - target) / scale).square().sum(-1)
    task = ((error * mask).sum(1) / mask.sum(1)).mean()
    activity = ((rates.square().mean(-1) * mask).sum(1) / mask.sum(1)).mean()
    prior = (network.a - network.a_initial).square().mean()
    decoder = network.D.square().mean()
    total = task + 1e-5 * activity + 1e-4 * prior + 1e-5 * decoder
    return total, dict(task=task, activity=activity, edge_prior=prior, decoder=decoder)
