"""Optional differentiable counterpart of the common rate equation."""

import numpy as np
import torch
from torch import nn

from .dynamics import initial_coefficients, numpy_rollout

LOSS_WEIGHTS = dict(activity=1e-5, edge_prior=1e-4, decoder=1e-5)


def recurrent_block(state, drives, recurrent_weight, bias, alpha):
    """Bounded tensor-only recurrence; return every new state for full BPTT."""
    states = []
    for drive in drives:
        target = torch.relu(bias + drive + state @ recurrent_weight)
        state = state + alpha * (target - state)
        states.append(state)
    return state, torch.stack(states, dim=1)


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
        self._compiled_block = None
        self._block_steps = 32

    def compile_recurrence(self, block_steps=32, backend="inductor"):
        """Opt in without changing checkpoint keys or truncating gradients.

        Compilation is lazy: first forward/backward use includes startup cost.
        Checks, input projection, decoder, loss and optimizer remain eager.
        No-grad evaluation stays eager to avoid compiling its many shapes.
        """
        if not isinstance(block_steps, int) or isinstance(block_steps, bool) or block_steps < 1:
            raise ValueError("Require a positive integer block size.")
        self._block_steps = block_steps
        self._compiled_block = torch.compile(recurrent_block, backend=backend,
                                             fullgraph=True, dynamic=False)
        return self

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
        compiled_training = self._compiled_block is not None and torch.is_grad_enabled()
        if compiled_training:
            # A time-major contiguous copy gives every [batch, neuron] view
            # the same stride regardless of total padded sequence length.
            # Unbind still assembles the drive gradients together once.
            input_steps = input_drive.transpose(0, 1).contiguous().unbind(dim=0)
        else:
            input_steps = input_drive.unbind(dim=1)
        recurrent_weight = weight[self.input_size:]
        alpha = -torch.expm1(-dt / self.tau[self.type_index])
        if compiled_training:
            chunks = []
            boundaries = [0]
            if reset_at is not None and 0 < reset_at < pfn.shape[1]:
                boundaries.append(reset_at)
            boundaries.append(pfn.shape[1])
            bias = self.bias[self.type_index]
            for start, end in zip(boundaries, boundaries[1:]):
                if start == reset_at:
                    state = torch.zeros_like(state)
                chunks.append(state.unsqueeze(1))
                for n in range(start, end - 1, self._block_steps):
                    drives = input_steps[n:min(n + self._block_steps, end - 1)]
                    # Keep short tails eager rather than compiling a new graph
                    # for every possible remainder length.
                    block = self._compiled_block if len(drives) == self._block_steps else recurrent_block
                    state, new_states = block(state, drives, recurrent_weight, bias, alpha)
                    chunks.append(new_states)
            rates = torch.cat(chunks, dim=1)
            return rates @ self.D.T + self.d, rates
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


def objective(prediction, target, rates, mask, scale, network, trajectory_weights=None, change=None):
    """Absolute task loss, optional change supervision and unchanged penalties.

    Frozen trajectory weights affect only the absolute prediction-error term.
    """
    if not np.isfinite(scale) or scale <= 0 or mask.ndim != 2 or not (mask.sum(1) > 0).all():
        raise ValueError("Require positive training-only scale and nonempty trajectory masks.")
    error = ((prediction - target) / scale).square().sum(-1)
    per_trajectory = (error * mask).sum(1) / mask.sum(1)
    if trajectory_weights is not None:
        if (trajectory_weights.shape != per_trajectory.shape or trajectory_weights.requires_grad
                or not torch.isfinite(trajectory_weights).all() or not (trajectory_weights > 0).all()):
            raise ValueError("Require one finite positive frozen weight per trajectory.")
        per_trajectory = per_trajectory * trajectory_weights
    task = per_trajectory.mean()
    activity = ((rates.square().mean(-1) * mask).sum(1) / mask.sum(1)).mean()
    prior = (network.a - network.a_initial).square().mean()
    decoder = network.D.square().mean()
    total = (task + LOSS_WEIGHTS["activity"] * activity + LOSS_WEIGHTS["edge_prior"] * prior
             + LOSS_WEIGHTS["decoder"] * decoder)
    terms = dict(task=task, activity=activity, edge_prior=prior, decoder=decoder)
    if change is not None and change["enabled"]:
        from .vector_change import change_loss
        terms["change"] = change_loss(prediction, target, mask, change)
        total = total + change["weight"] * terms["change"]
    return total, terms
