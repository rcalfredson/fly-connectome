"""Opt-in phenomenological hDeltaG sugar probe, separate from production dynamics.

This is candidate B (additional relaxation), not an identified cellular mechanism.
Only PFNs and an externally supplied scalar event gate enter the rollout.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SugarResponse:
    rise_tau_s: float
    recovery_tau_s: float
    relaxation_rate_per_s: float
    reference_activity: float

    def __post_init__(self):
        values = [self.rise_tau_s, self.recovery_tau_s,
                  self.relaxation_rate_per_s, self.reference_activity]
        if (not np.isfinite(values).all() or self.rise_tau_s <= 0
                or self.recovery_tau_s <= 0 or self.relaxation_rate_per_s < 0
                or self.reference_activity < 0):
            raise ValueError('Require finite positive timescales and nonnegative rate/reference.')


def event_gate(time_s, event_times_s, response):
    """Causal, bounded rise-and-decay kernel; max aggregation is an explicit assumption."""
    time, events = np.asarray(time_s, dtype=float), np.asarray(event_times_s, dtype=float)
    if (time.ndim != 1 or events.ndim != 1 or not np.isfinite(time).all()
            or not np.isfinite(events).all() or (time < 0).any() or (events < 0).any()
            or (np.diff(time) <= 0).any() or (np.diff(events) <= 0).any()):
        raise ValueError('Require finite increasing nonnegative times and event times.')
    gate = np.zeros_like(time)
    for event in events:
        elapsed = np.maximum(time - event, 0.)
        pulse = -np.expm1(-elapsed / response.rise_tau_s) * np.exp(-elapsed / response.recovery_tau_s)
        gate = np.maximum(gate, pulse)
    return gate


def relaxation_step(state, target, tau, dt, mask, gate, response):
    """Exact frozen-target decay per step, matching the native update when disabled.

    dr_i/dt = (target_i-r_i)/tau_i - gate*k*(r_i-reference), hDeltaG only.
    The target is recomputed from the current recurrent state every step.
    """
    alpha = -torch.expm1(-dt / tau)
    native = state + alpha * (target - state)
    if gate == 0. or response.relaxation_rate_per_s == 0.:
        return native
    extra = gate * response.relaxation_rate_per_s
    decay = 1. / tau + extra
    equilibrium = (target / tau + extra * response.reference_activity) / decay
    modified = state + (-torch.expm1(-dt * decay)) * (equilibrium - state)
    return torch.where(mask, modified, native)


def rollout(network, pfn, dt, gate, response, hdeltag_mask):
    """Frozen CPU eager rollout; no target, position, fitting or state reset argument."""
    if torch.is_grad_enabled():
        raise ValueError('Sugar probes require torch.no_grad().')
    if (pfn.ndim != 3 or pfn.shape[2] != network.input_size or pfn.shape[1] < 2
            or pfn.device.type != 'cpu' or not np.isfinite(dt) or not 0 < dt <= .0025
            or not torch.isfinite(pfn).all() or (pfn < 0).any()):
        raise ValueError('Require finite nonnegative CPU PFNs and dt <= 0.0025 s.')
    gates = np.asarray(gate, dtype=float)
    if (gates.shape != (pfn.shape[1],) or not np.isfinite(gates).all()
            or (gates < 0).any() or (gates > 1).any()):
        raise ValueError('Require one gate in [0,1] per state time.')
    if (hdeltag_mask.shape != (network.size,) or hdeltag_mask.dtype != torch.bool
            or hdeltag_mask.device != pfn.device):
        raise ValueError('Require an aligned boolean hDeltaG mask.')
    weight = network.weights()
    drives = (pfn @ weight[:network.input_size]).unbind(dim=1)
    recurrent = weight[network.input_size:]
    tau = network.tau[network.type_index]
    if not torch.isfinite(tau).all() or (tau <= 0).any():
        raise ValueError('Require positive finite frozen time constants.')
    state = pfn.new_zeros(pfn.shape[0], network.size)
    states = []
    for n in range(pfn.shape[1]):
        states.append(state)
        if n + 1 < pfn.shape[1]:
            target = torch.relu(network.bias[network.type_index] + drives[n] + state @ recurrent)
            state = relaxation_step(state, target, tau, dt, hdeltag_mask, float(gates[n]), response)
    rates = torch.stack(states, dim=1)
    if not torch.isfinite(rates).all() or (rates < 0).any() or rates.max() > 1e6:
        raise ValueError('Sugar probe exceeded the numerical activity guard.')
    return rates @ network.D.T + network.d, rates
