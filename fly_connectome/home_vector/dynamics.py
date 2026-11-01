"""NumPy forward reference and reproducible capacity-model initialization."""

import numpy as np

from .data import TYPES


def initial_coefficients(circuit, seed):
    rng = np.random.default_rng(seed)
    blocks = {}
    values = []
    for e in circuit.edges:
        block = (e["source_type"], e["target_type"])
        if block not in blocks:
            blocks[block] = 1. if e["pre"] < len(circuit.input_ids) else float(rng.normal(0, .1))
        values.append(blocks[block])
    return np.asarray(values)


def numpy_rollout(circuit, pfn, dt, a, bias, tau, recurrence=True):
    """Convenient frozen baseline; RateNetwork remains the independent reference."""
    n_input, size = len(circuit.input_ids), len(circuit.dynamic_ids)
    types = np.array([TYPES.index(t) for t in circuit.dynamic_types])
    weights = np.zeros((n_input + size, size))
    for value, edge in zip(a, circuit.edges):
        if recurrence or edge["pre"] < n_input:
            weights[edge["pre"], edge["post"]] = value * edge["contacts"] / edge["denominator"]
    drive = pfn @ weights[:n_input]
    rates = np.zeros((len(pfn), size))
    alpha = -np.expm1(-dt / np.asarray(tau)[types])
    for n in range(len(pfn) - 1):
        target = np.maximum(0, np.asarray(bias)[types] + drive[n] + rates[n] @ weights[n_input:])
        rates[n + 1] = rates[n] + alpha * (target - rates[n])
    return rates
