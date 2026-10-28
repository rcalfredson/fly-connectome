"""Discrete fluorescence reference and an explicitly hypothetical rate adapter.

The eight indices are idealized FB columns, NOT eight vDeltaE neurons or a
mapping of the 18 MaleCNS bodies. No anatomy or movement encoding is inferred.
"""
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .connectivity import ValidationError
from .rate_framework import RateNetwork, RateState, _finite


@dataclass(frozen=True)
class JankeParameters:
    # Figure 4.1 example, reported on a one-second basis. Sampling conventions
    # remain unresolved; these are explicit demonstration coefficients only.
    reference_dt_s: float = 1.
    k1: float = .99
    c: float = .001
    k2: float = .80
    a: float = .07
    b: float = 1.

    def validate(self):
        _finite(list(asdict(self).values()), "Janke parameters")
        if self.reference_dt_s <= 0 or not 0 < self.k1 <= 1 or not 0 < self.k2 <= 1:
            raise ValidationError("Use a positive reference timestep and retention factors in (0, 1].")


def _inputs(values, resets):
    values = _finite(values, "Prescribed vDeltaE column signal")
    if values.ndim != 2 or values.shape[1] != 8 or len(values) < 2 or (values < 0).any():
        raise ValidationError("Supply nonnegative signal samples for eight idealized columns.")
    if resets is None:
        resets = np.zeros(len(values), dtype=bool)
    resets = np.asarray(resets)
    if resets.shape != (len(values),) or resets.dtype != bool:
        raise ValidationError("Reset flags must be boolean and match signal samples.")
    return values, resets


def discrete_reference(values, parameters, *, resets=None):
    """Previous-sample drive; reset overrides all predicted states/output.

    Initial states are zero. An instantaneous full reset is a declared numerical
    convention; residual biological bumps and exact source update order remain
    unresolved. No rectification is applied to signed fluorescence components.
    """
    parameters.validate()
    values, resets = _inputs(values, resets)
    p = parameters
    centered = values - values.mean(axis=1, keepdims=True)
    opposite = (np.arange(8) + 4) % 8
    mean = np.zeros(len(values))
    integrated = np.zeros_like(values)
    output = np.zeros_like(values)
    for n in range(1, len(values)):
        if resets[n]:
            continue
        mean[n] = p.k1 * mean[n - 1] + p.c
        integrated[n] = p.k2 * integrated[n - 1] - p.a * centered[n - 1, opposite]
        output[n] = mean[n] + integrated[n] + p.b * centered[n - 1]
    if not np.isfinite(output).all():
        raise ValidationError("Nonfinite fluorescence reference.")
    return dict(time_s=np.arange(len(values)) * p.reference_dt_s, mean=mean,
                integrated=integrated, fluorescence=output, centered_input=centered,
                resets=resets.copy())


def absolute_input_control(values, parameters, *, resets=None):
    """Discrete mathematical control using absolute input; NOT the thesis model.

    Retains the declared coefficients and reset convention but does not subtract
    the spatial mean. The integrated state can therefore have a nonzero mean.
    """
    parameters.validate()
    values, resets = _inputs(values, resets)
    p = parameters
    mean = np.zeros(len(values))
    integrated = np.zeros_like(values)
    output = np.zeros_like(values)
    opposite = (np.arange(8) + 4) % 8
    for n in range(1, len(values)):
        if resets[n]:
            continue
        mean[n] = p.k1 * mean[n - 1] + p.c
        integrated[n] = p.k2 * integrated[n - 1] - p.a * values[n - 1, opposite]
        output[n] = mean[n] + integrated[n] + p.b * values[n - 1]
    if not np.isfinite(output).all():
        raise ValidationError("Nonfinite absolute-input mathematical control.")
    return dict(mean=mean, integrated=integrated, latent_output=output)


def rate_candidate(values, parameters, *, dt_s=.005, rate_tau_s=.1, resets=None,
                   input_mode="centered"):
    """Common rate solver with a slow excitability offset and imposed baseline.

    Embeds the discrete leak/increment exactly under held inputs, then adds an
    ordinary nonnegative rate state with provisional rate_tau_s. The baseline
    is prescribed, not an explained circuit mechanism. In centered mode the
    spatial mean correction is also imposed. Absolute mode removes that
    correction, testing a different hypothesis with the same coupling. Neither
    mode supplies a calibrated fluorescence observation model.
    """
    parameters.validate()
    values, resets = _inputs(values, resets)
    p = parameters
    if input_mode not in {"centered", "absolute"}:
        raise ValidationError("Candidate input_mode must be centered or absolute.")
    if p.k1 == 1 or p.k2 == 1 or not np.isfinite(dt_s) or dt_s <= 0:
        raise ValidationError("Candidate embedding requires 0 < k1,k2 < 1 and positive dt.")
    ratio = p.reference_dt_s / dt_s
    steps = round(ratio)
    if steps < 1 or not np.isclose(ratio, steps, rtol=0, atol=1e-8):
        raise ValidationError("Candidate dt must divide the reference timestep exactly.")
    if not np.isfinite(rate_tau_s) or rate_tau_s <= 0:
        raise ValidationError("Candidate rate tau must be finite and positive.")
    ids = [f"vE_column_{i}" for i in range(8)] + [f"hG_output_{i}" for i in range(8)]
    neurons = pd.DataFrame(dict(neuron_id=ids, cell_type=["prescribed_vE_column"] * 8 + ["candidate_hG"] * 8,
        tau_s=rate_tau_s, slow_tau_s=[0.] * 8 + [-p.reference_dt_s / np.log(p.k2)] * 8))
    edges = []
    for i in range(8):
        for pre, channel, weight in [(i, "rate", p.b), ((i + 4) % 8, "excitability", -p.a / (1 - p.k2))]:
            edges.append(dict(source_id=ids[pre], target_id=ids[8 + i], channel=channel,
                effective_weight=weight, sign_evidence="assumed",
                sign_detail="Janke compartment hypothesis; logical column coupling, not MaleCNS contact weights"))
    network = RateNetwork(neurons, pd.DataFrame(edges), rate_unit="arbitrary_rate_units")
    ticks = np.arange((len(values) - 1) * steps + 1)
    sampled = values[np.minimum(ticks // steps, len(values) - 1)]
    sampled_mean = sampled.mean(axis=1)
    reset_ticks = np.flatnonzero(resets) * steps
    mean = np.zeros(len(ticks))
    # Exact continuous embedding of the explicit discrete baseline recurrence.
    retention = np.exp(np.log(p.k1) / steps)
    increment = p.c * (-np.expm1(np.log(p.k1) / steps)) / (1 - p.k1)
    reset_set = set(reset_ticks)
    for n in range(1, len(ticks)):
        mean[n] = 0. if n in reset_set else retention * mean[n - 1] + increment
    drive = np.zeros((len(ticks), 16))
    drive[:, 8:] = mean[:, None]
    slow_drive = np.zeros_like(drive)
    if input_mode == "centered":
        drive[:, 8:] -= p.b * sampled_mean[:, None]
        slow_drive[:, 8:] = p.a / (1 - p.k2) * sampled_mean[:, None]
    rates, offsets = np.zeros_like(drive), np.zeros_like(drive)
    # Restart only at declared reset events; all other segment changes carry state.
    boundaries = sorted({0, len(ticks) - 1, *reset_ticks})
    state = None
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if start in reset_set:
            state = RateState(network.neuron_ids, np.zeros(16), np.zeros(16))
        trace = network.solve(drive[start:stop + 1], dt_s,
            prescribed={ids[i]: sampled[start:stop + 1, i] for i in range(8)},
            slow_drive=slow_drive[start:stop + 1], initial=state, method="exponential_euler")
        rates[start:stop + 1], offsets[start:stop + 1] = trace.rates, trace.excitability
        state = trace.final_state
    # A reset at the terminal sample also overrides the just-computed state.
    for tick in reset_ticks:
        rates[tick, 8:] = offsets[tick, 8:] = 0.
    native = ticks[::steps]
    selected_input = values - values.mean(axis=1, keepdims=True) if input_mode == "centered" else values
    previous = np.vstack((np.zeros(8), selected_input[:-1]))
    proxy = mean[native, None] + offsets[native, 8:] + p.b * previous
    proxy[resets] = 0.
    return dict(time_s=ticks * dt_s, rates=rates[:, 8:], excitability=offsets[:, 8:],
                mean=mean, native_indices=native, latent_proxy=proxy,
                fluorescence_proxy=proxy,  # Retained API alias; not calibrated fluorescence in either mode.
                neurons=network.neurons, coupling=network.coupling,
                assumptions=dict(rate_tau_s=rate_tau_s, rate_unit=network.rate_unit, input_mode=input_mode,
                    baseline="prescribed continuous embedding; generator not explained",
                    input="prescribed column signals, no movement encoding or MaleCNS body mapping",
                    centering=("prescribed instantaneous across-column mean correction" if input_mode == "centered"
                               else "none; absolute prescribed rates drive both coupling channels"),
                    coupling="compartment sign hypothesis; no feedback; no learning",
                    proxy="signed latent proxy is not a calibrated firing-rate observation model"))


def bump_readout(values):
    """First-harmonic modulation and phase; zero-amplitude phases are undefined."""
    values = _finite(values, "Column readout")
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValidationError("Readout requires eight columns.")
    phasor = (values - values.mean(axis=1, keepdims=True)) @ np.exp(1j * np.arange(8) * 2 * np.pi / 8) * 2 / 8
    amplitude = np.abs(phasor)
    phase = np.where(amplitude > 1e-12, np.angle(phasor), np.nan)
    return pd.DataFrame(dict(mean=values.mean(axis=1), peak=values.max(axis=1),
        trough=values.min(axis=1), first_harmonic_amplitude=amplitude, phase_rad=phase))
