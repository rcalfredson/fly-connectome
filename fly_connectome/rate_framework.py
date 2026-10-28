"""Small recurrent rate networks with explicit coupling and optional slow states.

Arrays use the declared neuron order; edges always use identifiers. No graph
expansion, sign inference, normalization, or weight learning occurs in the solver.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .connectivity import ValidationError


def _finite(values, name):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValidationError(f"{name} must be finite.")
    return values


def _ids(values, name):
    if pd.isna(values).any():
        raise ValidationError(f"{name} must not be missing.")
    result = [str(v) for v in values]
    if any(not v.strip() for v in result) or len(set(result)) != len(result):
        raise ValidationError(f"{name} must be nonempty and unique.")
    return result


def coupling_from_counts(neurons, contacts, rules, *, normalize_incoming=False):
    """Counts × declared type-pair strength × declared sign, with an audit table.

    Optional normalization divides counts within each receiving neuron and
    explicit presynaptic input_group. It never normalizes a biological total.
    Multiple channels for a type pair require separately assigned contact rows;
    the same anatomical contacts are not automatically duplicated.
    """
    neurons, contacts, rules = neurons.copy(), contacts.copy(), rules.copy()
    neurons["neuron_id"] = _ids(neurons.neuron_id, "Neuron IDs")
    if type(normalize_incoming) is not bool:
        raise ValidationError("Normalization must be an explicit boolean.")
    for column in ["source_id", "target_id"]:
        if contacts[column].isna().any():
            raise ValidationError("Contact endpoints must not be missing.")
        contacts[column] = contacts[column].astype(str)
        if not contacts[column].isin(neurons.neuron_id).all():
            raise ValidationError("Contact endpoint outside the modeled circuit.")
    counts = _finite(contacts.contacts, "Contact counts")
    if (counts < 0).any() or (counts % 1 != 0).any():
        raise ValidationError("Contact counts must be nonnegative integers.")
    contacts["contacts"] = counts.astype(np.int64)
    if "channel" not in contacts:
        contacts["channel"] = "rate"
    keys = ["source_type", "target_type", "channel"]
    if rules.duplicated(keys).any() or contacts.duplicated(["source_id", "target_id", "channel"]).any():
        raise ValidationError("Duplicate rules or contact pairs within a channel.")
    if not rules.sign.isin([-1, 1]).all():
        raise ValidationError("Select an explicit sign (-1 or +1), including for uncertain edges.")
    strengths = _finite(rules.strength, "Per-contact strengths")
    if (strengths < 0).any():
        raise ValidationError("Strengths must be nonnegative; use sign for polarity.")
    if not rules.sign_evidence.isin(["measured", "inferred", "assumed"]).all():
        raise ValidationError("Sign evidence must be measured, inferred or assumed.")
    if rules.sign_detail.isna().any() or rules.sign_detail.astype(str).str.strip().eq("").any():
        raise ValidationError("Provide evidence or an assumption description for every sign.")
    types = neurons.set_index("neuron_id").cell_type
    contacts["source_type"] = contacts.source_id.map(types)
    contacts["target_type"] = contacts.target_id.map(types)
    result = contacts.merge(rules, on=keys, how="left", validate="many_to_one", indicator=True)
    if not result._merge.eq("both").all():
        raise ValidationError("Every contact needs a declared source/target/channel rule.")
    result = result.drop(columns="_merge")
    result["normalization"] = "none"
    result["normalization_denominator"] = 1.
    if normalize_incoming:
        if "input_group" not in neurons or neurons.input_group.isna().any():
            raise ValidationError("Incoming normalization requires explicit input_group labels.")
        result["input_group"] = result.source_id.map(neurons.set_index("neuron_id").input_group)
        result["normalization_denominator"] = result.groupby(
            ["target_id", "input_group", "channel"]).contacts.transform("sum")
        result["normalization"] = "within_selected_source_group_per_target"
    denominator = result.normalization_denominator.to_numpy(dtype=float)
    normalized = np.divide(result.contacts.to_numpy(dtype=float), denominator,
                           out=np.zeros(len(result)), where=denominator > 0)
    result["effective_weight"] = normalized * result.strength * result.sign
    return result


@dataclass
class RateState:
    neuron_ids: tuple
    rates: np.ndarray
    excitability: np.ndarray


@dataclass
class RateTrace:
    time_s: np.ndarray
    neuron_ids: tuple
    rates: np.ndarray
    excitability: np.ndarray
    final_state: RateState


class RateNetwork:
    """tau_r dr/dt = -r + gain * ReLU(bias + input + q - threshold).

    Optional slow offset: tau_q dq/dt = -q + slow_input. This is a declared
    phenomenological excitability state, not automatically synaptic plasticity.
    All recurrent inputs use the preceding rates (simultaneous updates).
    """

    def __init__(self, neurons, coupling, *, rate_unit):
        if rate_unit not in {"Hz", "arbitrary_rate_units"}:
            raise ValidationError("Declare rate_unit as Hz or arbitrary_rate_units.")
        self.rate_unit = rate_unit
        self.neurons = neurons.copy()
        self.neuron_ids = tuple(_ids(neurons.neuron_id, "Neuron IDs"))
        if not self.neuron_ids or neurons.cell_type.isna().any():
            raise ValidationError("Declare a nonempty circuit and cell types.")
        self.neurons["neuron_id"] = self.neuron_ids
        self.index = {name: i for i, name in enumerate(self.neuron_ids)}
        for name, default in [("tau_s", None), ("bias", 0.), ("threshold", 0.),
                              ("gain", 1.), ("slow_tau_s", 0.)]:
            if name not in self.neurons:
                if default is None:
                    raise ValidationError("Declare tau_s for every neuron.")
                self.neurons[name] = default
            setattr(self, name, _finite(self.neurons[name], name))
        if (self.tau_s <= 0).any() or (self.slow_tau_s < 0).any() or (self.gain < 0).any():
            raise ValidationError("Use positive rate tau and nonnegative slow tau/gain.")
        self.coupling = coupling.copy()
        if not coupling.channel.isin(["rate", "excitability"]).all():
            raise ValidationError("Supported coupling channels are rate and excitability.")
        for name in ["source_id", "target_id"]:
            if coupling[name].isna().any() or not coupling[name].astype(str).isin(self.neuron_ids).all():
                raise ValidationError("Coupling endpoint outside the modeled circuit.")
            self.coupling[name] = coupling[name].astype(str)
        self.weights = _finite(coupling.effective_weight, "Effective weights")
        if not coupling.sign_evidence.isin(["measured", "inferred", "assumed"]).all():
            raise ValidationError("Every coupling requires an explicit sign-evidence label.")
        if coupling.sign_detail.isna().any() or coupling.sign_detail.astype(str).str.strip().eq("").any():
            raise ValidationError("Every coupling requires a sign description.")
        self.pre = self.coupling.source_id.map(self.index).to_numpy(dtype=int)
        self.post = self.coupling.target_id.map(self.index).to_numpy(dtype=int)
        self.slow_edges = coupling.channel.eq("excitability").to_numpy()
        if (self.slow_tau_s[self.post[self.slow_edges]] <= 0).any():
            raise ValidationError("Excitability targets require a positive slow_tau_s.")

    def solve(self, drive, dt_s, *, prescribed=None, slow_drive=None, initial=None,
              method="euler", record_every=1, max_rate=1e6):
        """Held inputs on [t[n], t[n+1]); final sample has no following interval.

        exponential_euler integrates leak exactly with preceding network input
        held fixed; it is NOT an exact solution of the coupled recurrent system.
        record_every reduces trace storage, not integration steps. Final state
        can restart a run; prescribed sources are externally clamped and cannot
        receive coupling, so feedback into them is never silently discarded.
        """
        drive = _finite(drive, "External drive")
        size = len(self.neuron_ids)
        if drive.ndim != 2 or drive.shape[1] != size or len(drive) < 2:
            raise ValidationError("Drive must have at least two samples and one column per neuron.")
        if not np.isfinite(dt_s) or dt_s <= 0 or method not in {"euler", "exponential_euler"}:
            raise ValidationError("Use positive dt and a supported integration method.")
        if type(record_every) is not int or record_every < 1 or not np.isfinite(max_rate) or max_rate <= 0:
            raise ValidationError("Use positive integer record_every and finite positive max_rate.")
        slow_drive = np.zeros_like(drive) if slow_drive is None else _finite(slow_drive, "Slow drive")
        if slow_drive.shape != drive.shape or (slow_drive[:, self.slow_tau_s == 0] != 0).any():
            raise ValidationError("Slow drive must match drive and target enabled slow states.")
        prescribed = {} if prescribed is None else prescribed
        clamps = {}
        for key, values in prescribed.items():
            if str(key) not in self.index:
                raise ValidationError("Prescribed neuron outside the circuit.")
            i = self.index[str(key)]
            if i in clamps:
                raise ValidationError("Duplicate prescribed neuron.")
            values = _finite(values, "Prescribed rates")
            if values.shape != (len(drive),) or (values < 0).any() or (values > max_rate).any():
                raise ValidationError("Prescribed rates must be nonnegative and match the time grid/guard.")
            if i in self.post or self.slow_tau_s[i] > 0 or drive[:, i].any():
                raise ValidationError("Prescribed neurons cannot receive edges, drive or slow states.")
            clamps[i] = values
        dynamic = np.array([i not in clamps for i in range(size)])
        enabled = self.slow_tau_s > 0
        taus = np.r_[self.tau_s[dynamic], self.slow_tau_s[enabled]]
        if method == "euler" and len(taus) and dt_s > taus.min() / 20:
            raise ValidationError("Euler requires dt <= every dynamic tau/20; recurrence still needs validation.")
        if initial is None:
            rates, offset = np.zeros(size), np.zeros(size)
        else:
            if tuple(initial.neuron_ids) != self.neuron_ids:
                raise ValidationError("Restart state identities/order differ from this circuit.")
            rates = _finite(initial.rates, "Initial rates").copy()
            offset = _finite(initial.excitability, "Initial excitability").copy()
            if (rates.shape != (size,) or offset.shape != (size,) or (rates < 0).any()
                    or (offset[~enabled] != 0).any() or (rates > max_rate).any()):
                raise ValidationError("Invalid initial rate/excitability state.")
        for i, values in clamps.items():
            rates[i] = values[0]
        ticks = np.unique(np.r_[np.arange(0, len(drive), record_every), len(drive) - 1])
        out_rate, out_offset = np.empty((len(ticks), size)), np.empty((len(ticks), size))
        alpha = dt_s / self.tau_s
        beta = np.zeros(size)
        beta[enabled] = dt_s / self.slow_tau_s[enabled]
        if method == "exponential_euler":
            alpha = -np.expm1(-alpha)
            beta = -np.expm1(-beta)
        cursor = 0
        for n in range(len(drive)):
            if n == ticks[cursor]:
                out_rate[cursor], out_offset[cursor] = rates, offset
                cursor += 1
            if n == len(drive) - 1:
                break
            with np.errstate(over="raise", invalid="raise"):
                try:
                    weighted = self.weights * rates[self.pre]
                    fast = np.bincount(self.post[~self.slow_edges], weights=weighted[~self.slow_edges], minlength=size)
                    slow = np.bincount(self.post[self.slow_edges], weights=weighted[self.slow_edges], minlength=size)
                    target = self.gain * np.maximum(0., self.bias + drive[n] + fast + offset - self.threshold)
                    next_rates = rates + alpha * (target - rates)
                    offset = offset + beta * (-offset + slow_drive[n] + slow)
                    rates[dynamic] = next_rates[dynamic]
                except FloatingPointError as exc:
                    raise ValidationError("Nonfinite dynamics; inspect gains and recurrent stability.") from exc
            for i, values in clamps.items():
                rates[i] = values[n + 1]
            if not np.isfinite(rates).all() or not np.isfinite(offset).all() or (rates > max_rate).any():
                raise ValidationError("Dynamics exceeded numerical guard; this is not biological saturation.")
        return RateTrace(ticks * dt_s, self.neuron_ids, out_rate, out_offset,
                         RateState(self.neuron_ids, rates.copy(), offset.copy()))
