from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from fly_connectome.connectivity import ValidationError
from fly_connectome.rate_framework import RateNetwork, RateState, coupling_from_counts
from fly_connectome.janke_reference import (JankeParameters, discrete_reference,
    absolute_input_control, rate_candidate, bump_readout)
from fly_connectome.trajectory import solve_continuous


def nodes(ids=("a", "b"), tau=.1, slow=0.):
    return pd.DataFrame(dict(neuron_id=list(ids), cell_type=list(ids), tau_s=tau, slow_tau_s=slow))


def edges(rows=()):
    return pd.DataFrame([dict(source_id=a, target_id=b, effective_weight=w, channel=channel,
        sign_evidence="assumed", sign_detail="Synthetic test coupling") for a, b, w, channel in rows],
        columns=["source_id", "target_id", "effective_weight", "channel", "sign_evidence", "sign_detail"])


def test_single_stage_matches_legacy_and_exact_held_drive():
    drive = np.zeros((1001, 2))
    drive[100:400] = [1., -2.]
    drive[400:700] = [3., 2.]
    initial = RateState(("a", "b"), np.array([.3, .4]), np.zeros(2))
    model = RateNetwork(nodes(), edges(), rate_unit="arbitrary_rate_units")
    euler = model.solve(drive, .001, initial=initial)
    legacy, exact = solve_continuous(drive, .001, .1, initial.rates)
    np.testing.assert_array_equal(euler.rates, legacy)
    exponential = model.solve(drive, .001, initial=initial, method="exponential_euler")
    np.testing.assert_array_equal(exponential.rates, exact)


def test_feedback_is_simultaneous_and_not_truncated_by_hops():
    model = RateNetwork(nodes(), edges([("a", "b", .4, "rate"), ("b", "a", .4, "rate")]), rate_unit="Hz")
    drive = np.tile([1., 0.], (5001, 1))
    trace = model.solve(drive, .001)
    # dr/dt=0: a=1+.4b and b=.4a. Paths of every length contribute.
    np.testing.assert_allclose(trace.rates[-1], [1 / .84, .4 / .84], rtol=1e-10)
    assert trace.rates[1, 1] == 0  # b sees the preceding a, not its just-updated value.
    permuted = RateNetwork(nodes(("b", "a")), model.coupling, rate_unit="Hz")
    reverse = permuted.solve(drive[:, ::-1], .001)
    np.testing.assert_array_equal(trace.rates, reverse.rates[:, ::-1])


def test_slow_offset_modulates_without_learning_and_can_be_negative():
    model = RateNetwork(nodes(slow=1.), edges(), rate_unit="Hz")
    drive = np.ones((5001, 2))
    slow = np.tile([1., -2.], (5001, 1))
    trace = model.solve(drive, .001, slow_drive=slow, method="exponential_euler")
    np.testing.assert_allclose(trace.excitability[-1], np.array([1., -2.]) * (1 - np.exp(-5)), atol=1e-12)
    assert trace.rates[-1, 0] > 1.9 and trace.rates[-1, 1] < .01
    assert np.all(trace.rates >= 0)
    assert model.coupling.empty


def test_inhibition_zero_drive_and_prescribed_boundary():
    model = RateNetwork(nodes(), edges([("a", "b", -1., "rate")]), rate_unit="Hz")
    drive = np.tile([0., 2.], (1001, 1))
    trace = model.solve(drive, .001, prescribed={"a": np.ones(1001)})
    np.testing.assert_allclose(trace.rates[-1], [1., 1.], atol=5e-5)
    with pytest.raises(ValidationError, match="cannot receive"):
        model.solve(drive, .001, prescribed={"b": np.ones(1001)})
    quiet = model.solve(np.zeros_like(drive), .001)
    np.testing.assert_array_equal(quiet.rates, 0)


def test_restart_and_record_stride_preserve_continuous_state():
    model = RateNetwork(nodes(slow=1.), edges([("a", "b", -.3, "excitability")]), rate_unit="Hz")
    drive = np.tile([1., 2.], (1002, 1))
    full = model.solve(drive, .001)
    first = model.solve(drive[:401], .001)
    second = model.solve(drive[400:], .001, initial=first.final_state)
    np.testing.assert_array_equal(np.vstack((first.rates[:-1], second.rates)), full.rates)
    np.testing.assert_array_equal(second.final_state.excitability, full.final_state.excitability)
    sparse = model.solve(drive, .001, record_every=30)
    ticks = np.unique(np.r_[np.arange(0, 1002, 30), 1001])
    np.testing.assert_array_equal(sparse.rates, full.rates[ticks])
    np.testing.assert_array_equal(sparse.final_state.rates, full.final_state.rates)
    wrong = RateState(("b", "a"), np.zeros(2), np.zeros(2))
    with pytest.raises(ValidationError, match="identities"):
        model.solve(drive, .001, initial=wrong)


def test_contact_scaling_normalization_signs_and_missing_rules():
    neurons = nodes(("a1", "a2", "b", "disconnected"))
    neurons["cell_type"] = ["A", "A", "B", "B"]
    neurons["input_group"] = ["A-L", "A-L", "B", "B"]
    contacts = pd.DataFrame(dict(source_id=["a1", "a2"], target_id=["b", "b"], contacts=[2, 6]))
    rules = pd.DataFrame([dict(source_type="A", target_type="B", channel="rate", sign=-1,
        strength=.5, sign_evidence="inferred", sign_detail="Synthetic receptor-based hypothesis")])
    raw = coupling_from_counts(neurons, contacts, rules)
    np.testing.assert_array_equal(raw.effective_weight, [-1, -3])
    normalized = coupling_from_counts(neurons, contacts, rules, normalize_incoming=True)
    np.testing.assert_array_equal(normalized.effective_weight, [-.125, -.375])
    np.testing.assert_array_equal(normalized.contacts, [2, 6])
    np.testing.assert_array_equal(normalized.normalization_denominator, [8, 8])
    assert len(RateNetwork(neurons, normalized, rate_unit="Hz").neuron_ids) == 4
    with pytest.raises(ValidationError, match="Every contact"):
        coupling_from_counts(neurons, contacts, rules.assign(target_type="C"))
    with pytest.raises(ValidationError, match="explicit sign"):
        coupling_from_counts(neurons, contacts, rules.assign(sign=np.nan))
    with pytest.raises(ValidationError, match="integers"):
        coupling_from_counts(neurons, contacts.assign(contacts=[1.5, 2]), rules)


def test_validation_and_divergence_are_not_silent_saturation():
    with pytest.raises(ValidationError, match="slow_tau"):
        RateNetwork(nodes(), edges([("a", "b", 1., "excitability")]), rate_unit="Hz")
    with pytest.raises(ValidationError, match="outside"):
        RateNetwork(nodes(), edges([("missing", "b", 1., "rate")]), rate_unit="Hz")
    model = RateNetwork(nodes(), edges([("a", "a", 10., "rate")]), rate_unit="Hz")
    with pytest.raises(ValidationError, match="guard"):
        model.solve(np.ones((1001, 2)), .001, max_rate=100)
    with pytest.raises(ValidationError, match="tau/20"):
        model.solve(np.ones((3, 2)), .1)
    with pytest.raises(ValidationError, match="finite"):
        model.solve(np.full((3, 2), np.nan), .001)
    with pytest.raises(ValidationError, match="rate_unit"):
        RateNetwork(nodes(), edges(), rate_unit="fluorescence")


def test_janke_reference_hand_update_mean_centering_reset_and_leak():
    p = JankeParameters(k1=.9, c=.1, k2=.8, a=.2, b=.5)
    values = np.ones((5, 8))
    values[0, 0] = 2
    ref = discrete_reference(values, p)
    u = np.array([.875] + [-.125] * 7)
    np.testing.assert_allclose(ref["integrated"][1], -.2 * np.roll(u, 4))
    np.testing.assert_allclose(ref["fluorescence"][1], .1 - .2 * np.roll(u, 4) + .5 * u)
    np.testing.assert_allclose(ref["integrated"][2], .8 * ref["integrated"][1])
    shifted = discrete_reference(values + 5, p)
    np.testing.assert_array_equal(ref["fluorescence"], shifted["fluorescence"])
    reset = discrete_reference(values, p, resets=np.array([False, False, True, False, False]))
    np.testing.assert_array_equal(reset["fluorescence"][2], 0)
    assert reset["mean"][3] == .1
    np.testing.assert_array_equal(reset["integrated"][3], 0)


def test_rate_candidate_embeds_reference_but_does_not_relabel_fluorescence():
    values = np.ones((8, 8))
    values[1:4] += .1 * np.cos(np.arange(8) * np.pi / 4)
    resets = np.zeros(8, dtype=bool)
    resets[[0, 5, 7]] = True
    p = JankeParameters()
    ref = discrete_reference(values, p, resets=resets)
    candidate = rate_candidate(values, p, dt_s=.01, resets=resets)
    fine = rate_candidate(values, p, dt_s=.005, resets=resets)
    np.testing.assert_allclose(candidate["fluorescence_proxy"], ref["fluorescence"], atol=1e-13)
    np.testing.assert_allclose(candidate["excitability"][::100], ref["integrated"], atol=1e-13)
    assert (candidate["rates"] >= 0).all()
    assert (ref["fluorescence"] < 0).any()
    assert not np.allclose(candidate["rates"][::100], ref["fluorescence"])
    assert np.max(np.abs(candidate["rates"] - fine["rates"][::2])) < 1e-3
    np.testing.assert_array_equal(candidate["rates"][[0, 500, 700]], 0)
    with pytest.raises(ValidationError, match="divide"):
        rate_candidate(values, p, dt_s=.03)
    with pytest.raises(ValidationError, match="embedding"):
        rate_candidate(values, replace(p, k2=1))


def test_bump_amplitude_is_not_peak_and_flat_phase_is_undefined():
    theta = np.arange(8) * np.pi / 4
    readout = bump_readout(np.array([2 + .3 * np.cos(theta - np.pi / 2), np.ones(8)]))
    assert readout.first_harmonic_amplitude[0] == pytest.approx(.3)
    assert readout.phase_rad[0] == pytest.approx(np.pi / 2)
    assert readout.peak[0] == pytest.approx(2.3)
    assert np.isnan(readout.phase_rad[1])


def test_absolute_mode_retains_common_input_and_matches_its_own_control():
    p = JankeParameters()
    values = np.ones((4, 8))
    control = absolute_input_control(values, p)
    np.testing.assert_allclose(control["integrated"][1], -.07)
    np.testing.assert_allclose(control["latent_output"][1], .001 - .07 + 1.)
    np.testing.assert_allclose(control["integrated"][2], -.8 * .07 - .07)
    candidate = rate_candidate(values, p, dt_s=.01, input_mode="absolute")
    np.testing.assert_allclose(candidate["latent_proxy"], control["latent_output"], atol=1e-13)
    assert np.max(np.abs(candidate["latent_proxy"] - discrete_reference(values, p)["fluorescence"])) > .5
    assert candidate["assumptions"]["input_mode"] == "absolute"
    assert candidate["excitability"][-1].mean() < 0
    resets = np.array([True, False, True, True])
    reset = rate_candidate(values, p, dt_s=.01, input_mode="absolute", resets=resets)
    np.testing.assert_allclose(reset["latent_proxy"], absolute_input_control(values, p, resets=resets)["latent_output"], atol=1e-13)
    with pytest.raises(ValidationError, match="input_mode"):
        rate_candidate(values, p, input_mode="unknown")


def test_uniform_pulse_cancels_only_in_centered_mode_with_matched_controls():
    p = JankeParameters()
    tonic = np.ones((21, 8))
    pulse = tonic.copy()
    pulse[10:15] += .05
    for mode in ["centered", "absolute"]:
        control = rate_candidate(tonic, p, dt_s=.01, input_mode=mode)
        perturbed = rate_candidate(pulse, p, dt_s=.01, input_mode=mode)
        pd.testing.assert_frame_equal(control["coupling"], perturbed["coupling"])
        difference = perturbed["rates"] - control["rates"]
        if mode == "centered":
            np.testing.assert_allclose(difference, 0., atol=1e-13)
        else:
            assert difference[1100].mean() > 0
            assert difference[1600].mean() < 0
        readout = bump_readout(perturbed["rates"])
        np.testing.assert_allclose(readout.first_harmonic_amplitude, 0., atol=1e-13)
