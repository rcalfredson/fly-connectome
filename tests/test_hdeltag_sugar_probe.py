"""Causal cue, population specificity, numerical update, and matched controls."""

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from fly_connectome.home_vector.data import Circuit, sample_protocol
from fly_connectome.home_vector.model import HomeVectorNetwork
from fly_connectome.home_vector.analysis.sugar_response import (
    SugarResponse, event_gate, relaxation_step, rollout,
)
from fly_connectome.home_vector.analysis.sugar_probe import insert_pause


def response(**overrides):
    return SugarResponse(**(dict(rise_tau_s=.1, recovery_tau_s=10.,
                                relaxation_rate_per_s=2., reference_activity=0.) | overrides))


def network(sign=1.):
    c = Circuit([1], ['PFNd'], [2, 3], ['hDeltaG', 'hDeltaB'], [0.], [0.], [
        dict(pre=0, post=0, contacts=1, denominator=1., source_type='PFNd', target_type='hDeltaG'),
        dict(pre=1, post=1, contacts=1, denominator=1., source_type='hDeltaG', target_type='hDeltaB'),
        dict(pre=2, post=0, contacts=1, denominator=1., source_type='hDeltaB', target_type='hDeltaG')])
    n = HomeVectorNetwork(c)
    with torch.no_grad():
        n.a.copy_(torch.tensor([1., sign*.5, .1]))
        n.bias.fill_(1.)
        n.D.copy_(torch.tensor([[1., -.5], [.2, .7]]))
        n.d.copy_(torch.tensor([.1, -.1]))
    return n


def test_event_kernel_is_causal_bounded_and_does_not_stack_above_one():
    t = np.arange(0., 3., .01)
    one = event_gate(t, [1.], response())
    many = event_gate(t, [1., 1.5], response())
    np.testing.assert_array_equal(one[t <= 1.], 0.)
    assert 0 < one.max() < 1.
    np.testing.assert_array_equal(many, np.maximum(one, event_gate(t, [1.5], response())))
    np.testing.assert_array_equal(event_gate(t, [], response()), 0.)


def test_extra_relaxation_matches_analytic_decay_and_affects_only_targeted_cells():
    r = torch.tensor([[10., 10.]], dtype=torch.float64)
    target, tau = torch.zeros_like(r), torch.tensor([2., 2.], dtype=torch.float64)
    out = relaxation_step(r, target, tau, .1, torch.tensor([True, False]), .5,
                          response(relaxation_rate_per_s=3.))
    np.testing.assert_allclose(out.numpy(), [[10*np.exp(-.2), 10*np.exp(-.05)]], atol=1e-14)
    # Stable convex update even at strong inhibition; no hard zeroing or overshoot.
    out = relaxation_step(r, target, tau, .1, torch.tensor([True, False]), 1.,
                          response(relaxation_rate_per_s=1e5, reference_activity=2.))
    assert 0 < out[0, 0] < 2.01


def test_disabled_gate_is_bitwise_identical_to_native_and_does_not_change_parameters():
    n = network(); pfn = torch.ones(1, 101, 1); saved = {k: v.clone() for k, v in n.state_dict().items()}
    with torch.no_grad():
        expected, rates = n(pfn, .0025)
        actual, observed = rollout(n, pfn, .0025, np.zeros(101), response(), torch.tensor([True, False]))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(observed, rates, atol=0, rtol=0)
    for k, v in n.state_dict().items():
        torch.testing.assert_close(v, saved[k], atol=0, rtol=0)


@pytest.mark.parametrize('sign', [1., -1.])
def test_untargeted_neuron_can_decrease_or_increase_through_recurrence(sign):
    n = network(sign); pfn = torch.ones(1, 101, 1); mask = torch.tensor([True, False])
    with torch.no_grad():
        _, control = rollout(n, pfn, .0025, np.zeros(101), response(), mask)
        _, sugar = rollout(n, pfn, .0025, np.ones(101), response(), mask)
    assert sugar[0, 1, 0] < control[0, 1, 0]
    assert sugar[0, 1, 1] == control[0, 1, 1]  # No direct change to hDeltaB.
    if sign > 0:
        assert sugar[0, 2, 1] < control[0, 2, 1]
    else:
        assert sugar[0, 2, 1] > control[0, 2, 1]


def test_no_effect_before_event_and_runtime_requires_explicit_no_grad():
    n = network(); pfn = torch.ones(1, 101, 1); mask = torch.tensor([True, False])
    gate = event_gate(np.arange(101)*.0025, [.1], response())
    with pytest.raises(ValueError, match='no_grad'):
        rollout(n, pfn, .0025, gate, response(), mask)
    with torch.no_grad():
        _, control = n(pfn, .0025)
        _, sugar = rollout(n, pfn, .0025, gate, response(), mask)
    torch.testing.assert_close(sugar[:, :41], control[:, :41], atol=0, rtol=0)
    assert not torch.equal(sugar[:, -1], control[:, -1])


def test_pause_retains_movement_and_displacement_without_inserting_a_reset():
    p = [[0., 0., 1., 0.], [1., 90., 1., 0.], [2., 90., 0., 0.]]
    _, original = sample_protocol(p, .0025)
    paused = insert_pause(p, 1., .5, .0025)
    c, y = sample_protocol(paused, .0025)
    np.testing.assert_allclose(y[-1], original[-1])
    np.testing.assert_array_equal(c[400:600, 1:], 0.)
    np.testing.assert_allclose(y[400:601], np.repeat(original[400:401], 201, axis=0))
    # Also support a pause within a held movement segment.
    _, y2 = sample_protocol(insert_pause(p, .5, .5, .0025), .0025)
    np.testing.assert_allclose(y2[-1], original[-1])


@pytest.mark.parametrize('overrides', [dict(rise_tau_s=0), dict(recovery_tau_s=np.nan),
                                      dict(reference_activity=-1), dict(relaxation_rate_per_s=-1)])
def test_invalid_response_parameters_are_rejected(overrides):
    with pytest.raises(ValueError):
        response(**overrides)
