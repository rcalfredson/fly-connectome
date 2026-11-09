"""Causal controls and decoder-space diagnostics need independent counterexamples."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from fly_connectome.home_vector.analysis.activity_checks import (
    decoder_spaces, perturb_states, sweep_records, assess)
from fly_connectome.home_vector.model import HomeVectorNetwork
from test_home_vector_data import circuit_fixture


def test_null_motion_is_hidden_but_visible_motion_changes_output():
    decoder = np.array([[1., 0., 0.], [0., 1., 0.]])
    rates = np.array([[1., 1., 0.], [1., 1., 3.], [2., 1., 3.]])
    result = decoder_spaces(rates, decoder)
    assert result['rank'] == 2
    np.testing.assert_allclose(result['hidden_change_norm'], [0, 3, 3])
    np.testing.assert_allclose(result['visible_change_norm'], [0, 0, 1])
    zero = decoder_spaces(rates, np.zeros((2, 3)))
    assert zero['rank'] == 0
    np.testing.assert_array_equal(zero['visible_change_norm'], 0.)


def test_clamp_stops_recurrent_propagation_and_sham_matches_default():
    model = HomeVectorNetwork(circuit_fixture())
    with torch.no_grad():
        model.bias.zero_()
        model.a.fill_(1.)
        model.D.copy_(torch.tensor([[1., 1., 1.], [0., 1., 0.]]))
        x = torch.zeros(2, 10, 2)
        initial = torch.tensor([[1., 0., 0.], [1., 0., 0.]])
        normal, normal_rates = model(x, .0025, initial=initial)
        mask = torch.tensor([[False, False, False], [True, False, False]])
        prediction, rates = model(x, .0025, initial=initial, silence_mask=mask)
    torch.testing.assert_close(prediction[0], normal[0], rtol=0, atol=0)
    torch.testing.assert_close(rates[0], normal_rates[0], rtol=0, atol=0)
    assert normal_rates[0, 1, 1] > 0  # First unit's old state excites second.
    torch.testing.assert_close(rates[1], torch.zeros_like(rates[1]), rtol=0, atol=0)
    # Merely deleting the first unit from readout leaves that second-unit signal.
    direct_removed = normal - normal_rates[:, :, 0, None] * model.D[:, 0]
    assert direct_removed[0, 1, 0] > 0
    with pytest.raises(ValueError, match='no_grad'):
        model(x, .0025, silence_mask=mask)
    with torch.no_grad(), pytest.raises(ValueError, match='mask'):
        model(x, .0025, silence_mask=torch.zeros(4, dtype=torch.bool))


def test_sweep_matches_distance_across_speeds_and_covers_lateral_cases():
    from fly_connectome.home_vector.data import sample_protocol
    rows = sweep_records(.0025, 2., [.5, 2.], 4)
    assert len(rows) == 32
    for row in rows:
        _, target = sample_protocol(row['protocol'], .0025)
        np.testing.assert_allclose(np.linalg.norm(target[-1]), 2., atol=1e-10)
    assert any(r['body_direction_deg'] == 90 for r in rows)
    with pytest.raises(ValueError, match='grid'):
        sweep_records(.0025, 1., [.7], 4)


def test_jitter_is_reproducible_nonnegative_and_preserves_sham_and_reset():
    state = np.array([0., 1., 3.], dtype=np.float32)
    a, metadata = perturb_states(state, [.001, .01], 4, 9)
    b, _ = perturb_states(state, [.001, .01], 4, 9)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a[0], state)
    np.testing.assert_array_equal(a[-1], 0)
    assert (a >= 0).all()
    for row in metadata[1:-1]:
        assert row['actual_relative_state_change'] == pytest.approx(row['fraction'], rel=1e-4)
    score = assess(np.zeros((3, 2)), np.zeros((3, 2)), np.array([[0.], [2e6], [3e6]]), .1)
    assert score['status'] == 'numerical_guard_failed' and score['first_invalid_s'] == .1
    assert score['vector_rmse_mm'] is None


def test_small_complete_diagnostics_bundle(tmp_path):
    from test_home_vector_checkpoint import fixtures
    from fly_connectome.home_vector.analysis import activity, activity_checks
    from fly_connectome.home_vector.data import verify_bundle, digest
    args = fixtures(tmp_path)
    activity.analyze(args)
    atlas = Path(args.output)
    before = {p: digest(p) for p in atlas.iterdir() if p.is_file()}
    output = tmp_path / 'checks'
    result = activity_checks.main(['--activity', str(atlas), '--source', args.source,
        '--output', str(output), '--angles', '2', '--speeds', '1', '--distance-mm', '.01',
        '--pause-seconds', '.05', '--jitter-fractions', '.01', '--repeats', '2'])
    assert result['sweep_conditions'] == 4
    assert result['neuron_interventions'] == 12 * 3 * 2
    assert result['pause_variants'] == 8
    assert verify_bundle(output)['status'] == 'complete'
    assert all(digest(p) == sha for p, sha in before.items())
    for name in ('contribution_sweep', 'neuron_interventions', 'pause_stability'):
        for extension in ('png', 'pdf'):
            assert (output / f'{name}.{extension}').stat().st_size > 0
    report = json.loads((output / 'pause_details.json').read_text())
    assert len(report) == 2
    with np.load(output / f'{report[0]["trajectory"]}_pause.npz') as arrays:
        assert arrays['variant_prediction'].shape == (4, 21, 2)
