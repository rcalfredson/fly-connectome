"""History probes must preserve memory, isolate trials and match heading controls."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from fly_connectome.home_vector.__main__ import batch
from fly_connectome.home_vector.analysis.history_tuning import (
    compare_maps, histories, main, map_stats, probe_from_state, response_views)
from fly_connectome.home_vector.analysis.activity_paths import resume_records
from fly_connectome.home_vector.data import digest, verify_bundle
from fly_connectome.home_vector.model import HomeVectorNetwork
from test_home_vector_data import circuit_fixture


def test_saved_state_probes_equal_uninterrupted_replay_and_do_not_contaminate_each_other():
    circuit = circuit_fixture()
    network = HomeVectorNetwork(circuit)
    dt = .0025
    history = histories(.05, 0., 1.)[1]
    records = resume_records(.01, 2, [1.], 0.)
    with torch.no_grad():
        x, _, _ = batch(circuit, [history], dt, 'cpu', network.a.dtype)
        _, r = network(x, dt)
    state = r[0, -1].numpy().copy()
    before = state.copy()
    _, probes = probe_from_state(network, circuit, records, state, dt, 3)
    np.testing.assert_array_equal(state, before)
    for k, record in enumerate(records):
        with torch.no_grad():
            probe_x, _, _ = batch(circuit, [record], dt, 'cpu', network.a.dtype)
            # Drop the endpoint input of the history: that input has not yet
            # been applied to its saved terminal state.
            _, uninterrupted = network(torch.cat([x[:, :-1], probe_x], dim=1), dt)
        np.testing.assert_allclose(probes[k], uninterrupted[0, x.shape[1] - 1:].numpy(), atol=2e-6, rtol=1e-6)
    _, reversed_probes = probe_from_state(network, circuit, list(reversed(records)), state, dt, 1)
    np.testing.assert_allclose(probes, reversed_probes[::-1], atol=2e-6, rtol=1e-6)


def test_stationary_subtraction_removes_passive_history_offset_without_recurrence():
    circuit = circuit_fixture()
    network = HomeVectorNetwork(circuit, recurrence=False)
    records = resume_records(.05, 4, [1.], 0.)
    outputs = []
    for state in [np.zeros(network.size, dtype=np.float32), np.full(network.size, 5., dtype=np.float32)]:
        _, rates = probe_from_state(network, circuit, records, state, .0025, 7)
        views, moving, controls = response_views(rates, state, records)
        for i, k in enumerate(moving):
            assert records[controls[i]]['kind'] == 'stationary'
            assert records[controls[i]]['heading_deg'] == records[k]['heading_deg']
        np.testing.assert_allclose(views['starting_delta'], views['raw'] - state)
        outputs.append(views)
    assert np.max(abs(outputs[0]['raw'] - outputs[1]['raw'])) > 1.
    np.testing.assert_allclose(outputs[0]['movement_delta'], outputs[1]['movement_delta'], atol=3e-6, rtol=1e-5)


def test_map_statistics_distinguish_offset_gain_shape_and_flat_or_tied_peaks():
    conditions = [dict(heading_deg=0., body_direction_deg=float(i * 90)) for i in range(4)]
    flat = map_stats(np.ones(4), conditions)
    assert flat['flat'] and flat['peak_heading_deg'] is None
    tied = map_stats([0., 2., 2., 0.], conditions)
    assert tied['peak_tied_points'] == 2
    assert compare_maps(np.ones(4), np.arange(4))['pattern_correlation'] is None
    shifted = compare_maps(np.arange(4), np.arange(4) + 10)
    assert shifted['mean_shift'] == 10 and shifted['centered_rms_difference'] == 0
    assert shifted['pattern_correlation'] == pytest.approx(1)
    assert compare_maps(np.arange(4), 2 * np.arange(4))['pattern_correlation'] == pytest.approx(1)
    assert compare_maps(np.arange(4), -np.arange(4))['pattern_correlation'] == pytest.approx(-1)


def test_complete_history_bundle_captures_every_neuron_and_preserves_sources(tmp_path):
    from test_home_vector_checkpoint import fixtures
    args = fixtures(tmp_path)
    circuit = circuit_fixture()
    frozen = {p: digest(p) for p in tmp_path.rglob('*') if p.is_file()}
    output = tmp_path / 'history'
    report = main(['--run', args.run, '--source', args.source, '--output', str(output),
        '--history-seconds', '.02', '--angles', '2', '--times', '.0025', '.01',
        '--focus', str(circuit.dynamic_ids[0]), '--chunk-size', '3'])
    assert report['checkpoint_update'] == 7
    assert report['moving_conditions'] == 4 and report['stationary_controls'] == 2
    assert verify_bundle(output)['status'] == 'complete'
    assert all(digest(p) == sha for p, sha in frozen.items())
    with np.load(output / 'probe_responses.npz') as saved:
        assert saved['rates'].shape == (3, 6, 5, len(circuit.dynamic_ids))
        assert saved['raw'].shape == (3, 4, 2, len(circuit.dynamic_ids))
        for h in range(3):
            np.testing.assert_array_equal(saved['rates'][h, :, 0],
                                          np.broadcast_to(saved['initial_states'][h], (6, len(circuit.dynamic_ids))))
    for plot in report['plots']:
        for extension in ('png', 'pdf'):
            assert (output / f'{plot["stem"]}.{extension}').stat().st_size > 0
    assert (output / 'index.html').exists()
    assert (output / 'history_comparisons.csv').exists()


def test_off_grid_sampling_fails_before_creating_output(tmp_path):
    from test_home_vector_checkpoint import fixtures
    args = fixtures(tmp_path)
    output = tmp_path / 'bad'
    with pytest.raises(ValueError, match='time grid'):
        main(['--run', args.run, '--source', args.source, '--output', str(output), '--times', '.003'])
    assert not output.exists()
