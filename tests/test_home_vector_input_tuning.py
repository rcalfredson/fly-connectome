"""Initial PFN mixtures must account for cancellation and rectification."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from fly_connectome.home_vector.analysis.input_tuning import first_step, main


def test_signed_inputs_cancel_before_relu_and_time_constant_sets_increment():
    pfn = np.array([[2., 3.], [5., 5.]])
    weight = np.array([[1., 0.], [-1., 0.]])
    bias, tau = np.array([.5, 1.]), np.array([.1, .2])
    drive, z, rates = first_step(pfn, weight, bias, tau, .0025)
    np.testing.assert_array_equal(drive, [[-1., 0.], [0., 0.]])
    np.testing.assert_array_equal(z, [[-.5, 1.], [.5, 1.]])
    alpha = 1 - np.exp(-.0025 / tau)
    np.testing.assert_allclose(rates, [[0., alpha[1]], [.5 * alpha[0], alpha[1]]])


def test_all_neuron_atlas_reconstructs_onset_and_preserves_inputs(tmp_path):
    from test_home_vector_checkpoint import fixtures
    from fly_connectome.home_vector.analysis.checkpoint import capture
    from fly_connectome.home_vector.__main__ import batch, begin, finish
    from fly_connectome.home_vector.data import Circuit, digest, verify_bundle, write_json
    args = fixtures(tmp_path)
    snap = capture(Path(args.run), Path(args.source))
    circuit = Circuit(**snap['circuit'])
    onset, record = begin(tmp_path / 'onset', 'home_vector_onset_probe_v1')
    conditions = []
    for heading in (0, 180):
        for forward, direction in [(1., 0), (-1., 180)]:
            conditions.append(dict(kind='moving', speed_mm_s=1., heading_deg=heading,
                body_direction_deg=direction, protocol=[[0, heading, forward, 0], [1, heading, 0, 0]]))
        conditions.append(dict(kind='stationary', speed_mm_s=0., heading_deg=heading,
            body_direction_deg=None, protocol=[[0, heading, 0, 0], [1, heading, 0, 0]]))
    with torch.no_grad():
        x, _, _ = batch(circuit, conditions, .0025, 'cpu', snap['network'].a.dtype)
        _, rates = snap['network'](x, .0025)
    np.savez(onset / 'onset_rates.npz', rates=rates.numpy(), time_s=np.arange(401) * .0025,
             input_at_onset=x[:, 0].numpy())
    (onset / 'current_checkpoint.pt').write_bytes(snap['blob'])
    write_json(onset / 'current_metadata.json', dict(circuit=snap['circuit'], run_manifest=snap['manifest'],
        checkpoint_sha256=snap['checkpoint_sha256']))
    write_json(onset / 'conditions.json', conditions)
    write_json(onset / 'report.json', dict(dt_s=.0025, checkpoint_update=7, checkpoint_sha256=snap['checkpoint_sha256']))
    finish(onset, record)
    before = {p: digest(p) for p in onset.iterdir() if p.is_file()}
    output = tmp_path / 'tuning'
    report = main(['--onset', str(onset), '--output', str(output)])
    assert verify_bundle(output)['status'] == 'complete'
    assert report['first_step_reconstruction_max_abs_error'] < 1e-6
    assert report['populations']['hDeltaG']['with_direct_pfn'] == 0
    assert report['populations']['hDeltaG']['initially_modulated'] == 0
    assert report['populations']['hDeltaB']['initially_modulated'] == 1
    assert len(report['plots']) == 27
    assert all(digest(p) == sha for p, sha in before.items())
    assert (output / 'input_connections.csv').stat().st_size > 0
    assert (output / 'neuron_input_summary.csv').stat().st_size > 0
    assert 'hDeltaG' in (output / 'index.html').read_text()
    for plot in report['plots']:
        assert (output / (plot['stem'] + '.png')).stat().st_size > 0
