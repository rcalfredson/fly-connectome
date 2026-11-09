"""Signed incoming-drive accounting must reconstruct rates and separate state terms."""

import csv
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from fly_connectome.home_vector.analysis.incoming_drive import decompose, main, verify_updates
from fly_connectome.home_vector.data import digest, verify_bundle
from fly_connectome.home_vector.model import HomeVectorNetwork
from test_home_vector_data import circuit_fixture


def test_drive_sums_and_own_state_term_reconstruct_production_updates():
    circuit = circuit_fixture()
    network = HomeVectorNetwork(circuit)
    with torch.no_grad():
        network.a.copy_(torch.linspace(-1., 1., len(network.a)))
    x = torch.linspace(.1, 2., 2 * 8 * network.input_size).reshape(2, 8, network.input_size)
    initial = torch.full((2, network.size), 3.)
    with torch.no_grad():
        _, r = network(x, .0025, initial=initial)
    targets = list(range(network.size))
    parts, wr = decompose(network, x.numpy(), r.numpy(), targets, .0025)
    assert verify_updates(dict(rates=r.numpy()), parts, targets) < 4e-5
    np.testing.assert_allclose(parts['recurrent_by_population'].sum(axis=-2), parts['recurrent_drive'], atol=1e-12)
    contributions = r.numpy().astype(float)[..., :, None] * wr
    np.testing.assert_allclose(contributions.sum(axis=-2), parts['recurrent_drive'], atol=1e-12)
    np.testing.assert_allclose(parts['target_increment'] + parts['own_state_increment'], parts['increment'], atol=1e-12)
    assert np.any(parts['preactivation'] < 0)
    np.testing.assert_array_equal(parts['rectified_target'][parts['preactivation'] < 0], 0)


def test_same_incoming_drive_can_coexist_with_different_rates_and_relaxation():
    circuit = circuit_fixture()
    network = HomeVectorNetwork(circuit, recurrence=False)
    pfn = np.ones((2, network.input_size))
    rates = np.stack([np.zeros(network.size), np.full(network.size, 10.)])
    parts, _ = decompose(network, pfn, rates, list(range(network.size)), .0025)
    np.testing.assert_array_equal(parts['input_drive'][0], parts['input_drive'][1])
    np.testing.assert_array_equal(parts['recurrent_drive'], 0)
    np.testing.assert_array_equal(parts['rectified_target'][0], parts['rectified_target'][1])
    assert np.all(parts['increment'][0] > parts['increment'][1])


def test_complete_audit_from_renamed_sources_preserves_inputs_and_exports_every_partner(tmp_path):
    from test_home_vector_trajectory_tuning import archive
    from fly_connectome.home_vector.analysis.trajectory_tuning import main as probe
    args, activity, cases = archive(tmp_path)
    probes = tmp_path / 'probes'
    probe(['--activity', str(activity), '--source', args.source, '--output', str(probes), '--cases', str(cases),
           '--angles', '2', '--times', '.0025', '.01'])
    renamed = tmp_path / 'renamed-trajectories'
    activity.rename(renamed)
    frozen = {p: digest(p) for p in tmp_path.rglob('*') if p.is_file()}
    output = tmp_path / 'drive'
    report = main(['--activity', str(renamed), '--probes', str(probes), '--source', args.source, '--output', str(output)])
    assert report['checkpoint_update'] == 7 and verify_bundle(output)['status'] == 'complete'
    assert all(digest(p) == sha for p, sha in frozen.items())
    assert max(r['maximum_next_state_reconstruction_error'] for r in report['reconstruction_checks']) < 4e-5
    with (output / 'identical_probe_first_updates.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert all(float(row['input_drive_difference']) == 0 for row in rows)
    for row in rows:
        assert float(row['increment_difference']) == pytest.approx(
            float(row['target_increment_difference']) + float(row['own_state_increment_difference']), abs=1e-12)
    for plot in report['plots']:
        for extension in ('png', 'pdf'):
            assert (output / f'{plot["stem"]}.{extension}').stat().st_size > 0
    assert (output / 'report.md').exists()
