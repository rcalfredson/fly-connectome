from dataclasses import replace
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from fly_connectome import __main__ as cli
from fly_connectome import rate_model as f
from fly_connectome import recurrent_model as r
from fly_connectome.connectivity import DATASET, SERVER, ValidationError, build_matrix


def event(ids=(1,), start=0., stop=1.):
    return {"body_ids": list(ids), "start_s": start, "stop_s": stop, "amplitude": 1., "label": "pulse"}


def test_zero_recurrence_is_bitwise_feedforward():
    p = f.Parameters(duration_s=1.)
    _, drive = f.make_drive([1, 2], [event(ids=(1,))], p)
    weights = np.array([[.137, .1], [0., .4]])
    for old, new in zip(f.integrate(weights, drive, p), f.integrate(weights, drive, p, np.zeros((2, 2)))):
        np.testing.assert_array_equal(old, new)


def test_exact_reference_matches_scalar_recurrent_solution():
    p = f.Parameters(duration_s=1.)
    times, drive = f.make_drive([1], [event()], p)
    weights, recurrent = np.array([[.3]]), np.array([[.5]])
    exact = r.continuous_reference(drive, r.reference_operators(weights, recurrent, p))
    expected = 2 * (1 - np.exp(-.5 * times / p.tau_s))
    np.testing.assert_allclose(exact[0][:, 0], expected, rtol=1e-12, atol=1e-13)
    coarse = f.integrate(weights, drive, p, recurrent)
    fine_p = replace(p, dt_s=p.dt_s / 2)
    _, fine_drive = f.make_drive([1], [event()], fine_p)
    fine = f.integrate(weights, fine_drive, fine_p, recurrent)
    for c, refined, theory in zip(coarse, fine, exact):
        assert f.relative_error(c, theory) < .01
        assert f.relative_error(refined[::2], theory) < f.relative_error(c, theory)


def test_reference_matches_feedforward_closed_form_with_overlap():
    p = f.Parameters(duration_s=1.)
    events = [event(start=.1, stop=.3), event(start=.2, stop=.5)]
    times, drive = f.make_drive([1], events, p)
    weights = np.array([[.137, .5]])
    reference = r.continuous_reference(drive, r.reference_operators(weights, np.zeros((1, 1)), p))
    analytical = f.analytical_response(weights, [1], events, times, p.tau_s)
    for exact, closed in zip(reference, analytical):
        np.testing.assert_allclose(exact, closed, atol=1e-13, rtol=1e-12)


def test_stability_rejects_critical_and_supercritical_gains():
    counts = np.array([[0., 2.], [3., 0.]])
    p = f.Parameters()
    info = r.stability(counts, .1, p)
    assert info['raw_spectral_radius'] == pytest.approx(np.sqrt(6))
    assert info['slowest_PFNd_mode_tau_s'] == pytest.approx(.1 / (1 - .1 * np.sqrt(6)))
    for gain in [1 / np.sqrt(6), .5, -1, np.inf, np.nan]:
        with pytest.raises(ValidationError):
            r.stability(counts, gain, p)


def test_recurrence_direction_and_recruitment():
    p = f.Parameters(duration_s=1.)
    _, drive = f.make_drive([1, 2, 3], [event()], p)
    rec = np.array([[0, .2, 0], [0, 0, 0], [0, 0, 0.]])
    pre, post = f.integrate(np.ones((3, 1)), drive, p, rec)
    assert pre[:, 1].max() > 0 and not pre[:, 2].any()
    assert pre[1, 1] == 0  # No within-step sequential update.
    assert pre[2, 1] > 0
    for rec in [np.eye(2), -np.eye(3), np.full((3, 3), np.nan)]:
        with pytest.raises(ValidationError):
            f.integrate(np.ones((3, 1)), drive, p, rec)


def test_decay_is_last_crossing_and_reports_censoring():
    times = np.arange(6.)
    signal = np.array([[0], [1], [.001], [.1], [.001], [0]])
    result = r.decay_metrics(times, signal, 1.)
    assert result['decay_to_1pct_after_offset_s'] == 3.
    assert result['decay_observed']
    assert not r.decay_metrics(times[:4], signal[:4], 1.)['decay_observed']
    assert not r.decay_metrics(times, np.zeros((6, 1)), 1.)['decay_observed']


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    root = tmp_path_factory.mktemp('recurrent-sources')
    baseline = root / 'baseline'
    baseline.mkdir()
    meta = {"uuid": "test-uuid", "latestMutationId": 1, "lastDatabaseEdit": "test-edit"}
    (baseline / 'metadata.json').write_text(json.dumps({"status": "complete", "server": SERVER, "dataset": DATASET, "dataset_metadata": meta}))
    ids = [26913, *range(1, 40)]
    neurons = pd.DataFrame({"bodyId": ids, "type": "PFNd", "instance": "PFNd(PB04)_L2_C1"})
    targets = pd.DataFrame({"bodyId": [515192, *range(100, 118)], "type": "hDeltaB", "instance": "hDeltaB_01_C1"})
    neurons.to_csv(baseline / 'PFNd_neurons.csv', index=False)
    targets.to_csv(baseline / 'hDeltaB_neurons.csv', index=False)
    edges = pd.DataFrame([[26913, 515192, 137], [1, 100, 20]], columns=['bodyId_pre', 'bodyId_post', 'weight'])
    edges.to_csv(baseline / 'PFNd_to_hDeltaB_connections.csv', index=False)
    build_matrix(edges, ids, targets.bodyId).to_csv(baseline / 'PFNd_to_hDeltaB_matrix.csv')
    spec = root / 'stimuli.json'
    spec.write_text(json.dumps({"experiments": [{"name": "pulse", "events": [event(ids=(26913,), start=.2, stop=.3)]}]}))
    feedforward = root / 'feedforward'
    f.run_rate_model(baseline, feedforward, {}, stimuli=spec)
    census = root / 'census'
    census.mkdir()
    neurons.to_csv(census / 'PFNd_neurons.csv', index=False)
    neurons.to_csv(census / 'upstream_neurons.csv', index=False)
    pd.DataFrame([[26913, 1, 100], [1, 26913, 100]], columns=['bodyId_pre', 'bodyId_post', 'weight']).to_csv(census / 'incoming_connections.csv', index=False)
    cm = {"status": "complete", "dataset": DATASET, "server": SERVER, "dataset_metadata": meta,
          "data_validated": True, "data_sha256": {p.name: f.sha256(p) for p in census.glob('*.csv')}}
    (census / 'inputs.json').write_text(json.dumps(cm))
    return feedforward, census


def test_end_to_end_matched_sweep_and_relocated_replot(sources, tmp_path, monkeypatch):
    feedforward, census = sources
    monkeypatch.setattr(cli, 'Client', lambda *a, **kw: pytest.fail('Recurrent simulation opened a network client'))
    output = tmp_path / 'sweep'
    assert cli.main(['simulate-recurrent', str(feedforward), '--census', str(census), '--output', str(output), '--recurrent-gains', '0', '.002']) == 0
    manifest = json.loads((output / 'recurrent_model.json').read_text())
    assert manifest['status'] == 'complete'
    assert manifest['validation']['zero_gain_matches_saved_feedforward']
    assert manifest['recurrent_contacts'] == 200
    summary = pd.read_csv(output / 'comparison_summary.csv')
    assert summary.iloc[1].recruited_unstimulated_PFNd == 1
    assert summary.iloc[1].hDeltaB_peak_ratio_to_feedforward > 1
    assert summary.iloc[1].hDeltaB_decay_to_1pct_after_offset_s > summary.iloc[0].hDeltaB_decay_to_1pct_after_offset_s
    counts = pd.read_csv(output / 'raw_recurrent_matrix.csv', index_col=0)
    assert counts.loc[26913, '1'] == 100
    assert not np.diag(counts).any()
    assert (output / 'pulse_comparison.png').stat().st_size > 1000
    moved = tmp_path / 'moved'
    output.rename(moved)
    assert cli.main(['simulate-recurrent-plot', str(moved), '--output', str(tmp_path / 'plots')]) == 0
    assert cli.main(['simulate-plot', str(moved / 'gain_0p002'), '--output', str(tmp_path / 'condition-plots')]) == 0
    with (moved / 'gain_0p002/experiments/pulse/activity.csv').open('a') as stream:
        stream.write('tampered\n')
    with pytest.raises(ValidationError, match='changed'):
        r.replot_recurrent_model(moved, tmp_path / 'bad-plots', {})


def test_incompatible_dataset_and_unsafe_gains_fail_before_output(sources, tmp_path):
    feedforward, census = sources
    copied = tmp_path / 'census'
    shutil.copytree(census, copied)
    meta = json.loads((copied / 'inputs.json').read_text())
    meta['dataset_metadata']['uuid'] = 'different'
    (copied / 'inputs.json').write_text(json.dumps(meta))
    with pytest.raises(ValidationError, match='UUID'):
        r.run_recurrent_model(feedforward, copied, tmp_path / 'invalid', {})
    for gains in [[.001], [0, .001, .001], [0, .1]]:
        with pytest.raises(ValidationError):
            r.run_recurrent_model(feedforward, census, tmp_path / 'invalid', {}, gains)
        assert not (tmp_path / 'invalid').exists()
