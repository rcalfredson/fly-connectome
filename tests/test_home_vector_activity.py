"""Frozen activity archives must preserve time alignment and signed readouts."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from fly_connectome.home_vector.__main__ import batch
from fly_connectome.home_vector.analysis import activity
from fly_connectome.home_vector.analysis.activity_summary import population_readout
from fly_connectome.home_vector.analysis.checkpoint import capture
from fly_connectome.home_vector.data import Circuit, digest, verify_bundle
from test_home_vector_checkpoint import fixtures


def test_signed_populations_cancel_and_intercept_is_added_once():
    rates = np.array([[1., 1., 0.], [3., 3., 2.]])
    decoder = np.array([[1., -1., 0.], [0., 0., 2.]])
    intercept = np.array([4., -2.])
    # Interleaved populations ensure grouping does not assume contiguous indices.
    contributions, total = population_readout(rates, decoder, intercept,
                                             ['A', 'B', 'A'], ['B', 'A'])
    np.testing.assert_array_equal(contributions[:, 0], [[-1., 0.], [-3., 0.]])
    np.testing.assert_array_equal(contributions[:, 1], [[1., 0.], [3., 4.]])
    np.testing.assert_array_equal(total, [[4., -2.], [4., 2.]])
    assert not np.array_equal(rates[0], rates[1])
    with pytest.raises(ValueError, match='axes'):
        population_readout(rates, decoder, intercept, ['A', 'B', 'A'], ['A'])
    rates[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        population_readout(rates, decoder, intercept, ['A', 'B', 'A'], ['B', 'A'])


def test_recorder_uses_input_before_next_state_and_preserves_decoder():
    from fly_connectome.home_vector.model import HomeVectorNetwork
    from test_home_vector_data import circuit_fixture
    circuit = circuit_fixture()
    model = HomeVectorNetwork(circuit, recurrence=False)
    with torch.no_grad():
        model.D.copy_(torch.tensor([[1., -.5, 0.], [0., 2., -1.]]))
        model.d.copy_(torch.tensor([.3, -.2]))
    row = dict(protocol=[[0, 0, 1, 0], [.005, 90, -1, 0], [.015, 90, 0, 0]])
    arrays, residual = activity.record_trajectory(model, circuit, row, .0025)
    np.testing.assert_array_equal(arrays['rates'][0], 0)
    assert len(arrays['rates']) == 7
    # Change the input at tick 2: state at tick 2 stays fixed, tick 3 responds.
    other = dict(protocol=[[0, 0, 1, 0], [.015, 0, 0, 0]])
    unchanged, _ = activity.record_trajectory(model, circuit, other, .0025)
    np.testing.assert_array_equal(arrays['rates'][:3], unchanged['rates'][:3])
    assert np.max(np.abs(arrays['rates'][3] - unchanged['rates'][3])) > 1e-6
    with torch.no_grad():
        x, _, _ = batch(circuit, [row], .0025, 'cpu', model.a.dtype)
        prediction, rates = model(x, .0025)
    np.testing.assert_array_equal(arrays['pfn'], x[0].numpy())
    np.testing.assert_array_equal(arrays['rates'], rates[0].numpy())
    np.testing.assert_array_equal(arrays['prediction'], prediction[0].numpy())
    np.testing.assert_allclose(arrays['population_contribution'].sum(axis=1) + [.3, -.2],
                               arrays['prediction'], atol=1e-6)
    assert residual < 1e-6


def test_activity_bundle_replays_all_cases_and_retains_independent_checkpoint(tmp_path):
    args = fixtures(tmp_path)
    before = {p: digest(p) for p in tmp_path.rglob('*') if p.is_file()}
    result = activity.analyze(args)
    output = Path(args.output)
    manifest = verify_bundle(output)
    assert manifest['kind'] == 'home_vector_activity_v1'
    assert manifest['device'] == 'cpu' and not manifest['refitting']
    assert all(digest(p) == sha for p, sha in before.items())
    report = json.loads((output / 'activity.json').read_text())
    assert len(report['trajectories']) == result['trajectories'] == 12
    with (output / 'neurons.csv').open() as stream:
        assert [int(row['body_id']) for row in csv.DictReader(stream)] == [10, 11, 12]
    with (output / 'neuron_summary.csv').open() as stream:
        summaries = list(csv.DictReader(stream))
    snap = capture(Path(args.run), Path(args.source))
    for i, row in enumerate(report['trajectories']):
        with np.load(output / f'{row["stem"]}.npz') as arrays:
            expected, _ = activity.record_trajectory(snap['network'],
                Circuit(**snap['circuit']),
                row['record'], report['dt_s'])
            for key in arrays.files:
                np.testing.assert_array_equal(arrays[key], expected[key])
            assert float(summaries[i * 3]['rate_mean']) == pytest.approx(arrays['rates'][:, 0].mean())
        for kind in ('atlas', 'atlas_detail', 'readout'):
            for extension in ('png', 'pdf'):
                assert (output / f'{row["stem"]}_{kind}.{extension}').stat().st_size > 0
    assert 'trajectory_00_atlas.png' in (output / 'index.html').read_text()
    Path(args.run).rename(tmp_path / 'moved_run')
    archived = capture(output, Path(args.source))
    assert archived['checkpoint_sha256'] == report['checkpoint_sha256']
    assert archived['input_kind'] == 'saved_activity_snapshot'
    assert archived['checkpoint']['update'] == 7


def test_output_guards_and_failed_recording(tmp_path, monkeypatch):
    args = fixtures(tmp_path)
    args.output = str(Path(args.run) / 'nested')
    with pytest.raises(ValueError, match='separate'):
        activity.analyze(args)
    args.output = str(tmp_path / 'audit')

    def fail(*_):
        raise RuntimeError('recording interrupted')

    monkeypatch.setattr(activity, 'record_trajectory', fail)
    with pytest.raises(RuntimeError, match='interrupted'):
        activity.analyze(args)
    assert json.loads((Path(args.output) / 'manifest.json').read_text())['status'] == 'failed'
