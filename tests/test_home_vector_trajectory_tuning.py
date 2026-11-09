"""Selected-state comparisons preserve checkpoint, history and effect interpretation."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from fly_connectome.home_vector.__main__ import begin, finish
from fly_connectome.home_vector.analysis.activity import record_trajectory, write_csv
from fly_connectome.home_vector.analysis.checkpoint import capture
from fly_connectome.home_vector.analysis.trajectory_tuning import default_cases, effect_metrics, main
from fly_connectome.home_vector.data import Circuit, digest, verify_bundle, write_json


def conditions():
    return [dict(heading_deg=h, body_direction_deg=b) for h in (0., 90.) for b in (90., 270.)]


def test_offset_gain_and_remapping_are_separate_and_flat_maps_are_undefined():
    a = np.array([1., 2., 3., 5.])
    offset = effect_metrics(a, a + 20., conditions())
    assert offset['mean_shift'] == 20
    assert offset['range_ratio_b_over_a'] == 1
    assert offset['fitted_gain_b_from_a'] == 1
    assert offset['affine_residual_rms'] == pytest.approx(0, abs=1e-12)
    gain = effect_metrics(a, 2 * a + 5., conditions())
    assert gain['fitted_gain_b_from_a'] == 2
    assert gain['fitted_offset_b_from_a'] == 5
    assert gain['range_ratio_b_over_a'] == 2
    assert gain['pattern_correlation'] == pytest.approx(1)
    changed = effect_metrics(a, [1., 5., 4., 3.], conditions())
    assert changed['affine_residual_rms'] > 0
    inverted = effect_metrics(a, -a, conditions())
    assert inverted['fitted_gain_b_from_a'] == -1
    assert inverted['right_left_order_reversed']
    flat = effect_metrics(np.ones(4), a, conditions())
    assert flat['fitted_gain_b_from_a'] is None
    assert flat['pattern_correlation'] is None
    assert flat['range_ratio_b_over_a'] is None
    assert not effect_metrics(a, a, conditions())['right_left_order_reversed']
    weak = effect_metrics(a, a * -1e-7, conditions())
    assert weak['right_left_order_reversed'] is None


def archive(tmp_path):
    from test_home_vector_checkpoint import fixtures
    args = fixtures(tmp_path)
    snap = capture(Path(args.run), Path(args.source))
    circuit = Circuit(**snap['circuit'])
    folder, manifest = begin(tmp_path / 'paths', 'home_vector_activity_paths_v1')
    (folder / 'current_checkpoint.pt').write_bytes(snap['blob'])
    write_json(folder / 'current_metadata.json', dict(run_manifest=snap['manifest'], circuit=snap['circuit'],
        history=snap['history'], training_config=snap['training_config'],
        checkpoint_sha256=snap['checkpoint_sha256'], checkpoint_update=snap['checkpoint']['update']))
    write_csv(folder / 'neurons.csv', [dict(body_id=body, cell_type=kind)
        for body, kind in zip(circuit.dynamic_ids, circuit.dynamic_types)])
    paths = []
    for i, forward in enumerate((1., -1.)):
        row = dict(protocol=[[0., 0., forward, 0.], [.03, 0., 0., 0.]])
        arrays, _ = record_trajectory(snap['network'], circuit, row, .0025)
        np.savez_compressed(folder / f'path_{i:02d}.npz', **arrays)
        paths.append(dict(path=f'path_{i:02d}', record=row))
    write_json(folder / 'raw_paths.json', paths)
    finish(folder, manifest)
    cases = [dict(case='example', title='Matched states', pairs=[[[0, .02], [1, .02]]],
                  focus=[circuit.dynamic_ids[0]])]
    casefile = tmp_path / 'cases.json'
    write_json(casefile, cases)
    return args, folder, casefile


def test_renamed_archive_produces_verified_complete_independent_probe_bundle(tmp_path):
    args, folder, casefile = archive(tmp_path)
    renamed = tmp_path / 'activity-by-trajectory'
    folder.rename(renamed)
    frozen = {p: digest(p) for p in tmp_path.rglob('*') if p.is_file()}
    output = tmp_path / 'selected'
    report = main(['--activity', str(renamed), '--source', args.source, '--output', str(output),
        '--cases', str(casefile), '--angles', '2', '--times', '.0025', '.01', '--chunk-size', '3'])
    assert report['checkpoint_update'] == 7
    assert len(report['states']) == 2 and len(report['pairs']) == 1
    assert report['replay_verification'][0]['rate_replay_max_abs_error'] == 0
    assert verify_bundle(output)['status'] == 'complete'
    assert all(digest(path) == sha for path, sha in frozen.items())
    for row in report['states']:
        with np.load(output / f'{row["key"]}.npz') as data:
            assert data['rates'].shape[:2] == (6, 5)
            np.testing.assert_array_equal(data['rates'][:, 0], np.broadcast_to(data['initial_state'], data['rates'][:, 0].shape))
            assert data['movement_delta'].shape[:2] == (4, 2)
    assert (output / 'report.md').exists() and (output / 'focus_summary.csv').exists()
    for plot in report['plots']:
        for extension in ('png', 'pdf'):
            assert (output / f'{plot["stem"]}.{extension}').stat().st_size > 0


def test_replay_rejects_saved_states_that_do_not_match_checkpoint_even_if_hashed(tmp_path):
    args, folder, casefile = archive(tmp_path)
    with np.load(folder / 'path_00.npz') as data:
        arrays = {key: data[key].copy() for key in data.files}
    arrays['rates'][8, 0] += .1
    np.savez_compressed(folder / 'path_00.npz', **arrays)
    finish(folder, json.loads((folder / 'manifest.json').read_text()))
    output = tmp_path / 'bad'
    with pytest.raises(AssertionError):
        main(['--activity', str(folder), '--source', args.source, '--output', str(output), '--cases', str(casefile)])
    assert not output.exists()


def test_default_plan_has_five_cases_seven_pairs_and_thirteen_unique_states():
    cases = default_cases()
    assert len(cases) == 5
    assert sum(len(row['pairs']) for row in cases) == 7
    assert len({tuple(state) for row in cases for pair in row['pairs'] for state in pair}) == 13
