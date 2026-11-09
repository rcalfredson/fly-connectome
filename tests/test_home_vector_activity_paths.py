"""Raw rates and fixed-reference state norms answer distinct questions."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from fly_connectome.home_vector.analysis.activity_paths import (
    rate_leaders, norm_traces, varied_paths, resume_records, add_varied_restarts)
from fly_connectome.home_vector.data import sample_protocol


def test_raw_leaders_share_ties_exclude_silence_and_zero_duration_endpoint():
    rates = np.array([[0., 0., 0.], [4., 1., 2.], [2., 2., 1.], [0., 0., 100.]])
    global_fraction, within = rate_leaders(rates, ['A', 'A', 'B'])
    np.testing.assert_allclose(global_fraction, [.75, .25, 0.])
    np.testing.assert_allclose(within, [.75, .25, 1.])
    silent, _ = rate_leaders(np.zeros((4, 3)), ['A', 'A', 'B'])
    np.testing.assert_array_equal(silent, 0.)


def test_fixed_reference_hidden_change_can_fall_while_absolute_norm_rises():
    rates = np.array([[2., 0., 4.], [4., 0., 3.]])
    decoder = np.array([[1., 0., 0.], [0., 1., 0.]])
    reference = np.array([0., 0., 1.])
    traces = norm_traces(rates, decoder, reference)
    assert traces['full_norm'][1] > traces['full_norm'][0]
    np.testing.assert_array_equal(traces['hidden_change_from_pause_onset_norm'], [3., 2.])
    np.testing.assert_array_equal(traces['hidden_norm'], [4., 3.])
    # This measurement must not silently reset its baseline on movement restart.
    assert traces['hidden_change_from_pause_onset_norm'][0] == 3.


def test_varied_paths_reproduce_world_geometry_across_body_motion_pairs():
    rows = varied_paths(4)
    assert rows == varied_paths(4)
    assert len(rows) == 16
    for i in range(0, len(rows), 2):
        assert rows[i]['family'] == rows[i + 1]['family']
        a, target_a = sample_protocol(rows[i]['protocol'], .0025)
        b, target_b = sample_protocol(rows[i + 1]['protocol'], .0025)
        np.testing.assert_allclose(target_a, target_b, atol=1e-10)
        assert not np.array_equal(a, b)
    assert {len(r['protocol']) - 1 for r in rows[:8]} == {3, 4}


def test_restart_grid_has_heading_only_controls_and_correct_displacement():
    rows = resume_records(2., 4, [.5, 1., 2.], 90.)
    assert len(rows) == 4 + 4 * 4 * 3
    assert rows[0]['kind'] == 'stationary' and rows[0]['heading_deg'] == 90.
    for row in rows:
        assert any(r['kind'] == 'stationary' and r['heading_deg'] == row['heading_deg'] for r in rows)
        _, target = sample_protocol(row['protocol'], .0025)
        np.testing.assert_allclose(np.linalg.norm(target[-1]), row['speed_mm_s'] * 2., atol=1e-10)


def test_varied_restart_controls_match_entire_heading_history():
    rows = add_varied_restarts([], 8., .0025, 0)
    assert len(rows) == 32
    for row in rows:
        if row['kind'] != 'path_moving':
            continue
        control = rows[row['matched_control_index']]
        a, _ = sample_protocol(row['protocol'], .0025)
        b, target = sample_protocol(control['protocol'], .0025)
        np.testing.assert_array_equal(a[:, 0], b[:, 0])
        np.testing.assert_array_equal(b[:, 1:], 0)
        np.testing.assert_array_equal(target, 0)


def test_small_complete_paths_bundle(tmp_path):
    from test_home_vector_checkpoint import fixtures
    from fly_connectome.home_vector.analysis import activity, activity_paths
    from fly_connectome.home_vector.data import verify_bundle, digest
    args = fixtures(tmp_path)
    activity.analyze(args)
    atlas = Path(args.output)
    before = {p: digest(p) for p in atlas.iterdir() if p.is_file()}
    output = tmp_path / 'paths'
    report = activity_paths.main(['--activity', str(atlas), '--source', args.source,
        '--output', str(output), '--angles', '2', '--pauses', '.05', '--resume-seconds', '.01'])
    assert report['straight_conditions'] == 12 and report['varied_paths'] == 16
    assert report['restart_cases'] == 2
    assert verify_bundle(output)['status'] == 'complete'
    assert all(digest(p) == sha for p, sha in before.items())
    for name in ['raw_rate_sweep', 'movement_after_pause', 'path_00_activity']:
        for extension in ['png', 'pdf']:
            assert (output / f'{name}.{extension}').stat().st_size > 0
    with np.load(output / 'resume_00.npz') as data:
        assert data['prediction'].shape[1:] == (5, 2)
        assert np.all(data['full_norm'] >= 0)
