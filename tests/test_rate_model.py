from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from fly_connectome import __main__ as cli
from fly_connectome import rate_model as m
from fly_connectome.connectivity import DATASET, SERVER, ValidationError, build_matrix


def event(ids=(1,), start=.2, stop=.3, amplitude=1.):
    return {"body_ids": list(ids), "start_s": start, "stop_s": stop,
            "amplitude": amplitude, "label": "test"}


def test_cascade_matches_independent_step_solution_and_converges():
    p = m.Parameters(duration_s=1.)
    events = [event(start=0, stop=1)]
    times, drive = m.make_drive([1], events, p)
    pre, post = m.integrate(np.array([[2.]]), drive, p)
    exact_pre = 1 - np.exp(-times / p.tau_s)
    exact_post = 2 * (1 - (1 + times / p.tau_s) * np.exp(-times / p.tau_s))
    assert pre[100, 0] == pytest.approx(1 - np.exp(-1), abs=.002)
    assert post[1, 0] == 0  # Simultaneous updates use the old presynaptic activity.
    np.testing.assert_allclose(pre[:, 0], exact_pre, atol=.002)
    np.testing.assert_allclose(post[:, 0], exact_post, atol=.005)
    fine_p = replace(p, dt_s=p.dt_s / 2)
    _, fine_drive = m.make_drive([1], events, fine_p)
    fine = m.integrate(np.array([[2.]]), fine_drive, fine_p)
    assert m.relative_error(fine[1][::2, 0], exact_post) < m.relative_error(post[:, 0], exact_post)


def test_pulse_returns_to_zero_and_analytic_superposition():
    p = m.Parameters(duration_s=3.)
    weights = np.array([[.137, 0.], [0., .2]])
    events = [event(), event(ids=(2,), start=.3, stop=.4)]
    times, drive = m.make_drive([1, 2], events, p)
    pre, post = m.integrate(weights, drive, p)
    exact = m.analytical_response(weights, [1, 2], events, times, p.tau_s)
    assert m.relative_error(post, exact[1]) < .01
    assert (pre >= 0).all() and (post >= 0).all()
    assert post[-1].max() < 1e-9
    assert post[:, 0].argmax() < post[:, 1].argmax()
    assert times[post[:, 0].argmax()] > .3  # A delayed peak after stimulus offset is expected.


def test_direction_axes_disconnected_targets_and_gain():
    p = m.Parameters(duration_s=1.)
    _, drive = m.make_drive([1, 2], [event(ids=(1,), start=0, stop=1)], p)
    weights = np.array([[2., 0., 0.], [0., 3., 0.]])
    pre, post = m.integrate(weights, drive, p)
    pre2, post2 = m.integrate(2 * weights, drive, p)
    assert post[:, 0].max() > 0
    assert not post[:, 1:].any()  # No accidental transposition or cross-neuron activation.
    np.testing.assert_array_equal(pre, pre2)
    np.testing.assert_allclose(post2, 2 * post, rtol=1e-14)
    for original, repeated in zip((pre, post), m.integrate(weights, drive, p)):
        np.testing.assert_array_equal(original, repeated)


def test_events_are_half_open_and_overlaps_add():
    p = m.Parameters(duration_s=1.)
    _, drive = m.make_drive([1], [event(), event(start=.25, stop=.35)], p)
    assert drive[199, 0] == 0 and drive[200, 0] == 1
    assert drive[250, 0] == 2 and drive[300, 0] == 1
    assert drive[350, 0] == drive[-1, 0] == 0


def test_zero_input_and_zero_coupling():
    p = m.Parameters(duration_s=1.)
    _, drive = m.make_drive([1], [], p)
    assert all(not x.any() for x in m.integrate(np.array([[1.]]), drive, p))
    _, drive = m.make_drive([1], [event()], p)
    pre, post = m.integrate(np.zeros((1, 1)), drive, p)
    assert pre.max() > 0 and not post.any()


@pytest.mark.parametrize("override", [
    {"tau_s": 0}, {"tau_s": float("nan")}, {"dt_s": -.1}, {"dt_s": .1},
    {"gain": -1}, {"gain": float("inf")}, {"duration_s": 0}, {"duration_s": .3005},
])
def test_invalid_parameters_rejected(override):
    with pytest.raises(ValidationError):
        replace(m.Parameters(), **override).validate()


@pytest.mark.parametrize("bad_event", [
    event(ids=(999,)), event(ids=(1, 1)), event(ids=()), event(start=-.1),
    event(start=.3, stop=.2), event(stop=4), event(start=.2005),
    event(amplitude=-1), event(amplitude=float("inf")),
    {**event(), "body_ids": [True]}, {**event(), "typo": 1},
])
def test_invalid_protocols_rejected(bad_event):
    spec = {"experiments": [{"name": "test", "events": [bad_event]}]}
    with pytest.raises(ValidationError):
        m.validate_experiments(spec, {1}, m.Parameters())


def test_protocol_names_cannot_escape_output_or_collide():
    for names in [["../escape"], ["same", "same"]]:
        with pytest.raises(ValidationError):
            m.validate_experiments({"experiments": [{"name": n, "events": []} for n in names]}, {1}, m.Parameters())


@pytest.fixture
def baseline(tmp_path):
    source = tmp_path / "baseline"
    source.mkdir()
    (source / "metadata.json").write_text(json.dumps({"status": "complete", "server": SERVER, "dataset": DATASET}))
    ids = [26913, *range(1, 40)]
    pfnd = pd.DataFrame({"bodyId": ids, "type": "PFNd", "instance": "PFNd(PB04)_L2_C1"})
    hd = pd.DataFrame({"bodyId": [515192, *range(100, 118)], "type": "hDeltaB", "instance": "hDeltaB_01_C1"})
    pfnd.to_csv(source / "PFNd_neurons.csv", index=False)
    hd.to_csv(source / "hDeltaB_neurons.csv", index=False)
    edges = pd.DataFrame([[26913, 515192, 137], [1, 100, 5]], columns=["bodyId_pre", "bodyId_post", "weight"])
    edges.to_csv(source / "PFNd_to_hDeltaB_connections.csv", index=False)
    build_matrix(edges, ids, hd.bodyId).to_csv(source / "PFNd_to_hDeltaB_matrix.csv")
    spec_path = tmp_path / "stimuli.json"
    spec_path.write_text(json.dumps({"experiments": [{"name": "pulse", "events": [event(ids=(26913,))]}]}))
    return source, spec_path


def test_end_to_end_cli_runs_offline_and_replots(baseline, tmp_path, monkeypatch):
    source, stimuli = baseline
    output = tmp_path / "run"
    monkeypatch.setattr(cli, "Client", lambda *a, **kw: pytest.fail("Simulation opened a network client"))
    assert cli.main(["simulate", str(source), "--stimuli", str(stimuli), "--output", str(output)]) == 0
    manifest = json.loads((output / "rate_model.json").read_text())
    assert manifest["status"] == "complete" and manifest["network"]["raw_contact_total"] == 142
    assert manifest["validation"]["max_relative_analytical_error"] < .02
    assert len(pd.read_csv(output / "parameter_sensitivity.csv")) == 5
    assert len(pd.read_csv(output / "neuron_summary.csv")) == 59
    activity = pd.read_csv(output / "experiments/pulse/activity.csv")
    assert activity.shape == (3001, 60)
    assert activity.hDeltaB_515192.max() > 0
    assert activity.hDeltaB_100.max() == 0
    weights = pd.read_csv(output / "effective_weight_matrix.csv", index_col=0)
    assert weights.loc[26913, "515192"] == pytest.approx(.137)
    assert weights.loc[26913, "100"] == 0
    assert (output / "pulse.pdf").read_bytes().startswith(b"%PDF")
    assert (output / "hDeltaB_peak_profiles.png").stat().st_size > 1000
    assert cli.main(["simulate", str(source), "--stimuli", str(stimuli), "--output", str(output)]) == 1
    # Replot a relocated run after its original baseline path also disappears.
    # The recorded absolute provenance paths must not be used to load data.
    moved = tmp_path / "moved-run"
    output.rename(moved)
    source.rename(tmp_path / "moved-baseline")
    assert cli.main(["simulate-plot", str(moved), "--output", str(tmp_path / "replot")]) == 0
    with (moved / "experiments/pulse/activity.csv").open("a") as f:
        f.write("tampered\n")
    with pytest.raises(ValidationError, match="changed"):
        m.replot_rate_model(moved, tmp_path / "tamper-replot", {})


def test_corrupt_matrix_rejected_before_creating_run(baseline, tmp_path):
    source, stimuli = baseline
    path = source / "PFNd_to_hDeltaB_matrix.csv"
    counts = pd.read_csv(path, index_col=0)
    counts.loc[26913, "515192"] = 138
    counts.to_csv(path)
    with pytest.raises(AssertionError):
        m.run_rate_model(source, tmp_path / "bad", {}, stimuli=stimuli)
    assert not (tmp_path / "bad").exists()


def test_failed_validation_preserves_traces_without_success_status(baseline, tmp_path, monkeypatch):
    source, stimuli = baseline
    monkeypatch.setattr(m, "relative_error", lambda *args: .1)
    output = tmp_path / "failed"
    with pytest.raises(ValidationError, match="validation failed"):
        m.run_rate_model(source, output, {}, stimuli=stimuli)
    assert (output / "experiments/pulse/activity.csv").exists()
    manifest = json.loads((output / "rate_model.json").read_text())
    assert manifest["status"] == "validation_failed" and not manifest.get("data_validated")


def test_hash_paths_cannot_escape_run(tmp_path):
    outside = tmp_path / "outside.csv"
    outside.write_text("data")
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValidationError):
        m.verify_model_data(source, {"data_validated": True, "data_sha256": {"../outside.csv": m.sha256(outside)}})
