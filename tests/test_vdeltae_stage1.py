import json

import numpy as np
import pandas as pd
import pytest

from fly_connectome import vdeltae_stage1 as stage
from fly_connectome import vdeltae_stage1_figures as figures
from fly_connectome.connectivity import ValidationError
from fly_connectome.inputs import save_json
from fly_connectome.vdeltae_spatial_pilot import record_artifact_hashes


@pytest.fixture
def example():
    bodies = pd.DataFrame(dict(bodyId=range(1, 27), type=["vDeltaE"] * 18 + ["hDeltaG"] * 8,
        instance=[f"synthetic_{v}" for v in range(26)],
        annotation_column=[0, 0, *range(1, 17), *range(1, 9)]))
    raw = pd.DataFrame(dict(contact_relationship_id=range(6), bodyId_pre=[1, 1, 1, 2, 2, 3],
        bodyId_post=[19, 19, 20, 19, 20, 21], compartment_post=["dendrite", "dendrite", "axon", "axon", "axon", "axon"],
        compartment_pre=["axon", "dendrite", "axon", "axon", "axon", "axon"],
        confidence_pre=[.9] * 6, confidence_post=[.9, .9, .9, .9, .5, .9],
        FB_post=[True, True, True, True, True, None]))
    config, _ = stage.prepare_config(dict(parameters=dict(reference_dt_s=.01, k1=.99, c=.001, k2=.8, a=.07, b=1.),
        dt_s=.001, sample_dt_s=.01, rate_tau_s=.02, onset_s=.1, pulse_s=.02, sustained_s=.1, duration_s=.5))
    counts = stage.projection_counts(bodies, raw, .5)
    return bodies, raw, config, counts


def test_contacts_are_disjoint_conserved_and_zero_pairs_retained(example):
    bodies, raw, _, counts = example
    assert len(counts) == 18 * 8 * 2
    assert counts.contacts.sum() == 4
    assert counts.loc[counts.compartment_post.eq("dendrite"), "contacts"].sum() == 2
    raw.loc[0, "compartment_post"] = "unknown"
    with pytest.raises(ValidationError, match="unassigned"):
        stage.projection_counts(bodies, raw, .5)


def test_dc_gain_matching_and_channel_assignment(example):
    bodies, _, config, counts = example
    fast, slow = [stage.build_network(bodies, counts, config, s) for s in stage.SCENARIOS]
    np.testing.assert_array_equal(fast.coupling.effective_weight, slow.coupling.effective_weight)
    dend = slow.coupling.compartment_post.eq("dendrite")
    assert slow.coupling.loc[dend, "channel"].eq("excitability").all()
    assert slow.coupling.loc[~dend, "channel"].eq("rate").all()
    assert fast.coupling.channel.eq("rate").all()
    assert fast.coupling.normalization.eq("none").all()
    assert fast.coupling.sign_evidence.eq("assumed").all()
    assert len(fast.neuron_ids) == 26
    assert not fast.neurons.slow_tau_s.any()
    expected = -2 * config["per_contact_gain"] * .07 / (1 - .8)
    assert slow.coupling.loc[(slow.coupling.bodyId_pre == 1) & (slow.coupling.bodyId_post == 19) & dend,
                             "effective_weight"].item() == pytest.approx(expected)


def test_stimulus_budgets_boundaries_and_body_identity(example):
    bodies, _, config, _ = example
    specs = stage.experiment_specs(bodies, config)
    assert len(specs) == 39
    times = np.arange(51) * .01
    ids = list(range(1, 19))
    single = next(v for v in specs if v["name"] == "body_1_pulse")
    population = next(v for v in specs if v["name"] == "population_pulse")
    a, b = [stage.prescribed_inputs(times, ids, spec, config) for spec in [single, population]]
    np.testing.assert_allclose(a.sum(axis=1), b.sum(axis=1))
    assert a[9].sum() == a[12].sum() == 0
    assert a[10, 0] == a[11, 0] == 1
    np.testing.assert_allclose(b[10], np.full(18, 1 / 18))
    assert a.sum() * .01 == pytest.approx(.02)


@pytest.mark.parametrize("change", [dict(dt_s=.02), dict(source_tonic_rate=-1), dict(duration_s=.2),
    dict(pulse_s=.105), dict(unknown=1), dict(total_increment=True),
    dict(parameters=dict(c="0.001")), dict(parameters=dict(b=True)),
    dict(parameters=dict(reference_dt_s=1, k1=1, k2=.8, a=.07, b=1, c=.001))])
def test_invalid_experiment_specs_rejected(example, change):
    with pytest.raises(ValidationError):
        stage.prepare_config({**example[2], **change})


def test_slow_tail_and_fast_control_differ_without_resets(example):
    bodies, _, config, counts = example
    spec = stage.experiment_specs(bodies, config)[1]
    residuals = []
    for scenario in stage.SCENARIOS:
        network = stage.build_network(bodies, counts, config, scenario)
        trace = stage.solve_trial(network, config, spec)
        fine = stage.solve_trial(network, config, spec, half_dt=True)
        control_spec = stage.experiment_specs(bodies, config)[0]
        control = stage.solve_trial(network, config, control_spec)
        fine_control = stage.solve_trial(network, config, control_spec, half_dt=True)
        metrics, rows, _ = stage.trial_metrics(network, trace, fine,
            [(control.rates, control.excitability), (fine_control.rates, fine_control.excitability)],
            config, spec, scenario)
        assert metrics["rate_half_dt_relative_error"] < config["numerical_tolerance"]
        assert metrics["post_stimulus_slow_decay_max_abs_error"] < 1e-14
        target = network.neurons.cell_type.eq("hDeltaG").to_numpy()
        np.testing.assert_array_equal(trace.rates[0, target], np.zeros(8))
        np.testing.assert_array_equal(trace.excitability[0], np.zeros(26))
        # A prescribed pulse is zero at offset; the receiver is not reset there.
        off = stage.grid_index(spec["stop_s"], config["sample_dt_s"])
        assert trace.rates[off, 0] == 0
        assert trace.rates[off, target].max() > 0
        np.testing.assert_allclose(np.ptp(control.rates[:, target], axis=1), 0)
        residuals.append(trace.rates[off + 10, 18] - control.rates[off + 10, 18])
    assert residuals[1] < residuals[0] < 0


@pytest.fixture
def saved_run(example, tmp_path, monkeypatch):
    bodies, raw, config, _ = example
    source = tmp_path / "source"
    source.mkdir()
    (source / "contacts.json").write_text("{}")
    source_hash = stage.sha256(source / "contacts.json")
    monkeypatch.setattr(stage, "load_stage1_source", lambda path: (
        bodies.copy(), raw.copy(), {"standard_confidence_threshold": .5}, source_hash))
    monkeypatch.setattr(stage, "verify_contact_audit", lambda path: {})
    monkeypatch.setattr(stage, "_protection", lambda *args: None)
    directory = tmp_path / "run"
    stage.run_stage1(source, directory, config)
    return directory


def test_saved_run_roundtrip_and_tampering(saved_run):
    record = stage.verify_stage1(saved_run)
    assert record["status"] == "complete"
    assert len(list(saved_run.glob("*.npz"))) == 78
    response = pd.read_csv(saved_run / "response_by_body.csv")
    assert len(response) == 78 * 8
    response.loc[0, "final_delta"] = 10
    response.to_csv(saved_run / "response_by_body.csv", index=False)
    # Even if artifact hashes are updated, independently reconstructed metrics reject the edit.
    record["artifact_sha256"] = record_artifact_hashes(saved_run, "stage1.json")
    save_json(saved_run / "stage1.json", record)
    with pytest.raises(ValidationError, match="body responses"):
        stage.verify_stage1(saved_run)


def test_changed_artifact_and_incomplete_run_rejected(saved_run):
    (saved_run / "summary.md").write_text("changed")
    with pytest.raises(ValidationError, match="hashes"):
        stage.verify_stage1(saved_run)
    record = json.loads((saved_run / "stage1.json").read_text())
    record["status"] = "failed"
    save_json(saved_run / "stage1.json", record)
    with pytest.raises(ValidationError, match="not complete"):
        stage.verify_stage1(saved_run)


def test_heatmap_shared_signed_scale_and_separate_c0_bodies(saved_run):
    import matplotlib.pyplot as plt
    responses = pd.read_csv(saved_run / "response_by_body.csv")
    bodies = pd.read_csv(saved_run / "selected_bodies.csv")
    figure, matrices = figures.offset_heatmap(responses, bodies)
    assert len(matrices) == 4
    assert all(table.shape == (18, 8) for table in matrices.values())
    assert all(list(table.index[:2]) == [1, 2] for table in matrices.values())
    norms = [axis.images[0].norm for axis in figure.axes if axis.images]
    assert all((norm.vmin, norm.vcenter, norm.vmax) == (norms[0].vmin, 0, norms[0].vmax) for norm in norms)
    figure.canvas.draw()
    plt.close(figure)


def test_nonzero_tonic_control_subtraction_obeys_same_slow_leak(example):
    bodies, _, config, counts = example
    config = {**config, "source_tonic_rate": .2}
    network = stage.build_network(bodies, counts, config, "slow_candidate")
    specs = stage.experiment_specs(bodies, config)
    trace, control = [stage.solve_trial(network, config, spec) for spec in [specs[1], specs[0]]]
    source = network.neurons.cell_type.eq("vDeltaE").to_numpy()
    np.testing.assert_allclose(control.rates[:, source], .2)
    assert np.any(control.excitability[-1] < 0)
    off = stage.grid_index(specs[1]["stop_s"], config["sample_dt_s"])
    delta = trace.excitability - control.excitability
    tau = -config["parameters"]["reference_dt_s"] / np.log(config["parameters"]["k2"])
    expected = delta[off] * np.exp(-(trace.time_s[off:] - trace.time_s[off])[:, None] / tau)
    np.testing.assert_allclose(delta[off:], expected, atol=1e-15, rtol=1e-10)


def test_failed_run_records_status_and_preserves_existing_outputs(saved_run, monkeypatch):
    output = saved_run.parent / "failed-run"
    config = json.loads((saved_run / "config.json").read_text())
    source = json.loads((saved_run / "stage1.json").read_text())["source"]
    def fail(*args, **kwargs):
        raise ValidationError("deliberate numerical failure")
    monkeypatch.setattr(stage, "solve_trial", fail)
    with pytest.raises(ValidationError, match="deliberate"):
        stage.run_stage1(source, output, config)
    record = json.loads((output / "stage1.json").read_text())
    assert record["status"] == "failed"
    assert record["finished_at_utc"]
    assert "config.json" in record["artifact_sha256"]
    before = (saved_run / "stage1.json").read_bytes()
    with pytest.raises(FileExistsError):
        stage.run_stage1(source, saved_run, config)
    assert (saved_run / "stage1.json").read_bytes() == before


def test_source_ancestor_directories_are_protected(tmp_path):
    coords, census, audit, probe, inputs = [tmp_path / name for name in ["coords", "census", "audit", "probe", "inputs"]]
    coords.mkdir()
    save_json(coords / "coordinates.json", dict(source_probe=str(probe), source_inventory=dict(
        annotation_audit_directory=str(audit), sources={"example": dict(source_manifest_file=str(inputs / "manifest.json"))})))
    source = tmp_path / "contacts"
    record = dict(sources=dict(coordinates=str(coords), census=str(census)))
    for root in [source, coords, census, audit, probe, inputs]:
        with pytest.raises(ValidationError, match="separate"):
            stage._protection(source, record, root / "new-output")
    stage._protection(source, record, tmp_path / "separate-output")
