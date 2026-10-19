import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fly_connectome import inputs as i
from fly_connectome import __main__ as cli
from fly_connectome.connectivity import DATASET, SERVER, ValidationError


@pytest.fixture
def census():
    targets = pd.DataFrame({
        "bodyId": [10, 11, 26913], "type": ["PFNd"] * 3,
        "instance": ["PFNd(PB04)_L2_C1", "PFNd(PB04)_L3_C2", "PFNd(PB04)_R2_C8"],
        "post": [6, 0, 15],
    })
    rows = []
    for pre, post, weight, typ, info, status in [
        (5, 26913, 10, "LNO2", {"FB": {"post": 6}, "FBl3": {"post": 6}, "NO": {"post": 3}, "CX": {"post": 10}}, "Traced"),
        (5, 10, 4, "LNO2", {"FB": {"post": 4}}, "Traced"),
        (6, 26913, 3, None, {"PB": {"post": 3}}, "Glia"),
        (7, 26913, 2, "EPG", {"PB": {"post": 2}}, "Traced"),
        (26913, 10, 2, "PFNd", {"NO": {"post": 2}}, "Traced"),
    ]:
        rows.append({"bodyId_pre": pre, "bodyId_post": post, "weight": weight, "type": typ,
                     "instance": None, "status": status, "statusLabel": None, "cropped": None,
                     "hemibrainType": typ, "flywireType": None, "consensusNt": None,
                     "labels": ["Segment", "Neuron"] if typ else ["Segment"], "roiInfo": json.dumps(info)})
    return targets, pd.DataFrame(rows)


def prepared(census):
    targets, raw = census
    u, e, r, a = i.prepare_inputs(raw, targets, {"PB", "NO", "FB"})
    return u, e, r, a, i.summarize_inputs(u, e, r, targets, a)


def test_denominators_coverage_and_zero_targets(census):
    u, e, r, a, tables = prepared(census)
    summary = tables["population_inputs.csv"].set_index("population")
    assert summary.loc["LNO2", "share"] == pytest.approx(14 / 21)
    assert summary.loc["LNO2", "coverage"] == pytest.approx(2 / 3)
    assert summary.loc["LNO2", "weight_median"] == 4
    assert summary.loc["LNO2", "targets_with_defined_fraction"] == 2
    assert summary.loc[i.UNTYPED, "weight"] == 3
    d = tables["input_denominators.csv"].set_index("bodyId")
    assert d.loc[26913, "all_segment_weight"] == 15
    assert d.loc[26913, "annotated_neuron_weight"] == 12
    assert d.loc[26913, "non_neuron_segment_weight"] == d.loc[26913, "flagged_weight"] == 3
    assert d.post_minus_edge_residual.eq(0).all()
    pt = tables["population_by_target.csv"]
    assert len(pt) == len(summary) * 3
    assert pt.loc[pt.bodyId_post == 11, "weight"].eq(0).all()
    assert pt.loc[pt.bodyId_post == 11, "input_fraction"].isna().all()
    assert tables["anatomical_group_inputs.csv"].weight.sum() == 21


def test_roi_overlap_residual_and_receiver_anatomy(census):
    u, e, r, a, tables = prepared(census)
    assert r[r.additive].weight.sum() == e.weight.sum() == 21
    assert r[r.roi == "NotPrimary"].weight.sum() == 1
    assert r[r.roi == "FBl3"].weight.sum() == 6
    assert not r.loc[r.roi == "FBl3", "additive"].any()
    ar = tables["anatomical_roi_inputs.csv"]
    # NO input to an R2/C8 target is not interpreted as contact in PB R2 or FB C8.
    no = ar[(ar.population == "LNO2") & (ar.roi == "NO")].iloc[0]
    assert (no.pb_side, no.pb_glomerulus, no.fb_column, no.weight) == ("R", 2, 8, 3)
    assert (u.source_category == "non_neuron_segment").sum() == 1


@pytest.mark.parametrize("case", ["duplicate", "unknown_target", "fractional", "infinite", "negative", "missing_roi", "bad_roi", "overlap", "annotation_drift"])
def test_invalid_census_stops(census, case):
    targets, raw = census
    raw = raw.copy(deep=True)
    if case == "duplicate":
        raw = pd.concat([raw, raw.iloc[:1]], ignore_index=True)
    elif case == "unknown_target":
        raw.loc[0, "bodyId_post"] = 99
    elif case == "fractional":
        raw["weight"] = raw.weight.astype(float)
        raw.loc[0, "weight"] = 1.5
    elif case == "infinite":
        raw["weight"] = raw.weight.astype(float)
        raw.loc[0, "weight"] = np.inf
    elif case == "negative":
        raw.loc[0, "weight"] = -1
    elif case == "missing_roi":
        raw.loc[0, "roiInfo"] = None
    elif case == "bad_roi":
        raw.loc[0, "roiInfo"] = '{"FB": {"post": 1.2}}'
    elif case == "overlap":
        raw.loc[0, "roiInfo"] = '{"FB": {"post": 8}, "NO": {"post": 8}}'
    else:
        raw.loc[1, "type"] = "changed"
    with pytest.raises((ValidationError, ValueError)):
        i.prepare_inputs(raw, targets, {"PB", "NO", "FB"})


def test_self_and_recurrent_edges_are_preserved(census):
    targets, raw = census
    raw.loc[raw.bodyId_pre == 26913, "bodyId_post"] = 26913
    u, e, r, a = i.prepare_inputs(raw, targets, {"PB", "NO", "FB"})
    assert len(e[e.bodyId_pre == e.bodyId_post]) == 1
    assert e.weight.sum() == 21


def test_no_edges_keeps_target_denominators(census):
    targets, raw = census
    u, e, r, a, tables = prepared((targets.assign(post=0), raw.iloc[:0]))
    assert len(tables["input_denominators.csv"]) == 3
    assert tables["input_denominators.csv"].all_segment_weight.eq(0).all()
    assert tables["population_inputs.csv"].empty


@pytest.fixture
def mock_run(census, tmp_path, monkeypatch):
    targets, raw = census
    source = tmp_path / "baseline"
    source.mkdir()
    (source / "metadata.json").write_text(json.dumps({"status": "complete", "dataset": DATASET, "server": SERVER}))
    targets.drop(columns="post").to_csv(source / "PFNd_neurons.csv", index=False)
    monkeypatch.setattr(i, "NeuronCriteria", lambda **kw: kw)
    def fetch_targets(criteria, **kw):
        assert criteria["type"] == "PFNd" and criteria["regex"] is False
        assert kw["omit_rois"] is True
        return targets.copy()
    monkeypatch.setattr(i, "fetch_neurons", fetch_targets)
    monkeypatch.setattr(i, "fetch_meta", lambda **kw: {"uuid": "test"})
    monkeypatch.setattr(i, "fetch_annotations", lambda *args: pd.DataFrame({"bodyId": [26913, 515192], "type": ["PFNd", "hDeltaB"]}))
    monkeypatch.setattr(i, "fetch_connections", lambda *args: pd.DataFrame([[26913, 515192, 137]], columns=[*i.PAIR, "weight"]))
    class Client:
        primary_rois = ["PB", "FB", "NO"]
        def fetch_custom(self, q):
            if "count(n)" in q:
                return pd.DataFrame({"type": ["PFNd", "hDeltaB"], "neuron_count": [3, 1]})
            assert "(s:Segment)" in q and "WHERE e.weight >= 1" in q
            assert "t.bodyId IN [10, 11, 26913]" in q
            assert "s.type =" not in q and "s.status =" not in q
            return raw.copy(deep=True)
    return Client(), source, tmp_path / "run", targets


def test_extraction_offline_replot_and_tamper_detection(mock_run, monkeypatch):
    client, source, output, targets = mock_run
    manifest = i.run_inputs(client, source, output, {"commit": "test", "dirty": True})
    assert manifest["status"] == "complete"
    assert manifest["validation"]["total_weight"] == 21
    assert manifest["validation"]["not_primary_weight"] == 1
    assert len(list(output.glob("*.png"))) == len(list(output.glob("*.pdf"))) == 4
    assert (output / "raw_connections.csv").exists()
    # Plot command must not initialize a network client or require a token.
    monkeypatch.setattr(cli, "Client", lambda *a, **kw: pytest.fail("Offline plot opened a client"))
    new = output.parent / "replot"
    assert cli.main(["inputs-plot", str(output), "--output", str(new)]) == 0
    assert len(list(new.glob("*.png"))) == 4
    with pytest.raises(FileExistsError):
        i.run_inputs(client, source, output, {})
    with (output / "population_inputs.csv").open("a") as f:
        f.write("tampered\n")
    with pytest.raises(ValidationError, match="changed"):
        i.replot_inputs(output, output.parent / "tampered-replot", {})


def test_failed_reconciliation_retains_diagnostics(mock_run, monkeypatch):
    client, source, output, targets = mock_run
    targets.loc[0, "post"] += 1
    monkeypatch.setattr(i, "render_inputs", lambda *a: pytest.fail("Invalid run rendered plots"))
    with pytest.raises(ValidationError, match="reconcile"):
        i.run_inputs(client, source, output, {})
    manifest = json.loads((output / "inputs.json").read_text())
    assert manifest["status"] == "validation_failed"
    assert not manifest.get("data_validated")
    assert (output / "raw_connections.csv").exists()
    assert (output / "input_denominators.csv").exists()


def test_failed_query_is_not_an_empty_census(mock_run, monkeypatch):
    client, source, output, _ = mock_run
    original = client.fetch_custom
    def fail(q):
        if "(s:Segment)" in q:
            raise RuntimeError("sensitive-server-response")
        return original(q)
    monkeypatch.setattr(client, "fetch_custom", fail)
    with pytest.raises(RuntimeError):
        i.run_inputs(client, source, output, {})
    text = (output / "inputs.json").read_text()
    assert json.loads(text)["status"] == "failed"
    assert "sensitive-server-response" not in text
    assert not (output / "population_inputs.csv").exists()


def test_offline_hash_paths_cannot_escape_run(tmp_path):
    outside = tmp_path / "outside.csv"
    outside.write_text("data")
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValidationError, match="outside"):
        i.verify_input_data(source, {"data_validated": True, "data_sha256": {"../outside.csv": i.sha256(outside)}})
