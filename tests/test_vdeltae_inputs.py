import json

import numpy as np
import pandas as pd
import pytest

from fly_connectome import vdeltae_inputs as v
from fly_connectome.connectivity import ValidationError
from fly_connectome.inputs import upstream_query


def fixture():
    targets = pd.DataFrame(dict(bodyId=[10, 11, 12], type=["vDeltaE"]*3, post=[9, 2, 0]))
    rows = []
    for pre, post, weight, typ in [(1, 10, 5, "hDeltaB"), (2, 11, 2, "PFNd"),
                                   (3, 10, 3, None), (10, 10, 1, "vDeltaE")]:
        row = {k: None for k in v.UPSTREAM if k != "bodyId"}
        row.update(bodyId_pre=pre, bodyId_post=post, weight=weight, type=typ,
                   labels=["Segment", "Neuron"] if typ else ["Segment"],
                   roiInfo='{"FB":{"post":1},"FBl6":{"post":1}}')
        rows.append(row)
    mapping = pd.DataFrame(dict(bodyId=[1, 2, 20, 30], type=["hDeltaB", "PFNd", "PFNd", "PFNv"]))
    return pd.DataFrame(rows), targets, mapping


def test_denominators_untyped_zero_targets_and_self():
    raw, targets, _ = fixture()
    tables = v.prepare(raw, targets)
    summary = tables["population_inputs.csv"].set_index("population")
    assert summary.loc["hDeltaB", "input_share"] == pytest.approx(5/11)
    assert summary.loc["hDeltaB", "coverage"] == pytest.approx(1/3)
    assert summary.loc["vDeltaE", "weight"] == 1  # self connection
    assert summary.loc[v.UNTYPED, "weight"] == 3
    den = tables["input_denominators.csv"]
    assert den.all_segment_weight.tolist() == [9, 2, 0]
    assert den.post_minus_edge_residual.eq(0).all()
    per = tables["population_by_target.csv"]
    assert len(per) == 12
    assert per[per.bodyId_post == 12].input_fraction.isna().all()
    assert per[(per.population == "hDeltaB") & (per.bodyId_post == 10)].input_fraction.iloc[0] == 5/9
    assert tables["incoming_connections.csv"].roiInfo.eq(raw.roiInfo.iloc[0]).all()


def test_model_overlap_keeps_disconnected_sources_and_zero_populations():
    raw, targets, mapping = fixture()
    tables = v.prepare(raw, targets)
    summary, matrices = v.overlap_tables(tables["incoming_connections.csv"], mapping, targets)
    assert matrices["PFNd"].shape == (2, 3)
    assert matrices["PFNd"].loc[20].eq(0).all()
    assert matrices["PFNv"].to_numpy().sum() == 0
    assert summary.set_index("population").loc["hDeltaB", "input_share"] == 5/11


@pytest.mark.parametrize("case", ["duplicate", "fractional", "negative", "missing", "outside", "inconsistent", "labels", "target_type"])
def test_invalid_data_rejected(case):
    raw, targets, _ = fixture()
    if case == "duplicate":
        raw = pd.concat([raw, raw.iloc[:1]])
    elif case in ["fractional", "negative", "missing"]:
        raw["weight"] = raw.weight.astype(float)
        raw.loc[0, "weight"] = {"fractional": 1.5, "negative": -1, "missing": np.nan}[case]
    elif case == "outside":
        raw.loc[0, "bodyId_post"] = 999
    elif case == "inconsistent":
        raw.loc[1, "bodyId_pre"] = 1
    elif case == "labels":
        raw.at[0, "labels"] = ["Neuron"]
    else:
        targets.loc[0, "type"] = "PFNd"
    with pytest.raises((ValidationError, ValueError)):
        v.prepare(raw, targets)


def test_empty_census_and_query_scope():
    raw, targets, _ = fixture()
    tables = v.prepare(raw.iloc[:0], targets.assign(post=0))
    assert len(tables["input_denominators.csv"]) == 3
    assert tables["population_inputs.csv"].empty
    query = upstream_query(targets.bodyId)
    assert "(s:Segment)" in query and "e.weight >= 1" in query
    assert "s.type =" not in query and "e.roiInfo AS roiInfo" in query


def test_offline_verify_reconciles_counts_and_detects_tampering(tmp_path):
    raw, targets, mapping = fixture()
    v.save_raw(tmp_path / "raw_connections.csv", raw)
    targets.to_csv(tmp_path / "vDeltaE_neurons.csv", index=False)
    tables = v.prepare(raw, targets)
    for name, table in tables.items():
        table.to_csv(tmp_path / name, index=False)
    mapping.to_csv(tmp_path / "modeled_neurons.csv", index=False)
    summary, matrices = v.overlap_tables(tables["incoming_connections.csv"], mapping, targets)
    summary.to_csv(tmp_path / "model_overlap_summary.csv", index=False)
    for pop, matrix in matrices.items():
        matrix.to_csv(tmp_path / f"{pop}_to_vDeltaE_matrix.csv")
    def manifest():
        v.save_json(tmp_path / "manifest.json", dict(status="complete", data_validated=True,
            data_sha256={p.name: v.sha256(p) for p in tmp_path.glob("*.csv")}))
    manifest()
    assert v.verify(tmp_path)["status"] == "complete"
    (tmp_path / "model_overlap_summary.csv").write_text("tampered")
    with pytest.raises(ValidationError, match="changed"):
        v.verify(tmp_path)
    summary.to_csv(tmp_path / "model_overlap_summary.csv", index=False)
    targets.loc[0, "post"] += 1
    targets.to_csv(tmp_path / "vDeltaE_neurons.csv", index=False)
    changed = v.prepare(raw, targets)
    for name, table in changed.items():
        table.to_csv(tmp_path / name, index=False)
    manifest()  # Internally hash-consistent but biologically inconsistent denominator.
    with pytest.raises(ValidationError, match="reconcile"):
        v.verify(tmp_path)
