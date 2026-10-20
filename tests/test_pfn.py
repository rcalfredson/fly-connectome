import json

import numpy as np
import pandas as pd
import pytest

from fly_connectome import __main__ as cli
from fly_connectome import pfn
from fly_connectome import rate_model as model
from fly_connectome.anatomy import annotation_order
from fly_connectome.connectivity import DATASET, SERVER, ValidationError, build_matrix


def test_pfnv_annotations_are_literal_not_pfnd_mapping():
    neurons = pd.DataFrame({"bodyId": [1, 2, 3], "type": "PFNv",
                            "instance": ["PFNv(PB05)_L9_C9", "PFNv(PB05)_R2_C9", "PFNv(PB05)_L7_C7"]})
    order = annotation_order(neurons, "PFNv").set_index("bodyId")
    assert order.loc[1, "fb_column"] == order.loc[2, "fb_column"] == 9
    assert order.loc[1, "pb_glomerulus"] == 9
    assert order.loc[3, "pb_glomerulus"] == order.loc[3, "fb_column"] == 7
    for bad in ["PFNv(PB05)_L2_C10", "PFNv(PB04)_L2_C1", "PFNv(PB05)_L1_C1"]:
        with pytest.raises(ValidationError):
            annotation_order(neurons.assign(instance=bad), "PFNv")


@pytest.fixture
def extraction(tmp_path, monkeypatch):
    root = tmp_path / "baseline"
    root.mkdir()
    meta = {"uuid": "test", "latestMutationId": 1, "lastDatabaseEdit": "same"}
    model.save_json(root / "metadata.json", {"status": "complete", "server": SERVER,
                                            "dataset": DATASET, "dataset_metadata": meta})
    populations = {
        "PFNd": pd.DataFrame({"bodyId": [26913, *range(1, 40)], "type": "PFNd", "instance": "PFNd(PB04)_L2_C1"}),
        "hDeltaB": pd.DataFrame({"bodyId": [515192, *range(100, 118)], "type": "hDeltaB", "instance": "hDeltaB_01_C1"}),
        "PFNv": pd.DataFrame({"bodyId": range(200, 220), "type": "PFNv", "instance": "PFNv(PB05)_L2_C1"}),
    }
    edges = {
        "PFNd": pd.DataFrame([[26913, 515192, 137]], columns=["bodyId_pre", "bodyId_post", "weight"]),
        "PFNv": pd.DataFrame([[200, 515192, 11], [201, 100, 3]], columns=["bodyId_pre", "bodyId_post", "weight"]),
    }
    for label in ["PFNd", "hDeltaB"]:
        populations[label].to_csv(root / f"{label}_neurons.csv", index=False)
    edges["PFNd"].to_csv(root / "PFNd_to_hDeltaB_connections.csv", index=False)
    build_matrix(edges["PFNd"], populations["PFNd"].bodyId, populations["hDeltaB"].bodyId).to_csv(root / "PFNd_to_hDeltaB_matrix.csv")
    monkeypatch.setattr(pfn, "fetch_meta", lambda **kw: meta.copy())
    monkeypatch.setattr(pfn, "type_inventory", lambda client: pd.DataFrame({"type": ["PFNd", "PFNv", "hDeltaB"]}))
    monkeypatch.setattr(pfn, "fetch_annotations", lambda client, criteria: populations[criteria.type[0]].sort_values("bodyId").reset_index(drop=True))
    monkeypatch.setattr(pfn, "fetch_connections", lambda client, source, target: edges["PFNv" if 200 in set(source) else "PFNd"].copy())
    return root, populations, edges, meta


def test_expansion_preserves_full_axes_and_original_weights(extraction, tmp_path):
    source, populations, _, _ = extraction
    output = tmp_path / "expanded"
    original = {p.name: model.sha256(p) for p in source.iterdir()}
    manifest = pfn.run_pfn_expansion(None, source, output, {})
    assert manifest["status"] == "complete"
    assert manifest["validation"]["PFNd_annotations_and_weights_unchanged"]
    counts, loaded, _, _, _ = pfn.load_pfn_network(output)
    assert counts.shape == (60, 19) and counts.to_numpy().sum() == 151
    assert counts.loc[200, 515192] == 11 and not counts.loc[219].any()
    assert counts.loc[26913, 515192] == 137
    assert set(loaded["hDeltaB"].bodyId) == set(populations["hDeltaB"].bodyId)
    assert original == {p.name: model.sha256(p) for p in source.iterdir()}
    with (output / "PFNv_to_hDeltaB_connections.csv").open("a") as stream:
        stream.write("tampered\n")
    with pytest.raises(ValidationError, match="changed"):
        pfn.load_pfn_network(output)


def test_metadata_mismatch_stops_before_output(extraction, tmp_path, monkeypatch):
    source, _, _, meta = extraction
    monkeypatch.setattr(pfn, "fetch_meta", lambda **kw: {**meta, "uuid": "changed"})
    with pytest.raises(ValidationError, match="UUID"):
        pfn.run_pfn_expansion(None, source, tmp_path / "invalid", {})
    assert not (tmp_path / "invalid").exists()


def test_changed_pfnd_weights_prevent_success(extraction, tmp_path):
    source, _, edges, _ = extraction
    edges["PFNd"].loc[0, "weight"] = 138
    output = tmp_path / "changed"
    with pytest.raises(AssertionError):
        pfn.run_pfn_expansion(None, source, output, {})
    assert json.loads((output / "metadata.json").read_text())["status"] == "validation_failed"


def test_combined_model_keeps_populations_separate_and_sums_targets(extraction, tmp_path, monkeypatch):
    source, _, _, _ = extraction
    expanded = tmp_path / "expanded"
    pfn.run_pfn_expansion(None, source, expanded, {})
    monkeypatch.setattr(cli, "Client", lambda *a, **kw: pytest.fail("Offline simulation opened a client"))
    events = [{"body_ids": [body], "start_s": .1, "stop_s": .2, "amplitude": 1.} for body in [26913, 200]]
    spec = {"experiments": [{"name": "d", "events": events[:1]},
                             {"name": "v", "events": events[1:]},
                             {"name": "both", "events": events}]}
    model.save_json(tmp_path / "stimuli.json", spec)
    output = tmp_path / "model"
    assert cli.main(["simulate-pfn", str(expanded), "--output", str(output), "--stimuli", str(tmp_path / "stimuli.json"), "--duration", ".6"]) == 0
    manifest = json.loads((output / "rate_model.json").read_text())
    assert manifest["network"]["PFNd"] == 40 and manifest["network"]["PFNv"] == 20
    frames = {name: pd.read_csv(output / f"experiments/{name}/activity.csv") for name in ["d", "v", "both"]}
    assert not frames["d"].filter(regex="^PFNv_").to_numpy().any()
    assert not frames["v"].filter(regex="^PFNd_").to_numpy().any()
    for prefix in ["PFNd_", "PFNv_", "hDeltaB_"]:
        np.testing.assert_allclose(frames["both"].filter(regex="^" + prefix),
                                   frames["d"].filter(regex="^" + prefix) + frames["v"].filter(regex="^" + prefix),
                                   rtol=1e-12, atol=1e-14)
    # The expanded d-only condition preserves the original solver's response.
    counts, _, _, _, _ = model.load_network(source)
    parameters = model.Parameters(duration_s=.6)
    _, drive = model.make_drive(counts.index, spec["experiments"][0]["events"], parameters)
    _, original_post = model.integrate(counts.to_numpy() * parameters.gain, drive, parameters)
    np.testing.assert_allclose(frames["d"][[f"hDeltaB_{body}" for body in counts.columns]], original_post, atol=1e-14)
    checks = pd.read_csv(output / "numerical_validation.csv")
    assert set(checks.population) == {"PFNd", "PFNv", "hDeltaB"} and checks.passed.all()
    assert (output / "both.pdf").read_bytes().startswith(b"%PDF")
    expanded.rename(tmp_path / "relocated-extraction")
    assert cli.main(["simulate-plot", str(output), "--output", str(tmp_path / "heatmaps"), "--heatmaps-only"]) == 0
    assert not (tmp_path / "heatmaps/hDeltaB_peak_profiles.png").exists()
    assert (tmp_path / "heatmaps/both.png").stat().st_size > 1000


def test_pfn_defaults_use_pb_groups_without_equalizing_counts():
    rows = [(200, "PFNv(PB05)_L2_C1"), (201, "PFNv(PB05)_L5_C4"),
            (202, "PFNv(PB05)_L8_C7"), (203, "PFNv(PB05)_R2_C9")]
    orders = {
        "PFNv": annotation_order(pd.DataFrame(rows, columns=["bodyId", "instance"]).assign(type="PFNv"), "PFNv"),
        "PFNd": annotation_order(pd.DataFrame({"bodyId": [26913, 1, 2, 3], "type": "PFNd", "instance":
                                                ["PFNd(PB04)_L2_C1", "PFNd(PB04)_L2_C1", "PFNd(PB04)_L5_C4", "PFNd(PB04)_L8_C7"]}), "PFNd"),
    }
    spec = pfn.pfn_experiments(orders)
    assert len(spec["experiments"]) == 12
    model.validate_experiments(spec, {26913, 1, 2, 3, 200, 201, 202, 203}, model.Parameters())
    assert spec["experiments"][-1]["events"][0]["body_ids"] == [1, 26913]
    assert spec["experiments"][-1]["events"][1]["body_ids"] == [200]
