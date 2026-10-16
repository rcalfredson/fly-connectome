import hashlib
import json

import pandas as pd
import pytest

from fly_connectome import __main__ as cli
from fly_connectome.anatomy import annotation_order
from fly_connectome.connectivity import ValidationError, build_matrix, EDGE_COLUMNS
from fly_connectome.plotting import load_matrix


def neurons(typ, ids, names):
    return pd.DataFrame({"bodyId": ids, "type": typ, "instance": names})


def test_numeric_columns_sides_and_duplicate_labels():
    h = neurons("hDeltaB", [3, 2, 1], ["hDeltaB_12_C10", "hDeltaB_02_C2", "hDeltaB_02_C2"])
    assert annotation_order(h, "hDeltaB").bodyId.tolist() == [1, 2, 3]
    p = neurons("PFNd", [1, 3, 2], ["PFNd(PB04)_R9_C1", "PFNd(PB04)_L2_C1", "PFNd(PB04)_L2_C1"])
    assert annotation_order(p, "PFNd").bodyId.tolist() == [2, 3, 1]


@pytest.mark.parametrize("name", [None, "PFNd", "PFNd(PB04)_L2_C8"])
def test_missing_unknown_and_inconsistent_annotations(name):
    with pytest.raises(ValidationError):
        annotation_order(neurons("PFNd", [1], [name]), "PFNd")


def saved_extraction(path):
    path.mkdir()
    (path / "metadata.json").write_text(json.dumps({"status": "complete", "dataset": "test"}))
    populations = {
        "PFNd": neurons("PFNd", [26913, 1], ["PFNd(PB04)_R2_C8", "PFNd(PB04)_L2_C1"]),
        "hDeltaB": neurons("hDeltaB", [515192, 2], ["hDeltaB_07_C6", "hDeltaB_12_C10"]),
    }
    for typ, table in populations.items():
        table.to_csv(path / f"{typ}_neurons.csv", index=False)
    for pre, post, edges in [
        ("PFNd", "hDeltaB", [(26913, 515192, 137)]),
        ("hDeltaB", "PFNd", [(2, 1, 5)]),
    ]:
        edges = pd.DataFrame(edges, columns=EDGE_COLUMNS)
        edges.to_csv(path / f"{pre}_to_{post}_connections.csv", index=False)
        build_matrix(edges, populations[pre].bodyId, populations[post].bodyId).to_csv(
            path / f"{pre}_to_{post}_matrix.csv")


def test_offline_artifacts_preserve_inputs_and_all_weights(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "anatomical"
    saved_extraction(source)
    original = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    def unexpected(*args, **kwargs):
        pytest.fail("Anatomical plotting attempted authentication")
    monkeypatch.setattr(cli, "Client", unexpected)
    monkeypatch.setattr(cli, "load_dotenv", unexpected)
    assert cli.main(["anatomical", str(source), "--output", str(output)]) == 0
    assert original == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    for stem in ["PFNd_to_hDeltaB", "hDeltaB_to_PFNd"]:
        before = load_matrix(source / f"{stem}_matrix.csv")
        after = load_matrix(output / f"{stem}_anatomical_matrix.csv")
        pd.testing.assert_frame_equal(after.reindex(index=before.index, columns=before.columns), before)
        assert (output / f"{stem}_anatomical.png").stat().st_size > 1000
        assert (output / f"{stem}_anatomical.pdf").read_bytes().startswith(b"%PDF")
    manifest = json.loads((output / "anatomical_ordering.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["validation"]["reference"]["observed_weight"] == 137
    assert cli.main(["anatomical", str(source), "--output", str(output)]) == 1


@pytest.mark.parametrize("corruption", ["reference", "weight", "population", "duplicate", "incomplete"])
def test_corruption_stops_before_output(tmp_path, corruption):
    source, output = tmp_path / "source", tmp_path / "anatomical"
    saved_extraction(source)
    matrix_path = source / "PFNd_to_hDeltaB_matrix.csv"
    if corruption in ["reference", "weight", "population"]:
        matrix = load_matrix(matrix_path)
        if corruption == "reference":
            matrix.loc[26913, 515192] = 136
            edges = pd.read_csv(source / "PFNd_to_hDeltaB_connections.csv")
            edges["weight"] = 136
            edges.to_csv(source / "PFNd_to_hDeltaB_connections.csv", index=False)
        elif corruption == "weight":
            matrix.loc[1, 2] = 5
        else:
            matrix = matrix.drop(index=1)
        matrix.to_csv(matrix_path)
    elif corruption == "duplicate":
        path = source / "PFNd_neurons.csv"
        table = pd.read_csv(path)
        pd.concat([table, table.iloc[:1]]).to_csv(path, index=False)
    else:
        (source / "metadata.json").write_text('{"status":"failed"}')
    assert cli.main(["anatomical", str(source), "--output", str(output)]) == 1
    assert not output.exists()
