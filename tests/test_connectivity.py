import json
import os

import numpy as np
import pandas as pd
import pytest

from fly_connectome import connectivity as c
from fly_connectome import __main__ as cli
from fly_connectome.plotting import load_matrix, order_matrix, plot_heatmap


def edges(rows=()):
    return pd.DataFrame(rows, columns=c.EDGE_COLUMNS)


def test_matrix_direction_zeros_and_disconnected_neurons():
    matrix = c.build_matrix(edges([(20, 300, 137), (10, 100, 1)]), [30, 20, 10], [300, 200, 100])
    assert matrix.index.tolist() == [10, 20, 30]
    assert matrix.columns.tolist() == [100, 200, 300]
    assert matrix.loc[20, 300] == 137
    assert matrix.loc[10, 100] == 1
    assert (matrix.loc[30] == 0).all()
    assert (matrix[200] == 0).all()
    assert (matrix.dtypes == np.dtype("int64")).all()


def test_empty_connections_and_csv_round_trip(tmp_path):
    matrix = c.build_matrix(edges(), [2, 1], [4, 3])
    assert matrix.shape == (2, 2)
    assert matrix.to_numpy().sum() == 0
    path = tmp_path / "matrix.csv"
    matrix.to_csv(path)
    loaded = load_matrix(path)
    assert loaded.index.tolist() == [1, 2]
    assert loaded.columns.tolist() == [3, 4]
    np.testing.assert_array_equal(loaded, matrix)


@pytest.mark.parametrize("rows", [
    [(1, 2, -1)], [(1, 2, 1.5)], [(1, 2, None)], [(1, 2, float("inf"))],
    [(1, 2, 1), (1, 2, 2)], [(9, 2, 3)], [(1, 9, 3)],
])
def test_reject_corrupt_connection_tables(rows):
    with pytest.raises(c.ValidationError):
        c.build_matrix(edges(rows), [1], [2])


@pytest.mark.parametrize("labels", [["PFNd_L", "hDeltaB"], ["pfnd", "hDeltaB"], ["PFNd"], ["PFNd", "hDeltaB", "hDeltaB_a"]])
def test_unexpected_type_labels_stop(labels):
    with pytest.raises(c.ValidationError):
        c.check_type_inventory(pd.DataFrame({"type": labels}))


def test_type_inventory_allows_unrelated_labels():
    c.check_type_inventory(pd.DataFrame({"type": ["PFNd", "hDeltaB", "PFNv", "other"]}))


def test_population_count_sanity_check_is_saved():
    populations = {
        "PFNd": pd.DataFrame({"bodyId": [1], "type": ["PFNd"]}),
        "hDeltaB": pd.DataFrame({"bodyId": [2], "type": ["hDeltaB"]}),
    }
    summary = c.population_summary(populations)
    assert summary["PFNd"]["matched_types"] == ["PFNd"]
    assert summary["PFNd"]["web_count_sanity_check"] == {"expected": 40, "matches": False}


def test_reference_annotations_and_weight():
    neurons = pd.DataFrame({"bodyId": [c.REFERENCE_POST, c.REFERENCE_PRE], "type": ["hDeltaB", "PFNd"]})
    c.check_reference_annotations(neurons)
    for observed in [137, 136, 0]:
        matrix = c.build_matrix(edges([(c.REFERENCE_PRE, c.REFERENCE_POST, observed)]), [c.REFERENCE_PRE], [c.REFERENCE_POST])
        result = c.reference_result(matrix)
        assert result["observed_weight"] == observed
        if observed == 137:
            c.check_reference_weight(result)
        else:
            with pytest.raises(c.ValidationError):
                c.check_reference_weight(result)
    neurons.loc[0, "type"] = "unexpected"
    with pytest.raises(c.ValidationError):
        c.check_reference_annotations(neurons)


def test_plot_order_is_independent_and_validated(tmp_path):
    matrix = c.build_matrix(edges([(2, 4, 137)]), [2, 1], [4, 3])
    original = matrix.copy()
    ordered = order_matrix(matrix, [2, 1], [4, 3])
    assert ordered.iloc[0, 0] == 137
    assert ordered.index.tolist() == [2, 1]
    pd.testing.assert_frame_equal(matrix, original)
    for rows in [[1], [1, 1], [1, 9]]:
        with pytest.raises(ValueError):
            order_matrix(matrix, rows)
    plot_heatmap(matrix, tmp_path / "heatmap", "PFNd", "hDeltaB", row_order=[2, 1])
    assert (tmp_path / "heatmap.png").stat().st_size > 1000
    assert (tmp_path / "heatmap.pdf").read_bytes().startswith(b"%PDF")


@pytest.fixture
def fake_neuprint(monkeypatch):
    """Mock only API boundaries, exercising actual selection and extraction logic."""
    pfnd = pd.DataFrame({
        "bodyId": [c.REFERENCE_PRE] + list(range(1, 40)),
        "type": ["PFNd"] * 40, "instance": ["unrelated-name"] * 40,
        "status": ["Traced"] * 39 + [None],
        "cropped": [False] * 39 + [True],
    })
    hdeltab = pd.DataFrame({
        "bodyId": [c.REFERENCE_POST, 500], "type": ["hDeltaB"] * 2,
        "instance": ["unrelated-name"] * 2, "status": ["Traced"] * 2,
        "cropped": [False] * 2,
    })
    neurons = pd.concat([pfnd, hdeltab], ignore_index=True)
    calls = []

    class FakeClient:
        def __init__(self, server, dataset, token):
            assert (server, dataset, token) == (c.SERVER, c.DATASET, "test-only-token")
        def fetch_custom(self, query):
            assert "count(n)" in query
            return pd.DataFrame({"type": ["PFNd", "hDeltaB"], "neuron_count": [40, 2]})
        def fetch_all_rois(self):
            return []

    def fetch_neurons(criteria, **kwargs):
        assert kwargs["omit_rois"] is True
        if isinstance(criteria, list):
            return neurons[neurons.bodyId.isin(criteria)].copy()
        assert criteria.regex is False
        assert not criteria.status
        assert criteria.cropped is None
        return neurons[neurons.type.isin(criteria.type)].copy()

    def fetch_adjacencies(sources, targets, **kwargs):
        calls.append((sources, targets))
        assert kwargs["omit_rois"] is True
        assert kwargs["weight_props"] == ["weight"]
        assert kwargs["min_total_weight"] == 1
        assert "rois" not in kwargs
        rows = [(c.REFERENCE_PRE, c.REFERENCE_POST, 137), (500, 1, 2)]
        return pd.DataFrame(), edges([row for row in rows if row[0] in sources and row[1] in targets])

    monkeypatch.setenv("NEUPRINT_TOKEN", "test-only-token")
    monkeypatch.setattr(cli, "Client", FakeClient)
    monkeypatch.setattr(cli, "fetch_meta", lambda **kwargs: {"dataset": c.DATASET, "uuid": "test-uuid"})
    monkeypatch.setattr(c, "fetch_neurons", fetch_neurons)
    monkeypatch.setattr(c, "fetch_adjacencies", fetch_adjacencies)
    return neurons, calls


def test_full_workflow_both_directions_and_summary_before_connections(fake_neuprint, tmp_path, monkeypatch, capsys):
    _, calls = fake_neuprint
    output = tmp_path / "run"
    original_fetch = cli.fetch_connections

    def checked_fetch(*args):
        summary = json.loads((output / "population_summary.json").read_text())
        assert summary["PFNd"]["neuron_count"] == 40
        assert summary["PFNd"]["reconstruction"]["status"]["unknown"] == 1
        assert summary["PFNd"]["reconstruction"]["cropped"]["True"] == 1
        assert summary["PFNd"]["reconstruction"]["statusLabel"] == {"unavailable": 40}
        return original_fetch(*args)

    monkeypatch.setattr(cli, "fetch_connections", checked_fetch)
    assert cli.main(["extract", "--output", str(output)]) == 0
    assert len(calls) == 3
    forward = load_matrix(output / "PFNd_to_hDeltaB_matrix.csv")
    reverse = load_matrix(output / "hDeltaB_to_PFNd_matrix.csv")
    assert forward.shape == (40, 2)
    assert reverse.shape == (2, 40)
    assert forward.loc[c.REFERENCE_PRE, c.REFERENCE_POST] == 137
    assert reverse.loc[500, 1] == 2
    assert forward.loc[1, 500] == 0
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["dataset_metadata"]["uuid"] == "test-uuid"
    assert metadata["matrix"]["zero_meaning"] == c.ZERO_MEANING
    assert "test-only-token" not in (output / "metadata.json").read_text()
    assert "test-only-token" not in capsys.readouterr().out
    assert (output / "PFNd_to_hDeltaB.png").exists()
    assert (output / "hDeltaB_to_PFNd.pdf").exists()
    assert cli.main(["extract", "--output", str(output)]) == 1
    assert len(calls) == 3


@pytest.mark.parametrize("discrepancy", ["annotation", "weight", "missing_edge", "query_failure"])
def test_stop_before_population_queries(fake_neuprint, tmp_path, monkeypatch, discrepancy):
    neurons, calls = fake_neuprint
    if discrepancy == "annotation":
        neurons.loc[neurons.bodyId == c.REFERENCE_PRE, "type"] = "wrong"
    else:
        def bad_connections(*args):
            calls.append("reference")
            if discrepancy == "query_failure":
                raise RuntimeError("test-only-token")
            if discrepancy == "missing_edge":
                return edges()
            return edges([(c.REFERENCE_PRE, c.REFERENCE_POST, 136)])
        monkeypatch.setattr(cli, "fetch_connections", bad_connections)
    output = tmp_path / "failed"
    assert cli.main(["extract", "--output", str(output)]) == 1
    assert len(calls) == (0 if discrepancy == "annotation" else 1)
    assert not list(output.glob("*_matrix.csv"))
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["status"] == ("failed" if discrepancy == "query_failure" else "validation_failed")
    assert "test-only-token" not in (output / "metadata.json").read_text()
    if discrepancy in ["weight", "missing_edge"]:
        assert json.loads((output / "reference_validation.json").read_text())["passed"] is False


def test_dotenv_precedence_and_missing_token(tmp_path, monkeypatch, fake_neuprint):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("NEUPRINT_TOKEN=test-only-token\n")
    monkeypatch.setenv("NEUPRINT_TOKEN", "")
    assert cli.main(["extract", "--output", str(tmp_path / "empty")]) == 1
    assert not (tmp_path / "empty").exists()
    monkeypatch.delenv("NEUPRINT_TOKEN")
    cli.extract(tmp_path / "dotenv")
    assert os.environ["NEUPRINT_TOKEN"] == "test-only-token"


def test_offline_plot_does_not_authenticate_or_read_dotenv(tmp_path, monkeypatch):
    matrix = c.build_matrix(edges([(1, 2, 1)]), [1], [2])
    path = tmp_path / "matrix.csv"
    matrix.to_csv(path)
    monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
    def unexpected(*args, **kwargs):
        pytest.fail("Offline plotting attempted authentication")
    monkeypatch.setattr(cli, "load_dotenv", unexpected)
    monkeypatch.setattr(cli, "Client", unexpected)
    assert cli.main(["plot", str(path), "--output-prefix", str(tmp_path / "offline"),
                     "--source-type", "PFNd", "--target-type", "hDeltaB"]) == 0


def test_changed_reference_stops_after_forward_population_query(fake_neuprint, tmp_path, monkeypatch):
    _, calls = fake_neuprint
    original = cli.fetch_connections
    def changed_reference(client, sources, targets):
        result = original(client, sources, targets)
        if len(sources) > 1:
            result.loc[result.bodyId_pre == c.REFERENCE_PRE, "weight"] = 136
        return result
    monkeypatch.setattr(cli, "fetch_connections", changed_reference)
    output = tmp_path / "changed"
    assert cli.main(["extract", "--output", str(output)]) == 1
    assert len(calls) == 2
    assert not list(output.glob("*_matrix.csv"))
    result = json.loads((output / "population_reference_validation.json").read_text())
    assert result["observed_weight"] == 136
    assert result["passed"] is False
