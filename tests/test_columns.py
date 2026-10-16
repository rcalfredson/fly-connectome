import hashlib
import json

import pandas as pd
import pytest

from fly_connectome import __main__ as cli
from fly_connectome import columns
from fly_connectome.anatomy import anatomical_heatmaps
from fly_connectome.columns import aggregate_columns, ranked_table
from fly_connectome.connectivity import ValidationError, build_matrix, EDGE_COLUMNS
from fly_connectome.plotting import load_matrix


def populations():
    source_columns = [1] * 12 + [2] * 4 + list(range(3, 9)) * 4
    target_columns = [1] * 8 + list(range(2, 13))
    source = pd.DataFrame({"bodyId": [26913] + list(range(1, 40)),
                           "type": "PFNd", "fb_column": source_columns})
    source["instance"] = [f"PFNd(PB04)_L{c + 1}_C{c}" for c in source_columns]
    target = pd.DataFrame({"bodyId": [515192] + list(range(101, 119)),
                           "type": "hDeltaB", "fb_column": target_columns})
    target["instance"] = [f"hDeltaB_{c:02d}_C{c}" for c in target_columns]
    return source, target


def sample_matrix():
    source, target = populations()
    edges = pd.DataFrame([(26913, 515192, 137), (1, 101, 7), (12, 108, 100)],
                         columns=EDGE_COLUMNS)
    return build_matrix(edges, source.bodyId, target.bodyId), source, target, edges


def test_unequal_sizes_zero_pairs_and_conservation():
    matrix, source, target, _ = sample_matrix()
    summary = aggregate_columns(matrix, source, target)
    assert len(summary) == 96
    block = summary.query("pfnd_column == 1 and hdeltab_column == 1").iloc[0]
    assert block.total_weight == 144
    assert block.source_neurons == 12 and block.target_neurons == 8
    assert block.possible_pairs == 96 and block.connected_pairs == 2
    assert block.mean_weight_per_possible_pair == 1.5
    assert block.connection_density == 2 / 96
    zero = summary.query("pfnd_column == 8 and hdeltab_column == 12").iloc[0]
    assert zero.possible_pairs == 4
    assert zero.total_weight == zero.mean_weight_per_possible_pair == zero.connected_pairs == 0
    assert summary.total_weight.sum() == 244
    assert summary.possible_pairs.sum() == 760
    assert summary.connected_pairs.sum() == 3


@pytest.mark.parametrize("corruption", ["duplicate", "missing", "axis", "column", "fraction", "negative", "nan"])
def test_invalid_coverage_or_weights(corruption):
    matrix, source, target, _ = sample_matrix()
    if corruption == "duplicate":
        source.loc[1, "bodyId"] = source.loc[0, "bodyId"]
    elif corruption == "missing":
        target = target.iloc[:-1]
    elif corruption == "axis":
        matrix = matrix.rename(index={26913: 999})
    elif corruption == "column":
        target.loc[target.fb_column == 12, "fb_column"] = 11
    else:
        matrix = matrix.astype(float)
        matrix.iloc[0, 0] = {"fraction": 0.5, "negative": -1, "nan": float("nan")}[corruption]
    with pytest.raises(ValidationError):
        aggregate_columns(matrix, source, target)


def test_rankings_and_ties_at_cutoff():
    matrix, source, target, _ = sample_matrix()
    matrix.loc[source.loc[source.fb_column == 1, "bodyId"], :] = 0
    # Totals favor C1's eight neurons; the per-pair means favor C2.
    matrix.loc[26913, [515192, 108, 109, 110, 111]] = [80, 20, 10, 10, 1]
    summary = aggregate_columns(matrix, source, target)
    group = summary.query("pfnd_column == 1").set_index("hdeltab_column")
    assert group.loc[1, "total_rank"] == 1
    assert group.loc[2, "mean_rank"] == 1
    assert group.loc[3, "total_rank"] == group.loc[4, "total_rank"] == 3
    line = next(line for line in ranked_table(summary).splitlines() if line.startswith("| C1 |"))
    assert line.startswith("| C1 | C1 (#1;")
    assert "C4 (#3;" in line
    assert "| C2 (#1;" in line
    assert "C5" not in line


@pytest.mark.parametrize("show_caveat", [False, True])
def test_offline_workflow_preserves_inputs(tmp_path, monkeypatch, show_caveat):
    matrix, source, target, edges = sample_matrix()
    raw, anatomical, output = [tmp_path / name for name in ["raw", "anatomical", "columns"]]
    raw.mkdir()
    (raw / "metadata.json").write_text(json.dumps({"status": "complete", "dataset": "test"}))
    source.drop(columns="fb_column").to_csv(raw / "PFNd_neurons.csv", index=False)
    target.drop(columns="fb_column").to_csv(raw / "hDeltaB_neurons.csv", index=False)
    matrix.to_csv(raw / "PFNd_to_hDeltaB_matrix.csv")
    edges.to_csv(raw / "PFNd_to_hDeltaB_connections.csv", index=False)
    reverse = pd.DataFrame([], columns=EDGE_COLUMNS)
    reverse.to_csv(raw / "hDeltaB_to_PFNd_connections.csv", index=False)
    build_matrix(reverse, target.bodyId, source.bodyId).to_csv(raw / "hDeltaB_to_PFNd_matrix.csv")
    anatomical_heatmaps(raw, anatomical)
    inputs = [p for directory in [raw, anatomical] for p in directory.iterdir()]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    def unexpected(*args, **kwargs):
        pytest.fail("Column analysis attempted authentication")
    monkeypatch.setattr(cli, "Client", unexpected)
    monkeypatch.setattr(cli, "load_dotenv", unexpected)
    notes = []
    plot = columns.plot_heatmap
    def record_plot(*args, **kwargs):
        notes.append(kwargs.get("note"))
        return plot(*args, **kwargs)
    monkeypatch.setattr(columns, "plot_heatmap", record_plot)
    args = ["columns", str(raw), "--anatomical", str(anatomical)]
    if show_caveat:
        args.append("--show-column-caveat")
    args.extend(["--output", str(output)])
    assert cli.main(args) == 0
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    manifest = json.loads((output / "column_aggregation.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["show_column_caveat"] == show_caveat
    assert len(notes) == 2 and all(bool(note) == show_caveat for note in notes)
    assert manifest["validation"]["source_files_unchanged"]
    for suffix in ["total", "mean"]:
        prefix = output / f"PFNd_to_hDeltaB_column_{suffix}"
        assert load_matrix(str(prefix) + "_matrix.csv").shape == (8, 12)
        assert prefix.with_suffix(".png").stat().st_size > 1000
        assert prefix.with_suffix(".pdf").read_bytes().startswith(b"%PDF")
    assert cli.main(args) == 1
    table = pd.read_csv(anatomical / "PFNd_anatomical_order.csv")
    table.loc[0, "fb_column"] = 8
    table.to_csv(anatomical / "PFNd_anatomical_order.csv", index=False)
    rejected = tmp_path / "rejected"
    args[-1] = str(rejected)
    assert cli.main(args) == 1
    assert not rejected.exists()
