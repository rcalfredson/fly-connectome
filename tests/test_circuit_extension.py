import json
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from fly_connectome import circuit_extension as c
from fly_connectome.connectivity import ValidationError


def fixture():
    sources = pd.DataFrame(dict(bodyId=[1, 2, 3], pre=[1, 1, 0]))
    rows = []
    # Weight can exceed source pre because contacts are polyadic.
    for pre, post, weight, typ, labels in [(1, 8, 5, "hDeltaG", ["Segment", "Neuron"]),
            (2, 8, 2, "hDeltaG", ["Segment", "Neuron"]),
            (1, 9, 3, None, ["Segment"]), (2, 2, 1, "hDeltaB", ["Segment", "Neuron"])]:
        row = dict(bodyId_pre=pre, bodyId_post=post, weight=weight,
                   labels=labels, roiInfo='{"FB":{"post":1}}')
        row.update({k: None for k in c.ANNOTATIONS})
        row.update(type=typ, pre=1, post=10)
        rows.append(row)
    return pd.DataFrame(rows), sources


def test_outgoing_denominators_zero_sources_untyped_and_self():
    raw, sources = fixture()
    partners, den, per, summary = c.summarize_outputs(raw, sources)
    summary = summary.set_index("population")
    assert summary.loc["hDeltaG", "output_share"] == pytest.approx(7 / 11)
    assert summary.loc["hDeltaG", "source_coverage"] == pytest.approx(2 / 3)
    assert summary.loc["hDeltaG", "weight_min"] == 0
    assert partners.set_index("bodyId").loc[9, "category"] == "non_neuron_segment"
    assert partners.set_index("bodyId").loc[9, "population"] == c.UNTYPED
    assert den.all_outgoing_segment_weight.tolist() == [8, 3, 0]
    g = per[per.population == "hDeltaG"].set_index("bodyId_pre")
    assert g.loc[1, "output_fraction"] == 5 / 8
    assert g.loc[2, "output_fraction"] == 2 / 3
    assert np.isnan(g.loc[3, "output_fraction"])
    assert summary.loc["hDeltaB", "weight"] == 1  # self connection retained
    assert len(per) == 9


@pytest.mark.parametrize("case", ["duplicate", "fractional", "negative", "nan", "unknown_source", "labels", "inconsistent", "reserved"])
def test_malformed_census_rejected(case):
    raw, sources = fixture()
    if case == "duplicate":
        raw = pd.concat([raw, raw.iloc[:1]])
    elif case in ["fractional", "negative", "nan"]:
        raw["weight"] = raw.weight.astype(float)
        raw.loc[0, "weight"] = dict(fractional=1.5, negative=-1, nan=np.nan)[case]
    elif case == "unknown_source":
        raw.loc[0, "bodyId_pre"] = 999
    elif case == "labels":
        raw.at[0, "labels"] = ["Neuron"]
    elif case == "inconsistent":
        raw.loc[1, "instance"] = "different"
    elif case == "reserved":
        raw.loc[0, "type"] = c.UNTYPED
    with pytest.raises((ValidationError, ValueError)):
        c.summarize_outputs(raw, sources)


def test_empty_census_retains_source_axes():
    raw, sources = fixture()
    partners, den, per, summary = c.summarize_outputs(raw.iloc[:0], sources)
    assert partners.empty and per.empty and summary.empty
    assert den.all_outgoing_segment_weight.tolist() == [0, 0, 0]


def test_query_scope_and_target_validation():
    query = c.output_query([2, 1])
    assert "(t:Segment)" in query and "e.weight >= 1" in query
    assert "s.bodyId IN [1, 2]" in query
    assert "status =" not in query and "roiInfo AS roiInfo" in query
    assert ("hDeltaC", "vDeltaE") in c.MOTIFS
    assert ("vDeltaE", "hDeltaG") in c.MOTIFS
    assert ("hDeltaG", "vDeltaE") in c.MOTIFS
    assert ("hDeltaG", "hDeltaG") in c.MOTIFS
    raw, _ = fixture()
    with pytest.raises(ValidationError, match="Target outside"):
        c.validate_edges(raw, [1, 2], [8])


def test_query_failure_is_not_success_or_zero(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "baseline").write_text("baseline")
    population = pd.DataFrame(dict(bodyId=[1], type=["hDeltaB"], instance=["hDeltaB_01_C1"]))
    meta = dict(uuid="test", latestMutationId=1, lastDatabaseEdit="test")
    monkeypatch.setattr(c, "load_network", lambda _: (None, {"hDeltaB": population}, None,
        {"dataset_metadata": meta}, {"baseline": c.sha256(source / "baseline")}))
    monkeypatch.setattr(c, "fetch_meta", lambda **_: meta)
    class Client:
        def fetch_custom(self, query):
            raise ConnectionError("Do not serialize server request or credentials")
    output = tmp_path / "failed"
    with pytest.raises(ConnectionError):
        c.run(Client(), source, output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error_type"] == "ConnectionError"
    assert "Do not serialize" not in (output / "manifest.json").read_text()
    assert not (output / "population_outputs.csv").exists()
    with pytest.raises(FileExistsError):
        c.run(Client(), source, output)


@pytest.fixture
def mocked_census(tmp_path, monkeypatch):
    import ast
    import re
    raw, sources = fixture()
    neurons = []
    for body, typ in [(1, "hDeltaB"), (2, "hDeltaB"), (3, "hDeltaB"), (8, "hDeltaG")]:
        row = {k: None for k in c.ANNOTATIONS}
        row.update(bodyId=body, type=typ, instance=f"{typ}_{body}", pre=1, post=10,
                   labels=["Segment", "Neuron"])
        neurons.append(row)
    neurons = pd.DataFrame(neurons)
    source = tmp_path / "source"
    source.mkdir()
    (source / "baseline").write_text("baseline")
    meta = dict(uuid="test", latestMutationId=1, lastDatabaseEdit="test")
    monkeypatch.setattr(c, "load_network", lambda _: (None,
        {"hDeltaB": neurons[neurons.type == "hDeltaB"]}, None,
        {"dataset_metadata": meta}, {"baseline": c.sha256(source / "baseline")}))
    monkeypatch.setattr(c, "fetch_meta", lambda **_: meta)
    monkeypatch.setattr(c, "fetch_connections", lambda *a: pd.DataFrame(
        [[26913, 515192, 137]], columns=[*c.PAIR, "weight"]))
    class Client:
        def fetch_custom(self, query):
            if "n.type IN" in query:
                return neurons.copy()
            if "(t:Segment)" in query:
                return raw.copy()
            if "count(n)" in query:
                return pd.DataFrame(dict(type=["hDeltaB", "hDeltaG"], neuron_count=[3, 1]))
            src = ast.literal_eval(re.search(r"s.bodyId IN (\[[^\]]*\])", query)[1])
            dst = ast.literal_eval(re.search(r"t.bodyId IN (\[[^\]]*\])", query)[1])
            return raw[raw.bodyId_pre.isin(src) & raw.bodyId_post.isin(dst)][[*c.PAIR, "weight", "roiInfo"]].copy()
    output = tmp_path / "complete"
    result = c.run(Client(), source, output)
    assert result["status"] == "complete"
    return output


def test_mocked_extraction_offline_reconstruction_and_hash_tampering(mocked_census):
    from fly_connectome.extension_analysis import validate_saved_census
    output = mocked_census
    validated, _, _, motifs = validate_saved_census(output)
    assert validated["validation"]["total_output_weight"] == 11
    absent = motifs[(motifs.source == "vDeltaE") & (motifs.target == "hDeltaG")].iloc[0]
    assert absent.query_status == "population_absent" and pd.isna(absent.weight)
    # Rehash each changed summary so rejection tests semantic validation rather
    # than file integrity. Exercise the real interpreter with and without -O.
    script = textwrap.dedent("""\
        import json
        from pathlib import Path
        import sys

        import pandas as pd
        from fly_connectome.connectivity import ValidationError
        from fly_connectome.extension_analysis import validate_saved_census
        from fly_connectome.inputs import save_json, sha256

        source = Path(sys.argv[1])
        summary_path = source / "motif_summary.csv"
        manifest_path = source / "manifest.json"
        original_summary = summary_path.read_bytes()
        original_manifest = manifest_path.read_bytes()
        validate_saved_census(source)
        try:
            for field in ["weight", "positive_pairs", "possible_pairs",
                          "all_target_post_denominator"]:
                summary = pd.read_csv(summary_path)
                index = summary.index[summary.query_status.eq("queried")][0]
                summary.loc[index, field] += 1
                summary.to_csv(summary_path, index=False)
                manifest = json.loads(original_manifest)
                manifest["data_sha256"][summary_path.name] = sha256(summary_path)
                save_json(manifest_path, manifest)
                try:
                    validate_saved_census(source)
                except ValidationError as exc:
                    if field not in str(exc):
                        raise RuntimeError(f"Unexpected validation error for {field}") from exc
                else:
                    raise RuntimeError(f"Accepted inconsistent {field}")
                summary_path.write_bytes(original_summary)
                manifest_path.write_bytes(original_manifest)
        finally:
            summary_path.write_bytes(original_summary)
            manifest_path.write_bytes(original_manifest)
        validate_saved_census(source)
    """)
    for flags in [[], ["-O"]]:
        result = subprocess.run(
            [sys.executable, *flags, "-c", script, str(output)],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    (output / "outgoing_connections.csv").write_text("tampered")
    with pytest.raises(ValidationError, match="changed"):
        validate_saved_census(output)


def test_report_preserves_absent_populations_and_queried_zeros(mocked_census, tmp_path):
    from fly_connectome.extension_analysis import analyze

    movement, trajectory = tmp_path / "movement", tmp_path / "trajectory"
    for directory, prefix in [(movement, "movement"), (trajectory, "trajectory")]:
        directory.mkdir()
        c.save_json(directory / f"{prefix}_config.json", dict(target_ids=[1, 2, 3]))
        if directory == movement:
            (directory / "baseline_1").mkdir()
            pairs = pd.DataFrame(dict(relation=["same_world_different_heading"],
                centered_cosine=[0.7], trial_a=["a"], trial_b=["b"]))
            for representation in ["raw", "movement_evoked"]:
                pairs.to_csv(directory / "baseline_1" / f"{representation}_pair_comparisons.csv", index=False)
        c.save_json(directory / f"{prefix}_model.json", dict(status="complete", data_validated=True,
            data_sha256={str(p.relative_to(directory)): c.sha256(p)
                         for p in directory.rglob("*") if p.is_file()}))

    # Extraction does not create edge files for absent populations.
    assert not (mocked_census / "hDeltaB_to_vDeltaE_connections.csv").exists()
    output = tmp_path / "analysis"
    analyze(mocked_census, movement, trajectory, output)
    manifest = json.loads((output / "analysis.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["validation"]["source_hashes_unchanged"]
    for name, digest in manifest["data_sha256"].items():
        assert c.sha256(output / name) == digest

    report = (output / "connectivity_summary.md").read_text()
    absent_line = next(line for line in report.splitlines() if line.startswith("| hDeltaB | vDeltaE |"))
    assert "population_absent" in absent_line and "undefined" in absent_line
    assert "undefined, not measured zeros" in report
    # Both hDeltaG and hDeltaB exist, but the reverse edge query returned no contacts.
    zero_line = next(line for line in report.splitlines() if line.startswith("| hDeltaG | hDeltaB |"))
    assert "queried" in zero_line and "undefined" not in zero_line
    targets = pd.read_csv(output / "hdeltab_input_by_candidate_neuron.csv")
    assert set(targets.type) == {"hDeltaB", "hDeltaG"}
    assert targets.set_index("bodyId").loc[8, "hdeltab_weight"] == 7
    assert targets.set_index("bodyId").loc[1, "hdeltab_weight"] == 0
