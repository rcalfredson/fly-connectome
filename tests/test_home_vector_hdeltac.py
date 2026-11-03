import copy
import json

import numpy as np
import pandas as pd
import pytest

from fly_connectome.home_vector.__main__ import begin, finish
from fly_connectome.home_vector.data import Circuit, TYPES, encode, write_json
from fly_connectome.home_vector.dynamics import initial_coefficients, numpy_rollout
from fly_connectome.home_vector.hdeltac import ADDED_PROJECTIONS, contact_matrix, extend, validate


@pytest.fixture
def anatomy(tmp_path):
    input_ids, dynamic_ids, hc = list(range(1, 61)), list(range(100, 145)), list(range(200, 220))
    dynamic_types = ["hDeltaB"] * 19 + ["vDeltaE"] * 18 + ["hDeltaG"] * 8
    input_types = ["PFNd"] * 40 + ["PFNv"] * 20
    edges = []
    for a, b in [(a, b) for a in TYPES for b in TYPES if (a, b) != ("vDeltaE", "hDeltaB")]:
        edges.append(dict(pre=60 + dynamic_types.index(a), post=dynamic_types.index(b),
                          contacts=2, denominator=2., source_type=a, target_type=b))
    for i in range(60):
        for j in range(45):
            if len(edges) < 1075:
                edges.append(dict(pre=i, post=j, contacts=1, denominator=1.,
                                  source_type=input_types[i], target_type=dynamic_types[j]))
    base = Circuit(input_ids, input_types, dynamic_ids, dynamic_types,
                   [0.] * 60, [-31.] * 40 + [137.] * 20, edges)
    neurons = pd.DataFrame(dict(bodyId=input_ids + dynamic_ids + hc,
                                type=input_types + dynamic_types + ["hDeltaC"] * 20))
    populations = {t: neurons.loc[neurons.type.eq(t), "bodyId"].tolist()
                   for t in ["PFNd", "PFNv", *TYPES, "hDeltaC"]}
    rows, tables = [], []
    directory, record = begin(tmp_path / "anatomy", "home_vector_hdeltac_anatomy_v1")
    snapshot = ["neuprint.janelia.org", "male-cns:v1.0", "example", 1]
    record.update(snapshot=snapshot, projections=[list(pair) for pair in ADDED_PROJECTIONS])
    write_json(directory / "base_circuit.json", base.as_dict())
    neurons.to_csv(directory / "neurons.csv", index=False)
    for pre, post in ADDED_PROJECTIONS:
        table = pd.DataFrame([], columns=["bodyId_pre", "bodyId_post", "weight"])
        if pre != "PFNd":
            table = pd.DataFrame([[populations[pre][0], populations[post][0], 3]], columns=table.columns)
        tables.append(table)
        prefix = f"{pre}_to_{post}"
        table.to_csv(directory / f"{prefix}_connections.csv", index=False)
        contact_matrix(table, populations[pre], populations[post]).to_csv(directory / f"{prefix}_matrix.csv")
        rows.append(dict(source_type=pre, target_type=post, contacts=int(table.weight.sum()), positive_pairs=len(table)))
    raw = pd.concat(tables, ignore_index=True)
    raw[raw.bodyId_post.isin(hc)].to_csv(directory / "incoming_connections.csv", index=False)
    raw[~raw.bodyId_post.isin(hc)].to_csv(directory / "outgoing_connections.csv", index=False)
    record["summary"] = rows
    finish(directory, record)
    provenance = dict(snapshot=snapshot, projections=[], source_sha256={}, omitted=["unchanged omissions"])
    return directory, record, base, provenance


def test_extension_preserves_original_edges_ids_input_encoding_and_seeded_coefficients(anatomy):
    directory, _, base, provenance = anatomy
    original = copy.deepcopy(base.as_dict())
    expanded, sources = extend(base, provenance, directory)
    assert base.as_dict() == original
    assert expanded.edges[:1075] == base.edges
    assert expanded.dynamic_ids[:45] == base.dynamic_ids
    assert expanded.population_types == (*TYPES, "hDeltaC")
    assert len(expanded.dynamic_ids) == 65
    assert len(expanded.edges) == 1083
    for seed in (0, 12, 99):
        np.testing.assert_array_equal(initial_coefficients(expanded, seed)[:1075],
                                      initial_coefficients(base, seed))
    controls = np.array([[10., 1., -.4], [130., 0., 0.], [45., -1., 2.]])
    np.testing.assert_array_equal(encode(expanded, controls), encode(base, controls))
    incoming = next(e for e in expanded.edges[1075:] if e["source_type"] == "PFNv")
    assert incoming["denominator"] == 3 / 20  # Includes all zero-receiver hDeltaC neurons.
    assert sources["omitted"] == provenance["omitted"]
    assert len(sources["projections"]) == 9  # The successful zero PFNd block is explicit.


def test_extension_rejects_mismatched_core_snapshot_and_fabricated_zeros(anatomy):
    directory, record, base, provenance = anatomy
    wrong = copy.deepcopy(base)
    wrong.edges[0]["contacts"] += 1
    with pytest.raises(ValueError, match="different core"):
        extend(wrong, provenance, directory)
    provenance["snapshot"][-1] += 1
    with pytest.raises(ValueError, match="snapshots differ"):
        extend(base, provenance, directory)
    # Rehash an edited matrix: semantic verification must still reject it.
    path = directory / "hDeltaC_to_vDeltaE_matrix.csv"
    table = pd.read_csv(path, index_col=0)
    table.iloc[:, :] = 0
    table.to_csv(path)
    finish(directory, record)
    with pytest.raises(AssertionError):
        validate(directory)


def test_contact_matrix_distinguishes_empty_results_from_invalid_contacts():
    empty = pd.DataFrame(columns=["bodyId_pre", "bodyId_post", "weight"])
    assert contact_matrix(empty, [1, 2], [3, 4]).shape == (2, 2)
    for values in ([[1, 3, 0]], [[1, 3, np.nan]], [[1, 3, 1.5]], [[1, 5, 1]], [[1, 3, 1], [1, 3, 2]]):
        with pytest.raises(ValueError):
            contact_matrix(pd.DataFrame(values, columns=empty.columns), [1, 2], [3, 4])


def test_extraction_failure_keeps_raw_response_and_failed_manifest(anatomy, tmp_path, monkeypatch):
    pytest.importorskip("neuprint")
    import neuprint
    from fly_connectome import extension_analysis
    from fly_connectome.home_vector.hdeltac import extract
    directory, _, base, provenance = anatomy
    source, source_record = begin(tmp_path / "prepared", "home_vector_prepared_v1")
    write_json(source / "circuit.json", base.as_dict())
    write_json(source / "trajectories.json", {})
    write_json(source / "task.json", {})
    write_json(source / "sources.json", provenance)
    finish(source, source_record)
    census = tmp_path / "census"
    census.mkdir()
    meta = dict(uuid="example", latestMutationId=1, lastDatabaseEdit="example")
    cm = dict(server=provenance["snapshot"][0], dataset=provenance["snapshot"][1], dataset_metadata=meta)
    write_json(census / "manifest.json", cm)
    neurons = pd.read_csv(directory / "neurons.csv")
    monkeypatch.setattr(extension_analysis, "validate_saved_census", lambda path: (cm, neurons, None, None))
    monkeypatch.setattr(neuprint, "fetch_meta", lambda **kwargs: meta)

    class Client:
        calls = 0

        def fetch_custom(self, query):
            self.calls += 1
            if self.calls == 1:
                return neurons
            if self.calls == 2:
                return pd.read_csv(directory / "incoming_connections.csv")
            raise RuntimeError("simulated query failure")

    output = tmp_path / "failed-extraction"
    with pytest.raises(RuntimeError):
        extract(Client(), source, census, output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert (output / "incoming_connections.csv").exists()
    assert not list(output.glob("*_matrix.csv"))
    with pytest.raises(ValueError, match="complete"):
        validate(output)


def test_four_population_dynamics_gradients_and_ablation(anatomy):
    torch = pytest.importorskip("torch")
    from fly_connectome.home_vector.model import HomeVectorNetwork, objective
    directory, _, base, provenance = anatomy
    expanded, _ = extend(base, provenance, directory)
    model = HomeVectorNetwork(expanded, seed=12, dtype=torch.float64)
    assert model.bias.shape == model.tau.shape == (4,)
    assert model.D.shape == (2, 65)
    assert HomeVectorNetwork(base).bias.shape == (3,)
    with torch.no_grad():
        model.a.fill_(.2)
        model.D.fill_(.1)
    pfn = np.ones((20, 60))
    prediction, rates = model(torch.tensor(pfn)[None], .0025)
    expected = numpy_rollout(expanded, pfn, .0025, model.a.detach().numpy(),
                             model.bias.detach().numpy(), model.tau.detach().numpy())
    np.testing.assert_allclose(rates[0].detach().numpy(), expected, atol=1e-14)
    loss, _ = objective(prediction, torch.ones_like(prediction), rates,
                         torch.ones((1, 20)), 1., model)
    loss.backward()
    assert model.tau.grad[3].abs() > 0
    assert model.bias.grad[3].abs() > 0
    assert model.a.grad[1075:].abs().max() > 0
    assert model.D.grad[:, 45:].abs().max() > 0
    model.recurrence = False
    assert model.weights()[60:].abs().sum() == 0
    assert model.weights()[:60, 45:].abs().sum() > 0
