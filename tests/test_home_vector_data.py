import copy
from collections import Counter

import numpy as np
import pytest

from fly_connectome.home_vector.data import (Circuit, datasets, encode, rewire,
                                            sample_protocol, target_scale, verify_bundle, write_json, digest)


def circuit_fixture():
    return Circuit([1, 2], ["PFNd", "PFNv"], [10, 11, 12],
        ["hDeltaB", "vDeltaE", "hDeltaG"], [0., 90.], [-31., 137.], [
        dict(pre=i, post=j, contacts=c, denominator=2., source_type=a, target_type=b)
        for i, j, c, a, b in [(0, 0, 2, "PFNd", "hDeltaB"), (1, 1, 1, "PFNv", "vDeltaE"),
            (2, 1, 3, "hDeltaB", "vDeltaE"), (3, 2, 2, "vDeltaE", "hDeltaG"),
            (4, 0, 1, "hDeltaG", "hDeltaB")]])


def test_geometry_sample_alignment_and_known_closure():
    controls, target = sample_protocol([[0, 0, 1, 0], [1, 90, 1, 0],
        [2, 180, 1, 0], [3, 270, 1, 0], [4, 0, 0, 0]], .25)
    np.testing.assert_array_equal(target[0], [0, 0])
    np.testing.assert_allclose(target[4], [-1, 0], atol=1e-12)
    np.testing.assert_allclose(target[8], [-1, -1], atol=1e-12)
    np.testing.assert_allclose(target[-1], 0, atol=1e-12)
    assert controls[4, 0] == 90  # New drive, preceding interval's accumulated target.


@pytest.mark.parametrize("protocol,dt", [([[0, 0, 1, 0], [.3, 0, 0, 0]], .2),
    ([[1, 0, 0, 0], [2, 0, 0, 0]], .1), ([[0, 0, 0, 0], [0, 0, 0, 0]], .1),
    ([[0, 0, 0, 0], [1, 0, 0, 0]], 0)])
def test_invalid_protocols_rejected(protocol, dt):
    with pytest.raises(ValueError):
        sample_protocol(protocol, dt)


def test_matched_families_cannot_be_solved_from_current_inputs_or_time():
    data = datasets(123, dict(train=8, validation=4, test=4))
    assert data == datasets(123, dict(train=8, validation=4, test=4))
    families = [{r["family"] for r in data[s]} for s in ["train", "validation", "test"]]
    assert not any(families[i] & families[j] for i in range(3) for j in range(i))
    a, b = data["train"][:2]
    ca, ya = sample_protocol(a["protocol"], .0025)
    cb, yb = sample_protocol(b["protocol"], .0025)
    onset = round(a["suffix_start_s"] / .0025)
    np.testing.assert_array_equal(ca[onset:], cb[onset:])
    np.testing.assert_array_equal(encode(circuit_fixture(), ca[onset:]), encode(circuit_fixture(), cb[onset:]))
    assert np.linalg.norm(ya[-1] - yb[-1]) > 0
    assert a["family"] == b["family"]


def test_scale_equal_weights_trajectories_and_uses_only_given_training_records():
    records = [dict(protocol=[[0, 0, 1, 0], [1, 0, 0, 0]]),
               dict(protocol=[[0, 0, 1, 0], [2, 0, 0, 0]])]
    expected = np.sqrt(np.mean([np.mean(sample_protocol(r["protocol"], .5)[1] ** 2, axis=0).sum()
                               for r in records]))
    assert target_scale(records, .5) == pytest.approx(expected)


def test_encoding_matches_existing_prescription():
    pytest.importorskip("neuprint")
    import pandas as pd
    from fly_connectome.movement import MovementParameters, encode_pfn
    circuit = circuit_fixture()
    mapping = pd.DataFrame(dict(preferred_heading_deg=circuit.preferred_heading_deg,
                                preferred_translation_deg=circuit.preferred_translation_deg))
    controls = np.array([[10, 1, -.4], [130, 0, 0], [45, -1, 2]])
    expected = np.array([encode_pfn(mapping, h, f, r, 1., MovementParameters()) for h, f, r in controls])
    np.testing.assert_allclose(encode(circuit, controls), expected, atol=1e-14)


def test_bundle_hashes_reject_tampering_and_failed_status(tmp_path):
    path = tmp_path / "data.json"
    write_json(path, dict(value=1))
    write_json(tmp_path / "manifest.json", dict(status="complete", artifacts={path.name: digest(path)}))
    verify_bundle(tmp_path)
    write_json(path, dict(value=2))
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_bundle(tmp_path)
    write_json(tmp_path / "manifest.json", dict(status="failed", artifacts={path.name: digest(path)}))
    with pytest.raises(ValueError, match="complete"):
        verify_bundle(tmp_path)


def test_numpy_dynamics_match_independent_common_framework():
    pytest.importorskip("neuprint")
    import pandas as pd
    from fly_connectome.rate_framework import RateNetwork
    from fly_connectome.home_vector.dynamics import initial_coefficients, numpy_rollout
    circuit = circuit_fixture()
    ids = circuit.input_ids + circuit.dynamic_ids
    a = initial_coefficients(circuit, 12)
    nodes = pd.DataFrame(dict(neuron_id=list(map(str, ids)),
        cell_type=circuit.input_types + circuit.dynamic_types, tau_s=.1, bias=[0.] * 2 + [.1] * 3))
    edges = pd.DataFrame([dict(source_id=str(ids[e["pre"]]),
        target_id=str(circuit.dynamic_ids[e["post"]]), effective_weight=v * e["contacts"] / e["denominator"],
        channel="rate", sign_evidence="assumed", sign_detail="Synthetic parity test") for e, v in zip(circuit.edges, a)])
    pfn = np.tile([1., 2.], (101, 1))
    expected = RateNetwork(nodes, edges, rate_unit="arbitrary_rate_units").solve(
        np.zeros((101, 5)), .0025, prescribed={str(i): pfn[:, k] for k, i in enumerate(circuit.input_ids)},
        method="exponential_euler")
    actual = numpy_rollout(circuit, pfn, .0025, a, [.1] * 3, [.1] * 3)
    np.testing.assert_allclose(actual, expected.rates[:, 2:], atol=1e-14)


def test_rewire_preserves_blockwise_degrees_counts_and_selfloops():
    ids = list(range(10, 18))
    edges = [dict(pre=i + 1, post=j, contacts=1 + i + j, denominator=4.,
                  source_type="hDeltaB", target_type="hDeltaB")
             for i in range(8) for j in [(i + 1) % 8, (i + 3) % 8]]
    original = Circuit([1], ["PFNd"], ids, ["hDeltaB"] * 8, [0], [-31], edges)
    saved = copy.deepcopy(original.as_dict())
    other, diagnostics = rewire(original, 42)
    assert original.as_dict() == saved
    assert Counter(e["pre"] for e in edges) == Counter(e["pre"] for e in other.edges)
    assert Counter(e["post"] for e in edges) == Counter(e["post"] for e in other.edges)
    assert sorted(e["contacts"] for e in edges) == sorted(e["contacts"] for e in other.edges)
    assert all(e["pre"] != 1 + e["post"] for e in other.edges)
    assert len({(e["pre"], e["post"]) for e in other.edges}) == len(edges)
    assert diagnostics[0]["accepted"] > 0
    assert diagnostics[0]["retained_fraction"] < 1
