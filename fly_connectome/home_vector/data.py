"""Audited circuit boundary and compact trajectory families. No neural training."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

TYPES = ("hDeltaB", "vDeltaE", "hDeltaG")
SUPPORTED_TYPES = TYPES + ("hDeltaC",)
OMISSIONS = ["PFNd/PFNv -> hDeltaG (unqueried)", "all edges into clamped PFNs",
             "all other populations and external boundary inputs"]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_bundle(source):
    source = Path(source)
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("status") != "complete" or not manifest.get("artifacts"):
        raise ValueError("Require a complete home-vector bundle.")
    for name, expected in manifest["artifacts"].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source.resolve()) or digest(path) != expected:
            raise ValueError(f"Bundle hash mismatch: {name}")
    return manifest


@dataclass
class Circuit:
    input_ids: list
    input_types: list
    dynamic_ids: list
    dynamic_types: list
    preferred_heading_deg: list
    preferred_translation_deg: list
    edges: list

    def __post_init__(self):
        ids = self.input_ids + self.dynamic_ids
        if not self.input_ids or not self.dynamic_ids or len(set(ids)) != len(ids):
            raise ValueError("Circuit IDs must be nonempty and unique.")
        if (len(self.input_types) != len(self.input_ids)
                or len(self.dynamic_types) != len(self.dynamic_ids)
                or not set(self.dynamic_types) <= set(SUPPORTED_TYPES)):
            raise ValueError("Invalid type/identity alignment.")
        for angles in [self.preferred_heading_deg, self.preferred_translation_deg]:
            if len(angles) != len(self.input_ids) or not np.isfinite(angles).all():
                raise ValueError("Every PFN requires finite prescribed angles.")
        seen = set()
        for edge in self.edges:
            i, j = edge["pre"], edge["post"]
            if (not 0 <= i < len(ids) or not 0 <= j < len(self.dynamic_ids)
                    or (i, j) in seen or edge["contacts"] <= 0
                    or edge["contacts"] % 1 or edge["denominator"] <= 0
                    or not np.isfinite([edge["contacts"], edge["denominator"]]).all()):
                raise ValueError("Invalid positive edge or duplicate pair.")
            seen.add((i, j))

    def as_dict(self):
        return dict(vars(self))

    @property
    def population_types(self):
        """Stable intrinsic-parameter order; old three-population checkpoints retain their shapes."""
        return tuple(t for t in SUPPORTED_TYPES if t in self.dynamic_types)


def assemble(movement, census, inputs, hdeltac=None):
    """Verify existing source formats before selecting a fixed circuit boundary."""
    import pandas as pd
    from fly_connectome.extension_analysis import validate_saved_census
    from fly_connectome.vdeltae_inputs import verify
    from fly_connectome.rate_model import verify_model_data

    movement, census, inputs = map(Path, [movement, census, inputs])
    manifests = [census / "manifest.json", inputs / "manifest.json",
                 movement / "movement_model.json"]
    before = {str(p.resolve()): digest(p) for p in manifests}
    cm, neurons, _, _ = validate_saved_census(census)
    im = verify(inputs)
    mm = json.loads(manifests[2].read_text())
    if mm.get("status") != "complete":
        raise ValueError("Movement source is incomplete.")
    verify_model_data(movement, mm)
    snapshots = [(m.get("server"), m.get("dataset"),
                  m.get("dataset_metadata", {}).get("uuid"),
                  m.get("dataset_metadata", {}).get("latestMutationId")) for m in [cm, im, mm]]
    if len(set(snapshots)) != 1 or any(x is None for x in snapshots[0]):
        raise ValueError("Source dataset snapshots do not match.")
    config = json.loads((movement / "movement_config.json").read_text())
    mapping = pd.read_csv(movement / "PFN_encoding_map.csv").set_index("bodyId_pre")
    input_ids = config["source_ids"]
    if set(mapping.index) != set(input_ids) or not mapping.index.is_unique:
        raise ValueError("PFN map and saved input identities disagree.")
    mapping = mapping.loc[input_ids]
    populations = {t: sorted(neurons.loc[neurons.type.eq(t), "bodyId"].astype(int)) for t in TYPES}
    if [len(populations[t]) for t in TYPES] != [19, 18, 8]:
        raise ValueError("Expected reviewed 19/18/8 dynamic populations.")
    if set(config["target_ids"]) != set(populations["hDeltaB"]):
        raise ValueError("Movement and census hDeltaB identities disagree.")
    overlap = pd.read_csv(inputs / "modeled_neurons.csv")
    expected = dict(zip(input_ids, mapping.type))
    expected.update({i: "hDeltaB" for i in populations["hDeltaB"]})
    if dict(zip(overlap.bodyId, overlap.type)) != expected or overlap.bodyId.duplicated().any():
        raise ValueError("Input survey and movement source identities disagree.")
    dynamic_ids = sum((populations[t] for t in TYPES), [])
    dynamic_types = sum(([t] * len(populations[t]) for t in TYPES), [])
    pre_index = {i: n for n, i in enumerate(input_ids + dynamic_ids)}
    post_index = {i: n for n, i in enumerate(dynamic_ids)}
    edges, projections, source_files = [], [], set(manifests)

    def add(path, pre_ids, post_ids, source_type, target_type):
        table = pd.read_csv(path, index_col=0)
        table.index = table.index.astype(int)
        table.columns = table.columns.astype(int)
        # The combined PFN->hDeltaB matrix is explicitly subset by source type.
        if (not table.index.is_unique or not table.columns.is_unique
                or not set(pre_ids) <= set(table.index) or set(post_ids) != set(table.columns)):
            raise ValueError(f"Matrix identities disagree: {path}")
        values = table.loc[pre_ids, post_ids].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or (values % 1).any():
            raise ValueError(f"Invalid contact counts: {path}")
        denominator = float(values.sum() / len(post_ids))
        for i, j in zip(*np.nonzero(values)):
            edges.append(dict(pre=pre_index[pre_ids[i]], post=post_index[post_ids[j]],
                              contacts=int(values[i, j]), denominator=denominator,
                              source_type=source_type, target_type=target_type))
        projections.append(dict(source_type=source_type, target_type=target_type,
                                contacts=int(values.sum()), positive_pairs=int((values > 0).sum()),
                                denominator=denominator, source_file=str(path.resolve())))
        source_files.add(path)

    for a in TYPES:
        for b in TYPES:
            add(census / f"{a}_to_{b}_matrix.csv", populations[a], populations[b], a, b)
    for a in ["PFNd", "PFNv"]:
        selected = [i for i in input_ids if mapping.loc[i, "type"] == a]
        add(movement / "raw_contact_matrix.csv", selected, populations["hDeltaB"], a, "hDeltaB")
        add(inputs / f"{a}_to_vDeltaE_matrix.csv", selected, populations["vDeltaE"], a, "vDeltaE")
    source_files.update([movement / "PFN_encoding_map.csv", movement / "movement_config.json",
                         movement / "effective_weight_matrix.csv",
                         census / "candidate_neurons.csv", inputs / "modeled_neurons.csv"])
    hashes = {str(p.resolve()): digest(p) for p in source_files}
    if any(digest(p) != h for p, h in before.items()):
        raise ValueError("Source manifest changed during assembly.")
    # Check source artifact hashes again after reading selected matrices.
    validate_saved_census(census)
    verify(inputs)
    verify_model_data(movement, mm)
    circuit = Circuit(input_ids, mapping.type.tolist(), dynamic_ids, dynamic_types,
                      mapping.preferred_heading_deg.tolist(), mapping.preferred_translation_deg.tolist(), edges)
    if len(circuit.input_ids) != 60 or len(edges) != 1075:
        raise ValueError("Circuit differs from reviewed 60-input/1075-edge boundary.")
    provenance = dict(source_sha256=hashes, snapshot=snapshots[0],
                      projections=projections, omitted=list(OMISSIONS))
    if hdeltac is not None:
        from .hdeltac import extend
        return extend(circuit, provenance, hdeltac)
    return circuit, provenance


def encode(circuit, controls):
    """Same fixed algebraic encoding as movement.encode_pfn; no geometry input."""
    controls = np.asarray(controls, dtype=float)
    if controls.ndim != 2 or controls.shape[1] != 3 or not np.isfinite(controls).all():
        raise ValueError("Controls must be finite [heading, forward, right] samples.")
    heading = np.deg2rad(controls[:, 0, None] - np.array(circuit.preferred_heading_deg))
    phi = np.deg2rad(circuit.preferred_translation_deg)
    projection = controls[:, 1, None] * np.cos(phi) + controls[:, 2, None] * np.sin(phi)
    return .5 * (1 + np.cos(heading)) * (1 + 5 * np.maximum(0, projection))


def sample_protocol(protocol, dt):
    """External task layer: sample held movement, integrate independent target."""
    protocol = np.asarray(protocol, dtype=float)
    if (protocol.ndim != 2 or protocol.shape[1] != 4 or len(protocol) < 2
            or not np.isfinite(protocol).all() or not np.isfinite(dt) or dt <= 0
            or protocol[0, 0] != 0 or (np.diff(protocol[:, 0]) <= 0).any()):
        raise ValueError("Invalid timestamped movement protocol.")
    boundaries = protocol[:, 0] / dt
    if not np.allclose(boundaries, np.round(boundaries), rtol=0, atol=1e-7):
        raise ValueError("Movement boundaries must fall on the simulation grid.")
    ticks = np.arange(int(round(boundaries[-1])) + 1)
    segment = np.searchsorted(np.round(boundaries).astype(int), ticks, side="right") - 1
    controls = protocol[segment, 1:]
    angle = np.deg2rad(controls[:, 0])
    vf, vr = controls[:, 1], controls[:, 2]
    velocity = np.column_stack((vf * np.cos(angle) - vr * np.sin(angle),
                                vf * np.sin(angle) + vr * np.cos(angle)))
    target = -np.vstack((np.zeros(2), np.cumsum(velocity[:-1] * dt, axis=0)))
    return controls, target


def datasets(seed=20261030, sizes=None):
    """Generate independent families per split; matched siblings never split."""
    sizes = sizes or dict(train=1024, validation=128, test=256)
    if set(sizes) != {"train", "validation", "test"} or any(
            type(n) is not int or n < 2 for n in sizes.values()):
        raise ValueError("Provide >=2 trajectories in each of the three splits.")
    result = {}
    for split, stream in zip(["train", "validation", "test"], np.random.SeedSequence(seed).spawn(3)):
        rng, records, family = np.random.default_rng(stream), [], 0
        while len(records) < sizes[split]:
            group = f"{split}-family-{family}"
            matched = family % 2 == 0 and len(records) + 2 <= sizes[split]
            if matched:
                duration = float(rng.integers(2, 9) / 4)
                theta, suffix_heading = rng.uniform(-180, 180, 2)
                speed = float(rng.uniform(.5, 2))
                # Identical five-second pause at the same heading and clock time.
                for sign in [-1, 1]:
                    protocol = [[0., float(theta), sign * speed, 0.],
                                [duration, float(suffix_heading), 0., 0.],
                                [duration + 5, float(suffix_heading), 0., 0.]]
                    records.append(dict(family=group, kind="matched_pause", suffix_start_s=duration,
                                        protocol=protocol))
            else:
                duration = int(rng.integers(16, 65)) / 4
                times, t = [0.], 0.
                while t < duration:
                    t = min(duration, t + int(rng.integers(1, 9)) / 4)
                    times.append(t)
                protocol = []
                for t in times:
                    theta, direction = rng.uniform(-180, 180, 2)
                    speed = 0. if rng.random() < .2 else rng.uniform(0, 2)
                    protocol.append([t, float(theta), float(speed * np.cos(np.deg2rad(direction))),
                                     float(speed * np.sin(np.deg2rad(direction)))])
                records.append(dict(family=group, kind="random", protocol=protocol))
            family += 1
        result[split] = records
    # Deliberately independent, named challenge protocols; no fitting on these.
    challenges = []
    for pause in [1, 5, 10]:
        for sign in [-1, 1]:
            challenges.append(dict(family=f"challenge-pause-{pause}", kind="matched_pause",
                suffix_start_s=2., protocol=[[0, 0, sign, 0], [2, 90, 0, 0], [2 + pause, 90, 0, 0]]))
    for duration in [32, 64]:
        challenges.append(dict(family=f"challenge-long-{duration}", kind="long",
                               protocol=[[0, 45, 1, 0], [duration, 45, 1, 0]]))
    challenges.extend([
        dict(family="challenge-square", kind="closed_loop", protocol=[
            [0, 0, 1, 0], [2, 90, 1, 0], [4, 180, 1, 0], [6, 270, 1, 0], [8, 270, 0, 0]]),
        dict(family="challenge-reversal", kind="reversal", protocol=[
            [0, 0, 1, 0], [2, 0, -1, 0], [4, 0, 0, 0], [9, 0, 0, 0]]),
        dict(family="challenge-world-equivalent", kind="forward", protocol=[
            [0, 0, 1, 0], [4, 0, 0, 0], [9, 0, 0, 0]]),
        dict(family="challenge-world-equivalent", kind="sideways", protocol=[
            [0, 90, 0, -1], [4, 0, 0, 0], [9, 0, 0, 0]])])
    result["challenge"] = challenges
    return result


def target_scale(records, dt):
    energy = [np.mean(np.sum(sample_protocol(r["protocol"], dt)[1] ** 2, axis=1)) for r in records]
    scale = float(np.sqrt(np.mean(energy)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Training target scale must be positive.")
    return scale


def rewire(circuit, seed, swaps_per_edge=5):
    """Directed blockwise degree null; self loops fixed, count multiset shuffled."""
    import copy
    other = copy.deepcopy(circuit)
    rng, offset = np.random.default_rng(seed), len(circuit.input_ids)
    diagnostics = []
    blocks = {}
    for n, e in enumerate(other.edges):
        blocks.setdefault((e["source_type"], e["target_type"]), []).append(n)
    for block, positions in blocks.items():
        movable = [n for n in positions if other.edges[n]["pre"] != offset + other.edges[n]["post"]]
        pairs = {(e["pre"], e["post"]) for e in other.edges}
        original = {(other.edges[n]["pre"], other.edges[n]["post"]) for n in positions}
        accepted, attempted = 0, 0
        goal = swaps_per_edge * len(movable)
        while len(movable) >= 2 and accepted < goal and attempted < max(100, goal * 50):
            attempted += 1
            x, y = rng.choice(movable, 2, replace=False)
            a, b = other.edges[x], other.edges[y]
            new_a, new_b = (a["pre"], b["post"]), (b["pre"], a["post"])
            if (a["pre"] == b["pre"] or a["post"] == b["post"]
                    or new_a in pairs or new_b in pairs
                    or new_a[0] == offset + new_a[1] or new_b[0] == offset + new_b[1]):
                continue
            pairs.remove((a["pre"], a["post"]))
            pairs.remove((b["pre"], b["post"]))
            a["post"], b["post"] = b["post"], a["post"]
            pairs.update([new_a, new_b])
            accepted += 1
        counts = rng.permutation([other.edges[n]["contacts"] for n in positions])
        for n, count in zip(positions, counts):
            other.edges[n]["contacts"] = int(count)
        retained = sum((other.edges[n]["pre"], other.edges[n]["post"]) in original for n in positions)
        diagnostics.append(dict(block=list(block), attempted=attempted, accepted=accepted,
                                retained_fraction=retained / len(positions)))
    return Circuit(**other.as_dict()), diagnostics
