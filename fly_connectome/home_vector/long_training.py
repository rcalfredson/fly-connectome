"""Augment a verified prepared task with long walks; preserve anatomy and loss scale."""

import argparse
import copy
from itertools import product
import json
from pathlib import Path
import shutil

import numpy as np

from .__main__ import begin, fail, finish, load
from .data import digest, verify_bundle, write_json


SPEEDS = (.25, .5, 1., 2.)
HEADINGS = tuple(range(0, 360, 45))
DIRECTIONS = (0, 45, 90, 127, 135, 180, 225, 233, 270, 315)
TRAIN_DURATIONS = (16, 32, 48, 64)
EXTRAPOLATION_DURATIONS = (96, 128)


def split_controls(seed):
    """Eight training, one validation and one test direction per speed/heading.

    The historical 45-degree forward challenge stays out of long-walk training.
    Splits group complete controls, not timestamps or shorter prefixes.
    """
    rng = np.random.default_rng(seed)
    result = dict(train=[], validation=[], test=[])
    for speed, heading in product(SPEEDS, HEADINGS):
        directions = list(map(int, rng.permutation(DIRECTIONS)))
        if speed == 1 and heading == 45:
            where = directions.index(0)
            directions[1], directions[where] = directions[where], directions[1]
        result["validation"].append((speed, heading, directions[0]))
        result["test"].append((speed, heading, directions[1]))
        result["train"].extend((speed, heading, direction) for direction in directions[2:])
    return result


def long_row(control, duration, split):
    speed, heading, direction = control
    angle = np.deg2rad(direction)
    forward, right = float(speed * np.cos(angle)), float(speed * np.sin(angle))
    return dict(family=f"long-{split}-s{speed:g}-h{heading:g}-b{direction:g}",
                kind="long_training" if split == "train" else "long_holdout",
                speed_mm_s=float(speed), heading_deg=float(heading), body_direction_deg=float(direction),
                protocol=[[0., float(heading), forward, right],
                          [float(duration), float(heading), forward, right]])


def augment(groups, seed):
    if set(groups) != {"train", "validation", "test", "challenge"}:
        raise ValueError("Require an unaugmented original prepared task with four standard splits.")
    if len(groups["train"]) != 1024:
        raise ValueError("The controlled 50:50 mixture requires the original 1024 training trajectories.")
    result = copy.deepcopy(groups)
    controls = split_controls(seed)
    for split, durations in [("train", TRAIN_DURATIONS), ("validation", (24, 56)), ("test", (32, 64))]:
        result[split].extend(long_row(control, duration, split)
                             for control in controls[split] for duration in durations)
    # Retain all historical challenges unchanged. This extra split is never
    # visited by train(), including its checkpoint-selection and final metrics.
    seen = controls["train"][::8]  # One seen control per speed/heading stratum.
    result["extrapolation"] = [
        dict(long_row(control, duration, f"extrapolation-{status}"), control_status=status)
        for status, selected in [("seen", seen), ("held_out", controls["test"])]
        for control in selected for duration in EXTRAPOLATION_DURATIONS]
    return result, controls


def prepare(source, output, seed=20261104):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Output must be separate from the original prepared bundle.")
    manifest = verify_bundle(source)
    if manifest.get("kind") != "home_vector_prepared_v1":
        raise ValueError("Require an original home-vector prepared bundle.")
    _, groups, task = load(source)
    hashes = {str(p): digest(p) for p in source.iterdir() if p.is_file()}
    result, controls = augment(groups, seed)
    output, record = begin(output, "home_vector_prepared_v1", seed=seed,
        source=str(source), source_manifest_sha256=digest(source / "manifest.json"),
        dataset_variant="long_straight_v1", circuit_variant=manifest["settings"].get("circuit_variant", "core"))
    try:
        for name in ("circuit.json", "sources.json", "legacy_weights.npy"):
            shutil.copyfile(source / name, output / name)
        write_json(output / "trajectories.json", result)
        task = copy.deepcopy(task)
        task["family_counts"] = {s: len({r["family"] for r in rows}) for s, rows in result.items()}
        task["trajectory_counts"] = {s: len(rows) for s, rows in result.items()}
        task["length_scale_policy"] = "Inherited original training-only scale; not recomputed after augmentation."
        write_json(output / "task.json", task)
        write_json(output / "long_training_design.json", dict(
            seed=seed, speeds_mm_s=SPEEDS, headings_deg=HEADINGS, body_directions_deg=DIRECTIONS,
            control_splits={s: [list(c) for c in rows] for s, rows in controls.items()},
            train_durations_s=TRAIN_DURATIONS, validation_durations_s=[24, 56], test_durations_s=[32, 64],
            extrapolation_durations_s=EXTRAPOLATION_DURATIONS,
            counts={s: len(rows) for s, rows in result.items()},
            old_counts={s: len(rows) for s, rows in groups.items()},
            long_training_sampling_probability=.5,
            selection="Original validation plus 64 long validation paths; extrapolation excluded.",
            initialization="Train from the same seed, not from a fitted checkpoint.",
            length_scale_mm=task["length_scale_mm"], source_sha256=hashes))
        if hashes != {str(p): digest(p) for p in source.iterdir() if p.is_file()}:
            raise ValueError("Source files changed during preparation.")
        finish(output, record)
    except Exception as error:
        fail(output, record, error)
        raise
    return dict(output=str(output), trajectories={s: len(rows) for s, rows in result.items()},
                length_scale_mm=task["length_scale_mm"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Original verified prepared task; core or hDeltaC")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.source, args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
