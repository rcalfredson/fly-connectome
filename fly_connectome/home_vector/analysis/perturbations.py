"""Acute state reset and recurrence removal, without fitting parameters."""

import numpy as np
import torch

from ..__main__ import batch, metrics
from ..data import write_json
from ..model import HomeVectorNetwork


def pause_pair(rows, family):
    indices = [i for i, row in enumerate(rows) if row["family"] == family]
    if len(indices) != 2:
        raise ValueError("Require exactly two challenge trajectories in the selected pause family.")
    records = [rows[i] for i in indices]
    a, b = records
    if (a["kind"] != "matched_pause" or b["kind"] != "matched_pause"
            or a["suffix_start_s"] != b["suffix_start_s"]):
        raise ValueError("Require matched pause timing and trajectory kinds.")
    np.testing.assert_array_equal(np.asarray(a["protocol"])[1:], np.asarray(b["protocol"])[1:])
    suffix = np.asarray(a["protocol"])[1:]
    if (suffix[0, 0] != a["suffix_start_s"] or (suffix[:, 2:] != 0).any()
            or suffix[-1, 0] - suffix[0, 0] < 1):
        raise ValueError("Require at least one second of identical stationary pause inputs.")
    return indices, records


def analyze(audit):
    torch.set_num_threads(audit.args.threads)
    audit.record["torch"] = torch.__version__
    circuit, groups, task = audit.circuit, audit.groups, audit.task
    checkpoint = torch.load(audit.run / "best.pt", map_location="cpu", weights_only=True)
    network = HomeVectorNetwork(circuit, seed=audit.manifest["settings"]["seed"],
                                recurrence=audit.manifest["settings"]["recurrence"])
    network.load_state_dict(checkpoint["model"])
    if not network.recurrence:
        raise ValueError("Acute recurrence-removal audit requires a recurrent training run.")
    indices, records = pause_pair(groups["challenge"], audit.args.family)
    x, y, mask = batch(circuit, records, task["dt_s"], "cpu", torch.float32)
    if not mask.all():
        raise ValueError("Require matched trajectory lengths.")
    reset_at = round(records[0]["suffix_start_s"] / task["dt_s"])
    saved, _ = audit.predictions("challenge")
    report = {}
    with torch.no_grad():
        for condition in ("unchanged", "reset_at_pause_start", "acute_recurrence_removal"):
            network.recurrence = condition != "acute_recurrence_removal"
            prediction, _ = network(x, task["dt_s"],
                                    reset_at=reset_at if condition == "reset_at_pause_start" else None)
            p, truth = prediction.numpy(), y.numpy()
            count = round(1 / task["dt_s"]) + 1
            report[condition] = dict(
                late_pause=metrics([p[0, -count:], p[1, -count:]],
                                   [truth[0, -count:], truth[1, -count:]]),
                terminal_predictions_mm=p[:, -1].tolist(),
                terminal_prediction_separation_mm=float(np.linalg.norm(p[0, -1] - p[1, -1])))
            if condition == "unchanged":
                error = max(float(np.max(np.abs(p[i] - saved[j]))) for i, j in enumerate(indices))
                report[condition]["cpu_vs_saved_gpu_max_abs_prediction_error_mm"] = error
                if error >= .002:
                    raise ValueError("CPU replay differs materially from saved predictions.")
    audit.record.update(parameters_retrained=False, selected_family=audit.args.family,
                        selected_indices=indices, reset_at_sample=reset_at,
                        note="Acute perturbations; these are not separately retrained controls.")
    write_json(audit.out / "perturbations.json", report)
