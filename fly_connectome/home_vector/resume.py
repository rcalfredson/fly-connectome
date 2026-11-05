"""Verified continuation of complete training bundles without restarting Adam."""

import json
from pathlib import Path

import numpy as np

from .data import digest, verify_bundle


def checkpoint(network, optimizer, update, rng):
    import torch
    state = dict(model=network.state_dict(), optimizer=optimizer.state_dict(), update=update,
                 numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state())
    if network.a.is_cuda:
        state["torch_cuda_rng"] = torch.cuda.get_rng_state_all()
    return state


def load_resume(source, prepared, settings, task, plan, change, network):
    """Check compatibility before the caller creates a new output directory."""
    import torch
    from .model import LOSS_WEIGHTS

    source = Path(source).resolve()
    manifest = verify_bundle(source)
    if manifest.get("kind") != "home_vector_training_v1":
        raise ValueError("Resume requires a complete home-vector training bundle.")
    parent = manifest["settings"]
    if parent.get("source_manifest_sha256") != digest(Path(prepared) / "manifest.json"):
        raise ValueError("Resume prepared dataset does not match the parent run.")
    defaults = dict(batching="padded", length_ratio=2., compile_block=0, loss_mode="original")
    for key, value in settings.items():
        if parent.get(key, defaults.get(key)) != value:
            raise ValueError(f"Resume setting mismatch: {key}; retain the parent training recipe.")
    if manifest.get("torch") != torch.__version__:
        raise ValueError("Resume requires the same PyTorch version as the parent run.")
    config = json.loads((source / "training_config.json").read_text())
    expected = dict(optimizer="Adam", learning_rate=.001, betas=[.9, .999], eps=1e-8,
                    gradient_norm_clip=1., dtype="float32", dt_s=task["dt_s"],
                    length_scale_mm=task["length_scale_mm"],
                    bounds=dict(edge_coefficient=[-5, 5], bias=[-2, 2], tau_s=[.05, .5]))
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Resume training configuration mismatch: {key}.")
    loss = config["loss"]
    for key, value in plan.items():
        # Older original-mode runs did not record an explicit mode/selection name.
        fallback = {"mode": "original", "selection_metric": "equal_trajectory_scaled_mse"}.get(key)
        if loss.get(key, fallback) != value:
            raise ValueError(f"Resume absolute-loss configuration mismatch: {key}.")
    for key, value in dict(activity=LOSS_WEIGHTS["activity"],
                           edge_initialization_prior=LOSS_WEIGHTS["edge_prior"],
                           decoder_weights=LOSS_WEIGHTS["decoder"]).items():
        if loss.get(key) != value:
            raise ValueError(f"Resume loss penalty mismatch: {key}.")
    old_change = loss.get("vector_change", dict(enabled=False, weight=0.))
    if old_change["enabled"] != change["enabled"] or (change["enabled"] and old_change != change):
        raise ValueError("Resume vector-change configuration mismatch; retain the parent objective.")

    # Load CPU tensors even for CUDA training: CPU RNG state must remain on CPU.
    last = torch.load(source / "last.pt", weights_only=False, map_location="cpu")
    if not {"model", "optimizer", "update", "numpy_rng", "torch_rng"} <= last.keys():
        raise ValueError("Resume checkpoint lacks optimizer or random-number state.")
    history = json.loads((source / "history.json").read_text())
    start = last["update"]
    if type(start) is not int or start < 1 or [r["update"] for r in history] != list(range(1, start + 1)):
        raise ValueError("Resume history does not match the last checkpoint update.")
    # Circuit identity and initialization buffers must not silently change.
    for key in ("pre", "post", "counts", "type_index", "a_initial"):
        if not torch.equal(network.state_dict()[key].cpu(), last["model"][key]):
            raise ValueError(f"Resume circuit or initialization mismatch: {key}.")
    network.load_state_dict(last["model"], strict=True)

    validations = [r for r in history if "validation_scaled_mse" in r]
    best, best_state = float("inf"), None
    if validations:
        score = lambda r: r.get("validation_selection_score", r["validation_scaled_mse"])
        selected = min(validations, key=score)
        best = score(selected)
        best_state = torch.load(source / "best.pt", weights_only=False, map_location="cpu")
        if not np.isfinite(best) or best_state["update"] != selected["update"]:
            raise ValueError("Resume best checkpoint does not match validation history.")
    metadata = dict(resume_source=str(source), resume_manifest_sha256=digest(source / "manifest.json"),
                    resume_checkpoint_sha256=digest(source / "last.pt"), starting_update=start,
                    cuda_rng_restored="torch_cuda_rng" in last)
    return dict(last=last, history=history, best=best, best_state=best_state, metadata=metadata)


def restore_rng(state, rng):
    import torch
    rng.bit_generator.state = state["numpy_rng"]
    torch.set_rng_state(state["torch_rng"])
    if "torch_cuda_rng" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_rng"])
