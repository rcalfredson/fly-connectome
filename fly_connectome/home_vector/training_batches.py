"""Partition an already sampled batch; preserve equal-trajectory gradient weights."""

from contextlib import nullcontext
import math


def length_groups(records, dt, max_ratio=2.):
    if not records or not math.isfinite(dt) or dt <= 0 or not math.isfinite(max_ratio) or max_ratio < 1:
        raise ValueError("Require records, positive dt and a finite length ratio >=1.")
    lengths = [round(r["protocol"][-1][0] / dt) + 1 for r in records]
    if min(lengths) < 2:
        raise ValueError("Require at least two samples per trajectory.")
    groups = []
    for index in sorted(range(len(records)), key=lambda i: lengths[i]):
        if not groups or lengths[index] > max_ratio * lengths[groups[-1][0]]:
            groups.append([])
        groups[-1].append(index)
    executed = sum(len(group) * max(lengths[i] for i in group) for group in groups)
    padded, valid = len(records) * max(lengths), sum(lengths)
    return groups, dict(groups=groups, subgroup_count=len(groups), valid_steps=valid,
                        original_padded_steps=padded, executed_steps=executed,
                        original_padding_fraction=1 - valid / padded,
                        remaining_padding_fraction=1 - valid / executed,
                        avoided_fraction=1 - executed / padded)


def backward_grouped(network, circuit, records, dt, device, scale, max_ratio=2., section=None):
    """Accumulate one sampled batch's gradient; caller clips/steps exactly once.

    Data terms are weighted by subgroup_size / original_batch_size. Parameter
    penalties are backpropagated once, separately from trajectory-dependent
    terms. No state is detached within any trajectory.
    """
    import torch
    from .__main__ import batch
    from .model import LOSS_WEIGHTS, objective

    section = section or (lambda name: nullcontext())
    groups, description = length_groups(records, dt, max_ratio)
    task, activity = network.a.new_zeros(()), network.a.new_zeros(())
    penalty = None
    for positions in groups:
        with section("batch_preparation_and_transfer"):
            selected = [records[i] for i in positions]
            x, target, mask = batch(circuit, selected, dt, device, network.a.dtype)
        with section("forward"):
            prediction, rates = network(x, dt)
        with section("objective_and_guard"):
            loss, terms = objective(prediction, target, rates, mask, scale, network)
            if not torch.isfinite(loss) or not torch.isfinite(rates).all() or rates.max() > 1e6:
                raise ValueError("Grouped training exceeded the numerical guard.")
            weight = len(positions) / len(records)
            data_loss = weight * (terms["task"] + LOSS_WEIGHTS["activity"] * terms["activity"])
            if penalty is None:
                penalty = LOSS_WEIGHTS["edge_prior"] * terms["edge_prior"] + LOSS_WEIGHTS["decoder"] * terms["decoder"]
                prior, decoder = terms["edge_prior"].detach(), terms["decoder"].detach()
            task = task + weight * terms["task"].detach()
            activity = activity + weight * terms["activity"].detach()
        with section("backward"):
            data_loss.backward()
        # Drop each subgroup's graph before preparing the next one.
        del x, target, mask, prediction, rates, loss, terms, data_loss
    with section("backward"):
        penalty.backward()
    return (task + LOSS_WEIGHTS["activity"] * activity + penalty.detach(),
            dict(task=task, activity=activity, edge_prior=prior, decoder=decoder), description)
