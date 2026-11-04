"""Two equivalent recurrent loops with different gradient bookkeeping.

Run this file directly in an environment with PyTorch installed:
    python examples/home_vector_gradient_assembly.py

No project imports, data files or GPU are needed. Both loops retain the whole
history; neither detaches tensors or truncates backpropagation.

Conceptually, selecting time 2 from four scalar time steps sends back a full
buffer [0, 0, g2, 0]. Selecting the other times creates similar buffers. They
are added together to obtain [g0, g1, g2, g3]. Unbind receives the separate
slice gradients together and assembles that result once. The recurrence and
the mathematical derivative are identical.
"""

import argparse
from collections import Counter

import torch


def indexed_loop(drive):
    """Before: each tensor selection gets its own backward operation."""
    state = torch.zeros_like(drive[0], requires_grad=False)
    for t in range(len(drive)):
        current_input = drive[t]  # Tensor indexing: SelectBackward for each t.
        state = 0.8 * state + current_input
    return state


def unbound_loop(drive):
    """After: expose all time slices together, then index/iterate a Python tuple."""
    time_slices = drive.unbind(dim=0)
    state = torch.zeros_like(time_slices[0], requires_grad=False)
    for current_input in time_slices:
        state = 0.8 * state + current_input
    return state


def graph_counts(output):
    """Count actual autograd nodes, without invoking the heavyweight profiler."""
    pending, seen, counts = [output.grad_fn], set(), Counter()
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        counts[type(node).__name__] += 1
        pending.extend(child for child, _ in node.next_functions if child is not None)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    if not 2 <= args.steps <= 1000:
        parser.error("Use 2 to 1000 steps; this example is for understanding, not timing.")

    # Movement is fixed. The upstream gain is learnable, like a simplified
    # input-edge weight. Gradients must travel through drive to that gain.
    movement = torch.arange(1, args.steps + 1, dtype=torch.float64)[:, None] / 10
    results = []
    for loop in [indexed_loop, unbound_loop]:
        gain = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
        drive = movement * gain
        drive.retain_grad()  # Only to display its gradient; not needed to learn gain.
        final_state = loop(drive)
        counts = graph_counts(final_state)
        loss = final_state.sum()  # Simple teaching objective, not the project loss.
        loss.backward()
        print(f"\n{loop.__name__}")
        print(f"  final state:          {final_state.detach().tolist()}")
        print(f"  d(loss)/d(drive[t]):  {drive.grad[:, 0].tolist()}")
        print(f"  d(loss)/d(gain):      {gain.grad.item():.12f}")
        print(f"  SelectBackward nodes: {counts['SelectBackward0']}")
        print(f"  UnbindBackward nodes: {counts['UnbindBackward0']}")
        results.append((final_state.detach(), drive.grad.clone(), gain.grad.clone()))

    for before, after in zip(*results):
        torch.testing.assert_close(before, after, atol=1e-12, rtol=1e-12)
    print("\nSame result and gradients; different gradient assembly.")
    print("Each drive[t] selection's backward can create a full-size zero gradient")
    print("and place that slice's gradient into it. Unbind assembles the slices together.")
    print("Analytic gradient at step t: 0.8 ** (steps - 1 - t).")
    print("The real model uses [batch, time, neuron] tensors and unbind(dim=1).")


if __name__ == "__main__":
    main()
