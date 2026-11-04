"""Freestanding CPU/GPU batch-size experiment for a small recurrent model.

Requires only PyTorch and the Python standard library. Run, for example:
    python examples/recurrent_batch_benchmark.py --devices cpu cuda --output /tmp/batches.json

The loop stays sequential in TIME. BATCH consists of independent trajectories
that can be processed together. Larger batches change optimization statistics;
better throughput is not proof of faster convergence to the same solution.
"""

import argparse
import copy
import json
import math
from pathlib import Path
import platform
import statistics
import time

import torch
from torch import nn


class SmallRecurrentModel(nn.Module):
    """Dense teaching model, not the connectome-constrained biological model."""

    def __init__(self, inputs=60, neurons=65):
        super().__init__()
        self.input_weight = nn.Parameter(torch.randn(inputs, neurons) * .03)
        self.recurrent_weight = nn.Parameter(torch.randn(neurons, neurons) * .03 / math.sqrt(neurons))
        self.bias = nn.Parameter(torch.full((neurons,), .1))
        self.decoder = nn.Parameter(torch.randn(neurons, 2) * .01)
        self.intercept = nn.Parameter(torch.zeros(2))

    def forward(self, inputs):
        # inputs: [batch, time, input_neuron]. Each view is [batch, neuron].
        drives = (inputs @ self.input_weight).unbind(dim=1)
        state = inputs.new_zeros(inputs.shape[0], self.bias.numel())
        states = [state]
        alpha = -math.expm1(-.0025 / .1)
        for drive in drives[:-1]:
            # Rows evolve independently; later time steps still need earlier states.
            target = torch.relu(self.bias + drive + state @ self.recurrent_weight)
            state = state + alpha * (target - state)
            states.append(state)
        rates = torch.stack(states, dim=1)
        return rates @ self.decoder + self.intercept, rates


def objective(model, inputs, targets):
    prediction, rates = model(inputs)
    # Average over time and trajectories, so batch size does not multiply loss.
    return (prediction - targets).square().sum(-1).mean() + 1e-5 * rates.square().mean()


def step(model, optimizer, inputs, targets):
    optimizer.zero_grad(set_to_none=True)
    loss = objective(model, inputs, targets)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite toy loss.")
    loss.backward()  # Full history, no detach or truncation.
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return loss.detach()


def statistics_for(batch_size, steps, seconds):
    median = statistics.median(seconds)
    return dict(median_update_s=median, mean_update_s=statistics.mean(seconds),
                min_update_s=min(seconds), max_update_s=max(seconds),
                trajectories_per_s=batch_size / median,
                trajectory_transitions_per_s=batch_size * (steps - 1) / median,
                seconds_per_1024_trajectory_passes=1024 * median / batch_size)


def measure(initial, inputs, targets, device, warmup, repeats):
    # Creation, copies and random-number generation are outside the timer.
    model = copy.deepcopy(initial).to(device)
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    synchronize = torch.cuda.synchronize if device == "cuda" else lambda: None
    for _ in range(warmup):
        step(model, optimizer, inputs, targets)
    synchronize()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    seconds = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        loss = step(model, optimizer, inputs, targets)
        synchronize()  # Include completed GPU work, not just Python enqueue time.
        seconds.append(time.perf_counter() - start)
    result = dict(status="complete", device=device, batch_size=inputs.shape[0], steps=inputs.shape[1],
                  update_seconds=seconds, final_loss=float(loss),
                  **statistics_for(inputs.shape[0], inputs.shape[1], seconds))
    if device == "cuda":
        result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
    return result


def check_cpu_gpu(initial, inputs, targets):
    """Check predictions/loss/all parameter gradients on a small common prefix."""
    records = []
    for device in ["cpu", "cuda"]:
        model = copy.deepcopy(initial).to(device)
        x, y = inputs[:2, :33].to(device), targets[:2, :33].to(device)
        prediction, _ = model(x)
        loss = objective(model, x, y)
        loss.backward()
        records.append((prediction.detach().cpu(), loss.detach().cpu(),
                        {n: p.grad.detach().cpu() for n, p in model.named_parameters()}))
    for a, b in zip(records[0][:2], records[1][:2]):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-4)
    for name in records[0][2]:
        torch.testing.assert_close(records[0][2][name], records[1][2][name], atol=1e-6, rtol=1e-4)
    return dict(passed=True, steps=min(33, inputs.shape[1]), batch_size=min(2, inputs.shape[0]),
                scope="Small-prefix outputs, loss and gradients; not bitwise full-training equivalence.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=[8, 32, 128, 512])
    parser.add_argument("--steps", type=int, default=1025)
    parser.add_argument("--neurons", type=int, default=65)
    parser.add_argument("--inputs", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--devices", choices=["cpu", "cuda"], nargs="+")
    parser.add_argument("--output", help="Optional fresh JSON file; existing files are refused")
    args = parser.parse_args(argv)
    if (min(args.batches) < 1 or len(set(args.batches)) != len(args.batches) or args.steps < 2
            or args.neurons < 1 or args.inputs < 2 or args.repeats < 1 or args.warmup < 0 or args.threads < 1):
        parser.error("Require distinct positive batches, steps >=2, inputs >=2, positive repeats/width/threads and warmup >=0.")
    devices = args.devices or (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    if "cuda" in devices and not torch.cuda.is_available():
        parser.error("CUDA explicitly requested but unavailable; no silent CPU substitution.")
    if len(set(devices)) != len(devices):
        parser.error("Devices must be distinct.")
    output = Path(args.output) if args.output else None
    if output is not None and output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    initial = SmallRecurrentModel(args.inputs, args.neurons)
    # Generate once on CPU: same initial weights and nested input batches on both devices.
    inputs = torch.rand(max(args.batches), args.steps, args.inputs)
    targets = torch.cumsum(inputs[..., :2] - .5, dim=1) * .0025
    report = dict(status="running", settings=vars(args), torch=torch.__version__, python=platform.python_version(),
                  cuda_version=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                  cuda_device=torch.cuda.get_device_name() if "cuda" in devices else None, results=[],
                  definitions=dict(
                      timing="Resident inputs; includes full forward/backward, clipping and Adam. Excludes data generation, transfer, initialization, validation and file I/O.",
                      workload="Dense 65-neuron toy by default; short equal-length trajectories, no connectome or padding. Not a full long-path training benchmark.",
                      throughput="Trajectory passes per second; larger batches change gradient noise and number of optimizer updates per data exposure."))
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation protects earlier results; update only the file this run created.
        with output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    if "cpu" in devices and "cuda" in devices:
        report["cpu_gpu_check"] = check_cpu_gpu(initial, inputs, targets)
    print("device  batch  median update (s)  trajectories/s  peak CUDA MiB", flush=True)
    for device in devices:
        for batch_size in sorted(args.batches):
            try:
                result = measure(initial, inputs[:batch_size], targets[:batch_size], device, args.warmup, args.repeats)
            except torch.cuda.OutOfMemoryError:
                result = dict(status="out_of_memory", device=device, batch_size=batch_size)
                if device == "cuda":
                    torch.cuda.empty_cache()
            report["results"].append(result)
            if result["status"] == "complete":
                memory = result.get("peak_cuda_allocated_bytes")
                memory_text = f"{memory / 1024 ** 2:.1f}" if memory is not None else "-"
                print(f"{device:6} {batch_size:5} {result['median_update_s']:18.4f} {result['trajectories_per_s']:15.1f} {memory_text:>14}", flush=True)
            else:
                print(f"{device:6} {batch_size:5} OUT OF MEMORY", flush=True)
            if output:
                output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    report["status"] = "complete" if all(r["status"] == "complete" for r in report["results"]) else "complete_with_oom"
    if output:
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Compare throughput AND update time. Equal trajectory exposure is not equal optimization.")
    return report


if __name__ == "__main__":
    main()
