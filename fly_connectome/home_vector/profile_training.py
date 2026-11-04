"""Profile disposable training updates without changing the scientific trainer."""

import argparse
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import resource
import time

import numpy as np

from .__main__ import batch, begin, fail, finish, load
from .data import digest, write_json


def batch_description(indices, rows, dt):
    lengths = [round(row["protocol"][-1][0] / dt) + 1 for row in rows]
    padded = len(rows) * max(lengths)
    return dict(indices=list(map(int, indices)), durations_s=[row["protocol"][-1][0] for row in rows],
                valid_steps=sum(lengths), padded_steps=padded, longest_steps=max(lengths),
                padding_fraction=1 - sum(lengths) / padded)


@contextmanager
def phase(name, timings, synchronize, segmented, annotate):
    """Synchronized wall phases include device completion, not just enqueue time."""
    import torch
    if segmented:
        synchronize()
    start = time.perf_counter()
    with torch.profiler.record_function(name) if annotate else nullcontext():
        yield
    if segmented:
        synchronize()
    timings[name] = timings.get(name, 0.) + (time.perf_counter() - start)


def run_pass(circuit, groups, task, args, batches, output, label, segmented=False, trace_steps=None):
    """Match train(): seeded initialization, sampled batches, Adam, clipping, projection.

    Deliberately retain loop locals across updates, as the real trainer does;
    releasing prior graph objects can otherwise move costs between phases.
    """
    import torch
    from .model import HomeVectorNetwork, objective
    from .training_batches import backward_grouped
    torch.manual_seed(args.seed)
    network = HomeVectorNetwork(circuit, args.seed).to(args.device)
    optimizer = torch.optim.Adam(network.parameters(), lr=.001)
    cuda = args.device == "cuda"
    synchronize = torch.cuda.synchronize if cuda else lambda: None
    history = []
    batching = getattr(args, "batching", "padded")
    for update, indices in enumerate(batches, 1):
        selected = [groups["train"][i] for i in indices]
        metadata = batch_description(indices, selected, task["dt_s"])
        print(f"{label}: update {update}/{len(batches)}, longest {max(metadata['durations_s']):g} s", flush=True)
        synchronize()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        started, times = time.perf_counter(), {}

        def section(name):
            return phase(name, times, synchronize, segmented, trace_steps is not None)

        grouping = None
        if batching == "padded":
            with section("batch_preparation_and_transfer"):
                x, target, mask = batch(circuit, selected, task["dt_s"], args.device, network.a.dtype)
                if trace_steps is not None:
                    x, target, mask = x[:, :trace_steps], target[:, :trace_steps], mask[:, :trace_steps]
            with section("zero_grad"):
                optimizer.zero_grad(set_to_none=True)
            with section("forward"):
                prediction, rates = network(x, task["dt_s"])
            with section("objective_and_guard"):
                loss, terms = objective(prediction, target, rates, mask, task["length_scale_mm"], network)
                if not torch.isfinite(loss) or not torch.isfinite(rates).all() or rates.max() > 1e6:
                    raise ValueError("Profiling update exceeded the training numerical guard.")
            with section("backward"):
                loss.backward()
        else:
            with section("zero_grad"):
                optimizer.zero_grad(set_to_none=True)
            loss, terms, grouping = backward_grouped(network, circuit, selected, task["dt_s"], args.device,
                task["length_scale_mm"], getattr(args, "length_ratio", 2.), section)
        with section("gradient_clipping"):
            norm = torch.nn.utils.clip_grad_norm_(network.parameters(), 1., error_if_nonfinite=True)
        with section("optimizer"):
            optimizer.step()
        with section("parameter_projection"):
            network.project_parameters()
        with section("scalar_metrics"):
            values = dict(loss=float(loss.detach()), gradient_norm=float(norm),
                          **{k: float(v.detach()) for k, v in terms.items()})
        synchronize()
        wall = time.perf_counter() - started
        row = dict(update=update, measured=update > args.warmup, wall_s=wall,
                   batch=metadata, **values)
        if grouping is not None:
            row["grouping"] = grouping
        if trace_steps is not None:
            row["trace_steps"] = int(x.shape[1])
        if segmented:
            row["phases_s"] = times
            row["unattributed_wall_s"] = wall - sum(times.values())
        if cuda:
            row["cuda_memory_bytes"] = dict(
                allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
        row["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        history.append(row)
        # Reporting I/O is outside the timed update; same fresh history file pattern as train().
        write_json(output / f"{label}_updates.json", history)
        print(f"{label}: update {update} finished in {wall:.3f} s", flush=True)
    state = {name: value.detach().cpu().clone() for name, value in network.state_dict().items()}
    gradients = {name: p.grad.detach().cpu().clone() for name, p in network.named_parameters() if p.grad is not None}
    torch.save(dict(model=state, optimizer=optimizer.state_dict(), update=len(batches),
                    purpose="Disposable profiling; not a selected scientific checkpoint"),
               output / f"{label}_final.pt")
    return history, state, gradients


def compare_passes(a, b):
    import torch
    result = {}
    for name, first, second in [("parameters_and_buffers", a[1], b[1]), ("final_gradients", a[2], b[2])]:
        if first.keys() != second.keys():
            raise ValueError("Profiling passes have different state keys.")
        maximum = max(float((first[k].to(torch.float64) - second[k].to(torch.float64)).abs().max()) for k in first)
        close = all(torch.allclose(first[k], second[k], atol=1e-6, rtol=1e-5) for k in first)
        result[name] = dict(max_abs_difference=maximum, passed=close)
        if not close:
            raise ValueError(f"Segmented profiling changed {name} materially.")
    return result


def operator_trace(circuit, groups, task, args, batches, output):
    import copy
    import torch
    trace_args = copy.copy(args)
    trace_args.warmup = 0
    activities = [torch.profiler.ProfilerActivity.CPU]
    if args.device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    steps = round(args.trace_seconds / task["dt_s"]) + 1
    # Use one prefix from the last selected batch; fresh model, not the timing pass state.
    with torch.profiler.profile(activities=activities, record_shapes=False,
                                profile_memory=False, with_stack=False) as profile:
        run_pass(circuit, groups, task, trace_args, [batches[-1]], output, "operators", trace_steps=steps)
    profile.export_chrome_trace(str(output / "operator_trace.json"))
    events = profile.key_averages()
    (output / "operators_cpu.txt").write_text(events.table(sort_by="self_cpu_time_total", row_limit=40))
    if args.device == "cuda":
        (output / "operators_device.txt").write_text(events.table(sort_by="self_device_time_total", row_limit=40))
    write_json(output / "operators.json", [dict(
        name=e.key, count=e.count, self_cpu_time_us=e.self_cpu_time_total,
        total_cpu_time_us=e.cpu_time_total, self_device_time_us=e.self_device_time_total,
        total_device_time_us=e.device_time_total) for e in events])


def profile_training(args):
    import torch
    batching, length_ratio = getattr(args, "batching", "padded"), getattr(args, "length_ratio", 2.)
    if batching not in ("padded", "length-groups") or not np.isfinite(length_ratio) or length_ratio < 1:
        raise ValueError("Require padded/length-groups batching and finite length ratio >=1.")
    if batching != "padded" and args.trace_seconds:
        raise ValueError("Operator tracing currently requires padded batching; grouped timings use full trajectories.")
    if (args.updates < 1 or args.warmup < 0 or args.batch_size < 1 or args.threads < 1
            or not np.isfinite(args.trace_seconds) or not 0 <= args.trace_seconds <= 1):
        raise ValueError("Require positive updates/batch/threads, nonnegative warmup, trace_seconds in [0,1].")
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Output must be separate from the prepared source.")
    circuit, groups, task = load(source)
    if args.trace_seconds and not np.isclose(args.trace_seconds / task["dt_s"],
                                            round(args.trace_seconds / task["dt_s"]), atol=1e-7, rtol=0):
        raise ValueError("Trace duration must fall on the task time grid.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; do not silently substitute CPU.")
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    batches = [rng.integers(0, len(groups["train"]), args.batch_size) for _ in range(args.warmup + args.updates)]
    hashes = {str(p): digest(p) for p in source.iterdir() if p.is_file()}
    output, record = begin(output, "home_vector_training_profile_v1", source=str(source),
        source_manifest_sha256=digest(source / "manifest.json"), seed=args.seed,
        updates=args.updates, warmup=args.warmup, batch_size=args.batch_size,
        device=args.device, threads=args.threads, trace_seconds=args.trace_seconds,
        batching=batching, length_ratio=length_ratio)
    record.update(torch=torch.__version__, cuda_version=torch.version.cuda,
                  device_name=torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
                  source_sha256=hashes, scientific_training=False, profiling_optimizer_updates=True)
    write_json(output / "manifest.json", record)
    try:
        baseline = run_pass(circuit, groups, task, args, batches, output, "baseline")
        # Separate passes avoid retaining GPU training tensors across measurements.
        if args.device == "cuda":
            torch.cuda.empty_cache()
        segmented = run_pass(circuit, groups, task, args, batches, output, "segmented", segmented=True)
        agreement = compare_passes(baseline, segmented)
        report = dict(agreement=agreement, measured_updates=args.updates,
            baseline_mean_update_s=float(np.mean([r["wall_s"] for r in baseline[0] if r["measured"]])),
            segmented_mean_update_s=float(np.mean([r["wall_s"] for r in segmented[0] if r["measured"]])),
            mean_phases_s={name: float(np.mean([r["phases_s"][name] for r in segmented[0] if r["measured"]]))
                           for name in segmented[0][0]["phases_s"]},
            padding_fraction=float(1 - sum(r["batch"]["valid_steps"] for r in baseline[0] if r["measured"])
                / sum(r["batch"]["padded_steps"] for r in baseline[0] if r["measured"])),
            definitions=dict(
                baseline="Device-synchronized wall time per update, without inter-phase barriers; excludes initialization, history writes, validation and checkpoint I/O.",
                segmented="Separate fresh model and identical batches; barriers between phases can change runtime. Unsynchronized phase timers are not reported.",
                trace="Optional instrumented short prefix, fresh model, one batch; cannot estimate full-length update speed from it.",
                rss="Linux process lifetime peak, cumulative across passes; not per-update CPU allocation.",
                scope="Disposable optimizer updates; no model-selection or scientific checkpoint."))
        measured = [r for r in baseline[0] if r["measured"]]
        executed = sum(r.get("grouping", {}).get("executed_steps", r["batch"]["padded_steps"]) for r in measured)
        report["executed_padding_fraction"] = float(1 - sum(r["batch"]["valid_steps"] for r in measured) / executed)
        report["avoided_slot_fraction"] = float(1 - executed / sum(r["batch"]["padded_steps"] for r in measured))
        write_json(output / "profile.json", report)
        if args.trace_seconds:
            operator_trace(circuit, groups, task, args, batches, output)
        if hashes != {str(p): digest(p) for p in source.iterdir() if p.is_file()}:
            raise ValueError("Prepared source files changed during profiling.")
        finish(output, record)
    except BaseException as error:
        fail(output, record, error)
        raise
    return dict(output=str(output), **report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batching", choices=["padded", "length-groups"], default="padded")
    parser.add_argument("--length-ratio", type=float, default=2.)
    parser.add_argument("--trace-seconds", type=float, default=0,
                        help="Optional short operator trace, <=1 s; timing passes retain full trajectories")
    args = parser.parse_args(argv)
    print(json.dumps(profile_training(args), indent=2))


if __name__ == "__main__":
    main()
