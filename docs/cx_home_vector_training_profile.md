# Profiling long-path training

The profiler measures disposable optimizer updates from initialization. It
does not alter the trainer, model, dataset, time step, gradient history or loss,
and does not produce a selected scientific checkpoint. Output directories
must be fresh and separate from the verified prepared input bundle.

## Run

From the repository root in the PyTorch 2.8 environment:

```bash
OPENBLAS_NUM_THREADS=1 python -m fly_connectome.home_vector.profile_training \
  results/cx-home-vector-hdeltac-long-data-20261104 \
  --output results/cx-home-vector-hdeltac-long-profile-cuda-NEW \
  --device cuda --seed 0 --updates 3 --warmup 0 --batch-size 8 --threads 1 \
  --trace-seconds 0.25
```

For a CPU comparison, change `--device cpu` and choose a different output.
For the previous short-data workload, supply its prepared bundle and another
output. No GPU is silently replaced with CPU. All source files are hashed
before and after profiling; manifests include code hashes and device/library
information. Verify outputs with the existing `home_vector verify` command.

## First GPU measurements, 4 November 2026

The completed bundle is
`results/cx-home-vector-hdeltac-long-profile-cuda-20261104`, using PyTorch
2.8.0+cu128 on an RTX 3060, batch size 8, one CPU thread and seed 0. Three
full-length updates were measured without warmup, matching the earlier
benchmark's sampled indices. This performs six disposable timing updates plus
one short-prefix trace update; no scientific fit has started.

| Phase | Mean synchronized wall time per update | Share of summed phase times |
| --- | ---: | ---: |
| Backpropagation | 25.786 s | 91.75% |
| Forward computation | 2.000 s | 7.11% |
| Batch preparation and transfer | 0.234 s | 0.83% |
| Objective and numerical guard | 0.073 s | 0.26% |
| Gradient clipping | 0.010 s | 0.04% |
| Adam update | 0.0013 s | <0.01% |

Whole-update means were 28.364 seconds without inter-phase barriers and
28.105 seconds with barriers. Their 0.9% difference is small in this run;
neither three-update sample provides a confidence interval. The parameters,
buffers and final gradients matched exactly between passes. Peak live CUDA
allocation was about 510.5 MiB in the segmented pass. This is allocator-tracked
tensor memory, not total device usage or a worst-case memory guarantee.
Padding occupied 48.29% of the sampled batch-time slots.

These measurements deprioritize input caching and Adam tuning as solutions
to the observed runtime: their measured costs are small. The main target is
backward computation through the existing recurrent implementation.

### A concrete backward-pass candidate

The current loop reads `input_drive[:, n]` separately at each time step.
The input drive depends on learnable weights. A small diagnostic of this
model's autograd graph confirms that each time selection records the complete
batch/time/neuron shape for its backward operation. In PyTorch 2.8, the
[selection backward implementation](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/TensorShape.cpp#L2164)
creates a zero tensor of that full shape and copies the selected gradient
into its corresponding slice.

**Inference from the source and graph structure:** repeating that operation
for every time step creates gradient-buffer work that can grow quadratically
with trajectory length, beyond the necessary recurrent computation. The
100-transition operator trace contains 100 `aten::select_backward` calls and
102 `aten::slice_backward` calls, consistent with this structure. It does not
measure their full-trajectory share of backward time.

A targeted next experiment is to expose time slices together, for example
through `unbind`, and hoist constant population-indexed bias out of the loop.
The aim would be to reduce redundant gradient assembly and repeated indexing
while preserving all states, loss terms and full backpropagation. This is a
candidate for equivalence and timing tests, not an implemented or measured
speedup. Compilation and padding reduction remain other options afterward.

### CPU comparison and validation

A matching CPU run, batch size 8 and one thread, was stopped after more than
five minutes without completing its first 48-second batch. The matched GPU
baseline update completed in 19.34 seconds. No completed CPU mean or phase
breakdown is claimed. The retained CPU bundle,
`results/cx-home-vector-hdeltac-long-profile-cpu-20261104`, is explicitly marked
failed/interrupted and contains an interruption record. Input hashes remained
unchanged. This deprioritizes switching to CPU for the unchanged long-path
implementation; it says nothing about a future optimized CPU implementation.

Focused tests cover exact agreement with canonical trainer parameters, Adam
state, losses and gradient norms; padding counts; warmup exclusion; short
trace export; fresh-output enforcement; source mutation; and interruption
handling. The completed GPU profiling bundle passes `home_vector verify`.
The focused profiler/model/run suite passed 16 tests, with one independent
framework-reference test skipped because neuPrint is absent from this PyTorch
environment. No neuPrint queries are needed for profiling.

## How measurement works

Each run makes two independently initialized timing passes through the same
seeded sequence of batches. They use the actual `batch`, `HomeVectorNetwork`
and `objective` functions, Adam settings, numerical guard, gradient clipping
and parameter projection used by training. Optimizer updates are real but
confined to this disposable experiment.

1. **Baseline:** synchronize at update boundaries, with no added barriers
   between phases. This is the primary whole-update runtime estimate.
2. **Segmented:** additionally synchronize at phase boundaries to separate
   preparation/transfer, zeroing gradients, forward, objective/guard, backward,
   gradient clipping, Adam, parameter projection and scalar logging.

The phase barriers change the execution schedule. Compare segmented totals
with baseline totals before interpreting their breakdown; synchronized phase
times are not independently measured kernel execution time. GPU execution is
asynchronous, so merely timing Python calls without synchronization would
misattribute pending GPU work to later phases. See [PyTorch CUDA timing
guidance](https://docs.pytorch.org/docs/2.8/notes/cuda.html#asynchronous-execution).

The profiler retains full trajectories and full backpropagation in both timing
passes. Initialization, progress/history file writes, checkpoint writes,
validation and final scientific evaluation are outside the reported update
times. Source loading and integrity hashing are also excluded. Batch generation
and transfer are included together, using the unchanged trainer's batch helper.
These measurements alone cannot separate their individual contributions.

`--warmup 0` reproduces the three batches from the seed-0 benchmark. For a
follow-up estimate, `--warmup 1` performs and records an additional first update
per pass but excludes it from reported means. That advances the model and
batch stream identically in both passes. It therefore changes which batches
are measured relative to the original benchmark.

The two passes' final parameters, buffers and clipped gradients are checked
for agreement. A material discrepancy fails the profiling bundle. Synthetic
tests also compare its updates with the canonical training function, including
Adam state, losses and gradient norms.

## Optional operator trace

`--trace-seconds 0.25` adds a third, fresh-model pass on one prefix from the
last selected batch. Prefix duration is restricted to at most one second to
bound instrumentation overhead. It does not shorten either timing pass.
Leave the option at its default of zero to disable tracing.
Preparation still processes the complete selected protocols before slicing
the neural inputs and targets; trace-wide phase fractions therefore do not
describe a uniformly shortened training workload.

This pass records CPU operators and, on CUDA, GPU activity. It exports a Chrome
trace and operator tables. Shapes, stacks and allocation-event tracing are
disabled to limit overhead. The short trace shows operator counts and the
relationship between CPU dispatch and GPU work; it does **not** establish the
full-length bottleneck, full-length throughput or a predictable speedup.
See the [PyTorch profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).

## Outputs

| Artifact | Meaning |
| --- | --- |
| `profile.json` | Measured means, phase means, padding fraction and agreement checks. |
| `baseline_updates.json`, `segmented_updates.json` | Per-update timings, exact sampled indices/durations, loss, gradient norm, memory statistics and warmup flags. |
| `baseline_final.pt`, `segmented_final.pt` | Disposable final states for diagnostic comparisons; not selected or resumable scientific runs. |
| `operator_trace.json`, `operators*.txt`, `operators.json` | Optional short-prefix trace and operator summaries; `operators_*` update/state files describe that extra disposable pass. |
| `manifest.json` | Completion/failure status, source/code/artifact hashes and profiling settings. |

CUDA memory fields distinguish live tensor allocation from allocator-reserved
memory and report their update peaks. CPU peak RSS is the Linux process's
lifetime maximum, cumulative across both passes; it is not a per-update memory
increment. Operator table times are in microseconds, while phase/update times
are in seconds. Total operator times include nested operations and must not be
summed indiscriminately; self times exclude nested operators.

## What to investigate next

Use measured phase costs to choose the next experiment. Large backward time
would favor investigating autograd and recurrent-loop execution; large batch
preparation time would favor input caching. Padding counts quantify the scope
for avoiding work after individual trajectory endpoints, but do not directly
predict a proportional wall-time saving. Compare CPU and GPU on identical
batches before assuming the small recurrent state benefits from the GPU.

No optimization is implemented by this tool. Changes to execution should be
validated against predictions, loss, gradients and an optimizer update before
being used for the long-training comparison.
