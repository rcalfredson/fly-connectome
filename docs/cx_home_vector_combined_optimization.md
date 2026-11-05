# Combining larger batches with recurrent-block compilation

This experiment tests whether the batch-size and compilation benefits
combine. This extends the [32-step compilation experiment](cx_home_vector_compilation.md).
These are disposable profiling updates, not scientific training runs.

## Results

| Batch size | Eager mean update | Compiled mean update | Compilation speedup |
|---|---:|---:|---:|
| 8 | 7.668 s | 2.400 s | 3.19× |
| 128 | 9.532 s | 4.554 s | 2.09× |

The optimizations combine, but the whole-update speedups do not multiply
independently. Applying the measured batch-eight compilation factor to the
fresh batch-128 eager time would predict 2.984 s; the actual result is 4.554 s,
about 53% longer. This fresh four-condition prediction differs from the earlier
3.53-second hypothesis, which used older 10.55-second/2.99× measurements.

At equal exposure (16,000 trajectory presentations), the fresh GPU estimates
are 2,000 × 7.668 s = 4.26 hours for batch-eight eager execution and
125 × 4.554 s = 9 minutes 29 seconds for batch-128 compiled execution, a
26.94× throughput improvement. These are update-computation estimates only.
The earlier 15.7-hour figure predates the `unbind` optimization; comparing
with it additionally includes that earlier improvement. None of these
estimates demonstrates equal learning quality.

Both eager/compiled pairs passed numerical agreement checks. Recorded loss
components and gradient norms matched exactly across the five updates.
Maximum final model differences were 9.69e-8 at batch eight and 2.98e-8 at
batch 128; maximum Adam-state differences were 1.86e-9 and 7.45e-9 respectively.
The prepared source and all input profiles remained unchanged during the audit.

## Design

Four fresh GPU profiles compare batch sizes 8 and 128, each with eager and
compiled-32 execution. All use the verified hDeltaC long-path bundle, seed 0,
one CPU thread, padded batching, full-history differentiation, the same dt,
Adam, clipping, numerical guards and objective. There is no length subgrouping,
mixed precision or change to the PFN encoding.

Within each batch size, eager and compiled cases receive identical sampled
batches and start from identical parameters and optimizer settings. Each
profile runs two warmup updates and three measured updates, then repeats
that sequence in a fresh model with synchronized timing of separate phases.
The first, unsplit pass is the primary wall-time comparison. Across batch
sizes, the sampler consumes different numbers of random draws per update;
the selected examples and valid-time fractions therefore differ. This is a
measurement of the actual sampler at each batch size, not a controlled test
with identical contents across different batch sizes.

The batch-eight warmup covers 48- and 64-second padded lengths. All five
batch-128 batches have a 64-second maximum. All measured updates in all four
cases have 25,601 samples per trajectory, including sample zero.

Independent multiplicative speedups would predict:

```text
compilation speedup at batch 8 = eager_8 / compiled_8
predicted compiled_128 = eager_128 / compilation speedup at batch 8
```

The comparison audits the sampled indices, measured-update flags, loss
components and gradient norms, then checks every final model tensor and Adam
state field within each eager/compiled pair. It verifies bundle provenance,
matching model/profiler code, and unchanged input artifacts. It does not
compare learned parameters between different batch sizes: those are different
optimization experiments.

## Reproduce

Run each command sequentially in the PyTorch 2.8 environment, with fresh
output names:

```bash
OPENBLAS_NUM_THREADS=1 python -m fly_connectome.home_vector.profile_training \
  results/cx-home-vector-hdeltac-long-data-20261104 \
  --output results/cx-home-vector-combined-b128-compiled-cuda-NEW \
  --device cuda --seed 0 --updates 3 --warmup 2 \
  --batch-size 128 --threads 1 --compile-block 32
```

Repeat without `--compile-block 32` for the batch-128 eager reference, and
repeat both settings at `--batch-size 8` using different output directories.
Compilation is lazy; use a separate `TORCHINDUCTOR_CACHE_DIR` if cold startup
needs to be measured. A cache reused from another profile reduces startup.

Audit the four completed profiles:

```bash
OPENBLAS_NUM_THREADS=1 python -m fly_connectome.home_vector.compare_compilation \
  results/cx-home-vector-hdeltac-long-data-20261104 \
  --small-eager results/cx-home-vector-combined-b8-eager-cuda-20261104 \
  --small-compiled results/cx-home-vector-combined-b8-compiled-cuda-20261104 \
  --large-eager results/cx-home-vector-combined-b128-eager-cuda-20261104 \
  --large-compiled results/cx-home-vector-combined-b128-compiled-cuda-20261104 \
  --output results/cx-home-vector-combined-comparison-NEW
```

Saved profiling bundles use the four paths above; the comparison bundle is
`results/cx-home-vector-combined-comparison-20261104`. All input manifests and
artifacts are hashed; completed results are not modified by the comparison.

## Limits

Three measured updates per case are a preliminary timing check, not a
statistical independence test or a sustained full-run benchmark. New shapes
can trigger recompilation; warmup timings include initialization of compiled
execution as well as ordinary update work. Update-based forecasts exclude
validation, final evaluation, checkpoint writing and compilation startup.

Equal sample exposure means sampled trajectory presentations, with replacement,
not necessarily distinct examples or equal optimization quality. Batch 128
uses 125 parameter updates for the exposure of 2,000 batch-eight updates.
The relevant learning comparison remains time to comparable held-out performance.
The scientific trainer's default behavior is unchanged.

## Interpretation of the batch-128 result

Compiled batch-128 updates averaged 4.554 s, versus 9.532 s eager (2.09×).
Measured compiled updates ranged from 4.415 to 4.682 s. At this measured rate,
125 updates imply 569 s, or about **9 minutes 29 seconds of update computation**.
This replaces the hypothetical 7-minute-20-second estimate; it is still not a
wall-time prediction for a full scientific training run.

The segmented passes help explain why a whole-update compilation speedup is
smaller at the larger batch size:

| Batch 128 phase | Eager | Compiled |
|---|---:|---:|
| Batch preparation and transfer | 1.972 s | 1.982 s |
| Forward | 2.062 s | 0.932 s |
| Backward | 5.302 s | 1.386 s |

Preparation/transfer is outside the compiled function and is essentially
unchanged. It accounts for about 46% of the compiled segmented update. At
batch eight, compiled preparation/transfer took 0.152 s, while compiled
forward/backward were 0.909/1.303 s: the recurrent work stays almost flat,
whereas preparing more examples costs substantially more time. These separate
phase measurements support an explanation involving the larger unaffected
preparation cost; they do not isolate CPU encoding from allocation and transfer,
or prove a unique cause for every timing difference.

The two optimizations therefore combine beneficially, but their **whole-update
speedup factors are not fully multiplicative** in these measurements. This does
not imply that batching cancels the recurrent compilation benefit.

GPU peak allocated memory was 7.183 GiB eager and 6.590 GiB compiled at batch
128. The first compiled update took 19.174 s, including first-use compilation;
the second took 4.607 s. These startup measurements are stored alongside the
warmed timings, not included in the measured-update mean.

Validation: all four profiles completed and their internal repeated-pass
checks passed. The cross-mode audit passed at both batch sizes. The comparison
and profiler test suite passed 9 tests, covering optimizer-moment comparisons,
resampled-batch and warmup rejection, source integrity, fresh output protection
and the existing canonical-trainer profiling checks. `git diff --check` passed.
