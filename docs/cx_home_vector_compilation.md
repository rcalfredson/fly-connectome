# Bounded recurrent-block compilation

The subsequent [combined-optimization experiment](cx_home_vector_combined_optimization.md)
tests the same compiled blocks at batch sizes 8 and 128. Measurements below
remain the original batch-eight reference.

On 2026-11-04 we tested `torch.compile` on blocks of 32 consecutive recurrent
transitions. This is an optional profiling experiment; the scientific trainer
still uses eager execution. Padded batching, PFN mapping, dt, loss, clipping,
Adam and full-history differentiation are retained.

## What is compiled

`model.recurrent_block` takes the current state, a tuple of input-drive slices,
recurrent weights, population bias and integration factor. It performs the
same ReLU/exponential-Euler updates and returns both the final state and every
intermediate state. `torch.compile(..., backend="inductor", fullgraph=True,
dynamic=False)` compiles this bounded function, including its backward graph.
The loop across blocks remains Python. Adjacent blocks share a differentiable
state: there is no detach, state reset or truncated backpropagation at block
boundaries. Protocol resets retain their original meaning.

Input validation, connectome weight assembly, the input projection, decoder,
objective and optimizer remain outside compilation. The existing single
input-drive `unbind` is retained. Short final blocks run eagerly, avoiding
specialization for every possible remainder length. No TF32, mixed precision,
CUDA graph mode or loss change is introduced.

The model exposes `compile_recurrence(32)` for this experiment. It compiles a
standalone function rather than wrapping the module, so checkpoint parameter
names remain unchanged. It does not persist the execution setting in the
state dict. The profiler records the setting in its manifest.

## Reproduce

In the PyTorch 2.8 environment, with a fresh output directory:

```bash
OPENBLAS_NUM_THREADS=1 python -m fly_connectome.home_vector.profile_training \
  results/cx-home-vector-hdeltac-long-data-20261104 \
  --output results/cx-home-vector-compile32-cuda-NEW \
  --device cuda --seed 0 --updates 3 --warmup 2 \
  --batch-size 8 --threads 1 --compile-block 32
```

For the matched reference, omit `--compile-block 32`, keeping all other
settings and using a different fresh output directory. Run device/case
benchmarks sequentially. CPU uses `--device cpu`. Do not turn on length groups.

The first five sampled batches have longest durations 48, 64, 64, 64 and
64 seconds. Warmup excludes the first two **optimizer updates**, not examples
from the batch. Both passes use the same batches and fresh seed-0 models.
Warmup updates still affect optimizer state in both conditions.

## Startup and interpretation

Compilation is lazy. A first-use pilot with a fresh compiler cache took
25.66 s for the first CPU update and 19.15 s for the first GPU update. The
second update took 10.41 s and 10.59 s respectively, including specialization
for the new 64-second shape. The third updates took 3.00 s and 2.49 s.
These startup figures include the entire update, not just compiler time.

Input-drive views retain strides dependent on padded trajectory length.
Consequently a new padded length, batch size, dtype or gradient configuration
may require another graph. Two warmups cover the shapes in this measurement;
they do not establish that arbitrary future training batches avoid recompilation.
The compiler cache can reduce startup on later processes. Cold startup and
warm update times must therefore be reported separately. The profiler's
segmented second pass also benefits from cache entries created in its first
pass; use the first pass (`baseline`) for the primary throughput comparison.

The default profiler summary only excludes the requested number of warmups;
it does not automatically detect subsequent compilations. A mean with only
one warmup was contaminated by the second shape's compilation, so it is not
our steady-speed comparison.

## Correctness checks

The focused tests compare predictions, every rate sample, loss, gradients of
all parameters, PFN inputs and the initial state, plus two clipped/projected
Adam updates and optimizer moments. They exercise float32/float64, resets at
zero, inside a block, at a boundary and at the final sample, continuation,
masked samples and short remainders. Most cases use AOTAutograd's `aot_eager`
backend to test graph differentiation without repeated native compilation;
a separate float32 case uses actual Inductor on CPU and CUDA when available.

Full-length matched profiles compare final parameters, optimizer moments,
loss components and gradient norms across eager and compiled execution.
Numerical agreement within tolerance is required; operation fusion can alter
floating-point rounding. These checks establish implementation agreement on
the tested cases, not identical convergence over a full scientific run.

## Saved experiments

Initial startup pilots:

- `results/cx-home-vector-compile32-cpu-20261104`
- `results/cx-home-vector-compile32-cuda-20261104`

Matched five-update comparisons use the `compile32` and `compile-reference`
`*-warm-20261104` bundles, with three measured updates after two warmups.
Each bundle records source hashes and code hashes; none is a selected
scientific checkpoint. The existing prepared source bundle is unchanged.

## Matched measurements

Batch size 8, one CPU thread, seed 0, two warmup updates followed by three
measured updates. All measured batches have 25,601 samples (64 seconds).
Numbers include batch preparation/transfer, full forward/backward, clipping,
Adam, projection and scalar metrics; they exclude validation, checkpoint I/O
and initialization. Eager already includes the earlier `unbind` optimization.

| Device | Eager mean update | Compiled 32-step mean update | Speedup |
|---|---:|---:|---:|
| CPU, one thread | 5.172 s | 3.053 s | 1.69× |
| RTX 3060 GPU | 7.629 s | 2.548 s | 2.99× |

These are short isolated measurements, not a full-run time forecast. Only
32-step blocks and batch size 8 were tested here; results do not establish the
best block size or the benefit at batch size 128. Most of the GPU segmented
improvement is in backward execution (5.47 s to 1.35 s); forward also improves
(2.11 s to 0.91 s). Segmented timings come from separate passes and are not
additive explanations of the unsplit first-pass wall-time difference.

Across the five matched updates, recorded loss components and gradient norms
were identical on each device. Maximum final model differences were
5.96e-8 (CPU) and 9.69e-8 (GPU); maximum Adam-state differences were 1.86e-9
on both. GPU peak allocated memory declined from 510.5 MiB to 472.2 MiB.
The verified comparison summary is saved in
`results/cx-home-vector-compilation-comparison-20261104/comparison.json`,
with hashes of its input profiles and the prepared source manifest.

This result supports keeping compiled blocks as an optimization candidate.
Before enabling them in the scientific trainer, examine recompilation across
its full length distribution and the intended larger batch size. It does not
show that the trained model integrates longer home vectors: no full scientific
training or model-selection run was performed in this experiment.

Validation: the home-vector and batch-toy suite passed 91 tests, with four
optional-neuprint skips and one unavailable-CUDA skip inside the sandbox.
The separately executed CUDA Inductor correctness case passed on the GPU
(92 tests passed in total across those invocations). `git diff --check` passed.
