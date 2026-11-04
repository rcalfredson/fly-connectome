# Batch size and the CPU/GPU crossover

Exploratory engineering measurements, 4 November 2026. This follows the
[gradient-assembly change](cx_home_vector_gradient_assembly.md). The scientific
training batch size has not changed, and no full training run has started.

## Freestanding toy

```bash
OPENBLAS_NUM_THREADS=1 python examples/recurrent_batch_benchmark.py \
  --devices cpu cuda --batches 8 32 128 512 --steps 1025 \
  --warmup 1 --repeats 3 --threads 1 \
  --output results/cx-home-vector-batch-toy-NEW/benchmark.json
```

The [script](../examples/recurrent_batch_benchmark.py) requires only PyTorch
and Python's standard library. It can run outside the repository with no
project imports or experiment files. Omit CUDA from `--devices` for a CPU-only
run. Explicit CUDA requests fail if CUDA is unavailable; the script never
silently substitutes CPU. Existing output files are refused.

Read `SmallRecurrentModel.forward` first: time remains a sequential loop,
while rows of the batch evolve independently through the same weights. The
`measure` function shows warmup and synchronization around completed updates.
Those are the two parts that directly express the experiment.

The default dense model has 60 inputs, 65 recurrent neurons and a two-component
readout. It uses the optimized unbind approach, ReLU, a leaky rate update and
full backpropagation through time. It is a teaching analogue, not the
connectome model: weights are dense, biases are per neuron, and there are no
type-based parameter gathers, anatomical constraints or padded trajectories.
Each sequence contains 1,025 samples (1,024 transitions), substantially shorter
than the 25,601-sample 64-second scientific paths.

Every device/batch case starts from the same weights and fresh Adam state.
Inputs are generated once on CPU and the batches are nested prefixes of the
same synthetic dataset. The loss averages trajectories and time. Inputs and
targets are already resident on the chosen device before measurement. The
timer includes forward, loss, backward, clipping and Adam; initialization,
data generation, transfers, validation and I/O are excluded. CUDA is
synchronized at update boundaries. There is one warmup update followed by
three timed updates; the table uses their median.

## Toy results

The saved result is `results/cx-home-vector-batch-toy-20261104/benchmark.json`,
using PyTorch 2.8.0+cu128, one CPU thread and an RTX 3060.

| Batch size | CPU update (s) | GPU update (s) | CPU trajectories/s | GPU trajectories/s |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.0986 | 0.2009 | 81.1 | 39.8 |
| 32 | 0.1251 | 0.1918 | 255.8 | 166.8 |
| 128 | 0.2936 | 0.1913 | 435.9 | 669.1 |
| 512 | 0.9265 | 0.1990 | 552.6 | 2,572.7 |

The toy has a CPU/GPU crossover between sampled batch sizes 32 and 128. At
512, the GPU is about 4.66 times faster than CPU on the same batch, and its
update time is nearly the same as at batch 8. This is consistent with
amortizing overhead across more parallel work, but does not isolate the
hardware cause. It does not show that batch 512 fits a full 64-second path:
the toy GPU allocation peaks near 1,055 MiB at batch 512 on much shorter data.

CPU/GPU predictions, loss and all parameter gradients were checked on a small
common prefix before benchmarking. Three synthetic tests cover mean-loss
gradient scaling when trajectories are repeated, throughput calculations,
CPU CLI execution and output-file protection. This is an execution check,
not a test of training convergence or biological behavior.

## Reading the two performance measures

**Seconds per update** tells us how long one optimizer step takes. **Trajectory
passes per second** tells us how much sampled data it processes per unit time.
The latter is batch size divided by seconds per update; one trajectory pass
does not imply a new independent trajectory when training samples repeat.

At fixed 2,000 optimizer updates, batch 128 processes sixteen times as many
trajectory samples as batch 8. A throughput gain therefore does not directly
translate into shorter training at the same update count. Holding total
trajectory exposure fixed would instead reduce the number of updates, which
changes the learning process. Gradient noise, Adam state evolution, learning
rate needs and validation scheduling can all differ. This is fundamentally
different from unbind, which preserves the existing update's mathematics.

Larger batches are a promising training hyperparameter experiment, not an
equivalence-preserving replacement for the current run configuration.

## One actual-model check

The unbind-optimized connectome model was also checked at batch size 128 using
the verified long-path dataset, seed 0 and one CPU thread. Both devices used
the same 128 sampled indices, protocols and masks; the longest path was 64
seconds. Each profiling run makes one baseline and one independently
initialized segmented update. There is no warmup in this check.

| Measurement | CPU | RTX 3060 |
| --- | ---: | ---: |
| Baseline update | 19.33 s | 10.55 s |
| Segmented update | 15.95 s | 9.22 s |
| Baseline trajectory passes/s | 6.62 | 12.13 |
| Segmented forward phase | 4.60 s | 1.98 s |
| Segmented backward phase | 8.69 s | 4.97 s |

The GPU baseline is about **1.83 times faster** on this matched batch. Unlike
the short toy, the real batch includes data preparation/transfer and substantial
padding (63.10% of batch-time slots). GPU live tensor allocation peaked at
about **7.18 GiB**. This fits the tested device but is a first-update observation,
not a guarantee for sustained updates, other processes or larger batches.

The CPU baseline/segmented passes match exactly in final parameters and
gradients, as do the GPU passes. Across CPU and GPU, the final parameter/buffer
maximum absolute difference is approximately 5.96e-8, and the loss difference
is approximately 7.63e-6. The two complete profiling bundles verify:

- `results/cx-home-vector-hdeltac-batch128-profile-cpu-20261104`
- `results/cx-home-vector-hdeltac-batch128-profile-cuda-20261104`

The baseline/segmented runtime differences are appreciable, and each contains
only one update. These numbers establish a promising case, not a precise
steady-state crossover or the optimal batch size. They also do not establish
training quality or justify keeping the same update budget after increasing
the batch sixteenfold. No production training configuration has changed.

To repeat the diagnostic with a fresh output:

```bash
OPENBLAS_NUM_THREADS=1 python -m fly_connectome.home_vector.profile_training \
  results/cx-home-vector-hdeltac-long-data-20261104 \
  --output results/cx-home-vector-hdeltac-batch128-profile-cuda-NEW \
  --device cuda --seed 0 --updates 1 --warmup 0 --batch-size 128 --threads 1
```

Use `--device cpu` and a separate output for the matching CPU case. A later
multi-update, warmed benchmark and a convergence comparison would be required
before choosing a larger batch for the full scientific fit.
