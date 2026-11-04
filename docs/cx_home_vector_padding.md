# Padding reduction by grouping within a sampled batch

Investigated 4 November 2026, after the unbind optimization. The first
length-grouped implementation is correct within floating-point tolerance
and reduces padded work and peak memory, but **did not reduce runtime** on
the tested eight-trajectory batches. Ordinary padded batching remains the
default. No full scientific training run has started.

## Preserve the sampled batch and optimizer update

The sampler still draws eight indices uniformly with replacement. Only those
eight trajectories are sorted and partitioned; duplicates keep their original
multiplicity. A greedy subgroup can contain lengths up to twice its shortest
sample count, controlled by `--length-ratio`. Each subgroup is padded to its
own longest member and starts from zero state as usual.

For an example split into groups of five, one and two trajectories, accumulate:

```text
data gradient = (5/8 × gradient of group-five mean loss)
              + (1/8 × gradient of group-one mean loss)
              + (2/8 × gradient of group-two mean loss)

then add parameter-penalty gradients once
then clip the combined gradient once
then perform one Adam update and one parameter projection
```

Task and activity terms remain equally weighted per trajectory, including their
original means over valid time samples. Edge and decoder penalties use the
same shared loss coefficients as the padded objective. Each trajectory's full
history is retained; nothing is detached within a trajectory. Validation and
test evaluation are unchanged, and checkpoint keys/shapes remain compatible.

This differs from selecting length-homogeneous batches from the entire dataset:
that alternative changes which trajectories co-occur in an optimizer update.
It could be useful but is a different sampling experiment. The implementation
here first isolates an objective-preserving change to execution of the same
already sampled batch.

## Measurements

Fresh padded references and grouped runs used the same prepared hDeltaC
long-path task, seed 0, batch size 8, one CPU thread and three sampled batches,
without warmup. Whole-update measurements synchronize at update boundaries;
separate segmented passes also record phase costs. The primary means below
exclude inter-phase barriers, initialization, validation and output I/O.

| Device | Padded mean update | Length-grouped mean update | Change |
| --- | ---: | ---: | ---: |
| CPU | 4.747 s | 5.246 s | 10.5% slower |
| RTX 3060 | 7.114 s | 9.258 s | 30.1% slower |

Across these batches, executed trajectory/time slots decrease by **34.89%**.
Padding falls from **48.29% of original slots** to **20.59% of remaining
executed slots**. Peak live CUDA allocation in segmented passes decreases
from about **510.5 MiB to 397.7 MiB**. A slot is one trajectory at one sampled
time, containing all of its neurons; slot counts are not wall-time estimates.

Why can less work be slower? Each subgroup has its own sequential time loop.
Summed loop iterations increase from **70,403 to 99,808**, about **41.8%**,
even though each loop operates on fewer trajectories. That count includes
the final sample iteration, as the actual forward loop does. Repeated dispatch
and autograd work therefore increase. The measured increase in forward and
backward time is consistent with that tradeoff; this experiment does not
independently attribute every part of the slowdown.

The first batch illustrates the difference. A single eight-member loop spans
48 seconds. Grouping produces loops spanning 8.25, 16 and 48 seconds. Their
smaller batch dimensions save many padded tensor entries, but the execution
now walks through 72.25 seconds of sequential subgroup time in total.

This is a three-batch engineering result, not an optimum search over grouping
rules or a claim that every padding-reduction method is slower. In particular,
it does not test packed execution that removes finished trajectories while
retaining one overall time loop, nor global length-based batch formation.

## Correctness and retained outputs

Synthetic checks compare the full padded objective with accumulated subgroup
gradients, all parameter gradients, loss terms and three Adam updates. They
cover float32/float64, three/four populations, unequal subgroup sizes, repeated
trajectories and ratios producing one or several groups. The profiler also
matches the real trainer in each mode, including Adam state and loss history.

After the three actual updates, the largest padded/grouped model-state
difference is about **5.96e-8** on both devices. The largest Adam-state tensor
difference is about **1.05e-7** on CPU and **2.14e-8** on GPU; scalar losses
differ by at most **1.53e-5**. Reduction order changes, so equivalence is
numerical rather than bitwise. Each mode's independent timing passes still
match exactly in final parameters and gradients.

The focused home-vector and batch-toy suite passed **79 tests**, with four
neuPrint-dependent skips. Source bundles remain unchanged. Retained complete
and hash-verified bundles are:

- `results/cx-home-vector-hdeltac-padding-reference-cpu-20261104`
- `results/cx-home-vector-hdeltac-padding-reference-cuda-20261104`
- `results/cx-home-vector-hdeltac-length-groups-cpu-20261104`
- `results/cx-home-vector-hdeltac-length-groups-cuda-20261104`
- `results/cx-home-vector-padding-comparison-20261104`

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 python -m fly_connectome.home_vector.profile_training \
  results/cx-home-vector-hdeltac-long-data-20261104 \
  --output results/cx-home-vector-hdeltac-length-groups-cuda-NEW \
  --device cuda --seed 0 --updates 3 --warmup 0 --batch-size 8 --threads 1 \
  --batching length-groups --length-ratio 2
```

Change to `--batching padded` and a fresh output for the reference; use
`--device cpu` and another fresh output for CPU. `padding_fraction` in the
JSON report describes the original sampled batch; `executed_padding_fraction`
describes the actual grouped work, and `avoided_slot_fraction` measures its
reduction. Per-update `grouping` metadata records positions and counts.
Phase totals accumulate repeated subgroup phases. Operator tracing currently
requires padded mode; grouped timing passes retain full trajectories.

The training CLI accepts the same `--batching` and `--length-ratio` flags and
records them in the manifest/config/history. They are available for explicit
experiments; the observed batch-8 results do not support enabling grouping
for a speed improvement. Omitting the flags retains the original padded
behavior. Compilation has not been implemented.
