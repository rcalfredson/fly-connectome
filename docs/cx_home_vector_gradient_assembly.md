# Reducing repeated time-slice gradient assembly

Implemented 4 November 2026 after the
[training profile](cx_home_vector_training_profile.md) identified backward
computation as the main cost. This changes execution of the existing model,
not its equations, task, parameterization or training distribution. Padding
reduction is a separate follow-up and has not been implemented.

## Start with the freestanding example

```bash
python examples/home_vector_gradient_assembly.py
```

The [example](../examples/home_vector_gradient_assembly.py) imports only Python
standard-library modules and PyTorch. It needs neither project imports nor
experiment data, and runs on CPU. The two functions near the top are the key
comparison:

```python
# Before: tensor indexing creates a separate selection-backward operation.
for t in range(len(drive)):
    state = 0.8 * state + drive[t]

# After: expose views together; the loop still visits every time step.
for current_input in drive.unbind(dim=0):
    state = 0.8 * state + current_input
```

The 0.8 retention factor is a teaching example, not a replacement for the
model's time constants. The scalar upstream `gain` is learnable; the movement
inputs are fixed. Both functions compute exactly the same recurrence. The
script differentiates a simple final-state sum, displays gradients of the
drive and gain, and checks agreement. It also counts the actual autograd
nodes: four time selections versus one unbind for the default four steps.
It does not perform optimizer updates or present a timing benchmark.

For the four scalar steps, the final-state gradients with respect to drive
are `[0.512, 0.64, 0.8, 1.0]`. Conceptually, separate selections construct
full-size gradient buffers such as `[0, 0.64, 0, 0]`, then combine them.
Unbind's backward receives the slice gradients together and assembles the
result once. This avoids repeated zero-buffer work without removing any
dependence on earlier states. The source behavior underlying this explanation
is documented in the profiling report.

## Main-model change

The model computes the same `input_drive` tensor of shape
`[batch, time, dynamic_neuron]`. It now calls:

```python
input_steps = input_drive.unbind(dim=1)
```

Inside the existing recurrence, `input_steps[n]` replaces
`input_drive[:, n]`. The first expression indexes a Python tuple of views;
it does not create a new tensor selection node at each step. Neither path
copies new input values in the forward pass. All states, including the final
sample, are retained as before. The final drive sample remains unused because
there is no integration step beyond the trajectory endpoint.

The activation function, exponential-Euler update, state initialization,
continuation/reset behavior, decoder, loss, masks, optimizer, clipping and
parameter bounds are unchanged. All previous checkpoint keys and shapes are
preserved. There is no detachment, truncated backpropagation, coarser time
step, compilation, bias-index hoisting or change to padded batches.

## Matched GPU measurements

The new verified profiling bundle is
`results/cx-home-vector-hdeltac-long-profile-cuda-unbind-20261104`. Its reference
is `results/cx-home-vector-hdeltac-long-profile-cuda-20261104`. Both use the
same prepared source, RTX 3060, float32, seed 0, batch size 8, one CPU thread
and three batches without warmup. Their longest trajectories are 48, 64 and
64 seconds, respectively.

| Measurement | Indexed implementation | Unbind implementation |
| --- | ---: | ---: |
| Mean whole-update wall time, no phase barriers | 28.364 s | 6.813 s |
| Mean forward phase, synchronized | 2.000 s | 1.653 s |
| Mean backward phase, synchronized | 25.786 s | 4.589 s |
| Mean whole-update wall time with phase barriers | 28.105 s | 6.513 s |
| Peak live CUDA allocation, segmented pass | 510.5 MiB | 510.5 MiB |
| Padding fraction | 48.29% | 48.29% |

The primary update-time measurement improves by **4.16 times** (about 76%
less time). Backward phase time improves by **5.62 times**. At the observed
new update average, 2,000 updates alone would take roughly **3.8 hours**, versus
15.8 hours at the original profile's average. These are short engineering
measurements, not full-run predictions: validation, final evaluation,
checkpoint/history I/O, longer-term variation and different sampled batches
are excluded from that extrapolation.

The optional 100-transition trace changes from 100 `aten::select_backward`
calls to none, with one `UnbindBackward0`. `aten::slice_backward` calls decrease
from 102 to two. The two remaining slices select the input and recurrent
weight blocks; they are not per-time-step input-drive selections. This
confirms that the intended gradient graph changed. The short trace is not
the source of the full-length speed measurements.

## Equivalence checks

### CPU follow-up

Because the original CPU attempt did not complete its first update within
five minutes, CPU performance was checked again after this change. A one-update
48-second check completed in 3.34 seconds. A separate three-update run on the
same batches then averaged **4.60 seconds per update**, compared with 6.81
seconds on GPU. Its bundle is
`results/cx-home-vector-hdeltac-long-profile-cpu-unbind-three-updates-20261104`.
This makes CPU a credible candidate again for this small network. It is not
a full-job device comparison or a guarantee for other batch sizes.

The optimized CPU baseline/segmented passes match exactly. Comparing the CPU
and GPU three-update final states gives a largest absolute parameter/buffer
difference of about 1.19e-7; one update's scalar loss differs by about 7.63e-6.
Cross-device float32 execution therefore agrees closely but is not bitwise
identical. No change to the scientific training device has been made.

### Original versus optimized GPU implementation

The retained comparison bundle is
`results/cx-home-vector-gradient-assembly-comparison-20261104`:

- The exact sampled trajectory indices and lengths match across both GPU runs.
- Every loss, loss component and pre-clipping gradient norm matches exactly
  for the three updates.
- All final model parameters and buffers match exactly.
- All final Adam tensors and parameter-group settings match exactly.
- The optimized profiler's independent baseline/segmented passes also match
  exactly in their final parameters and clipped gradients.

Synthetic tests retain an independent copy of the original indexed forward
loop. They compare predictions, rates, all loss terms, every parameter
gradient, input gradients and carried-state gradients across three Adam
updates. Coverage includes three/four populations, float32/float64, zero
initial state, continuation, reset, and masks with padded tails. A graph test
checks that the per-time selection-backward nodes are replaced with one
unbind-backward node. Existing finite-difference, NumPy-reference and
checkpoint/ablation checks continue to run.

The home-vector test suite passed **61 tests**, with four neuPrint-dependent
tests skipped in this PyTorch environment. Prepared data and existing trained
results were preserved. No full scientific fit has started.

## Reproduce the optimized profile

```bash
OPENBLAS_NUM_THREADS=1 python -m fly_connectome.home_vector.profile_training \
  results/cx-home-vector-hdeltac-long-data-20261104 \
  --output results/cx-home-vector-hdeltac-long-profile-cuda-unbind-NEW \
  --device cuda --seed 0 --updates 3 --warmup 0 --batch-size 8 --threads 1 \
  --trace-seconds 0.25

python -m pytest -q tests/test_home_vector_gradient_assembly.py
```

The next separate implementation candidate is avoiding work on padded tails
while preserving the same sampled trajectories and combined gradient.
The subsequent [batch-scaling experiment](cx_home_vector_batch_scaling.md)
also investigates CPU/GPU throughput with a freestanding toy and a larger-batch
actual-model check; changing batch size is a separate training choice.
