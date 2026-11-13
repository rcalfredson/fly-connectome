# Matched recurrent training under original and sugar dynamics

## Purpose and status

The previous decoder comparison left recurrent parameters optimized for the
original dynamics. The long-loop probe showed a later benefit from suppression
on familiar circles, despite worse two-loop performance. A matched training
experiment can ask how both dynamics perform when their recurrent circuits
and readouts have an equivalent opportunity to adapt.

The opt-in trainer is `fly_connectome/home_vector/sugar_training.py`. The paired
seed-0 scientific run completed all **2,000 updates** on CUDA; its verified bundle
is `results/cx-sugar-paired-training-seed0-20261112-r1`. Both arms independently
selected update 1950 using validation data. Results are summarized below.
The first launch failed after six completed updates because static compilation
specialized on too many subgroup shapes; see the correction below. Archived
two-update checks are
`results/cx-sugar-paired-training-smoke-20261112` (eager) and
`results/cx-sugar-paired-training-compiled-smoke-20261112` (32-step Inductor blocks).
The final implementation with runtime dependency snapshots and hashes was
checked again at `results/cx-sugar-paired-training-verified-smoke-20261112`.
Their validation values are
engineering checks, not scientific performance estimates. Production dynamics,
the previous checkpoint, prescribed encoder, and archived analyses are unchanged.

## Controlled design

- Train two arms from the same seeded initialization, rather than giving either
  the previously optimized weights. Both use identical anatomical support and
  synapse normalization. No new anatomical connections are added.
- Optimize all anatomical edge coefficients, population biases and time
  constants, and the decoder/intercept, matching the classes of parameters in
  the original full trainer. Input-edge gains may adapt; prescribed PFN activity
  remains fixed. Sugar response constants remain fixed and untrained.
- Use `results/cx-sugar-decoder-data-20261112`. Its existing examples and eight
  variants per return family include post-event continuation. The original arm
  receives the same movements and targets, while its gate remains zero. The
  sugar arm receives the explicit sensory gate on treated trials.
- Sample half the expected training mass from existing examples and half from
  return examples, then sample families and their variants equally. The same
  trajectory indices feed both arms at every update. Loss terms are equal per
  sampled trajectory with uniform physical-time weighting; final states are
  evaluated separately, not added as time intervals.
- Use Adam with learning rate 0.001, gradient norm clipping at 1, the existing
  activity/edge-prior/decoder penalties (`1e-5`, `1e-4`, `1e-5`), and existing
  parameter bounds. Inherit the original training-only target length scale.
  Integrate at 0.0025 seconds with full-trajectory backpropagation; do not detach
  state between loops or computational blocks.
- The completed budget was 2,000 paired updates, batch size 8, seed 0,
  with validation every 50 updates. Each arm independently selects its best
  balanced validation error at the same candidate update counts. The test set
  is evaluated after checkpoint selection. Familiar circles and their longer
  repeats do not select checkpoints.
- Treat one seed as an initial comparison, not evidence about reproducibility
  across training seeds. If proceeding to a robust comparison, repeat both arms
  with matched additional seeds and the same predeclared recipe.

The dataset has one sugar delivery per treated return trajectory and trajectories
up to 50 seconds in that domain. Consequently this first full training did
not specifically train repeated feeding or 16-loop behavior. Those longer,
repeated-delivery conditions remain extrapolation probes. If they are later
added to training, that requires a new derived dataset for both arms, preserving
the present held-out comparison rather than silently reusing it for selection.

## Completed seed-0 comparison

The verified results are in
`results/cx-sugar-paired-training-seed0-20261112-r1/comparison.json`. Both
selected checkpoints are from update 1950; the final update-2000 checkpoints
are retained separately and are not substituted for the validation selections.

| Metric (mm) | Original | Sugar |
| --- | ---: | ---: |
| Selected validation vector RMSE | 7.452847 | 7.454228 |
| Selected validation endpoint RMS error | 13.310144 | 13.261253 |
| Held-out test vector RMSE | 5.743460 | 5.675652 |
| Held-out test endpoint RMS error | 9.898948 | 9.609298 |

These metrics use equal existing/return domain mass, then equal families and
variants. Here "balanced" means that sampling/aggregation policy, not division
by each domain's target energy. Endpoint errors cover all test trajectories;
they are not exclusively errors at geometric returns. The test vector RMSE is
about 1.18% lower for sugar, while validation vector RMSE is slightly higher.
One seed with a small aggregate difference does not establish a reliable sugar
advantage, convergence, or preservation of the prior frozen circuit's long-loop
benefit. Condition-specific assessment remains a follow-up. Extended-loop replay
of these newly trained checkpoints is now documented in
[cx_home_vector_sugar_retrained_loops.md](cx_home_vector_sugar_retrained_loops.md).

The older frozen-circuit decoder-only comparison used a different optimization
history and objective. These results do not establish improvement over that
earlier model. The proposed loss-balance and many-loop changes are documented
separately in `docs/cx_home_vector_many_loop_training_plan.md`; neither has been
applied to this completed run.

## Implementation checks

The differentiable sugar step uses the same effective relaxation equation as
the frozen probe. At zero gate it selects the exact native update. The model
retains the original checkpoint parameter names. Optional compiled blocks
accelerate gradient calculation without truncating trajectories; initial blocks
and short tails remain eager, and evaluation does not compile new shapes.

Tests establish native states/predictions/parameter gradients when disabled,
agreement with the frozen sugar rollout when enabled, nonzero gradients reaching
recurrent coefficients, and agreement between eager and compiled full-history
gradients. The associated comparison, sugar-probe, and repeated-loop suites
passed together initially: 25 tests. The two-update smoke run exercises complete
post-event paths, identical initial states, paired optimizer steps, validation,
checkpoint selection and archival hash checks. The compiled actual-circuit
smoke run produces the same two-update losses as the eager run. It skips scientific test
evaluation and uses only two training and two validation examples.
The eager/compiled final parameter discrepancies are below `1.2e-7`;
recurrent coefficients change by up to approximately `0.00163` after the two
updates in both arms. This confirms actual recurrent optimization, rather
than a decoder-only update.

The trainer writes both initial, selected and last checkpoints, optimizer/RNG
states, sampled indices and losses, validation history, configuration, circuit,
runtime scripts and hashes. It now also writes an atomic `paired_last.pt` after
each completed paired update, containing both models, both optimizers, and RNG
states. It does not implement automatic resume yet; saved
optimizer states are preserved for a future verified continuation path.

## Commands

```bash
# Completed engineering check; choose a new output to repeat it.
OPENBLAS_NUM_THREADS=1 python -m fly_connectome.home_vector.sugar_training \
  --data results/cx-sugar-decoder-data-20261112 \
  --output results/cx-sugar-paired-training-smoke-20261112 \
  --updates 2 --batch-size 2 --validate-every 2 --smoke

# Full comparison; use a fresh output directory for a corrected restart.
OPENBLAS_NUM_THREADS=1 python -m fly_connectome.home_vector.sugar_training \
  --data results/cx-sugar-decoder-data-20261112 \
  --output results/cx-sugar-paired-training-seed0-NEW \
  --updates 2000 --batch-size 8 --validate-every 50 --seed 0 \
  --compile-block 32 --device cpu
```

Use the PyTorch environment. CUDA may be requested explicitly if available;
there is no silent fallback. Compilation startup and full validation cost are
not estimated by the eager two-update smoke timing. Both eager and compiled
actual-circuit smoke runs have completed; these do not benchmark the longest
training paths or full validation workload. The
trainer executes equal update counts for both arms and does not compare runs
stopped at independently chosen wall-time limits.

Suggested additional commit boundaries after the preceding decoder comparison:
`feat(analysis): compare sugar schedules over extended circular paths`, then
`feat(training): add matched recurrent training with fixed sugar dynamics`.

## Correction after the first full launch

The first full launch, historically at `results/cx-sugar-paired-training-seed0-20261112`,
completed six updates, then failed in Dynamo's eight-variant recompilation
limit. That directory is no longer present in the current workspace. It predated per-update
recovery checkpoints and contains initialization snapshots/history, not a
resumable update-six checkpoint; a corrected launch must restart from seed 0
in a fresh directory.

The original `dynamic=False` compile setting specialized the recurrent block
for different length-group batch sizes. It also specialized on the initial
state's `requires_grad=False` versus continuation states' `requires_grad=True`.
The fixed path runs each trajectory's first block eagerly and marks only the
batch dimension of subsequent block inputs dynamic (size-one inputs have their
own specialization). Full-history gradients remain intact. Evaluation and
short tails remain eager. The global compilation limit is unchanged.
This follows PyTorch's guidance on
[dynamic dimensions and recompilation](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html).

The added regression exercises both zero and nonzero gates at batch sizes
1–8, with variable sequence lengths, and compares predictions, states, and
parameter gradients with eager execution. A counting backend confirms at most
two compiled graphs across that workload, and no compilation during evaluation.

The new `--benchmark` option uses the real sampled training workload but skips
validation and test evaluation and never selects a checkpoint. It is distinct
from `--smoke`, which uses only two fixed examples. It permits checks that cross
the original failure point without consuming held-out data for debugging.

The corrected Inductor path completed 12 seed-0 paired updates on the real
batch-size-8 workload, originally recorded at
`results/cx-sugar-paired-training-varied-benchmark-20261112`. That disposable
directory is no longer present in the current workspace; the completed CUDA
scientific run above is retained. The 12-update benchmark crossed the
previous failure at update 7; its first six batch indices match the failed run
and losses agree within floating-point tolerance. The full focused suite passes
26 tests. The atomic paired snapshot at update 12 matches both final model
snapshots and RNG states. Original inputs and runtime hashes are verified.
This real-workload check ran on CPU; the user's failed full run used CUDA,
which is unavailable in the current execution environment. No validation or
test data were evaluated in this engineering benchmark.

All six completed batches in the failed run contained treated trajectories
(counts 2, 4, 1, 2, 1, 2 out of 8). Near-identical initial losses therefore do not
indicate absent sugar delivery. The decoder starts at zero, making the first
task prediction identical in both arms, while activity regularization has a
small coefficient. Small early parameter differences are expected under shared
initialization and paired training.

## Pre-commit review, 13 November 2026

The focused dataset/decoder, long-loop, training, sugar-probe, repeated-loop and
measurement-inventory suites passed together: **30 tests**. Additional end-to-end
training tests check that both arms share initial parameters, selection uses
validation before final test evaluation, and atomic checkpoints preserve both
models and optimizer/RNG states. An injected failure during the second arm's
optimizer step leaves the previous complete paired checkpoint intact. The
long-loop geometry test now lives separately from the training tests.

The derived data, decoder fit, long-loop and component bundles and the completed
CUDA training bundle were checked against their manifests and original-input
hashes. Both selected update-1950 checkpoints match validation history; the
atomic update-2000 snapshot matches the two final model snapshots and RNG states.
Current training runtime hashes still match the completed run. Production
encoder/dynamics files and the original frozen checkpoint remain unchanged.
