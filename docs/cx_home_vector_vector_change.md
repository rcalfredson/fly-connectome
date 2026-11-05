# Optional home-vector change supervision

Status, November 5: implementation for a follow-up experiment; no scientific
training with vector-change supervision has been launched as part of this work.
The [balanced absolute-vector experiment](cx_home_vector_balanced_loss.md)
remains a separate comparison. Default commands keep their previous objective.

## First experiment

Keep the prepared dataset, network, PFN mapping, initialization, optimizer,
batch size, update budget and absolute loss mode fixed. Add supervision of
home-vector changes over **one second** (400 simulation steps at dt = 0.0025 s).
Use coefficient **1** initially; this is a starting experimental choice, not a
calibrated optimum. A weight of **0**, the default, disables the additional term.
Both original and balanced absolute modes can optionally use it.

For a window starting at t:

```
predicted_change(t) = predicted_home(t + window) - predicted_home(t)
correct_change(t)   = target_home(t + window) - target_home(t)

change_loss = mean_trajectories(
    mean_valid_windows(
        ||predicted_change - correct_change||² / change_scale²
    )
)

total_loss = absolute_task_loss
             + change_weight * change_loss
             + existing_activity_edge_and_decoder_penalties
```

Use all overlapping windows at simulation-step spacing. Both endpoints must
lie within the valid trajectory, excluding padding. Batches use contiguous
valid prefixes, so valid endpoints imply the entire interval is valid. Every
trajectory must have at least one full window; unsupported window lengths are
rejected before creating an output bundle. Boundaries inside a window, such as
a reversal or movement-to-pause transition, remain included. The correct change
is the net displacement over the window, not the instantaneous velocity.

A single fixed `change_scale` is computed before optimization as the RMS
training target change, averaging windows within each trajectory and then
trajectories equally. For the existing 2,048-trajectory augmented bundle and
a one-second window it is **0.9288319663 mm**. Validation/test targets do not
set this scale. No per-window normalization is performed, so pauses with zero
correct change remain well defined. Absolute-distance group weights are not
applied to this term: the magnitude of a fixed-window displacement does not
inherently increase with total trajectory length. Uniform sampling still gives
the equal-size training groups equal expected influence on this new term.

Both prediction endpoints keep their gradients. State is not reset or detached
at window boundaries, and full-trajectory backpropagation remains in use.
Within-batch length grouping also supports the new term, weighting subgroup
means by trajectory count and applying parameter penalties once.

## Interpretation and limitations

The term penalizes a plateau during walking because predicted changes vanish
while target changes persist. During pauses it penalizes drift. Constant offsets
cancel out of the change term, which is why the absolute-vector loss remains.
Correct one-second changes alone do not guarantee correct long-term memory or
reversal dynamics within a window. The current long training examples still
contain uninterrupted constant motion; changing those examples is a separate
intervention. The new term cannot supply information absent from PFN inputs.

## Logging and selection

`training_config.json` records the coefficient, physical window, step count,
training target energy and frozen scale. The manifest records coefficient and
window. With the feature enabled:

- History `task` remains the absolute-task component; `change` is the unweighted change component. Total `loss` includes `change_weight * change`.
- Each validation row adds `validation_change`, containing equal-trajectory MSE in mm² and normalized MSE, overall and by original/long group.
- Final validation, test and challenge metrics contain the same diagnostics under `vector_change`.

Checkpoint selection **continues to use the absolute-vector validation
criterion** chosen by `--loss-mode`. Change diagnostics do not select checkpoints.
The new term therefore tests a training-signal change while holding selection
fixed. Compare both selected and last checkpoints if their behavior differs.
Half-step diagnostics keep the physical window fixed, doubling its step count;
the existing numerical convergence gate remains unchanged.

The engineering profiler retains its existing objective; use
`train --benchmark` with these flags if timing the new training objective.
The change term uses the existing prediction sequence and adds no recurrent
simulation passes for training or validation.

## Example command

Use a fresh output name. This initializes a new model with seed 0 rather than
continuing the current balanced run:

```bash
python -m fly_connectome.home_vector train \
  results/cx-home-vector-hdeltac-long-data-20261104 \
  --output results/cx-home-vector-hdeltac-long-balanced-change1s-seed0-u500-NEW \
  --device cuda --seed 0 --updates 500 --batch-size 128 \
  --compile-block 32 --loss-mode balanced \
  --change-weight 1 --change-window 1 \
  --threads 1 --validate-every 25 --max-seconds 86400
```

`NEW` is a placeholder for a distinct run label. Omit the two `--change-*`
options, or specify `--change-weight 0`, to retain the existing objective.
Judge the follow-up using original/long held-out errors, sustained magnitude
growth, loop closure, pause stability and reversal behavior, in addition to
the new diagnostic scores.

## Verification

The home-vector regression suite passed on CPU: 124 passed, 8 skipped.
The focused vector-change tests passed all 14 cases with CUDA available,
including eager/compiled agreement, complete trainer evaluation and checkpoint
selection on both devices. Mathematical checks cover masked endpoint gradients,
pauses and reversals, training-only normalization, unchanged disabled objectives,
grouped gradients, saved-prediction diagnostics and physical windows at a finer
timestep. These checks verify implementation, not scientific improvement.
