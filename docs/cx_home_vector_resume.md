# Continuing a completed home-vector training run

Resume support continues optimization from `last.pt` in a **complete, verified
training bundle**, writing into a fresh output directory. It restores model
parameters and buffers, Adam moments and step counters, NumPy sampling state,
and PyTorch CPU random-number state. New CUDA checkpoints additionally save
and restore CUDA random-number states. The original 500-update balanced bundle
predates that CUDA-state field; its current model has no stochastic CUDA
operations, so its saved sampling and optimizer states suffice for continuation.

The vector-change feature is preserved separately in commit `cb6d903` and remains
disabled unless explicitly requested. Resuming the balanced-only fit retains a
zero change-loss coefficient.

## Continue the balanced experiment

The completed 500-update run selected update 500 and passed its numerical
convergence gate. Its original-path validation MSE is 0.159895, compared with
0.0469264 for the earlier short-trained hDeltaC model on the same original
validation subset. Both original and long validation errors were still falling
at the update limit. Reaching the reference accuracy remains an experimental
question; continuing the unchanged objective tests whether more optimization
can close that gap.

For a first extension, run **1,500 additional updates**, reaching **2,000 total**:

```bash
python -m fly_connectome.home_vector train \
  results/cx-home-vector-hdeltac-long-data-20261104 \
  --resume results/cx-home-vector-hdeltac-long-balanced-seed0-u500-2026-11-05 \
  --output results/cx-home-vector-hdeltac-long-balanced-seed0-u2000-resumed-2026-11-05 \
  --device cuda --seed 0 --updates 1500 --batch-size 128 \
  --compile-block 32 --loss-mode balanced --change-weight 0 \
  --threads 1 --validate-every 25 --max-seconds 86400
```

This command has not been launched as part of implementation. Choose another
fresh output name if that directory already exists. `--updates` always means
updates to execute in the current invocation, including when `--resume` is used.
The first new history entry is update 501; the planned final entry is update
2000. A further continuation can resume this new output bundle after completion.

## Compatibility and selection

Retain the parent's dataset, seed, recurrence/rewiring, loss mode, batch size,
batching/length-ratio settings, device, threads, compilation settings and
PyTorch version. Pass their values explicitly when they differ from CLI
defaults. The trainer checks those settings, frozen loss scales/weights,
optimizer configuration, parameter bounds and circuit/initialization buffers
before creating the new output. An objective change is a separate experiment,
not an accepted resume operation. Older original-loss bundles without explicit
loss-mode metadata and balanced bundles without vector-change metadata are
supported through their historical defaults.

Keep validation every 25 updates for this comparison. Validation cadence does
not affect gradient updates, but changing it changes the candidate checkpoints;
a segment ending outside the regular cadence also adds a terminal validation.
The wall-time cap applies to the new segment. There is no new automatic stopping
rule at the reference MSE or at a validation plateau.

Resume uses **last**, not **best**, to preserve the optimization trajectory.
The earlier best checkpoint and its validation score are carried forward:
a worse continuation cannot replace it. Selection continues to use the balanced
absolute-vector criterion, not original-path error alone. Track the separate
original and long subset scores to judge whether approaching 0.04693 costs
long-path performance. The historical challenge plots and numerical gate remain
necessary checks of what improving MSE represents.

## Output and provenance

The parent directory remains unchanged. The new bundle records the parent path,
manifest and last-checkpoint hashes, and starting update. The parent is checked
again before the new bundle is marked complete. Source paths must remain
available through the run's completion.

`history.json` carries forward the parent's rows unchanged and appends new rows
with cumulative update numbers. Its elapsed-time field continues from the last
recorded parent value; it excludes idle time between invocations and is not
end-to-end experiment wall time. `timing.json` reports only the current segment's
elapsed time, executed updates and mean update duration, together with
`starting_update` and `final_update`. Final evaluation uses the best checkpoint
across the full history, which may precede the resumed segment.

Completed bundles are required; recovery from an interrupted/unverified bundle
is outside this interface. Retaining Adam and random-number state distinguishes
this operation from initializing a new optimizer with previously fitted weights.

## Verification

The home-vector regression suite passed on CPU: 136 passed, 9 skipped.
All 13 focused resume cases passed with CUDA available. Splitting and resuming
small runs matched uninterrupted training exactly, including parameters, Adam,
random-number states, loss/validation history and best-checkpoint selection.
Tests cover compiled CUDA continuation, compiled vector-change continuation,
chained resumes, inherited best checkpoints, legacy metadata, incompatible
recipes and artifact tampering. A read-only compatibility check also accepted
the actual completed 500-update balanced bundle; no scientific updates were run.
