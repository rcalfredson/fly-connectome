# Balanced-loss follow-up

Status, November 5: implementation verified; the first 500-update balanced-loss
experiment has been launched. Its scientific assessment is pending.
The original loss remains the default, explicitly selectable with
`--loss-mode original`. Existing commands retain their original objective.

## Purpose and fixed controls

The [500-update long-path pilot](cx_home_vector_long_u500_assessment.md)
improved some long-distance errors while losing short-path, loop and pause
performance. This experiment tests a change in the relative importance of
those training groups. It does not establish whether the circuit can sustain
accurate integration indefinitely.

Reuse `results/cx-home-vector-hdeltac-long-data-20261104` unchanged. Keep the
PFN mapping, four-population circuit, fresh seed-0 initialization, uniform
sampling with replacement, batch 128, 500 updates, Adam learning rate 0.001,
gradient clipping at 1, compiled blocks of 32 steps, and validation every
25 updates. No curriculum, extra inputs or incremental-displacement loss is
introduced. Original and long trajectories already each have 50% sampling
probability; changing group sampling alone is not the proposed intervention.

## Frozen group weights

Let `E[g]` be the mean across training trajectories in group `g` of the
mean across valid timestamps of `||target||²`, in mm². Both means weight
trajectories equally. Compute these values once from training targets only:

```
u[g] = 1 / E[g]
w[g] = u[g] / mean(u[original], u[long])

prediction loss = mean_trajectories(
    w[group] * mean_valid_time(
        ||predicted_home - target_home||² / L²
    )
)
```

The inherited scale remains `L = 2.045091886931525 mm`. For this bundle:

| Group | Training trajectories | RMS target magnitude (mm) | Frozen weight |
| --- | ---: | ---: | ---: |
| Original | 1,024 | 2.045091887 | 1.990207496 |
| Added long straight paths | 1,024 | 29.155139091 | 0.009792504 |

These weights give the two groups equal expected task contributions for an
all-zero predictor. Their mean over training examples is 1. This does **not**
preserve the previous numerical task loss or gradient scale, and it does not
guarantee equal group contributions after learning. Activity, edge-prior and
decoder penalties retain their original definitions and coefficients. Batch
weights are not renormalized by their sampled sum. The implementation requires
equal nonempty training groups with finite positive target energies.

Group membership comes from saved trajectory kinds: `long_training` and
`long_holdout` are added paths; the historical challenge kind `long` remains
historical. There is no normalization by individual path length or instantaneous
home-vector magnitude, so origin crossings and pauses introduce no singularity.

## Validation and checkpoint selection

Balanced mode selects the minimum **equal average of the two weighted group
validation means**. Validation contains 128 original and 64 long trajectories;
averaging all weighted validation rows would therefore give original examples
twice the intended group influence. Both groups must be present. Validation
and test targets never determine training weights.

Each validation history row contains:

- `validation_scaled_mse`: the historical, unweighted mean across all validation trajectories, unchanged in meaning.
- `validation_subsets`: original/long counts and unweighted scaled MSEs.
- `validation_selection_score` and `validation_selection_metric`: the actual checkpoint criterion. Original mode selects the historical score.

`training_config.json` records target energies, RMS values, weights and the
selection rule. Training history retains all four loss components and adds
`gradient_clipped`; final metrics include the clipped-update fraction and
selected validation scores. The comparison tool's validation plot continues
to show the historical unweighted score, with its title identifying that
meaning; it is not necessarily the balanced run's checkpoint criterion.

Regularization and clipping should be inspected when comparing runs, because
changing relative task weights can change their influence. Compare raw
original/long validation and test errors, historical challenge plots, long
straight-path growth and extrapolation; a smaller balanced score alone does
not demonstrate better performance across those skills. The automatic final
numerical convergence gate remains in place.

## First training command

Use a fresh output name; this starts from the normal seeded initialization,
not a previous fitted checkpoint:

```bash
python -m fly_connectome.home_vector train \
  results/cx-home-vector-hdeltac-long-data-20261104 \
  --output results/cx-home-vector-hdeltac-long-balanced-seed0-u500-2026-11-05 \
  --device cuda --seed 0 --updates 500 --batch-size 128 \
  --compile-block 32 --loss-mode balanced \
  --threads 1 --validate-every 25 --max-seconds 86400
```

Change only `--loss-mode balanced` to `--loss-mode original` and choose another
fresh output directory to reproduce the previous objective with this trainer.
The engineering profiler retains its historical original-loss objective;
`train --benchmark --loss-mode balanced` can time balanced updates if needed.

## Verification

The home-vector regression suite passed on CPU (111 passed, 7 skipped). The balanced-loss
test file passed all 8 cases with CUDA available, including compiled training,
validation, checkpoint reload and metrics output on both CPU and GPU.
Checks also cover masked task gradients, unchanged regularizers, weighted
subgroup accumulation and deliberately conflicting raw/balanced checkpoint
rankings. Existing profiler/trainer tests retain exact agreement for the
original objective. These are small implementation checks, not evidence of
scientific improvement from the new objective.
