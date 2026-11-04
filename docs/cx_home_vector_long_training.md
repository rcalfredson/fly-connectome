# Training-distribution experiment: longer straight walks

Prepared 4 November 2026. The new data bundle is ready; the full scientific
training run has not started. A three-update GPU benchmark has completed;
[profiling tools](cx_home_vector_training_profile.md) investigate its cost
before launching the full run. The model, PFN encoding and
optimizer implementation are unchanged. The comparison is the existing
hDeltaC short-trained checkpoint versus a newly initialized hDeltaC fit with
the same seed, trained on an augmented trajectory distribution.

## What changes and what stays fixed

`results/cx-home-vector-hdeltac-long-data-20261104` derives from the verified
`results/cx-home-vector-hdeltac-data-20261101` bundle. Circuit, anatomy
provenance and legacy weights are copied byte for byte. Every original
trajectory remains unchanged at the start of its original split, including
all 12 historical challenges. The time step remains 0.0025 s and the loss
scale remains **2.045091886931525 mm**, inherited rather than recomputed.

Long paths cover the same four speeds, eight global headings and ten body
directions as the factorial part of the
[straight-walk sweep](cx_home_vector_straight_sweep.md). The separate matched
world-path sweep remains an evaluation probe; its additional headings are not
added to training. Body direction is atan2(rightward, forward) velocity.

Within each of the 32 speed/heading combinations, eight body directions are
assigned to training, one to validation and one to test. All durations for a
given control triple stay in that control's partition. The seeded split
reserves speed 1 mm/s, heading 45 degrees, forward walking for test, preserving
the historical long challenge as a held-out long-motion control. Split seed
20261104 is distinct from the network initialization seed, 0.

| Split | Original trajectories retained | Added straight paths | Total | Added durations |
| --- | ---: | ---: | ---: | --- |
| Train | 1,024 | 1,024 | 2,048 | 16, 32, 48, 64 s at each of 256 controls |
| Validation | 128 | 64 | 192 | 24, 56 s at each of 32 held-out controls |
| Test | 256 | 64 | 320 | 32, 64 s at each of 32 other held-out controls |
| Historical challenge | 12 | 0 | 12 | Unchanged |
| Extrapolation | 0 | 128 | 128 | 96, 128 s at 32 trained and 32 test controls |

At 2 mm/s, training paths reach 128 mm; the longer evaluation paths reach
192 and 256 mm. At slower speeds their absolute distances are smaller, but
their durations and distances exceed training examples at the same controls.

The existing trainer samples trajectories uniformly with replacement, so
long paths have a 50% sampling probability. The loss still averages time
within each trajectory, then averages trajectories. Equal sampling does
**not** imply equal loss contributions: larger targets can make long-path
errors dominate. Keeping the old scale avoids changing task loss relative
to regularization merely because the normalizing length changed. Short-path,
loop, reversal and pause behavior must be checked for tradeoffs afterward.

Checkpoint selection uses the combined validation set, including its new long
paths. Neither the test set nor the `extrapolation` split selects checkpoints.
The existing trainer deliberately evaluates only validation, test and challenge
at the end; it does not automatically evaluate the extra extrapolation split.
Use the frozen 128-second sweep below for duration extrapolation, including
96-second prefixes. Its complete grid includes both trained and held-out
controls; their assignments are saved in `long_training_design.json`.

## Reproduce preparation

Run from the repository root in the project environment:

```bash
python -m fly_connectome.home_vector.long_training \
  results/cx-home-vector-hdeltac-data-20261101 \
  --output results/cx-home-vector-hdeltac-long-data-NEW

python -m fly_connectome.home_vector verify \
  results/cx-home-vector-hdeltac-long-data-NEW
```

No neuPrint access is required. Existing output directories are refused.
Source files are hashed before and after preparation. The module also accepts
the original core's prepared task, allowing a later matched architecture
comparison with identical added trajectories. Do not feed an already augmented
task back into preparation.

## Benchmark and train

The already prepared November 4 bundle can be used directly. A benchmark has
already completed at `results/cx-home-vector-hdeltac-long-benchmark-seed0-20261104`;
it averaged 28.35 seconds per update over three updates, compared with 2.14
seconds in the earlier hDeltaC benchmark. At that observed average, 2,000
updates would take about 15.7 hours before validation and final evaluation.
These are pre-optimization measurements; the
[gradient-assembly change](cx_home_vector_gradient_assembly.md) subsequently
reduced the measured GPU update average to 6.81 seconds. Neither short
benchmark is a reliable full-run prediction. Full-trajectory
backpropagation through 64 seconds is substantially larger than the old
short-path workload. To repeat the GPU check, use a small,
explicitly unselected benchmark:

```bash
python -m fly_connectome.home_vector train \
  results/cx-home-vector-hdeltac-long-data-20261104 \
  --output results/cx-home-vector-hdeltac-long-benchmark-seed0-NEW \
  --device cuda --seed 0 --updates 3 --batch-size 8 \
  --threads 1 --max-seconds 1800 --benchmark
```

This benchmark performs optimizer updates in its own run. It is not resumed
or used to initialize the scientific fit. A three-update benchmark may miss
the worst sequence mix; it does not guarantee peak memory or convergence.

For the initial comparison, start from the same seed and retain Adam, batch
size, parameter bounds, regularization, full-trajectory state history and
validation interval:

```bash
python -m fly_connectome.home_vector train \
  results/cx-home-vector-hdeltac-long-data-20261104 \
  --output results/cx-home-vector-hdeltac-long-anatomy-seed0-NEW \
  --device cuda --seed 0 --updates 2000 --batch-size 8 \
  --threads 1 --validate-every 50 --max-seconds 86400
```

The 24-hour cap is a guard, not a predicted runtime. Training stops at the
update limit or the wall-time cap, checked between updates; it has no
validation-plateau early-stopping rule. Check `timing.json` for the completed
update count and stopping reason. Equal updates do not mean equal wall time
or equal short-example exposure. If memory requires a different batch size,
record that as another changed factor and consider a matched short-data
control. A plateau after an insufficient or capped optimization run is not
evidence of an intrinsic capacity limit.

## Evaluate after training

Replace the training-run path with the completed run. This uses independent
verified source bundles and explicitly permits their intended task difference:

```bash
OPENBLAS_NUM_THREADS=1 python -m fly_connectome.home_vector.analysis.straight \
  --reference-run results/cx-home-vector-hdeltac-anatomy-seed0-2026-11-01 \
  --reference-source results/cx-home-vector-hdeltac-data-20261101 \
  --run results/cx-home-vector-hdeltac-long-anatomy-seed0-NEW \
  --source results/cx-home-vector-hdeltac-long-data-20261104 \
  --reference-label "hDeltaC, short training" \
  --label "hDeltaC, long training" \
  --allow-task-differences --duration 128 --sample-times 16 32 64 96 128 \
  --output results/cx-home-vector-hdeltac-long-comparison-NEW
```

The opt-in requires byte-identical circuits and identical time steps; each run
must still match its own recorded source manifest. Default comparisons retain
the stricter matching-task requirement. New reports identify the models as
`reference` and `current`, with human-readable labels in `model_labels`.
The older November 3 sweep retains its historical `original`/`hdeltac` keys
and trace filenames; that existing bundle is unchanged.

The CSV includes the full-duration rows and the requested prefix rows; filter
by `duration_s` before aggregation. Prefix rows leave the full-run hidden-state
diagnostic columns blank. Prefix summaries also appear in JSON under each
model's `prefix_summaries`; the full-duration rows are under `trajectories`.
All prefixes come from one continuous zero-initialized rollout, without
resets at sampling times. The NPZ archives retain the full decoded traces.

Inspect fit on trained controls through 64 seconds, generalization to held-out
controls through 64 seconds, and extrapolation at 96/128 seconds separately.
The original challenge arrays remain directly comparable. The original 256
test trajectories are the first 256 entries of the enlarged test archive, so
recompute their metrics separately rather than comparing the new combined
test NRMSE directly with the old short-test NRMSE. Do the same for validation.

Success would show that the earlier training distribution contributed to the
plateau. Failure to fit the added training paths would leave optimization,
parameter bounds, dynamics, inputs and readout as competing explanations.
Across-seed replication and adequate optimization would still be needed before
making a stronger capacity claim about this model, much less the biology.

## Setup validation

Sixteen focused tests passed across dataset preparation, the straight sweep,
existing comparisons and run utilities. They check disjoint movement-control
partitions, preservation of original trajectories and bundle files, inherited
loss scale, evaluation-only longer durations, physical distance integration,
fresh outputs, and guards for cross-distribution comparisons. The prepared
November 4 bundle passes `home_vector verify`.

A separate frozen self-comparison of the existing hDeltaC checkpoint tested
custom labels, CSV prefix rows and ten-second summaries within a continuous
16-second rollout. Both sides produced identical predictions and metrics;
the smoke bundle hashes verified. This check performed no optimizer updates
and is not a new trained result.
