# Recurrent training with reliable sugar at home

Status on 13 November 2026: the seed-0 paired scientific run is in progress at
`results/cx-sugar-reliable-return-training-seed0-20261113`. The dataset and
engineering checks below are complete; completed-run evaluation remains pending.

## Log entry

- The next training comparison tests whether recurrent dynamics can learn to
  use reliable sugar-triggered hDeltaG suppression to maintain an accurate home
  vector over diverse excursions and repeated returns.
- In the sugar-response network, every eligible home-patch reentry after
  departure triggers sugar delivery. The original network receives identical
  movements and pauses with its gate always zero. Paths without a home reentry
  naturally receive no sugar.
- Include regular-loop controls, out-and-back routes, irregular multisegment
  returns, and sequences of different excursions, including departures after
  home contact. Contact does not zero neural activity or change the correct
  home-vector target.
- Use fixed, training-only loss normalization, fresh held-out route families,
  and longer extrapolation tests. Train both networks from matched
  initialization and assess tracking between contacts as well as errors at
  returns. Physiology and integration performance are distinct assessments.

## Implemented task

The sealed dataset is `results/cx-sugar-reliable-return-data-20261113`.
All motion protocols, family/split identities, original home anchor, PFN
encoder, anatomy and response constants are preserved. Each distinct movement
within a return family appears once per body/duration/pause configuration;
`source_variant_ids` retain its provenance. Record IDs describe the new task.

| Split | Movements | With home deliveries | Without home reentry |
| --- | ---: | ---: | ---: |
| Train | 2,400 | 381 | 2,019 |
| Validation | 304 | 116 | 188 |
| Test | 512 | 202 | 310 |
| 16-return extrapolation | 48 | 48 | 0 |

There are also 12 historical challenge cases, 8 familiar diagnostic cases,
and 4 near-miss/pause-away controls. All use the same contact rule. Counts
without delivery reflect absence of eligible reentry, not withheld sugar.
The sampling policy remains equal mass across four strata, then equal families
and siblings: raw counts are not sampling proportions. All 256 retained
two-return and 96 diverse-return training movements receive sugar.

The new diverse routes still have 4/8 returns and span 24–96 seconds, while
16-return extensions span 96–192 seconds. Current geometry and reliable-gate
previews are at `results/cx-sugar-reliable-return-preview-20261113`. These show
prescribed paths and sensory inputs, not fitted neural responses.

## Exact event semantics

- The environment starts inside the home patch, without an initial sugar event.
- A visit is armed after departure beyond 0.15 mm from the anchor. Entry within
  0.1 mm generates one delivery and consumes that visit. Remaining inside the
  patch, including a home pause, does not generate new events each time step.
- There is no refractory interval, schedule-based withholding or return-count
  exception. Contacts on the final approach are delivered when a following
  simulation update exists. A contact occurring only at the terminal
  zero-duration state has no measurable following interval and is excluded.
- This rule applies to every example, including retained ordinary paths and
  diagnostics. A near miss or pause away from home does not deliver sugar.
- The environment uses geometry to generate the sensory event. The networks
  receive prescribed PFN activity and, in the sugar arm, the resulting gate;
  they do not receive position, true home vector or a geometric-reset flag.
  Correct targets are never zeroed or reanchored on contact, and a patch-edge
  contact need not have exactly zero correct home vector.

Consumption at patch reentry remains a modeling assumption. The hDeltaG rule
is fixed: 1-second rise, 300-second recovery, maximum event-kernel aggregation,
and 1-per-second relaxation toward zero reference activity. No physiology
parameters are fitted using decoder performance.

## Normalized paired training

Training strata contain 1,024 existing-other, 1,024 long-straight, 256
two-return and 96 diverse-return movements. Fixed target energies (mm squared)
are 4.811142, 849.933595, 19.923782 and 6.381247, respectively. The energies
agree to rounding precision with those of the preserved motion distribution.

```text
task_loss = batch_mean(time_mean(||prediction - target||^2) / E_stratum)
```

Time averages exclude the terminal zero-duration state. Training-only energies
and the `1e-8 mm^2` floor are frozen in `loss_plan.json`; validation and test do
not re-estimate scales. Both networks share initial parameters, movement draws,
normalization, regularizers, optimizer and update budget. All anatomical edge
coefficients, population biases/time constants and decoder/intercept can learn.
PFN activities, anatomical support and sugar constants stay fixed.

The original arm always receives zero gate. The sugar arm always receives all
recorded contact events; there are no deliberately sugar-withheld primary
training examples. Validation selects each arm's checkpoint using
`selection_score`, a dimensionless mean normalized MSE. Raw vector/endpoint
errors in millimeters and per-stratum scores are also retained. Atomic paired
checkpoints preserve both networks and optimizer/RNG states after every
completed update. Start the comparison from scratch, not a stopped-run state.

## Preparation and launch

```bash
python -m fly_connectome.home_vector.reliable_sugar \
  --source results/cx-sugar-diverse-return-data-20261113 \
  --output results/cx-sugar-reliable-return-data-20261113

# Scientific launch used for the active run; choose a fresh output to repeat.
MPLCONFIGDIR=/tmp/fly-connectome-mpl \
TORCHINDUCTOR_CACHE_DIR=/tmp/fly-connectome-inductor \
OPENBLAS_NUM_THREADS=1 \
/home/tracking/miniconda3/envs/py3.13-torch/bin/python -u \
  -m fly_connectome.home_vector.sugar_training \
  --data results/cx-sugar-reliable-return-data-20261113 \
  --output results/cx-sugar-reliable-return-training-seed0-20261113 \
  --updates 2000 --batch-size 8 --validate-every 50 --seed 0 \
  --compile-block 32 --device cuda
```

Outputs must be fresh directories. The builder and trainer both verify that no
eligible contact is missing a delivery and no spurious event is present. The
trainer validates the full event policy before taking an optimizer step.

Focused tests cover contact hysteresis, initial occupancy and pauses, closely
spaced returns, final-approach delivery, source-preserving duplicate removal,
rejection of missing events, and an end-to-end paired run checking that only
the sugar arm receives a nonzero gate. They run alongside route, normalization,
checkpoint, gradient and frozen-probe tests: **52 tests pass**.

The compiled two-update smoke is archived at
`results/cx-sugar-reliable-return-training-smoke-20261113`. It exercises all
four strata with full-history gradients and normalized validation selection.
The longest-batch engineering check is
`results/cx-sugar-reliable-return-longest-benchmark-20261113`: eight 96-second
training examples with repeated contacts, both arms, no held-out scoring.
Checks run on CPU; the supplied scientific launch requests CUDA explicitly.

The completed reference runs, stopped run and its input dataset are preserved
unchanged. The user launched the primary scientific run; do not edit its
imported runtime files while it is active. Validation-selected checkpoint
evaluation will follow once it is complete.
