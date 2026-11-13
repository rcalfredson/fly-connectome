# Planned successor: learning over repeated returns

Status: design only. No new dataset or scientific training run has been started.
The completed seed-0 paired run, its source files, original encoder, frozen
checkpoint and all archives remain unchanged.

## Question

Can original and sugar-response dynamics learn useful home-vector tracking
when training explicitly includes repeated returns and post-sugar continuation?
The existing run asks about shorter-return training; the proposed successor
asks about longer-duration training. Neither asks that their losses diverge.

The initial seed-0 run completed 2,000 paired updates and independently selected
update 1950 for both arms. Original/sugar held-out test vector RMSEs were
5.743460/5.675652 mm; selected validation RMSEs were 7.452847/7.454228 mm.
The small single-seed difference does not settle whether suppression improves
generalization or extended-loop behavior after recurrent training.
The previous frozen 16-loop diagnostic provides motivation, not a training
target to optimize selectively or independent evidence from an untouched test.

## Dataset proposal

Derive a new, separately archived bundle from
`results/cx-sugar-decoder-data-20261112`, preserving its existing examples and
split identities. Add independently generated repeated-return families:

| Split | New families | New trajectories | Proposed total trajectories |
| --- | ---: | ---: | ---: |
| Training | 24 | 288 | 2,848 |
| Validation | 12 | 144 | 464 |
| Test | 24 | 288 | 800 |

- Each new family supplies 4-loop and 8-loop versions, forward/rightward body
  traversal, and no-sugar/single-delivery/repeated-delivery schedules: 12 related
  examples per family. No pauses are added to this first many-loop extension;
  the retained two-loop examples already include pause controls.
- Use randomized orientations, speeds of 0.5–2 mm/s, clockwise/counterclockwise
  traversal, and regular polygons with 4, 8 or 24 sides. Initial loop periods
  of 6 or 12 seconds keep segment boundaries aligned with the 0.0025-second
  integration grid. Training durations then span 24–96 seconds. Stratify shape,
  period and turn direction where family counts permit.
- Keep every duration, traversal and delivery variant of a family in one split.
  Check exact protocols and geometric parent identities for leakage. Fresh test
  families at trained loop counts assess generalization to new path parameters.
- Add a separate 16-loop extrapolation split from those test families: 144
  examples, 96–192 seconds. Their 4/8-loop relatives remain in test, not training
  or validation. These are related tests of extension, not independent samples.
  Retain the already inspected paths 8/9/14/15 as historical diagnostics.

This is a restricted periodic-path experiment, not a demonstration of general
path integration. Additional irregular trajectories and open-ended departures
would be useful challenges after the first controlled comparison.

## Sugar delivery and fixed physiology approximation

Use the same gate kernel and hDeltaG relaxation constants as the existing run.
Retain the original anchor and initialize activity once per trajectory. Give
neither the network nor decoder true position, loop count or a closure flag.

For new long examples, environmental reentry into the 0.1 mm patch after
departure beyond 0.15 mm defines eligible contact. Single delivery uses the
first contact; repeated delivery uses each nonfinal return contact. The new
repeated-delivery protocol deliberately removes the previous 30-second minimum
interval, which would otherwise skip most returns. This is an explicit delivery
assumption, not an inferred physiological refractory period. Deliveries refresh
the gate through maximum aggregation, without unbounded additive suppression.
The retained two-loop examples keep their original event schedules.

## Loss balance needs explicit attention

Equal sampling weight does not imply comparable task-error contributions.
For the current training set, under a zero-vector predictor and its exact
family-weighted sampling, the MSE contributions are:

| Examples | Sampling mass | Contribution to total MSE (mm squared) |
| --- | ---: | ---: |
| Existing other paths | 36.37% | 1.74974 |
| Existing long straight walks | 13.63% | 115.85889 |
| Two-loop returns, all arms | 50.00% | 9.96189 |

Thus long straight walks account for approximately 90.8% of this initial
task-error baseline; return examples account for approximately 7.8%. These
are target-energy measurements, not a causal decomposition of current gradients
or proof that weighting caused the observed learning behavior.

The proposed successor should distinguish four strata: existing other paths,
existing long straight walks, two-loop returns, and many-loop returns. A concrete
starting recipe is equal sampling mass per stratum, equal family/variant mass
within a stratum, and frozen training-only target-energy normalization:

```text
E_g = family-weighted mean over training trajectories in stratum g
      of time_mean(||correct_home_vector||^2)

task_loss = batch_mean(time_mean(||prediction - target||^2) / E_stratum)
```

All strata have positive target energy; include a declared numerical floor and
derive/freeze scales before any optimization. Do not divide by instantaneous
home-vector magnitude, which approaches zero at returns. Use the same scales
and objective for both dynamics, and apply them to validation selection without
re-estimating scales from validation data. Retain raw errors in mm separately.
Keep the existing regularization coefficients initially, recording that changing
task normalization changes their relative influence.

This introduces both a duration expansion and a loss-balance change. The current
run remains informative, but comparing it with the successor cannot attribute
differences solely to longer examples. If that attribution is needed, add a
shorter-only run with the same balancing policy: 25% existing other, 25% existing
long straight, and 50% two-loop return sampling. It is an additional control,
not a prerequisite for the paired original/sugar comparison on the expanded task.

## Paired training and evaluation

Train both dynamics from the same initialization, using identical sampled
examples and update budgets. Optimize the same recurrent/input-edge coefficients,
population biases/time constants, and decoder/intercept. Keep anatomical support,
PFN activity mapping, and sugar constants fixed. Do not initialize one arm with
the old optimized circuit while initializing the other from scratch.

Use a matched initial budget of 2,000 paired updates, batch size 8, and validation
every 50 updates. These are initial experimental settings, not assurance of
convergence. Select each arm independently using the same predefined balanced
validation criterion. Test/extrapolation and familiar circles do not select
parameters. Start with seed 0; additional matched seeds are needed for claims
about reproducibility across training runs.

Measure raw and normalized errors by stratum, sugar schedule, traversal, loop
number and phase. Include both x/y predictions and signed errors, closure error,
within-loop vector RMSE, mean-error/centered-error decomposition, activity growth
and direct hDeltaG suppression. Always compare with zero-vector prediction:
bounded output and correct closure alone can conceal poor tracking between
returns. No physiological fit or identified reset mechanism is established by
decoding performance.

Benchmark the longest representative batch and a varied workload before a full
launch. A 96-second path has 38,401 states at the fixed integration step, versus
25,601 for the previous 64-second maximum. Full-history backpropagation costs
more memory and compute; record processed steps and wall time. Keep gradients
through every loop. Compilation/checkpointing may change execution, but do not
silently detach histories or enlarge the time step to fit the budget. Equal
updates across dataset versions do not imply equal compute.

## Implementation boundary

Prepare a new generator, bundle, frozen scale table and reproducible experiment
configuration. Tests should cover closures and world-path matching, first/repeated
delivery and hysteresis, cross-split family/prefix protection, training-only
scales, common weights across arms and full-history gradients. The existing
trainer's loss/sampling/selection metadata and implementation must then explicitly
support the new strata and normalized objective; its current printed 50/50
description must not silently mislabel the new task.

The first paired run is now complete. Preserve its bundle and archived runtime
scripts when implementing the successor. For any future concurrent run, do not
edit its imported runtime files: it hashes them and would reject changes at
completion. Stage concurrent implementation in separate files and an isolated
checkout instead.
