# Planned successor: learning over diverse returns

The active primary delivery policy and launch command are now specified in
[cx_home_vector_reliable_return_training.md](cx_home_vector_reliable_return_training.md):
the sugar arm receives every eligible home-patch reentry and the original arm
always disables its gate. Schedule variants and counts below are a record of
the earlier planning stage, not the current primary training specification.

Status: dataset construction and engineering checks are implemented; no full
scientific training run has started. Realized counts, normalization and commands
are in [cx_home_vector_diverse_return_training.md](cx_home_vector_diverse_return_training.md).
The completed seed-0 paired run, its source files, original encoder, frozen
checkpoint and all archives remain unchanged.
The scope was broadened on 13 November 2026 from repeated regular loops to
diverse excursions and returns. Regular loops remain one reference class.

## Question

Can original and sugar-response dynamics learn useful home-vector tracking
when training explicitly includes diverse repeated returns and post-sugar continuation?
The existing run asks about shorter-return training; the proposed successor
asks about longer-duration training. Neither asks that their losses diverge.

The initial seed-0 run completed 2,000 paired updates and independently selected
update 1950 for both arms. Original/sugar held-out test vector RMSEs were
5.743460/5.675652 mm; selected validation RMSEs were 7.452847/7.454228 mm.
The small single-seed difference does not settle whether suppression improves
generalization or extended-loop behavior after recurrent training.
The previous frozen 16-loop diagnostic provides motivation, not a training
target to optimize selectively or independent evidence from an untouched test.
Evaluation of the selected jointly trained models now also shows that delivery
limits long-loop drift, including in a same-weights gate-disabled comparison,
but does not uniformly improve tracking relative to the earlier decoder-only
refit. See `docs/cx_home_vector_sugar_retrained_loops.md`. This is consistent
with limited training coverage but does not isolate a decoder deficit: both
recurrent state dynamics and their readout may need to adapt. Longer and more
varied training is a test of that possibility, not an assurance of improvement.

## Dataset proposal

Derive a new, separately archived bundle from
`results/cx-sugar-decoder-data-20261112`, preserving its existing examples and
split identities. Add independently generated repeated-return families:

| Split | New families | New trajectories | Proposed total trajectories |
| --- | ---: | ---: | ---: |
| Training | 24 | 288 | 2,848 |
| Validation | 12 | 144 | 464 |
| Test | 24 | 288 | 800 |

- Each new family supplies 4-return and 8-return versions, forward/rightward body
  traversal, and no-sugar/single-delivery/repeated-delivery schedules: 12 related
  examples per family. The counts above remain the initial proposed budget.
- Balance the new families across four route classes: regular polygon loops
  for continuity with earlier experiments; retraced out-and-back excursions;
  irregular multisegment excursions with different outward and homebound routes;
  and sequences mixing excursions of different shapes, extents and durations.
  A sequence's later excursions need not repeat its first excursion.
- Vary world orientation, approach direction, speed (initially 0.5–2 mm/s),
  and clockwise/counterclockwise turns where applicable. Retain forward/rightward
  variants of the same world path. Initially target total durations of 24–96
  seconds with all segment boundaries on the 0.0025-second integration grid.
  Exact duration choices and realized counts must be sealed in the generated
  manifest before training; irregular routes need not share a common lap period.
- Include families with brief home dwell periods, and others that leave home
  again and end away from it, within the same overall duration budget. Both
  networks and every delivery variant of a family must have identical movement
  and pause schedules. Keep the original short pause controls as well.
- Keep every duration, traversal and delivery variant of a family in one split.
  Check exact protocols and geometric parent identities for leakage. Fresh test
  families at trained loop counts assess generalization to new path parameters.
- Add a separate 16-return extrapolation split from those test families: initially
  144 examples, targeting 96–192 seconds. Their 4/8-return relatives remain in test, not training
  or validation. These are related tests of extension, not independent samples.
  Retain the already inspected paths 8/9/14/15 as historical diagnostics.

Include held-out near-miss trajectories that pass close to home without entering
the patch and pauses away from home. These are additional diagnostic controls,
not a source of training/selection feedback. Keep related duration prefixes,
rotations, body-direction variants and delivery variants within one split;
splitting timestamps or obvious geometric copies does not create independent
test families. Diverse route classes still do not establish general path
integration or generalization beyond the chosen movement distribution.

## Sugar delivery and fixed physiology approximation

Use the same gate kernel and hDeltaG relaxation constants as the existing run.
Retain the original anchor and initialize activity once per trajectory. Give
neither the network nor decoder true position, loop count or a closure flag.

For new long examples, environmental reentry into the 0.1 mm patch after
departure beyond 0.15 mm defines eligible contact. Single delivery uses the
first contact; repeated delivery uses the first and later eligible contacts with subsequent
continuation. Initial occupancy does not deliver sugar. No-sugar variants follow
the identical movements but receive no deliveries. The new
repeated-delivery protocol deliberately removes the previous 30-second minimum
interval, which would otherwise skip most returns. This is an explicit delivery
assumption, not an inferred physiological refractory period. Deliveries refresh
the gate through maximum aggregation, without unbounded additive suppression.
The retained two-loop examples keep their original event schedules.
Contact is an effective delivery/consumption approximation. Do not use mere
proximity or geometric closure to change neural state or the target. Contact
at the patch edge can leave a small nonzero correct home vector. Geometry may
be used to construct a route and environmental contacts, but only the resulting
prescribed PFN activity and explicit sensory gate enter the network.

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
existing long straight walks, two-loop returns, and diverse repeated returns. A concrete
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

This introduces duration/route expansion and a loss-balance change. The current
run remains informative, but comparing it with the successor cannot attribute
differences solely to longer examples or route diversity. If that attribution is needed, add a
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
