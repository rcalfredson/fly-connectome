# Diverse returns: prepared task and normalized paired training

The active primary goal and launch command are now documented in
[cx_home_vector_reliable_return_training.md](cx_home_vector_reliable_return_training.md).
This document describes the archived dataset preparation and delivery schedules
associated with its own dated bundle, rather than the current primary protocol.

The new dataset is complete at `results/cx-sugar-diverse-return-data-20261113`.
Preparation preserves the shorter task and completed runs. The original PFN
encoder, anatomy, home anchor, sugar-response constants and archived frozen
checkpoint are unchanged. This document records dataset preparation and
engineering checks; it does not claim a completed scientific performance result.

## Notes for the log

- Sugar suppression limits accumulated drift in the extended-loop probes, but
  does not ensure accurate tracking. The successor exposes training to longer
  histories with repeated opportunities to return home and receive sugar.
- Broaden route coverage beyond repeated circles: retain regular-loop controls,
  add retraced out-and-back routes, irregular multisegment returns, and sequences
  of different excursions. Vary speed, approach direction, body traversal and
  pauses, and include departure after the final return so trials can end away
  from home.
- Match all movements and pauses across no-sugar, single-delivery and repeated-
  delivery variants. Sugar is an explicit sensory event at home-patch contact;
  neither contact nor closure zeros state or reanchors the correct home vector.
- Normalize task errors with fixed values derived only from training examples,
  so large straight-walk home vectors do not overwhelm return examples. Preserve
  fresh family-level test splits and longer extrapolation cases.
- This changes route/duration coverage and loss balancing together. The primary
  controlled comparison is original versus sugar dynamics under identical new
  conditions; improvement over the earlier run would not isolate which change
  helped.

## Realized dataset

| Split | Retained examples | Added diverse returns | Total |
| --- | ---: | ---: | ---: |
| Train | 2,560 | 288 | 2,848 |
| Validation | 320 | 144 | 464 |
| Test | 512 | 288 | 800 |

The new families number 24/12/24, equally divided among the four route classes.
Each family has 4-return and 8-return versions, forward/rightward body traversal,
and three delivery schedules: 12 siblings. Test families additionally supply
144 16-return extrapolation examples. These extensions share lineage with test
examples, not training or validation, and are not independent test replicates.
Four near-miss/pause-away diagnostic controls are separate. The old 12 challenge
and 16 familiar diagnostic examples are retained.

New fit/evaluation examples span 24–96 seconds; extrapolation spans 96–192
seconds. Movement speeds lie in 0.5–2 mm/s. Segment times are integer multiples
of the unchanged 0.0025-second integration step. Nonperiodic excursions use
different shapes, durations, orientations and speeds; out-and-back excursions
retrace their outward segment, while regular-loop controls repeat a template.
Home dwell durations are 0, 0.25, 0.5 or 1 second. Open-ending families use
0.5–1 second of their final home interval for an outward movement instead.

All body/sugar siblings have the same world path and target at every sample.
The generator verifies each declared return and rejects unintended extra home
contacts. A canonical parent signature groups the initial four excursions
across duration prefixes and heading/body derivatives. Split checks reject
cross-split families, exact protocols and matching new parent signatures.
Independent random streams use seed 20261113. Existing examples retain their
source protocols, IDs, family identities and split assignments; only stratum
metadata is added.

Contact uses the existing 0.1 mm entry/0.15 mm departure hysteresis. There is no
initial delivery. The new repeated schedule delivers at each eligible return,
including a final return followed by open departure; a terminal return without
further movement is excluded. The new protocol has no 30-second minimum delivery
interval. This approximates consumption on contact, not identified feeding
physiology. The prior examples retain their original schedules. A patch-edge
contact need not have exactly zero correct home vector. The imposed gate still
uses 1-second rise, 300-second recovery, maximum pulse aggregation, and a
1-per-second hDeltaG relaxation term toward zero reference activity.

## Frozen normalization and selection

Sampling assigns equal mass to four strata, then to families and siblings
within each stratum. Target energies are calculated on the training split only,
using uniform time excluding each final zero-duration state:

| Stratum | Training examples | Mean target energy E_g (mm squared) |
| --- | ---: | ---: |
| Existing other paths | 1,024 | 4.811142 |
| Long straight walks | 1,024 | 849.933595 |
| Retained two-loop returns | 512 | 19.923782 |
| New diverse returns | 288 | 6.381247 |

```text
task_loss = batch_mean(time_mean(||prediction - target||^2) / E_stratum)
```

The floor is `1e-8 mm^2`; no energy requires clipping in the realized task.
No instantaneous target norm enters the denominator. A zero predictor has
normalized task loss 1 on the fully weighted training distribution; validation
and test need not have that baseline because they use the frozen training
scales. The same scales and sample indices serve both dynamics.

The loss retains the existing activity, edge-prior and decoder penalties and
parameter bounds, so normalization changes their relative influence. Validation
selection uses `selection_score` (dimensionless mean normalized MSE), not raw
millimeter RMSE. Raw vector/endpoint RMSE and per-stratum metrics remain in
history/results. Raw aggregate scores also change their sampling policy and
cannot be directly compared with the earlier 50/50 task as though nothing else
changed. Legacy datasets still use their prior sampling and raw-RMSE selection.

## Commands and validation

```bash
python -m fly_connectome.home_vector.diverse_returns \
  --source results/cx-sugar-decoder-data-20261112 \
  --output results/cx-sugar-diverse-return-data-20261113

# Completed engineering smoke check; use a fresh output to repeat.
OPENBLAS_NUM_THREADS=1 python -m fly_connectome.home_vector.sugar_training \
  --data results/cx-sugar-diverse-return-data-20261113 \
  --output results/cx-sugar-diverse-return-training-smoke-NEW \
  --updates 2 --batch-size 4 --validate-every 2 --compile-block 32 --smoke

# Completed longest treated batch check; no validation/test selection.
OPENBLAS_NUM_THREADS=1 python -m fly_connectome.home_vector.sugar_training \
  --data results/cx-sugar-diverse-return-data-20261113 \
  --output results/cx-sugar-diverse-return-longest-benchmark-NEW \
  --updates 1 --batch-size 8 --compile-block 32 --benchmark --benchmark-longest

# Prepared full comparison; NOT launched.
OPENBLAS_NUM_THREADS=1 python -m fly_connectome.home_vector.sugar_training \
  --data results/cx-sugar-diverse-return-data-20261113 \
  --output results/cx-sugar-diverse-return-training-seed0-NEW \
  --updates 2000 --batch-size 8 --validate-every 50 --seed 0 \
  --compile-block 32 --device cuda
```

Use the PyTorch environment; CUDA must be available when requested. Engineering
checks ran on CPU. The smoke uses one example per stratum and exercises repeated
delivery, full-history gradients, normalization and checkpoint selection. The
longest-batch benchmark uses eight copies of the longest training trajectory,
preferring the most delivered events on duration ties. Neither is a scientific
fit. No held-out test or extrapolation case is used by the benchmark.

Completed check bundles are `results/cx-sugar-diverse-return-training-smoke-20261113`,
`results/cx-sugar-diverse-return-longest-treated-benchmark-20261113`, and
`results/cx-sugar-diverse-return-varied-benchmark-20261113`. The last contains
four real-distribution paired updates. The longest treated check uses eight
96-second examples with eight deliveries each. The whole process took 23.56 s
with maximum resident memory about 2.08 GiB on CPU, including initialization and
archival work. This is a workload-fit check, not a warmed throughput estimate or
a prediction of total CUDA training time. Resource logs are retained separately
at `results/cx-sugar-diverse-return-benchmark-resources-20261113`.

Focused tests cover route closures, world-path matching, speed bounds, contacts,
open endings, diagnostic controls, family/parent leakage, family/stratum sampling,
zero-predictor normalization, frozen evaluation scales, end-to-end normalized
selection, and selected-checkpoint loading when normalized and raw rankings
disagree. Existing training/decoder/probe tests also pass: **47 tests** total.

The builder archives inputs and hashes, fixed scales, protocols, summary and
its runtime script. Training archives both arms and atomic recovery snapshots.
The preview below is generated only from prescribed paths and gates; it is not
a neural response or a fitted result.

![Diverse routes](../results/cx-sugar-diverse-return-preview-20261113/route_examples.png)

![Delivery gates](../results/cx-sugar-diverse-return-preview-20261113/gate_examples.png)
