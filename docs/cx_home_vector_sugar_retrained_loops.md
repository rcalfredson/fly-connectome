# Extended circular paths after recurrent sugar-response training

## Setup

The completed evaluation is archived at
`results/cx-sugar-retrained-long-loops-20261113`. It reuses the exact protocols,
world-space targets, 0.0025-second step, and delivery schedules from
`results/cx-sugar-long-loops-20261112`: paths 8, 9, 14, and 15 each traverse
16 uninterrupted 8-second loops. One zero-state initialization occurs per trial;
there is no state reset at a return and the original home anchor is retained.

Both networks come from
`results/cx-sugar-paired-training-seed0-20261112-r1`. Each uses its independently
validation-selected **update-1950** recurrent parameters and its own jointly
trained decoder/intercept. Neither the final update-2000 weights nor any newly
refitted readout is substituted. All parameters and response constants remain
fixed during replay.

Four conditions distinguish delivery effects from learned parameter differences:

- `original_trained`: original-trained network, no sugar.
- `sugar_trained_single`: sugar-trained network, one delivery at 8 seconds.
- `sugar_trained_repeated`: the same sugar-trained network and decoder, delivery
  at 8, 16, ..., 120 seconds.
- `sugar_trained_no_sugar`: the same sugar-trained network with its gate disabled,
  as an additional control.

Delivery is explicitly scheduled, as in the prior diagnostic; there is no
30-second minimum interval. Gates combine by maximum rather than addition.
The networks receive PFN activity and, where applicable, the sensory gate;
geometric position and loop count never enter the network or decoder.

## Results at the sixteenth return

| Path | Original-trained closure error (mm) | Sugar: one delivery (mm) | Sugar: repeated delivery (mm) | Sugar-trained, gate disabled (mm) |
| --- | ---: | ---: | ---: | ---: |
| 8 | 16.11758 | 1.13152 | 1.31282 | 24.39603 |
| 9 | 21.45441 | 0.88478 | 0.97202 | 27.90691 |
| 14 | 12.42553 | 1.69207 | 1.78942 | 21.45701 |
| 15 | 21.56391 | 1.39731 | 1.53335 | 27.80526 |

Sugar delivery still substantially limits late readout drift on these examples.
The disabled-gate control is particularly informative: the same sugar-trained
parameters yield much larger errors without the delivered gate. This is a
controlled delivery effect within that trained model, not simply a favorable
difference between two sets of learned parameters.

At loop 16, the sugar-trained network's mean hDeltaG activity is 46.94–57.48
without the gate and 0.425–0.485 with repeated delivery, in model activity units.
The original-trained model has mean hDeltaG activity 25.94–37.66. The added
suppression and recurrent consequences therefore remain substantial after joint
training. These observations do not identify a biological mechanism or establish
physiological calibration.

## Tracking remains imperfect

| Path | Prior frozen circuit + refit: repeated-sugar loop-16 RMSE (mm) | Jointly trained: repeated-sugar loop-16 RMSE (mm) | Jointly trained: single-sugar loop-16 RMSE (mm) |
| --- | ---: | ---: | ---: |
| 8 | 1.60334 | 1.29111 | 1.29697 |
| 9 | 1.58619 | 0.94379 | 0.93650 |
| 14 | 1.11797 | 1.85033 | 1.74241 |
| 15 | 1.20085 | 2.12195 | 1.96894 |

The constant zero-vector predictor has loop RMSE **1.80063 mm** on every path.
Repeated-sugar tracking improves relative to the prior decoder-only refit on
paths 8/9 but worsens on 14/15; on those latter paths it is worse than constant
zero prediction. Thus limiting accumulation does not establish accurate
within-loop home-vector representation. Offsets and waveform distortions remain,
as shown by the component plots. Single delivery gives lower final closure error
than repeated delivery on all four new trials, while per-loop RMSE differences
between schedules are not uniformly in that direction.

This comparison does not show a uniform improvement from joint retraining.
The earlier and new models differ in recurrent parameters, readouts, and the
training/selection objective, so their differences cannot be assigned to a
single mechanism. The current model was trained on at most two return loops
with one delivery; many-loop and repeated-delivery cases remain diagnostic
extrapolation. These familiar circles and one training seed do not establish
generalization, convergence, or stability for arbitrary durations.

## Artifacts, reproduction, and verification

The report contains full-duration plots and loop-16 x/y predictions and signed
component errors for all four paths. Dashed purple in the full-duration plots
is the additional disabled-gate control. `previous_comparison.png` contrasts
the prior frozen-circuit/refitted-decoder closure errors (dashed) with the newly
trained models (solid). `loop_metrics.csv` retains all 256 path/condition/loop
measurements, including raw component errors, a zero-output baseline and
population activity. `previous_comparison.csv` aligns the primary three
conditions with the previous experiment. NPZ files retain full-resolution rates,
predictions, targets, time and gates. Final-loop time averages exclude the
zero-duration endpoint, which is scored separately as closure error.

```bash
OPENBLAS_NUM_THREADS=1 python -m fly_connectome.home_vector.analysis.sugar_retrained_loops \
  --training results/cx-sugar-paired-training-seed0-20261112-r1 \
  --reference results/cx-sugar-long-loops-20261112 \
  --output results/cx-sugar-retrained-long-loops-20261113
```

Use a fresh output directory. The command validates each selected checkpoint
against validation history and saved selection metadata, verifies anatomy and
response constants, and requires targets/times to match the original replay.
Each model independently reproduces its native first loop before sugar; the
two different models are not required to have identical starting-loop activity.
Model parameters and all original-input hashes are checked unchanged afterward.

Tests cover independent checkpoint selection rather than final-weight loading,
rejection of inconsistent checkpoint metadata, and the distinction between zero
closure error and useful tracking. The new tests and existing long-loop/training
tests passed together: 11 tests. Independent checks reproduce all 256 loop RMSE
and closure values from saved predictions, reconstruct predictions from rates
and the selected decoders, and verify that single/repeated schedules produce
identical predictions through the second return.

![Path 14 full duration](../results/cx-sugar-retrained-long-loops-20261113/path_14.png)

![Path 14 component close-up](../results/cx-sugar-retrained-long-loops-20261113/path_14_loop_16_components.png)
