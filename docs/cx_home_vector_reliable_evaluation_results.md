# Completed reliable home-sugar comparison

15 November 2026. The seed-0 paired run completed 2,000 updates. Both arms
independently selected update 2,000 by minimum normalized validation MSE.
The evaluation follows [the fixed evaluation plan](cx_home_vector_reliable_evaluation_plan.md),
using each selected network and its jointly trained decoder. No parameters,
response constants, readouts, targets or event times were fitted or changed.

Training: `results/cx-sugar-reliable-return-training-seed0-20261113`.
Dataset: `results/cx-sugar-reliable-return-data-20261113`.
Results: `results/cx-sugar-reliable-evaluation-20261115-v2`.
The earlier output without `-v2` is an incomplete engineering attempt, not a
scientific result. The v2 scoring pass finished before an interruption during
plotting; completion regenerates representative replays and numerical checks
without changing saved all-trajectory scores. Both evaluator sources are archived.

## Primary results

Lower scores are better. Raw vector RMSE and normalized MSE use the fixed
stratum/family/sibling weights. Time averages exclude the terminal state;
endpoint and contact errors are separate measurements.

| Held-out measurement | Original | Sugar | Zero-vector baseline |
| --- | ---: | ---: | ---: |
| Overall normalized MSE | 0.37402 | 0.42493 | 1.01091 |
| Overall vector RMSE, mm | 10.537 | 9.601 | 17.013 |
| Diverse 4/8-return RMSE, mm (96 paths) | 1.704 | 2.086 | 2.741 |
| Diverse 16-return RMSE, mm (48 paths) | 1.799 | 2.293 | 2.834 |

Sugar has 13.6% higher normalized test error, despite 8.9% lower raw overall
RMSE. These are different objectives: large-vector straight walks dominate the
raw squared-distance measure even after giving strata equal total weight.
Diverse-return RMSE is 22.4% higher for sugar on test and 27.4% higher on the
longer extensions. Both models beat the zero predictor on whole-path tracking.
Good closure alone would not establish this: the zero predictor can be exactly
correct at home while failing during excursions.

| Test stratum: RMSE, mm | Original | Sugar |
| --- | ---: | ---: |
| Existing other paths | 0.942 | 1.081 |
| Long straight walks | 20.819 | 18.870 |
| Retained two-return paths | 2.627 | 2.664 |
| Diverse return paths | 1.704 | 2.086 |

Long straight examples have no deliveries; the difference there reflects the
different learned parameters, not an acute benefit of sugar on those walks.
Sugar is worse in aggregate for all four diverse route classes, on both test
and extrapolation. It wins within 3/24 test diverse families and 4/24 extended
families. Related body/duration variants and their extensions are not independent
replicates. This single training seed does not estimate reproducibility.
For forward/rightward traversal respectively, diverse test RMSE is
1.852/1.543 mm for original and 2.063/2.109 mm for sugar; sixteen-return
RMSE is 1.965/1.617 versus 2.253/2.332 mm.

## The sugar signal is being used

Disabling the gate while retaining **the sugar model's own weights and decoder**
causes large errors. This intervention begins at the original zero state and
removes every delivery; it does not switch the gate off midway through a trial.

| RMSE, mm | Original | Sugar | Same sugar weights, gate disabled |
| --- | ---: | ---: | ---: |
| Diverse test | 1.704 | 2.086 | 9.304 |
| Sixteen-return extensions | 1.799 | 2.293 | 39.338 |

Mean hDeltaG activity over the 16-return paths is 1.361 model units in the
original network, 0.482 with sugar, and 25.410 with the sugar gate disabled.
At geometric center returns 1–4 versus 13–16, weighted return RMS errors are:

- Original: 1.646 to 1.513 mm.
- Sugar: 1.722 to 1.820 mm.
- Sugar weights without the gate: 5.161 to 70.435 mm.

Thus the gate strongly stabilizes the dynamics learned by the sugar arm.
It is not being ignored. Removing it is an out-of-training intervention;
the result does not show that the independently trained original network needs
sugar, nor that sugar provides a superior attainable solution.
On the four no-contact near-miss/pause-away controls, sugar and disabled-gate
outputs agree exactly, as expected.

## Relation to the earlier many-loop advantage

Replaying the four familiar 16-circle movements under the current contact policy
gives aggregate RMSE 1.478 mm for original and 1.397 mm for sugar, versus
1.801 mm for a zero predictor. Sugar retains a small advantage on these familiar
motions. They are illustrative diagnostics, not fresh generalization evidence.
The movement matches the historical routes, but current patch-contact delivery
replaces the older scheduled pulses; this is not a strictly matched comparison
with the old model figures.

The crucial change is the comparator: the new original arm was also trained on
diverse repeated returns. It now avoids the large progressive errors seen in
the older original model. Sugar's earlier benefit addressed a failure of that
particular model; it did not establish that suppression must outperform a
network trained to solve repeated-return tasks without it.

## What remains unresolved

The completed comparison establishes a performance trade-off under this task,
training budget, seed, circuit and effective sugar equation. It does not isolate
the best achievable performance of either dynamics.

1. **Optimization is unfinished as a question.** Both best checkpoints lie at
   the final update. Normalized validation errors at updates 1,750 / 1,950 /
   2,000 are 0.4489 / 0.4177 / 0.4131 for original and 0.5152 / 0.5064 /
   0.4966 for sugar. Neither arm has demonstrated convergence.
2. **This is not solely held-out generalization failure.** An additional,
   explicitly post-hoc diagnostic on all 96 diverse training paths gives
   RMSE 1.442 mm for original and 1.842 mm for sugar. It cannot separate
   insufficient optimization, competing objectives, readout fitting or a
   dynamical capacity limit.
3. **Remaining error includes the excursion waveform.** On diverse test paths,
   error variation around each trajectory's own mean error is 1.209 versus
   1.684 mm RMS. In the union of the first two seconds after contact, total
   RMSE is 1.625 versus 1.602 mm; later after contact it is 1.797 versus
   2.365 mm. Training paths also show a larger later-phase error for sugar
   (1.455 versus 1.990 mm). These summaries have different eligible time
   samples and are descriptive, not a causal decomposition.
4. **The gate is prolonged suppression, not an instantaneous reset.** Recovery
   is 300 seconds. Suppression remains active throughout much of a repeated-
   return trial and may trade stability against displacement maintenance or
   updating. Learned recurrent compensation is plausible, especially given
   disabled-gate instability, but this evaluation does not identify that
   mechanism. The equation is our effective approximation, not a demonstrated
   cellular mechanism or a quantitative physiological fit.
5. **Constraints could matter.** Most population time constants reach their
   allowed 0.5-second maximum in both arms. No anatomical coefficient is at
   the magnitude-5 boundary. Bound occupancy warrants investigation; it does
   not establish that widening a bound is appropriate or would favor sugar.

## Recommended successor comparisons

Keep this completed comparison frozen. There is no demonstrated change that
will guarantee sugar outperforms original.

The smallest informative training change is a matched extension of the update
budget, preserving both optimizer states, data, objective and response constants,
with independent validation selection and a predefined stopping budget. The
saved paired checkpoint includes optimizer/RNG states, but a verified resume
path for this paired trainer must be implemented before launching it. This
would test insufficient optimization without simultaneously changing the task.
A second initialization is needed before attributing a small difference to
the dynamics rather than a particular optimization trajectory.

Before changing the loss, localize the same phase-dependent errors in training
and validation. If accurate between-visit displacement is the chosen priority,
a predeclared phase-balanced or displacement-update term could prevent good
near-home performance from hiding poor excursion tracking. Apply the identical
new objective and training budget to both arms. A return-focused objective can
also be justified, but it changes the scientific target and is not evidence
that the present sugar model was already better. Do not choose coefficients
using the current test/extrapolation errors; those sets are now inspected.

A separate, explicitly new readout experiment could fit an optimal linear
readout on frozen training activities, with matched fitting/selection for both
arms. That would ask whether joint decoder optimization is limiting the result;
it would not repair missing information or establish a recurrent mechanism.
No such readout was fitted in this evaluation.

Only after these distinctions should suppression strength/recovery or temporal
bounds be varied. Treat them as sensitivity experiments, retain physiological
plausibility as a separate criterion, and use equal search budgets for both
models. Stronger suppression is not justified by the present result: sugar
already limits activity, while the unresolved difficulty is accurate tracking.

## Artifacts and verification

The result bundle contains trajectory, event, excursion, family and subset CSVs,
zero baselines, selected checkpoint copies/hashes, full-resolution arrays and
full-duration/component plots for the eight prespecified examples and four
familiar circles. Event scores distinguish patch entry from center return.
Post-hoc training-fit and historical-circle diagnostics are explicitly labeled.
Half-step checks preserve saved event times and compare predictions and neural
states on the original time grid. Source-bundle hashes and model parameters
are checked before/after replay. No training or decoder refitting is performed.

The completed bundle passes its artifact hash verification. Independent scoring
of all 32 saved representative prediction arrays reproduces the trajectory
scores; replay reproduces the completed training summary. Native zero-gate
predictions and states agree with saved prefixes. Halving the integration step
changes predictions by at most 0.00494 mm RMS for original and 0.00207 mm for
sugar across the four selected extensions; corresponding RMSE changes are at
most 0.00308 and 0.000763 mm. State-entry RMS differences are at most 0.00172
and 0.00132 model units. The unstable disabled-gate control is more sensitive
(0.0982 mm prediction RMS difference), far below its tens-of-mm error.
Forty relevant scoring, dataset, training and replay tests pass.

Representative plots:

- [Out-and-back, full duration](../results/cx-sugar-reliable-evaluation-20261115-v2/out_and_back_extrapolation.png)
- [Out-and-back, final-excursion components](../results/cx-sugar-reliable-evaluation-20261115-v2/out_and_back_extrapolation_components.png)
- [All prespecified plots and numerical tables](../results/cx-sugar-reliable-evaluation-20261115-v2/report.md)

Reproduce the frozen evaluation in a fresh output directory:

```bash
MPLCONFIGDIR=/tmp/fly-connectome-mpl OPENBLAS_NUM_THREADS=1 \
/home/tracking/miniconda3/envs/py3.13-torch/bin/python \
-m fly_connectome.home_vector.analysis.reliable_evaluation \
--output results/cx-sugar-reliable-evaluation-reproduction
```

The optional `--complete-existing` flag completes plotting/checks only after
all trajectory scores have been saved by an interrupted evaluation. It refuses
a completed bundle. A subsequent code review corrected CSV metadata restoration
so missing values, integer return counts and booleans retain the same subset
labels as a fresh run. Completion also checks event and excursion coverage,
so an interruption between CSV writes cannot silently produce a partial report.
The archived evaluation and its numerical results were not changed.
