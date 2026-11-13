# Evaluation plan: reliable home-sugar recurrent training

This plan is specified while the seed-0 run is still training. Familiar circles
and early training diagnostics have already been inspected; this is not a claim
of formal preregistration or wholly blind analysis. No additional model outputs
are needed to define the comparisons below.

The machine-readable specification is
`docs/data/cx_home_vector_reliable_evaluation_plan.json`. It identifies the sealed
dataset and representative plots before completed-run results are examined.

## Questions and fixed choices

Assess two separate questions: does suppression limit activity accumulation,
and does it improve accurate home-vector tracking? Neither divergent training
losses nor bounded activity alone answers the second question.

Use `results/cx-sugar-reliable-return-training-seed0-20261113` only after its
manifest is complete. Independently load each arm's validation-selected best
checkpoint. Check its update and validation score against history and saved
selection metadata. Selection uses the recorded normalized `selection_score`,
not minimum raw RMSE or the final update. Preserve its jointly trained decoder
and intercept. Do not refit readouts, tune sugar constants, select another
checkpoint on test performance, or change the original home anchor.

Use the exact sealed protocols and contact events from
`results/cx-sugar-reliable-return-data-20261113`. Keep the saved training-only
target energies, physical integration step and family weights. Each trial starts
at zero once; all returns within it retain continuous neural state.

## Conditions

| Condition | Parameters and decoder | Gate during replay | Purpose |
| --- | --- | --- | --- |
| Original | Selected original-trained checkpoint | Always zero | Primary comparator |
| Sugar | Selected sugar-trained checkpoint | Every recorded home reentry | Primary intervention |
| Sugar, gate disabled | Same selected sugar checkpoint | Always zero | Same-weights intervention control |
| Zero-vector prediction | No network | Not applicable | Guard against apparently good closure with poor tracking |

The disabled-gate replay starts from the same zero state and follows the same
movement as the normal sugar replay. The resulting histories can differ; this
measures the effect of the gate over the whole trial. It is an out-of-training
intervention, not an additional training condition. A difference establishes
gate influence in that model, not superiority over the original-trained model.

If an individual missed-delivery test is later added, omit that event's kernel
while retaining the causal effects of previous events. Do not instantly set
the existing slow-recovery gate to zero and call it a single missed delivery.
That additional robustness experiment is outside the first evaluation pass.

## Evaluation sets

1. **Held-out test: 512 movements.** Report the whole set and all four strata:
   existing other paths, long straight walks, retained two-return paths, and
   diverse 4/8-return paths. Report route class, body traversal, return count,
   and ending at home versus away where metadata supports them.
2. **Extrapolation: 48 sixteen-return movements, 96–192 seconds.** They extend
   the 24 held-out diverse families; their shorter relatives are in test, not
   training. Do not treat related extensions as additional independent families.
3. **Controls: 4 near-miss/pause-away cases and 12 historical challenges.** Check
   that proximity or a pause is not mistaken for a reset. Retain correct nonzero
   vectors when a trajectory ends away from home.
4. **Eight familiar diagnostics and the historical sixteen-loop circles.** These
   illustrate behavior but are not fresh generalization evidence. For the circles,
   preserve the old movement protocol but derive all contact events using the
   current reliable policy. Any comparison to an older figure must identify
   event-timing/selection/objective differences. A strictly matched historical
   model comparison would require replaying both models under the same schedule.

For 4/8/16-return relatives, verify actual controls before claiming an identical
prefix: open-ending short examples may depart during the interval when a longer
relative remains home before its next excursion. Compare only genuinely shared
prefixes as such; retain whole-trajectory scores for each declared task.

## Metrics and aggregation

Report the primary held-out normalized task MSE and the diverse-return stratum's
raw vector RMSE together. Also report all four stratum scores; do not choose
whichever subset favors sugar after inspecting results.

```text
trajectory_MSE = time_mean(||predicted_vector - correct_vector||^2)
normalized_MSE = weighted_mean(trajectory_MSE / training_energy[stratum])
vector_RMSE_mm = sqrt(weighted_mean(trajectory_MSE))
delta = sugar_metric - original_metric   # Negative favors sugar for errors.
```

Use equal stratum/family/sibling weights, as in the training task. For a subset
or phase, renormalize these fixed weights among eligible trajectories, and
average physical time within each eligible trajectory/phase. Exclude terminal
zero-duration states from time averages and score them separately. Do not
estimate normalization from test or extrapolation targets. A zero predictor's
normalized score is exactly one on the full training distribution, not
necessarily on these held-out distributions.

Additional measurements:

- Whole-path and moving-only vector error, plus endpoint error. Retain dwell
  performance separately rather than improving the average merely by spending
  more time near zero displacement.
- Error at each actual patch-entry event, using its actual correct vector.
  Record geometric-return error separately at annotated center-return times,
  after verifying the target is approximately zero there. Entry into the finite
  patch does not necessarily mean zero displacement. Legacy two-return times
  can be recovered from the known loop period, pause and terminal metadata;
  do not invent center-return annotations for arbitrary paths.
- Per-excursion RMSE and error versus return number, including the 16-return
  extensions. Split first two seconds after contact from later time using
  nonduplicated union masks when windows overlap.
- Signed x/y errors, their mean error vector and variation around that mean.
  Plot raw predictions; the centered-error summary is not a baseline correction.
- Mean activity per population, maximum single-neuron activity, state norm,
  and activity at returns. Measure hDeltaG suppression and recovery relative
  to the same-weights gate-disabled control. No upper physiological firing cap
  is assumed; model activity units are not calibrated firing rates.

For each condition, include the zero-vector baseline on the same time samples.
For paired comparisons, preserve family identity across models and body/duration
variants. Report family-level paired errors and their distribution rather than
treating time points as independent replicates. A single training seed cannot
establish reproducibility across model initializations; no time-point-based
significance claim is planned.

## Plots and artifacts

Choose representative families by route metadata and IDs, never by error:
regular `test-diverse-return-0000`, out-and-back `0005`, irregular `0002`, and
mixed `0007`. Use forward traversal for first-pass plots, with 8-return examples
and their 16-return relatives. The fixed choices include home and away endings.
Report rightward traversal quantitatively and retain matching plots as needed.

Produce full-duration vector error, x/y predictions versus truth, signed component
errors, per-return error and population activity. Include a final-excursion close-up
and a paired family summary. Tables should include all families so selected
figures cannot conceal heterogeneity.

Archive a report, machine-readable metrics by trajectory/family/subset, zero
baselines, exact protocols/gates, selected checkpoint hashes, verification results,
and full-resolution arrays for representative plots. An independent calculation
from saved predictions should reproduce displayed errors. No decoder or activity
offset should be altered to improve a figure.

## Checks and interpretation

Verify source bundle hashes, selected checkpoint identities, anatomy, fixed
response constants and reliable contact policy. Every replay keeps parameters
unchanged. Check disabled-gate native equivalence and repeatability. Compare
matched body variants' world targets and record their actual PFN inputs/gates.

As a numerical spot check, replay the four preselected sixteen-return examples
at half the step, evaluating the same physical protocol and saved event times.
Compare states/predictions at the original sample times; do not re-detect contacts
and inadvertently change cue timing. Report whether differences between models
are comparable to step-size sensitivity rather than declaring a small contrast
meaningful on its own.

- Better held-out tracking with restrained accumulation supports a useful
  computational role in this tested model and task, not physiological validation.
- Lower activity growth without lower error separates an anti-accumulation
  effect from improved integration.
- Similar strong performance suggests little additional benefit under these
  conditions; similar poor performance requires examining learning curves and
  representational limitations before concluding the mechanism cannot help.
- Opposing route/duration effects should be reported as conditional benefits
  and costs. A favorable long-run figure must not replace the aggregate result.

Do not change the ongoing training or its imported files to prepare this plan.
Implementation of the evaluator is separate work; no completed-run result or
new evaluation is claimed here.
