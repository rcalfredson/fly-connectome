# Is the decoded output a home vector? First-pilot claim assessment

31 October 2026 · Run: `gpu-initial-v1-anatomy-seed0` · Selected update: 2,000

## Assessment for the log

**Partially supported, within a limited tested regime.** The trained recurrent
network supports a stateless readout that approximates the fixed-home vector
on held-out trajectories and retains displacement-dependent information during
five-second pauses. This is a substantial improvement over the frozen-network
and current-input baselines. However, the readout drifts during longer pauses,
underestimates large accumulated displacement, leaves residual vectors after
returning home, and depends on how the same world movement is generated.
The demonstrated capability is an approximate movement-history representation
useful for short-range home-vector prediction. A general, stable home-vector
integrator, a unique alternative computation, and a biological implementation
have not been established.

The subsequent [network-change audit](cx_home_vector_network_changes.md)
quantifies how much the trained effective coupling and intrinsic parameters
departed from the contact-count initialization.

## 1. What claim is being assessed?

Home is established at trajectory initialization. The supervised target is the
Cartesian world-frame vector **from the current position back to that fixed
home**, in millimeters. The decoder is an affine map of the current 45 neural
rates. The network stores any history; the decoder supplies no temporal memory.
Thus “the decoder tracks home” means that its output remains a useful estimate
of this geometric target as the network receives movement-related inputs.

A home-vector computation should jointly satisfy these behavioral properties:

| Property | What the output should do |
| --- | --- |
| Follow displacement | Update in the opposite direction to world-frame movement, with appropriate magnitude |
| Preserve history | Distinguish different displacements even when current heading, velocity and elapsed time agree |
| Remain stable without translation | Keep the same vector during a pause, including when heading changes |
| Cancel inverse movements | Return toward zero after movements that undo the outbound displacement |
| Depend on net displacement | Give the same final vector for equal final displacement reached through different histories |
| Respect world/body coordinates | Decode equivalent world paths consistently despite different heading/body-velocity decompositions |
| Generalize over task scale | Preserve these relationships across the claimed ranges of duration, distance and speed |

No single aggregate score establishes all of these properties. A constant-zero
predictor can pass a closure test; an opposite-current-velocity predictor can
point home during straight outbound movement. The combined tests matter.

## 2. Evidence and scope

This report uses the hash-verified completed training run, the full ridge baseline
run, the saved-prediction/geometry audit, and the acute frozen-checkpoint
perturbation audit. It adds descriptive statistics computed from existing test
predictions, with no refitting or change to the original results.

The target audit independently reconstructed geometric displacement, reproducing
all validation/test/challenge targets within 1.91e-6 mm. Recomputed metrics match
the saved values. Code inspection confirms that position, target and elapsed time
are absent from the neural forward inputs. No exact protocol copies were found
between training, validation and test splits. The numerical half-step check
passes. These checks support the validity of the inspected measurement pipeline;
they do not rule out all task-distribution shortcuts.

The experiment is a permissive capacity model: prescribed PFN inputs drive 45
dynamic hDeltaB/vDeltaE/hDeltaG bodies; signs and per-edge efficacies can train,
while anatomy fixes the selected edge mask and informs initialization. It is
neither a complete CX model nor a model constrained by measured physiological
efficacies. Only one training seed has been assessed here.

## 3. Where the home-vector claim is supported

### Held-out Cartesian prediction improves substantially

| Condition | Test Cartesian RMSE (mm) | Test normalized RMSE |
| --- | ---: | ---: |
| Current PFN inputs, trained ridge decoder | 2.076 | 0.968 |
| Original frozen hDeltaB model, trained ridge decoder | 2.046 | 0.954 |
| Frozen 45-cell network, trained ridge decoder | 2.001 | 0.933 |
| Trained 45-cell network and decoder | **0.700** | **0.326** |

Validation NRMSE is 0.329, close to test NRMSE 0.326. On these trajectories,
training the dynamics and decoder therefore provides a real generalizing
improvement over fitting readouts from the fixed features. The comparison does
not by itself identify which physiological parameters or anatomical motifs
produce that improvement.

The 0.0773 validation value in the training log is a **scaled mean-squared error**,
not NRMSE. It uses equal trajectory weights and fixed training length scale
L = 2.04509 mm. The NRMSE column pools timestamps and uses evaluated target
energy. In that metric, 0 is perfect and 1 is the error of always predicting
zero. Neither value is a percentage of correct predictions.

### The readout retains information that current inputs cannot determine

The held-out test set includes 85 matched pairs, each with opposite outbound
displacements followed by identical five-second pause inputs within the pair.
On the final second of these pauses, trained-model NRMSE is **0.231** and
Cartesian RMSE is **0.409 mm**. The earlier frozen baselines lose usable
separation on the designated pause challenge.

In that designated challenge, the final targets are (2,0) and (-2,0) mm. After
the same five seconds of stationary input, trained predictions are approximately
(1.747,-0.149) and (-1.592,-0.611) mm. This is genuine retained history: a
deterministic function of only the matched current inputs could not produce
those distinct outputs.

Acute tests of the same checkpoint further locate this history in the model:

| Condition | Final prediction separation (mm) | Late-pause NRMSE |
| --- | ---: | ---: |
| Unchanged checkpoint | 3.371 | 0.250 |
| Reset dynamic neural state at pause onset | 0 | 1.001 |
| Remove recurrent contributions from the start of replay | 0.000027 | 1.001 |

These are two challenge trajectories, not the full test-set pause score above.
Resetting state eliminates the information needed to distinguish the histories.
Acute recurrence removal also eliminates useful separation, supporting recurrent
reliance in this trained solution. It changes the operating point as well, so it
does not establish that a separately retrained nonrecurrent architecture must
fail, or that any particular cell type or biological motif is necessary.

## 4. Where the stronger claim fails or remains untested

| Requirement | Observation | Current assessment |
| --- | --- | --- |
| Accurate home-vector prediction on held-out task distribution | Test NRMSE 0.326; meaningful but nontrivial errors | Partially supported |
| Retention across identical five-second pauses | Histories remain distinguishable; late-pause test NRMSE 0.231 | Supported approximately, with drift/error |
| Stable retention during longer pauses | For true 2 mm vectors, ten-second-pause predicted lengths shrink to 1.33 and 1.17 mm, with angular drift | Not achieved in these probes |
| Correct large accumulated displacement | Straight-path estimates are 18.77 mm at true 32 mm and 20.60 mm at true 64 mm | Not achieved in these probes |
| Exact cancellation after returning home | Square and reversal residual lengths are 1.062 and 0.928 mm | Approximate cancellation only |
| Consistent world-motion decoding | Equivalent forward/sideways paths have different predictions and endpoint errors 2.338/2.558 mm | Not achieved in this pair |
| Broad path independence and additivity | One square, one reversal and a small challenge set | Insufficiently tested |
| A special contribution of biological wiring | No matched trained-null ensemble assessed here | Not established |
| Biological plausibility or use by the fly | No independent physiological/behavioral validation in this experiment | Not established |

The 32/64-second straight probes change both duration and accumulated distance
beyond the ordinary training examples. They demonstrate an extrapolation limit,
but do not separately identify time-dependent leak versus representational
range, nonlinear compression, or another dynamical effect. The apparent plateau
near 20 mm applies to that input condition; it is not a measured global capacity
bound for every heading, speed or trajectory.

Known differences in the prescribed PFN encoding between world-equivalent body
movements remain part of this experiment. They can contribute to unequal decoded
outputs; these results alone do not localize the error to the encoder or the
recurrent network. Responses should not be aligned afterward to conceal it.

The run does not meet the proposal's provisional overall NRMSE <=0.20,
late-five-second-pause NRMSE <=0.20, or square residual <=1 mm criteria. These
are computational acceptance choices rather than biological thresholds. They
should remain visible alongside the substantial improvements.

## 5. What is it doing well, including where it is not an accurate home vector?

### A. Remembering the direction of a simple displacement

The clearest capability is preserving a signed directional displacement signal
after translation stops. This is more than current-heading or current-velocity
coding, because the matched histories have identical current inputs.

| Held-out test subset | Trajectories | Vector RMSE (mm) | NRMSE | Median angular error |
| --- | ---: | ---: | ---: | ---: |
| Straight outbound movement followed by matched pause, entire trajectories | 170 | 0.308 | 0.184 | 3.66 degrees |
| Random multi-segment paths | 86 | 0.984 | 0.377 | 12.65 degrees |

Angles are evaluated only where true home-vector length is at least 1 mm. In an
additional descriptive calculation, 99.2% of those timestamps in the simple
matched-pause trajectories are within 15 degrees of the correct direction;
their 90th-percentile angular error is 9.05 degrees. This statistic covers the
whole trajectories, not just the pauses. Together with the late-pause results,
it supports good short-range directional retention for this task family.

### B. Approximate displacement magnitude within the simpler task family

The model does not merely classify outbound direction. Across the simple
held-out trajectories, magnitude RMSE is **0.264 mm**, and vector RMSE is
0.308 mm. It can therefore supply useful approximate lengths as well as
directions within that distribution. This does not establish linear distance
scaling across all speeds, durations or body-motion decompositions.

In the long straight challenge, direction stays close to correct while magnitude
becomes severely underestimated. That is a concrete example of a useful
directional output without an accurate home vector. On a straight outbound path,
however, opposite-current-velocity direction would also point home; the matched
pause evidence is needed to establish retained history.

### C. Partial combination of successive movements

Performance on the random multi-segment paths is weaker but still informative:
NRMSE 0.377, vector RMSE 0.984 mm, and median angular error 12.65 degrees.
The model can track some cumulative displacement through changing inputs and
reduces error relative to a zero-vector reference on these paths as well.

This is also where the phrase “right direction, wrong distance” becomes
inadequate. Only **57.0%** of non-near-home random-path timestamps are within
15 degrees, and the 90th-percentile angle error is **35.46 degrees**. Among
random-path timestamps with vector error over 1 mm, only **30.9%** retain a
direction within 15 degrees. Thus many larger misses involve directional error
as well as incorrect magnitude, rather than a single correctable length scale.

These percentages are post-hoc descriptive statistics. The 15-degree and 1-mm
cutoffs were not preregistered success gates, and correlated timestamps are not
independent trials. They characterize observed errors, not a confidence level
or a mechanistic identification.

## 6. What computation should we say it implements?

The most defensible functional description is:

> A trained recurrent movement-history representation with a stateless readout
> that approximates short-range home displacement, retains useful directional
> information over several seconds, and exhibits drift, limited scaling and
> coordinate-dependent errors outside its strongest task regime.

“Leaky integrator,” “compressed displacement code,” and “memory of recent
movement” are plausible explanations for parts of the behavior, but none is
uniquely established. We have not compared their predictions systematically,
estimated a validated effective leak law, or identified the underlying neural
mechanism. All three trained intrinsic time constants approach their allowed
0.5-second maximum; this alone is not an estimate of the effective memory
timescale of a recurrent network.

Likewise, the evidence does not justify a single alternative label such as
“it is only tracking current velocity.” Identical-input pauses directly contradict
that explanation. “It always gets direction right but scales distance wrongly”
is also contradicted by the multi-segment errors. Different conditions reveal
different successful and unsuccessful aspects of the learned computation.

## 7. Next tests that would sharpen the claim

Keep the checkpoint and decoder frozen for these evaluations:

1. **Separate time from distance:** reach the same displacement using different
   speeds/durations, and test multiple pause lengths, headings and displacements.
2. **Test path algebra:** compare equal displacements reached by different
   segment orders/routes, exact inverse paths, and repeated loops. Report both
   accuracy along the path and residuals at home.
3. **Match moving suffixes:** give different prefixes the same later movement
   sequence, matching heading, velocity and elapsed time. The appropriate offset
   between decoded home vectors should persist through that common suffix.
4. **Test coordinate invariance:** vary heading/body-motion decompositions and
   heading changes without translation, preserving the original PFN encoding.
5. **Distinguish candidate dynamics:** evaluate ideal displacement, leaky
   displacement and simple recent-movement/clock-based reference models on
   dedicated holdouts. Any fitted reference parameters should use training or
   validation data, not the final evaluation set.

Then assess independently retrained memory ablations, degree/type-matched
randomized graphs and multiple optimization seeds under comparable budgets.
These address architecture-specific capacity; they are distinct from acute
perturbations of an already trained solution. If the model or task changes in
response to these exploratory findings, use fresh confirmatory holdouts for
stronger claims.

## Sources and reproducibility

The [analysis replay guide](cx_home_vector_analysis_walkthrough.md) provides
versioned commands for the prediction audit, acute perturbations and descriptive
statistics. They accept relocated bundles through explicit run/source arguments.
The historical artifacts linked below remain outside Git.

- [Completed pilot metrics](../results/gpu-initial-v1-anatomy-seed0/metrics.json)
  and [training configuration](../results/gpu-initial-v1-anatomy-seed0/training_config.json).
- [Full ridge baseline metrics](../results/gpu-initial-v1-baselines/metrics.json).
- [Saved-prediction and geometric audit](../results/gpu-initial-v1-anatomy-seed0-analysis-20261031/audit.json),
  [diagnostic figure](../results/gpu-initial-v1-anatomy-seed0-analysis-20261031/diagnostics.png)
  and [exportable PDF figure](../results/gpu-initial-v1-anatomy-seed0-analysis-20261031/diagnostics.pdf).
- [Acute checkpoint perturbations](../results/gpu-initial-v1-anatomy-seed0-perturbations-20261031/perturbations.json).
- [Additional descriptive error statistics](../results/gpu-initial-v1-anatomy-seed0-claim-statistics-20261031/statistics.json).
- [Original metric audit and interpretation](cx_home_vector_first_training_interpretation.md)
  and [experiment proposal](cx_home_vector_proposal.md).

The three analysis directories retain replay scripts and source/output hashes.
Original checkpoints, predictions and task data were preserved. Results are
ignored local artifacts and are not supplied by a fresh Git checkout. The report
records the numerical findings in prose so the assessment remains readable
without those local artifacts. No new training, circuit expansion or commit
was performed for this report.
