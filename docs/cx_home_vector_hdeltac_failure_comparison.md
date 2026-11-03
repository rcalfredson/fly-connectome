# hDeltaC pilot: quantified improvements and failure mechanisms

Reviewed 3 November 2026. This compares the selected checkpoints from
`gpu-initial-v1-anatomy-seed0` (update 2,000) and
`cx-home-vector-hdeltac-anatomy-seed0-2026-11-01` (update 1,950).
Both completed 2,000 optimization updates, using identical task/trajectory files
and the same PFN firing prescription. Original result bundles were preserved.
This is an exploratory comparison of one fit per architecture, not an estimate
of across-seed uncertainty or a demonstrated biological role for hDeltaC.

## Main finding

The expanded model is more accurate on ordinary test trajectories and several
short-range challenges, but saturates at a substantially shorter distance under
sustained forward walking. It learns useful cancellation and history retention
without learning a reliable integrator across all tested durations and inputs.
There is no single ordering of the two models that captures all these properties.

## Turning the visual notes into explicit measurements

The [unified companion table](cx_home_vector_hdeltac_comparison_table.md)
combines these metrics with pause endpoint errors, drift and square-loop
component closure errors in one table for use beside the comparison figure.

| Measurement | Original | With hDeltaC |
| --- | ---: | ---: |
| Test normalized RMSE | 0.3261 | 0.2504 |
| Random-test normalized RMSE | 0.3774 | 0.2930 |
| Random-test median direction error, true distance ≥1 mm | 12.65° | 8.60° |
| First sustained 10% magnitude underestimation on long path | 15.15 mm | 7.45 mm |
| First sustained 20% magnitude underestimation on long path | 19.52 mm | 9.49 mm |
| Predicted magnitude after a true 64 mm displacement | 20.60 mm | 10.69 mm |
| Square-loop x-component RMSE | 0.350 mm | 0.297 mm |
| Square-loop y-component RMSE | 0.553 mm | 0.115 mm |
| Square-loop return-home error | 1.062 mm | 0.084 mm |
| Reversal turnaround lag | 0.3125 s | 0.1025 s |
| Further negative-x excursion after reversal | 0.121 mm | 0.011 mm |
| Reversal error at the true return time | 0.498 mm | 0.521 mm |
| Reversal error after the subsequent five-second pause | 0.928 mm | 0.071 mm |

The long-path distance criterion is deliberately narrower than “maximum
trackable distance”: after displacement reaches 2 mm, find the first interval
of at least 0.5 s with predicted magnitude below 90% or 80% of truth. It ignores
overestimation and direction error, which must be assessed separately. These
thresholds were chosen descriptively after viewing the pilots. They are not
preregistered gates. The two long protocols yield the same crossing distances.
The body movement is forward with fixed heading 45°; travelled distance equals
true home-vector magnitude. The 64 mm endpoint is a finite-time observation,
not an exact asymptotic limit.

Reversal lag is the time from the 2 s switch to the minimum decoded x during
the following backward-walk interval. The extra excursion measures how far x
continues in the old direction after that switch; it does not include an error
already present at the switch. Faster reversal and better late closure can
coexist with slightly worse accuracy at the instant of true return.

## Pauses: drift is different from accuracy

For each opposite-history challenge, the target is ±2 mm along x. Define drift
as the Euclidean distance between the prediction at pause onset and at pause
end. Define error relative to the true vector separately.

| Five-second pause measurement | Original | With hDeltaC |
| --- | ---: | ---: |
| Drift after backward walking | 0.320 mm | 0.258 mm |
| Final error after backward walking | 0.294 mm | 0.119 mm |
| Drift after forward walking | 0.752 mm | 0.316 mm |
| Final error after forward walking | 0.735 mm | 0.098 mm |
| Error at pause onset after forward walking | 0.121 mm | 0.385 mm |

Thus the expanded model's larger initial forward-walk overshoot partly relaxes
toward the correct vector. Movement of its prediction during the pause is
still a failure of ideal constancy, but should not be equated with increasing
error. Conversely, a stable prediction can retain a pre-existing error.
Looking only at x misses transverse drift. The new comparison figure plots
full-vector error for both histories.

Backward walking is the negative body-forward prefix, which produces a
positive home-x target. Forward walking produces a negative home-x target.
The colored trace's displacement sign is not the walking-direction label.

The improvement is not universal across pause durations. After a one-second
pause, final errors are slightly worse with hDeltaC: backward 0.121 → 0.157 mm,
forward 0.263 → 0.499 mm. After ten seconds, both improve but still drift:
backward final error 0.870 → 0.418 mm, forward 1.538 → 0.465 mm.

The five-second pair's late-pause vector RMSE is 0.500 → 0.122 mm in the existing
CPU perturbation audits. Resetting state at pause onset gives identical outputs
for the two histories; acute recurrence removal leaves only about 0.00003 mm
of terminal separation. This supports dependence on recurrent movement history,
not an input-only or purely instantaneous explanation. It does not show that
hDeltaC itself is necessary; no selective hDeltaC intervention was performed.

In these pause protocols heading changes from 0° to 90° at movement cessation.
Holding the home vector should tolerate that, but observed drift can combine
responses to the heading transition with imperfect retention. A matched pause
without a heading change would isolate these effects more cleanly.

## Same world path, different body motions

| Measurement | Original | With hDeltaC |
| --- | ---: | ---: |
| Forward-walk vector RMSE during movement | 0.335 mm | 0.380 mm |
| Sideways-walk vector RMSE during movement | 0.952 mm | 0.534 mm |
| RMS separation of predictions during movement | 1.108 mm | 0.713 mm |
| Final error, forward case | 2.338 mm | 1.833 mm |
| Final error, sideways case | 2.558 mm | 2.023 mm |
| Separation of the two final predictions | 2.538 mm | 0.640 mm |

The smaller forward/sideways discrepancy is partly a genuine sideways
improvement; movement-phase forward accuracy worsens slightly. Both final
errors improve, but remain large relative to the 4 mm displacement. Predictions
can agree with each other while sharing an incorrect magnitude, so invariance
and absolute accuracy must both be reported.

## What the long-path failure is doing internally

CPU replay matches the saved long-path predictions within 0.000006 mm. With
constant heading and velocity, the PFN input is constant. At 64 s:

| Diagnostic | Original | With hDeltaC |
| --- | ---: | ---: |
| Hidden-state fractional change during the last 10 s | 0.736% | 0.168% |
| Magnitude of decoded instantaneous velocity from the rate equation | 0.0226 mm/s | 0.00254 mm/s |
| Correct home-vector speed | 1 mm/s | 1 mm/s |
| Largest real eigenvalue of final active-set continuous Jacobian | -0.0146 /s | -0.0883 /s |
| Associated local linear relaxation time | 68.4 s | 11.3 s |

The hidden state is itself changing very little; the output plateau is not
solely a graphical or magnitude-decoding effect. Both final active sets have
locally relaxing dynamics, more strongly so in the expanded fit. These are
local diagnostics at the observed final state. They do not prove global
stability, give a universal memory time constant, or show that all conditions
share the same active neurons. In particular, the local relaxation time must
not be substituted for a population's intrinsic tau.

This explains how better pause memory and worse sustained integration can
coexist: the recurrent dynamics and active ReLU set depend on the input and
state. Retaining an already established small vector under pause input does
not guarantee continued accumulation under constant walking input.

## Separating candidate causes

**Training/task evidence.** Of 1,024 training trajectories, 682 are matched-pause
examples and 342 are random. The longest moving control interval is 2 s; total
trajectory duration never exceeds 16 s. Maximum home-vector magnitude per
training trajectory has median 1.97 mm, 95th percentile 5.47 mm and maximum
12.25 mm. A 64 s uninterrupted walk is therefore a strong temporal and spatial
generalization test. Lower ordinary validation error can select a model with
poorer extrapolation. The present evidence does not establish that optimization
has converged or that additional updates alone would solve this problem.

**Input evidence.** The known current-encoder ambiguity concerns particular
sideways/backward velocities. This long challenge uses forward body translation,
where both forward-preferring groups respond; it is not that counterexample.
The known ambiguity therefore cannot explain all observed failures. More
movement information could help other cases, but does not by itself establish
an accumulating dynamical mechanism. Heading changes during pauses and input-
dependent transients remain distinct input-interface questions.

**Dynamic-network evidence.** The long-path replay directly shows a tendency
for activity to settle under continued movement. This implicates the learned
dynamics/readout solution for this condition, rather than a target-integration
bug or missing training input after some distance. The graph, initialization,
loss, parameter bounds and training distribution jointly selected this
solution; one run does not isolate which imposed the limitation. Added neurons
also introduce parameters and additional routes for the existing PFN signals.
A performance difference is not by itself evidence for a specific anatomical
computation by hDeltaC.

**Square y improvement.** It is substantial, but does not yet establish a
special y mechanism. Rotate the square and reverse its traversal order with
weights fixed: does the improvement follow the world axis, the body-motion
sequence, or a particular turn/history? This is an inexpensive next diagnostic
before assigning the effect to a cell type or decoder component.

## Recommended next experiments

1. Frozen-checkpoint tests first: rotate and reverse loops; compare pauses with
   and without a heading change; examine full-vector error and reversal
   transients. These distinguish coordinate and input-transition effects.
2. Training-distribution control: hold circuit/input encoding fixed, train on
   new longer uninterrupted movements and diverse pauses, and reserve fresh
   long-duration holdouts for confirmation. Keep old challenges as exploratory
   diagnostics once they influence model design.
3. Input control: hold circuitry and training distribution fixed while changing
   the encoding or adding a justified movement channel. An ideal-kinematic-input
   model is a useful computational control, explicitly outside the biological
   input-boundary claim.
4. Dynamics control: under a fixed input/task, test a memory-supporting dynamics
   change or initialization. Altering intrinsic tau alone can slow relaxation
   without creating an integrator. Multiple optimization seeds and matched
   budgets are needed before claiming a reliable architecture advantage.

These should be separate factors. Changing inputs, task duration and recurrence
at once could improve performance without resolving the present attribution
question. No new parameter fitting or selective intervention was done here.

## Artifacts, replay and validation

- [Comparison figure](../results/hdeltac-seed0-comparison-final-20261103/comparison.png)
  and [PDF](../results/hdeltac-seed0-comparison-final-20261103/comparison.pdf).
- [Machine-readable comparison](../results/hdeltac-seed0-comparison-final-20261103/comparison.json)
  and [validation-history plot](../results/hdeltac-seed0-comparison-final-20261103/validation_history.png).
- [Input/code hashes and completion manifest](../results/hdeltac-seed0-comparison-final-20261103/manifest.json).

The versioned comparison tool reads explicit bundles, verifies matching tasks,
checks source integrity before/after analysis, refuses existing output folders,
and runs optional long-path replays on CPU. Original bundles remain unchanged.
Result artifacts are ignored local files and are not supplied by Git.

```bash
python -m fly_connectome.home_vector.analysis.compare \
  --reference-run results/gpu-initial-v1-anatomy-seed0 \
  --reference-source results/cx-home-vector-data-20261030 \
  --run results/cx-home-vector-hdeltac-anatomy-seed0-2026-11-01 \
  --source results/cx-home-vector-hdeltac-data-20261101 \
  --output results/hdeltac-comparison-NEW --replay-dynamics
```

Synthetic checks distinguish drift from accuracy, transient reversal lag from
late closure, isolated threshold breaches from sustained underestimation, and
component error from full-vector error.
