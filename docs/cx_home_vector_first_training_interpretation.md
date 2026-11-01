# First trained home-vector pilot: metric audit and interpretation

Reviewed 31 October 2026. Source: the complete, hash-verified
`results/gpu-initial-v1-anatomy-seed0` run, selected checkpoint update 2,000.
This analysis did not retrain parameters or modify the original run/data bundle.

The [home-vector claim assessment](cx_home_vector_claim_assessment.md) organizes
these findings around the specific claim, what is supported, and what the model
does well where full home-vector tracking fails.

The [analysis replay guide](cx_home_vector_analysis_walkthrough.md) documents
the versioned commands that reproduce these audits with explicit bundle paths.
Historical result links below refer to separately stored local artifacts.

## What the validation numbers mean

The final history value **0.0773096** is `equal_trajectory_scaled_mse`, not
normalized RMSE. It averages squared Cartesian error within each trajectory,
then across trajectories, dividing by the fixed training-set length scale
squared (L = 2.04509 mm). Taking its square root gives 0.27805, an RMS error
relative to that particular training scale.

The separately reported `normalized_rmse` pools evaluated timestamps and divides
total squared vector error by total squared target-vector magnitude before
taking a square root. Validation NRMSE is **0.32859**; held-out test NRMSE is
**0.32611**. Validation Cartesian RMSE is 0.69362 mm; test is 0.69966 mm.
These are distinct aggregations, not inconsistent implementations. NRMSE is
not a percentage of correct predictions or a per-timestamp relative error.

NRMSE = 1 is the error of always predicting a zero vector on the same set. The
trained test result improves substantially over both that reference and the
frozen 45-cell baseline (NRMSE 0.93289), but it does not pass the proposal's
provisional NRMSE <=0.20 gate. The test late-five-second-pause score 0.23101
also exceeds its <=0.20 gate; square closure residual 1.06187 mm exceeds the
provisional 1 mm criterion. These gates were design choices, not biological
thresholds, and should not be revised merely to declare this pilot successful.

## What was audited

The source dataset manifest hash matches the training record. The checkpoint,
predictions, targets and run artifacts pass their manifest hash checks.
Recomputed prediction metrics reproduce the saved values. An independent
piecewise-segment integration of heading/body velocity reproduces every saved
validation, test and challenge target within 1.91e-6 mm maximum absolute error
(float32 storage precision). No exact protocol copies were found between the
training, validation and test splits.

Code inspection confirms that the forward model takes only PFN traces, initial
neural state and integration settings. It contains no target/position/time input;
the affine decoder reads current rates without a recurrent decoder state. These
checks address the inspected code/data paths, not every conceivable shortcut.
The existing half-timestep validation also passes: relative rate RMS difference
0.000424 and task-RMSE change 0.0000672.

## Where the improvement occurs

| Held-out test subset | Trajectories | Cartesian RMSE (mm) | NRMSE |
| --- | ---: | ---: | ---: |
| All | 256 | 0.700 | 0.326 |
| Straight movement followed by matched five-second pause, entire trajectories | 170 | 0.308 | 0.184 |
| Random multi-segment movement | 86 | 0.984 | 0.377 |
| Final second of the matched five-second pauses | 170 | 0.409 | 0.231 |

Two-thirds of test trajectories are the simpler matched-pause construction.
The rest demand accumulation across multiple movements/heading changes. Report
both, since one aggregate obscures their different difficulty. Pooled metrics
give longer trajectories more samples; checkpoint selection weights trajectories
equally. Most samples within a trajectory are correlated, not independent trials.

## Evidence for retained history

The designated challenge pair travels to opposite positions, then receives
identical heading/zero-velocity inputs for five seconds. True home vectors at
the end are (2,0) and (-2,0) mm. Predictions are approximately (1.747,-0.149)
and (-1.592,-0.611) mm: distinct and substantially aligned with the two targets.
Their predicted separation is 3.371 mm versus the true 4 mm.

Acute diagnostics on the same frozen checkpoint and stateless decoder:

| Condition | Final prediction separation (mm) | Last-second NRMSE |
| --- | ---: | ---: |
| Unchanged checkpoint | 3.371 | 0.250 |
| Reset all dynamic rates at pause onset | 0 | 1.001 |
| Remove recurrent contributions from the start of replay | 0.000027 | 1.001 |

The unmodified CPU replay matches saved GPU predictions within 1.08e-6 mm.
This supports retention of history in the simulated neural state and reliance
on recurrence in this trained solution. Acute recurrence removal also changes
the operating point; it is not a substitute for retraining the no-recurrence
architecture under a matched optimization budget. Nor does it identify a
particular biological motif or population as necessary.

## Failures relevant to a fixed-home vector

- **Long-path saturation:** at 32 seconds of straight 1 mm/s movement, predicted
  length is 18.77 mm versus 32 mm. At 64 seconds it is 20.60 versus 64 mm.
  Correct direction alone would conceal most of this error.
- **Pause drift:** with the same true vectors of length 2 mm, ten-second pause
  predictions shrink to lengths 1.33 and 1.17 mm, with angular drift as well.
- **Closure:** the square ends at home but the predicted vector has length
  1.062 mm. The reversal challenge ends with a 0.928 mm residual.
- **Coordinate/decomposition sensitivity:** two equivalent world paths have
  the same final target (-4,0) mm, but forward versus sideways body motion
  produces final errors 2.338 and 2.558 mm and different predicted vectors.
  The PFN encoder's known decomposition differences remain; do not remove them
  through response alignment to improve these results.

All three intrinsic time constants reached approximately the allowed upper
bound of 0.5 s. This indicates pressure toward slower intrinsic responses under
these bounds; it does not demonstrate a measured biological timescale or prove
that increasing the bound would solve general integration. There were 212 edge
sign changes relative to initialization; learned signs remain permissive
computational parameters, not measured physiological signs.

## Interpretation and next discriminating tests

The result supports a trained, finite-horizon displacement-history representation
decodable from this recurrent rate network. It does not yet establish an accurate
duration-independent home-vector integrator, a special benefit of CX wiring,
a biological mechanism, or behavioral homing.

Keep this checkpoint frozen for evaluation of pause invariance, inverse-path
cancellation, equal displacement reached through different paths and repeated
loops, matched moving suffixes, world-motion decompositions, heading rotations,
and duration/distance shifts. Broaden the small challenge set before making
general claims; distinguish exploratory diagnostics from fresh confirmatory
holdouts after any model/task revisions. Test separately retrained no-recurrence
models, degree/type-matched rewires and multiple training seeds under equal
optimization/selection budgets.

Machine-readable audits, plotting scripts and figures are saved separately:

- `results/gpu-initial-v1-anatomy-seed0-analysis-20261031/`: saved-prediction,
  geometry, subset and matched-history-difference audit, plus diagnostic PNG/PDF.
- `results/gpu-initial-v1-anatomy-seed0-perturbations-20261031/`: acute frozen
  checkpoint state-reset/recurrence diagnostics, source hash and replay code.

For individual straight-path subsets a Cartesian target component can have zero
or effectively zero variance. Its component R² is then undefined or numerically
unstable and should not be interpreted; Cartesian errors remain meaningful.
