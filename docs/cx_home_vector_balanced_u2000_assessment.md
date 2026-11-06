# Balanced-loss training close-out and dynamics study

Reviewed November 6, 2026. The unchanged balanced absolute-vector recipe ran
500 updates initially and 1,500 additional updates from `last.pt`, reaching
**2,000 cumulative updates**. Validation selected **update 1975**, which is the
model assessed below. Update 2000 is retained in `last.pt`; its selection score
was worse. Vector-change supervision remained disabled. No new training or
model-development change was made for this assessment.

## Fixed model and evaluated artifacts

The dynamic circuit contains hDeltaB (19), vDeltaE (18), hDeltaG (8) and hDeltaC
(20): 65 rate units. PFNd/PFNv provide prescribed inputs. The prepared dataset
is `results/cx-home-vector-hdeltac-long-data-20261104` and the completed run is
`results/cx-home-vector-hdeltac-long-balanced-seed0-u2000-resumed-2026-11-05`.

Existing tools generated:

- [Completed prediction audit](../results/cx-home-vector-balanced-u2000-predictions-20261106/audit.json): saved metrics reproduce and independent geometric targets agree.
- [Selected home-vector component traces](../results/cx-home-vector-balanced-u2000-challenges-20261106/home_vectors.png).
- [Challenge comparison with the short-trained hDeltaC reference](../results/cx-home-vector-balanced-u2000-challenges-20261106/comparison.png).
- [Joint long-path view](../results/cx-home-vector-balanced-u2000-challenges-20261106/long_path.png).
- [Quantitative challenge table](../results/cx-home-vector-balanced-u2000-challenges-20261106/challenge_metrics.csv).
- [Full challenge report and checkpoint identities](../results/cx-home-vector-balanced-u2000-challenges-20261106/challenge_metrics.json).

The full trainer numerical gate passed: relative state RMS difference at half
step was 0.0012083; relative Cartesian RMSE change was 0.0006106. The separate
challenge replay used CPU, with no fitting or parameter changes.

## Held-out performance

All normalized MSEs below are equal-trajectory means of squared Cartesian
error divided by the same inherited L², with L = 2.045091886931525 mm.
Original validation and test membership is retained exactly in the expanded
bundle. Long-subset scores are not directly comparable to original-subset
scores as measures of relative reconstruction quality because target scales
are very different; the balanced selection criterion applies frozen weights.

| Measure | Earlier short-trained hDeltaC | Balanced update 1000 | Balanced selected update 1975 |
| --- | ---: | ---: | ---: |
| Original validation normalized MSE | 0.0469264 | 0.0277614 | 0.00932791 |
| Added-long validation normalized MSE | Not part of that fit's saved validation set | 10.6561 | 3.92852 |
| Original test normalized MSE | 0.0475658 | Not evaluated in that snapshot | 0.0115527 |

Relative to the earlier short-trained model, original validation MSE is about
80.1% lower and original test MSE about 75.7% lower. Equal-trajectory vector
RMSEs for the selected model are 0.198 mm (original validation), 0.220 mm
(original test), 4.053 mm (long validation) and 4.396 mm (long test).
The raw subgroup values were recomputed from the saved prediction archives
using the existing `subset_summary` definition; no additional model fitting or
new selection based on test performance was performed.

This is one seed. Comparison with the short-trained fit changes the training
distribution, batch size and optimization schedule as well as the objective;
there is no matched 2,000-update unbalanced control here. These results show
what this fitted system can do, not which intervention alone caused each gain.

The final scheduled validation at update 2000 had original/long normalized
MSEs 0.0108701 / 4.69869 and balanced score 0.0338228. Update 1975 scored
0.00932791 / 3.92852 and 0.0285172. This explains selection of the earlier
checkpoint; it does not establish a capacity floor or a permanent plateau.

## Challenge behavior and remaining limitations

| Measure | Short-trained reference | Balanced update 1000 | Balanced selected update 1975 |
| --- | ---: | ---: | ---: |
| 64-second straight-path vector RMSE (mm) | 28.280 | 8.962 | 7.244 |
| 64-second straight-path endpoint vector error (mm) | 53.310 | 21.854 | 14.126 |
| Final error after backward walk + 5 s pause (mm) | 0.119 | 0.351 | 0.027 |
| Final error after forward walk + 5 s pause (mm) | 0.098 | 0.223 | 0.141 |
| Square-loop endpoint error (mm) | 0.084 | 0.664 | 0.357 |
| Reversal + pause endpoint error (mm) | 0.071 | 0.567 | 0.424 |
| World-equivalent forward moving-phase RMSE (mm) | 0.380 | 0.109 | 0.036 |
| World-equivalent sideways moving-phase RMSE (mm) | 0.534 | 0.232 | 0.376 |

Five-second pause endpoint drifts for the selected model are 0.065 mm after
backward walking and 0.126 mm after forward walking. These differ from final
target errors because some error is already present at pause onset. The square
and reversal trajectories have lower whole-trajectory RMSE than the old fit
(0.167 versus 0.318 mm; 0.327 versus 0.381 mm), despite worse final closure.
Likewise, later training did not improve every measure relative to update 1000:
the sideways moving phase worsened while its full trajectory RMSE improved
(0.542 versus 0.723 mm). Metrics and complete traces should be retained together.

The long challenge is one held-out control: forward walking at 1 mm/s with
heading 45 degrees. The selected model's vector errors at 10, 20, 32 and 64 mm
are approximately 0.675, 2.830, 5.400 and 14.126 mm. At 20 mm this is about
14.2% relative vector error; at 64 mm it is about 22.1%. Predicted terminal
magnitude is 78.04 mm for a true 64 mm, compared with 10.69 mm in the old fit.
The final ten-second magnitude-growth slope is 1.297 mm/s for true 1 mm/s.
Sustained accumulation has replaced the old early plateau, but over-integration
remains. This single condition does not establish a universal accurate range,
including for sideways walking or across headings/speeds.

Recommended wording for the log:

> The balanced-loss circuit exhibits approximate path integration, retention
> of distinct movement histories, and partial cancellation around closed paths.
> Accuracy depends on movement direction, displacement, history and metric;
> residual pause drift, loop/reversal closure errors and long-path overgrowth
> remain. Near-zero output after a loop is evidence of cancellation, not yet
> evidence for an explicit reset/zeroing mechanism. About 2 cm is a useful
> reference displacement on one probe, not a demonstrated network-wide range.

## Next: characterize the frozen solution

Pausing model development to examine the selected fit is well motivated.
Freeze update 1975, its decoder, PFN mapping and task definitions before
interpreting activity. Start with the existing challenge set and controlled
variants that vary speed, direction, heading and pauses independently. These
are diagnostic probes, not extra training data.

1. **Observe activity and inputs through events.** Save all 65 rates alongside
   PFN activity, target and decoded vectors; align to starts, stops, reversals
   and turns. Plot per-neuron traces/heatmaps and population summaries. Increasing
   mean activity is not required for integration: displacement could be stored
   in opposing or distributed patterns.
2. **Separate neural-state changes from readout changes.** The decoder is a
   fixed instantaneous linear map, `h = D r + d`; it contains no independent
   temporal state. During a pause, `delta_h = D delta_r`. Decompose activity
   changes into the decoder row space and null space to distinguish a genuinely
   stationary state from changing activity that leaves the decoded vector
   nearly fixed. Compare matched pauses after opposite histories with identical
   current PFN inputs. Report each population's actual `D_population r_population`
   contribution as well as its activity; decoder weight magnitude alone is not
   a causal importance measure.
3. **Locate accumulation and retention mechanisms.** Along constant-control
   paths, compare activity with displacement, speed and elapsed time using
   matched-distance and matched-duration probes. Examine input drive, recurrent
   drive, leak, activation changes and local slow/unstable directions. Local
   linearization is descriptive; do not assume a line attractor or a global
   stability result before checking behavior over the relevant states.
4. **Test causal necessity with controlled perturbations.** Resetting state,
   perturbing selected populations and removing selected recurrent pathways
   without refitting the decoder can test the observed mechanism. Use matched
   perturbations and controls for generic activity disruption. Existing
   `analysis perturbations` supports reset-at-pause and a recurrence-free replay,
   but its recurrence removal applies to the entire trial. It does not isolate
   recurrence's contribution during retention; pause-onset-only removal would
   require a targeted extension. Existing `analysis network` provides parameter,
   projection and decoder summaries rather than full activity dynamics.
5. **Compare specific signatures with physiology.** After identifying the
   computational role, compare cell-type-resolved tuning, movement/heading
   dependence, memory dynamics, interaction signs and timescales with published
   measurements. Separate measured constraints, assumptions, unmeasured
   properties and contradictions. A good decoder or plausible-looking activity
   pattern alone does not establish a biological replica; freely learned edge
   signs and gains particularly require inspection. Identify a concrete mismatch
   before choosing new physiological constraints or another training change.

The sensible goal is to identify which navigation-relevant computations and
physiological correspondences the model supports, not to decide from a single
fit whether it is a full replica of the central complex. Closed-loop homing,
robustness outside these probes, and variability across training seeds remain
separate questions.

## Reproduce the existing evaluations

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fly-connectome-mpl \
python -m fly_connectome.home_vector.analysis predictions \
  results/cx-home-vector-hdeltac-long-balanced-seed0-u2000-resumed-2026-11-05 \
  --source results/cx-home-vector-hdeltac-long-data-20261104 \
  --output results/cx-home-vector-balanced-u2000-predictions-NEW

OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fly-connectome-mpl \
python -m fly_connectome.home_vector.analysis.checkpoint \
  --run results/cx-home-vector-hdeltac-long-balanced-seed0-u2000-resumed-2026-11-05 \
  --source results/cx-home-vector-hdeltac-long-data-20261104 \
  --reference-run results/cx-home-vector-hdeltac-anatomy-seed0-2026-11-01 \
  --reference-source results/cx-home-vector-hdeltac-data-20261101 \
  --allow-task-differences \
  --label "Balanced final selected" --reference-label "Short-trained hDeltaC" \
  --threads 1 --output results/cx-home-vector-balanced-u2000-challenges-NEW
```

Choose fresh output names. The first command audits the saved selected-checkpoint
predictions; the second independently replays that checkpoint and the reference
on CPU, recording identities and hashes. It evaluates only challenges, not the
validation/test subsets. Result artifacts remain local ignored files; the report
and reproducible commands can be committed separately from evaluator code.
