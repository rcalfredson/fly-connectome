# Completed long-path training: 500-update assessment

Reviewed November 5, 2026. The run completed all 500 updates and selected
update 500 by the combined validation objective. It passed the half-step
numerical convergence gate. Broader long-path exposure has improved some
long-path errors, but the decoder still has a saturating response and large
regressions on short trajectories, pauses and loops. This is not yet a
successful home-vector integrator across the task range.

## Figures and machine-readable results

- [500 updates versus the short-trained hDeltaC fit](../results/cx-home-vector-long-u500-comparison-20261105/comparison.png) · [PDF](../results/cx-home-vector-long-u500-comparison-20261105/comparison.pdf)
- [125 versus 500 updates on the same expanded task](../results/cx-home-vector-long-u125-u500-comparison-20261105/comparison.png)
- [Individual-run diagnostic traces, including pause x components](../results/cx-home-vector-long-u500-predictions-20261105/diagnostics.png)
- [Straight-path duration summary, including 96/128-second extrapolation](../results/cx-home-vector-long-u500-assessment-20261105/straight_duration_summary.png)
- [Unified challenge table CSV](../results/cx-home-vector-long-u500-assessment-20261105/challenge_comparison.csv)
- [Complete assessment summary JSON](../results/cx-home-vector-long-u500-assessment-20261105/summary.json)

Result artifacts are ignored local files, not supplied by Git. The reference
is the same four-population hDeltaC circuit trained on the original task,
selected at update 1950. Both long-path pilots started afresh at seed 0.
The 125- and 500-update pilots share batch size 128 and the expanded task;
the old fit used batch size 8 and the original task. Thus the comparison does
not isolate dataset coverage from optimization schedule.

## Matched challenge results

| Measurement | Unit | Short-trained fit | 125-update pilot | 500-update fit |
|---|---|---:|---:|---:|
| Long path: predicted magnitude after 64 mm | mm | 10.690 | 1.862 | 30.097 |
| Long path: whole-trajectory vector RMSE | mm | 28.280 | 35.664 | 18.416 |
| Long path: final vector error | mm | 53.310 | 62.495 | 35.724 |
| Five-second pause after backward walking: final vector error | mm | 0.119 | 1.964 | 1.912 |
| Five-second pause after forward walking: final vector error | mm | 0.098 | 2.276 | 2.671 |
| Five-second pause after backward walking: endpoint drift | mm | 0.258 | 6.414 | 5.501 |
| Five-second pause after forward walking: endpoint drift | mm | 0.316 | 4.096 | 7.575 |
| Square loop: final vector error | mm | 0.084 | 4.233 | 12.200 |
| Same world path, forward walking: moving-phase vector RMSE | mm | 0.380 | 2.341 | 9.776 |
| Same world path, sideways walking: moving-phase vector RMSE | mm | 0.534 | 2.290 | 8.048 |
| Reversal: error at actual return home | mm | 0.521 | 6.364 | 2.944 |
| Reversal: final error after pause | mm | 0.071 | 0.910 | 0.201 |

### The larger long-path output is not a longer accurate integration range

The 64-second challenge is constant forward walking at 1 mm/s and heading
45 degrees, held out from the added long-path training examples. The new
prediction grows too quickly and settles near 30 mm:

| Elapsed time / true magnitude | Predicted magnitude | Full-vector error |
|---|---:|---:|
| 1 s / 1 mm | 6.412 mm | 5.583 mm |
| 2 s / 2 mm | 10.884 mm | 9.096 mm |
| 8 s / 8 mm | 26.332 mm | 18.814 mm |
| 16 s / 16 mm | 29.646 mm | 14.796 mm |
| 32 s / 32 mm | 30.090 mm | 8.192 mm |
| 64 s / 64 mm | 30.097 mm | 35.724 mm |

The descriptive onset of sustained 10% magnitude underestimation moves from
7.45 to 33.44 mm. Here that statistic is misleading as an accuracy range:
it ignores the severe early overestimation and directional error. The new
final vector is (-25.997, -15.166) mm, versus truth (-45.255, -45.255) mm.
A larger response amplitude reduces some long-path losses without producing
continuous accumulation of displacement.

A CPU replay matches saved GPU predictions within 1.15e-5 mm. Hidden-state
change over the final ten seconds is negligible at float32 precision; the
local decoded-velocity estimate is about 0.00074 mm/s versus the required
1 mm/s. This supports saturation on this trajectory. The local Jacobian
relaxation estimate is about 4.08 s; it is not a universal memory time constant.

### History dependence has changed, but pause accuracy is poor

Unlike the 125-update pilot, the two opposite-history predictions have not
collapsed to one value after five seconds. They end near (+3.907, +0.134) and
(-4.514, +0.903) mm, separated by 8.456 mm rather than the required 4 mm.
Their pause-onset errors are already 7.41 and 10.21 mm. Much of the subsequent
decrease in error is relaxation from those oversized responses, not evidence
of a stable correct memory. Five-second endpoints are finite-time samples;
10-second endpoints differ further and must not be called steady states.

Square-loop closure is markedly worse. A few endpoint statistics nevertheless
improve: the reversal final error falls relative to the 125-update pilot,
and sideways same-world-path final error is 0.534 mm, lower than the old
2.023 mm. These do not establish improved trajectory tracking: sideways
moving-phase RMSE is 8.048 mm, compared with 0.534 mm in the old fit.

## Validation tradeoff at completion

Scores are equal-trajectory squared error divided by the inherited L²,
L = 2.045091886931525 mm. The old reference was evaluated on the added long
validation paths in the earlier audited snapshot; that checkpoint hash is
verified before reusing its scores here.

| Validation subset | Short-trained reference | Update 125 | Update 350 | Update 500 |
|---|---:|---:|---:|---:|
| Original 128 trajectories | 0.04693 | 2.14735 | 9.27361 | 6.97117 |
| Added 64 long paths | 105.86224 | 157.35926 | 70.53360 | 62.09786 |
| Combined 192 trajectories | 35.31870 | 53.88466 | 29.69361 | 25.34673 |

The final fit improves long-path MSE by 41.3% and combined MSE by 28.2%
relative to the reference on the same expanded validation set. Original-task
MSE remains about 149 times higher. However, it has recovered by 24.8% since
update 350: the earlier divergence is not monotonically worsening through
the end of this run. Both subsets improve between updates 350 and 500.

Validation improves from 25.869 to 25.347 over the final 25 updates, about
2.0%. The run stopped at its update limit, not because of a demonstrated
plateau. The last selection objective alone does not establish convergence
or imply that extending again will recover the original behaviors.

On the untouched original 256 test trajectories, normalized RMSE is 2.923,
versus about 0.250 in the old fit. The full expanded test NRMSE is 0.607;
these use different target distributions and must not be conflated. Added
long test paths have NRMSE 0.593 and equal-trajectory vector RMSE 19.729 mm.
Thus the short-task regression appears on held-out data as well as validation.

## Direction/speed sweep and longer-duration evaluation

The frozen sweep covers four speeds, eight headings and ten body directions:
320 factorial combinations, plus 40 same-world-path cases. Each rollout lasts
128 seconds, with summaries at 16, 32, 64, 96 and 128 seconds. Summaries share
one continuous rollout, with no resets at the prefix boundaries. Evaluation
uses saved weights, no fitting, and the existing NumPy float32 recurrence;
each sweep verifies a replay against the saved long challenge.

| Movement-combination partition | Conditions | 64 s median vector RMSE, old → new | 128 s median vector RMSE, old → new |
|---|---:|---:|---:|
| Assigned to long training | 256 | 17.68 → 15.06 mm | 44.92 → 39.25 mm |
| Assigned to long validation | 32 | 17.09 → 15.69 mm | 43.85 → 40.15 mm |
| Assigned to long test | 32 | 17.01 → 14.61 mm | 43.90 → 39.85 mm |

The changes are not confined to the one plotted long challenge: median
long-path error is lower across all three partitions at these durations.
At 16 and 32 seconds, however, the new median error is higher in all three
partitions, despite those durations being included in long-path training.
However, errors still grow strongly with duration. Median late along-track
gain is effectively zero at 64 and 128 seconds for the new fit (ideal: 1),
including on trained movement combinations. Improved error does not mean the
model continues integrating throughout those bouts. At 96/128 seconds all
movement combinations are outside the training duration range, even those
whose heading, speed and body direction were trained.

The initial all-speed sweep process was terminated with exit 143 before
writing results and is marked failed. Its incomplete directory is excluded.
All four smaller per-speed sweeps completed and were combined for this
assessment; together they cover the full intended grid.

## Numerical and integrity checks

The half-step gate passes: state relative RMS difference is 1.24e-4 and
relative Cartesian-RMSE change is 1.16e-4. The independent prediction audit
reconstructs geometric targets, reproduces saved metrics and checks split
protocol overlap. Completed run/source bundles are verified and hashed;
comparison audits confirm the inputs remain unchanged. These checks support
interpreting the observed model behavior but do not turn the single-seed
pilot into a capacity or biological conclusion.

The assessment bundle stores `summary.json`, the challenge CSV, duration
plots and `reproduce_assessment.py`. Recreate the summary from its verified
inputs, from the repository root, with a fresh output path:

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fly-connectome-mpl PYTHONPATH=. \
python results/cx-home-vector-long-u500-assessment-20261105/reproduce_assessment.py \
  results/cx-home-vector-long-u500-assessment-NEW
```

Prediction audits use `analysis predictions`; challenge plots use
`analysis.compare` with explicit run/source paths and labels, plus
`--allow-task-differences` only for the short-task comparison. The 125/500
comparison uses identical tasks and needs no opt-in. The four straight sweeps
use `analysis.straight --duration 128 --sample-times 16 32 64 96 128`, each
with one of `--speeds 0.25`, `0.5`, `1`, `2`. Their verified bundles are named
`results/cx-home-vector-long-u500-straight-s025-20261105`, `s05`, `s1`, and `s2`
with the same prefix/date convention.

The next experiment can test objective balance and optimization settings,
using separate subset curves and full challenge traces. The present results
show a higher-amplitude saturating solution with some long-path benefit,
not an intrinsic impossibility proof and not a uniformly successful integrator.
