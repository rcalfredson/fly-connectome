# Long-path pilot: 125-update challenge assessment

Reviewed November 5, 2026. The first batch-128 long-path pilot has completed.
Its selected checkpoint is update 125; the previous short-trained hDeltaC
reference selected update 1950. Both have the same four-population circuit,
PFNd/PFNv encoding, initialization seed, dt and loss scale. They differ in
training distribution, batch size and optimizer-update budget. This is an
interim optimization diagnostic, not a settled comparison of the training
distributions or of network capacity.

The long-path pilot performs substantially worse on the matched challenge
cases. Its validation loss was still decreasing, so these results support
continuing the optimization investigation rather than concluding that long
examples are ineffective.

[Comparison plot](../results/cx-home-vector-long-pilot125-comparison-final-20261105/comparison.png)
· [PDF](../results/cx-home-vector-long-pilot125-comparison-final-20261105/comparison.pdf)
· [Metrics JSON](../results/cx-home-vector-long-pilot125-comparison-final-20261105/comparison.json)
· [Individual-run diagnostic plot](../results/cx-home-vector-long-pilot125-predictions-20261105/diagnostics.png)

## Matched challenge measurements

| Measurement | Unit | Short-trained hDeltaC | Long-path pilot, 125 updates |
|---|---|---:|---:|
| Long straight path: predicted magnitude after 64 mm | mm | 10.690 | 1.862 |
| Long straight path: final vector error | mm | 53.310 | 62.495 |
| Long straight path: onset of sustained 10% magnitude underestimation | mm | 7.453 | 2.077 |
| Five-second pause after backward walking: final vector error | mm | 0.119 | 1.964 |
| Five-second pause after forward walking: final vector error | mm | 0.098 | 2.276 |
| Five-second pause after backward walking: endpoint drift | mm | 0.258 | 6.414 |
| Five-second pause after forward walking: endpoint drift | mm | 0.316 | 4.096 |
| Square loop: x-component RMSE | mm | 0.297 | 2.194 |
| Square loop: y-component RMSE | mm | 0.115 | 1.801 |
| Square loop: final vector error | mm | 0.084 | 4.233 |
| Same world path, forward walking: moving-phase vector RMSE | mm | 0.380 | 2.341 |
| Same world path, sideways walking: moving-phase vector RMSE | mm | 0.534 | 2.290 |
| Reversal: error at true return home | mm | 0.521 | 6.364 |
| Reversal: final vector error after pause | mm | 0.071 | 0.910 |

Long-path onset uses the existing descriptive criterion: at least 0.5 seconds
more than 10% below true magnitude, considering displacements beyond 2 mm.
It is not a maximum fully accurate distance. The long challenge is 1 mm/s at
global heading 45 degrees with forward body motion, deliberately held out
from the added long training examples. It therefore probes generalization
as well as sustained integration.

## What the traces show

**Long path:** the new decoder quickly approaches a magnitude of about
1.86 mm and gives the same endpoint at 32 and 64 seconds. Its decoded output
has settled rather than continuing to grow with displacement. The previous
fit also saturates, but near 10.69 mm on the 64-second case. Magnitude alone
misses directional error: the new endpoint is (-1.837, -0.305) mm, while the
correct vector is (-45.255, -45.255) mm.

**Opposite histories, identical pause inputs:** the two correct terminal
vectors are (+2, 0) and (-2, 0) mm. The new predictions both approach about
(+0.166, +0.702) mm, separated by only 0.00010 mm after five seconds, instead
of the required 4 mm. The old predictions remain separated by 4.195 mm.
This is strong evidence that the current decoder loses the distinction
between these histories during the pause. It does not establish that every
hidden-state trace of history has disappeared. A stable but incorrect shared
endpoint should not be counted as successful memory retention.

**Square loop:** component errors are large throughout and the final vector
remains 4.23 mm from home. This is worse both as a trajectory estimate and as
a closure estimate; it is not merely a different transient on the way to an
accurate endpoint.

**Same world path, different body motions:** forward and sideways errors
become similar during the final pause because the decoder approaches nearly
the same wrong vector, about (+0.842, -0.345) mm, when truth is (-4, 0) mm.
Agreement between those two predictions does not by itself demonstrate
correct body-direction invariance.

**Reversal:** the reported turnaround lag is zero in the pilot, versus
0.1025 seconds previously. That isolated value must not be interpreted as
an improvement: error at the actual return home grows from 0.52 to 6.36 mm.
A poorly shaped response can score well on this narrow timing statistic.

## Optimization and numerical checks

Validation scaled MSE was 63.63, 61.27, 59.04, 56.33 and 53.88 at updates
25, 50, 75, 100 and 125. The final interval improves by 4.3%, with no clear
plateau. Equal sample exposure has not yielded comparable performance on these
challenges; both the training distribution and optimizer schedule changed.

The numerical convergence gate passed: halving dt changed the validation
rate trajectories by a relative RMS of 3.19e-5 and Cartesian RMSE by a
relative 1.63e-5. The prediction audit independently checked geometric
targets and reproduced saved metrics. Thus there is no numerical-gate
failure to explain away the observed challenge errors.

Do not directly compare whole validation/test scores between the old and
expanded datasets. Only the unchanged challenge protocols are directly
matched here. The larger validation set legitimately includes longer,
higher-magnitude targets; its value is useful for tracking this run's own
progress and selecting its checkpoint.

A longer optimization budget is motivated by the still-improving validation
curve. These exploratory challenge results provide a baseline for that next
comparison. They do not establish an intrinsic anatomical limit, nor isolate
the effects of large batches, loss balance and duration coverage.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fly-connectome-mpl \
python -m fly_connectome.home_vector.analysis predictions \
  results/cx-home-vector-hdeltac-long-anatomy-seed0-2026-11-05 \
  --source results/cx-home-vector-hdeltac-long-data-20261104 \
  --output results/cx-home-vector-long-pilot125-predictions-NEW

OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fly-connectome-mpl \
python -m fly_connectome.home_vector.analysis.compare \
  --reference-run results/cx-home-vector-hdeltac-anatomy-seed0-2026-11-01 \
  --reference-source results/cx-home-vector-hdeltac-data-20261101 \
  --run results/cx-home-vector-hdeltac-long-anatomy-seed0-2026-11-05 \
  --source results/cx-home-vector-hdeltac-long-data-20261104 \
  --allow-task-differences \
  --reference-label "Short-trained hDeltaC" \
  --label "Long-path hDeltaC (125 updates)" \
  --output results/cx-home-vector-long-pilot125-comparison-NEW
```

The opt-in allows different datasets only with byte-identical circuits,
identical dt and loss scales, and exactly matching challenge metadata and
protocols. Each run still matches its own verified source. Comparisons use
`reference`/`current` keys with explicit model labels; the default matched-task
architecture comparison retains its historical `original`/`hdeltac` keys.
The validation-history figure explicitly flags that its curves use different
validation sets. Audits hash their inputs and confirm they are unchanged.
Result files are ignored local artifacts, not distributed with Git.

Sixteen focused comparison, audit and straight-sweep tests passed; the new
guard tests reject unrequested dataset differences and mismatched challenge
protocols, circuits, time steps or normalization scales.
