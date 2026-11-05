# Original versus added-long validation: update-350 snapshot

Reviewed November 5, 2026. The batch-128 long-path run was still active when
its selected `best.pt` was copied at update 350. Evaluation used that independent
copy on the CPU with one thread, without changing, locking or stopping the
active GPU run. The reference is the short-trained hDeltaC fit's selected
update-1950 checkpoint. Both were evaluated on exactly the same 128 original
and 64 added-long validation trajectories. No test or challenge outcomes
selected this comparison.

## Result

All scores below are equal-trajectory, time-averaged squared Cartesian error,
divided by the same inherited L² (L = 2.045091886931525 mm).

| Validation subset | Short-trained reference, update 1950 | Long-path pilot, update 125 | Active long-path fit, snapshot 350 |
|---|---:|---:|---:|
| Original 128 trajectories | 0.04693 | 2.14735 | 9.27361 |
| Added 64 long straight paths | 105.86224 | 157.35926 | 70.53360 |
| Combined 192 trajectories | 35.31870 | 53.88466 | 29.69361 |

The update-125 column was recomputed from the completed pilot's saved
`validation_predictions.npz`, using the exact retained prefix and appended
paths in its source bundle. Its combined score reproduces the saved validation
MSE. The 350/reference columns come from independent CPU checkpoint evaluation.
The current combined and reference-original scores reproduce their respective
logged selection objectives within CPU/GPU tolerance.

The active fit has genuinely improved the added long-path score relative to
the old fit (33.4% lower MSE), and has a 15.9% lower combined score on this
shared expanded validation set. However, it is much worse on the original
trajectories. The old 0.04693 and new 29.69361 scores originally appeared
incomparable because they used different sets; once both models are evaluated
on the same expanded set, the old score is 35.31870.

In physical units, equal-trajectory vector RMSE is 0.443 mm versus 6.228 mm on
the original subset, and 21.042 mm versus 17.176 mm on added long paths
(reference versus snapshot 350). These RMSEs are square roots of the
trajectory-averaged MSE and should not be confused with the trainer's separate
pooled-over-time `cartesian_rmse_mm` metric.

## What is changing during training

Between updates 125 and 350, the new fit's original-subset MSE grows by 4.32×,
while its long-path MSE falls by 55.2%. Its combined score improves by 44.9%.
Thus the progress is not uniformly improving behavior across the two subsets.
It reflects a measured tradeoff over this interval, not merely the presence
of harder examples in the validation set.

This does not isolate the cause. The new run has a different training
mixture, batch size and optimization budget. Large long-path residuals can
contribute strongly to the squared-error objective even under equal
trajectory weighting. More optimization might change the tradeoff; these
measurements alone do not establish whether the original behaviors will
recover or identify an intrinsic network limit.

The combined score decomposes as:

```text
combined = (2/3 × original) + (1/3 × added_long)
         = 6.18240 + 23.51120
         = 29.69361
```

Long paths account for about 79.2% of the snapshot's combined loss. The
original trajectories are not negligible, but their worsening can coexist
with improvement in the larger long-path term.

It is reasonable to finish the planned 500-update run. A further extension
should consider these subset curves and the desired balance of behaviors,
not just whether the aggregate validation objective is falling. Useful next
instrumentation is to log both validation components at every check. Loss
balance, sampling mixture and learning-rate comparisons remain subsequent
controlled experiments, rather than explanations already demonstrated here.

## Reproduce and audit

```bash
OPENBLAS_NUM_THREADS=1 python -m fly_connectome.home_vector.analysis.validation_subsets \
  --run results/cx-home-vector-hdeltac-long-anatomy-seed0-u500-2026-11-05 \
  --source results/cx-home-vector-hdeltac-long-data-20261104 \
  --reference-run results/cx-home-vector-hdeltac-anatomy-seed0-2026-11-01 \
  --reference-source results/cx-home-vector-hdeltac-data-20261101 \
  --output results/cx-home-vector-validation-subsets-snapshot-NEW
```

A new invocation snapshots whichever selected checkpoint exists then, not
necessarily update 350. The historical update-350 copy and all measured
outputs are preserved in
`results/cx-home-vector-validation-subsets-snapshot-20261105`:

- `current_checkpoint.pt` and `reference_checkpoint.pt`: independent checkpoint bytes.
- `current_metadata.json` and `reference_metadata.json`: copied run metadata/history and exact checkpoint hashes/updates.
- `validation_subsets.json`: subset scores, units, matching checks and definitions.
- `*_per_trajectory.json`: individual trajectory errors for each subset.
- `manifest.json`: verified source/reference hashes, snapshot artifact hashes, code identity and completion status.

The active run is explicitly allowed to evolve; its changing whole directory
is not falsely treated as an immutable completed bundle. The prepared sources
and completed reference bundle are verified before evaluation and checked for
changes afterward. Only stable-looking file copies are read, with bounded
retry on incomplete checkpoint serialization. Validation membership, circuit,
time step, loss scale and checkpoint buffers are checked before evaluation.
Result artifacts are ignored local files, not distributed by Git.

Validation: 12 focused subset-snapshot and existing audit tests passed. They
cover exact loss recombination, running-run acceptance without input writes,
snapshot independence from later checkpoint writes, retained-prefix rejection,
and retries for incomplete serialization. `git diff --check` passed.
