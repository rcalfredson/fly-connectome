# Challenge evaluation during training

The reusable CPU evaluator takes an independent snapshot of `best.pt` from a
running or completed training run, replays the standard challenge suite, and
writes plots, metrics and predictions to a fresh directory. No training files
are written or locked, no decoder is refitted, and CUDA is not used. It reuses
the existing validation-snapshot reader and completed-run challenge definitions.

## Evaluate the current extension

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fly-connectome-mpl \
python -m fly_connectome.home_vector.analysis.checkpoint \
  --run results/cx-home-vector-hdeltac-long-balanced-seed0-u2000-resumed-2026-11-05 \
  --source results/cx-home-vector-hdeltac-long-data-20261104 \
  --label "Balanced training" --threads 1 \
  --output results/cx-home-vector-balanced-live-challenges-NEW
```

Each invocation captures whichever **best validated checkpoint** is available
then, not necessarily the latest optimizer update. Update and checkpoint hash
are recorded; plot legends include the actual captured update. A later checkpoint
write cannot affect that evaluation. `NEW` is a placeholder for a fresh run label.
The current trainer does not periodically save every optimizer update; this tool
does not promise arbitrary past checkpoints or a live view of unsaved weights.

## Reload a historical evaluation

Every successful evaluation preserves its checkpoint bytes, metadata and
predictions. Although the training run overwrites `best.pt`, a checkpoint copied
into an evaluation directory remains available. Pass that completed evaluation
directory as `--run` to replay its archived model, even if the original training
directory has moved or is no longer available:

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fly-connectome-mpl \
python -m fly_connectome.home_vector.analysis.checkpoint \
  --run results/cx-home-vector-balanced-live-challenges-900 \
  --source results/cx-home-vector-hdeltac-long-data-20261104 \
  --label "Balanced training" \
  --output results/cx-home-vector-balanced-u900-home-vectors-NEW
```

The matching prepared source is still required and checked against the archived
run manifest. `--snapshot-key current` is the default; choose `reference` to use
the reference checkpoint from a saved comparison instead. A reference input can
also be an evaluation directory, with `--reference-snapshot-key current` or
`reference` selecting its archived model. The corresponding prepared source
must match that selected model. All outputs go into a fresh directory; older
evaluations stay unchanged. Missing models or inconsistent checkpoint identities
are rejected. Snapshot input files are verified and checked for changes.

Saved activity atlases and per-trajectory activity bundles can also supply their
archived checkpoints through `--run`. Their matching prepared source remains
required; renaming a bundle's directory does not change its checkpoint identity.

This is model re-evaluation, not optimizer continuation: the training-resume
interface still takes a completed training bundle. Historical updates without
a copied checkpoint are not directly loadable; history scores alone cannot
reconstruct their weights.

## Compare with a reference

Add a reference run and its matching prepared source to the same command. Both
checkpoints are independently copied and replayed on CPU. For comparison with
the earlier short-trained hDeltaC fit:

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fly-connectome-mpl \
python -m fly_connectome.home_vector.analysis.checkpoint \
  --run results/cx-home-vector-hdeltac-long-balanced-seed0-u2000-resumed-2026-11-05 \
  --source results/cx-home-vector-hdeltac-long-data-20261104 \
  --reference-run results/cx-home-vector-hdeltac-anatomy-seed0-2026-11-01 \
  --reference-source results/cx-home-vector-hdeltac-data-20261101 \
  --allow-task-differences \
  --label "Balanced training" --reference-label "Short-trained hDeltaC" \
  --threads 1 --output results/cx-home-vector-balanced-live-comparison-NEW
```

Default comparisons require identical task and trajectory files. The explicit
`--allow-task-differences` opt-in permits different training/validation/test sets
while requiring identical circuits, simulation timestep, length scale and exact
challenge protocols. Only matching challenges are replayed and compared. Both
reference arguments must be supplied together; the reference may also be active.

## Outputs

For neural activity rather than only decoded vectors/errors, use the separate
[activity recorder and atlas](cx_home_vector_activity.md). It accepts the same
training and saved-evaluation checkpoints. Activity bundles also preserve a
`current` checkpoint that this evaluator can reload.

- `comparison.png` / `.pdf`: joint x–y home-vector tips, long-path magnitude and full vector error, pause errors after forward/backward walking, square-loop x/y errors, and equivalent world paths traversed with different body motions. With no reference, these show the current checkpoint alone.
- `long_path.png` / `.pdf`: a larger, focused view of the three long-path panels, also generated by completed-run comparison and prediction audits.
- `reversal.png` / `.pdf`: predicted and correct x/y components during reversal and the return-home pause, with walking reversal and pause onset marked.
- `home_vectors.png` / `.pdf`: actual predicted and correct home-vector x/y traces for the long straight path, opposite-history pauses, square loop and world-equivalent forward/sideways motions. These show the vectors themselves rather than prediction errors. Gray vertical lines mark movement transitions. Reversal components are shown in the separate reversal figure.
- `validation_history.png`: copied historical **unweighted** validation scores through each captured update. Different validation memberships are flagged; this is not the balanced checkpoint-selection score.
- `challenge_metrics.json`: shared per-trajectory metrics, aggregate challenge errors, world-path prediction separation, checkpoint identities and definitions.
- `challenge_metrics.csv`: one row per model/trajectory, including vector/component RMSE, endpoint error, pause drift, late magnitude growth and reversal lag where applicable. Empty cells mean a metric does not apply.
- `current_predictions.npz` and optional `reference_predictions.npz`: Cartesian predictions and targets indexed in prepared challenge order.
- `*_checkpoint.pt` and `*_metadata.json`: exact copied checkpoint bytes/hash, source/run paths, run manifest, circuit, training configuration, and copied history.
- `manifest.json`: completion/failure status, code and artifact hashes, immutable prepared inputs, and CPU execution settings.

## Interpretation and limits

Metrics retain the existing meanings: final target error differs from drift
across a pause; a sustained magnitude-underestimation threshold is not a general
maximum accurate distance; reversal lag measures when predicted x turns around
within the backward-walking interval. Full definitions and additional measures
remain in the JSON report. Targets are checked against an independent geometric
integration before reporting.

In the joint x–y panel, points are **home-vector tips**, not the fly's physical
path. Distance from the origin shows magnitude, and the ray from the origin
shows direction. Equal x/y axis scales preserve angles; +y points downward,
matching the simulation convention. Markers show five matched timestamps, with
times labeled on the correct vector. Dotted connectors join correct and decoded
tips at the same time, so their lengths are the full Cartesian vector errors.
The adjacent time plots separate magnitude changes from full vector accuracy.

A fresh rendering of the saved update-700 and update-800 snapshots is available
in `results/cx-home-vector-balanced-u700-u800-vector-plots-20261105`.
Its three-panel `long_path.png` shows how update 800 can have greater magnitude
overshoot and smaller vector error simultaneously. This rendering does not
rerun training or alter the original snapshot bundles.

This command evaluates only the standard challenge suite. It does not recompute
validation/test subset scores, run the larger straight-path sweep, or perform a
new half-step numerical convergence gate. The existing `analysis predictions`
and `analysis.compare` commands remain available for full completed-run audits;
`analysis.validation_subsets` separately supports active-run validation replay.

History is copied separately from the checkpoint and may briefly lag it during
a concurrent save. Evaluation always uses the recorded checkpoint bytes. Repeated
challenge inspection is exploratory; the running trainer continues to select
checkpoints using validation alone. Prepared sources must remain unchanged
through evaluation, and output directories must be separate from all inputs.

## Verification and first live replay

The home-vector regression suite passed: **146 passed, 9 skipped**. New tests
cover single/comparison reports, exact agreement with direct model replay and
shared metrics, recurrence-disabled checkpoints, saved-array/hash integrity,
replacement of the active checkpoint during evaluation, input/path guards and
failure markers. Existing completed-run comparison and snapshot tests pass.
Plot checks explicitly distinguish vectors with identical magnitudes but wrong
directions, verify same-time connector endpoints and Cartesian error curves,
and handle zero vectors without inventing an angle. A completed short-trained
prediction audit also passed with the shared joint long-path panels.
Historical reload tests reproduce both archived current and reference predictions
exactly after moving the original training directories, preserve input bundles,
and reject missing models or inconsistent snapshot identities. The saved
update-900 model was also replayed into
`results/cx-home-vector-balanced-u900-home-vectors-20261105`, including the new
actual x/y component traces.

The first live comparison captured balanced-training update **700** and the
short-trained hDeltaC reference at update **1950**. Its verified output is
`results/cx-home-vector-balanced-live-comparison-20261105`. This is a fixed
snapshot, not the latest state of the continuing training. Long-path magnitude
growth is much closer to the target, but the 64-second endpoint still has about
21.5 mm of **vector** error; correct magnitude alone does not establish correct
direction. The saved plots and tables expose additional pause/loop tradeoffs.
