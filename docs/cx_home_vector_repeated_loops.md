# Two uninterrupted closed loops with the frozen model

The separate-state figures raised a question: can the model maintain a correct
home-vector readout while its full activity does not return to its initial state
after a closed path? This probe repeats the exact archived closed circles of
paths 8/9 and 14/15 twice. Paths 10/11 and 12/13 are half-circles and are not
used as the repeated closed-loop controls.

## Controlled continuation

- Use the same encoder, dynamic network and decoder from best checkpoint 1975,
  SHA256 `2aab91f13258883819f66f12897919b2bf509617e7927120ca7f9db069d9a31f`.
- Each loop lasts eight seconds; run 16 seconds continuously at the original
  0.0025-second step. Start from zero activity once, without resetting at 8 s.
- Repeat the exact held heading/body-velocity sequence. Remove only the unused
  stationary terminal input between loops; the original endpoint state is
  unaffected because input at t drives the update to t+dt.
- Verify identical controls and PFNs across the two loops and independently
  integrated targets. Correct home vectors return essentially to zero at both
  loop endpoints and match between forward/rightward partners.
- Replay the original protocols and check the full first-loop prefix, including
  the endpoint. Rates, predictions and targets match the archive exactly.

Each loop's RMSE uses its states preceding updates, excluding the endpoint
from that time average. Closure error is the absolute distance of the endpoint
prediction from the true zero home vector. Endpoint drift relative to the
nonzero initial decoder intercept is separately recorded in CSV.

## Results

| Path | Traversal | Loop 1 / 2 vector RMSE (mm) | Loop 1 / 2 closure error (mm) |
| --- | --- | ---: | ---: |
| 8 | Forward | 0.149 / 1.044 | 0.245 / 1.544 |
| 9 | Rightward | 0.251 / 0.731 | 0.481 / 1.125 |
| 14 | Forward | 0.124 / 0.640 | 0.215 / 1.044 |
| 15 | Rightward | 0.456 / 0.644 | 0.526 / 1.142 |

Every trial has greater time-averaged error and greater closure error in the
second loop. Activity remains finite, but endpoint state norms rise from about
88–106 after loop 1 to 143–177 after loop 2. Most of this state is in the
decoder null space, while the row-space closure state also changes. Peak
single-neuron activities rise from about 60–87 in loop 1 to 69–99 in loop 2;
these are model units, not calibrated firing rates.

The model has lower rectification at zero but no upper firing-rate cap. The
linear decoder has no hard output range. Thus these results are not literal
saturation or numerical overflow. They demonstrate worsened displacement
readout and loop closure on an uninterrupted repeated movement sequence.

The frozen decoder still exactly reads the activities to numerical precision;
the accumulated representation is less accurate. A correct home-vector system
should support displacement cancellation on closed paths within its intended
operating conditions. These results expose a limitation of this model on the
specified repeated circles, despite useful accuracy over the earlier one-loop
evaluation. No new pass/fail tolerance or biological accuracy claim is imposed.

## What remains unresolved

Returning home does not require every neuron to return to its initial activity.
Extra null activity would be compatible with correct decoding if it did not
compromise subsequent performance. Here, additional activity and worse decoding
occur together, but the probe does not establish which part of the inherited
state causes the error. In particular, it does not isolate null-state growth
from decoder-visible drift, initial transients or other recurrent mechanisms.

Two loops do not establish whether activities eventually diverge, approach a
bounded orbit, or settle at a plateau. The model could remain bounded and still
decode poorly. Nor do these findings identify a change to the PFN encoder or
justify retraining. Mechanistic follow-ups would need separate controls.

## Artifacts and reproduction

The completed output is
`results/cx-home-vector-balanced-u1975-repeated-loops-20261112/`, with browser,
PNG/PDF figures, CSV metrics, full-resolution NPZ arrays, exact protocols and
script, checkpoint/metadata and a completion manifest. Prepared data and all
archived source files are hash-checked before and after the run.

Use the project dependencies, including Torch, and a fresh output directory:

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fly-connectome-mpl \
python -m fly_connectome.home_vector.analysis.repeated_loops \
  --activity results/cx-home-vector-balanced-u1975-activity-by-trajectory-20261106 \
  --source results/cx-home-vector-hdeltac-long-data-20261104 \
  --output results/cx-home-vector-balanced-u1975-repeated-loops-20261112

python -m pytest -q tests/test_home_vector_repeated_loops.py
```
