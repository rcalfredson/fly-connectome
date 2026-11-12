# Decoder contributions on matched world paths

This analysis extends the paired-state row/null-space comparison by identifying
the individual neurons' addends in the **same frozen decoder**. It uses all four
archived forward/rightward pairs (8/9, 10/11, 12/13 and 14/15), all 65 dynamic
neurons and the best checkpoint at update 1975. PFNs, dynamics, decoder weights,
and archived results remain unchanged. There is no training or new simulation.

## What is attributed

For neuron i, with activity r and its two decoder weights:

```text
absolute_i(t) = r_i(t) * (D_xi, D_yi)
prediction(t) = intercept + sum_i absolute_i(t)

from_initial_i(t) = (r_i(t) - r_i(0)) * (D_xi, D_yi)
prediction(t) - prediction(0) = sum_i from_initial_i(t)

paired_difference_i(t) = absolute_B_i(t) - absolute_A_i(t)
prediction_B(t) - prediction_A(t) = sum_i paired_difference_i(t)
```

A is forward traversal; B is rightward traversal. The intercept is a separate
term, never assigned to individual neurons. The paired difference retains any
initial difference; it is not calculated from individually centered paths.
Each vector has signed x/y components in mm.

The absolute and initial-referenced views address persistent offsets versus
contributions developing during movement. In these particular archives, every
initial dynamic activity is zero, so the two views coincide exactly.

Rank neurons separately for each trajectory by duration-weighted RMS vector
magnitude. Rank paired differences separately: a neuron can dominate a
prediction without dominating the difference, or vice versa.

```text
duration_k = time[k+1] - time[k]
score_i = sqrt(sum_k duration_k * norm(contribution_i[k])^2
               / sum_k duration_k)
```

Each state represents the following interval. The terminal state has zero
duration and is excluded from these summaries, but retained in arrays and
snapshots. Full-rank comparisons use Spearman correlation with average tied
ranks. Top-ten overlap is an illustrative summary, not a selected significance
threshold. Body ID breaks display-order ties, which are flagged in the report.
There are no inferential tests across the four related trajectory pairs.

## Outputs and safeguards

- Full-resolution per-neuron contribution arrays for both trajectories and B−A.
- All-neuron scores/ranks, signed snapshots at 0, 2, 4, 6, 7.5 and 8 seconds,
  and population summaries.
- Plots compare large contributors, signed differences, and cancellation.
- Per-population summaries distinguish RMS of the signed population sum from
  the sum of individual RMS scores. These are different quantities.
- Decoder, neuron identities, exact script and input hashes are preserved.

The source must be the completed paired-state bundle with its exact checkpoint
identity and neuron order. That earlier bundle includes replay verification of
all eight trajectories. This analysis checks its artifact hashes and verifies
the contribution sums against archived predictions, prediction changes, and
paired decoded differences. Input hashes are checked again on completion.
The new output must be separate and fresh. The manifest records completion or
failure; no archived input is overwritten.

## Interpretation

These are **arithmetic contributions**, not unique information or causal
necessity. Magnitudes can be large even when their signed sum is small. Neither
the intercept nor cross-neuron cancellation should be omitted when interpreting
the prediction. Large contributions do not allocate the prediction error to
individual neurons, and deleting an addend does not simulate deleting a neuron
from the recurrent network.

The neuron-level signed differences sum to the same decoded difference
reproduced by the earlier row-space component. The present attribution uses the
actual neuron activities, not the coordinates of the projected row component:
projecting first would redistribute addends among neurons. The resulting ranks
describe this decoder and these trajectories, not a universal cell importance.
For a fixed neuron, contribution magnitude is activity magnitude multiplied by
the norm of its decoder-weight vector. Reusing those same weights can itself
favor similar rankings; high rank agreement is not independent evidence that
the two activity patterns or represented home vectors are equivalent.

## First results

All four pairs share nine of their ten largest duration-RMS contributors:

| Pair | All-neuron rank correlation | Top-ten overlap |
| --- | ---: | ---: |
| 8 / 9 | 0.9265 | 9 / 10 |
| 10 / 11 | 0.9036 | 9 / 10 |
| 12 / 13 | 0.9604 | 9 / 10 |
| 14 / 15 | 0.9443 | 9 / 10 |

hDeltaB 17816 ranks first in all eight trajectories. hDeltaG 20565 and 28258
are also prominent; the complete body-ID rankings are retained rather than
reducing the result to cell-type averages. Rank agreement coexists with
different contribution magnitudes and prediction errors.

For paths 12/13 at 4 seconds, hDeltaB 17816 contributes approximately
`(+1.1094, -1.0260) mm` to B−A, hDeltaG 20565 contributes
`(+0.7744, -0.6732) mm`, and hDeltaC 17728 contributes
`(-0.3234, -0.8362) mm`. Other neurons supply opposing terms. Across all 65,
the individual difference-vector magnitudes sum to 13.3158 mm, while their
signed sum is only `(0.7606, 0.0387) mm`. This reproduces the earlier decoded
disagreement and illustrates why the largest individual terms alone cannot
explain the net result without examining cancellation.

All predictions reconstruct within 0.000004 mm per coordinate, and all paired
prediction differences within 0.000005 mm. The small residuals reflect using
double-precision arithmetic on saved rates/weights versus archived predictions.
The source bundle's file hashes remain unchanged.

## Run

Use the project Python 3.11 environment with NumPy, SciPy and Matplotlib. Torch
is not required for this analysis of the saved arrays.

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fly-connectome-mpl \
python -m fly_connectome.home_vector.analysis.decoder_contributions \
  --source results/cx-home-vector-balanced-u1975-state-spaces-20261110-v2 \
  --output results/cx-home-vector-balanced-u1975-decoder-contributions-20261112-v2

python -m pytest -q tests/test_decoder_contributions.py
```

The completed browser is `index.html` in that output directory. The first run
without `-v2` predates expanded reporting of identified neurons and the worked
snapshot; its numerical attribution is unchanged.
