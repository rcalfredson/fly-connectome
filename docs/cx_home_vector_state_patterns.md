# Shared decoder-visible patterns and differing null patterns

This visualization projects each trajectory's actual network state separately,
using the same frozen home-vector decoder. It complements the earlier analysis
of B−A and the neuron-level decoder-addend analysis. All four archived matched
world paths are retained. There is no refitting, simulation, or input change.

## Separate projections

Let D be the fixed two-output decoder, d its intercept, and Q an orthonormal
basis for its row space. For each trajectory:

```text
row_A(t)  = activity_A(t) @ Q.T @ Q
null_A(t) = activity_A(t) - row_A(t)
row_B(t)  = activity_B(t) @ Q.T @ Q
null_B(t) = activity_B(t) - row_B(t)

prediction_A(t) = d + row_A(t) @ D.T
prediction_B(t) = d + row_B(t) @ D.T
null_A(t) @ D.T = null_B(t) @ D.T = (0, 0)
```

Activities are not centered or normalized before projection. The projected
vectors retain one entry per neuron, in the original 65-neuron coordinate
system. Their entries can be negative and are not separately measured firing
rates. Each actual activity is the sum of its row and null entries.

For this fixed decoder, equal predictions imply equal row projections. A
similar projected pattern is therefore a way of visualizing output agreement,
not independent evidence for a shared biological mechanism. Even a neuron
whose actual activity is small can have a nonzero row entry that is canceled
by its null entry. Neither component defines a separate biological population.

## Select examples without requiring agreement

Each figure uses five examples per set by default:

```text
row_score_i  = min(RMS(row_A_i), RMS(row_B_i))
null_score_i = RMS(null_B_i - null_A_i)
```

The row rule selects large projected entries in both trajectories, without
rewarding matching signs, correlations or small differences. It does not use
the correct home vector or select for accuracy. The null rule selects the
largest decoder-canceling differences. Body ID breaks ties deterministically.
All 65 scores are saved, so the examples need not stand for the entire state.

RMS is weighted by each state's following interval. The terminal sample has no
following interval and is excluded from summaries, but remains in the plots
and saved arrays. Descriptive temporal correlations also exclude that endpoint;
the archived sampling intervals are uniform. Constant traces have undefined
correlation, serialized as a blank field.

The figure begins with both decoded x/y time courses and the shared correct
home vector. Below those:

- Left two columns: selected row-space traces and their actual activities.
- Right two columns: selected null-space traces and their actual activities.
- Blue is forward traversal A; dashed orange is rightward traversal B.
- An asterisk marks a neuron selected in both sets, if any. Corresponding row
  and null examples on the same visual line need not be the same neuron.

Axes use model activity units for both actual and projected traces, but their
scales vary by panel. The home-vector plots use mm. Comparing the numeric axes
matters: projected and actual activity can have very different magnitudes.

## First results

| Pair | RMS row A / B | RMS row B−A | RMS null B−A |
| --- | ---: | ---: | ---: |
| 8 / 9 | 3.006 / 2.999 | 0.294 | 41.349 |
| 10 / 11 | 6.875 / 6.688 | 0.387 | 52.163 |
| 12 / 13 | 4.859 / 4.919 | 1.218 | 33.923 |
| 14 / 15 | 2.796 / 2.967 | 0.831 | 42.791 |

These are duration-weighted full-population norms in model activity units.
They show much larger differences in null than row components, but row
differences remain appreciable for some pairs. Norms are not fractions of
information or measures of home-vector error.

For paths 12/13, the large-in-both row examples are vDeltaE 118715,
hDeltaG 49952, hDeltaG 57625, vDeltaE 63612 and hDeltaG 20565. Their projected
traces resemble one another across traversal conditions more closely than
several corresponding actual activity traces do. The selected null examples
are vDeltaE 142911, hDeltaB 17816, hDeltaB 17820, hDeltaC 23630 and hDeltaC
521644. Some differ strongly in amplitude; others also differ in time-course
shape. The two selected sets do not overlap in any of these four pairs,
although overlap is permitted and tested.

These examples are not the same selection as the largest decoder addends.
Notably, hDeltaB 17816 was the largest duration-RMS decoder contributor in all
eight trajectories, yet its null-component difference is prominent here. This
is possible because the projections redistribute a population pattern among
neuron coordinates; large signed contributions can cancel at the decoder.

This is a descriptive view of shared decoder-visible patterns alongside
differing activity that cancels at the current readout. It does not identify
the recurrent mechanism producing those patterns, or establish that null
activity is irrelevant to future dynamics. Both predictions must still be
compared with their correct home vector.

## Verification and reproduction

The analysis reads only the complete, hash-verified paired-state bundle,
including its exact frozen checkpoint and neuron order. Each separate state
must reconstruct from its components. Row projections must reproduce its
archived prediction after adding the intercept; null projections must decode
to zero. Their B−A differences must reproduce the previously archived row/null
decomposition. All source-file hashes are checked again on completion.

Use the project Python 3.11 environment:

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fly-connectome-mpl \
python -m fly_connectome.home_vector.analysis.state_patterns \
  --source results/cx-home-vector-balanced-u1975-state-spaces-20261110-v2 \
  --output results/cx-home-vector-balanced-u1975-state-patterns-20261112 \
  --examples 5

python -m pytest -q tests/test_home_vector_state_patterns.py tests/test_decoder_contributions.py
```

Use a fresh output directory. The completed browser is `index.html`; PDF and
PNG figures, full arrays, neuron scores, timecourses and validation report are
alongside it. The exact analysis script and source hashes are archived.
