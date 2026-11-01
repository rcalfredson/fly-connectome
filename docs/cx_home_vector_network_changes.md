# How much did the CX capacity network change during training?

1 November 2026 · Run: `gpu-initial-v1-anatomy-seed0` · Selected update: 2,000

## Finding

The anatomical edge mask and contact counts were preserved, while the learned
effective internal coupling changed substantially. Its weight norm increased
**7.71-fold**, **212 of 478 internal edges changed sign**, and the learned
within-projection strengths often departed from simple proportionality to
contact counts. The three intrinsic time constants increased from 0.1 s to
approximately the allowed maximum of 0.5 s. PFN input couplings changed more
moderately and retained their initially positive signs.

This documents how much parameter flexibility the first capacity solution used.
It supports describing the result as an anatomically constrained, extensively
trained recurrent model. It does not show that measured physiological weights
underwent these changes, identify a unique mechanism, or establish that biological
wiring contributed more than a matched randomized graph.

## What is the reference for “connectome-set weights”?

The connectome supplies a raw contact count C for each selected neuron pair and
the directed positive-edge mask. Counts are not measured physiological weights.
For a source-type/target-type projection a→b, the model uses:

```text
S_ab = total selected projection contacts / number of selected receiving bodies
W_initial[i,j] = (C[i,j] / S_ab) * a_initial[i,j]
W_trained[i,j] = (C[i,j] / S_ab) * a_trained[i,j]
```

Receiving bodies with no contacts remain in the denominator's population count.
At initialization, every PFN input coefficient is +1. Internal coefficients
are seeded Normal(0, 0.1) values, shared within each nonempty directed type pair.
Those initial internal signs/scales were chosen computational parameters, not
biological measurements. During training each edge gets its own signed
coefficient, with a weak penalty on departure from its initial value.

The comparison here is with **this training initialization**, not with the
historical frozen PFN→hDeltaB model's different normalization/gain conventions.
All effective weights are in model units and have no Hz/conductance calibration.

Counts continue to affect more than the initial value: because coefficients are
bounded to [-5,5], each edge's allowed effective magnitude is bounded by
5*C/S. The coefficient-change penalty also acts in coefficient space, not in
absolute effective-weight space. Thus counts still affect allowed scales and
the prior, even when trained relative count ratios are no longer fixed.

## What stayed fixed?

- The same 60 prescribed PFNs, 45 dynamic bodies and 1,075 positive anatomical
  body-pair edges remain selected: 597 input and 478 internal edges.
- Raw counts are unchanged: 24,619 input and 5,205 internal contacts.
- No absent edge was added; no selected edge became exactly zero in this
  checkpoint. All 1,075 coefficients differ from initialization by more than
  1e-6, a numerical reporting threshold rather than a functional criterion.
- The zero-report vDeltaE→hDeltaB block remains zero. Omitted projections and
  feedback into clamped PFNs remain outside the model.
- PFN encoding, the fixed normalization denominators, rate gain/threshold,
  absence of slow excitability states, and zero reset at trajectory initialization
  were not learned.

These statements refer to the modeled boundary. They do not establish biological
absence for zero-report or unqueried connections.

## Overall coupling changes

Here, weight norm means sqrt(sum of squared effective weights), equivalent to
the matrix Frobenius norm. Relative change is ||W_trained-W_initial|| /
||W_initial||. It measures change across a collection of edges, not the average
percentage change of its individual edges.

| Edge group | Edges | Initial norm | Trained norm | Trained/initial norm | Relative change | Sign changes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PFN inputs | 597 | 18.554 | 15.507 | 0.836× | 17.9% | 0 |
| Internal CX coupling | 478 | 0.537 | 4.142 | **7.715×** | **740.1%** | **212 (44.4%)** |
| All modeled edges | 1,075 | 18.562 | 16.051 | 0.865× | 27.9% | 212 (19.7%) |

The internal trained and initial signed weight vectors have cosine similarity
**0.372**, compared with **0.997** for the input weights. This is consistent with
larger changes in the internal signed pattern. These are geometric parameter
comparisons, not activity correlations or causal measures of circuit importance.

Large relative changes partly reflect weak internal initialization. The RMS
absolute coefficient change is 0.331 internally versus 0.426 for PFN inputs.
Therefore “740%” should not be read as a physiological effect size or a statement
that each internal edge became 7.4 times stronger.

### Why the whole-network percentage can conceal internal changes

The single reported PFNd→vDeltaE contact is divided by S = 1/18, giving an
initial effective weight of **18**, despite its raw count of one. Specifically:

```text
PFNd 77360 → vDeltaE 142911
raw contacts = 1
initial coefficient = 1
trained coefficient = 0.824354
effective weight: 18 → 14.8384
```

This one edge accounts for **94.0% of the initial sum of squared weights** and
85.5% of the trained sum. Consequently, the global norm and its high cosine
similarity are dominated by this normalization choice. They are poor standalone
summaries of how much the recurrent circuit changed.

This is a consequence of the declared projection normalization, not a changed
contact count or a newly discovered strong anatomical pathway. It motivates an
input-normalization sensitivity analysis. Its large coefficient-weight product
does not by itself show that the trained function depends on this edge.

## Changes by directed projection

| Presynaptic → postsynaptic | Edges | Initial coefficient, shared within block | Trained/initial weight norm | Sign changes |
| --- | ---: | ---: | ---: | ---: |
| hDeltaB → hDeltaB | 143 | +0.01257 | 27.445× | 86 |
| hDeltaB → vDeltaE | 21 | −0.01321 | 12.288× | 5 |
| hDeltaB → hDeltaG | 46 | +0.06404 | 4.908× | 14 |
| vDeltaE → vDeltaE | 65 | +0.01049 | 39.386× | 21 |
| vDeltaE → hDeltaG | 118 | −0.05357 | 4.929× | 58 |
| hDeltaG → hDeltaB | 9 | +0.03616 | 13.049× | 6 |
| hDeltaG → vDeltaE | 56 | +0.13040 | 3.155× | 18 |
| hDeltaG → hDeltaG | 20 | +0.09471 | 8.149× | 4 |
| PFNd → hDeltaB | 432 | +1 | 1.291× | 0 |
| PFNd → vDeltaE | 1 | +1 | 0.824× | 0 |
| PFNv → hDeltaB | 126 | +1 | 0.868× | 0 |
| PFNv → vDeltaE | 38 | +1 | 1.006× | 0 |

The largest fold changes occur in two blocks initialized with very small
positive coefficients. All eight nonempty internal projections grow in norm.
This is not simply a uniform gain increase: several initially sign-uniform
blocks become mixtures of positive and negative effective coupling. Internal
edges end with 253 positive and 225 negative coefficients, versus all-positive
PFN input edges. No coefficient reaches its ±5 bound; the largest absolute
coefficient is 2.102.

“Sign changed” is relative to the seeded initial scenario, not to a measured
transmitter/receptor effect. This fast-rate model does not resolve physiological
or compartment-specific signs and does not enforce source-wide sign consistency.

## Are strengths still proportional to contact counts?

To distinguish block gain changes from reweighting individual pairs, the audit
finds the single signed scalar g that best describes the saved trained block as
g*(C/S). This is a descriptive projection of weights, not retraining or replacing
the network. Its relative residual is:

```text
residual = ||W_trained - g_best*(C/S)|| / ||W_trained||
```

Zero means exact count proportionality. A value near one means that a single
signed block gain describes little of the trained signed weight pattern.

| Projection | Relative residual after best count scaling |
| --- | ---: |
| hDeltaB → hDeltaB | 77.9% |
| hDeltaB → vDeltaE | 80.9% |
| hDeltaB → hDeltaG | 77.8% |
| vDeltaE → vDeltaE | 83.1% |
| vDeltaE → hDeltaG | **98.8%** |
| hDeltaG → hDeltaB | **99.8%** |
| hDeltaG → vDeltaE | 87.7% |
| hDeltaG → hDeltaG | **10.2%** |
| PFNd → hDeltaB | 35.2% |
| PFNv → hDeltaB | 22.4% |
| PFNv → vDeltaE | 16.6% |

The singleton PFNd→vDeltaE projection has zero residual trivially and cannot
test preservation of a multi-edge pattern. hDeltaG→hDeltaG is a notable internal
exception: it is approximately a scaled count pattern by this norm measure,
despite four sign changes on individual edges. The measure weights strong edges
more heavily than weak ones, so these findings are compatible. Most other
internal blocks cannot be summarized well as count-proportional coupling with
one modified gain. These comparisons do not establish which departures are
needed to perform the task.

## Intrinsic dynamics and decoder

| Population | Initial tau (s) | Trained tau (s) | Initial bias | Trained bias |
| --- | ---: | ---: | ---: | ---: |
| hDeltaB | 0.100 | 0.499883 | 0.100 | −0.009845 |
| vDeltaE | 0.100 | 0.500000 | 0.100 | 0.538339 |
| hDeltaG | 0.100 | 0.500000 | 0.100 | 0.132285 |

The learned solution uses slower intrinsic responses, near the allowed upper
bound, as well as changed recurrence. These taus are not effective memory time
constants of the coupled network; they do not alone explain its several-second
retention or drift. Bias changes affect operating points, so the consequences
of weight changes depend on activity and rectification.

The 90 decoder weights and two intercepts also learned from zero. The trained
decoder weight norm is 1.032; intercepts are approximately (-0.00418, 0.13381) mm.
Their complete table is exported separately. Decoder magnitudes do not establish
population necessity or how the fly would read this representation.

## What this adds before a follow-up experiment

The pilot's home-vector improvement was accompanied by substantial adjustment
of internal coupling, signs and intrinsic dynamics. Its success cannot be
described as exposing a ready-made home-vector code through a trained readout
or making only small physiological refinements to known weights. It remains a
valid test of the permissive computational capacity question we specified.

Three follow-up comparisons are now especially informative:

1. **Count proportionality:** train with one coefficient per type-pair block
   and compare against per-edge training under matched task/selection budgets.
   This directly tests the importance of within-block reweighting.
2. **Input normalization:** assess sensitivity to the normalization of sparse
   input projections, especially the single-contact PFNd→vDeltaE case. Declare
   alternative scales and their coefficient bounds before comparing fits.
3. **Anatomical structure:** use matched trained rewires and multiple seeds to
   determine whether this particular mask helps beyond generic recurrent
   capacity. Add evidence-based sign restrictions in a separate constraint
   stage, rather than treating learned signs as physiological findings.

Acute perturbations or parameter-reversion experiments could test reliance on
particular changes, but large weight changes alone do not identify mechanism.
Removing them can also change the operating point; decoder refitting and full
retraining answer different compensation questions. No such new experiment was
run for this audit.

## Artifacts and validation

The [analysis replay guide](cx_home_vector_analysis_walkthrough.md) documents
the tracked `network` command, explicit bundle paths and fresh-output behavior.
The links below identify historical artifacts stored separately from Git.

The [machine-readable summary](../results/gpu-initial-v1-network-change-audit-20261101/summary.json)
and [edge-by-edge CSV](../results/gpu-initial-v1-network-change-audit-20261101/edge_changes.csv)
retain body identities, raw counts, normalization denominators, initial/trained
coefficients, effective weights and sign changes. Additional artifacts include
projection summaries, intrinsic/decoder tables and full body-ID matrices.

- [Internal coupling comparison](../results/gpu-initial-v1-network-change-audit-20261101/internal_weight_changes.png)
  ([PDF](../results/gpu-initial-v1-network-change-audit-20261101/internal_weight_changes.pdf)):
  initial, trained and difference matrices use a common symmetric color scale;
  presynaptic bodies are rows and postsynaptic bodies columns, ordered by type
  then body ID. This does not assign shared columns or angles.
- [Projection comparisons](../results/gpu-initial-v1-network-change-audit-20261101/projection_changes.png)
  ([PDF](../results/gpu-initial-v1-network-change-audit-20261101/projection_changes.pdf)):
  strength ratios and departures from a scaled contact pattern. Blue marks
  PFN input projections and orange internal projections.
- [Replay script](../results/gpu-initial-v1-network-change-audit-20261101/analysis.py)
  and [manifest](../results/gpu-initial-v1-network-change-audit-20261101/manifest.json).

The audit reconstructs the seeded initial coefficients exactly, verifies the
checkpoint's contact/pre/post buffers, and reproduces both initial and trained
effective-weight matrices exactly through the model implementation. Source and
output hashes were verified and all original run/data files were checked for
changes. No retraining, anatomical extraction, original-result modification or
commit was performed. Result artifacts remain ignored local files; the report
records the main numerical findings for a fresh checkout.
