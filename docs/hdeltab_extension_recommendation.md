# Historical circuit selection — 24 October 2026

**Status updated 26 October 2026: superseded as the next modeling milestone.**
The current plan is to [extract the vDeltaE–hDeltaG experimental observations
and model specification from the thesis](vdeltae_hdeltag_next_steps.md) before
implementing dynamics. The hDeltaB→hDeltaG transfer experiment below was not
implemented. Its proposal is retained as a historical hypothesis, alongside
the anatomical results, assumptions and reproducible commands. The gain, time
constant and protocol below are provisional choices from that proposal, not
parameters adopted for a new hDeltaG model.

## Original recommendation

Recommend **the eight-cell hDeltaG population as a feedforward input-transfer
component**, receiving the existing hDeltaB output. This is the smallest
literature-motivated component in the investigated shortlist with direct input
from all selected hDeltaB neurons and anatomical routes toward goal circuitry.
It is a test of a partial input pathway, not a claim that hDeltaG is a relay-only
cell type. **No investigated candidate is sufficiently constrained here to add
a demonstrated neural displacement-memory mechanism.** Defer integration,
plasticity, reset and steering dynamics.

The choice is conditional on the narrow next question: does the measured
hDeltaB→hDeltaG projection deliver useful, distinguishable movement-dependent
patterns to a population implicated in a memory motif? If the next milestone
instead requires reproducing physiological memory, obtain the full motif model
and missing inputs first. Adding a long time constant would not satisfy that goal.

## Repository baseline and scope

Inspected README, movement/trajectory/decoder code, tests, saved movement and
trajectory configurations, connectivity helpers and the existing PFNd incoming
census. No applicable AGENTS.md was found in the repository or ancestor search.
The working tree was initially clean. Existing commands and dynamics are unchanged.

The selected model retains prescribed literature PFN tuning, b=1, per-receiver
normalization within PFNd-L/R and PFNv-L/R, PFNd ×6, PFNv ×1, gain 0.001,
hDeltaB tau 0.1 s and default dt 0.001 s. Continuous trajectory state carries
across boundaries. Geometry and return-to-origin remain external references;
there is no fitted decoder, neural displacement memory or behavioral controller.

The live extraction uses `neuprint.janelia.org`, `male-cns:v1.0`, UUID
`4b2087c0fbe046bfaf0d60bc970e3e5d`, mutation ID `1006591300`, with full metadata
in its manifest. It checks all 19 hDeltaB IDs/instances against the original
extraction, the reference 26913→515192 = 137, and dataset metadata before/after.
The outgoing query includes **all Segment partners**, weight ≥1, with no status,
cropping, ROI or hemisphere filters. It retains raw annotations, labels, ROI JSON
and untyped fragments. A fixed list of 54 directed population pairs supplements
the one-hop census; no discovered partner triggers further expansion. PFR subtype
cross-connections and direct hDeltaA/G↔PFL3 checks were added after reviewing the
primary anatomical/steering papers. The earlier census remains separate.

## Anatomical results

There are **100,030 outgoing contacts**, **29,346 positive pairs** and **23,272
partner bodies**. Of these contacts, 62,879 reach typed neurons, 29 reach two
untyped neurons, and 37,122 reach 22,168 non-Neuron segments. Together untyped
partners contribute 37,151 / 100,030 = **37.14%**. Fragment body counts are not
counts of complete biological neurons. Reconstruction labels remain provisional;
missing cropping values remain unknown. No partner is removed for these reasons.

| Direct hDeltaB target | Complete target population | Contacts | Fraction of all 100,030 outgoing contacts | Fraction of complete target recorded input | Targets reached |
| --- | --- | --- | --- | --- | --- |
| PFR_a | 28 | 6,717 | 6.715% | 6,717 / 42,484 = 15.811% | 28/28 |
| PFR_b | 16 | 5,787 | 5.785% | 5,787 / 25,874 = 22.366% | 16/16 |
| hDeltaA | 12 | 2,216 | 2.215% | 2,216 / 50,360 = 4.400% | 12/12 |
| hDeltaG | 8 | 736 | 0.736% | 736 / 28,991 = 2.539% | 8/8 |
| vDeltaE | 18 | 57 | 0.057% | 57 / 14,583 = 0.391% | 12/18 |

The first four rows each receive input from all 19 hDeltaB sources; vDeltaE
receives input from 13/19. Coverage never substitutes for contact strength.
Target input denominators are sums of recorded `post` counts, not sums within
this selected subgraph; they were not independently reconciled by a complete
candidate incoming census. Outgoing denominators are contact weights; unique
presynaptic site counts cannot be used because a site can have multiple partners.
Per-source fractions include all 19 sources and save undefined zero-denominator
fractions as blank cells. Every matrix includes all possible pairs, including zeros.

| Directed motif | Contacts | Positive / possible pairs | Interpretation |
| --- | --- | --- | --- |
| hDeltaC→vDeltaE | 1,604 | 130/360 | Much larger candidate input route than hDeltaB→vDeltaE; not a physiological transfer function |
| vDeltaE→hDeltaG | 2,405 | 118/144 | Specific integration-motif input route; 8.296% of hDeltaG recorded input |
| hDeltaG→vDeltaE | 653 | 56/144 | Reciprocal anatomy; 4.478% of vDeltaE recorded input |
| hDeltaG→hDeltaG | 361 | 20/64 | Recurrence exists; 1.245% of recorded input, no reported self-edges |
| vDeltaE→vDeltaE | 102 | 65/324 | Recurrence; 0.699% of recorded input, no reported self-edges |
| hDeltaA→hDeltaA | 284 | 33/144 | Recurrence; 0.564% of recorded input |
| hDeltaB→hDeltaB | 882 | 143/361 | Recurrence; 1.880% of recorded input, currently disabled |
| PFR_a→PFR_a; PFR_b→PFR_b | 387; 1,722 | 229/784; 119/256 | Recurrence is not evidence of memory |
| PFR_a→PFR_b; PFR_b→PFR_a | 579; 2,199 | 214/448; 269/448 | Subtypes interact; a one-subtype model omits this input |
| hDeltaG→FC2B; hDeltaG→FC2C | 1,711; 1,460 | 74/216; 106/376 | Anatomical access to goal circuitry |
| FC2B→hDeltaG; FC2C→hDeltaG | 81; 110 | 45/216; 67/376 | Weaker reciprocal routes in raw counts |
| hDeltaA→PFL3; hDeltaG→PFL3 | 3,881; 582 | 92/288; 48/192 | Direct steering-population routes; neither specifies steering computation |
| PFL3→hDeltaA; PFL3→hDeltaG | 47; 5 | 18/288; 2/192 | Small reciprocal counts |

hDeltaG→FC2A has zero reported contacts; hDeltaB→FC2A/B/C totals only 146,
and hDeltaB→PFL3 totals 32. FC2A/B/C→PFL3 totals 7,690 contacts across all
24 PFL3 targets. Thus hDeltaB→hDeltaG→FC2B/C→PFL3 is an anatomical path, not
a demonstrated sequence from velocity through memory and goal selection to turning.
hDeltaA has a stronger direct PFL3 route but adds 12 cells and an unresolved
working-memory mechanism. PFR has stronger hDeltaB input and peer-reviewed
travel-signal evidence, but neither subtype is smaller or established as a store.

All 54 pairs, reciprocal counts and denominators are in `motif_summary.csv`.
The full ranked one-hop census is retained, including stronger targets such as
hDeltaK/J/I and vDeltaM. Their large counts alone do not justify expanding their
networks or assigning integration. This is a smallest justified **shortlist**
selection, not a proof of a global optimum over all possible circuits.

## Evidence boundary and required inputs

See [the primary-source evidence ledger](hdeltab_extension_evidence.md) for
publication status and source limits. The specific vDeltaE–hDeltaG integration
claim is dissertation evidence, not a verified peer-reviewed mechanism in this
review. Its reported ramp/leak process cannot be replaced by an arbitrary
accumulator. The anatomical vDeltaE→hDeltaG route is independently present here;
hDeltaB also reaches hDeltaG directly, but the two inputs are not interchangeable.

Selected hDeltaG bodies are **19798, 20565, 28258, 42440, 48642, 49952, 57625,
515960**. Their hDeltaB input fractions range from 1.899% to 3.187%; about 97.46%
of total recorded input lies outside the hDeltaB projection. This makes the
proposed component a deliberately partial input test. Missing signals include
vDeltaE's movement/state dependence and upstream drive, hDeltaG's other inputs,
reset/sugar signals, context and baseline-ramp regulation, receptor mechanisms,
cell-specific time constants and functional arbor coordinates. No claim is made
that the missing inputs are silent in vivo. hDeltaC→vDeltaE was an anatomical
control only; hDeltaC activity is not supplied by the current model.

`consensusNt` predicts acetylcholine for hDeltaB, hDeltaG, vDeltaE and hDeltaA;
PFR_a is `unclear`, PFR_b acetylcholine. These are transmitter annotations,
not measurements of the sign or efficacy of each connection. Start the proposed
input-transfer null with positive hDeltaB→hDeltaG coupling, explicitly assumed.
Do not infer an inhibitory sign from the reported vDeltaE-dependent slowing of
a ramp. Recurrence/reciprocity alone supplies neither a gain nor an integration rule.

## Proposed next simulation (not implemented)

Add only eight hDeltaG rates, with raw 19×8 contacts C, and compare a static
projection d(t)=a hDeltaB(t) C to a fast tracking null
`tau_G dg/dt = -g + ReLU(d(t))`. This filters an input; it does not encode a
biological displacement integral. Use no recurrence, fitted angles, plasticity,
reset or geometry input. No FC2/PFL3 dynamics are added. Candidate gain `a=0.001`
per contact and `tau_G=0.1 s` are provisional numerical anchors, not measured
parameters; sweep gain ×{0.1,1,10} and tau_G {0.05,0.1,0.2} s. Leave the PFN→hDeltaB
model unchanged. As a sensitivity control, divide C by its hDeltaB column sum
per target; label this as normalization of the selected projection, never of all
biological input. Keep all body identities. The static projection is a reference
for separating anatomy from added filtering, not a new decoder.

Required model inputs are the carried hDeltaB rates, the exact body-aligned C,
and the declared gain/tau assumptions. Omitted inputs are set to zero **only in
this reduced null model**. A constant tonic-drive sensitivity control can test
rectification effects but cannot stand in for measured vDeltaE or ramp inputs.
Independent biological mapping would be required before interpreting an angular
readout. No within-type column label is converted to a world angle here.

1. Replay the existing 10 s trajectory unchanged, alongside an independently
   replayed zero-translation counterfactual with identical heading history,
   timing and initial conditions. Save both raw traces and their difference;
   the difference is an input decomposition, not stored memory.
2. Repeat the existing 8 world directions ×8 headings at speeds 0.5, 1 and
   2 mm/s and durations 0.25, 1 and 4 s, with a common 1 s pre-rest and 10 s
   post-rest. Include stationary controls for every heading. Save unnormalized
   amplitudes, centered pattern similarities and undefined-pattern flags.
3. Compare equal world movement at heading/body combinations (0°,1,0),
   (90°,0,-1), (180°,-1,0), (270°,0,1), with velocities in mm/s. Use both
   equal-distance/different-speed and equal-speed/different-duration trials.
   Include the inherited worst evoked pair: world 0°, headings 45° versus 315°.
4. Use two different movement histories ending in the same heading and rest;
   compare after 1, 3 and 10 s. Repeat with opposite legs and closed squares,
   holding final heading fixed and matching total duration. Geometry is only an
   external label. Test pulse pairs and reversal, not merely post-pulse decay.
5. Keep all parameters fixed across trials; repeat half-dt and exact held-drive
   checks, body-ID permutation checks, and zero-input/continued-state controls.
   Use source-row shuffles within the selected hDeltaB projection as an anatomical
   specificity control, with one permutation held across the whole condition grid.

Predictions: a scalar gain will rescale the linear regime without fixing pattern
differences; long constant movement reaches equilibrium instead of indefinitely
growing with distance. At identical final rest input, history differences should
vanish after several hDeltaB/hDeltaG time constants. A closed path need not have
zero neural activity or a unique closure signature. The route may preserve,
attenuate or worsen upstream differences; improved centered similarity is not
proof of a world-vector code and may reflect loss of directional information.

Predeclare descriptive gates: all outputs finite and deterministic; relative
half-dt and exact-solver errors <2%; no inference for centered norms ≤1e-12.
For the provisional null, failure to approach the same final rest state to
within 1% of peak evoked activity by 3 s (max tau_G 0.2 s) flags an implementation
error or an unmodeled input/state, not successful integration. If every direction
collapses to an indistinguishable centered pattern, reject this component as
a useful transfer stage under these assumptions. If same-world similarity
increases while different-world discrimination collapses, reject a claimed
improvement in world invariance. Report full pair distributions and amplitudes;
do not tune gains or rotations to make a pass.

The simplest biological challenge is to record hDeltaG under two matched-final-
heading/rest conditions with different prior displacements, while controlling
feeding/context and measuring vDeltaE when possible. A reproducible
history-specific component after the fast-null response has settled challenges
the relay-only interpretation. It does not identify its mechanism by itself.
Support for displacement memory additionally needs independent coordinate
calibration, duration/speed scaling, cancellation under opposite movements,
turn/closure controls, and perturbations separating vDeltaE input, recurrence,
baseline ramp and reset. Retention, recurrence or slow decay alone is insufficient.

## Inherited limitations and reproducibility

For b=1 the saved model's same-world/different-heading mean centered cosine is
0.71636 raw and 0.82324 matched-rest-subtracted; minima are **0.10357** and
**0.28920** across 224 pairs. A downstream relay is not presumed to repair
these differences. Fixed-heading stationary subtraction is valid only for the
independent steady trials; changing-heading replay requires the continuous
counterfactual above. No peak normalization enters temporal amplitude analysis.

Activate the existing `fly-connectome` environment. Extract into a new directory:

```bash
python -m fly_connectome.circuit_extension results/pfnd-hdeltab-v1.0 \
  --output results/hdeltab-extension-census-final-20261024
python -m fly_connectome.extension_analysis \
  results/hdeltab-extension-census-final-20261024 \
  --movement results/pfn-hdeltab-movement-groupnorm-pfnd6-v1.0 \
  --trajectory results/pfn-hdeltab-trajectory-v1.0 \
  --output results/hdeltab-extension-analysis-20261024
python -m pytest -q
```

Those dated outputs now exist; choose new names when reproducing. Extraction
uses the local ignored `.env` without printing credentials. It saves safe error
types on failure; failed network requests are never treated as empty graphs.
The failed sandbox attempt and earlier census are separate diagnostic directories.
The final census records exact Cypher queries, raw and complete-axis matrices,
package versions, source hashes, source-code hashes, dataset metadata and checks.
The offline analysis verifies hashes before/after, reconstructs every matrix and
summary, and verifies model/census body identities. It writes readable
`connectivity_summary.md`, per-candidate-neuron fractions, reconstruction summaries
and inherited-difference tables. Numerical/data validation does not validate
biological efficacy or the proposed computation. Generated results remain ignored;
preserve both output directories and the uncommitted source changes together.
