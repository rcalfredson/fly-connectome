# CX home-vector capacity: first-experiment proposal

Prepared 30 October 2026. **Reviewed; initial code setup is now implemented.**
See the [implementation walkthrough](cx_home_vector_walkthrough.md) for current
commands, engineering checks and work remaining before the full experiment.
The repository assessment below describes the baseline at proposal time.
This is a separate capacity subproject alongside the existing vDeltaE–hDeltaG
mechanism investigation. It does not replace that investigation or its results.

## Recommendation and feasibility

Recommend **B: train connectome-constrained recurrent dynamics and a stateless
readout**, using prescribed PFNd/PFNv inputs and a bounded, body-resolved
hDeltaB/vDeltaE/hDeltaG network. Run **A: frozen dynamics, trained readout** as
a small baseline on the same trajectory task. A asks whether the specified
dynamics already contain decodable information; B asks whether this selected
architecture and parameter family can learn to retain it. Failure of A does not
answer B. Success of B is a computational existence result under declared
assumptions, rather than evidence that a fly implements the solution.

The repository has sufficient saved anatomy and forward-simulation machinery to
specify a small experiment offline. It has no automatic differentiation,
trajectory training pipeline, fitted Cartesian home-vector decoder, or training
checkpoint workflow. These require new implementation after review. The main
scientific uncertainties are input adequacy, how tightly to constrain efficacies
and signs, the circuit boundary, and how training success generalizes in time.
They are separable from software feasibility.

The recommended first capacity model permits signed edge efficacies to train.
This deliberately weakens physiological constraints: contact counts determine
the mask and initialization, with a soft prior during training. Subsequent
experiments can fix count ratios and impose signs or compartment dynamics.
Do not describe the first model as having measured physiological weights.

## Repository inspection and reusable evidence

Inspected README, the [common framework](common_rate_framework.md), movement,
trajectory, PFN, recurrent and stage-1 modules, their relevant tests, example
JSON files, saved movement/trajectory settings, and the anatomical workflow
guides and summaries. No applicable AGENTS.md was found in the repository or
searched ancestors. The initial working tree contained the user's untracked
dissertation PDF. Credentials were not read; no live query is needed for this
proposal. Existing results and presentation exports remain untouched.

| Existing component | What can be reused | Limitation for this project |
| --- | --- | --- |
| `movement.py`: `heading_mapping`, `encode_pfn`, `movement_weights` | Literal body identities, prescribed heading/body-velocity encoding, legacy normalization and comparison conditions | Algebraic PFN rates, no learned dynamics; transferred tuning is not MaleCNS physiological calibration |
| `trajectory.py`: `sample_inputs`, `world_velocity`, `geometric_reference` | Held-input timing, clockwise coordinates, external geometric targets, continuous sequences | Current default is one short demonstration trajectory, not a training distribution |
| `rate_framework.py`: `RateNetwork`, `RateState`, `coupling_from_counts` | ID alignment, synchronous updates, signed coupling, continuous restart, count audit, optional slow states | NumPy forward solver; parameters do not learn; clamped sources cannot receive feedback |
| `recurrent_model.py` and framework tests | Stability/reference examples and recurrence checks | Earlier restricted PFNd recurrence is not a general CX home-vector model |
| `circuit_extension.py`, `extension_analysis.py`, `vdeltae_inputs.py` | Saved all-Segment counts, full selected populations, source validation and hashes | Bounded surveys, not a complete CX graph; omitted inputs are substantial |
| `vdeltae_contact_audit.py`, `vdeltae_stage1.py` | Contact/compartment provenance and numerical controls for later constraints | Prescribed vDeltaE stimulation, assumed signs, no movement encoding or learned home vector |
| Existing manifests/export conventions | Fresh output directories, explicit assumptions, hashes, failed-run status | Training also needs split, optimizer, RNG and checkpoint provenance |

Relevant saved sources are local and ignored by Git:

- `results/pfn-hdeltab-movement-groupnorm-pfnd6-v1.0`: 40 PFNd,
  20 PFNv and 19 hDeltaB; 24,509 PFN→hDeltaB contacts over 558 positive pairs.
- `results/pfn-hdeltab-trajectory-v1.0`: continuous feedforward replay;
  exact-reference relative error 0.001532 and half-step error 0.000767.
- `results/common-framework-pfn-pilot-20261026`: legacy replay agreement to
  1.39e-17 maximum absolute rate difference. It adds no recurrence.
- `results/hdeltab-extension-census-final-20261024` and
  `results/vdeltae-inputs-live-20261024`: the bounded counts tabulated below.
- `results/vdeltae-hdeltag-contact-audit-20261030`: 2,405 standard all-ROI
  contacts; 2,402 with receiving FB flag, including 1,381 receiving-axon and
  1,021 receiving-dendrite contacts. These scopes must not be mixed.
- `results/vdeltae-hdeltag-stage1-20261030`: 39 prescribed stimuli in two
  DC-matched fast/slow scenarios. The slow tail is imposed by a hypothesized
  excitability state; it is not a discovered displacement accumulator.

The saved PFN model uses baseline 1, velocity modulation 5, reference speed
1 mm/s, group-normalized PFNd×6/PFNv×1 coupling, common gain 0.001 and hDeltaB
tau 0.1 s. Same-world-direction comparisons are **not invariant** to heading/
body-velocity decomposition: the saved raw mean/min centered cosine is
0.7164/0.1036 across 224 pairs (movement-evoked: 0.8232/0.2892). Preserve these
differences and their cause in the prescribed encoding; do not repair them by
aligning responses, peak-normalizing rows, or using targets to modify inputs.

Offline source verifiers passed for the final downstream census, vDeltaE input
census, contact audit, stage-1 run and PFN common-framework run during this review.
Hash verification establishes file consistency, not physiological accuracy.
The existing test suite also passed: **383 tests**, with 14 dependency
deprecation warnings. These checks validate the existing repository baseline;
they do not validate the proposed, unimplemented experiment.

## Primary-source modeling precedents

Publication status below refers to the linked version examined on 30 October
2026. The comparison distinguishes simulation design from biological evidence.

| Source and status | What was constrained and optimized | Relevance and boundary |
| --- | --- | --- |
| [Lappalainen et al., Nature 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/), peer-reviewed | Visual-system model with 45,669 neurons and 64 types; fixed count-derived connectivity and type-pair signs. Trained type-shared time constants, resting potentials and nonnegative type-pair synapse scales, jointly with an instantaneous convolutional decoder, using optic-flow supervision and BPTT. Spatial sharing reduces neural parameters to 734. | Strong precedent for B and for keeping decoder memory absent. Validation against physiology is separate from task fitting. Its short visual sequences, spatial convolution, sign evidence and voltage/release model do not establish CX integration or justify transferring its physiology. |
| [Stone et al., Current Biology 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC6196076/), peer-reviewed | Bee physiology, light anatomy and selected EM motivate a hand-designed rate circuit. CPU4 memory explicitly accumulates compass-modulated speed; CPU1 circuitry steers simulated homing. | Demonstrates a constructive PI design. It does not train an adult Drosophila synapse-count network end to end; accumulation is an assigned operation rather than a learned result of measured recurrent weights. |
| [Hulse et al., eLife 2021](https://elifesciences.org/articles/66039), peer-reviewed anatomy and conceptual modeling | Hemibrain synaptic connectivity exposes PFN phase shifts, FB recurrence and candidate vector transformations. Discusses motifs that could support velocity computation and integration. | Measured connectome is central to the anatomical argument. Proposed PI roles do not constitute a task-trained home-vector network or proof that those motifs implement PI. |
| [Lu et al., Nature 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC10759448/), peer-reviewed | Physiology-based PFNd/PFNv tuning and count-derived PFN→hDeltaB connectivity model a body-to-world movement transformation. | Direct basis for this repository's prescribed-input branch. Instantaneous world-centric movement is distinct from accumulated displacement; this is not end-to-end training of a home-memory circuit. |
| [Lyu et al., Nature 2022](https://www.nature.com/articles/s41586-021-04067-0), peer-reviewed | Physiological measurements, perturbations and a vector-computation circuit model link PFN signals to allocentric travelling direction. | Supports input/coordinate-transform candidates. A travelling-direction signal does not itself retain the initial home location; the paper is not a trained CX home-vector capacity test. |
| [Goulard, Heinze & Webb, PLOS Computational Biology 2023](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011480), peer-reviewed theoretical model | Integrates compass, assigned hΔ temporal accumulation, steering and reward-triggered vector-memory plasticity. Uses anatomically motivated population patterns and chosen gains/rules, rather than fitting every measured body-pair contact. | Particularly relevant PI/goal precedent, but its simulations use constant speed without holonomic motion. Exact steering connectivity and parts of FBt/DAN wiring are simplified/speculative. Its reward plasticity is not BPTT training of the proposed MaleCNS network. |
| [Janke, Rockefeller 2025](https://digitalcommons.rockefeller.edu/student_theses_and_dissertations/808/), dissertation | Imaging/perturbation evidence motivates vDeltaE–hDeltaG dynamics. A phenomenological fluorescence model fits a baseline leak/rise, opposite-column leaky component and direct same-column input to co-imaging using `curve_fit`. | Strong motivation for testing history retention, with important measurement limits. The fitted equations contain no explicit population feedback term and are not a trained contact-resolved home-vector circuit. Recent-path integration with leak is not equivalent to an accurate fixed-home vector. |
| [Beiran & Litwin-Kumar, Nature Neuroscience 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12648571/), peer-reviewed theory | Teacher–student recurrent networks share fixed weights while unknown single-neuron properties are optimized. Tests task-output versus partial-neural-activity constraints. | Explains why task success need not identify dynamics or parameters. Partial recordings can improve predictability, even while parameter recovery remains ambiguous; their synthetic teacher setting is not direct evidence for a CX home mechanism. |
| [Cho, Kang & Kim, arXiv v1, 7 October 2026](https://arxiv.org/html/2610.10014v1), preprint | Whole-MaleCNS topology with one trained signed scalar per edge in a leaky tanh recurrence; fixed input/readout interfaces. Counts define the mask only. Trains bounded arithmetic and relational language tasks, comparing directed degree-preserving rewires. | Recent precedent for anatomical-topology capacity with permissive parameters. It is neither CX-specific PI nor a physiologically signed rate model. Its null design is useful, but reported task success does not validate a biological navigation mechanism. |

The local dissertation's Chapter 4 and model methods were also checked as text;
see the [full-text reading guide](janke_thesis_model_reading_guide.md) for equations,
page references and unresolved sampling/reset conventions. Do not import its
fluorescence coefficients as per-contact rate parameters.

These precedents already establish anatomical PI models and trainable
connectome-constrained networks. This bounded search does **not** establish a
novelty claim for their intersection. The useful deliverable here is a controlled
capacity result on a specified MaleCNS subnetwork, followed by biological tests.

## One minimal experiment

### Circuit and boundary

Use 45 dynamic bodies: **19 hDeltaB, 18 vDeltaE and eight hDeltaG**, plus the
existing 60 externally prescribed PFNd/PFNv sources. Retain every body, including
disconnected cases and duplicate/C0 annotation labels. The three dynamic types
offer a known movement-related entry point, recurrent anatomy and a candidate
history-sensitive motif; they are a manageable starting hypothesis, not a claim
that vDeltaE→hDeltaG is necessary or sufficient.

All nine internal directed type-pair matrices already exist:

| Presynaptic population ↓ / Postsynaptic population → | hDeltaB | vDeltaE | hDeltaG |
| --- | ---: | ---: | ---: |
| hDeltaB | 882 | 57 | 736 |
| vDeltaE | 0 | 102 | 2,405 |
| hDeltaG | 9 | 653 | 361 |

Entries are raw standard all-ROI contacts, totaling **5,205 contacts over 478
positive body pairs**. Zero means no connection reported in this snapshot.
Retain reported self-connections, but never invent an identity edge for memory.

Use the following saved projections as the **declared input boundary**:

| Projection | Positive body pairs | Contacts |
| --- | ---: | ---: |
| PFNd and PFNv → hDeltaB, combined | 558 | 24,509 |
| PFNd → vDeltaE | 1 | 1 |
| PFNv → vDeltaE | 38 | 109 |
| Total input boundary | 597 | 24,619 |

Together with 478 internal pairs, this gives **1,075 modeled positive edges**. It is not
the complete induced graph over all 105 bodies: PFN→hDeltaG is unqueried here
and explicitly omitted, and every edge into a clamped PFN is excluded by design.
An optional later bounded query of exactly those two missing input projections
would test this boundary assumption; do not label missing queries as zero.

All other sensory, tangential, compass, reward and downstream steering populations
are outside the model. For perspective, modeled PFNd/PFNv/hDeltaB sources explain
only 167/14,583 vDeltaE input contacts before including hDeltaG and vDeltaE
itself. The model cannot claim a complete physiological account of vDeltaE.
Keep a table of retained and omitted projections. No partner triggers expansion.
Internal recurrence runs at every timestep without any extraction-hop horizon.

### Inputs and neural dynamics

Supply only current heading and forward/right body velocity to the existing
PFN encoder, with baseline 1 and the saved tuning and modulation constants.
The neural simulator receives the resulting PFN rates `u(t)`, not geometric
velocity integrals, time, position or target. Do not invent movement tuning for
vDeltaE or hDeltaG. Heading is externally prescribed; a learned head-direction
system is outside this first experiment.

For dynamic neuron j, use the existing framework's rate family:

```text
tau[type(j)] dr_j/dt = -r_j + ReLU(b[type(j)]
                         + sum_i W_input[i,j] u_i(t)
                         + sum_k W_rec[k,j] r_k(t))
home_hat(t) = D r(t) + d
```

Rates use arbitrary units; gain is fixed to 1 and threshold to 0 to avoid
redundant gain/threshold/bias parameters. Use synchronous held-input updates.
Primary model has **no slow excitability state, no imposed ramp, no explicit
accumulator and no plasticity during inference**. Effective recurrence must
extend memory beyond the short intrinsic time constants if the task is solved.

Reset dynamic rates to zero only at trajectory initialization, when home is
established. Do not reset at pauses, turns, input segments or a return to home.
Zero is an experimental reset convention, not a reproduction of sugar reset.

### Fixed, trained and sensitivity quantities

| Quantity | Primary specification |
| --- | --- |
| Fixed | Bodies, directed positive-edge mask, raw contact table, PFN encoding, absence of slow states, gain/threshold, initial state, task/split rules |
| Trained | One signed coefficient `a_ij` per retained anatomical edge; three type-shared biases; three type-shared time constants; linear `D` (2×45) and intercept `d` |
| Time constants | Initialize 0.1 s; constrain 0.05–0.5 s. These are exploratory bounds, not measured values |
| Biases | Initialize 0.1; bound to [-2, 2] arbitrary input units |
| Coupling | `W_ij = (C_ij / S_ab) * a_ij`, only on retained edges. `S_ab` is the mean total incoming count across **all** selected targets of type b for projection a→b; all-zero blocks stay absent |
| Initialization/prior | Set input coefficients to +1 and recurrent coefficients to small shared-per-block random values (SD 0.1); perturb across seeds. A weak quadratic penalty on `a - a_initial` favors the initial count profile |
| Sign | Free signed coefficients, bounded to [-5, 5]; learned signs are computational parameters, not transmitter/receptor measurements. No source-wide Dale constraint is claimed |
| Normalization | `S_ab` is one fixed scalar per projection, preserving initial within-block count ratios. No dynamic normalization, population centering, row peak scaling or normalization by all biological input |
| Decoder | Stateless affine map of all 45 current rates; no PFN skip, clock, temporal window, recurrent decoder or latent slow state |

There are **1,173 trainable scalars** (1,075 edges + six intrinsic parameters +
92 decoder parameters). Count ratios are **not fixed after training**: counts
provide sparsity, initialization and a soft prior. Report coefficient changes,
sign changes, bound hits and the influence of the prior. This permissive capacity
test must not be presented as the stronger proportional-count model in the
visual-system precedent.

After the primary comparison, the first restrictive sensitivity is one signed
coefficient per directed type pair, fixing relative count strengths exactly.
Further one-at-a-time sensitivities: count-only versus legacy PFN group
normalization/×6 initialization, fixed tau=0.1 s, evidence-labeled sign scenarios,
and the audited receiving-compartment split with optional slow states. Do not
silently reuse the legacy gain 0.001 after the new normalization. A slow-state
extension is a distinct memory mechanism and requires its own ablation; do not
count one anatomical contact in both fast and slow channels without allocation.

### Geometric task and history safeguards

Use repository coordinates: heading zero faces +x, clockwise positive, +y
right/down. An external task generator computes, on held intervals,

```text
v_world = (vf*cos(theta) - vr*sin(theta),
           vf*sin(theta) + vr*cos(theta))
p[0] = home = (0, 0)
p[n+1] = p[n] + dt * v_world[n]
target[n] = home - p[n]
```

`target[n]` aligns with `r[n]`, before the next interval is integrated. Positions
and targets live in the task/loss/evaluation layer, never in the neural update.
No teacher-forced state or trajectory-specific initial condition is permitted.

Start with 1,024 training, 128 validation and 256 test trajectories of 4–16 s.
Draw segment durations from 0.25–2 s and speeds from 0–2 mm/s; independently vary
heading and body-translation direction across the full circle, with backward and
lateral movements and 20% zero-translation segments. Randomize duration,
orientation, segment counts and order; balance velocity signs. Include matched
history trials in every split. Defaults are design choices, not fitted biology.

For example, a pair travels east versus west for equal times, then both adopt
the same heading with zero translation for 5 s. At each matched timestamp,
current inputs, heading, speed and elapsed time agree while home vectors differ.
Use diverse prefixes and identical suffixes, including moving suffixes, and
randomly rotate the paired construction. Include loop-versus-outbound histories
matched for duration and approximate path length. Evaluate separation throughout
the common suffix, not just immediately after the prefixes.

Split by complete trajectory **families**, grouping paired histories, rotations,
reversals and replays together before assignment. Never split adjacent time
samples or place transformed copies on opposite sides of a split. Fit all
feature/target scales using training data only. Supply no elapsed-time input.
A nuisance baseline that receives current inputs plus time tests any residual
dataset shortcut; time is not supplied to the CX model.

Keep separate challenge sets: reversals; exact closed squares and non-square
loops; 1/5/10 s pauses away from home; heading changes at zero translation;
lateral/backward motion; and the same world path generated using different
heading/body-velocity histories. Hold out intermediate headings, speeds and
turn sequences. Test 32 and 64 s paths after training on ≤16 s, reporting
extrapolation separately from in-distribution performance.

### Readout, fitting and numerical checks

Train dynamics and the affine readout jointly by full-sequence BPTT and Adam
(initial learning rate 1e-3, batch size 8, maximum 2,000 updates per fit), selecting
checkpoints by validation Cartesian MSE. Sample whole trajectories; weight each
trajectory equally and average loss over its valid timestamps. Optimize both
Cartesian components using one training-set RMS length scale `L`, fixed across
time and splits. No per-trajectory target normalization.

Provisional dimensionless objective:

```text
loss = mean_over_trajectories(
           mean_over_time(
               squared_norm((home_hat - target) / L)
           )
       )
       + 1e-5 * mean(r²) + 1e-4 * mean((a-a_initial)²)
       + 1e-5 * mean(D²)
```

Use gradient-norm clipping at 1. Report task error independently of penalties.
Any tuning uses a small declared validation-only grid and the same budget for
anatomical and randomized graphs. Never optimize against the challenge/test sets.

Start exponential Euler at dt=2.5 ms with tau≥50 ms. It is a held-recurrent-drive
approximation, not an exact recurrent solution. Before training, compare the
differentiable forward pass with `RateNetwork`, verify finite-difference gradients
on a tiny recurrent example, and check deterministic restart and ID permutation.
Evaluate trained checkpoints at half dt with the same physical inputs and taus;
require <2% RMS relative state discrepancy and <5% change in task RMSE. A failed
gate requires a finer grid and renewed validation, not a performance claim.
Reject divergent/nonfinite runs; report rate/bias/weight bound hits and do not
mistake the numerical rate guard for physiological saturation.

Fit A using ridge regression on frozen current-rate features, choosing ridge on
validation trajectories. First use the unchanged legacy PFN→hDeltaB replay;
also freeze the proposed 45-cell model at its recorded initialization to isolate
the effect of dynamics training on identical topology. These are specified
frozen models, not an optimized survey of all untrained dynamics. Neither the
old movement-angle decoder nor the thesis fluorescence proxy is a Cartesian
home-vector decoder.

### Small control set

1. **Current-input information:** fit an affine decoder and a small stateless
   MLP (one hidden layer, 32 units) directly from current PFN rates to target.
   Also test current heading/body velocity plus time as nuisance features.
   All should fail the matched-history separation test. As a diagnostic only,
   decode instantaneous world velocity from PFN rates; poor performance flags
   an input/encoding limitation before attributing failure to recurrence.
2. **Memory:** remove recurrent edges and retrain with the same short-tau
   states and decoder. Short filtering remains, so inspect late common-suffix
   errors. Separately reset the trained network's rates at the start of a
   matched suffix: prediction separation should disappear. No slow states are
   present to retain hidden memory in this primary model.
3. **Randomized wiring:** train an ensemble of directed double-edge-swap nulls
   independently under identical task, optimizer and validation budgets.
   Swap within source-type/target-type blocks, keeping each body's blockwise
   in/out degrees, node identities, input/output memberships, edge count and
   parameter count. Keep self-loops fixed; reject duplicate/new self edges.
   Shuffle the original positive contact counts over the rewired edges within
   each block, preserving its count multiset and therefore `S_ab`. Apply this
   to input and recurrent projections. This preserves degree and coarse type
   composition but not each body's weighted strengths, spatial alignment,
   reciprocity or higher-order motifs. Report those changes rather than
   claiming they are preserved. Dense or tiny blocks may admit few swaps;
   report retained-edge fraction and mixing diagnostics before interpreting a
   null. A mere neuron relabeling is not an appropriate null with a free decoder.

Use five training seeds for anatomy, five independently rewired graphs with
paired training seeds, and the same ablations. For an economical pilot, use three
seeds and three null graphs before scaling the comparison. Report all fits,
including failures; resample complete trajectory families for uncertainty and
show graph/seed variability separately. A small null ensemble is exploratory,
not a precise tail-probability test.

### Predictions and failure criteria

Report Cartesian RMSE in mm, normalized RMSE, component R², magnitude error,
closure residual, and error versus elapsed time, distance and pause duration.
Define normalized RMSE as `sqrt(sum ||error||² / sum ||target||²)` over the
specified evaluation set. For angles use wrapped error only where true length
is ≥1 mm; below this threshold report Cartesian error without direction. Count
predictions of near-zero length for non-near-home targets as direction failures,
not as conveniently omitted samples. Report near-home sample counts.

Provisional capacity gate, fixed before training: test normalized RMSE ≤0.20,
at least 50% lower Cartesian RMSE than the best current-input control, and
≤0.20 normalized RMSE on the late (last second) samples of 5 s matched pauses.
The pause denominator uses that subset's target energy. Report median angular
error, but do not substitute it for the Cartesian gate. Closed-loop terminal
error should be ≤1 mm on the designated speed/path range. Also report all
challenge results even when these primary gates pass.

Expect A to retain short recent-motion information but degrade over pauses;
expect a successful B to maintain distinguishable states after identical
suffixes and reverse displacement correctly. Whether biological wiring exceeds
the degree/type-matched null is an open prediction, not an assumed outcome.

Failures have distinct interpretations:

- A fails: these frozen dynamics/readout lack usable information under this task.
- B fits training only: memorization, shortcuts or limited generalization; no
  capacity claim on the held-out distribution.
- B fails reproducibly: this boundary, input, parameter family and optimizer
  did not produce a solution. Check input ambiguity and numerical/optimization
  controls; this is not a proof that CX anatomy cannot integrate.
- B passes short tests but loses the vector over long pauses/paths: finite-horizon
  or leaky history retention, not a reliable longer-term fixed-home code.
- B matches rewired controls: generic recurrent capacity remains a sufficient
  explanation. Do not claim that biological wiring specifically supports PI.
- Anatomy consistently outperforms nulls: evidence for a contribution beyond
  the preserved graph statistics in this task/model family, not a unique
  biological mechanism.

## Staged implementation and compute

1. **This review:** agree on the permissive capacity question, 45-cell boundary,
   count role, memory-free decoder and evaluation gates. Preserve this proposal
   separately from the existing mechanism documents. Review preceded the user's
   authorization to start code setup; subsequent engineering runs are documented
   in the implementation walkthrough.
2. **Data/task foundation:** after review, add an isolated home-vector package
   and fresh output root. Reuse verifiers and ID tables to assemble the explicit
   105-node/1,075-edge boundary. Save inclusion/omission tables and immutable
   source hashes. Build trajectory-family generation, splits and external
   targets, with independent geometry/timing and leakage checks. No neuPrint
   credential is required for the selected saved-data boundary.
3. **Baselines and differentiable parity:** implement stateless current-input
   and frozen-state fitting; then a batched PyTorch counterpart of the common
   equations. Add meaningful parity, gradient, reset and convergence tests.
   Keep the NumPy solver as the independent forward reference. Add training
   dependencies separately/pinned after choosing a compatible build; do not
   change existing commands or silently add GPU requirements to them.
4. **Capacity pilot and controls:** benchmark 100 updates, train B and ablations,
   then run rewires with matched selection budgets. Save complete configs,
   seeds, split family IDs/hashes, raw/effective coupling, code hashes, optimizer
   state, checkpoints and numerical checks. Separate checkpoint selection from
   final test evaluation and keep failed manifests.
5. **Constraint ladder and interpretation:** fix count ratios, add sign evidence,
   test audited compartment channels/slow states and targeted boundary changes
   one at a time. Fit measured activity only in a separately defined stage with
   held-out physiological validation. Continue the existing vDeltaE mechanism
   work without treating its hypotheses as validated by this task.

The graph is small; sequence unrolling is the likely bottleneck. A 16 s rollout
at 2.5 ms has 6,400 steps. Batch 8×6,400×45 float32 rate samples occupy about
9.2 MB, but autodiff intermediates, input traces and graph bookkeeping can cost
much more. One GPU with roughly 8–16 GB is a sensible pilot target; CPU inference
and baselines should be feasible. No GPU or PyTorch installation was assumed
or benchmarked during this review. Do not extrapolate wall time from node count.

Budget at most 2 GPU-hours for benchmarking/the first fit, then estimate the
ensemble from measured time per update. A planning range is hours per fit and
tens of GPU-hours for a small controlled ensemble, with substantial uncertainty
from long unrolls and implementation overhead. Full 5×5 graph/seed comparison
is 30 trained fits before ablations, not one run. Prefer full-trajectory gradients
and activation checkpointing over silently truncating the memory horizon.
If projected cost exceeds a provisional 24 GPU-hour pilot budget, resolve that
resource choice before launching the ensemble. These are future budget caps,
not work started or a claim about available hardware.

## What success would mean

A stateless decoder succeeding on matched-history tests establishes that current
simulated neural state contains recoverable home-vector information. Retention
during identical pauses plus state-reset/recurrence ablations supports storage
of displacement-related history in this model. An advantage over well-matched
rewires supports an anatomical contribution beyond the statistics preserved by
that null. These are progressively different conclusions.

Compare separate-population and joint decoders on the **same trained network**,
with held-out regularization selection, to ask whether information is more
accessible jointly. Such differences depend on decoder class, noise and sample
size; they do not establish that the fly uses a distributed code.

Decoder weights alone cannot identify a circuit mechanism. Follow successful
models with transient state perturbations during a pause, blockwise connection
silencing, selective hDeltaB/vDeltaE/hDeltaG suppression, and vDeltaE→hDeltaG
or hDeltaG→vDeltaE ablation while recording recovery and error components.
Distinguish acute ablation with a frozen decoder from decoder refitting and
whole-network retraining: these test reliance, residual accessibility and
compensatory capacity respectively.

Biological plausibility additionally requires independent sign, input, timescale,
compartment and activity constraints. Actual use by the fly requires relevant
behavioral and causal evidence. This supervised task contains no homing policy,
learning rule used by the animal, sensory position estimate or proof of a
fixed-home representation in vDeltaE, hDeltaG or any other single population.
