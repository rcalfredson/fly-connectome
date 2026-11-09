# Frozen-network activity inspection: implementation plan

Proposed November 6, 2026. **Milestone 1's recorder, activity atlas and population
readout are now implemented:** see the [usage guide](cx_home_vector_activity.md).
Decoder row/null-space accounting, extended pauses and initial causal/robustness
checks are also available in the [follow-up report](cx_home_vector_activity_checks.md).
As of November 9, signed drive reconstruction and matched-state tuning probes
are also implemented: see the [incoming-drive audit](cx_home_vector_incoming_drive.md)
and [trajectory-state comparisons](cx_home_vector_trajectory_tuning.md).
Event zooms and pause-only pathway interventions remain planned. This historical plan
turns the [dynamics study outline](cx_home_vector_balanced_u2000_assessment.md#next-characterize-the-frozen-solution)
into code and output milestones. No training or model changes are proposed for
the first two milestones.

## Fixed subject and questions

Study `best.pt`, selected at **update 1975**, from
`results/cx-home-vector-hdeltac-long-balanced-seed0-u2000-resumed-2026-11-05`,
with source `results/cx-home-vector-hdeltac-long-data-20261104`. Preserve the
checkpoint bytes/hash, decoder, PFN encoder and challenge definitions. The
65 dynamic units are hDeltaB (19), vDeltaE (18), hDeltaG (8) and hDeltaC (20).
Vector-change supervision was disabled in this fit.

| Question | First measurement | Human-readable output |
| --- | --- | --- |
| What changes at starts, stops, turns and reversals? | Every unit's rate, aligned to control transitions | Activity heatmaps, individual traces and event zooms |
| Is a stable output also a stable neural state? | State changes during pauses, separated by what the decoder can observe | Pause-state and decoded-drift panels |
| Which populations contribute to the home vector? | Actual signed contributions through the frozen decoder | Population contributions to home-vector X and Y |
| How does activity keep accumulating, or stop accumulating? | Activity and input/recurrent drive through constant-motion segments | Drive and state-change panels alongside displacement |

The decoder has no independent temporal memory. It still performs an important
spatial readout and can conceal changes in neural activity through cancellation.
Thus stable output alone does not establish stable activity or a physiological
memory mechanism. Activity here has **model units**, not calibrated firing rates
in Hz.

## Milestone 1: reusable recorder and activity atlas

Add three analysis modules:

- `analysis/activity.py`: CLI, checkpoint capture, trajectory replay, provenance
  and output orchestration. Reuse `analysis.checkpoint.capture` and the existing
  protocol/encoder/model interfaces.
- `analysis/activity_summary.py`: pure array-based summaries and readout
  decomposition, independently usable in notebooks.
- `analysis/activity_plots.py`: figures built from the saved arrays and summaries.

`HomeVectorNetwork.forward` already returns both predictions and all neural
rates; `evaluate` also returns states. Use these interfaces rather than writing
a second simulator or adding recording logic to the trainer. Replay one
trajectory at a time on CPU, without gradients, and save no padded rows.

Begin with all 12 existing challenge trajectories, retaining their identities
and protocols. This connects each internal trace to behavior already inspected.
The proposed interface follows the checkpoint evaluator:

```bash
# Recorder/atlas command; implemented follow-ups are linked above.
python -m fly_connectome.home_vector.analysis.activity \
  --run results/cx-home-vector-hdeltac-long-balanced-seed0-u2000-resumed-2026-11-05 \
  --source results/cx-home-vector-hdeltac-long-data-20261104 \
  --output results/cx-home-vector-activity-u1975-NEW \
  --threads 1
```

Support archived checkpoint-evaluation bundles through the existing capture
interface as well as training directories. A fresh output directory must retain
the selected checkpoint and metadata so that the analysis remains reproducible.

### Saved data

Each trajectory gets a compressed NPZ with full-resolution arrays:

- `time_s[T]`, controls `[T,3]` with declared heading/forward/right conventions;
- encoded PFN activity `[T,60]` and dynamic rates `[T,65]`;
- predicted and target home vectors `[T,2]`.

Save neuron/input IDs, types, array conventions and model units in accompanying
tables/metadata. An index identifies each trajectory, its protocol, event times
and artifact filenames. Record source/checkpoint/code hashes using existing
audit conventions. Use flat artifact paths compatible with bundle hashing.

Keep the full 2.5 ms resolution in saved traces and quantitative summaries.
Display-only downsampling must be declared and preserve event boundaries. Drives
can be reconstructed from the checkpoint and traces; avoid saving multiple
redundant full-size arrays by default.

### First figures

1. **Activity atlas:** one panel per population, neurons on rows and time on
   columns, with starts/stops/turns marked. Include control traces and decoded
   versus true vectors for context. Use a shared raw-activity scale, plus a
   separately labeled change-from-event-baseline view when useful.
2. **Population readout:** signed X/Y contributions from each population, their
   sum plus decoder intercept, and the true vector. Use line plots that preserve
   negative contributions and cancellation.
3. **Pause detail:** activity change and output drift after opposite movement
   histories, aligned to pause onset. Milestone 2 adds the decoder-subspace view.

Order neurons by population and recorded body ID/index. The current circuit
bundle has no dynamic-neuron anatomical column metadata; arbitrary ID order
must not be presented as spatial order or evidence of a traveling bump.
Population means alone can hide opposing activity changes, so retain all units.

Export PNG/PDF figures, per-neuron/population/event CSV summaries and a small
linked HTML index with checkpoint identity and figure captions. A later optional
neuron/case selector can help exploration; standalone figures remain the
shareable scientific artifacts. No new web application is needed.

## Milestone 2: explain readout stability and activity changes

### Exact decoder accounting

For population p, compute its contribution using the observed activity:

```text
home_p(t) = D[:, population_p] × rates_population_p(t)
predicted_home(t) = sum_over_populations(home_p(t)) + d
```

Include the intercept once. A large decoder coefficient alone does not establish
a neuron's importance: actual activity, cancellation and perturbation responses
also matter. Decoder coefficient signs are readout signs, not synaptic signs.

For a pause, define `delta_r(t) = r(t) - r(pause_onset)`. Use an SVD of the
fixed decoder to separate this change into its row space and null space:

```text
delta_r_visible = projection of delta_r onto the decoder row space
delta_r_hidden = delta_r - delta_r_visible
delta_home = D × delta_r = D × delta_r_visible
D × delta_r_hidden = 0   (to numerical precision)
```

The selected decoder is a rank-two 2-by-65 matrix, leaving 63 null-space
dimensions. This is a diagnostic opportunity, not evidence that the actual
trajectory drifts in those dimensions. Record singular values and the rank
tolerance. Plot neural-state norms and decoded drift on separate, labeled axes:
they have different units. This distinguishes a nearly stationary state from
state changes hidden by the readout.

### Reconstruct the actual update

Use effective weights from `network.weights()`, including contact-count
normalization and trained coefficients. With source rows and target columns:

```text
input_drive(t) = PFN(t) @ W_input
recurrent_drive(t) = rates(t) @ W_recurrent
z(t) = input_drive(t) + recurrent_drive(t) + population_bias
q(t) = relu(z(t))
alpha = 1 - exp(-dt / population_tau)
rates(t+1) - rates(t) = alpha * (q(t) - rates(t))
```

The recorded state at t precedes the update driven by input at t. Respect this
alignment when marking control changes and comparing predicted increments.
Input, recurrence and bias add **before** ReLU; do not treat them as independent
additive causes of the post-activation rate change. The exact update separates
into `-alpha * rates` and `alpha * q`.

Plot pre-activation drives, activation and rate increments for each population
and selected neurons. Declare selection criteria, such as the largest observed
readout contribution, while retaining all-unit data. For the long straight path,
show whether changes remain aligned with displacement or increasingly produce
excess growth. A single constant-speed path confounds elapsed time and distance;
resolve that with matched probes in the next milestone.

## Milestone 3: controlled probes, then causal tests

After the first activity report, add diagnostic trajectories varying speed,
heading and body-centered direction, with matched duration and matched distance.
Compare the same final displacement reached by different histories, and
different histories followed by identical stationary inputs. Separate
displacement-related patterns from responses to current input or elapsed time.
These are evaluation probes; do not fit a new decoder to make them agree.

Then test candidate mechanisms through state resets, population perturbations
or pathway suppression at a specified event, using matched controls for generic
activity disruption. Existing recurrence-removal analysis removes recurrence
throughout the trial; it cannot isolate retention after a pause begins. A
pause-only intervention needs a targeted implementation and its own validation.
Local Jacobians/slow modes can follow the descriptive analysis, but do not by
themselves establish a global attractor or a physiological mechanism.

## Verification and completion criteria

Before interpreting figures, verify:

- Checkpoint identity, source hashes and replay agreement with existing challenge
  predictions; no parameters or task definitions change during recording.
- Correct lengths, IDs, population membership, finite arrays and event timing.
- Decoder contributions sum to the original prediction including the intercept.
- Reconstructed state increments agree with the model, including at switches.
- Decoder-visible/hidden changes reconstruct the original state change, are
  orthogonal, and hidden changes have negligible decoded contribution.
- An artificial example distinguishes null-space activity drift with constant
  output from visible drift. This checks the intended interpretation.
- Plot downsampling does not affect metrics, and quiet/zero-variance units do
  not break summaries or scaling.

Use meaningful numerical tolerances for replay/matrix-operation rounding, with
the existing checkpoint replay guard as a reference. Tests should include a
control transition to catch a one-sample timing error.

The first deliverable is complete when the existing challenges have reproducible
full activity archives, an activity atlas and population readout plots. The
second adds pause-state and drive accounting with verified reconstruction.
Keep causal interventions and literature comparison as later, separate changes.

For physiological comparison, turn observed mechanisms into specific testable
claims about cell-type tuning, memory dynamics, interaction signs and timescales.
Separate published measurements from model assumptions and unknowns. Plausible
plots alone cannot establish that the fitted circuit replicates the CX.
