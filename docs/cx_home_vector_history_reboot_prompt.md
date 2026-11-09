# Starter prompt: explain history dependence from first principles

Copy the text below into a new thread. Written November 9, 2026.

---

Work in `/home/tracking/code/fly-connectome`. I want to restart the analysis of
our frozen CX home-vector model from first principles, using previous work as
raw material rather than following its exploratory sequence.

My central question is: **How does the same instantaneous heading and
body-centered movement produce different neural population activity depending
on the preceding trajectory, and which parts of that dependence actually store
and update the home vector?** I suspect recurrent dynamics matter, but want a
concrete mechanistic explanation rather than just naming recurrence.

Start by reading the actual model and existing analysis tools. First explain
the state update clearly, then propose a small, ordered investigation with
explicit questions, controls, expected observations, and interpretation limits.
Please discuss that plan with me before adding more analysis code. Do not
train another model or change the encoder, weights, decoder, or task.

## Frozen subject

- Use the archived **best checkpoint at update 1975**, not the final update
  2000, from `results/cx-home-vector-balanced-u1975-activity-20261106`.
  Checkpoint SHA256:
  `2aab91f13258883819f66f12897919b2bf509617e7927120ca7f9db069d9a31f`.
- Original training run:
  `results/cx-home-vector-hdeltac-long-balanced-seed0-u2000-resumed-2026-11-05`.
- Prepared source:
  `results/cx-home-vector-hdeltac-long-data-20261104`.
- Dynamic circuit: 65 rate units—hDeltaB 19, vDeltaE 18, hDeltaG 8, hDeltaC 20;
  prescribed PFNd/PFNv inputs, 60 units. Rates have model units, not calibrated Hz.
- State step: 2.5 ms. For row-vector activity and source-row/target-column weights:

  ```text
  input drive = PFN(t) @ W_input
  recurrent drive = r(t) @ W_recurrent
  target rate = relu(bias + input drive + recurrent drive)
  r(t+dt) = r(t) + alpha * (target rate - r(t))
  alpha = 1 - exp(-dt/tau)
  predicted home(t) = r(t) @ D.T + d
  ```

- The decoder is instantaneous and linear, with no separate memory. Stored
  state is in neural rates; the model has no extra slow-state variable.
  The inspected rates at t precede the update driven by the input at t.

## Interpretation safeguards

1. Match speed as well as heading and body direction, and verify actual PFN
   input equality. A matched world path alone does not imply identical PFN input.
2. PFN-to-dynamic responses are overlapping mixtures, not a one-to-one mapping
   of one movement combination to one neuron. Multiple neurons share sampled
   peaks. The prescribed PFN encoder, anatomical counts, normalization, trained
   coefficients, bias and time constants all affect the responses.
3. Carrying the target neuron's own activity forward can produce short-lived
   history dependence even without recurrence. Separate this from recurrent
   support of longer memory. Timescale bounds in this fit are 0.05–0.5 s.
4. Distinguish an activity offset, changed response strength, changed tuning
   shape, and changed preferred direction. An unchanged peak does not rule out
   shape changes; high activity does not establish a directional preference.
5. Distinguish **different displacement with identical current input**, where
   a useful integrator must retain different home vectors, from **equal
   displacement with different motor histories**, where different neural states
   may encode information irrelevant to the decoded home vector. Raw state
   differences alone do not prove functional path-integration memory.
6. Signed weighted-input accounting is descriptive local mechanism, not proof
   of necessity. Interventions need controls for generic disruption. Neither
   state dependence nor preserved tuning establishes what training created.

## Existing raw material

- Standard challenge activity, every neuron and full PFN traces:
  `results/cx-home-vector-balanced-u1975-activity-20261106/index.html`.
- Mixed and curved trajectories, every neuron in `path_XX.npz`:
  `results/cx-home-vector-balanced-u1975-activity-by-trajectory-20261106/index.html`.
  This directory was renamed from
  `cx-home-vector-balanced-u1975-raw-paths-restart-20261106`.
  Its displayed traces show only six highest mean-rate units plus 142911;
  absence from a trace legend does not mean absence from the saved data.
- Matched-state heading/body grids for five selected cases:
  `results/cx-home-vector-balanced-u1975-trajectory-history-checks-20261108/index.html`.
- Signed incoming PFN/recurrent drive, population/partner contributions,
  rectification and explicit own-state update accounting:
  `results/cx-home-vector-balanced-u1975-incoming-drive-20261108/index.html`.
- Earlier onset atlas and pretraining comparison:
  `results/cx-home-vector-balanced-u1975-pfn-input-atlas-20261106/index.html` and
  `results/cx-home-vector-balanced-u1975-onset-initial-comparison-20261107/comparison.json`.
  Much initial tuning precedes training in this encoding/circuit/model setup;
  that is not an anatomy-only or no-learning conclusion.

Useful code lives in `fly_connectome/home_vector/model.py`, `data.py`, and
`analysis/{activity,activity_checks,activity_paths,history_tuning,trajectory_tuning,incoming_drive}.py`.
The corresponding guides are in `docs/cx_home_vector_*.md`. Read the small
reports/CSVs before opening large metadata files or regenerating figures.

## Concrete anchor, not a required analysis sequence

Paths 6 and 7 share the same world path and target vector. From 6–8 s their
PFN inputs are identical, but their neural states differ. At 6 s, for 17816,
PFN drive is 1.999 in both states, recurrent drive is 26.847 versus 18.601,
and actual rate is 26.668 versus 19.602. Much of the recurrent-drive difference
comes from hDeltaG 20565. This identifies a local input difference, but does
not explain the full feedback mechanism or establish that the difference stores
the home vector. Pair this example with a different-displacement/identical-input
example from the standard pause challenges if useful.

Our selected neurons in cases 1–4 mostly preserved movement-response tuning
despite different activity levels. Case 5, especially 59176, had changes in
amplitude and shape with unchanged sampled peaks, accompanied by different
rectification. These are results to explain, not premises that all neurons
are hard-wired or all histories behave alike.

## What I want first

Give me a compact causal picture of what is already implied by the equations
versus what remains an empirical question about this trained solution. Then
recommend the smallest set of analyses needed to explain:

- where trajectory information resides in the population state;
- what preserves it versus letting it decay, and how new movement updates it;
- which state differences affect the decoded home vector versus cancel in the
  decoder's null space;
- what additional evidence would connect particular recurrent feedback paths
  to these functions.

For each proposed step, state what result would support or contradict the
explanation. Prefer one worked example and a small, interpretable experiment
over another broad collection of tuning maps. Keep biological plausibility and
cross-task training comparisons as later questions unless needed to resolve
the mechanism first.
