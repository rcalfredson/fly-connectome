# History-dependent heading/body-movement tuning

For matched probes from the five selected mixed/curved-trajectory cases, see
the [trajectory-state follow-up](cx_home_vector_trajectory_tuning.md).

This frozen-model experiment asks whether identical probe inputs elicit different
responses after different trajectories. It uses the production rate equation and
prescribed PFN encoder; it performs no training or decoder refitting.

The default histories last four seconds, all at heading 0°: stationary, forward
walking at 1 mm/s, and backward walking at 1 mm/s. Each begins at zero state.
Their endpoint home vectors are respectively (0, 0), (−4, 0), and (4, 0) mm.
Stationary means zero velocity, **not** zero PFN activity.

Every history's complete endpoint neural state seeds 64 movement probes
(eight global headings × eight body-centered directions, at 1 mm/s) and eight
stationary controls, one at each probe heading. Every probe receives its own
copy of that state. The state evolves during each probe; there is no reset or
carryover between probes. We inspect 2.5, 50, and 250 ms after the probe begins.
The state at time t precedes the update driven by the input at t, so the saved
terminal history state has not received the terminal zero-velocity control.

## Run

Use the archived activity snapshot to select the same frozen update 1975 as
the earlier atlases, rather than a subsequently replaced checkpoint:

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fly-connectome-mpl \
python -m fly_connectome.home_vector.analysis.history_tuning \
  --run results/cx-home-vector-balanced-u1975-activity-20261106 \
  --source results/cx-home-vector-hdeltac-long-data-20261104 \
  --output results/cx-home-vector-balanced-u1975-history-tuning-20261107 \
  --threads 1
```

The output directory must be new. `--run` also accepts the training runs and
saved challenge snapshots supported by `analysis.checkpoint.capture`.
`--snapshot-key reference` selects the reference model in a saved snapshot.
Inputs are verified and preserved; checkpoint bytes, identity, configuration,
protocols, code hashes and complete probe traces are archived with the results.

Options include `--history-seconds`, `--heading`, `--history-speed`, `--speed`
(probe speed), `--angles`, `--times`, `--focus`, and `--chunk-size`. Durations
and sampling times must fall on the prepared simulation's time grid.
The default focus is vDeltaE 142911, hDeltaB 17816, and hDeltaC 20231.
Focus selection affects plots only; measurements retain every dynamic neuron.

## Read the output

Open `index.html`. For each focus neuron, columns compare the three histories
and rows compare probe times. The three response views are:

| View | Definition | Interpretation |
| --- | --- | --- |
| Actual firing rate | r_moving(t) | Includes activity carried from the history. |
| Change from starting rate | r_moving(t) − r_history_endpoint | Separates the subsequent change from the inherited level. |
| Movement minus stationary | r_moving(t) − r_stationary(t) | Compares movement with the same-heading continuation from the exact same state. |

The last control includes the ongoing recurrent evolution and any response to
the new heading, rather than assuming the state would remain constant without
walking. It is a paired response difference, not a decomposition into independent
biological mechanisms. Changing probe heading can be abrupt, identically for
moving and stationary conditions within each history.

Raw and difference figures share one scale across all histories/times within a
neuron/view figure. Additional movement-response pattern figures normalize each
map separately to 0–1, with flat maps gray. They reveal shape while discarding
offset and amplitude. Rates are model units, not calibrated Hz.

Subtracting one starting rate per neuron shifts every grid point equally:
**raw and change-from-start maps have identical tuning shape at a fixed time
and history.** Their numerical levels answer different questions, but changes
in peak location or correlation cannot differ merely because of this subtraction.

`tuning_metrics.csv` records map levels, modulation ranges, sampled peaks and
tied-peak counts for every neuron/history/view/time. `history_comparisons.csv`
records pairwise mean shifts, RMS differences, centered RMS differences and
Pearson correlations. Correlation ignores additive offsets and positive gains;
it is undefined for flat maps. Peaks alone are fragile near ties, and the
45° grid does not locate a continuous optimum or establish statistical significance.

`probe_responses.npz` stores full traces in history × probe-condition × time ×
neuron (or xy) order; sampled views use history × moving-condition × sampled-time
× neuron order. History order is stationary, forward, backward. Trial order is
in `probe_conditions.json`; paired moving/control indices and sample ticks are
saved in the NPZ. `neurons.csv` defines neuron order. The three `history_*.npz`
files preserve full history traces, PFNs, predictions, and targets.

## What this establishes

A changed tuning shape would demonstrate history-dependent responses in this
fixed model, not changing weights. Even the first probe update includes the
history-dependent recurrent drive; rectification can change which inputs are
expressed. Later probes also include the evolution of that drive.

This initial comparison does not separate home-vector memory from other effects
of movement history, nor establish that training for integration created the
effect. Follow-ups could compare a closed-loop history with stationary history
(equal final displacement), histories reaching equal displacement by different
routes, and initialized versus trained models or matched alternative tasks.
