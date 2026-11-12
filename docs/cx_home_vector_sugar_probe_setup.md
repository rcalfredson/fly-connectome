# Opt-in hDeltaG sugar-response experiment

This implements a first **experimental setup**, not a calibrated physiological
reset. It is separate from `HomeVectorNetwork.forward`; production behavior,
the checkpoint, the PFN encoder, and all archived references are unchanged.
There is no training, decoder refitting or parameter search.

## Log recap

- Two uninterrupted closed loops increased decoding and closure errors in the
  frozen network. Activity accumulation accompanied the error, but its causal
  role was not isolated.
- Janke's thesis motivates a sugar-response hypothesis in hDeltaG: sugar
  consumption reduces baseline and bump activity, followed by slower recovery.
  The sensory pathway and cellular mechanism are unresolved.
- We can model an effective response without claiming that mechanism. The
  physiological observations determine the desired response; existing model
  activity does not supply empirical reset targets.
- Recurrent consequences could go either way: other cells could drive hDeltaG
  upward again, or hDeltaG suppression could reduce activity elsewhere. Record
  all populations rather than prescribing either outcome.
- The first candidate adds relaxation toward a low reference in hDeltaG only.
  Its timing, strength, reference and event schedule are explicit assumptions.
- Sugar is an external event input, not an instruction derived from true home
  vector, loop count or decoded error. The setup uses experimenter-scheduled
  delivery; a floor-contact/consumption detector is not yet implemented.
- Compare sugar/no-sugar under identical movement, both with and without a
  matched pause. Keep physiological assessment separate from decoder outcomes.
- The checked-in example verifies numerical and experimental wiring only. It
  is not a fitted response, an accepted reset rule, or a decoding improvement.

## Candidate equation and event kernel

For hDeltaG only:

```text
I_i = bias_i + PFN_drive_i + recurrent_drive_i
T_i = max(0, I_i)
dr_i/dt = (T_i - r_i)/tau_i - g(t)*k_reset*(r_i - r_reference)
```

All other populations retain their original equation. They can change through
the existing recurrent connections; no direct reset or clamp is applied to them.
The reference is a uniform scalar in model activity units, supplied explicitly.
It is not estimated from the correct home vector or from decoding performance.
The implementation does not inject a separately prescribed hDeltaG waveform.

For an event at time t_event, the imposed effective gate is:

```text
s = max(0, t - t_event)
g_event(t) = (1 - exp(-s/tau_rise)) * exp(-s/tau_recovery)
g(t) = max_over_events(g_event(t))
```

This gate is causal and bounded between zero and one. It is an assumed effective
modulation, not a measured neural signal. Maximum aggregation across events is
also an assumption; no habituation, refractory circuit or additive sensitization
has been established. The runner currently requires at least 30 seconds between
scheduled deliveries, reflecting the thesis protocol rather than a demonstrated
physiological refractory period. Its example contains a single event.

`tau_recovery` is a decay time of the imposed gate. It does **not** prescribe
the recovery time of the actual network activity, which depends on ongoing
inputs and recurrence. Suppression can persist or rebound differently from the
gate. This short setup cannot validate minutes-long recovery.

With the illustrative 300-second gate constant, the gate remains near its peak
eight seconds after delivery. This does not mean the cells are clamped silent:
the original drive continues to act. A minutes-long mean-activity rise in the
thesis does not establish that such a long-lived gate exists. The calibration
comparison must also consider a brief suppressive event followed by recovery
through network dynamics. The current example contains only one sugar event,
not a delivery at every circular return.

## Stable numerical integration

At each time step, recompute the original recurrent target T_i, then hold that
target and gate fixed over the step. For targeted cells:

```text
decay_i = 1/tau_i + g*k_reset
equilibrium_i = (T_i/tau_i + g*k_reset*r_reference) / decay_i
r_next_i = r_i + (1-exp(-dt*decay_i))*(equilibrium_i-r_i)
```

This is the exact exponential update for the frozen-target step, not an exact
solution of the continuously changing recurrent system. It avoids a new
forward-Euler instability at strong relaxation rates. Nonnegative references
and targets keep the update nonnegative; it never directly assigns activity
to zero. When the gate or relaxation strength is zero, use the original update
operation sequence exactly. The frozen native time constants are not modified.

## Explicit engineering configuration

`docs/data/hdeltag_sugar_probe_example.json` contains:

| Parameter | Example | Status |
| --- | ---: | --- |
| Gate rise constant | 1 s | Engineering choice; unresolved by the coarse figure |
| Gate decay constant | 300 s | Engineering choice; not a fitted activity-recovery constant |
| Extra relaxation strength | 1 per second | Engineering choice |
| Reference activity | 0 model units | Engineering choice; does not force zero firing |
| Sugar delivery | 8 s | Experimenter-scheduled single delivery |
| Matched pause | 8–9 s | Engineering control |

None is selected to improve decoding or claimed to fit the thesis. The coarse
40–60-second residual ratio is not used as an instantaneous reset multiplier.
The runner intentionally accepts only the explicit
`uncalibrated_engineering_example` status at this stage, preventing this setup
from being labeled physiologically validated without extending its assessment.

## Four-arm comparison and immutable reference

Repeat the original closed-circle movement of paths 8, 9, 14 and 15 twice:

1. Continuous movement, no sugar.
2. Continuous movement, sugar.
3. One-second pause between loops, no sugar.
4. The identical pause/movement schedule, sugar.

Every trial starts once from zero, with no state reset between loops. Pause
trials last 17 seconds; continuous trials last 16. Compare sugar effects within
matched schedules, not by treating different elapsed times as matched movement
phases. Scheduled sugar is identical across those schedules. Targets retain the
original origin, which is where the first loop closes; this does not implement
changing the reference on discovering a new food patch elsewhere.

The network rollout receives only PFNs and a scalar gate; it receives no true
position, target, or loop counter. The task layer independently integrates
targets for evaluation outside the rollout. The configured event is delivered
by the experimenter at a known time; it is not evidence that the network can
recognize home or food without another sensory input.

Verification requires:

- Gate-disabled trial rates/predictions are bitwise identical to the native
  forward pass, including its ordinary recurrence.
- Continuous controls reproduce the original two-loop archive.
- Sugar/control partners have exactly equal controls, PFNs and targets.
- Activities are identical up to and including the first event state.
- Every frozen state-dict tensor remains unchanged, and all source hashes match.
- Targets match an independent geometric integral; predictions reconstruct from
  actual activities with the same decoder and intercept.

Saved NPZ arrays retain all neuron activities and gates, not just population
means. CSVs give population measures and separately labeled decoder outcomes.
Cross-cell RMS contrast is not called bump amplitude without a spatial phase
map. Model units are not converted to Hz or fluorescence by this experiment.

## Engineering run and next decision

The completed wiring check is
`results/cx-hdeltag-sugar-setup-smoke-20261112/index.html`. It ran all 16 trials
and passed the controls above. The illustrative response changes hDeltaG and
other populations through recurrence; decoding is not improved by this example.
These outcomes do not select or reject a physiological mechanism, because the
example was not calibrated to physiological measurements. No parameter was
changed after viewing decoder outcomes.

Before a biological comparison: constrain the drop, residual signal, and
recovery against matched observables; resolve or bracket the fast timing; and
use a fixed, justified observation mapping. A population mean and a pooled
single-cell spike curve are not automatically the same observable. A calcium
comparison additionally needs an observation model. Longer response protocols
and a verified columnar map are needed for recovery and bump validation.
The earlier measurement assessment remains separate and should be read first.

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fly-connectome-mpl \
python -m fly_connectome.home_vector.analysis.sugar_probe \
  --activity results/cx-home-vector-balanced-u1975-activity-by-trajectory-20261106 \
  --source results/cx-home-vector-hdeltac-long-data-20261104 \
  --baseline results/cx-home-vector-balanced-u1975-repeated-loops-20261112 \
  --config docs/data/hdeltag_sugar_probe_example.json \
  --output results/cx-hdeltag-sugar-setup-reproduction

python -m pytest -q tests/test_hdeltag_sugar_probe.py
```

Use a fresh output directory. Requires the project dependencies including Torch.
The manifest preserves configuration, checkpoint and input hashes; both runtime
scripts are copied into the run. This experiment remains opt-in and untrained.
