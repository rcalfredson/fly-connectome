# Current travel direction and history in trained hDeltaB

18 November 2026. Frozen standard checkpoint selected at update 2,000 in
`cx-sugar-reliable-return-training-seed0-20261113`. This probe asks whether
the hDeltaB population in the current recurrent network still represents
world-centered travel direction, as in the earlier PFN-to-hDeltaB model.

## Direction probe

Replay the earlier grid of eight headings and eight world travel directions
at 1 mm/s. Different headings for the same world direction require different
body-centered translations. Start each trial from a zero state, allow 0.2 s
of heading-matched stationary input, then run the movement or the matched
stationary control. Average hDeltaB activity at 0.8–1.0 s and 2.8–3.0 s
after movement begins. The early window matches the archived feedforward
window relative to movement onset.

Use the archived transfer of the published population-vector convention.
The 19 hDeltaB body IDs match the reference inventory. Average cells that
share one of the 14 anatomical positions and take the circular population
vector with the original angles and offset. No new angle alignment, regression
or home-vector readout enters this measure. The cross-dataset anatomical map
is an assumption, and these are simulated rates. The earlier and current
models also differ in weights, time constants and recurrence, so the comparison
cannot isolate the effect of training or recurrence alone.

For an independent comparison of pattern structure, subtract each trial's
stationary control and compute centered cosine across the 19 cells. Compare
pairs with the same world direction and different headings against pairs with
different world directions and the same heading. These are descriptive averages
over a deterministic stimulus grid.

| Model/window | Mean absolute travel-angle error | Mean absolute heading-angle error | Same-world cosine | Different-world cosine | Cosine contrast |
| --- | ---: | ---: | ---: | ---: | ---: |
| Earlier feedforward | 15.81° | 78.93° | 0.823 | −0.093 | 0.916 |
| Recurrent, 0.8–1.0 s | 28.56° | 105.41° | 0.739 | 0.287 | 0.453 |
| Recurrent, 2.8–3.0 s | 21.89° | 95.34° | 0.775 | 0.173 | 0.601 |

Angles use raw activity. Cosines use the signed responses after stationary
subtraction. The recurrent population still distinguishes world travel
directions across headings. In particular, different directions at the same
heading evoke more similar patterns than in the earlier feedforward model.
The fixed readout also has larger angular errors in both recurrent windows.

## Common input after different histories

Reuse physically reached states after four seconds at rest, 4 mm forward,
4 mm rightward, and a closed square. The four histories end at the same
heading. From each state apply the same eight movement probes at that heading
and a stationary control. Clone the starting state for each probe. For each
world direction, compare the hDeltaB population angle with the angle following
the stationary history, then take the median absolute circular shift across
the eight directions.

| Earlier history | 0.8–1.0 s into common input | 2.8–3.0 s into common input |
| --- | ---: | ---: |
| Forward 4 mm | 81.58° | 10.29° |
| Rightward 4 mm | 82.08° | 32.20° |
| Closed square | 13.04° | 3.11° |

The latest hDeltaB activity carries a travel-direction signal and is sensitive
to the state inherited from previous movement. The history probe measures the
angle of the complete hDeltaB profile. It allows inherited activity offsets as
well as altered tuning to affect that angle. It does not isolate displacement
from other features of the prefix or demonstrate a displacement code in hDeltaB.
The fixed-speed grid tests direction rather than speed calibration.

## Presentation use

A single hDeltaB results slide can follow the biological evidence table. Its
takeaway is **hDeltaB still signals travel direction, and its activity also
reflects earlier movement**. Use the travel-angle error and the history-dependent
angle shift together. The vDeltaE–hDeltaG and hDeltaC proposals remain separate
questions. Give each a results slide if its experiment supports an interpretable
claim.

- [Slide figure](../results/cx-home-vector-hdeltab-direction-20261118-v2/hdeltab_direction_slide.png)
- [Detailed plots](../results/cx-home-vector-hdeltab-direction-20261118-v2/hdeltab_direction_probe.pdf)
- [Per-trial direction and phase data](../results/cx-home-vector-hdeltab-direction-20261118-v2/direction_trials.csv)
- [Per-trial history data](../results/cx-home-vector-hdeltab-direction-20261118-v2/history_trials.csv)

The bundle includes source hashes, fixed body-to-angle mapping, all replayed
hDeltaB rate traces, and the script. Reproduce in a fresh directory with
`python -m fly_connectome.home_vector.analysis.hdeltab_velocity --output OUTPUT`
in the home-vector environment.
