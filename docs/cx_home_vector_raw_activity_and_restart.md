# Raw activity leaders and movement after pauses

November 6, 2026. This addresses raw firing-model activity, separately from the
[decoder-contribution checks](cx_home_vector_activity_checks.md). The frozen
balanced hDeltaC model remains update **1975**; no training, fitting, weight or
decoder changes were made.

The [per-trajectory activity browser](../results/cx-home-vector-balanced-u1975-activity-by-trajectory-20261106/index.html)
is the renamed `cx-home-vector-balanced-u1975-raw-paths-restart-20261106` bundle
used to select mixed/curved-path examples. Its path arrays retain all neurons;
its trace panels show only the six largest mean rates plus the focus neuron.
The separate [extended movement/restart results](../results/cx-home-vector-balanced-u1975-raw-rate-movement-restart-20261106/index.html)
contain the varied-restart controls and norm statistics reported below. Renaming
the first directory does not rename or supersede the second experiment.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fly-connectome-mpl \
python -m fly_connectome.home_vector.analysis.activity_paths \
  --activity results/cx-home-vector-balanced-u1975-activity-20261106 \
  --source results/cx-home-vector-hdeltac-long-data-20261104 \
  --output results/cx-home-vector-balanced-u1975-raw-rate-movement-restart-NEW \
  --threads 1
```

Output must be fresh and separate from the verified source/activity bundles.
The tool preserves the checkpoint and metadata, records code/input/artifact
hashes and checks that inputs remain unchanged. Settings include `--angles`
(default eight), `--pauses` (5, 30, 60 s), `--resume-seconds` (8 s), `--seed`
(0), `--focus-body-id` (142911), and `--chunk-size` (8). CPU execution only.

## Which neurons have the highest raw rates?

Replay 192 straight 8 mm paths over eight headings, eight body-centered movement
directions and speeds 0.5, 1 and 2 mm/s. For each unit, measure raw mean/peak rates
and its fraction of active sampled time tied for highest rate, both globally and
within its own cell type. Ties share credit; silent samples and the zero-duration
final sample are excluded. No decoder coefficients enter these calculations.

Neuron 142911 is genuinely prominent in raw activity:

- It is the most frequent global leader in **92/192** conditions; neuron 17816
  leads in 52, and nine other units lead in the remaining 48.
- It is highest globally for more than half the active samples in **91/192**
  conditions and highest within vDeltaE for more than half in **138/192**.
- Its leading-time fraction ranges from zero to one: dominance is condition
  dependent, not universal.

Add four seeded three/four-segment world paths, each traversed by forward-only
versus mixed body movements (eight trials), plus four curved world paths, each
traversed with forward versus rightward movement (eight trials). Curves use
held headings updated every 0.1 s, angular velocities ±22.5/±45 degrees/s, speed
1 mm/s and duration 8 s. The paired world paths match, but body inputs differ.

142911 remains the most frequent global leader in 15 of these 16 trajectories,
but the leader can change within a trajectory. In `path_03`, a four-segment mixed
walk, the most frequent global leader within each segment is:

| Segment | Body ID | Cell type | Fraction of segment's active samples highest |
| --- | ---: | --- | ---: |
| 0–2 s | 32432 | hDeltaC | 51.1% |
| 2–4 s | 20231 | hDeltaC | 86.4% |
| 4–6 s | 142911 | vDeltaE | 96.2% |
| 6–8 s | 17816 | hDeltaB | 80.0% |

For the +22.5-degree/s curve, forward traversal is led by 17816 for 68.4% of
active time; matched rightward traversal is led by 142911 for 71.6%. Thus raw
leaders can depend on body motion even when the world path is matched. This
does not establish physiological tuning or a neuron's causal coding role.

## What happens when movement resumes?

Take each original forward/backward history's state at pause onset, replay the
same stationary heading-dependent PFN inputs for 5, 30 or 60 s, then branch from
the resulting state. Each branch lasts 8 s. Test eight headings, eight body
directions and three speeds: **192 moving branches per case**, or **1,152** over
the six history/pause cases. Also replay all 16 varied-path templates after each
pause, stretching each to the 8 s restart duration and quantizing control times
to the simulation grid. Include stationary controls at each fixed heading and
for each complete varying-heading sequence, to separate heading changes from
walking history. Each default case has 232 branches: 208 moving and 24 controls.
Targets continue from the pre-existing home vector rather than resetting home
at restart. Original pause-prefix predictions reproduce within 0.002 mm.

Keep these norms distinct, with the decoder's fixed row/null projections:

```text
absolute state norm          = ||r(t)||
absolute hidden norm         = ||Pnull r(t)||
hidden change from pause     = ||Pnull (r(t) - r(pause onset))||
```

The last reference is held fixed after restarting; it is not replaced by the
restart state. A change relative to an old pattern can shrink while absolute
activity increases.

**None of the 1,248 moving branches reduced the absolute state norm or absolute
hidden norm below its value at restart**, even transiently. This does not prove
that every possible action sequence would fail to reduce them. Absolute norm
is not globally monotonic: for example, in the forward history it drops from
81.69 at pause onset to 42.79 after five seconds, before movement resumes.

However, a norm can decrease from a peak reached after restart without going
below the restart value. The sustained straight restarts have no such absolute
norm declines, but the multi-segment/curved restarts do:

| History | Pause | Varied restarts with a full-norm decline from an earlier peak | Largest decline (model units) |
| --- | ---: | ---: | ---: |
| Backward | 5 s | 12/16 | 78.558 |
| Backward | 30 s | 9/16 | 54.668 |
| Backward | 60 s | 7/16 | 39.946 |
| Forward | 5 s | 12/16 | 77.626 |
| Forward | 30 s | 9/16 | 54.476 |
| Forward | 60 s | 7/16 | 40.209 |

For example, after the forward history and 5 s pause, the four-segment
`random-1 / 4-segment-forward` restart begins at norm **42.794**, reaches
**197.484** at 6 s, then falls to **119.858** at 7.2425 s after the final turn.
The matching heading-only history has a maximum decline of only **1.282**.
Thus movement sequences can reduce the norm again; the decrease does not erase
the growth since restart or show recovery to the state before the pause.

Changes relative to pause onset behave differently. Count a decrease only if
it exceeds `max(0.0001, 0.00001 * initial_norm)` model units:

The following counts refer specifically to the 192 constant-motion branches:

| History | Pause | Moving branches with any hidden-change decrease | Still below restart hidden-change value after 8 s |
| --- | ---: | ---: | ---: |
| Backward | 5 s | 148/192 | 0/192 |
| Backward | 30 s | 0/192 | 0/192 |
| Backward | 60 s | 0/192 | 0/192 |
| Forward | 5 s | 158/192 | 13/192 |
| Forward | 30 s | 91/192 | 0/192 |
| Forward | 60 s | 41/192 | 0/192 |

Example: after the forward history and 5 s pause, restart at heading 315 degrees,
body-forward speed 2 mm/s. At 0.275 s, the hidden change from pause onset falls
from **70.665 to 23.092**, while the absolute state norm rises from **42.794 to
88.434**. The stationary heading-315 control has hidden change **64.972** at
that time, so the stronger reduction involves movement as well as heading.
All these norms use model units, not millimeters.

This supports a distinction between restoring a movement-dependent hidden
pattern and reducing total activity. It does not show that the hidden state
has reset or that prediction improved: legitimate movement also changes the
correct home vector. Nor does it establish that state growth depends only on
elapsed time. Identifying the specific drives sustaining growth is a separate
mechanistic question.

## Artifacts

- `raw_rate_sweep.png/pdf`, `raw_sweep_conditions.csv`, `raw_sweep_neurons.csv`,
  `raw_sweep_protocols.json`: raw leaders and every neuron's rates over the grid.
- `raw_paths.json`, `raw_path_neurons.csv`, `path_XX.npz`,
  `path_XX_activity.png/pdf`: varied-path protocols, raw rates, inputs, vectors,
  leader timelines and selected raw-rate traces. Curves and matched variants
  are explicitly identified. Trace selection is six largest mean rates plus
  the focus unit; full arrays contain every neuron.
- `movement_after_pause.png/pdf`, `resume_cases.json`, `resume_conditions.csv`,
  `resume_XX.npz`, `resume_XX_protocols.json`: every branch's full-resolution
  norms, predicted/true vectors, reference/restart states and control protocol.
  Variant axes follow protocol/CSV order. Raw rates for all restart branches
  are not saved; they are reproducible from checkpoint, initial state and inputs.
- Restart plots include continued pause, heading-only controls, fixed movement
  examples, and explicitly selected illustrative branches with largest hidden
  reduction and full-norm drawdown. This is exploratory selection, not an
  independent validation case. CSV maximum-drawdown fields distinguish a drop
  from an earlier peak from dropping below the restart value.

All statements concern this frozen seed and finite probe set. Model rates are
uncalibrated; highest raw activity is distinct from direct readout contribution,
causal importance and physiological realism.
