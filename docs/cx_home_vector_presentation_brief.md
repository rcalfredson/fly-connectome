# Presentation brief: learned home-vector tracking in a partial fly CX circuit

Presentation outline and evidence inventory.

The talk was split into two parts. Part I introduces the home vector, central
complex anatomy, the modeled circuit, and three tests that motivate the
results talk. The [Part I speaker guide](cx_home_vector_part1_speaker_guide.md)
follows its seven-page slide order, including the title card. Part II will
recap the setup and present the results. The slide numbers below refer to the
combined working draft and serve as an index to its results and figures.

## Suggested central question and answer

**Question:** Can a small, anatomically constrained fly circuit learn a
movement-history representation useful for home-vector tracking, and does its
learned activity reproduce the biological computations of its cell types?

**Current answer:** The model supports approximate home-vector decoding and
history-dependent responses. Training on repeated returns yields nearly repeating
neural activity without imposed sugar suppression. But the current standard
solution strongly attenuates displacement offsets under common moving inputs,
so it is not a demonstrated persistent, exact home-vector integrator. Its
anatomy and input prescription have biological foundations; its learned
cell-type computations and effective synaptic dynamics remain hypotheses.
A fading memory can be computationally inadequate for exact home retention
while qualitatively resembling a biologically useful leaky integrator. These
are different assessments.

## Slide 1 working draft

**Title** — What does it mean to keep track of home?

**Layout** — Place the route diagram on the right and two short text blocks
on the left. The dashed route shows the walk from home to the current
position. The solid arrow points from the current position back to the fixed
home location.

> The home vector gives the direction and straight-line distance from the
> fly’s current position to its starting point.
>
> Can a network constrained by fly central-complex anatomy learn to track
> this vector from movement signals?

**Spoken transition** — “As the fly moves, the arrow changes. To keep it
up to date from movement alone, the network has to keep track of the
displacement it has accumulated. We’ll ask whether this circuit can learn
that task, and how closely its activity resembles what the real cell types
do. First, here’s what the model receives and what it learns.”

Figure exports in [PNG](../results/cx-home-vector-slide01-20261118/cx_home_vector_definition_slide01.png),
[editable SVG](../results/cx-home-vector-slide01-20261118/cx_home_vector_definition_slide01.svg),
and [vector PDF](../results/cx-home-vector-slide01-20261118/cx_home_vector_definition_slide01.pdf).

## The current setup: one slide

| Element | Specification | Evidence status |
| --- | --- | --- |
| Inputs | 40 PFNd + 20 PFNv prescribed activities, from heading and forward/rightward velocity | Published simplified encoding prescription transferred to selected MaleCNS cells; not measured activity for these 60 cells |
| Dynamic network | 19 hDeltaB, 18 vDeltaE, 8 hDeltaG, 20 hDeltaC; 65 scalar rate states | Named anatomical populations; simplified single-state neurons |
| Edges | 1,626 selected directed body-pair edges: 693 input, 933 internal | Anatomical support/contact counts; incomplete circuit boundary |
| Dynamics | Rectified drive plus relaxation, recurrent coupling; 0.0025-s simulation step | Our effective rate model; activity in arbitrary units |
| Learned quantities | Per-edge signed coefficients, type-specific biases/time constants, affine decoder | Supervised task optimization, not fitted physiological efficacies |
| Target | World-frame vector from current position to original home | Externally integrated geometric supervision; absent from neural inputs |
| Readout | Two affine outputs from current 65-neuron activity | No decoder memory; no identified biological downstream readout |
| Sugar variant | Extra hDeltaG relaxation with 1-s gate rise, 300-s recovery, rate 1/s, reference zero | Qualitative effective approximation motivated by thesis measurements |

```text
heading + body-centered velocity
             ↓
     prescribed PFNd/PFNv
             ↓
65 recurrent rates on selected anatomical edges
             ↓
  instantaneous affine x/y readout
```

The standard arm receives no sugar signal. In the sugar arm, every eligible
home-patch reentry triggers delivery; geometry generates this external sensory
event but is not provided as position or a home flag. The anchor never changes.
No state is zeroed at a return. Feedback into prescribed PFNs, much of the CX,
compartment-specific processing, slow intrinsic states and many sensory inputs
are outside the modeled boundary. PFN-to-hDeltaG connections were unqueried,
not established absent.

## How the setup evolved: one slide

Use milestones rather than presenting all historical numerical scores as if
these were a single controlled series:

1. Reproduced the published PFN-to-hDeltaB coordinate-transform prescription
   on an anatomical subnetwork; this is the movement-signal foundation.
2. Added recurrent hDeltaB/vDeltaE/hDeltaG, then hDeltaC, and trained a home-vector
   readout with the dynamics. This asks a capacity question beyond the original
   feedforward velocity model.
3. Examined matched world paths with forward/rightward body movement and
   decoder row/null components. Equal world displacement need not produce equal
   neural activity; performance imbalance alone did not diagnose the encoder.
4. Investigated prescribed-input ambiguities and physiology. Retained the
   production PFN rule rather than adopting a suppression change that lacked
   adequate empirical support. No extra prescribed sensory population was added.
5. Repeated circles exposed accumulation in the older checkpoint. Sugar-triggered
   hDeltaG suppression reduced drift, motivating decoder refits and paired
   recurrent training with more diverse returns.
6. Both currently trained arms avoid the old severe circle drift. The new
   standard solution becomes repeatable, but the latest common-input probes
   reveal attenuation of displacement history that should persist.

Network size, training distribution, normalization, checkpoint and evaluation
conditions changed across milestones. Earlier 45-cell metrics, local 4-second
errors, decoder-only refits and current 65-cell test RMSEs are not interchangeable.
See the individual reports before placing any old number on a slide.

## Slide 4 working draft

**Title** — The early standard model tracks a short circle

**Layout** — Put the route diagram on the left, with a direction-of-travel
arrow and a shared start/return marker. On the right, stack home-vector
`x` and `y` traces on the same 0–8 s time axis, with a solid prediction and
dashed target. Use one legend and mark the return at 8 s on both plots.
The example is path 8, a counterclockwise circle traversed while walking
forward. Draw the route relative to its start from the negative of the
archived target, which points back to the start.

**Slide assets** — [Route PNG](../results/cx-home-vector-slide04-20261118/cx_home_vector_early_standard_path08_route.png)
and [component traces PNG](../results/cx-home-vector-slide04-20261118/cx_home_vector_early_standard_path08_first_loop_components.png).
Both are also available as SVG and PDF in the same directory. The route
uses the repository's positive-down y convention. The component plots
show the archived values with a common vertical scale.

**Training recap** — Fit the recurrent circuit and readout together to
predict the home vector along trajectories. Validation selected update 1975.
The later experiments keep the supervised prediction task while changing
the training examples and loss weighting.

**Figure callout** — 0.149 mm vector RMSE over the circle.

**Held-out context** — On the original test subset, vector RMSE was
0.220 mm. On the added long-path test subset, it was 4.396 mm. Keep those
subset scores together and apart from the example-circle callout.

**Spoken transition** — “The readout follows both components over one
circle, although it still misses home by about a quarter of a millimeter
at the return. What happens if we keep the model running and give it the
same movement a second time?”

Data and metrics come from the
[early standard-model assessment](cx_home_vector_balanced_u2000_assessment.md)
and the [archived path 8 replay](../results/cx-home-vector-balanced-u1975-repeated-loops-20261112/path_08.npz).
The first-loop RMSE covers samples before the return. The component plots
should include the return sample at 8 s, where the vector error is 0.245 mm.

## Slide 5 working draft

**Title** — Repeating the same circle exposes drift and activity buildup

Continue path 8 through a second identical 8 s loop, carrying the neural
state across the first return. Show decoded vector error above hDeltaG
mean activity and all eight individual hDeltaG traces, on a common time
axis with markers at 8 and 16 s. On path 8, vector RMSE rose from
0.149 to 1.044 mm and return error rose from 0.245 to 1.544 mm. Across the
four circles, both errors and the endpoint full-state activity norm rose
on loop 2. The co-occurrence motivates a perturbation without isolating
which activity changes contribute to the error.

**Slide asset** — [Path 8 error and hDeltaG activity](../results/cx-home-vector-slide05-20261118/cx_home_vector_early_standard_path08_two_loop_drift_activity.png).
SVG and PDF exports are in the same directory. The hDeltaG population
mean at the two returns was 10.54 and 21.14 model units.

Use the [repeated-loop report](cx_home_vector_repeated_loops.md) and
[archived figure](../results/cx-home-vector-balanced-u1975-repeated-loops-20261112/paths_08_09.png).
Bridge to the sugar comparison by asking whether sugar-triggered hDeltaG
suppression could limit drift and whether a standard circuit could learn
to do so with different training examples.

## Slides 6–7: the sugar rule and trained comparison

Current seed-0 reliable-return training: 2,000 paired updates; both selected
update 2,000 by normalized validation MSE. The objective scales trajectory errors
by typical squared home-vector magnitude within each training group. This
prevents large-displacement straight walks from dominating merely by scale.
Both validation curves were still improving near the budget boundary.

| Measurement | Standard | Sugar |
| --- | ---: | ---: |
| Overall normalized test MSE | 0.3740 | 0.4249 |
| Overall raw test RMSE | 10.537 mm | 9.601 mm |
| Diverse 4/8-return RMSE | 1.704 mm | 2.086 mm |
| Diverse 16-return RMSE | 1.799 mm | 2.293 mm |
| Four familiar 16-circle aggregate RMSE | 1.478 mm | 1.397 mm |

Sugar does not win uniformly. It is better on long straight walks (which have
no deliveries, so this is a learned-parameter difference), and on familiar
circles 8/9; standard is better on circles 14/15 and aggregate diverse-return
tracking. Disabling the gate in the sugar-trained model increases diverse
16-return RMSE to 39.338 mm. That is dependence in that trained solution, not
proof that an independently trained network needs the gate.

Use [the complete comparison](cx_home_vector_reliable_evaluation_results.md),
including zero baselines and per-stratum metrics. Do not cherry-pick the raw or
normalized score according to which model wins.

## The latest mechanistic finding: two slides

**Repeatability slide assets** — [Successive-lap full-state differences](../results/cx-home-vector-loop-repeatability-slides-20261118/cx_home_vector_standard_path09_successive_lap_state_difference.png)
and [individual hDeltaG activity at laps 1, 8 and 16](../results/cx-home-vector-loop-repeatability-slides-20261118/cx_home_vector_standard_path09_hdeltag_lap_activity.png).
Both use path 9 (counterclockwise, rightward traversal) in the retrained
standard network. The line plot compares all 65 neurons at matching times
within each pair of successive laps. The heatmaps share an activity scale
and a fixed row order for the eight hDeltaG neurons. SVG and PDF exports are
in the same directory.

The standard network's phase-matched full-state differences decrease by
approximately 1,300–33,000-fold from loops 2-versus-1 to 16-versus-15. This is
repeatability of actual neurons, not only cancellation in the decoder. It is a
finite periodically driven replay, not a stability theorem.

The remaining loop-16 squared error is 90.8–99.3% mean vector offset. Timing,
amplitude and waveform errors remain, but are smaller. A post-hoc descriptive
Fourier decomposition does not correct the predictions or train a new decoder.

**Persistent-error slide assets** — [Path 9 home-vector x on laps 15 and 16](../results/cx-home-vector-persistent-error-slides-20261118/cx_home_vector_standard_path09_late_lap_home_x.png)
and [mean-offset share of lap 16 squared error](../results/cx-home-vector-persistent-error-slides-20261118/cx_home_vector_standard_lap16_mean_offset_share.png).
The left panel shows one component from the same frozen replay as the
repeatability slide. The offset shares use both components of the vector.
The mean offset is the prediction minus the correct vector, averaged over
the lap. SVG and PDF exports are in the same directory.

The stronger retention probe starts from six equal-duration physically reached
states and supplies identical continuations. For rest versus a 4-mm forward
prefix, decoded separation starts at 3.424 mm and falls to 0.0200 mm after
128 seconds of clockwise/rightward circular input, although the correct
separation remains 4 mm. The full-state distance falls from 43.413 to 0.0400
model units. All four circles strongly attenuate these meaningful offsets.

A rest-versus-closed-square pair has the same correct home vector, and its
history difference also shrinks. Thus the tested dynamics remove both irrelevant
history and relevant displacement. Pause responses are more heterogeneous:
some information survives 32 seconds, with direction-dependent distortion.
This is not simply a demonstration that only current velocity is represented.

**Defensible mechanism-level statement:** this selected solution attenuates
initial-state differences under common movement, including differences needed
to retain the original home anchor. That supports forgetting within the learned
dynamics. It does not establish a specific causal connection/motif, a universal
leak constant, the cause of every baseline circle offset, or an architecture-wide
inability to integrate. Another decoder could be tested separately; no such
readout fitting was done here.

Full methods, numerical checks and plots:
[displacement retention report](cx_home_vector_displacement_retention.md).

## What are the biological jobs, and what have we actually reproduced?

The primary papers distinguish current travel signals from temporal integration.
The following is a bounded source-based assessment, not a comprehensive claim
that these cells have no other functions.

| Cell type | Primary physiological evidence | What our model does / what remains untested |
| --- | --- | --- |
| PFNd/PFNv | Joint tuning to heading and body-centered translation during walking; PFNd perturbation affects path integration. [Lu et al.](https://www.nature.com/articles/s41586-021-04191-x) | We prescribe a simplified version of that encoding. It is an input assumption, not a learned discovery. Exact ambiguities in the prescription are not a demonstration that real PFNs share them. |
| hDeltaB | Allocentric translation/travel direction and a vector coordinate transformation. [Lu et al.](https://www.nature.com/articles/s41586-021-04191-x), [Lyu et al.](https://www.nature.com/articles/s41586-021-04067-0) | The trained recurrent population still signals world travel direction. Its population angle also depends on earlier movement. See slide 13. |
| vDeltaE | Imaging and perturbation evidence for involvement in hDeltaG integration. The thesis reports rapid post-sugar rebound in hDeltaG under silencing. [Janke thesis](https://digitalcommons.rockefeller.edu/student_theses_and_dissertations/808/) | Silencing modeled vDeltaE raises pre-sugar hDeltaG activity. Post-sugar activity remains suppressed in the tested replay. See slide 14. |
| hDeltaG | Sugar-associated suppression; slow mean-activity recovery and a faster bump-amplitude signal associated with recent walked distance, interpreted using a leaky-integration model. [Janke thesis](https://digitalcommons.rockefeller.edu/student_theses_and_dissertations/808/) | Our gate reproduces a qualitative suppression idea. Model memory attenuation is qualitatively compatible with leaky integration, but neither its timescale nor population bump phase/amplitude has been matched to those measurements. |
| hDeltaC | Odor-gated wind-direction tuning and perturbation evidence for upwind olfactory navigation. [Matheson et al.](https://www.nature.com/articles/s41467-022-32247-7) | Brief silencing weakens early displacement retention in the tested replays, partly through changes in other populations. See slide 15. The measured odor and wind response remains outside this task. |

The Lu/Lyu movement-coordinate evidence does not itself demonstrate a persistent
home-vector memory in hDeltaB. Janke's source is a thesis, not a peer-reviewed
journal article; repository year is 2025, while the supplied PDF title page is
September 2024. Its experimental claims and formal model are distinct from our
added sugar-gate equation. Matheson's experimental hDeltaC tuning is distinct
from the paper's simplified steering model.

Recent PFNa physiology also illustrates omitted computations, not justification
for adding a generic extra input: [Ishida et al., Cell 2026](https://doi.org/10.1016/j.cell.2025.11.040)
describe calcium-spike-mediated vector inversion in a different PFN class. That
is outside the current PFNd/PFNv-only boundary and is not a property we modeled.

## Slide 12 working draft

**Title** — How closely does the model resemble the fly circuit?

Open with one line beneath the title. Anatomy constrains the connections,
PFN tuning is prescribed, and the recurrent circuit is trained to predict
the home vector.

Use an editable table to lead into the three population-specific slides.

| Population | Evidence from the fly | Question for the model |
| --- | --- | --- |
| hDeltaB | Signals travel direction in world coordinates | Does this direction signal survive recurrent training? |
| vDeltaE | Silencing changes hDeltaG recovery after sugar | Does the sugar-response model reproduce that shift in recovery? |
| hDeltaC | Responds to wind direction in an odor-guided task | How does it contribute to retaining prior displacement in our circuit? |

The first two probes ask how the model compares with physiological evidence.
The hDeltaC probe asks what role this population acquired in the home-vector
task. The model receives no odor or wind input. The PFN prescription and
hDeltaG suppression rule are already covered on slides 2 and 6.

Reference footer — Lu et al. (2022) and Lyu et al. (2022) for hDeltaB, Janke's thesis for
vDeltaE, and Matheson et al. (2022) for hDeltaC. See the evidence table above
for source links and the current assessment by population. The earlier
five-row matrix exports in `results/cx-home-vector-slide09-biology-20261117`
predate the three probes.

Spoken transition — “The circuit learned activity that can be read out as
a home vector. We can now ask which aspects of that activity resemble what
we see in flies. First, does hDeltaB still tell us which way the fly is
traveling?”

## Current parameter flexibility: an essential qualification

A fresh checkpoint audit, rather than reuse of old 45-cell numbers, finds:

| Property | Standard | Sugar |
| --- | ---: | ---: |
| Internal coefficients changing sign from seeded initialization | 378 / 933 | 389 / 933 |
| Internal effective-weight norm / initial norm | 6.034 | 6.597 |
| Dynamic source neurons with both positive and negative effective outgoing weights | 65 / 65 | 65 / 65 |

These are model parameters relative to an arbitrary seeded initialization,
not biological synaptic plasticity measurements. The model does not calibrate
signs to transmitter/receptor or compartment-specific physiology. Mixed signs
alone do not prove physiological impossibility, but they preclude treating the
learned couplings as verified biological excitation/inhibition. Most intrinsic
time constants reach their allowed 0.5-second maximum; these are not estimates
of effective network memory duration.

Anatomy restricts the edge support and influences scales/priors. It does not
fix the computation. No matched retrained shuffled-connectome ensemble has
established a special functional advantage of the biological topology here.
No latest-checkpoint population-specific intervention establishes which cell
class is necessary for its memory.

## Combined draft slide sequence and figure selection

| Slide | Question / message | Evidence or visual |
| --- | --- | --- |
| 1 | What does it mean to keep track of home? | Define the home vector with a route schematic and introduce the guiding question |
| 2 | What enters the model and what learns? | Current setup table/pipeline; anatomy versus learned coefficients |
| 3 | Three tests of the learned home vector | Preview early performance, ways to limit drift, and displacement memory |
| 4 | How well did the early standard model track home? | Path 8 first-loop component traces, training recap and held-out RMSE |
| 5 | What happens when the early standard model repeats the same circle? | Path 8 two-loop error and activity traces, with results across all four circles |
| 6 | What does a sugar event do in the model? | Matched path 8 hDeltaG replay, one scheduled sugar delivery versus control |
| 7 | Does sugar improve tracking after retraining? | Whole-route RMSE for familiar circles and held-out diverse returns |
| 8 | What became stable? | [Phase-aligned activity](../results/cx-standard-loop-phases-20261116/path_09_phase_comparison.png) and [all-neuron heatmaps](../results/cx-standard-loop-phases-20261116/path_09_phase_activity.png) |
| 9 | What is still wrong? | [Final-loop error budget](../results/cx-standard-displacement-retention-20261117/circle_error_budget.png); offsets dominate |
| 10 | Does meaningful history survive? | [Focused two-panel retention figure](../results/cx-home-vector-slide07-retention-20261117-v2/cx_home_vector_standard_retention_cw_rightward_16loops_slide07.png); required offsets versus merging states |
| 11 | Is forgetting universal? | [Pause control](../results/cx-standard-displacement-retention-20261117/pause_32s_retention.png); history survives imperfectly and differently by direction |
| 12 | How closely does the model resemble the fly circuit? | Three cell-population questions grounded in the experimental evidence |
| 13 | Does hDeltaB still signal world travel direction? | [Direction and history probes](../results/cx-home-vector-hdeltab-direction-20261118-v2/hdeltab_direction_slide.png) with the same fixed population-angle readout |
| 14 | Does silencing vDeltaE accelerate hDeltaG recovery after sugar? | [hDeltaG recovery and matched controls](../results/cx-home-vector-vdeltae-silencing-20261118-v3/vdeltae_hdeltag_recovery.png) |
| 15 | How does hDeltaC support retained displacement? | [Forward-history circle probe](../results/cx-home-vector-hdeltac-slide12-20261118/cx_home_vector_hdeltac_forward_circles_slide12.png) and the breakdown at clamp release |
| 16 | What has been established, and what would discriminate the remaining explanations? | Conclusions and targeted next measurements below |

Slide 10's focused figure uses rest versus the forward-4mm prefix under path 15
(clockwise circles, rightward traversal). It shows decoded vector separation
and the Euclidean distance between complete 65-neuron states, not activity
projections. The time axis begins after the four-second prefixes. The original
home anchor is retained, so the required vector separation stays 4 mm. Both
neural states are reached by movement; neither is manually edited. Initial and
final decoded separations are 3.424 and 0.020 mm; state distances are 43.413 and
0.040 model units. This is a selected illustrative comparison, not a pooled
score or an assertion that every continuation forgets at the same rate.

The same output folder contains editable SVG and vector PDF exports, exact
plotted values, provenance and a standalone caption. Its predecessor without
`-v2` is a layout draft. The full multi-pair plots remain in the original
retention experiment. Reproduce the focused figure in a fresh directory:

```bash
MPLCONFIGDIR=/tmp/fly-connectome-mpl OPENBLAS_NUM_THREADS=1 \
/home/tracking/miniconda3/envs/py3.13-torch/bin/python \
-m fly_connectome.home_vector.analysis.slide_retention \
--output results/cx-home-vector-slide07-retention-reproduction
```

For slide layouts that manage explanatory text separately, minimal independent
panels are available in `results/cx-home-vector-slide07-retention-basic-20261117-v2`:
files ending `_decoded_basic` show home-vector separation, while `_state_basic`
shows Euclidean distance between the complete activity vectors. Each has PNG,
editable SVG and PDF versions with curves, axes and ticks only. The dashed
horizontal line in the decoded panel is the required 4-mm separation; add its
legend in the slide. The state panel's label is shortened to "State distance"
for layout. The original annotated combined figure remains available.

## Slide 11 working draft: stationary continuation

**Title:** Displacement memory persists imperfectly during a pause.

Keep slide 10's same frozen standard network and the same four-second histories:
A is stationary; B walks forward 4 mm. Replace the common circular continuation
with 32 seconds of stationary input at heading 45 degrees. Home remains at the
original origin, so the required vector separation stays 4 mm. A pause means
zero translational velocity, not zero PFN activity: the prescribed heading-
dependent baseline remains, and neural activity continues to evolve.

Use the same two-panel layout and colors as slide 10. Minimal PNG/SVG/PDF panels
are in `results/cx-home-vector-slide08-pause-basic-20261117`, with filenames
ending `_decoded_basic` and `_state_basic`. No title, legend, setup text or
endpoint annotations are embedded. The blue-panel dashed line is 4 mm.

- Decoded separation: **3.42 → 2.13 mm**, versus required **4 mm**.
- Complete neural-state distance: **43.4 → 4.22 model units**.
- Takeaway: **Earlier histories remain distinguishable during the pause, but
  the required 4-mm offset is not maintained.**

The response is nonmonotonic: decoded separation reaches about 1.57 mm at
17.48 seconds, then rises. Do not describe it as a fitted exponential leak,
complete forgetting, or evidence of indefinite retention. This is one specified
pair; the rightward-prefix pause response differs and includes substantial
directional distortion.

Speaker-note comparison: slide 10 covers 128 seconds, while slide 11 covers 32.
At a matched 32 seconds, the clockwise/rightward movement on slide 10 produces
0.58-mm separation versus 2.13 mm during the pause for this same prefix pair.
This supports input-dependent retention in this example without confounding
input condition with duration. The separation norm alone does not establish
correct direction; the archived pause pair also has a signed target gain of
0.531 and a perpendicular separation of 0.154 mm at 32 seconds.

```bash
MPLCONFIGDIR=/tmp/fly-connectome-mpl OPENBLAS_NUM_THREADS=1 \
/home/tracking/miniconda3/envs/py3.13-torch/bin/python \
-m fly_connectome.home_vector.analysis.slide_pause \
--output results/cx-home-vector-slide08-pause-reproduction
```

## Slide 13 working draft

**hDeltaB signals travel direction and reflects earlier movement**

Put the two-panel figure across the middle of the slide. Add a short question
above each panel and one sentence below the figure. The title carries the
takeaway, so the bottom sentence can explain the comparison rather than repeat it.

| Panel | Heading above the plot | Setup line below it |
| --- | --- | --- |
| Left | How well does hDeltaB track world travel direction? | 64 combinations of heading and travel direction at 1 mm/s. |
| Right | Does earlier movement change the hDeltaB angle? | The same movement probes follow four seconds of different histories. |

**Caption beneath both panels** — Both panels use the same fixed readout of
the hDeltaB population angle.

**Speaker notes.** On the left, the same fixed angle readout estimates travel
direction from hDeltaB population activity in both models. Mean absolute error
is 15.8° in the earlier feedforward model and 28.6° in the recurrent model during
the matching 0.8–1.0 s window. In the recurrent model it falls to 21.9° at
2.8–3.0 s. The models differ in direct PFN weights, time constants and recurrent
connections, so the comparison does not isolate the effect of any one change.

On the right, every movement direction is tested after a four-second stationary
history and after each of the three movement histories. Zero shift is the
expectation for a signal that depends only on current travel direction.
The bars show the median absolute difference from the stationary-history
reference across eight directions. Following the 4-mm forward and rightward
histories, the early shifts are about 82° and the late shifts are about 10°
and 32°, respectively. The closed square ends at its starting position and
produces smaller shifts. These probes establish sensitivity to earlier movement
and leave open what aspect of that history hDeltaB represents.

Keep the six-panel diagnostic in backup for questions about individual trials,
angle wrapping or activity-pattern similarity. The
[hDeltaB methods and results](cx_home_vector_hdeltab_direction.md) give the
controls and interpretation.

## Slide 14 working draft

**Silencing vDeltaE raises hDeltaG activity, but recovery stays slow**

Replay the same circular movement in the frozen sugar-trained model with
vDeltaE intact or silenced throughout. Deliver one sugar stimulus after the
first loop and include a matched no-sugar run for each condition.

Show the first two panels of the [recovery figure](../results/cx-home-vector-vdeltae-silencing-20261118-v3/vdeltae_hdeltag_recovery.png).
The first shows mean hDeltaG activity around sugar onset. Dashed lines indicate
the no-sugar controls. The second shows mean activity in the two sugar runs
over each subsequent eight-second circle.

- Mean hDeltaG activity in the last second before sugar was 1.87 model units
  with vDeltaE intact and 4.54 with vDeltaE silenced.
- After sugar, both runs stayed suppressed relative to their respective
  no-sugar controls.
- The rapid post-sugar rebound reported in the fly experiment did not occur
  in this replay. [Janke thesis, Fig. 3.7d](https://digitalcommons.rockefeller.edu/student_theses_and_dissertations/808/).

The pre-sugar difference explains why higher activity after sugar alone is
insufficient to demonstrate faster recovery. The gate stays active across
the replay, and its contribution to the mismatch remains open. See the
[methods and results](cx_home_vector_vdeltae_silencing.md).

## Slide 15 working draft

**hDeltaC supports early displacement retention through the recurrent circuit**

Compare a stationary history with a 4-mm forward history in the frozen standard
network. Both are followed by identical circular movement. Replay with hDeltaC
intact or silenced for the first two seconds of the shared input.

Show the [standalone forward-history circle panel](../results/cx-home-vector-hdeltac-slide12-20261118/cx_home_vector_hdeltac_forward_circles_slide12.png).
Read the y-axis as the part of the two histories' decoded difference that
points along the correct 4-mm offset. The orange line is hDeltaC's own decoder
term in the intact replay. At the two-second silencing boundary, the decoded
separation along the offset falls from 2.52 to 1.84 mm. Removing hDeltaC's own
term accounts for 0.05 mm of the drop. Changes in the contributions of the other
cells account for 0.63 mm.

Use the [four-panel figure](../results/cx-home-vector-hdeltac-retention-20261118-v2/hdeltac_retention.png)
in backup. Early separation weakens in all four comparisons at two seconds.
Later effects depend on the prior history and the shared input. The
[methods and results](cx_home_vector_hdeltac_retention.md) describe the full
vector errors and the limits of the intervention.

## Slide 16 working draft

**Title** — The circuit learned to track home approximately

**What we found**

- The retrained standard network limited drift on repeated circles.
- During identical later movement, its neural activity and readout
  increasingly lost a displacement difference that should have persisted.
- hDeltaB retained a travel-direction signal, vDeltaE silencing failed to
  reproduce the rapid post-sugar rebound, and hDeltaC supported early
  displacement retention through the modeled recurrent circuit.

**What remains open**

1. **Model** — Can this circuit keep activity bounded while preserving
   displacement accurately over long paths?
2. **Fly** — When does the fly's estimate return toward zero through path
   integration, and when do sensory cues correct it or set a new home anchor?

In the narration, distinguish cancellation of displacement on a closed path
from cue-driven correction at a known site and re-anchoring to a new site.
The standard model's return errors remained nonzero and the sugar variant
implemented hDeltaG suppression. The two interventions needed to test the
open questions are different. For the model, test bounded activity and
retention of nonzero displacement offsets under matched training and
held-out movement. For the fly, vary the cue independently of physical
return to the starting point and assess the inferred home vector. If cues
reliably correct errors, the biological question about stability without
cues can focus on how long displacement needs to be retained between cues.

## What is worth mining next, if needed for the biological interpretation?

1. A verified mapping from model neurons/compartments to physical FB columns,
   followed by fixed phase/amplitude/mean measurements under matched movement.
   Neuron-index heatmaps and decoder projections are not evidence of the
   anatomical population bump by themselves. Do not choose a different phase
   alignment for each trajectory to improve agreement.
2. The first [vDeltaE silencing probe](cx_home_vector_vdeltae_silencing.md)
   raised hDeltaG activity before sugar but did not reproduce the rapid
   post-sugar rise reported in the fly experiment. Slide 14 shows this
   comparison. A quantitative comparison requires an explicit link between
   model rates and the recorded observable.
3. The [hDeltaC retention probe](cx_home_vector_hdeltac_retention.md) found
   a small direct readout contribution along the correct offset and a
   larger circuit effect in the forward-circle example. Brief silencing
   weakened the target-aligned separation early in all four comparisons.
   Later effects depended on the history and continuation.
4. For performance explanations, matched optimization continuation and another
   seed before concluding an architectural limit. Revised tasks require fresh
   confirmatory holdouts. Do not tune physiology parameters to make sugar win.

The new retention evidence is sufficient to sharpen the presentation's
computational conclusion. The biological question remains open rather than
answered negatively: omitted inputs and different target computations matter,
and the measured fly signal need not implement our exact nonleaky objective.

## Appendix A1 — How a trajectory trains the circuit and decoder

Use three boxes for the forward pass and a full-width strip for supervision.
The equations refer to the final paired run.

Transparent equation PNGs and the [LaTeX source](../results/cx-home-vector-appendix-equations-20261118/equations.tex)
are in `results/cx-home-vector-appendix-equations-20261118`.

1. **Fixed PFN encoding.** Convert heading and forward/rightward velocity
   into 60 prescribed activities. For PFN `i`, `p_i` is the projection of
   body velocity onto its preferred translation axis.
2. **Recurrent update.** Start with zero activity once per trajectory.
   Combine PFN input, current recurrent activity, and bias, rectify the
   result, then relax toward it at the population's learned time constant.
3. **Instantaneous readout.** Predict world-frame `x, y` with one affine
   map of the current 65-neuron state.

```text
PFN_i = 0.5 * (1 + cos(heading - preferred_heading_i)) * (1 + 5 * max(0, p_i))

drive = ReLU(W_input * PFN[t] + W_rec * rates[t] + bias)
rates[t+1] = rates[t] + (1 - exp(-dt / tau)) * (drive - rates[t])
predicted_vector[t] = D * rates[t] + intercept

correct_vector[t] = starting_position - current_position[t]
task loss = batch mean of (time mean of squared vector error / E_group)
```

`dt = 2.5 ms`. The update equation shown is the standard rule. The sugar
variant adds gate-dependent relaxation toward zero in hDeltaG. The target is
computed externally by integrating world-frame movement. `E_group` is the
typical squared target length for that trajectory group, computed from training
data and kept fixed. Add penalties for activity, edge coefficients moving
away from initialization, and decoder weights. Backpropagation through
the complete trajectory updates the edge coefficients, population biases
and time constants, and decoder weights and intercept together.

## Appendix A2 — How the standard and sugar-response models were compared

| What was matched | Final paired run |
| --- | --- |
| Initialization and architecture | Same seed-0 starting parameters, fixed PFN encoder, and anatomical edge support and contact scaling |
| Training examples | Same trajectory draws and targets for both arms. Equal sampling probability across existing short/mixed routes, long straight walks, two-return routes, and diverse returns. Sample a family, then a variant within the chosen group. |
| Optimization | Adam with learning rate 0.001, batch size 8, gradient norm clipped at 1, and 2,000 paired updates |
| Sensory intervention | The standard gate is zero. In the sugar arm, each eligible patch reentry after departure drives hDeltaG suppression. The response uses a 1 s rise constant, a 300 s decay constant, and extra relaxation of up to 1/s toward zero. The home anchor stays fixed in both arms. |

**Selection and evaluation.** Evaluate every 50 updates. Select each arm's
checkpoint independently by its lowest normalized validation vector MSE,
before evaluating the test split. Both arms selected update 2,000. The
dataset contains 2,400 training, 304 validation, and 512 test trajectories.
The 16-return extensions and familiar circle replays are separate evaluations.
Report whole-route errors in mm separately from the normalized selection
score. The paired run uses one seed.

Link to the [archived run configuration](../results/cx-sugar-reliable-return-training-seed0-20261113/training_config.json),
[run manifest](../results/cx-sugar-reliable-return-training-seed0-20261113/manifest.json),
and [evaluation report](cx_home_vector_reliable_evaluation_results.md).

## Appendix A3 — How the training routes changed

Use two strips of archived training examples.
[Early routes and the first two-return addition](../results/cx-home-vector-training-routes-slides-20261118/cx_training_routes_early_and_two_return.png)
shows the early standard model's three kinds of movement in the first
three panels and the first paired run's new polygon example in the fourth.
[Diverse repeated returns](../results/cx-home-vector-training-routes-slides-20261118/cx_training_routes_diverse_repeated_returns.png)
shows the four route classes added for the final paired run. Open circles
mark each route's start. Filled dots mark distinct endpoints. Each panel
has its own distance scale. Both strips are also available as SVG and PDF.

- **Early standard model.** 682 walk-then-pause examples, 342 randomized
  movement sequences, and 1,024 long straight walks, for 2,048 trajectories.
  The short circle on slide 4 was a separate evaluation probe.
- **First paired training.** Added 256 distinct two-lap polygon movements,
  represented by 512 training records with matched sugar schedules.
- **Final paired training.** Retained 256 two-return movements with one
  record per movement and added 96 diverse movements with four or eight
  returns. The final 2,400 trajectories were sampled using the four equally
  weighted groups from A2. The 16-return extensions were used in evaluation.

The route chronology describes the training sets. The loss and event
schedule also changed across runs, so the chronology alone does not isolate
which change improved repeated-return behavior. See the
[early model assessment](cx_home_vector_balanced_u2000_assessment.md),
[first paired evaluation](cx_home_vector_sugar_retrained_loops.md), and
[final dataset and training report](cx_home_vector_reliable_return_training.md).

## Evidence provenance and verification

- Current comparison: `results/cx-sugar-reliable-evaluation-20261115-v2`.
- Phase comparison: `results/cx-standard-loop-phases-20261116`.
- New probes: `results/cx-standard-displacement-retention-20261117`.
- Parameter/sign audit and half-step checks:
  `results/cx-home-vector-presentation-evidence-20261117`.

All result bundles are local ignored artifacts; reports and reproducible code
are prepared for version control.
The new scripts archive source, selected checkpoint identity and immutable input
hashes. Half-step replay includes prefix generation, not just continuation.
The full-rate arrays retain all six prefixes and all six branches; pair tables
include all 15 comparisons rather than only the illustrated cases. All claims
refer to the stated checkpoint and task; none is an estimate of fly homing
accuracy or a demonstration of the full CX's function.
