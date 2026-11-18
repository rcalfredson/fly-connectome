# Persistent error and displacement retention in the frozen standard network

17 November 2026. This extends the [matching-phase analysis](cx_home_vector_loop_phases.md)
without changing the encoder, checkpoint, dynamics, decoder or home anchor.

Artifacts: `results/cx-standard-displacement-retention-20261117`.
Numerical checks and current parameter inventory:
`results/cx-home-vector-presentation-evidence-20261117`.
Both use the standard network's independently validation-selected update-2000
checkpoint from `cx-sugar-reliable-return-training-seed0-20261113`.

## Main findings

1. Final-loop error on the four familiar circles is predominantly a persistent
   vector offset: 90.8–99.3% of squared error. Cyclic gain/timing errors exist,
   especially on path 9, but are secondary in these examples.
2. Under identical circular continuations, the network strongly attenuates
   differences between physically reached states whose correct home vectors
   remain different. This affects the complete neural state as well as the
   current decoder output.
3. These observations support input-dependent forgetting in this learned
   solution. They do not show that forgetting is the sole cause of the original
   circle offsets, that stabilization necessarily requires forgetting, or that
   the architecture could not learn a better integrator.

## Describing the error without correcting it

For every loop, fit each raw predicted and correct Cartesian component to a
constant plus one cycle of cosine and sine. This is a descriptive Fourier
projection using the whole cycle, not a new decoder or an improvement applied
to predictions. No fitted quantity is fed back to the model.

```text
error(phase) = prediction(phase) - target(phase)
error = mean_error + first_harmonic_error + remaining_error
MSE = ||mean_error||^2 + mean(||first_harmonic_error||^2)
                         + mean(||remaining_error||^2)
```

The terms are orthogonal on the uniformly sampled complete eight-second cycle.
The terminal state is excluded from the phase average. Positive peak lag means
that the predicted component's fitted harmonic peak occurs later than truth;
negative means earlier. Cartesian component peak lags are not a single global
vector delay. Amplitude ratios compare the fitted cyclic amplitudes and do not
include the component's mean value. Values below are for loop 16, not whole-run
averages.

| Path | Mean x error, mm | Mean y error, mm | Offset share of squared error | x amplitude ratio | y amplitude ratio | x peak lag, s | y peak lag, s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | +1.8768 | −0.4251 | 99.25% | 1.119 | 0.991 | +0.001 | +0.127 |
| 9 | +1.3848 | +0.4051 | 90.84% | 0.829 | 0.593 | −0.336 | +0.003 |
| 14 | −0.9301 | +1.1974 | 97.38% | 0.981 | 1.235 | −0.041 | +0.103 |
| 15 | −0.9593 | +0.8675 | 98.29% | 0.874 | 1.042 | −0.051 | +0.128 |

A single global intercept correction would not remove all these differently
directed, trajectory-dependent offsets. No such correction is made here.

![Error budget](../results/cx-standard-displacement-retention-20261117/circle_error_budget.png)

![Signed errors](../results/cx-standard-displacement-retention-20261117/circle_error_components.png)

## Common-input probe design

Six equal-duration, four-second histories start from zero neural activity and
from the original home origin:

- Remain stationary at heading 45 degrees.
- Walk forward 2 or 4 mm at heading 45 degrees (0.5 or 1 mm/s).
- Move rightward 2 or 4 mm at that heading (0.5 or 1 mm/s).
- Walk a closed four-segment square and finish at the original home.

All terminate at heading 45 degrees, have equal elapsed time, and produce actual
reachable neural states. The final heading/control change is instantaneous under
the model's held-control convention, not a forced change in neural state. Each
state then receives exactly the same continuation within a branch:

- A 32-second stationary pause at heading 45 degrees.
- A 32-second forward straight walk at heading 45 degrees.
- Each of the four archived 128-second circular movements (16 loops), including
  both rotation directions and forward/rightward traversal.

Different branches can have different initial headings. Within each branch,
all six histories have identical future controls, PFN inputs and clock time.
Each continuation begins with the prefix's final neural state. There is no
reset, decoder adjustment or sugar dynamics. The correct target remains
referred to the original home, not to the start of the continuation.

```text
correct_home_vector_i(t) = prefix_home_vector_i + continuation_home_vector(t)
correct_home_vector_B(t) - correct_home_vector_A(t) = constant
```

In particular, returning to the *continuation's* start is not returning to home
for a displaced prefix. Different constant offsets must persist after each
circular return. All 15 pairs of histories are retained in the numerical tables;
plots show rest versus forward-4mm, rest versus right-4mm, and rest versus square,
chosen in the design before scoring.

## Different home vectors are not preserved through repeated movement

The required separation in both rest-versus-4mm comparisons is exactly 4 mm.
It is imperfectly encoded at the beginning: decoded separations are 3.424 mm
for the forward prefix and 2.157 mm for the rightward prefix. Their subsequent
collapse therefore establishes further attenuation, rather than just a failure
to encode the prefix in the first place.

| Continuation | Duration | Final decoded separation: forward-4mm versus rest | Final decoded separation: right-4mm versus rest |
| --- | ---: | ---: | ---: |
| Pause | 32 s | 2.12983 mm | 5.14609 mm |
| Forward straight | 32 s | 0.39744 mm | 0.69835 mm |
| Path 8: counterclockwise, forward | 128 s | 0.00088 mm | 0.00054 mm |
| Path 9: counterclockwise, rightward | 128 s | 0.01125 mm | 0.00027 mm |
| Path 14: clockwise, forward | 128 s | 0.00113 mm | 0.00124 mm |
| Path 15: clockwise, rightward | 128 s | 0.02002 mm | 0.01793 mm |

Norm alone does not determine whether the *direction* of separation is correct.
The tables also record its error vector and signed gain along the correct offset:

```text
signed_target_gain = dot(decoded_B - decoded_A, correct_B - correct_A)
                     / ||correct_B - correct_A||^2
```

Ideal gain is 1. For the forward-4mm and right-4mm pairs it begins at 0.828 and
0.536, respectively. After a 32-second pause it is 0.531 and 0.650; the latter
has a large incorrectly directed component despite its 5.146-mm norm. After
32 seconds of forward walking, gains are 0.081 and 0.100. After 128 seconds of
circular input they are at most 0.00431 across these eight comparisons.
The behavior is input- and direction-dependent, not a demonstrated universal
single-exponential leak or an exclusively instantaneous-velocity response.

Complete state differences shrink too. For path 15, forward-4mm versus rest
falls from 43.413 to 0.0400 model units; right-4mm versus rest falls from 25.052
to 0.0359. These are Euclidean population distances, not firing-rate estimates.
The model is not merely preserving a large hidden difference that the current
decoder happens to ignore. Residual finite differences do remain; these
measurements do not prove identical states or the impossibility of recovering
some signal with another readout.

![Retention under clockwise/rightward circles](../results/cx-standard-displacement-retention-20261117/path_15_retention.png)

![Retention during a pause](../results/cx-standard-displacement-retention-20261117/pause_32s_retention.png)

## Equal home vectors: complementary control

The rest and closed-square histories end at the same home vector, so their
required decoded separation is zero. Initial prediction separation is 1.111 mm,
with state separation 75.429 model units. Under path 15 these become 0.0113 mm
and 0.0226 units. Removing this history difference is appropriate for the
immediate target, whereas removing the 4-mm displacement difference is not.
The present dynamics attenuate both. These probes do not establish which
specific recurrent interactions cause the attenuation.

## Implications and limits

A persistent integrator need not return every neuron to zero or avoid all
contraction. It could remove irrelevant differences while preserving a family
of states or repeating cycles indexed by accumulated displacement. These
sampled histories instead approach nearly the same driven response even when
their home vectors should remain different.

This supports a computational limitation of the **selected learned dynamics**,
not proof of an architectural impossibility or of why training selected that
solution. Finite optimization, task distribution, normalization, regularization,
parameter bounds and imperfect readout remain possible contributors. No
optimization intervention or alternative decoder is tested here. No causal
connection-specific mechanism has been established.

Leaky memory is not automatically a physiological failure: the relevant
experimental literature includes imperfect, recent-motion integration. Whether
the model reproduces that activity requires matched cell-type observables and
conditions, independently of the exact-home-vector target. See the
[presentation brief](cx_home_vector_presentation_brief.md) for the evidence map.

## Verification and reproduction

The output stores full-rate states, raw predictions and geometric targets for
all six prefixes and all six continuations, with all pair metrics. Its selected
checkpoint and immutable inputs are hashed. Parameters remain unchanged.
Waveform decomposition, signed separation and reachable-prefix geometry have
unit tests. Independent array calculations and half-step checks are reported
with the presentation inventory. A half-step replay recomputes both prefixes
and continuations at 0.00125 seconds; it does not start from the coarse prefix
states. No numerical threshold is interpreted as a biological criterion.

Both new bundles pass artifact hash verification. Independent calculations
reproduce constant correct pair offsets and reconstruct all six branches'
predictions from the saved activities and selected affine decoder. Across
the pause/path-9/path-15 half-step checks, the largest prediction RMS difference
is 0.00800 mm and the largest change in final pair separation is 0.00407 mm.
For path 15, forward-4mm versus rest ends at 0.02002 mm separation at the original
step and 0.01992 mm at half step, versus the required 4 mm. Forty-eight relevant
tests pass. These checks support the scale of the observed effect without
claiming exact convergence to identical states.

```bash
MPLCONFIGDIR=/tmp/fly-connectome-mpl OPENBLAS_NUM_THREADS=1 \
/home/tracking/miniconda3/envs/py3.13-torch/bin/python \
-m fly_connectome.home_vector.analysis.displacement_retention \
--output results/cx-standard-displacement-retention-reproduction

MPLCONFIGDIR=/tmp/fly-connectome-mpl OPENBLAS_NUM_THREADS=1 \
/home/tracking/miniconda3/envs/py3.13-torch/bin/python \
-m fly_connectome.home_vector.analysis.presentation_inventory \
--retention results/cx-standard-displacement-retention-reproduction \
--output results/cx-home-vector-presentation-evidence-reproduction
```

Choose fresh output directories. These are exploratory probes on familiar motion
protocols, not additional independent test families or a multi-seed estimate.
