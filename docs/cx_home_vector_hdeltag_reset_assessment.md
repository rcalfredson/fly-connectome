# hDeltaG sugar response: measurements, candidate equations, and assumptions

Assessment dated 12 November 2026. This is a physiology-first feasibility study,
not a reset implementation. No encoder, dynamic equation, weight, checkpoint,
decoder or training run has changed. Existing two-loop arrays supply model
measurements; no new simulation is performed. No parameter is chosen using
home-vector decoding error.

The primary source is the supplied [Janke thesis](https://digitalcommons.rockefeller.edu/student_theses_and_dissertations/808/),
*A Neuronal Circuit Motif for Leaky Vector Integration*. The repository lists
2025; the PDF title page is dated September 2024. It is a thesis, not a
peer-reviewed journal paper. The supplied PDF has SHA256
`5477fe83b8fce13de34b736123333b8f509524aa13e329c4b5111c98d50e879d`.

## 1. Which measurements define the target?

Use experimental measurements to constrain the desired response. Use the model
to measure its existing activity and assess how a candidate interacts with it.
The present model has no sugar input, so it cannot supply an empirical reset
target or reset time constant.

| Evidence | What it constrains | What it does not establish |
| --- | --- | --- |
| Fig. 3.1i/j, PDF p.39, printed p.30 | Mean and bump amplitude decrease, with residual bump; paired before/after normalized calcium measurements from 17 flies | Firing rates, exact zero, or a fast reset time constant |
| Fig. 3.2, PDF p.40 | Speed-matched transitions do not reproduce the sugar-associated drop | A causal sugar-to-hDeltaG pathway |
| Fig. 3.3, PDF p.41 | Qualitative result survives an alternative fluorescence normalization | A linear calcium-to-spiking transfer function |
| Fig. 3.4a/b, PDF p.42 | Membrane potential and spike-rate reductions followed by recovery; pooled 36 drops across 13 flies | A simultaneous eight-cell population mean, or transfer parameters for every modeled neuron |
| Figs. 3.4–3.6 | Minutes-long baseline rise, behavior-dependent recovery, residual activity and saturation | One universal recovery timescale or a reset at every return home |
| Fig. 4.1 and Methods, printed pp.71–79 and 107 | Published baseline/bump model and its fitting simplifications | A measured cellular reset mechanism |

Figure 3.1 compares one second before the first sugar drop with one second
after the last drop. Multiple drops can intervene; it is not a single-step
kinetics experiment. Imaging examples have 3–5 Hz acquisition and some displayed
traces use two-second rolling averages. Those observations cannot establish an
arbitrarily fast time constant from the coarse plotted traces.

The thesis model separates a slow mean signal from a faster baseline-subtracted
columnar signal, with different leak rates. Its reported approximately 70-second
baseline and 4-second bump time constants are model-fit persistence parameters,
not measured sugar-reset time constants. The author also warns that the fitted
baseline persistence may be underestimated. In fitting, predicted hDeltaG
fluorescence is set to zero at sugar drops; that boundary condition does not
demonstrate instantaneous zero firing or establish which circuit implements it.

### Coarse numerical extraction

Digitized only the pooled black spike-rate mean in Fig. 3.4b's left panel. The
calibration and exact extraction script are versioned. An overlay is archived
for visual inspection. This is a small set of coordinates on a displayed mean,
not recovered raw recordings or independent trials.

- Five pre-event samples have median about 8.1 Hz.
- Samples at 40 and 60 seconds have median about 1.3 Hz, approximately 16% of
  that pre-event level. They are already recovering and are not the immediate
  post-sugar minimum. Do not set a reset multiplier to 0.16 from this alone.
- Displayed mean estimates are about 2.5 Hz at 120 seconds, 5.8 Hz at 600
  seconds and 6.7 Hz at 1100 seconds. This supports a much slower recovery
  than the initial decline, not a unique exponential recovery law.
- The calibrated image has about 8.5 seconds per horizontal pixel. The narrow
  transition at sugar is unresolved for fitting fast reset kinetics. Enlarging
  the raster would not restore missing temporal information.

Pixel envelopes include stroke extent and a padding pixel; they are extraction
uncertainty, not physiological confidence intervals. Color/threshold sensitivity
and raw event-level uncertainty have not been estimated. No physiological
parameter fit or statistical model comparison is claimed at this stage.

### Current model measurements

The eight modeled hDeltaG cells have IDs 19798, 20565, 28258, 42440, 48642,
49952, 57625 and 515960. Their actual arithmetic mean, in model units, is:

| Path | At start | At 8 s | At 16 s |
| --- | ---: | ---: | ---: |
| 8 | 0 | 10.54 | 21.14 |
| 9 | 0 | 8.35 | 16.87 |
| 14 | 0 | 10.82 | 21.61 |
| 15 | 0 | 8.67 | 17.34 |

These trials have no sugar event. Their 16-second ramps are not matched to the
thesis's experimental histories and cannot identify a minutes-long recovery
law. Cross-cell RMS about the mean is also recorded, but it is **not called
bump amplitude**: the archive does not provide a verified per-cell anatomical
phase map. Body-ID order is not column order. A spatial harmonic/phase-nulling
comparison requires that map first.

## 2. What scaling is permissible?

A fixed measurement link can express a model observable in experimental units:

```text
predicted_spike_measurement = a * model_activity + b
```

The gain must be positive and fixed across before/after periods and comparison
conditions. An offset needs justification; it is not an additional freedom to
remove a mismatch. Single-cell pooled spike measurements and simultaneous
population means are different observables and must not silently be equated.
No specific gain or offset is estimated from the unmatched histories above.

With consistently defined before/after anchors, an affine-invariant shape is:

```text
normalized_response(t) = (signal(t) - after_anchor)
                         / (before_anchor - after_anchor)
```

This can compare response shape despite a fixed gain/offset. It forces the
anchor values to match by construction, so it cannot independently validate
the drop's depth or residual activity. Those need separately measured absolute
levels or justified ratios. Raw post/pre ratios are invariant to a positive
gain, but not an unknown additive offset.

Fluorescence needs a separate observation model for calcium filtering and
indicator nonlinearity if quantitative comparison with rates is attempted.
Normalized calcium values, spike rates and membrane potentials must not be
pooled as interchangeable measurements. Time stays in seconds: amplitude
scaling does not authorize compressing a minutes-long recovery into our
eight-second loop period.

The observation link does not rescale internal network states. Changing hDeltaG
activity itself would alter its recurrent and decoder contributions unless
connected weights were correspondingly transformed. That would be a different
model intervention, not a cosmetic units conversion.

## 3. Candidate equations for discussion, not implementation

Write the present rate rule as:

```text
I_i = bias_i + PFN_drive_i + recurrent_drive_i
F_i = (-r_i + max(0, I_i)) / tau_i
```

Initially restrict candidate effects to hDeltaG. In each case g(t) represents
an observed sugar-consumption event, not a loop counter or a true-position
error signal. The pathway generating g(t) is not identified by these data.

**A. Temporarily reduce the driven target.**

```text
dr_i/dt = (-r_i + max(0, I_i - g(t)*J)) / tau_i
```

This is an effective inhibitory-drive hypothesis. It preserves the current
rate structure, but a common subtractive drive can change weak and strong
columns unequally and clip some at zero. The thesis does not identify J, the
transmitter, or the synaptic route. Recovery after the pulse remains governed
by the original network and need not match the observed slow trajectory.

**B. Add relaxation toward a low reference.**

```text
dr_i/dt = F_i - g(t)*k_reset*(r_i - r_reset_i)
```

This adds a decay tendency, rather than just shortening the ordinary tau.
With a common reference and isolated from other drive, it contracts the
population mean toward that reference and scales column deviations together.
Ongoing recurrent/input drive can oppose the contraction. Drop depth, pulse
duration and k_reset can trade off, and are not identified by a before/after
measurement alone. The reference need not be zero. Reducing tau alone would
only speed approach to the existing driven target, which could remain high.

**C. Distinguish population mean and contrast.** A descriptive event map is:

```text
m = mean_i(r_i)
u_i = r_i - m
m_after = m_reference + q_mean*(m - m_reference)
u_after_i = q_contrast*u_i
r_after_i = max(0, m_after + u_after_i)
```

This permits different suppression of baseline and columnar contrast, unlike
a single proportional contraction. It is a phenomenological boundary rule,
not a measured cellular mechanism. Rectification can alter the intended mean
and contrast. Separate q values should be introduced only if independently
measured baseline/bump suppression requires them; the current coarse spike
extraction cannot determine them. Contrast alone is not a spatial bump fit.

No candidate is selected yet. The existing figures constrain a response class
more strongly than a unique mechanism or kinetic parameter set. A sugar pulse
alone may reproduce a drop without reproducing the long recovery; an additional
slow state or modified drive would be a further assumption, requiring evidence.

In particular, a minutes-long rise in mean activity is not a measurement of
minutes-long loss of responsiveness or of sustained inhibition. Calibration
should distinguish a brief suppressive event followed by network-driven
recovery from a persistent, slowly fading influence. Neither can be selected
from the baseline recovery curve alone. Do not assign the observed recovery
duration directly to an inhibitory gate's decay constant.

## Task assumptions and decision boundary

A floor patch may generate an explicit sugar-contact/consumption signal in the
environment; the network receives that signal, not coordinates. A purely
visual marker is not equivalent to sugar consumption. The original food site
and a newly encountered food site also have different reference implications.
This is an explicit extension beyond PFN-only prescribed inputs.

The thesis sugar-delivery procedure used a minimum 30-second interval and
distance/return criteria. This is a protocol constraint, not a demonstrated
physiological refractory period. It does not validate resetting every eight
seconds in our repeated-circle test. Pausing alone is an essential control,
given the speed-matched evidence. Compare single-event and repeated-event
responses separately rather than assuming identical effectiveness.

Before any physiological implementation: obtain/digitize paired baseline and
bump suppression, establish a spatial map if bump fits are intended, and assess
whether event-aligned traces resolve timing. Fit a fixed observation link and
candidate response to matched physiological observables, not decoding error.
Reserve conditions/events for checking predictions if raw data become available.
Do not use plotted points on the same pooled line as independent validation.

Only after that discussion should a supported rule be tested against the
unchanged frozen reference. Improved home-vector decoding is a separate result,
not the criterion for choosing physiological parameters. If timing or observation
mapping remains underconstrained, report a bounded hypothesis/sensitivity study
instead of claiming a calibrated biological reset mechanism.

## Reproduction

The completed measurement browser is
`results/cx-hdeltag-reset-assessment-20261112/index.html`. It includes the source
page, extraction overlay, separate physiology/model plots, both CSV tables,
calibration, script and a manifest. The PDF and model input hashes are verified;
all archived model files remain unchanged. Source page imagery stays in ignored
results, and the supplied thesis PDF is not added to version control.

With the project Python 3.11 environment and Poppler's `pdftoppm` installed:

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fly-connectome-mpl \
python -m fly_connectome.home_vector.analysis.hdeltag_reset_assessment \
  --pdf 'A Neuronal Circuit Motif for Leaky Vector Integration.pdf' \
  --model results/cx-home-vector-balanced-u1975-repeated-loops-20261112 \
  --calibration docs/data/janke_hdeltag_reset_calibration.json \
  --output results/cx-hdeltag-reset-assessment-reproduction

python -m pytest -q tests/test_hdeltag_reset_assessment.py
```
