# Janke thesis: reading guide for the hDeltaG activity model

Targeted review, 26 October 2026. This extracts model-relevant passages from the
user-supplied full dissertation; it is not a review of every result and adds no
neural dynamics. Evidence remains dissertation evidence.

Source: Abby Janke (2025), *A Neuronal Circuit Motif for Leaky Vector Integration*,
[Rockefeller record](https://digitalcommons.rockefeller.edu/student_theses_and_dissertations/808/),
DOI 10.48496/947v-bw85. Local copy: `A Neuronal Circuit Motif for Leaky Vector Integration.pdf`.
The 127-page PDF includes preliminary material: throughout the numbered main
text, **PDF page = printed page + 9**. The local PDF is a temporary research
input, not a proposed repository artifact.

SHA-256 of the examined PDF:
`5477fe83b8fce13de34b736123333b8f509524aa13e329c4b5111c98d50e879d`.

## Priority reading

| Read | Printed pages | PDF pages | Why it matters |
| --- | --- | --- | --- |
| Figure 4.1, especially panel b and caption | 77 | 86 | Actual discrete-time equations, opposite/same-column input, anatomical schematic, example parameters. Start here. |
| Chapter 4 explanation | 70–74 | 79–83 | Separate baseline/bump states, hypothesized input effects, parameter fitting and limitations. |
| Model methods, Figures 4.1–4.3 paragraph | 107 | 116 | Mean subtraction, fitting procedure, reset, parameter bounds, sampling conversion and fit metric. |
| Figures 4.2–4.3 | 78–79 | 87–88 | Good and poor fits, parameter spread, reported median explained variance. |
| Perturbation argument and Figure 3.7 | 37–44 | 46–53 | Experimental basis for a persistent negative vDeltaE effect; hDeltaB silencing control. |
| Bump construction and amplitude tests, Figures 3.9–3.11 | 48–58 | 57–67 | Growth through decreasing troughs; walking/rest and displacement-related comparisons. |
| Phase analysis, Figures 3.12–3.15 | 59–69 | 68–78 | Evidence concerning angular history and comparison with delayed travel direction. |
| Reset and electrophysiology discussion, Figures 3.1–3.6 | 24–36 | 33–45 | Reset versus stopping, baseline behavior, saturation and measurement limitations. |
| Imaging analysis methods | 103–106 | 112–115 | Normalization, phase/amplitude extraction, epoch selection and trajectory definition. |
| Alternative interpretations | 80–82 | 89–91 | Limits on the memory interpretation and alternative explanations. |

Suggested first pass: printed p. 77 → pp. 70–74 → p. 107 → pp. 78–79 →
pp. 38–41. Then read the amplitude and phase results before defining validation
targets.

## What Figure 4.1 actually specifies

The source gives discrete-time recursions, not a membrane-potential differential
equation. In renamed notation, let `u_i[n]` be vDeltaE activity after subtracting
its across-column mean at each sample; `opp(i)` is the opposite column in the
source's eight-column representation. Let `m` be the shared hDeltaG baseline,
`z_i` the integrated, mean-subtracted component, and `y_i` predicted full signal.
Figure 4.1b, the Chapter 4 explanation and p. 107 together give:

```text
m[n]   = k1 * m[n-1] + c
z_i[n] = k2 * z_i[n-1] - a * u_opp(i)[n-1]
y_i[n] = m[n] + z_i[n] + b * u_i[n-1]
```

The last equation combines the figure's mean-subtracted output with the
separately predicted baseline, as described in the methods. The figure's
"instantaneous" term uses the preceding input sample; do not silently add an
extra biological delay when translating the sampling convention.

- `m`: a slow baseline rise with leak, modeled independently of vDeltaE despite
  perturbation evidence that vDeltaE can affect the baseline.
- `z_i`: a separate, faster memory state driven by opposite-column vDeltaE,
  with a hypothesized sign inversion associated with dendritic input.
- `b * u_i`: a direct same-column contribution, hypothesized to reflect input
  onto axon terminals, independent of the integration process.
- `y_i`: a fitted fluorescence signal, not a calibrated firing rate. Centered
  inputs and states may be negative; adding ReLU would change this model.

The opposite-column relationship depends on whether an hDeltaG cell is indexed
by its dendrite or axon. It is not permission to assign shared functional angles
from MaleCNS instance labels or to collapse all contacts into one positive weight.
The thesis anatomy uses hemibrain material; see pp. 102–103 for its FB ROI and
population-aggregation conventions, which differ from our all-Segment MaleCNS
census.

There is **no explicit hDeltaG→vDeltaE or within-population network feedback term
in these equations**. Retention of each state from the previous timestep is
temporal memory, not a claim that the anatomical reciprocal connections have
been modeled. This makes an externally driven reproduction a reasonable next
candidate; it does not establish that biological recurrence is unnecessary.

## Observations and assumptions to keep separate

| Evidence | Source location | Consequence and boundary |
| --- | --- | --- |
| Sugar reduces hDeltaG activity; speed-matched transitions do not reproduce the reset. Electrophysiology also shows reset/rise and eventual saturation. | pp. 24–27; Figs. 3.2–3.5 | Reset is more than a stopping response. Calcium and firing-rate signals are not quantitatively interchangeable. A residual bump can remain after sugar, unlike an exact zero reset. |
| hDeltaB silencing leaves broadly comparable hDeltaG dynamics in the described assay; a silencing efficacy control is reported as data not shown. | pp. 38–39; Fig. 3.7c | Supports deferring an hDeltaB-driven mechanism. Does not show all hDeltaB input is irrelevant in every condition. |
| vDeltaE silencing removes a clear hDeltaG bump and produces rapid post-reset rise/saturation; reset still occurs. | p. 39; Fig. 3.7d | Strong constraint on the input's role. The simplified baseline equation does not explain the full baseline perturbation phenotype. |
| Five-second vDeltaE activation produces a declining hDeltaG calcium signal with a persistent post-stimulus reduction. | pp. 40–41; Fig. 3.7e | Supports a persistent negative effect under this perturbation. A conductance-change/metabotropic mechanism is proposed, not demonstrated. The separate positive axonal contribution is a model hypothesis. |
| Bump amplitude grows during selected stable-phase walking epochs and shrinks during low-forward-speed epochs; trough reduction contributes to growth. | pp. 48–51; Figs. 3.9–3.10 | A model should track baseline, peak, trough and modulation amplitude separately. Do not identify amplitude with peak activity alone. |
| Short-window amplitude changes show different trends for movement away from versus toward the sugar-drop reference. | pp. 51–54; Fig. 3.11 | This is more informative than post-input decay alone. It is not proof of accurate long-term homing. |
| vDeltaE and hDeltaG phase analyses favor recent-path interpretations over sampled fixed delays, with limitations. | pp. 59–69; Figs. 3.12–3.15 | Phase was investigated; our earlier abstract-only review left this unresolved. vDeltaE is not established here as a pure instantaneous velocity input. Direct hDeltaG/hDeltaB co-imaging was technically unavailable. |

The trajectory comparisons in Figures 3.11 and 3.13–3.15 use heading and forward
speed while ignoring sideslip (methods pp. 105–106). They therefore do not
directly resolve our same-world-movement/different-heading-body-velocity tests.
The phase comparisons use finite trajectory windows; their best windows are
not interchangeable with an exponential decay time constant.

## Fitting, numbers and unresolved conventions

- The source fits co-imaged vDeltaE and hDeltaG fluorescence, using one parameter
  set per fly across eight columns. It subtracts each population's spatial mean
  at every timepoint for the columnar component (p. 107).
- Fitting uses `scipy.optimize.curve_fit`; `k1` and `k2` are bounded to [0.5, 1]
  at the recording timestep, with other parameters unbounded. The methods reset
  predicted hDeltaG fluorescence to zero at sugar drops. Initial states before
  the first reset and the exact reset/input update order need clarification.
- Figure 4.1's example, reported on a 1-second basis, has `k1=0.99`, `c=0.001`,
  `k2=0.80`, `a=0.07`, `b=1.0`, and reported explained variance 63%. These are
  fitted fluorescence-model parameters, not measured per-synapse conductances.
- Figure 4.3 reports median values `k1=0.95`, `c=0.008`, `k2=0.78`, `a=0.11`,
  `b=0.95` and median explained variance 54% across 20 flies. Chapter 4 also
  discusses poor amplitude fits and baseline inadequacy. This is fitting
  performance, not demonstrated held-out predictive accuracy.
- **Time-constant inconsistency:** p. 73 defines `tau=-dt/log(k)`. At `dt=1 s`,
  `k2=0.78` gives about 4.0 s, but `k1=0.95` gives about 19.5 s, not the roughly
  70 s baseline timescale stated alongside it on p. 79 and in p. 74's text.
  Preserve both reported statements; do not silently treat them as equivalent.
- **Sampling conversion needs checking:** p. 107 says `c`, `a` and `b` are
  multiplied by the sampling period for its 1-second reporting conversion,
  whereas `k` is exponentiated by the inverse sampling period. Do not import
  this as a verified resampling rule: `c/a` enter accumulated increments and
  `b` is an algebraic direct term. Recover code or clarify units before using
  the reported numbers at our 1 ms timestep.
- **Fit metric needs checking:** p. 107 describes a residual sum of squares
  divided by `(N-1) * variance(model predictions)` rather than the usual
  data-variance denominator. Retain the source's definition when reproducing
  its metric and report a conventional metric separately if needed.

## Implications for the next specification

Start by specifying the source's discrete recurrence at its native sampling
interval, including fluorescence preprocessing and reset semantics. If choosing
an ODE representation later, distinguish a continuous-time embedding from a
literal transcription. For example, homogeneous decay requires
`lambda=-log(k)/dt`; keeping `k` fixed while reducing `dt` changes the model.
Driven coefficients must also be converted consistently with the chosen solver.
No conversion or new dynamics are implemented here.

The first candidate is a phenomenological, externally driven baseline-plus-bump
model with a direct observation contribution. Its slow state should not be
equated to a long membrane time constant. A model of the perturbation phenotype,
biophysical receptors, explicit recurrence, or the origin of vDeltaE drive would
each add separate hypotheses beyond the Chapter 4 fit.

Before coding, resolve parameter conversion and reset conventions, decide
whether we have measured input traces or only controlled hypothetical inputs,
and declare which observations will be independent validation targets. Do not
call controlled pulses a reproduction of the fitted walking data. Keep
displacement-related history, home vector, goal direction and steering separate.

## Reproducing the text scan

From the repository root, with Poppler tools installed:

```bash
pdfinfo 'A Neuronal Circuit Motif for Leaky Vector Integration.pdf'
pdftotext -layout 'A Neuronal Circuit Motif for Leaky Vector Integration.pdf' /tmp/janke-thesis-layout.txt
rg -n 'CHAPTER 4|Figures 4.1-4.3|Figure 3.7|Figure 3.13' /tmp/janke-thesis-layout.txt
pdftoppm -f 86 -l 86 -scale-to 1900 -png -singlefile \
  'A Neuronal Circuit Motif for Leaky Vector Integration.pdf' /tmp/janke-figure-4-1
```

The equations are embedded in a figure: inspect PDF page 86 visually rather
than relying on extracted text. Figure 4.3 on PDF page 88 was also inspected
visually for the reported parameter values.
