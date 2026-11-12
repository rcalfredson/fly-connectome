# Measurements for assessing the hDeltaG sugar-response parameters

Reviewed 12 November 2026. The aim is to assess the existing engineering
parameters, not optimize decoding. No model, parameter, decoder or training
change was made. This pass adds source discovery and figure-derived measurements.

## Public source search

The primary source remains the supplied [Janke thesis](https://digitalcommons.rockefeller.edu/student_theses_and_dissertations/808/).
Its record was accessible; direct PDF download returned 403, so the already
supplied, hash-identified local PDF was used. There is no need to obtain another
copy of the thesis. No event-level dataset link was identified on the record.
No public hDeltaG sugar recording release was located through the official
[laboratory publication list](https://maimonlab.rockefeller.edu/publications/),
the [MaimonLab GitHub organization](https://github.com/MaimonLab), or bounded
title/author/cell-type searches of data repositories and preprints.

GitHub search returned twelve public organization repositories, including
general imaging tools and code for different experiments; none was identified
as these recordings. A repository title search returned no match. This is not
proof that the data are unavailable. Another author's PFNa/FC2 dataset cannot
substitute for hDeltaG sugar measurements. Secondary connectome summaries were
not used as physiological evidence. Search scope and repository names are
preserved in `docs/data/janke_hdeltag_data_search_20261112.json`.

## Additional measurements recoverable now

Extracted the bold population-mean dots from six panels, rather than attempting
to recover individual flies from overlapping symbols. The overlay was inspected.
Values below are approximate **displayed normalized calcium ordinates**, not
firing rates or raw fluorescence. Calibration uses manual tick centers and
isolated dot patches; pixel envelopes reflect extraction, not biological spread.

| Measurement | Figure / PDF page | Before | After |
| --- | --- | ---: | ---: |
| hDeltaG mean, sugar | 3.1i / 39 | 0.59 | 0.12 |
| hDeltaG bump amplitude, sugar | 3.1i / 39 | 0.207 | 0.105 |
| hDeltaG mean, speed-matched transition | 3.2a / 40 | 0.50 | 0.49 |
| hDeltaG bump, speed-matched transition | 3.2a / 40 | 0.158 | 0.151 |
| vDeltaE mean, sugar | 3.8c / 56 | 0.38 | 0.78 |
| vDeltaE bump amplitude, sugar | 3.8c / 56 | 0.157 | 0.162 |

The sugar summaries compare one second before the first sugar drop with one
second after the last drop; multiple drops can intervene. The speed controls
match those speed transitions without sugar. They are not single-event kinetic
traces or arbitrarily selected eight-second post-event windows.

The plotted after/before ratio is approximately 0.21 for hDeltaG mean and 0.51
for its bump amplitude. Those are ratios of normalized fluorescence ordinates;
they are not estimates of remaining spike rate. Per-column percentile
normalization changes the interpretation of a baseline offset. A model
observation link must reproduce the analysis before comparing such ratios.

This strengthens three constraints: baseline and bump suppression differ;
speed changes alone do not reproduce them; and vDeltaE does not simply undergo
the same reset. The latter is an additional check on the engineering example's
indirect decreases in vDeltaE. It is not evidence for directly prescribing an
extra vDeltaE response without specifying and testing another mechanism.

The earlier extraction of Fig. 3.4b provides a separate coarse spike-rate mean:
approximately 8.1 Hz before sugar, 1.3 Hz at 40–60 s, and 5.8 Hz at 600 s.
Those samples are already recovering, not an instantaneous minimum. They must
not be pooled with the calcium ordinates. The displayed time axis does not
resolve a one-second gate rise well enough for a fitted reset constant.

## Additional controls in the thesis

Figure 3.7d (PDF p.54; discussion on printed p.39) reports sugar-associated
hDeltaG decreases even when vDeltaE is silenced, followed by a rapid rise to
high activity. Normal vDeltaE input therefore contributes to the slow subsequent
rise; this does not identify an inverse relationship with a sugar-gate parameter.
Initial suppression and recovery need not be the same process.

Figure 3.7e reports persistent hDeltaG reductions following five-second vDeltaE
activation. This constrains the vDeltaE–hDeltaG interaction, not the duration of
the sugar signal itself. Figures 3.4–3.6 show behavior- and indicator-dependent
recovery and saturation, making a universal exponential recovery assumption
questionable. Figure 3.8 supplies the simultaneous vDeltaE comparison.

## What a full parameter assessment still needs

| Required quantity | Why it matters | Current availability |
| --- | --- | --- |
| Precise sugar onset, duration/consumption and all intervening drops | Separates gate timing, cumulative delivery and response latency | Event markers visible; precise arrays not located |
| Event-aligned hDeltaG firing or membrane traces around onset | Tests fast suppression independently of calcium filtering | Coarse pooled plots only; event-level series needed |
| Continuous hDeltaG signals over minutes, with movement history | Tests recovery and pre-event conditioning | Examples and pooled curves available, numerical histories not located |
| Eight-column fluorescence, normalization metadata and column identities | Separates population mean, bump amplitude and phase | Summary figures; numerical per-column arrays not located |
| Matched speed/heading and no-sugar transitions | Ensures sugar effect is not fitted using a different behavioral regime | Methods and summary controls available, aligned arrays not located |
| Simultaneous vDeltaE or perturbation-condition traces | Distinguishes a persistent sugar gate from recovery controlled by recurrent input | Figures 3.7/3.8 and captions; aligned numerical series not located |
| Per-fly/event identities, uncertainty and exclusion flags | Checks consistency across animals and reserves independent observations | Counts and descriptions; source-level records not located |
| Verified phase map for the eight modeled hDeltaG cells | Compares model bump observables with the experimental columnar analysis | Not present in the current model archive |

A useful minimal data request would ask for processed arrays underlying Figs.
3.1–3.4, 3.7 and 3.8: timestamps, sugar-event times, eight-column fluorescence
or mean/amplitude/phase, normalization constants, movement controls, fly/event
IDs, and perturbation labels. For the Sethi electrophysiology in Figs. 3.4/3.5,
ask for spike times or spike-rate time series, bin/filter definitions, membrane
potential if shareable, and aligned behavior/sugar events. Raw imaging movies
are not necessary for an initial response assessment if processed observables
and provenance are supplied. No correspondence has been sent.

## Consequences for the existing parameter set

The current `rise_tau_s=1`, `recovery_tau_s=300`, `relaxation_rate_per_s=1`, and
`reference_activity=0` remain hypotheses. The new measurements supply useful
summary targets, but do not identify those four values uniquely. A slowly
rising mean does not directly determine the duration of an imposed inhibitory
gate. Recovery-controlled-by-vDeltaE and persistent-gate hypotheses should be
compared explicitly, using the actual neural response rather than the gate.

The next defensible assessment has two levels: compare the available summary
constraints with a declared observation mapping and uncertainty; then, if
event-level data become available, assess onset and recovery on matched
histories and held-out events/flies. Plot-derived points on the same mean curve
are not independent validation data. Even full event data may leave mechanisms
or parameter combinations observationally indistinguishable; report that rather
than selecting values using decoder accuracy.

Thus a useful partial assessment is possible now. A full empirical validation
of fast kinetics, recovery and condition generalization is not yet supported
by the public materials located here. Neither model training nor decoder
refitting resolves those missing measurements.

## Reproduction and artifacts

`results/cx-hdeltag-measurement-inventory-20261112/index.html` contains the seven
reviewed source pages, dot overlay, coordinate/pixel-envelope CSV, six-panel
summary, calibration, exact script and completion manifest. The supplied PDF
and all model archives are preserved. This run contains no model simulation.

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fly-connectome-mpl \
python -m fly_connectome.home_vector.analysis.hdeltag_measurement_inventory \
  --pdf 'A Neuronal Circuit Motif for Leaky Vector Integration.pdf' \
  --calibration docs/data/janke_hdeltag_measurement_inventory.json \
  --output results/cx-hdeltag-measurement-inventory-reproduction

python -m pytest -q tests/test_hdeltag_measurement_inventory.py
```

Use a fresh output directory and the project Python environment with Poppler.
Calibration is bound to the source PDF's hash. Threshold/patch sensitivity and
biological uncertainty have not been quantified in this coarse extraction;
quoted values should retain only approximate precision.
