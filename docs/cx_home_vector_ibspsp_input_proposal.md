# IbSpsP input: published evidence and a bounded next step

Reviewed 1 November 2026. Research/design only; no simulation code or active
training artifact was changed.

## Search outcome

Lu et al., *Transforming representations of movement from body- to world-centric
space* (Nature 2022), Extended Data Figure 7, reports lateral-movement dependence
of IbSpsP population calcium bump amplitude. This is not a calibrated per-neuron
spike-rate equation. Its Extended Data Figure 5 places IbSpsP among PFN inputs.
[Primary article](https://doi.org/10.1038/s41586-021-04191-x),
[author manuscript](https://wilson.hms.harvard.edu/sites/g/files/omnuum8421/files/wilson-lab/files/lu_et_al.pdf).

The accompanying repository explicitly models PFNd/PFNv/hDeltaB. Inspection of
`translational_utils.py` and the simulation notebook found no IbSpsP encoding
implementation. The notebook has 26 cells and no IbSpsP references in its source.
Repository revision inspected: `0d29bca6c54eaf302116c8c2cbac1f01748acde8`.
[Model repository](https://github.com/druckmann-lab/Translational-velocity-and-heading-model),
[implementation](https://github.com/druckmann-lab/Translational-velocity-and-heading-model/blob/0d29bca6c54eaf302116c8c2cbac1f01748acde8/translational_utils.py).

Exact-name searches with model, velocity, encoding model and firing-rate terms
located physiological/anatomical evidence but no additional usable encoding
rule. This is a bounded search result, not proof that no such model exists.
Direct retrieval of figure PDFs/images was blocked or failed during this review;
no curve was digitized, and no signed or absolute-value response shape is claimed
here from the plotted curves. Indexed caption text was available.

## Qualification of the previous recommendation

Lateral sensitivity makes IbSpsP a candidate; it does not prove that its response
is a monotonic signed-velocity channel or that its measured activity eliminates
our encoder's ambiguity. That earlier implication was too strong. The proposed
mapping must be checked in combination with the existing PFN encoder.

One physical input variable also does not mean one model parameter. At minimum,
response shape, baseline and gain must be specified. Temporal filtering and
assignment of activity across individual cells are additional choices if used.
Population calcium amplitude must not be labeled a firing rate in Hz without a
separate calibration/observation model.

## Proposed minimal model, if no existing rule is available

Fit a static response curve to the published lateral-velocity measurements,
preferably using source data; digitized figure points would be an explicitly
approximate fallback. Choose the simplest curve supported by those measurements
rather than choosing its shape to solve the home-vector task.

```text
u(t) = baseline + gain × fitted_response(lateral_body_velocity(t))
```

The variable has units mm/s in our walking task. The curve must preserve the
paper's ipsilateral/contralateral convention. Represent the output in declared
relative activity units, with nonnegativity handled explicitly if it is used
as a rate proxy. Do not add a heading bump, hemisphere-specific sign rule or
all-cell uniformity without marking it as an additional assumption.

Before expensive training:

1. Test whether the two previously indistinguishable movements have different
   combined PFN-plus-new-input activity at training precision.
2. Test distinguishability across the complete direction and speed range,
   including both signs of lateral velocity. Breaking one pair is insufficient
   to establish complete angular coverage or equal sensitivity.
3. Check direct anatomical access to the current dynamic populations. If the
   available route is through PFNs, a fixed prescribed PFN trace cannot transmit
   the new upstream signal. Use an explicitly revised PFN model, or label any
   direct injected scalar as a separate computational control.
4. Freeze the input-curve fit independently of the home-vector training target
   for a physiology-derived comparison. Any later task-trained encoder would
   answer a different question.

## Saved anatomy already checked

The verified `results/vdeltae-inputs-live-20261024` inventory contains 31 exact-type
IbSpsP neurons. None occur among the direct vDeltaE input partners in that full
incoming census. This means no reported direct connection there in the saved
snapshot, not biological absence. It says nothing about unqueried IbSpsP routes
to hDeltaB, hDeltaC or hDeltaG, or routes through PFNs. No new live query was made.

The next concrete preparation is therefore source-curve acquisition/fit plus a
bounded outgoing-connectivity check. A new IbSpsP training variant has not been
implemented or launched.
