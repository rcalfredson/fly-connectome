# PFNd response curves: first digitization assessment

Completed 11 November 2026. This analysis estimates coordinates from a published
figure and fits descriptive response laws. It does not change the prescribed
encoder, train the home-vector network, or alter the frozen checkpoint or its
archived results.

The reviewable output is
`results/cx-pfn-physiology-ed4-20261111/index.html`, with methods and results in
`report.md`, calibrated coordinates and fits in CSV, extraction overlays,
PNG/PDF comparisons, the source image, and a completion manifest. The output
also copies the exact analysis script and calibration for reproducibility.

## Source and calibration

The source is [Lu et al. 2022, Extended Data Fig. 4a](https://authors.library.caltech.edu/records/73yvc-je285/files/41586_2021_4191_Fig8_ESM.webp?download=1):
individual PFNd firing-rate curves in two heading conditions. The paper's Methods
define 120-degree heading bins and fit firing-versus-projected-velocity relations
over -2 to 6 mm/s. This figure displays a narrower range. The underlying
recordings and error estimates were not obtained.
[Paper and Methods](https://wilson.hms.harvard.edu/sites/g/files/omnuum8421/files/wilson-lab/files/lu_et_al.pdf).

The source image is 2062 by 1920 pixels, SHA256
`0948bc59cd2a7dda5101f7a461664becca46f887a02c05f77a265fa8672f94a6`.
The checked-in calibration records manually identified tick centers for each
panel. Cells 5 and 12 have truncated vertical axes, beginning at 10 and 5
spikes/s respectively; these offsets are preserved.

Gold/gray color masks extract curve coordinates on a fixed half-mm/s grid.
Some coordinates fall on connecting segments, rather than original vertices.
The original bin locations are not recovered. Missing or overlapping gray traces
and samples obscured by axes remain missing. They are never replaced with zero
or with the other heading condition's curve.

The overlay was inspected, including endpoint and near-axis extraction.
An initial extraction picked up gray axis antialiasing; the final method masks
the horizontal-axis band and gray samples at the vertical axis. A regression
test checks that such axis pixels do not become firing-rate measurements.

## Descriptive comparison

For projected velocity `p`, the candidates are:

```text
published fixed shape: a * (1 + 5 * max(0, p))
proposed fixed shape:  a * max(0, 1 + 5 * p)
free positive-only:    b + g * max(0, p)
free signed:           max(0, b + g * p)
residual floor:        f + max(0, b + g * p)
```

All coefficients are nonnegative. Each curve is fitted separately, so these
are comparisons of response shape, not a full fit of heading tuning or a shared
population encoder. The fixed shapes each have one free rate scale; the free
positive-only and signed laws have two parameters; the floor law has three.

All 28 curves have enough extracted coordinates for descriptive fitting.
The final extraction retains 314 coordinates and marks 50 as unresolved.
Across 14 preferred-heading curves, median fitting RMSEs are:

| Response law | RMSE, spikes/s |
| --- | ---: |
| Published fixed shape | 1.808 |
| Proposed fixed shape | 1.889 |
| Free positive-only | 1.043 |
| Free signed | 1.149 |
| Residual floor | 1.027 |

These medians summarize different cells with different firing ranges; they are
not pooled estimates of recording noise. Each law is compared on the same
available coordinates for a given curve. Better training fit with an extra
parameter does not by itself establish a preferred physiological law.

The report also gives leave-one-sample-out errors. Because coordinates can lie
on connecting lines, this is a curve-prediction check, not independent recording,
cell or fly validation. No physiological confidence intervals or significance
tests are inferred from the coordinates.

## What this establishes

The proposed fixed shape predicts zero for `p <= -0.2 mm/s`. Among the 28
preferred-heading coordinates sampled at -1 or -0.5 mm/s, 25 have positive lower
pixel-envelope bounds, spanning 13 cells. This is a geometric disagreement with
the plotted curves; those 25 coordinates are not independent experimental
replicates. An overall rate scale cannot repair the forced zeros.

Allowing baseline/gain flexibility improves the descriptive fit. This does not
establish that signed suppression is superior: the free positive-only law fits
the preferred-heading curves at least comparably, and model preference varies
with cell and heading condition. Several displayed curves are nonmonotonic.
The narrow candidate set also cannot express every response shape, including a
positive-velocity onset when the pre-threshold intercept is constrained positive.

The two color-mask variants have median absolute coordinate change zero and
maximum change about 0.106 spikes/s where both extractions succeed. The artifact
records availability changes too. Two-pixel coordinate envelopes and six
deterministic perturbations leave the broad descriptive pattern intact. These
are extraction sensitivity checks, not biological uncertainty estimates.

## Implication for the encoder discussion

Do not adopt the proposed fixed cutoff on the strength of these curves.
No single replacement baseline/gain is estimated here. A zero-projection bin is
not necessarily a stationary animal, and a fitted floor threshold outside the
sampled velocity range is extrapolation. The residual-floor parameters can be
unidentifiable if no plateau is sampled.

For a signed response, `threshold = -b/g`. Avoiding its floor over the task's
entire speed range would be sufficient to preserve signed projections, but that
requirement has not been imposed on these fits. These PFNd-only, heading-band
fits do not justify assigning parameters to PFNv or claiming that all four input
groups eliminate the ambiguous sectors. Information preservation remains a
separate assessment after a defensible population mapping is specified.

## Reproduce

Use a Python environment with NumPy, SciPy, Pillow, Matplotlib and pytest. The
analysis performs no network access. Obtain the exact hash-identified source
image above, or reuse the archived `source.webp`.

```bash
MPLCONFIGDIR=/tmp/fly-connectome-mpl OPENBLAS_NUM_THREADS=1 \
python3 -m fly_connectome.home_vector.analysis.pfn_physiology \
  --image results/cx-pfn-physiology-ed4-20261111/source.webp \
  --calibration docs/data/lu2022_pfn_ed4_calibration.json \
  --output results/cx-pfn-physiology-ed4-reproduction

MPLCONFIGDIR=/tmp/fly-connectome-mpl OPENBLAS_NUM_THREADS=1 \
python3 -m pytest -q tests/test_pfn_physiology.py
```

The output directory must be fresh. Source hash and dimensions are checked
before extraction. The final manifest hashes the generated artifacts and
records the analysis-script hash.
