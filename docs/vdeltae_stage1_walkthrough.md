# Body-aligned vDeltaE → hDeltaG: first firing experiments

This stage asks how specified input patterns propagate through the audited contact
anatomy, and what changes when a hypothesized dendritic influence is slow. It
does not infer movement tuning, a home-vector decoder, or biological integration
from a slow response. A distributed home-vector representation is a reasonable
working hypothesis: this pathway could contribute one signal among several.
Neither a positive result here nor failure of this small model establishes where
the fly represents a fixed home location.

The new modules use the existing `RateNetwork`; the engine and the older PFN →
hDeltaB and idealized Janke experiments are unchanged. No new online extraction
or credentials are required.

## 1. Load verified anatomy

`vdeltae_stage1.load_stage1_source` verifies the saved contact audit and its
sources, then loads the selected bodies and raw contacts. `projection_counts`
uses standard-confidence contacts whose **receiving endpoint is FB-marked**.
This run allocates 2,402 contacts: 1,381 onto receiving axonal compartments and
1,021 onto dendritic compartments. Three contacts lacking the receiving FB flag
remain in the allocation inventory but are excluded; absence of that flag is
not proof they lie outside the FB. Source-side compartment does not restrict
selection.

All 18 vDeltaE and eight hDeltaG bodies remain distinct. The two C0 bodies stay
separate. The count table contains every 18 × 8 × 2 combination, including zeros.
No mapping of ten vDeltaE annotation labels onto eight ideal angular columns is
invented. Source/receiver body IDs define the actual mapping.

## 2. Specify two controlled hypotheses

For source body i and receiving body j, let A_ij and D_ij denote selected counts
onto receiving axon and dendrite. Define:

```text
P_j(t) = g b Σ_i A_ij v_i(t)
N_j(t) = -g [a / (1-k2)] Σ_i D_ij v_i(t)
tau_q  = -reference_dt / log(k2)
m(t)   = [c / (1-k1)] [1 - exp(t log(k1) / reference_dt)]
```

The **fast control** uses:

```text
tau_r dr_j/dt = -r_j + ReLU(m(t) + P_j(t) + N_j(t))
q_j = 0
```

The **slow candidate** uses:

```text
tau_r dr_j/dt = -r_j + ReLU(m(t) + P_j(t) + q_j(t))
tau_q dq_j/dt = -q_j + N_j(t)
```

The steady-state dendritic gain is identical between scenarios. This comparison
isolates its timescale, rather than giving the slow candidate a stronger static
negative input. Coupling rows preserve compartment-specific counts, channels,
sign assumptions and global strength. Two disjoint compartment components may
share a source/receiver/channel in the fast control; they add without counting
any contact twice.

Both signs are **assumed**, motivated by the thesis model; compartment-specific
positive and negative effects have not independently been established. The
negative slow state is phenomenological excitability modulation, not demonstrated
synaptic plasticity. It is assigned to hDeltaG; recurrence is omitted. Thus a
slow tail is explicitly built into this candidate, not an emergent discovery.

`m(t)` interpolates the thesis's discrete baseline on a continuous time axis. It
is an imposed drive inside the rate equation. Finite `tau_r` filters it, so the
recorded population mean need not equal `m(t)`. This is not an exact reproduction
of the thesis's recursive output equations. Absolute prescribed firing rates
enter the coupling; there is **no input centering or incoming normalization**.
Mean subtraction appears only in descriptive spatial RMS and tonic-relative
readouts.

## 3. Declare uncalibrated scales and time grid

`examples/vdeltae_hdeltag_stage1.json` records every configurable field. Defaults:

| Quantity | Value | Status |
|---|---:|---|
| Reference step | 1 s | Thesis Figure 4.1 demonstration grid |
| k1, c | 0.99, 0.001 | Demonstration baseline coefficients |
| k2, a, b | 0.8, 0.07, 1 | Demonstration model coefficients |
| Baseline time constant | ≈99.5 s | Derived from k1 |
| Slow time constant | ≈4.48 s | Derived from k2 |
| Rate time constant | 0.1 s | Provisional, carried from framework experiments |
| Global per-contact gain g | 0.001 | Fixed exploratory scale |
| Integration / recording interval | 0.005 / 0.01 s | Half-step comparison also saved |
| Source tonic rate | 0 | Chosen control baseline |
| Total added source rate | 1 | Arbitrary rate units |

These are not fitted firing-rate or per-synapse measurements. In particular,
transferring the thesis coefficients to individual contact counts does not
calibrate the anatomy. See the [thesis reading guide](janke_thesis_model_reading_guide.md)
and [dissertation](https://digitalcommons.rockefeller.edu/student_theses_and_dissertations/808/).

`prepare_config` rejects unknown fields, nonfinite scales, invalid retention,
off-grid event times, too-large rate steps and insufficient recovery time. The
primary run fixes these parameters for every stimulus; a later gain/leak
sensitivity analysis would be a distinct experiment.

## 4. Prescribe stimuli and retain state

`experiment_specs` declares 39 stimuli:

* One tonic/no-added-input control.
* A one-second pulse to each of the 18 individual vDeltaE bodies.
* A ten-second sustained input to each body.
* Population-wide pulse and sustained inputs, with the same **total** added rate
  divided equally across the 18 source bodies.

Each trial begins at t=0 with dynamic hDeltaG rates and slow states zero. Sources
are clamped to the tonic value. Added input begins at 30 s, ends at 31 or 40 s,
and the trial ends at 70 s. There are no within-trial resets. Separate trials
restart at zero. Inputs are held on `[onset, offset)`; the recorded receiver
state at offset still reflects preceding stimulation.

Matching total source activity avoids a population-size stimulus advantage.
It does not match delivered postsynaptic drive, since anatomical counts differ.
Pulse and sustained trials deliberately differ in duration and total dose.

## 5. Read out responses and challenge the interpretation

Each NPZ retains all 26 body rates and slow states, the imposed baseline,
half-step states on the same recording grid and readouts. The readouts include
population mean, peak, trough, spatial RMS and responses relative to the
**time-matched tonic control**. Body response CSVs include extrema, offset and
post-offset residuals, and final values.

The spatial RMS is spread across the eight receiving bodies after subtracting
their mean. It does not assume sinusoidal tuning or decode an angle. Rate-zero
fractions and fractions of nonpositive ReLU arguments are separate: positive
rates can decay toward zero without reaching it exactly.

Predictions and failure criteria:

* Both scenarios should follow signed contact structure; individual-body and
  matched-total population stimuli can differ because contact distributions
  differ. C0 bodies are assessed separately rather than pooled.
* Fast responses should relax on the rate timescale after stimulus offset.
  Slow-candidate responses may retain a spatially structured suppression and
  positive axonal responses may outlast their input only through rate filtering.
* Slow-state differences from tonic control must decay with the declared tau_q
  after offset. This is a **numerical check of an assumption**, not evidence
  for integration. Rectification can obscure a corresponding rate tail.
* Maximum absolute coarse/fine discrepancy divided by the fine trace's peak
  absolute response must be ≤2%. Stimulated trials use control-relative
  responses for this calculation; the baseline cannot hide an error. The tonic
  control uses its raw states. Post-input slow-state decay error must be ≤1e-10.
  Failing either condition marks the run failed; do not interpret its plots.
* If apparent retention is only a shared baseline, is erased by mean subtraction,
  or is dominated by clipping, it does not support spatial memory. Inspect raw,
  mean and spatial components together.
* Longer suppression in the slow candidate is predicted by its equation.
  Physiological perturbations showing equally fast dendritic-associated and
  axonal-associated effects, or different signs, would challenge this assignment.
  The simulation by itself cannot validate it.

Opposed patterns, cancellation, movement encoding and fixed-home-vector tests
remain later stages. A leaky recent-displacement signal and a fixed anchored
home vector have different requirements. No downstream relay is presumed to
repair heading/body-velocity differences in the existing PFN model.

## 6. Run, verify and export

From the repository root, in the project environment:

```bash
python -m fly_connectome.vdeltae_stage1 run \
  --contacts results/vdeltae-hdeltag-contact-audit-20261030 \
  --config examples/vdeltae_hdeltag_stage1.json \
  --output results/vdeltae-hdeltag-stage1-20261030

python -m fly_connectome.vdeltae_stage1 verify \
  --run results/vdeltae-hdeltag-stage1-20261030

python -m fly_connectome.vdeltae_stage1 export \
  --run results/vdeltae-hdeltag-stage1-20261030 \
  --output results/vdeltae-hdeltag-stage1-figures-20261030

python -m fly_connectome.vdeltae_stage1 verify-figures \
  --run results/vdeltae-hdeltag-stage1-figures-20261030
```

Output directories must be fresh and separate from source directories. Use new
names when repeating a run. The simulation manifest records complete/failed
status, input hashes, module hashes, package versions and output hashes. The
verifier rechecks saved anatomy, coupling specifications, clamped stimuli,
initial states and metrics recomputed from saved coarse/fine traces. It does
not rerun the differential equations or cryptographically authenticate a
manifest. Code hashes record what was present at execution, rather than requiring
the current code to remain unchanged. Original saved contact sources must remain
available for verification; the copied JSON config is the run's specification.

Exports include four offset heatmaps sharing a signed color scale; every
single-body trial is included. Representative raw/delta/slow trace panels are
chosen before examining responses: both C0 bodies, the smallest body in C1 and
C5, population controls and tonic control. All other traces remain in NPZs.
PNG and PDF exports are separate from raw simulation artifacts.

## 7. Tests

```bash
python -m pytest -q tests/test_vdeltae_stage1.py
```

Synthetic cases check confidence/ROI exclusions and count conservation, distinct
source compartments, zero pairs, DC-matched channel assignment, input budgets and
boundaries, configuration rejection, continuous state at offset, analytic slow
decay, convergence, saved-run reconstruction, artifact changes and shared figure
scales. These do not test biological validity.

## First run: 30 October 2026

The named default run completed all 78 trials. Maximum rate-response half-step
error was approximately **0.0112%** (acceptance limit 2%); maximum slow-state
half-step error was approximately 3.89e-15 relative, and maximum post-input
slow-tail absolute error was 3.47e-17. No receiver had a nonpositive ReLU
argument during the analyzed stimulation/recovery period at this gain.

Five seconds after sustained stimulation ended, the strongest retained negative
rate difference among the tested conditions was approximately -0.00712 arbitrary
units in the slow candidate. Fast-control differences were zero to numerical
precision at that checkpoint. The slow candidate therefore produces a spatially
uneven negative trace under these assumptions; the fast control does not retain
it beyond its ordinary rate relaxation. This is expected from the assigned slow
state. It neither establishes biological necessity of that state nor tests
environmental displacement direction or home-vector cancellation.

The raw metrics behind these statements can be inspected without plotting:

```python
from pathlib import Path
import pandas as pd

run = Path("results/vdeltae-hdeltag-stage1-20261030")
numerics = pd.read_csv(run / "numerical_validation.csv")
responses = pd.read_csv(run / "response_by_body.csv")
print(numerics.groupby("scenario").max(numeric_only=True))
print(responses.groupby(["scenario", "kind"])[
    ["delta_5s_after_offset", "fraction_rectified_target"]
].agg(["min", "max"]))
```
