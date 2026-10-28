# Common rate framework: initial pilot

Implemented 26 October 2026. This is an experimental foundation for bounded
circuit models, not a validated whole-brain model. Existing extraction,
movement, trajectory, decoder and presentation commands are unchanged.

## State and time conventions

`fly_connectome.rate_framework.RateNetwork` uses:

```text
tau_r[i] * dr[i]/dt = -r[i] + gain[i] * ReLU(bias[i] + I[i] + q[i] - threshold[i])
I[i] = external_drive[i] + sum_j W_rate[j,i] * r[j]
tau_q[i] * dq[i]/dt = -q[i] + external_slow_drive[i] + sum_j W_slow[j,i] * r[j]
```

`r` is a nonnegative rate. `q` is an optional signed excitability offset in
rate-equivalent input units; `slow_tau_s=0` disables it. A slow offset is a
phenomenological state, not a demonstrated cellular mechanism, synaptic learning
rule or home-vector accumulator. No connection weights change during a run.
Bias and threshold use input units, gain is dimensionless, and all time constants
and timesteps use seconds. Share parameters by cell type when justified; the
node table makes every per-neuron assignment explicit.

Declare either `Hz` or `arbitrary_rate_units` for the network. The PFN compatibility
case and the hDeltaG candidate both use arbitrary rate units. There is no
empirical Hz calibration or general rate-to-fluorescence observation model yet.

All fast and slow states update simultaneously from the preceding state, so
neuron order does not determine propagation. Inputs are held on `[t[n],t[n+1])`;
the terminal sample has no following interval. There is no extra synaptic delay.
Two integration methods are available:

- `euler`, the default, retains the existing rate solver's arithmetic and requires
  `dt <= min(dynamic time constants)/20`.
- `exponential_euler` integrates each leak with the preceding input held fixed.
  It is exact for a single state under a held drive, not for the full coupled
  recurrent network. Accuracy still requires convergence tests.

Every retained recurrent edge participates at every timestep. A maximum-hop
parameter is not used to truncate feedback. Circuit membership and omitted
boundary inputs must be selected separately. A prescribed neuron is clamped
to its supplied nonnegative rate trace; the solver rejects incoming edges,
external drive or slow-state dynamics on it rather than silently discarding
feedback. To study feedback into that neuron, make it dynamic and specify its
input generation instead of clamping it.

## Connectivity and evidence

The neuron table requires `neuron_id`, `cell_type` and `tau_s`. Optional fields
are `bias=0`, `threshold=0`, `gain=1`, `slow_tau_s=0`. IDs are canonical strings
inside the engine; real body IDs are preserved in the PFN adapter's config and
tables. Edges refer to IDs, not row positions.

`coupling_from_counts` takes a contact table (`source_id`, `target_id`, `contacts`,
optionally `channel`) and explicit source-type/target-type/channel rules. Each
rule declares `sign`, nonnegative `strength`, `sign_evidence` and `sign_detail`.
Evidence labels are `measured`, `inferred` or `assumed`; unresolved signs require
an explicit scenario choice and explanation. No transmitter or receptor lookup
is performed automatically.

Without normalization:

```text
effective_weight = contacts * strength * sign
```

Optional normalization divides contacts within each receiving neuron and
declared presynaptic `input_group`/channel, before applying strength and sign.
The audit table retains raw counts, the normalization denominator and effective
weights. This is a fraction within a selected projection, not normalization by
all biological input. Disconnected neurons remain in the node table.

The engine also accepts explicitly declared effective coupling tables, which
must include source/target IDs, `effective_weight`, `channel`, `sign_evidence`
and `sign_detail`. The idealized hDeltaG candidate uses this route: its logical
column couplings are not labeled as anatomical contact counts.

The first implementation has fast-rate and slow-excitability channels. It does
not yet model local terminal release, receptor conductances or morphological
compartments. Multiple channels can represent declared effects, but do not
automatically allocate or duplicate anatomical contacts between compartments.
The MaleCNS census's body-pair totals do not determine that allocation.

## Initial experiments

Activate the existing project environment and run from the repository root.
These commands are offline and use separate, new output directories. No new
dependencies or credentials are required beyond the existing environment.

### PFNd/PFNv→hDeltaB compatibility

```bash
python -m fly_connectome.framework_experiments pfn-reference \
  --movement results/pfn-hdeltab-movement-groupnorm-pfnd6-v1.0 \
  --trajectory results/pfn-hdeltab-trajectory-v1.0 \
  --output results/common-framework-pfn-pilot-20261026
```

Requires complete saved movement and trajectory runs, not included in Git.
Reconstructs coupling from raw contacts with per-receiver PFNd-L/R and PFNv-L/R
normalization, PFNd×6/PFNv×1, gain 0.001, b=1, tau=0.1 s and the saved timestep.
Preserves all source/target identities and continuous state across movement
changes. Checks against the saved encoding and trajectory, the original solver,
an exact held-drive solution and a half-timestep run. This case retains the
existing feedforward topology; generic feedback is tested separately on
synthetic networks.

`inherited_heading_differences.csv` carries the existing independent-grid
statistics. It does not recompute or repair their differences. Geometry and
return-to-origin are not inputs to the new solver.

### vDeltaE→hDeltaG reference and hypothetical rate candidate

```bash
python -m fly_connectome.framework_experiments janke-comparison \
  --output results/common-framework-janke-pilot-20261026
```

No saved datasets or thesis PDF are needed to run this controlled calculation.
The [thesis reading guide](janke_thesis_model_reading_guide.md) identifies the
source equations and unresolved conventions. `janke_reference.py` implements:

```text
u_i[n] = v_i[n] - mean_j(v_j[n])
m[n] = k1 * m[n-1] + c
z_i[n] = k2 * z_i[n-1] - a * u_opp(i)[n-1]
y_i[n] = m[n] + z_i[n] + b * u_i[n-1]
```

The eight indices are idealized FB columns, not eight vDeltaE neurons. The
18-body MaleCNS vDeltaE population has not been assigned to these columns.
The default coefficients are Figure 4.1's example, **explicitly interpreted on
a 1-second reference grid for this demonstration**: k1=.99, c=.001, k2=.8,
a=.07, b=1. They are not calibrated firing-rate or per-synapse parameters.

The rate candidate uses a separate slow excitability state and a provisional
0.1 s rate relaxation. Under a held input, the slow state reproduces the
reference increment using:

```text
tau_q = -reference_dt / log(k2)
slow_target_i = -a / (1-k2) * u_opp(i)
```

The baseline is imposed using the analogous exact leak/increment embedding.
By default, spatial centering is imposed through boundary-drive corrections to
the raw nonnegative prescribed signals. This is an experiment-specific choice,
not a behavior of the common solver. These operations are not explained as a
biological circuit. The ordinary rate stage tracks the rectified combination
of baseline, slow offset and direct same-column input.

The separately saved signed fluorescence proxy combines the latent states
with the previous native input sample. Matching that proxy to the source
recurrence validates the mathematical embedding only. It does not validate
the rate stage or constitute a physiological observation model. Rate
rectification and relaxation can change the bump and its timing; differences
are reported rather than fitted away.

The default conditions are a uniform control, a pulse, a sustained pattern and
two opposite spatial patterns. The prescribed signals are `1 + 0.05*cos(...)`
during events. "Opposite" is a 180° change in a controlled column pattern,
not a measured movement reversal. No movement-to-vDeltaE tuning is invented.
Zero initial states and resets at 0 and 15 s are explicit conventions. A reset
overrides baseline, integrated state and rate/output at that sample. The source's
exact reset order and initial states remain unresolved.

To change the protocol, copy a completed run's `config.json` to a new local
configuration file, edit it, and pass `--config path/to/config.json` with a new
output name. The file specifies reference coefficients/timestep, candidate dt,
rate tau, duration, reset times, candidate `input_mode`, and named events
(start/stop, patterned amplitude/phase and/or an additive `uniform_offset`).
Negative uniform offsets can test silencing, provided total prescribed inputs
remain nonnegative.
Candidate dt must divide the reference interval. Never hold a retention factor
fixed while changing its physical update interval.

### Centered versus absolute candidate input

Select `input_mode` in the configuration, or override it with
`--input-mode centered` / `--input-mode absolute`. Existing configurations without
this field retain centered behavior. The switch affects the candidate only:
the thesis fluorescence reference always remains mean-subtracted.

- `centered`: uses `u_i = v_i - mean(v)` in both candidate coupling channels,
  implemented through the external mean corrections described above.
- `absolute`: removes those corrections. The same direct and opposite-column
  couplings now receive `v_i` itself. The slow offset can have a nonzero spatial
  mean, and `m` is just the prescribed baseline component, not the total
  population mean. This is a different model hypothesis, not a reproduction of
  the thesis or a calibrated absolute-Hz model.

The [uniform-input configuration](../examples/janke_uniform_input_comparison.json)
provides a tonic control and an otherwise identical five-second uniform pulse.
No parameters are retuned between modes:

```bash
python -m fly_connectome.framework_experiments janke-comparison \
  --config examples/janke_uniform_input_comparison.json --input-mode centered \
  --output results/common-framework-uniform-centered-20261026
python -m fly_connectome.framework_experiments janke-comparison \
  --config examples/janke_uniform_input_comparison.json --input-mode absolute \
  --output results/common-framework-uniform-absolute-20261026
```

Use new output names when repeating. Compare each pulse against its own mode's
tonic control: absolute input also changes the background drive, so comparing
raw levels between modes confounds tonic and perturbation effects. The centered
candidate cancels this uniform pulse. With the example coefficients, the absolute
candidate has an initial positive response and a negative post-pulse tail relative
to control; removing centering alone does not explain the reported decline
during vDeltaE activation. Neither uniform condition should create a directional
bump. No anatomical or receptor sign has been established by this comparison.

In absolute mode, the latent proxy is checked against a separate absolute-input
discrete mathematical control. Its difference from the unchanged centered thesis
reference is reported, not treated as numerical failure. The compatibility field
`fluorescence_proxy` remains in activity files, but denotes an uncalibrated latent
proxy; the explicit `{experiment}_latent_control.npz` records the selected
mathematical control and candidate proxy. Neither is the firing-rate output.

## Outputs and checks

Each run writes `framework_run.json` with settings/provenance, code and source
hashes, numerical validation and hashes of its own outputs. Existing output
directories are refused; failures after creation receive a failed manifest.
Sources are read-only and checked for changes. The temporary thesis PDF is
neither read nor copied by these experiments.

- PFN outputs: raw contact matrix, node and coupling/rule tables, sampled input,
  common and legacy/exact activity, error metrics, inherited heading statistics,
  summary and comparison PNG/PDF.
- Janke outputs: explicit config/assumptions, idealized nodes/coupling, separate
  reference fluorescence and candidate rate/offset arrays, numerical comparisons,
  column plots, and mean/peak/trough/first-harmonic amplitude/phase readouts.
  Flat-pattern phases are undefined, not zero-angle estimates. This harmonic
  readout is not an exact reproduction of the thesis's analysis pipeline.

```bash
python -m fly_connectome.framework_experiments verify results/common-framework-pfn-pilot-20261026
python -m fly_connectome.framework_experiments verify results/common-framework-janke-pilot-20261026
python -m pytest -q tests/test_rate_framework.py tests/test_framework_experiments.py
```

Verification checks completeness and output hashes; it does not rerun dynamics
or provide physiological validation. Results remain under ignored `results/`.

Numerical gates are legacy agreement to floating-point tolerance, PFN exact and
half-dt relative errors <2%, latent proxy agreement with the selected centered
reference or absolute mathematical control within 1e-12, and
candidate rate half-dt relative error <2%. Recurrent examples are checked against
an independently solved equilibrium, permutation invariance and synchronous
update expectations. Slow states are checked against analytical held-drive
decay. Continuous restarts, input clamping, signed coupling, undefined phases,
malformed data, failed manifests and tampering have regression coverage.

Time-step restrictions are local numerical checks, not proofs of recurrent
stability. The rate guard stops a divergent run; it is not a biological ceiling.
`record_every` reduces output storage while keeping every integration step and
an exact restart state. Runtime scales with selected edges and timesteps, not
an enumeration of feedback paths. Inputs are currently supplied as in-memory
sampled arrays, so this pilot is intended for small subsets, not a whole CNS.

## Minimal custom recurrent circuit

The Python API can test a bounded network without modifying existing commands:

```python
import numpy as np
import pandas as pd
from fly_connectome.rate_framework import RateNetwork, coupling_from_counts

neurons = pd.DataFrame({
    "neuron_id": ["A", "B"], "cell_type": ["A", "B"], "tau_s": [.1, .1],
})
contacts = pd.DataFrame({
    "source_id": ["A", "B"], "target_id": ["B", "A"], "contacts": [2, 2],
})
rules = pd.DataFrame([
    {"source_type": pre, "target_type": post, "channel": "rate",
     "sign": 1, "strength": .2, "sign_evidence": "assumed",
     "sign_detail": "Synthetic reciprocal test, not biological evidence"}
    for pre, post in [("A", "B"), ("B", "A")]
])
coupling = coupling_from_counts(neurons, contacts, rules)
model = RateNetwork(neurons, coupling, rate_unit="arbitrary_rate_units")
drive = np.tile([1., 0.], (5001, 1))
trace = model.solve(drive, .001, record_every=10)
# Continue with initial=trace.final_state; time in each returned trace starts at 0.
```

For anatomy-based subsets, populate nodes and contacts from verified saved
censuses, declare all signs/scales and omitted inputs, and retain zero-connected
nodes. The API does not select those boundaries or infer unknown dynamics.

## Interpretation and next decision

This pilot establishes a common engine and reference checks. It does not
establish a calibrated hDeltaG rate model, necessary biological recurrence,
plasticity, displacement memory, a home vector, goal direction or steering.
The proposed dendritic/axonal signs, baseline generator, spatial mapping and
measurement conversion remain separate unresolved questions. Before adding
more profiled cell types, choose the observation or perturbation that the next
addition should explain and supply constraints on its dynamics and boundary
inputs.
