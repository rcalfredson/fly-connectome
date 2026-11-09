# Mapping PFN inputs to initial dynamic-neuron tuning

November 6, 2026; frozen balanced checkpoint **update 1975**. The
[all-neuron input atlas](../results/cx-home-vector-balanced-u1975-pfn-input-atlas-20261106/index.html)
connects the prescribed PFN encoding and learned input weights to the initial
responses of all 65 dynamic units. No training or fitting is performed.

## What is inherited from PFN inputs?

These are overlapping response maps, not a one-to-one assignment of movement
combinations to neurons. Multiple PFNs can drive each neuron, a neuron responds
over many conditions, and multiple neurons share a sampled peak. The trained
atlas alone does not separate anatomy, prescribed encoding and learned gains.
Similar peaks or preserved tuning shapes do not establish that training had no
effect, nor that the neuron lacks history-dependent activity.

At the initial zero state, recurrence contributes zero. For neuron j:

```text
input_drive_j = sum_over_PFN_i(PFN_rate_i * effective_input_weight_ij)
target_j = relu(population_bias_j + input_drive_j)
rate_j(dt) = (1 - exp(-dt / population_tau_j)) * target_j
```

Each PFN rate is prescribed from heading and forward/rightward velocity.
Connectivity determines which PFNs contribute; trained gains and signs determine
their mixture. Bias and rectification can suppress parts of that mixture, while
the time constant scales the first rate increment. This can copy a dominant
input's tuning or combine several inputs into a different pattern. Decoder
coefficients have no role in this first neural response.

The atlas reconstructs the observed first update across all 200 onset conditions
and all 65 units from these terms. Maximum absolute reconstruction error is
approximately **1.03e-7 model units**. Structural input counts and directional
modulation on the 1 mm/s onset grid are:

| Population | Units | With direct PFN input | Directionally modulated at first update |
| --- | ---: | ---: | ---: |
| hDeltaB | 19 | 19 | 19 |
| vDeltaE | 18 | 16 | 16 |
| hDeltaG | 8 | 0 | 0 |
| hDeltaC | 20 | 20 | 20 |

Initial modulation means nonzero variation over sampled heading/body-direction
conditions, not a physiological tuning measurement. A flat first response does
not mean the unit is unimportant or cannot become directionally sensitive through
recurrence later.

## Read the atlas

Each population has maps at **2.5 ms**, **50 ms** and **1 s**, at 1 mm/s. The first
is a pure input/bias response from zero; later maps include recurrent dynamics
developing under the same standardized starting conditions. Headings and body
directions are sampled in 45-degree increments.

Three views answer different questions:

- **Pattern:** each neuron's map is rescaled from its own minimum to maximum.
  This exposes tuning shape when amplitudes differ greatly. Flat maps are gray.
  Amplitude is removed: do not use this view to decide which neuron fires most.
- **Raw:** actual rates, with a common model-unit scale across units within the
  population/time figure. Use this for amplitude comparisons. Figure scales
  differ between populations and times.
- **Movement delta:** moving response minus a stationary response at the same
  heading and elapsed time, both starting from zero. PFN input at rest is not
  zero, so subtraction separates movement-associated changes from the heading-
  dependent stationary response. Rectification means this is not generally an
  additive attribution of final activity to individual input edges.

The population tables link each body ID to its direct input-pair count, largest
movement-drive input and one discrete first-step maximum. Flat first responses
have no assigned maximum. The CSV gives tied-maximum counts; a sampled maximum
does not establish a unique or continuous preferred angle.

## Trace the underlying connections

`input_connections.csv` exports every direct PFN edge with source/target IDs and
types, assigned PFN heading/body-direction preferences, anatomical contacts,
normalization denominator and learned effective weight. Each edge also has raw
and movement-minus-stationary **preactivation drive RMS** over the 1 mm/s grid.

`neuron_input_summary.csv` gives each dynamic unit's bias/time constant, numbers
of positive/negative effective input weights, initial tuning range, discrete
peak and strongest movement-drive input. The strongest-input share divides its
drive RMS by summed input drive RMS; it is not explained variance or causal
importance, and opposite signed drives can cancel before ReLU.

There are meaningful differences in how inputs are combined. Each hDeltaB unit
has **24–35** direct PFN pairs, while connected vDeltaE units have **1–5** and
hDeltaC units have **2–8**. For 142911, three direct inputs are present and PFNd
77360 contributes approximately **96.65% of summed edge movement-drive RMS**
over the specified grid. Thus that unit is closer to a dominant-input example
than the hDeltaB population, where no one edge exceeds half the summed movement-
drive RMS in this calculation.

These assigned PFN preferences and fitted gains describe this computational
model. They do not establish physiological preferred directions, signs or gains
of the dynamic neurons.

## Reproduce the mapping

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fly-connectome-mpl \
python -m fly_connectome.home_vector.analysis.input_tuning \
  --onset results/cx-home-vector-balanced-u1975-onset-20261106 \
  --output results/cx-home-vector-balanced-u1975-pfn-input-atlas-NEW
```

Use a fresh output label. This postprocessor requires the completed onset bundle,
which includes rates, actual initial PFN inputs, conditions and checkpoint.
The onset bundle's `reproduce.py` records its generation recipe; its output
path must be changed before rerunning into a fresh directory. The atlas verifies
input hashes, checkpoint/circuit identity, encoded PFN inputs, zero starting
states and first-update reconstruction. Outputs include PNG/PDF figures, CSVs,
`input_tuning.json`, a linked HTML index and a hashed completion manifest. The
source onset bundle is checked for changes and remains untouched.

## Log-ready initial maps

The [three-column export](../results/cx-home-vector-balanced-u1975-initial-tuning-log-20261106/export.json)
contains the hDeltaB, vDeltaE and hDeltaC `pattern` maps at 2.5 ms, with
freestanding filenames for wiki uploads. Data, neuron order, axes, colors and
normalization match the original atlas; subplot arrangement changes to three
columns. PDF versions are included alongside the PNGs.

The same directory includes a per-body preferred-pair CSV and Markdown table,
a peak-occupancy CSV grouping bodies that share a sampled maximum, and suggested
log wording. All 57 displayed units are retained: 55 have a sampled directional
maximum, and the two vDeltaE units without direct PFN input have blank preferred
pairs. There are 30 distinct occupied ordered pairs.

For future complete atlas runs, use `--columns 3` to select this arrangement;
the default remains five columns. The log export's `reproduce.py` records the
three-figure subset export and verifies its peak table against the original
summary. Choose a fresh output directory before rerunning that recipe.
