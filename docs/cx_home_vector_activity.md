# Frozen checkpoint activity atlas

The CPU recorder replays all 12 standard challenge trajectories using a snapshot
of the selected `best.pt`. It saves every dynamic neuron's rate, the actual PFN
inputs, controls and home vectors, then plots neural activity and signed
population contributions through the unchanged decoder. It performs no fitting
and does not change the model or training run.

The first completed atlas is
[the update-1975 activity index](../results/cx-home-vector-balanced-u1975-activity-20261106/index.html).
It contains 70,012 time samples across the 12 challenges. Predictions and targets
match the earlier checkpoint challenge evaluation exactly. The maximum absolute
residual when reconstructing predictions from population contributions plus the
intercept is 0.00003052 mm (floating-point summation order).

## Run

In the project's Torch environment:

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fly-connectome-mpl \
python -m fly_connectome.home_vector.analysis.activity \
  --run results/cx-home-vector-hdeltac-long-balanced-seed0-u2000-resumed-2026-11-05 \
  --source results/cx-home-vector-hdeltac-long-data-20261104 \
  --output results/cx-home-vector-balanced-u1975-activity-NEW \
  --threads 1
```

Replace `NEW` with a fresh output label. The selected update and checkpoint hash
are recorded; `best.pt` need not be the last optimizer update. For this completed
run it is update **1975**, not 2000.

`--run` also accepts a completed challenge-evaluation or activity bundle. Both
preserve the original checkpoint, allowing replay after the training directory
has moved. For a challenge comparison bundle, `--snapshot-key reference` chooses
its reference checkpoint instead of the default `current`. A single-model
activity bundle contains only `current`. The matching prepared source is still
required and verified. Output must be fresh and separate from all inputs.

## Read the outputs

Open `index.html` in the output directory. It links each trajectory's data and
figures and displays the detail activity atlas and population readout.

| Artifact | Contents |
| --- | --- |
| `trajectory_XX_atlas_detail.png` / `.pdf` | Heading, body velocities, all neural rates grouped by population, and decoded/true home vectors; color scale shared within this trajectory |
| `trajectory_XX_atlas.png` / `.pdf` | Same panels with a common raw activity scale across all 12 trajectories |
| `trajectory_XX_readout.png` / `.pdf` | Signed population contributions to home-vector X/Y, decoder intercept, total prediction and truth |
| `trajectory_XX.npz` | Full-resolution arrays listed below; no padding or display downsampling |
| `activity.json` | Trajectory/protocol index, events, checkpoint identity, array conventions and readout reconstruction residuals |
| `neurons.csv`, `inputs.csv` | Original array-index order, body IDs/types; neuron table also gives decoder coefficients |
| `neuron_summary.csv` | Per-trajectory activity minimum/maximum/mean/initial/final, and each neuron's actual signed output contribution summaries |
| `population_summary.csv` | Per-trajectory population activity and actual contribution summaries |
| `current_checkpoint.pt`, `current_metadata.json` | Exact checkpoint bytes and captured run/circuit/history metadata |
| `manifest.json` | Status, code/input/artifact hashes, execution settings and immutable-input checks |

Trajectories follow prepared challenge order. The JSON index retains each
complete protocol, so opposite-history cases remain distinguishable even when
their family/kind labels match. Internal control transitions are marked with
gray lines. Figures sort units by population and body ID; arrays retain the
original circuit order given in the CSV files. This ordering is not anatomical.

The detail view makes small activity changes visible when the long-path case
sets a much larger global maximum. Neither view normalizes each neuron
individually. Rates are uncalibrated model units, not Hz. Quantitative summaries
use all recorded samples, regardless of figure resolution.

### Array schema

Each NPZ contains:

```text
time_s                    [T]
controls                  [T, 3]  heading degrees, forward mm/s, rightward mm/s
pfn                       [T, I]  actual encoded inputs passed to the model
rates                     [T, N]  dynamic neuron order in neurons.csv
prediction                [T, 2]  home x/y, mm
target                    [T, 2]  home x/y, mm
population_contribution   [T, P, 2]  population_order from activity.json
```

For update 1975, I=60, N=65 and P=4. Recording uses the production forward pass
at the prepared timestep (2.5 ms for this dataset). At time t the recorded state
precedes the update driven by input t; its effect appears in state t+dt. The
initial state is zero. Targets are checked against independent trajectory
geometry and never enter the network as inputs.

The population contribution is the observed population activity multiplied by
its columns of the frozen decoder. Summing these contributions and adding the
intercept **once** reproduces the original prediction, checked numerically on
every trajectory. Signed contributions can cancel. A changing population
contribution during a stable decoded pause is evidence that the readout is
concealing some activity changes; it does not establish which recurrent
pathways cause or maintain that behavior. Decoder signs are not synaptic signs.

## Scope

The follow-up [contribution, intervention and pause checks](cx_home_vector_activity_checks.md)
are available as `python -m fly_connectome.home_vector.analysis.activity_checks`.
They quantify questions raised by this atlas without refitting the model.

This implements the recorder, atlas and population readout milestone of the
[inspection plan](cx_home_vector_activity_inspection_plan.md). Input/recurrent
drive reconstruction, event-aligned zooms and pause-only pathway interventions
remain follow-ups. Existing challenges are developmental
probes, and descriptive activity patterns alone do not establish physiological
correspondence or causal necessity.
