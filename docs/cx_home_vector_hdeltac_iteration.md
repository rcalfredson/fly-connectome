# hDeltaC expansion: anatomy, implementation and run commands

Prepared 1 November 2026. This iteration adds the full 20-neuron hDeltaC
population to the existing hDeltaB/vDeltaE/hDeltaG model. The original core
remains the default. At that date, engineering checks had passed; a full trained
hDeltaC experiment had not yet been run.

**Status update, 3 November 2026:** The first full hDeltaC training run has now
completed 2,000 optimization updates. The [comparison report](cx_home_vector_hdeltac_failure_comparison.md)
and [unified companion table](cx_home_vector_hdeltac_comparison_table.md)
document its results against the original model. The setup and engineering
checks below describe the original 1 November implementation.

## What changes

| Quantity | Original core | With hDeltaC |
| --- | ---: | ---: |
| Prescribed PFNd/PFNv input neurons | 60 | 60 |
| Dynamic neurons | 45 | 65 |
| Positive input-edge coefficients | 597 | 693 |
| Positive internal-edge coefficients | 478 | 933 |
| Total edge coefficients | 1,075 | 1,626 |
| Population biases and time constants | 3 + 3 | 4 + 4 |
| Affine decoder weights and intercepts | 90 + 2 | 130 + 2 |
| Total trainable scalars | 1,173 | 1,766 |

hDeltaC has its own shared bias and time constant, initialized to 0.1 as for
the existing populations. Its tuning is not prescribed: activity develops from
the incoming anatomical connections and recurrent dynamics. PFNs remain
clamped sources, with the same heading/translation mapping and firing curves.
The decoder reads all 65 dynamic rates.

The nine added directed projections were fetched in a bounded live extraction:

| Projection | Contacts | Positive neuron pairs |
| --- | ---: | ---: |
| hDeltaB → hDeltaC | 297 | 70 |
| vDeltaE → hDeltaC | 64 | 55 |
| hDeltaG → hDeltaC | 58 | 34 |
| hDeltaC → hDeltaB | 6 | 4 |
| hDeltaC → vDeltaE | 1,604 | 130 |
| hDeltaC → hDeltaG | 133 | 42 |
| hDeltaC → hDeltaC | 355 | 120 |
| PFNd → hDeltaC | 23 | 18 |
| PFNv → hDeltaC | 780 | 78 |
| **Added total** | **3,320** | **551** |

Thus the expansion includes reciprocal links between hDeltaC and every
existing dynamic population, plus within-hDeltaC recurrence. The 96 new PFN
edges use the existing PFN rates; no additional physical measurements or
learned input encoder are introduced. Self-connections are included where
reported. These are standard contact counts, not physiological efficacies.

## What stays fixed

The original 45 dynamic neuron IDs and all 1,075 edges retain their order and
values. hDeltaC neurons and new edges are appended. The existing seeded
initialization rule draws by first occurrence of each nonempty projection;
appending new blocks preserves all old coefficient draws for a given seed.
New recurrent blocks receive subsequent draws from that same generator;
new PFN-edge coefficients start at 1. No trained parameters are transferred
from the previous pilot: the proposed comparison trains from initialization.

The rate equation, ReLU, initial state, time step, parameter bounds, loss
coefficients, optimizer, checkpoint-selection behavior and training defaults
are unchanged. Normalization remains contacts divided by mean projection input
across the complete selected target population, including zero receivers.
The trajectory generator, seed and split sizes are unchanged; preparation
reproduced the original trajectory/task files and legacy weights byte for byte.
The fixed training length scale remains 2.0450918869 mm.

The existing omissions remain explicit, including unqueried PFNd/PFNv → hDeltaG
edges and all feedback into clamped PFNs. This is an hDeltaC expansion of the
declared boundary, not a complete induced CX graph. Existing normalization and
physiological-sign limitations are retained for this controlled comparison.

## Anatomy provenance

`results/cx-home-vector-hdeltac-anatomy-20261101` is a complete, verified new
bundle. Extraction used two disjoint connection queries: all selected sources
into hDeltaC (including hDeltaC itself), and hDeltaC into the original 45
dynamic neurons. Exact full-population annotations were checked against the
saved core and the census's 20 hDeltaC IDs. Dataset UUID, mutation ID and edit
metadata had to agree with the original census before and after extraction.
No ROI, status, cropped or confidence filters were introduced.

The two hDeltaC/vDeltaE projections matched their saved matrices exactly;
hDeltaB → hDeltaC matched the existing complete hDeltaB outgoing census.
Every projection has raw pair counts and a full matrix, including explicit
zero blocks if a successful query returns none. Failed queries create failed
bundles and cannot be used as evidence of absent edges. Preparation verifies
raw/matrix agreement, summaries, hashes, core identity and snapshot identity.

The original sources and trained results were preserved. New result bundles
remain ignored local artifacts. Source paths can be supplied explicitly.

## Commands

The complete expanded prepared task is already available locally at
`results/cx-home-vector-hdeltac-data-20261101`. To repeat anatomy extraction
and preparation in new directories, use the normal project environment with
neuPrint authentication configured locally:

```bash
python -m fly_connectome.home_vector.hdeltac \
  --source results/cx-home-vector-data-20261030 \
  --census results/hdeltab-extension-census-final-20261024 \
  --output results/cx-home-vector-hdeltac-anatomy-NEW

python -m fly_connectome.home_vector prepare \
  --hdeltac results/cx-home-vector-hdeltac-anatomy-NEW \
  --output results/cx-home-vector-hdeltac-data-NEW
```

Omitting `--hdeltac` prepares the original core. Use the environment with
PyTorch installed for the following commands. The full-data ridge baseline
fits its own decoder and does not initialize subsequent network training:

```bash
python -m fly_connectome.home_vector baselines \
  results/cx-home-vector-hdeltac-data-20261101 \
  --output results/cx-home-vector-hdeltac-baselines-NEW

python -m fly_connectome.home_vector train \
  results/cx-home-vector-hdeltac-data-20261101 \
  --output results/cx-home-vector-hdeltac-anatomy-seed0-NEW \
  --device cuda --seed 0 --updates 2000 --batch-size 8 \
  --threads 1 --validate-every 50 --max-seconds 7200

python -m fly_connectome.home_vector verify \
  results/cx-home-vector-hdeltac-anatomy-seed0-NEW
```

The [analysis replay commands](cx_home_vector_analysis_walkthrough.md) also
accept the expanded prepared task and training run. The network-change audit
now derives its intrinsic-parameter labels and plot boundaries from the
selected circuit, preserving the original three-population checkpoint format.

## Verification so far

- Original checkpoint loaded without conversion. All original edges,
  normalization values and initial coefficients matched; coefficient checks
  covered seeds 0, 1, 12 and 99, including the saved seed-0 initialization.
- Original trajectory/task/legacy files matched byte for byte. The original
  network-change audit was replayed as a backward-compatibility check.
- Synthetic checks cover four-population NumPy/PyTorch agreement, gradients
  into hDeltaC edges/intrinsics/decoder weights, recurrence ablation, complete
  target normalization, semantic tampering and failed-query handling.
- The full repository suite passed: 408 tests, three optional-dependency skips.
  Focused PyTorch checks passed: 31 tests, four optional-dependency skips.
- A small 8/4/4-trajectory ridge smoke run completed in
  `results/cx-home-vector-hdeltac-smoke-baselines-20261101`; this checks the
  expanded frozen-dynamics path, not full-data baseline performance.
- A two-update CPU engineering run completed in
  `results/cx-home-vector-hdeltac-cpu-smoke-20261101`.
- A three-update CUDA benchmark at batch size 8 completed in
  `results/cx-home-vector-hdeltac-gpu-benchmark-20261101`: 6.41 seconds total,
  about 2.14 seconds per update. This short measurement includes startup
  effects and does not establish convergence or a reliable full-run duration.

Full training and the long-path, closure, matched-body-motion and pause-memory
evaluations are still needed to assess whether hDeltaC improves the result.
