# Replaying the first home-vector pilot analyses

For matched original-versus-hDeltaC failure metrics and overlays, see the
[comparison report](cx_home_vector_hdeltac_failure_comparison.md), which
documents the additional `analysis.compare` command and its exploratory
threshold definitions.

For the frozen-checkpoint speed, heading and body-direction sweep, see the
[straight-walk assessment](cx_home_vector_straight_sweep.md), including its
`analysis.straight` command and time-versus-distance interpretation.

The [long-walk training experiment](cx_home_vector_long_training.md) retains
the model while extending training and reserving longer evaluation paths.

These commands reproduce the analyses supporting the
[pilot interpretation](cx_home_vector_first_training_interpretation.md),
[home-vector claim assessment](cx_home_vector_claim_assessment.md), and
[network-change audit](cx_home_vector_network_changes.md). They do not fit
parameters, change the model, or require neuPrint access or a GPU.

## Required inputs and environment

The analysis code is versioned under `fly_connectome/home_vector/analysis/`.
The trained checkpoint, predictions, prepared task, historical audit outputs
and figures remain external experiment artifacts under ignored `results/`;
a fresh checkout does not contain them. Obtain the original bundles separately
or produce new ones using the [training walkthrough](cx_home_vector_walkthrough.md).
Newly trained runs need not reproduce the historical numerical findings.

Use the repository environment with `requirements-home-vector.txt` installed.
`predictions` uses NumPy and Matplotlib; `statistics` uses NumPy;
`network` additionally uses pandas and PyTorch; `perturbations` uses PyTorch
on the CPU. Run commands from the repository root or otherwise put the package
on Python's import path. No machine-specific interpreter location is embedded
in the tools.

Every command requires three explicit locations:

- Positional training-run directory: a complete `home_vector_training_v1`
  bundle, including its manifest and command-specific artifacts.
- `--source`: the matching complete prepared-task bundle. Its manifest hash
  must match the hash recorded in the run. This explicit override allows the
  bundles to be relocated without editing the old absolute paths in manifests.
- `--output`: a new directory, separate from both input bundles. Existing
  directories are refused, including previous failed audit directories.

All listed bundle artifacts are verified before analysis. All top-level input
files are hashed before and after analysis. Output manifests record the chosen
paths, source hashes, current analysis/model code hashes and library versions;
they mark an audit failed if its calculation or final integrity check fails.
Use the existing `verify` command to validate a completed audit's artifact hashes.
Hash verification establishes artifact identity, not biological validity.

## Commands

The paths below identify the original pilot. Replace them with the locations of
your bundles and use fresh output names for every replay.

```bash
python -m fly_connectome.home_vector.analysis predictions \
  results/gpu-initial-v1-anatomy-seed0 \
  --source results/cx-home-vector-data-20261030 \
  --output results/pilot-prediction-audit-NEW

python -m fly_connectome.home_vector.analysis statistics \
  results/gpu-initial-v1-anatomy-seed0 \
  --source results/cx-home-vector-data-20261030 \
  --output results/pilot-direction-statistics-NEW

python -m fly_connectome.home_vector.analysis perturbations \
  results/gpu-initial-v1-anatomy-seed0 \
  --source results/cx-home-vector-data-20261030 \
  --family challenge-pause-5 --threads 1 \
  --output results/pilot-perturbations-NEW

python -m fly_connectome.home_vector.analysis network \
  results/gpu-initial-v1-anatomy-seed0 \
  --source results/cx-home-vector-data-20261030 \
  --output results/pilot-network-audit-NEW

python -m fly_connectome.home_vector verify results/pilot-network-audit-NEW
```

| Command | Calculations and outputs |
| --- | --- |
| `predictions` | Recompute saved validation/test/challenge metrics, compare targets against an independent geometric integral, check exact split duplicates, calculate matched-history differences, write `audit.json` and diagnostic PNG/PDF. |
| `statistics` | Write `statistics.json`: direction and parallel/perpendicular error statistics for the test set, pooled across timestamps and grouped by trajectory kind. Undefined subsets are represented by JSON `null`. |
| `perturbations` | Replay a named pair of pause challenges unchanged, with state reset at pause onset, and with recurrence removed for the whole replay; write `perturbations.json`. |
| `network` | Verify initial coefficient reconstruction and checkpoint buffers, compare effective weights and intrinsic parameters, write `summary.json`, eight CSV tables/matrices, and two PNG/PDF figures. |

The prediction figure uses the original v1 challenge families (long 64-second
path, five-second opposite-history pause pair, square loop and matched world
paths). It is not an arbitrary-protocol plotter. Perturbation selection uses
family labels, checks identical stationary suffix controls and derives reset
timing from metadata rather than fixed trajectory indices. Both checkpoint
commands require a recurrent training run; recurrence removal is an acute
perturbation rather than a retrained control. They load `best.pt` on the CPU
and preserve its parameters. The network audit remains a comparison against
the current model's declared initialization; incompatible future architectures
must be handled explicitly rather than silently reinterpreted.

The direction thresholds remain descriptive choices from the original audit:
true distance at least 1 mm, angular error at most 15 degrees, and a separate
subset with vector error greater than 1 mm. These are correlated timestamp
statistics, not independent-sample confidence intervals or preregistered gates.

## Refactor validation

On 1 November 2026, all four commands were replayed on the unchanged original
pilot into new temporary output directories. Every report field matched its
historical counterpart, including 1,046 numeric values compared with absolute
and relative tolerances of `1e-12`. All eight network CSV files matched byte
for byte. Input hashes were checked before and after each audit. Original
scripts and outputs remain preserved in their historical directories; current
replays use the tracked modules above and record their own provenance.

Focused synthetic tests cover relocated bundles, source mismatch, fresh-output
enforcement, failed audits, input changes, prediction-archive alignment, known
geometric/directional errors and metadata-based pause selection:

```bash
python -m pytest -q tests/test_home_vector_analysis.py
```
