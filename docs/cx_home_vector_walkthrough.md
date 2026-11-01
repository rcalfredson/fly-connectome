# CX home-vector capacity: initial code setup

Implemented 30 October 2026 following review of the
[experiment proposal](cx_home_vector_proposal.md). This is an initial executable
setup with engineering smoke runs, not a completed capacity experiment.
Existing simulation/extraction commands, source results and presentation exports
are preserved. No live query, credential access or commit is required.

## Circuit and one corrected total

Loading the saved matrices revealed an arithmetic error in the proposal's
original edge total. The nine internal projections have **478 positive body
pairs**, not 578. The 597 PFN input pairs were counted correctly. The selected
circuit therefore has 60 clamped PFNs, 45 dynamic neurons, **1,075 trainable edge
coefficients**, six intrinsic parameters and 92 decoder parameters:
**1,173 trainable scalars**. No population or connection was removed to obtain
this correction. The proposal now records the corrected totals.

`home_vector.data.assemble` calls the existing source verifiers, checks matching
dataset UUID/mutation/server, checks identities and matrix axes, retains all
selected bodies and loads only the declared projections. It writes explicit
projection counts, normalization denominators, source hashes and omissions.
No connection into a PFN is simulated; PFN→hDeltaG remains explicitly unqueried.

## Environments

Preparation and ridge baselines use the existing project environment. PyTorch
is optional and imported only by the training/model code. For a future unified
environment, the optional dependency file is:

```bash
python -m pip install -r requirements-home-vector.txt
```

That file includes the existing development requirements plus `torch==2.8.0`.
No package was installed or existing environment modified during setup. Locally,
the `fly-connectome` environment has the existing extraction dependencies and
no torch; `py3.13-torch` has torch 2.8.0, NumPy/pandas/pytest, and no neuPrint.
The prepared bundle allows training in the latter environment without importing
neuPrint or reading the original extraction files. CUDA was unavailable in the
original setup session; CPU runs are supported. After a reboot on the same day,
the loaded kernel module and userspace driver both report **580.178.04** and
PyTorch successfully initializes an **RTX 3060 with 12 GB memory**. The sandbox
still hides NVIDIA device nodes: GPU checks and the benchmark required execution
outside it. No packages or drivers were changed by this workflow.

## Prepare compact data offline

From the repository root in the existing project environment:

```bash
python -m fly_connectome.home_vector prepare \
  --output results/cx-home-vector-data-NEW
```

Defaults reuse the reviewed movement, downstream and vDeltaE input runs. Override
`--movement`, `--census` and `--inputs` to use other complete compatible saved
runs. Preparation uses no token or network access. Output must be a fresh path.

The bundle contains:

| File | Contents |
| --- | --- |
| `circuit.json` | Ordered IDs/types, fixed PFN angles, positive edges, raw counts and denominators |
| `trajectories.json` | Compact timestamped movement protocols, trajectory kinds and family IDs |
| `task.json` | dt, coordinates, training-only RMS length scale and split counts |
| `sources.json` | Source hashes, common dataset snapshot, selected projections and omissions |
| `legacy_weights.npy` | Exact saved PFN→hDeltaB effective coupling for the legacy frozen baseline |
| `manifest.json` | Completion/failure status, settings, code hashes and output hashes |

The standard preparation has 1,024 training, 128 validation and 256 test
trajectories plus 12 named challenge trajectories. A small engineering bundle
can be generated with `--train 8 --validation 4 --test 4`; it retains the full
anatomical circuit but is far too small for the proposed scientific comparison.

Movement is generated independently per family/split. Matched siblings share a
five-second stationary suffix, current inputs and elapsed time but have opposite
preceding displacements. Both members stay in the same split. Random trajectories
contain heading changes, lateral/backward translation and pauses. Challenge
protocols include matched 1/5/10-second pauses, reversals, a square loop,
32/64-second extrapolation and equivalent world motion using forward versus
sideways body movement.

The neural forward function receives only sampled PFN activity. Position/home
targets are integrated separately in the task layer. Targets and current rates
align before each subsequent held-input interval. No trajectory state is reset
at a segment boundary, pause or return home.

## Frozen and current-input ridge baselines

```bash
python -m fly_connectome.home_vector baselines \
  results/cx-home-vector-data-NEW \
  --output results/cx-home-vector-baselines-NEW
```

This fits three stateless affine decoders: current PFNs alone; the original
feedforward hDeltaB replay; and the new 45-cell network frozen at seed-0
initialization. Each decoder uses training-only feature standardization,
equal-trajectory fitting weights and validation-selected ridge from a fixed
four-value grid. Predictions have two Cartesian components. The command streams
features per trajectory instead of caching every PFN/rate trace.

Saved outputs include fitted weights/scales and Cartesian, magnitude, direction
and component-R² metrics. Aggregate normalized RMSE uses target energy over all
evaluated samples; it is distinct from the equal-trajectory training loss.
Near-home angles are omitted; near-zero predictions away from home count as
direction failures. The intercept is not ridge-penalized.

## Benchmark before selecting a training budget

In an environment with torch:

```bash
python -m fly_connectome.home_vector train \
  results/cx-home-vector-data-NEW \
  --output results/cx-home-vector-benchmark-NEW \
  --benchmark --updates 100 --batch-size 8 --max-seconds 120
```

`--benchmark` performs real gradient updates but does no checkpoint selection or
test evaluation, and labels its purpose as an engineering benchmark. The time
cap is checked between full-trajectory updates, so one update may overshoot it.
No temporal truncation is introduced to meet a wall-time limit.

Use measured time per update, batch lengths and hardware to choose the pilot
budget. A three-update batch-2 smoke benchmark took about 0.89 seconds/update on
this CPU session; this is a short, small-batch measurement and is not an ensemble
runtime prediction. A standard-data batch-8 benchmark is saved separately.
That three-update batch-8 run averaged 18.62 seconds/update (55.85 seconds
total), with substantial variation across updates while other tests were also
running. Budget from a longer isolated benchmark before launching an ensemble;
the short batch-2 result is too optimistic for the standard configuration.

After the reboot, the same standard-data, seed-0, three-update batch-8 benchmark
on CUDA averaged **1.84 seconds/update** (5.51 seconds total), versus 18.62 on the
earlier CPU run. This is about a 10× observed speed difference in these short
runs; the CPU measurement also overlapped other tests. The CUDA results are in
`results/cx-home-vector-gpu-batch8-benchmark-20261030`, with a complete, verified
manifest. A longer isolated benchmark is still needed for ensemble budgeting.

## A single training pilot

```bash
python -m fly_connectome.home_vector train \
  results/cx-home-vector-data-NEW \
  --output results/cx-home-vector-pilot-NEW \
  --seed 0 --updates 2000 --batch-size 8 --max-seconds 7200
```

The implementation uses the proposed signed per-edge coefficients, type-shared
bias/tau, fixed count normalization, exponential Euler at 2.5 ms, full-sequence
BPTT, Adam and gradient clipping. It applies parameter bounds after each update.
The affine decoder has no recurrent state or PFN skip input. Loss terms follow
the proposal, with equal trajectory weights and padding excluded from both
prediction and activity terms. All loss components are logged separately.

Checkpoints include model, optimizer, torch RNG, NumPy sampler state and update
number. `best.pt` is selected by validation error; `last.pt` records the final
update. Automatic resume is not implemented yet. `training_config.json` records
the optimizer, loss coefficients, bounds, initialization and numerical settings.

After training, evaluation reports validation/test/challenge metrics and
per-trajectory errors and saves Cartesian predictions/targets. Five-second
matched pauses have a separate late-suffix metric. A half-timestep validation
check compares rates and task RMSE at the same physical times. A failed
convergence check is explicitly marked as unvalidated performance, even if the
run completed without a numerical exception. `status: complete` means the
workflow finished and hashes were recorded; it is **not** the capacity gate.

To retrain without neural recurrence, add `--no-recurrence`. The intrinsic
short-time filtering remains; the recurrent contribution is zeroed in every
forward pass. To train on a null topology, add `--rewire-seed 0`. Rewiring uses
blockwise directed double-edge swaps, fixes self-loop locations, preserves
blockwise body degrees and count multisets, and reports mixing/retained-edge
fractions. It does not preserve per-body weighted strengths or spatial motifs.
The coefficients on inactive recurrent edges remain in the ablation parameter
table but cannot affect its activity.

Every experiment requires a fresh output directory. A post-creation exception
marks its manifest failed. Hashes of the prepared source are rechecked before
successful completion. All new runs remain under ignored `results/`.

```bash
python -m fly_connectome.home_vector verify results/cx-home-vector-pilot-NEW
```

Verification checks output integrity/completion, not scientific validity or a
fresh replay. Training works from the prepared bundle without the old source
directories; preparation/source-audit requires those directories.

## Engineering evidence and the next experiment

The local setup produced these separate runs:

- `results/cx-home-vector-data-20261030`: full default prepared task and circuit.
- `results/cx-home-vector-smoke-data-20261030`: 8/4/4 engineering task.
- `results/cx-home-vector-smoke-baselines-20261030`: three ridge controls on that
  tiny task; their scores are not scientific estimates.
- `results/cx-home-vector-cpu-benchmark-20261030`: three batch-2 updates.
- `results/cx-home-vector-cpu-batch8-benchmark-20261030`: standard-data batch-8
  benchmark; inspect `timing.json` and `history.json` for actual costs.
- `results/cx-home-vector-smoke-training-20261030`: three updates, checkpoint
  selection, all evaluation outputs and the half-step check. Rate discrepancy
  was 8.19e-6 and relative task-RMSE change 5.15e-5, passing the numerical gates.

On the full 105-body circuit, double-precision torch and the independent existing
`RateNetwork` agree to 7.11e-15 maximum absolute rate difference in the bounded
parity protocol. Tests also cover finite-difference gradients for recurrent/input
parameters, tau and decoder; loss weighting/padding; continuation/reset behavior;
current-input ambiguity in matched families; encoder compatibility; geometric
closure; null invariants; and artifact tampering/failure handling.

Run core tests in the project environment:

```bash
python -m pytest -q
```

Without torch, differentiable tests skip. In a torch environment run:

```bash
python -m pytest -q tests/test_home_vector_data.py \
  tests/test_home_vector_model.py tests/test_home_vector_run.py
```

Checks requiring neuPrint imports skip if that optional environment lacks the
existing extraction dependencies. The full-circuit cross-environment parity
comparison above was also executed directly during setup.

Setup verification: the complete existing-environment suite passed **398 tests**
(one optional torch module skipped; 14 existing dependency warnings). In the
separate torch environment, the three new test modules passed **20 tests** with
three extraction-dependency checks skipped. Source and new-run hash verifiers
also passed. These are engineering checks, not capacity or biological evidence.

Before a full capacity claim, complete the larger baseline/control comparison:
the nonlinear current-input/nuisance-time controls, broader moving-suffix and
non-square-loop challenges, input-to-world-velocity diagnostic, matched seeds
and null ensemble, confidence intervals and automatic application of the
predeclared capacity gates. Add state-reset perturbation reporting as well as
the already tested reset operation. The present random continuous-angle tasks
do not constitute a systematic held-out-angle grid. No ensemble, statistical
anatomical advantage or biological interpretation has been established.
