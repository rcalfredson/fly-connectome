# Activity-analysis review and proposed commits

Reviewed November 9, 2026. No files have been staged or committed by the agent.

## Assessment

No blocking correctness issues were found in the activity-analysis changes.
Retain the tools as reusable research diagnostics, including the tuning probes:
a hypothesis that was not supported does not make its measurement tool useless.
They archive all-neuron data, preserve model identity and original inputs, and
provide useful controls for future mechanistic work. The new model hook only
clamps units during explicit no-gradient interventions; normal training and
checkpoint parameter keys remain unchanged.

These tools establish different things: observed trajectories and signed drive
accounting, effects of explicit interventions, or responses to matched probes.
None of those alone establishes physiological realism, what learning created,
or that every history-dependent state difference stores the home vector.
The PFN mapping is an overlapping weighted encoding, not one-to-one wiring.
Same-world-path cases can contain history information irrelevant to the task.
Those distinctions are explicit in the new-thread prompt.

Review cleanups update the historical inspection plan to acknowledge completed
drive/tuning tools, identify the renamed per-trajectory archive separately from
the extended-restart report, and clarify the input atlas's interpretation.
Existing completed result bundles were not modified.

## Validation

- Torch environment `/home/tracking/miniconda3/envs/py3.13-torch/bin/python`:
  `pytest -q tests/test_home_vector*.py` — **175 passed, 9 skipped**.
  This includes the new recorders, perturbations, tuning, drive accounting and
  existing model/training/compilation regressions.
- Project environment `/home/tracking/miniconda3/envs/fly-connectome/bin/python`:
  `pytest -q` — **424 passed, 27 skipped**. This environment has the neuPrint
  and dotenv dependencies but lacks Torch, so it does not replace the Torch run.
- Full-suite collection in the Torch-only environment fails because neuPrint
  and python-dotenv are unavailable there. No dependency changes were made.
- Both runs used `OPENBLAS_NUM_THREADS=1` and
  `MPLCONFIGDIR=/tmp/fly-connectome-mpl`. Optional dependency/device skips and
  dependency warnings remain; this review does not claim fresh CUDA execution.
- `git diff --check` passes. Saved artifact hashes verify for the activity atlas,
  renamed per-trajectory archive, PFN atlas, initialization comparison, simple
  history probes, selected-trajectory probes and incoming-drive audit.

## Proposed commit sequence

These boundaries are ordered by code dependencies. Keep documentation together
in the last commit so no manual splitting of README or cross-reference hunks
is necessary. No generated `results/` files or literature PDFs belong in this
series.

### 1. `feat(home-vector): record and probe frozen network activity`

Include these complete files:

```text
fly_connectome/home_vector/model.py
fly_connectome/home_vector/analysis/checkpoint.py
fly_connectome/home_vector/analysis/activity.py
fly_connectome/home_vector/analysis/activity_plots.py
fly_connectome/home_vector/analysis/activity_summary.py
fly_connectome/home_vector/analysis/activity_checks.py
fly_connectome/home_vector/analysis/activity_checks_plots.py
fly_connectome/home_vector/analysis/activity_paths.py
fly_connectome/home_vector/analysis/activity_paths_plots.py
tests/test_home_vector_activity.py
tests/test_home_vector_activity_checks.py
tests/test_home_vector_activity_paths.py
```

This supplies the recorder/readout atlas, snapshot support, opt-in silencing,
contribution/robustness checks and mixed/curved/restart probes. The small model
change belongs with the intervention code that uses and tests it.

### 2. `feat(home-vector): compare tuning across movement histories`

```text
fly_connectome/home_vector/analysis/input_tuning.py
fly_connectome/home_vector/analysis/history_tuning.py
fly_connectome/home_vector/analysis/trajectory_tuning.py
fly_connectome/home_vector/analysis/trajectory_tuning_plots.py
tests/test_home_vector_input_tuning.py
tests/test_home_vector_history_tuning.py
tests/test_home_vector_trajectory_tuning.py
```

This adds initial input-mixture atlases, standardized history probes and probes
from selected saved states, with separate offset/amplitude/shape metrics and
paired stationary controls. It depends on the recorder and movement helpers in
commit 1. The initial atlas is a postprocessor for a completed onset bundle;
its documented generation recipe is archived with that bundle.

### 3. `feat(home-vector): audit presynaptic drive in saved trajectories`

```text
fly_connectome/home_vector/analysis/incoming_drive.py
fly_connectome/home_vector/analysis/incoming_drive_plots.py
tests/test_home_vector_incoming_drive.py
```

This accounts for signed PFN and recurrent inputs, per-population/per-partner
changes, rectification and the separate own-state update term. It reconstructs
actual production updates and consumes the completed probes from commit 2.

### 4. `docs(home-vector): document activity tools and reset history analysis`

```text
README.md
docs/cx_home_vector_checkpoint_evaluation.md
docs/cx_home_vector_activity.md
docs/cx_home_vector_activity_checks.md
docs/cx_home_vector_activity_inspection_plan.md
docs/cx_home_vector_raw_activity_and_restart.md
docs/cx_home_vector_input_tuning.md
docs/cx_home_vector_history_tuning.md
docs/cx_home_vector_trajectory_tuning.md
docs/cx_home_vector_incoming_drive.md
docs/cx_home_vector_history_reboot_prompt.md
docs/cx_home_vector_analysis_review_20261109.md
```

This completes the CLI/output guides, scoped results, status corrections,
reboot prompt and this review. All documentation cross-references then refer to
implemented tools from the preceding commits.

## New-thread handoff

Copy [the starter prompt](cx_home_vector_history_reboot_prompt.md) into a new
thread. It asks for a first-principles explanation and a small, ordered plan
before further implementation. It preserves the frozen model, exact data paths,
controls and useful prior findings without requiring the prior conversation.

Leave `A Neuronal Circuit Motif for Leaky Vector Integration.pdf` outside these
commits. It was not part of the code review and is unrelated to the proposed
source/documentation boundaries.
