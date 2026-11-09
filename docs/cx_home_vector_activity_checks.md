# Frozen activity checks: contributions, interventions and pause stability

Implemented and run November 6, 2026 on the balanced hDeltaC model selected at
**update 1975**. The checkpoint, decoder and PFN mapping remained fixed. These
checks investigate the fitted solution; they do not establish physiological
realism or select a new model.

For the separate question of which units have the highest **raw rates**, and
whether movement after a pause reduces neural-state norms, see the
[raw activity and restart probes](cx_home_vector_raw_activity_and_restart.md).

Open the [completed results index](../results/cx-home-vector-balanced-u1975-mechanism-checks-20261106/index.html)
for three PNG/PDF figures and the underlying tables. The input is the
[full activity atlas](cx_home_vector_activity.md).

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fly-connectome-mpl \
python -m fly_connectome.home_vector.analysis.activity_checks \
  --activity results/cx-home-vector-balanced-u1975-activity-20261106 \
  --source results/cx-home-vector-hdeltac-long-data-20261104 \
  --output results/cx-home-vector-balanced-u1975-mechanism-checks-NEW \
  --threads 1
```

Use a fresh output label. CPU execution leaves CUDA available. The activity/source
bundles are verified and checked for changes. Checkpoint bytes, source metadata,
code hashes, probe settings and results are saved in a hashed completion bundle.
No training or decoder refitting occurs.

Defaults can be changed with `--distance-mm`, `--angles`, `--speeds`,
`--pause-seconds`, `--jitter-fractions`, `--repeats`, `--seed`, and
`--focus-body-id`. `--chunk-size` bounds memory. Durations must lie on the prepared
simulation grid.

## Direct contribution across movement conditions

The grid contains **192** straight paths: eight global headings and eight
body-centered movement directions in 45-degree increments, at 0.5, 1 and 2 mm/s.
Each travels **8 mm**, so duration varies with speed. Forward/rightward velocities
are `speed*cos(body_direction)` and `speed*sin(body_direction)`.

```text
contribution_RMS_j = sqrt(mean_time(norm(D[:,j] * rate_j(t))^2))
share_j = contribution_RMS_j / sum_over_neurons(contribution_RMS)
```

Shares describe the size of direct decoder terms, not fractions of explained
behavior, information or net output; opposing terms can cancel. All neuron
ranks/contributions, maximum activity and late activity slope per millimeter
are saved alongside each condition's task error and exact protocol.

Neuron **142911** ranks **4th–64th** across conditions; its share ranges from
**0.014% to 5.39%** and it never ranks first. Across all conditions, the largest
individual share is **5.66%–13.82%**. Neurons 17816 (hDeltaB) and 28258 (hDeltaG)
lead in 123 and 64 conditions respectively; other units lead the remaining five.
Thus conspicuous activity does not amount to direct decoder dominance. These
measurements alone do not determine recurrent necessity or redundancy.

## Readout removal versus silencing

For **all 65 units on all 12 existing challenges**, compare:

1. **Readout removed:** subtract `D[:,j] * rate_j(t)` from the original prediction;
   the neural trajectory is untouched and the intercept remains.
2. **Silenced throughout:** clamp unit j to zero at every recorded state, before
   it can influence the next recurrent update. Parameters remain fixed.

This gives 1,560 intervention/case rows. The production model's optional
`silence_mask` is accepted only in no-gradient evaluation. Its default path
retains the original behavior. Masks may differ across batch members. Every
chunk includes an unmasked sham checked against the baseline. Numerical guard
failures are recorded without a finite performance score; none occurred here.

For neuron 142911, whole-trajectory Cartesian vector RMSEs are:

| Challenge | Baseline (mm) | Remove readout (mm) | Silence throughout (mm) |
| --- | ---: | ---: | ---: |
| Backward walk + 5 s pause | 0.0507 | 0.0626 | 0.5893 |
| Forward walk + 5 s pause | 0.0902 | 0.2788 | 2.0696 |
| 64-second straight path | 7.2442 | 6.3133 | 46.3225 |
| Square loop | 0.1669 | 0.3212 | 2.9198 |

This supports a role in recurrent computation despite its small direct readout
term. It does not identify a unique storage neuron or separate encoding from
retention: the clamp acts throughout the trial. Removing a term can improve a
particular task error because baseline errors and contributions have signs;
this does not establish that the neuron is globally harmful. All-unit controls
and per-case task errors are saved. The scatter instead shows equal-trajectory
RMS prediction change from baseline, whose absolute scale can be dominated by
long paths; it is a different measure from the task RMSEs above.

## Extended pauses and small state perturbations

Start from each saved five-second-pause challenge's state at pause onset, after
forward or backward walking. Continue the **same stationary, heading-dependent
PFN input** for 60 seconds; the correct home vector remains fixed. The first
five seconds reproduce the original replay within the 0.002 mm maximum-component
tolerance.

Use the fixed decoder's SVD to split state changes from pause onset into
orthogonal row-space and null-space components. The decoder has rank two, with
63 instantaneous output-null dimensions. Observed state changes lie predominantly
in that null space:

| History | Pause | Total state-change norm | Decoder-visible norm | Output drift (mm) | True-vector error (mm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Backward | 5 s | 25.654 | 0.129 | 0.0654 | 0.0271 |
| Backward | 60 s | 218.551 | 1.565 | 1.0359 | 1.0162 |
| Forward | 5 s | 70.665 | 0.248 | 0.1258 | 0.1411 |
| Forward | 60 s | 225.567 | 3.262 | 1.8707 | 1.8675 |

Neural norms are in model units, not mm. Hidden state changes can influence
future visible activity through recurrence. These results support short-pause
cancellation and a limit to sustained retention, not a stationary neural state.

Perturb the pause-onset state once using seeded multiplicative Gaussian jitter,
rescaled to **0.1%, 1% or 5% of the initial state's L2 norm**, with four draws per
amplitude/history. Keep rates nonnegative and report the actual applied norm.
This samples active-unit changes, not all possible perturbation directions.
Include an unchanged sham and a complete state reset per history: 28 variants.
Matched shams in every chunk control for batch-shape rounding; their maximum
prediction difference across chunks was zero in this run.

At 1% jitter, additional output differences from the matched unperturbed pause
after 60 s range from **0.0267–0.0509 mm** after backward walking and
**0.0205–0.0822 mm** after forward walking. At 5%, the combined range is
**0.0642–0.5030 mm**. These are separate from baseline drift. The sampled small
perturbations do not cause catastrophic failure, although some differences
amplify. Jitter already changes decoder-visible activity at onset, so later
differences alone do not prove corruption of an initially unchanged home vector.
No continuous noise or pause-only pathway suppression was tested.

## Outputs and limits

- `contribution_sweep.png/pdf`, `sweep_conditions.csv`, `sweep_neurons.csv` and
  `sweep_protocols.json`: direct contributions, condition errors and exact probes.
- `neuron_interventions.png/pdf`, `neuron_interventions.csv`: all-unit comparisons
  and guard status. In `trajectory_XX_silencing.npz`, prediction index 0 is the
  baseline; indices 1..N follow the saved `body_ids` order.
- `pause_stability.png/pdf`, `pause_interventions.csv`, `pause_details.json`:
  drift and perturbation metadata. `trajectory_XX_pause.npz` stores baseline
  full neural rates, true/predicted vectors, state-space norms, variant initial
  states/predictions and paired differences from shams. Variant order follows
  `pause_details.json`; time is relative to pause onset.
- `diagnostics.json`, `manifest.json`, copied checkpoint/metadata and neuron
  identities: definitions, settings, completion status and reproducibility.

The present interpretation is: direct readout is distributed, 142911 participates
in recurrent computation, and short-pause output stability conceals state changes
that eventually accompany substantial drift. Jitter results narrow the robustness
concern rather than proving extreme fragility. Conclusions concern one frozen
seed and this probe/intervention set. No physiological correspondence, unique
memory mechanism, new model selection or new half-step convergence gate is
established here.
