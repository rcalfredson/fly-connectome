# Frozen-checkpoint straight-walk sweep

Assessed 3 November 2026. Results below compare the same selected checkpoints
as the [hDeltaC failure report](cx_home_vector_hdeltac_failure_comparison.md).
No training was performed.

## Results

The completed bundle is
`results/hdeltac-seed0-straight-sweep-20261103`:
[comparison figure](../results/hdeltac-seed0-straight-sweep-20261103/straight_sweep.png),
[PDF](../results/hdeltac-seed0-straight-sweep-20261103/straight_sweep.pdf),
[full numerical table](../results/hdeltac-seed0-straight-sweep-20261103/straight_sweep.csv).
These generated artifacts remain local and ignored; the numerical summaries
below are retained in Git.

### Speed changes both the onset time and the eventual decoded magnitude

For forward walking at heading 45 degrees, onset means sustained 10% magnitude
underestimation beyond 2 mm. Along-track values are signed projections onto
the correct home direction, rather than vector magnitudes.

| Speed (mm/s) | Original onset (s) | hDeltaC onset (s) | Original onset (mm) | hDeltaC onset (mm) | Original along-track value at 64 s (mm) | hDeltaC along-track value at 64 s (mm) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 29.10 | 27.44 | 7.28 | 6.86 | 7.00 | 6.62 |
| 0.50 | 24.74 | 13.33 | 12.37 | 6.67 | 13.92 | 8.00 |
| 1.00 | 15.15 | 7.45 | 15.15 | 7.45 | 20.59 | 10.69 |
| 2.00 | 8.69 | 4.55 | 17.39 | 9.11 | 32.14 | 16.16 |

The expanded model's onset distances cluster more closely than its onset
times. This contradicts a simple speed-independent time-to-failure account
for this probe, and is more consistent with a limited accumulated range.
However, it is not a fixed hard distance cap: its late decoded values increase
with speed. A simple linear leaky integrator with a speed-independent time
constant and proportional gain would have the same relative-error curve
against time at all speeds; these observations do not follow that simple
description either. Nonlinear, input-dependent recurrent dynamics remain a
plausible explanation, without being uniquely identified by this experiment.

At 1 mm/s, the final-ten-second along-track slope is only 0.0176 mm/s for the
original and 0.00354 mm/s for hDeltaC, versus the correct 1 mm/s. Hidden-state
change over that interval is 0.736% and 0.168%, respectively. Both the internal
state and decoded output are nearly settled under this constant drive. The
expanded model's late direction error is just 0.31 degrees at this heading:
the principal endpoint failure is magnitude, not orientation. These finite-time
observations do not prove an exact asymptote or exclude a useful decoder refit.

### The hDeltaC deficit is strongly heading dependent

At 1 mm/s with forward body motion:

| Heading (degrees) | Original magnitude-underestimation onset (mm) | hDeltaC onset (mm) |
| ---: | ---: | ---: |
| 0 | 3.85 | 5.58 |
| 45 | 15.15 | 7.45 |
| 90 | 15.14 | 9.39 |
| 135 | 8.84 | 12.06 |
| 180 | 7.81 | 15.77 |
| 225 | 10.36 | 12.53 |
| 270 | 12.16 | 7.94 |
| 315 | 2.85 | 5.96 |

Thus the earlier 45-degree challenge does not establish uniformly worse
long-distance performance after adding hDeltaC. Its onset is later at five
of eight sampled headings and earlier at three. These eight conditions are
descriptive probes of two fits, not independent replicates of an architecture
effect. Magnitude alone can also hide angular error: for example, hDeltaC's
64-second forward-walk direction error at heading 135 degrees is 36.58 degrees.

### Identical world motion still yields different predictions

For a 1 mm/s world path at 45 degrees, hDeltaC's 64-second along-track value
ranges from 8.82 to 14.25 mm across the ten sampled body directions, despite
the identical 64 mm target. The original range is 13.92 to 20.59 mm. All ten
expanded-model endpoint vector errors exceed the corresponding original
errors in this long-path set. This does not negate the earlier short-path
improvements; it exposes their limited duration/range of generalization.

The full-vector threshold is deliberately stricter than magnitude alone.
For forward motion at heading 45 degrees and 1 mm/s, both models already
exceed 10% vector error at the first eligible distance of 2 mm, even though
sustained magnitude underestimation begins at 15.15 and 7.45 mm. This can
include angular error, overestimation and transient error. A first breach is
not a claim of permanent failure, and a magnitude-onset distance should not
be described as a fully accurate tracking range.

### What this narrows down

The reduced range in the original 45-degree probe is reproducible, but is
part of a direction- and speed-dependent learned response rather than a
universal 8 mm limit. Internal settling under constant drive supports
investigating the recurrent operating states alongside decoder calibration.
Training has only short constant-control bouts, so poor generalization to
these long inputs remains a credible contributor. The sweep does not separate
that explanation from architecture or optimization effects; it introduces no
training intervention.

The oblique 127- and 233-degree trials explore the known encoder ambiguity
sectors, but do not constitute an identical-PFN-input counterexample: heading
changes in the matched-world set, and these individual trajectories need not
have duplicate PFN traces. Input nonidentifiability is established separately
in the input-boundary assessment. It cannot by itself explain the forward
45-degree plateau.

The most useful next inspection is the time course of signed parallel and
transverse errors at selected headings, alongside population activity. That
would distinguish early calibration errors from the later loss of incremental
integration before deciding whether a training-distribution change, a recurrent
intervention or a decoder refit is the appropriate follow-up.

## Protocol and reproduction

This experiment replays the original and hDeltaC selected checkpoints without
optimization, decoder refitting or altered PFN encoding. Each trial starts at
zero dynamic rates and holds movement controls constant for 64 seconds at the
original 0.0025-second time step.

- Speeds: 0.25, 0.5, 1 and 2 mm/s, reaching 16, 32, 64 and 128 mm respectively.
- Headings: 0 through 315 degrees at 45-degree intervals.
- Body walking directions: 0, 45, 90, 127, 135, 180, 225, 233, 270 and 315 degrees.
  Zero means forward, 90 means rightward, and 180 means backward. The 127- and
  233-degree probes fall within the encoder ambiguity sectors identified in the
  [input-boundary assessment](cx_home_vector_input_boundary_assessment.md).
- An additional matched-world-path set adjusts heading to `45 - body_direction`
  modulo 360 degrees, producing identical world velocity for every body motion
  at each speed. The factorial grid contributes 320 trials and the matched set
  another 40 per checkpoint. Some controls repeat across these two sets.

Run from the repository root in the PyTorch analysis environment:

```bash
OPENBLAS_NUM_THREADS=1 python -m fly_connectome.home_vector.analysis.straight \
  --reference-run results/gpu-initial-v1-anatomy-seed0 \
  --reference-source results/cx-home-vector-data-20261030 \
  --run results/cx-home-vector-hdeltac-anatomy-seed0-2026-11-01 \
  --source results/cx-home-vector-hdeltac-data-20261101 \
  --output results/hdeltac-seed0-straight-sweep-NEW
```

`--duration` and `--speeds` override the defaults. Duration must be at least
10 seconds and fall on the task time grid. Outputs are a JSON report, a CSV
with one row per trial/checkpoint, full-time-resolution decoded vectors and
hidden-state norms in NPZ archives, a four-panel PNG/PDF, and a manifest.
Trace order matches the report's condition list; time is sample index times
the recorded task time step. Targets are negative elapsed time times world
velocity, derived from each row's heading, speed and body direction.

The CPU implementation batches constant drives using the model's exact update
rule in NumPy float32. Synthetic tests compare it with both Torch and the
independent NumPy rollout. Each real checkpoint is also checked against its
saved 64-second challenge prediction, with a maximum allowed absolute
component difference below 0.002 mm. Input bundle hashes are verified before
and after the sweep, and output hashes are recorded in the manifest. Existing
output directories are refused.

## Measurement definitions

Two threshold metrics use the first sustained breach lasting at least 0.5 s,
considering only true displacements of at least 2 mm:

- Magnitude underestimation: predicted magnitude below 90% of true magnitude.
- Full-vector error: Euclidean prediction error above 10% of true magnitude.

Each onset is reported as both elapsed time and travelled distance. A missing
onset means no qualifying interval was observed within the trial; it does not
establish indefinitely accurate integration. A breach at 2 mm can reflect an
earlier failure excluded by the minimum-distance rule. These are descriptive
thresholds, not independent trial statistics or preregistered criteria.

Along-track prediction is the signed projection of the decoded vector onto
the true home direction. Endpoint gain divides this projection by true
distance; ideal is 1, and a negative value points away from home. Endpoint
direction error and transverse error are recorded separately. Late gain is
the fitted along-track slope over the final ten seconds divided by walking
speed; ideal integration gives 1 and a settled prediction gives approximately
0. The hidden-state change is the norm of the final state minus its state ten
seconds earlier, divided by the final-state norm. The final decoded speed
uses the instantaneous rate-equation derivative rather than the fitted slope.

Comparing forward-walk curves against both time and distance tests whether a
common millimetre ceiling explains the failure. A plateau proportional to
speed is compatible with leaky temporal integration. Nonlinear inputs and
state-dependent recurrence can make the effective relaxation depend on speed,
so this sweep does not fit a universal memory time constant or isolate the
decoder from the dynamic network. Both the very long bouts and some travelled
distances exceed the training distribution. Only one checkpoint per
architecture is assessed.

## Validation of this run

The optimized CPU replay differs from the saved original challenge by at most
0.00000859 mm per component, and from the saved hDeltaC challenge by at most
0.00000525 mm. All input hashes remained unchanged, and the completed output
bundle passes the existing `home_vector verify` command. Eight focused tests
passed across the straight-sweep and existing comparison modules, including
world-path equivalence, directional error with correct magnitude, threshold
timing and censoring, and independent rollout agreement.
