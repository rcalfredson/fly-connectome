# Identical probes from selected trajectory states

For signed presynaptic inputs underlying these activity differences, see the
[incoming-drive audit](cx_home_vector_incoming_drive.md).

This follow-up asks whether apparent differences in neuronal activity along
paths reflect an inherited activity offset, changed movement sensitivity, or
a changed heading/body-movement tuning pattern. It compares full saved network
states with the **same** subsequent probes. It does not train or refit anything.

## Five cases

All path numbers refer to the renamed
`results/cx-home-vector-balanced-u1975-activity-by-trajectory-20261106` archive.
Times identify the state before the input at that time is applied.

| Case | State pairs A / B | Focus neurons | Question |
| --- | --- | --- | --- |
| 1 | Paths 6 / 7 at 6 s, and again at 7.5 s | 142911, 17816, 20231 | Same world path; identical PFN inputs during 6–8 s. How much history dependence persists during that common segment? |
| 2 | Paths 12 / 13 at 2 s, and again at 7.5 s | 142911, 17816 | Forward/lateral traversals of the same curved world path. Do opposite activity contrasts reflect changed probe sensitivity? |
| 3 | Path 6 at 1.5 / 7.5 s | 17816, 142911, 20231 | Same instantaneous PFN inputs and equal time into each straight segment, but different earlier history and displacement. |
| 4 | Paths 2 / 3 at 4 s | 20231, 142911 | Different body motions along the same world path, with a substantial hDeltaC activity contrast. |
| 5 | Paths 12 / 13 at 4 s | 20951, 59176 | Does being nearly silent correspond to suppression of subsequent movement sensitivity? |

There are seven state-pair comparisons and thirteen unique states. Reused
states are probed once. Every state receives 64 movement probes (eight headings
× eight body directions, 1 mm/s) and eight stationary probes, matched by heading:
936 independent continuations in total. Default sampling is 2.5, 50, 250 and
1,000 ms. Every probe gets its own copy of the complete starting state.

The probe speed is held fixed even when a source trajectory used another speed.
Abrupt changes of heading occur equally in the moving and stationary controls.
Thus the experiment asks how histories affect responses to standardized new
inputs; it does not simply replay the original future path.

## Measurements

For each neuron, history state and probe time, form a map over heading and
body-centered movement direction:

```text
actual activity = moving-probe rate
change from start = moving-probe rate − saved starting rate
movement response = moving-probe rate − stationary-probe rate at the same heading
```

The stationary continuation starts from the exact same full-network state.
It includes heading-only responses and autonomous recurrent evolution.
Subtracting the saved starting rate alone cannot change a map's shape, because
it shifts every grid point by the same amount.

The following measurements describe the pair B relative to A:

| Measurement | Meaning |
| --- | --- |
| Raw mean shift | Average actual-activity difference across the grid; captures an overall activity offset. |
| Movement-response range A and B | Maximum minus minimum movement response; separates response strength from its pattern. |
| Range ratio B/A | Amplitude comparison; undefined if A is flat. |
| Pattern correlation | Pearson correlation over the grid, ignoring offset and positive gain. Undefined if either map is flat. |
| Fitted gain and offset | Least-squares approximation `map B = offset + gain × map A`. A negative gain is an inversion. |
| Residual RMS | Remaining difference after that fit; quantifies departures from an offset/gain explanation. Also reported divided by B's range. |
| Right-minus-left contrast | Response to rightward minus leftward walking at heading 90°, separately for A and B; tests one concrete preference ordering. |
| Sampled peaks and tied counts | Descriptive preferred grid points, retained in the all-neuron CSV; not continuous physiological preferences. |

A right/left ordering reversal is reported only when both contrasts exceed
`max(1e-6, 1% of the larger map range)` in magnitude; otherwise it is left
indeterminate. This is a descriptive threshold, not a significance test.
A flat map has a range of at most 1e-6 model units. Inspect amplitudes alongside
correlations: weak maps, large gains from tiny inputs, and normalized residuals
with small denominators can be misleading. The production replay uses float32;
very small differences may reflect arithmetic precision.

## Run and share

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fly-connectome-mpl \
python -m fly_connectome.home_vector.analysis.trajectory_tuning \
  --activity results/cx-home-vector-balanced-u1975-activity-by-trajectory-20261106 \
  --source results/cx-home-vector-hdeltac-long-data-20261104 \
  --output results/cx-home-vector-balanced-u1975-trajectory-history-checks-20261108 \
  --threads 1
```

Use a fresh output directory. The tool verifies input bundle hashes and neuron
ordering, loads the checkpoint archived with the trajectory data, and replays
every used path against its saved inputs, rates, targets and predictions before
selecting states. It supports renamed input directories without rewriting them.

- `index.html`: browsable results with per-case summary tables and figures.
- `report.md`: portable written report with the same measurement definitions
  and relative links to figures.
- `*_raw.png/pdf` and `*_movement_delta.png/pdf`: A, B and B−A maps at every
  probe time. A/B share one color scale across times within each figure;
  differences have their own symmetric scale. No per-map normalization.
- `focus_summary.csv`: offset, amplitude, correlation, fitted gain, residual,
  and right/left comparisons for the selected neurons at all sampling times.
- `all_neuron_comparisons.csv`: every dynamic neuron, all three response views,
  including sampled peaks and ties.
- `pair_checks.csv`: whether current PFN inputs match, target-vector difference
  and initial neural-state difference for each pair.
- `state_endpoints.csv`: source path/time/tick, controls, target and state norm.
- `path_XX_at_XXXXXXXXus.npz`: full probe rates and predictions, saved starting
  state, PFNs at onset, control indices and sampled response maps. Rates use
  probe-condition × time × neuron order; sampled maps use moving-condition ×
  sampled-time × neuron order. `neurons.csv` identifies neurons.
- `cases.json`, `probe_conditions.json`, `raw_paths.json`, checkpoint/metadata,
  `trajectory_tuning.json` and hashed `manifest.json`: archived protocols,
  checkpoint identity, replay verification, definitions and provenance.

Customize `--times`, `--speed`, `--angles` and `--chunk-size`. An optional
`--cases` JSON file has the format returned by `default_cases()` in the module;
cases specify labels, pairs of `[path_number, time_s]`, and focus neuron IDs.
Times must lie on the simulation time grid. All-neuron measurements are saved
regardless of focus selection.

## Scope of inference

These comparisons test history dependence **within one frozen trained model**.
Changed responses do not imply changing weights; rectification and recurrent
state can change responsiveness without structural changes. Stable maps do not
imply an absence of memory: history can appear as an activity offset instead.
Different source histories also differ in hidden state and sometimes final
displacement. This does not isolate home-vector memory from other history
effects, establish biological realism, or show that training created the effect.

## First completed analysis: 8 November 2026

The default command above completed for update 1975. All six source paths
replayed with exactly zero maximum absolute rate error. The results browser is
`results/cx-home-vector-balanced-u1975-trajectory-history-checks-20261108/index.html`.

For the **selected neurons in cases 1–4**, movement-response maps were very
similar between the paired states: at 250 ms all correlations exceeded 0.99999,
and modulation ranges differed by at most 0.215%. At 1 s all correlations
exceeded 0.99978, and range differences were at most 1.96%. Thus large observed
activity differences in these examples mostly coexist with preserved movement
sensitivity; they do not demonstrate substantial tuning remapping.

Case 5 provides a more informative exception. At 250 ms, comparing path 12's
state at 4 s (A) with path 13's state at 4 s (B):

| Neuron | Movement-response range A | Range B | Shape correlation | Residual RMS / B range after offset/gain fit |
| --- | ---: | ---: | ---: | ---: |
| hDeltaC 20951 | 0.8467 | 0.8063 | 0.9981 | 1.54% |
| vDeltaE 59176 | 1.7268 | 1.0701 | 0.8940 | 11.51% |

At 1 s, the corresponding correlations were 0.9617 and 0.9647, respectively.
These show history-dependent sensitivity and some departures from an
offset/gain-only explanation, particularly for 59176 at 250 ms. Both neurons
retained their sampled peak combinations across the two histories at these
times. No right/left ordering reversal meeting the stated contrast threshold
was reported for the focus comparisons; weak contrasts remain indeterminate.
These observations describe this checkpoint and these selected states, not an
invariance claim across all possible histories.
