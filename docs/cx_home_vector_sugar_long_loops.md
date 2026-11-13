# Sugar suppression over sixteen circular loops

The frozen long-loop comparison is archived at
`results/cx-sugar-long-loops-20261112`. Its `report.md`, `loop_metrics.csv`,
four path plots, and full-resolution NPZ arrays preserve the results.
The analysis is `fly_connectome/home_vector/analysis/sugar_long_loops.py`.

## Conditions

The exact archived 8-second circular movements for paths 8/9 and 14/15 repeat
16 times (128 seconds). Each trial starts at zero once. There are no pauses,
geometric resets, readout refits, or changes to the original anchor. PFN inputs
repeat with the movement. The recurrent parameters remain those optimized
under the original dynamics. The decoder-only comparison supplies the fixed
refitted readouts; the original frozen decoder is also scored.

- `original_refit`: original dynamics with its previously refitted decoder.
- `sugar_single_refit`: sugar dynamics, one delivery at 8 seconds, using its
  previously refitted decoder.
- `sugar_repeated_refit`: the same sugar dynamics and decoder, with scheduled
  deliveries at 8, 16, ..., 120 seconds. No delivery occurs at the final endpoint.

The single and repeated conditions differ only in delivery schedule. The gate
combines event kernels by maximum, not addition. With a 300-second recovery
constant, a single event still has a substantial effect at 128 seconds;
repeated events maintain suppression closer to its peak. Scheduled delivery
every eight seconds is an experimental assumption, distinct from the derived
dataset's environmental patch and 30-second minimum-delivery interval. Neither
schedule identifies how often a real fly would consume sugar.

## Results

The table uses each dynamics' own previously refitted readout. Closure errors
are absolute distances from the correct zero home vector at that return.

| Path | Original: return 2 (mm) | Repeated sugar: return 2 (mm) | Original: return 16 (mm) | Single sugar: return 16 (mm) | Repeated sugar: return 16 (mm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.764 | 1.544 | 34.018 | 1.001 | 1.805 |
| 9 | 0.239 | 1.474 | 21.532 | 1.000 | 1.779 |
| 14 | 0.599 | 1.608 | 32.253 | 2.299 | 1.362 |
| 15 | 0.263 | 1.338 | 19.359 | 1.921 | 1.178 |

Thus the disadvantage after two loops does reverse over these longer runs.
At loop 16, original per-loop RMSE is 17.527–30.764 mm, compared with
1.118–1.603 mm for repeated sugar. The largest activity within that loop is
543.495–694.859 in original dynamics and 57.352–85.239 under repeated sugar.
Repeated-sugar trajectories settle toward a repeating activity pattern over
this interval; this is not a proof of asymptotic boundedness.

The single-delivery condition changes more gradually as its gate recovers.
It gives lower final closure error than repeated delivery on paths 8/9 and
higher error on paths 14/15. More frequent delivery is not uniformly better.

Lower long-duration error does not alone establish accurate integration. A
constant zero-vector predictor has per-loop RMSE 1.80063 mm on all four
circles. The repeated-sugar readout does better, but its 1.118–1.603 mm error
remains substantial. Activity stabilization, useful within-loop tracking, and
accuracy at return are separate properties. The original circuit has no upper
activity cap, and activity units are not physiological firing rates.

These are familiar diagnostic paths, extending beyond the decoder dataset's
durations. They do not select response constants or weights and do not replace
fresh held-out assessment. The previously trained recurrent weights remain a
potential source of disadvantage for the sugar dynamics. This motivates the
separate matched training setup in
[cx_home_vector_sugar_training.md](cx_home_vector_sugar_training.md).

## Reproduction and checks

```bash
OPENBLAS_NUM_THREADS=1 python -m fly_connectome.home_vector.analysis.sugar_long_loops \
  --data results/cx-sugar-decoder-data-20261112 \
  --fit results/cx-sugar-decoder-fit-20261112-v3 \
  --output results/cx-sugar-long-loops-20261112 --loops 16
```

Use a fresh output directory and a PyTorch-enabled environment. Inputs and
source archives are hash-checked; checkpoint parameters remain identical.
Every loop's controls repeat exactly and its true closure vector is checked.
The initial loop agrees with native dynamics. All two-loop predictions agree
with the preceding decoder comparison within floating-point tolerance. An
independent calculation from saved predictions reproduced all 384 per-loop
RMSE and closure measurements (maximum RMSE discrepancy below `4e-15` mm).

![Path 8](../results/cx-sugar-long-loops-20261112/path_08.png)
![Path 9](../results/cx-sugar-long-loops-20261112/path_09.png)
![Path 14](../results/cx-sugar-long-loops-20261112/path_14.png)
![Path 15](../results/cx-sugar-long-loops-20261112/path_15.png)

## Component-wise close-ups of loop 16

The separate, derived bundle `results/cx-sugar-loop16-components-20261112`
contains x/y predictions and signed component errors during 120–128 seconds.
It uses the archived predictions directly: no model replay, fitting, reference
change, or baseline correction. The original long-loop archive is unchanged.

Each `path_*_loop_16_components.png` has three columns (original, one sugar
delivery, repeated sugar) and four rows (predicted/correct x, predicted/correct
y, signed x error, signed y error). Vertical scales differ across panels to
keep the smaller sugar-condition errors visible. Each accompanying
`*_sugar_zoom.png` overlays the two sugar conditions on common axes. PDFs,
selected full-resolution arrays, and `component_metrics.csv` are also retained.
When exporting a standalone filename for a log, copy the PNG and retain the
original filename inside the manifest-backed archive so verification and report
links continue to work.

- [Path 8: components](../results/cx-sugar-loop16-components-20261112/path_08_loop_16_components.png)
  and [sugar comparison](../results/cx-sugar-loop16-components-20261112/path_08_loop_16_sugar_zoom.png).
- [Path 9: components](../results/cx-sugar-loop16-components-20261112/path_09_loop_16_components.png)
  and [sugar comparison](../results/cx-sugar-loop16-components-20261112/path_09_loop_16_sugar_zoom.png).
- [Path 14: components](../results/cx-sugar-loop16-components-20261112/path_14_loop_16_components.png)
  and [sugar comparison](../results/cx-sugar-loop16-components-20261112/path_14_loop_16_sugar_zoom.png).
- [Path 15: components](../results/cx-sugar-loop16-components-20261112/path_15_loop_16_components.png)
  and [sugar comparison](../results/cx-sugar-loop16-components-20261112/path_15_loop_16_sugar_zoom.png).

Much of the error is a nonzero mean signed error over this loop; cyclic
tracking is not simply absent. For path 14 under repeated sugar, mean error
is `(-0.89452, +0.56091) mm`, with vector norm `1.05584 mm`. The error's RMS
variation about its mean is `0.36750 mm`, giving total RMSE `1.11797 mm`.
The waveform retains distortions as well as offset. Mean error changes across
paths and sugar schedules, so this does not justify a universal intercept
correction. The centered statistic describes error variation and does not
represent a corrected model or establish a causal source of error.

```bash
OPENBLAS_NUM_THREADS=1 python -m fly_connectome.home_vector.analysis.sugar_loop_components \
  --source results/cx-sugar-long-loops-20261112 \
  --output results/cx-sugar-loop16-components-20261112 --loop 16
```

The command checks each loop RMSE and closure error against the parent metrics
and checks the decomposition `RMSE^2 = mean_error_norm^2 + centered_error_RMS^2`.
Hash checks verify preservation of the parent archive.
