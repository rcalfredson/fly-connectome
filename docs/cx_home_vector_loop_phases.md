# Matching phases across repeated circular movements

16 November 2026. First descriptive step toward understanding the frozen
standard network's repeated-return behavior. No training, decoder fitting,
sugar dynamics, neural-state reset or perturbation is introduced.

## Setup

Use the independently validation-selected standard checkpoint (update 2,000)
from `results/cx-sugar-reliable-return-training-seed0-20261113`. Replay the four
familiar 16-loop circles, each with an eight-second period, from
`results/cx-sugar-long-loops-20261112/protocols.json`. State starts from zero
once and evolves continuously for 128 seconds. The terminal stationary control
has no following update and does not reset the closing state.

Output: `results/cx-standard-loop-phases-20261116`.

Verify identical controls and identical encoded float32 PFN inputs across all
loops at corresponding phases. Also verify matching world-centered targets and
zero displacement at loop boundaries. Compare all 65 dynamic neurons, in their
fixed anatomical order. The first loop contains the transient from zero initial
activity; also report intermediate loops so that interpretation is not based
solely on the first-to-second comparison.

```text
phase = time within each eight-second movement cycle
activity_change(loop, phase) = activity(loop, phase) - activity(loop - 1, phase)
state_difference_RMS = sqrt(mean_over_phase(sum_over_neurons(activity_change^2)))
```

This is the phase RMS of the full population's Euclidean difference, not the
RMS of an individual neuron. Raw per-neuron differences and per-population
summaries are also saved. Average over phases in `[0, 8 s)`; closing states are
scored separately to avoid duplicating boundaries. No activity normalization,
recentering or alignment to neural peaks is performed.

## Results

| Path | Movement | Loop 2 minus 1: state difference RMS | Loop 16 minus 15: state difference RMS |
| --- | --- | ---: | ---: |
| 8 | Counterclockwise, forward | 8.29666 | 0.000558 |
| 9 | Counterclockwise, rightward | 11.00739 | 0.008467 |
| 14 | Clockwise, forward | 12.53722 | 0.000563 |
| 15 | Clockwise, rightward | 11.12730 | 0.000333 |

All state differences are in model activity units. The first-to-last decrease
is approximately 1,300–33,000-fold. Intermediate loop differences also shrink:

| Path | Loop 4 minus 3 | Loop 8 minus 7 | Loop 12 minus 11 |
| --- | ---: | ---: | ---: |
| 8 | 1.17440 | 0.065616 | 0.004850 |
| 9 | 1.06068 | 0.146798 | 0.030076 |
| 14 | 0.16488 | 0.021376 | 0.003473 |
| 15 | 0.17265 | 0.007974 | 0.000937 |

At loop 16, adjacent-loop state differences are 0.00078–0.01995% of the
corresponding raw state RMS. The largest individual-neuron change RMS on each
path is 0.000325, 0.004509, 0.000323 and 0.000181 units respectively. Thus this
finding is not merely cancellation in population means: actual individual
activities become more repeatable at matched phases. The state remains
nonzero and fluctuates substantially within each loop.

Repeatability does not establish accurate tracking:

| Path | Loop 16 minus 15: prediction difference RMS, mm | Loop 16 tracking RMSE, mm |
| --- | ---: | ---: |
| 8 | 0.000268 | 1.93164 |
| 9 | 0.004565 | 1.51379 |
| 14 | 0.000293 | 1.53644 |
| 15 | 0.000177 | 1.30457 |

These tracking errors refer to the final loop, not the previous report's
whole-128-second averages. Path 9 illustrates the distinction particularly
well: the predictions settle toward a repeating waveform while tracking RMSE
increases toward a persistent error. Small changes in the output can therefore
coexist with a substantial systematic error.

## Interpretation and next question

Over these finite, periodically driven replays, the **full neural state**
approaches a repeating activity cycle. Stable decoding is not explained solely
by large ongoing state changes that cancel at the decoder. A row/null split can
still characterize the residual changes, but is not required to establish that
the complete differences shrink here.

This does not prove an asymptotic attractor, stability to perturbation,
unlimited boundedness, accurate integration of arbitrary paths, or a learned
home-reset mechanism. The observations are at the recorded integration step
and float32 precision, with repeated inputs and a single initialization per
motion. Familiar circles are diagnostic examples, not fresh generalization
evidence. No comparison here isolates training exposure as the causal reason
for this behavior.

The next mechanistic step is to account for the remaining within-loop and
loop-to-loop changes using PFN drive, recurrent drive, bias, rectification and
relaxation. Small net changes over a cycle alone do not identify which of those
terms produce the balance. Pause and continuation probes remain a later test
of whether useful displacement information is preserved beyond these motions.

## Follow-up framing for the presentation

Keep the next analysis focused on whether repeatable activity preserves the
displacement information needed for an accurate home vector, or partly reflects
forgetting that information. First describe the persistent prediction error:
separate mean x/y offsets from differences in waveform amplitude, timing and
shape. These are descriptions of the frozen model, not corrections to its
predictions. Timing and amplitude errors have not yet been quantified.

Then compare physically reachable states with different correct home vectors
under identical subsequent inputs. Their correct vector difference should
remain constant. Loss of decoded separation would indicate a memory/readout
failure; loss of complete state separation would additionally implicate the
dynamics. States with equal home vectors but different histories provide a
complementary control. This test has not yet been run.

For the presentation, distinguish computational performance, mechanisms
demonstrated by interventions, and agreement with measured cell-type activity.
Anatomical support and repeatable activity alone do not establish physiological
validity. The current heatmaps describe model activity in arbitrary units;
comparison with physiology requires appropriate experimental conditions and
measurement-specific observables. Preserve this distinction when connecting
the machine-learning results to the underlying biology.

## Plots and checks

- [Path 8: phase comparisons](../results/cx-standard-loop-phases-20261116/path_08_phase_comparison.png)
- [Path 9: phase comparisons](../results/cx-standard-loop-phases-20261116/path_09_phase_comparison.png)
- [Path 14: phase comparisons](../results/cx-standard-loop-phases-20261116/path_14_phase_comparison.png)
- [Path 15: phase comparisons](../results/cx-standard-loop-phases-20261116/path_15_phase_comparison.png)
- [Path 9: actual activity of all neurons](../results/cx-standard-loop-phases-20261116/path_09_phase_activity.png)
- [All plots and summary](../results/cx-standard-loop-phases-20261116/report.md)

Phase comparison plots overlay all 16 loops, show full-state differences across
phase, and separate tracking/closure errors from prediction repeatability.
Activity heatmaps show loops 1, 8 and 16 with a common scale across loops within
each population. Scales differ between populations. Neuron IDs and types are
retained in full-resolution NPZs and per-neuron CSVs.

The output bundle passes artifact hash verification. Independent calculations
from the saved arrays reproduce the final-loop state differences. Whole-path
RMSEs reproduce the earlier standard-network circle evaluation. Input hashes
and parameters are unchanged after replay. Targeted tests distinguish hidden
individual drift under constant population means, inaccurate but repeatable
outputs, terminal-state scoring, and incompatible phase targets.

```bash
MPLCONFIGDIR=/tmp/fly-connectome-mpl OPENBLAS_NUM_THREADS=1 \
/home/tracking/miniconda3/envs/py3.13-torch/bin/python \
-m fly_connectome.home_vector.analysis.loop_phases \
--output results/cx-standard-loop-phases-20261116
```

Use a fresh output directory when reproducing the analysis.
