# Home-vector decoder comparison: original versus hDeltaC

| Measurement | Unit | Without hDeltaC | With hDeltaC |
| --- | --- | ---: | ---: |
| Long path: onset of sustained 10% magnitude underestimation | mm | 15.15 | 7.45 |
| Long path: predicted magnitude after 64 mm displacement | mm | 20.60 | 10.69 |
| Pause after backward walking: final vector error | mm | 0.294 | 0.119 |
| Pause after forward walking: final vector error | mm | 0.735 | 0.098 |
| Pause after backward walking: endpoint drift | mm | 0.320 | 0.258 |
| Pause after forward walking: endpoint drift | mm | 0.752 | 0.316 |
| Square loop: x-component RMSE | mm | 0.350 | 0.297 |
| Square loop: y-component RMSE | mm | 0.553 | 0.115 |
| Square loop: final absolute x error | mm | 0.439 | 0.005 |
| Square loop: final absolute y error | mm | 0.967 | 0.084 |
| Square loop: final vector error | mm | 1.062 | 0.084 |
| Same world path, forward walking: moving-phase vector RMSE | mm | 0.335 | 0.380 |
| Same world path, sideways walking: moving-phase vector RMSE | mm | 0.952 | 0.534 |
| Same world path, forward walking: final vector error | mm | 2.338 | 1.833 |
| Same world path, sideways walking: final vector error | mm | 2.558 | 2.023 |
| Reversal: turnaround lag | s | 0.3125 | 0.1025 |
| Reversal: final vector error after subsequent pause | mm | 0.928 | 0.071 |
| Random trajectories: test normalized RMSE | dimensionless | 0.3774 | 0.2930 |
| All test trajectories: normalized RMSE | dimensionless | 0.3261 | 0.2504 |

Pause rows use the five-second pause. Final vector error is distance from the correct home vector at that endpoint; drift is distance from the prediction at pause onset. These are finite-time measurements, not established steady states.

Long-path onset is the first interval of at least 0.5 s with magnitude more than 10% below truth, considering distances beyond 2 mm. It is not a general maximum accurate distance. The 64 mm magnitude should be compared with the true value of 64 mm.

The same-world-path protocols comprise four seconds of movement followed by a five-second pause. Square-loop component RMSE covers the entire loop; final errors measure closure. Reversal final error follows its five-second pause.

Values compare one selected checkpoint per architecture on identical trajectories; thresholds are descriptive, without across-seed uncertainty estimates.

[Comparison plot](../results/hdeltac-seed0-comparison-final-20261103/comparison.png) · [PDF](../results/hdeltac-seed0-comparison-final-20261103/comparison.pdf) · [Detailed interpretation](cx_home_vector_hdeltac_failure_comparison.md)
