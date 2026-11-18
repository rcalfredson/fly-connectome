# hDeltaC contribution to retained displacement

18 November 2026. This probe uses the frozen standard model and the identical
continuations from slides 7–8. It asks what the affine decoder reads directly
from hDeltaC and how silencing hDeltaC changes the other populations' contribution.

Compare rest versus a 4-mm forward history and rest versus a 4-mm rightward
history. For each pair, use the archived clockwise/rightward circular
continuation (path 15) and the stationary continuation (32 s). The correct
difference between the two home vectors stays 4 mm throughout each continuation.

First decompose the intact decoded difference into the hDeltaC addend and the
sum of the other cells' addends. For each group, multiply the difference in
its activities by its decoder weights. The intercept cancels between histories.
Then replay the same inputs from the same archived starting states, clamping
hDeltaC to zero during the first two seconds in both histories. Release the
clamp at 2 s. At that boundary hDeltaC is still at zero, and the altered states
evolve normally thereafter. The intact replay reproduces the archived states
and predictions within float32 tolerance. At the first clamped sample the
other cells' states exactly match the baseline.

## Readout and circuit effects

The table measures the component along the correct offset at the release
boundary. The desired value is +4 mm. The signed contributions add. The change
in the other populations is the change in their **combined decoded term**
under silencing. Full vector error and the norm of each relevant change are
saved separately.

| Continuation | Prior movement | Intact total | Direct hDeltaC term in intact replay | Total with hDeltaC silenced | Change in the other populations |
| --- | --- | ---: | ---: | ---: | ---: |
| Circles | Forward | +2.520 | +0.049 | +1.844 | −0.627 |
| Circles | Rightward | +1.940 | +0.090 | +1.698 | −0.152 |
| Pause | Forward | +2.297 | +0.017 | +1.990 | −0.290 |
| Pause | Rightward | +1.969 | +0.105 | +1.657 | −0.207 |

All values are in mm. In the forward/circles example, the direct term is only
+0.049 mm along the target, whereas the other populations lose +0.627 mm
from that component when hDeltaC is silenced. Even including the perpendicular
component, the direct hDeltaC addend's norm at that time is 0.110 mm, smaller
than the 0.644-mm change in the other populations' decoded difference.

All four comparisons show weaker target-aligned separation at the release
boundary. The later effects depend on the pair and movement. For example,
at 16 s in the circles, the prior silencing still weakens the forward
separation (1.295 to 0.974 mm) and increases the rightward separation
(0.855 to 1.195 mm). The offset's perpendicular component can change
independently. For the rightward-history pause at 32 s, target-aligned
separation falls from 2.600 to 2.310 mm, while full vector error improves
from 4.656 to 3.480 mm.

These replays show a circuit influence on displacement retention in the
trained checkpoint, most clearly early in the forward/circles pair. The
acute intervention can affect multiple recurrent pathways. Attribution to
a particular connection or a general memory function would require additional
tests. The assay addresses a learned role in this simulated tracking task.
The measured role of hDeltaC in odor-guided navigation concerns a different
sensory context.

- [Four comparisons, first 16 s](../results/cx-home-vector-hdeltac-retention-20261118-v2/hdeltac_retention.png)
- [Numerical summaries](../results/cx-home-vector-hdeltac-retention-20261118-v2/summary.csv)

The shaded region marks silencing, and the orange line is the intact direct
hDeltaC addend along the target. The blue and purple traces are full decoder
outputs projected onto the correct offset direction. Reproduce in the
home-vector environment with
`python -m fly_connectome.home_vector.analysis.hdeltac_retention --output OUTPUT`.
