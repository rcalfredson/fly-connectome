# vDeltaE silencing and hDeltaG recovery in the frozen sugar-response network

18 November 2026. The [Janke thesis, Fig. 3.7b–d and Methods](https://digitalcommons.rockefeller.edu/student_theses_and_dissertations/808/)
reports that with continuous vΔ silencing, hDeltaG calcium activity still
dropped after sugar and then rose rapidly. The experimental driver lines also
labeled other dorsal vΔ cells. This in-silico probe selectively silences the
modeled vDeltaE population and compares the direction and timing of the
population-mean response in model units.

Replay archived path 8 (16 identical 8-second circles) in the selected frozen
sugar-trained checkpoint. Deliver sugar once at 8 seconds, then follow recovery
for 120 seconds. Run the same movement with vDeltaE intact or held at zero
from initialization, and cross each condition with sugar on or off. The
silencing removes incoming drive to the clamped cells and, because their rates
remain zero, their outgoing drive to the network. The same native update and
gate parameters apply in all conditions. Sugar-free conditions use the same
checkpoint with the gate set to zero. Controls, time grid and starting state are
shared. One deterministic movement protocol is used.

| Mean hDeltaG activity (model units) | Intact | vDeltaE silenced |
| --- | ---: | ---: |
| Last second before delivery | 1.865 | 4.538 |
| Minimum during first 8 s after delivery | 0.248 | 1.151 |
| Average, first full post-event circle (0–8 s) | 0.861 | 1.992 |
| Average, second full post-event circle (8–16 s) | 0.439 | 1.007 |
| Average, final circle (112–120 s) | 0.606 | 1.351 |

The hDeltaG mean is higher with vDeltaE silenced even before sugar. Both runs
show an initial drop with within-circle fluctuations. Comparing the same eight
seconds of movement across successive loops, the mean drops further on the
second loop in both runs and then rises gradually. This replay does not
reproduce the rapid, sustained rise to high activity described after silencing
in the thesis. The matched no-sugar control for each silencing condition also
shows that suppression persists. These controls develop different baseline
trajectories over time, so the difference from the no-sugar run describes the
effect of including sugar on that replay, not a standalone recovery rate.

The imposed gate is still about 0.67 of its unit scale 120 s after delivery.
It is a plausible contributor to the persistent suppression. The experiment
does not isolate the gate from the learned recurrent interactions as the cause
of the mismatch. It also does not quantify agreement with calcium or match the
spatial bump because the measurement units and column correspondence differ.

**Presentation.** Slide 11 follows the hDeltaB slide and uses the first two
panels of the recovery figure. “Silencing modeled vDeltaE raises hDeltaG
activity even before sugar, and hDeltaG remains suppressed for many loops
compared with the corresponding no-sugar run. This model does not reproduce
the rapid post-sugar rebound reported in the fly perturbation.” This concerns
cell-type physiology, while the main deck’s tracking and memory claims come
from the home-vector experiments.

- [Population mean and matched controls](../results/cx-home-vector-vdeltae-silencing-20261118-v3/vdeltae_hdeltag_recovery.png)
- [Individual hDeltaG traces](../results/cx-home-vector-vdeltae-silencing-20261118-v3/hdeltag_cell_traces.png)
- [Cycle averages](../results/cx-home-vector-vdeltae-silencing-20261118-v3/cycle_means.csv)

Reproduce with `python -m fly_connectome.home_vector.analysis.vdeltae_silencing
--output OUTPUT` in the home-vector environment. The run checks that the intact
replay matches the native checkpoint through the first two seconds after sugar
and that vDeltaE rates are zero in both silenced runs. The sugar and no-sugar
trajectories within each silencing condition agree through event onset.
