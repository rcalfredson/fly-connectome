# Additional prescribed input with PFNd/PFNv unchanged

Reviewed 11 November 2026. Research and read-only anatomy extraction only. No
input prescription, dynamical circuit, checkpoint, decoder or training run was
changed. Earlier PFNd digitization work remains separate.

**Result:** PFNa and PFNp_c have direct access and useful primary physiology,
but neither currently supplies a validated extra walking-velocity prescription.
PFR_a supports a different boundary experiment using a prescribed downstream
travel-direction signal. None is yet recommended for a production extension.

## Anatomical access: complete incoming census

The earlier complete incoming survey covered vDeltaE only. The saved hDeltaC
`incoming_connections.csv` is a whitelist of selected projections, not a full
input census. A new query therefore retained every Segment source with reported
weight >=1 into the exact 65 archived dynamic body IDs, without type, status,
cropping or ROI filters.

The new census contains **18,145 directed pairs, 134,706 contacts, 3,719 source
bodies and 248 source populations including untyped sources**. Source bodies
are not necessarily whole biological neurons. The server, dataset, UUID and
mutation match the archived model:

```text
neuprint.janelia.org / male-cns:v1.0
UUID 4b2087c0fbe046bfaf0d60bc970e3e5d
latestMutationId 1006591300
```

All target post totals, unique pairs, positive integer weights and stable
before/after metadata were checked. The vDeltaE pairs reproduce the entire
earlier census exactly. All 1,626 contact pairs in the prepared 65-neuron model
agree with the new query; the reference PFNd 26913 -> hDeltaB 515192 remains 137.

| Target | Selected cells | All incoming contacts |
| --- | ---: | ---: |
| hDeltaB | 19 | 46,924 |
| hDeltaC | 20 | 44,208 |
| vDeltaE | 18 | 14,583 |
| hDeltaG | 8 | 28,991 |

Contact counts below are totals across all selected cells of the receiving
population, not functional weights or measures of independent information.
Zero means no direct connection reported in this snapshot, not biological
absence or absence of an indirect route.

| Candidate | Full exact-type cells | -> hDeltaB | -> hDeltaC | -> vDeltaE | -> hDeltaG | Assessment |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| PFNa | 58 | 6 | 1,303 | 25 | 23 | Conditional sensory candidate, particularly via hDeltaC |
| PFNp_c | 40 | 1 | 570 | 497 | 0 | Conditional sensory candidate via hDeltaC/vDeltaE |
| PFR_a | 28 | 763 | 202 | 33 | 150 | Downstream travel-direction cue; a different boundary experiment |
| FC3_a | 17 | 14 | 205 | 367 | 400 | Secondary PFNa-related route; subtype/output mapping unresolved |
| FC3_b | 20 | 13 | 251 | 238 | 114 | Same qualification |
| FC3_c | 15 | 15 | 342 | 347 | 362 | Same qualification |
| PFNp_b | 136 | 0 | 34 | 303 | 0 | Exact-subtype physiological mapping unresolved |
| PFNm_a | 21 | 0 | 319 | 162 | 0 | No transferable exact-subtype walking rule identified |
| PFNm_b | 26 | 0 | 131 | 96 | 0 | Same qualification |
| FB5AB | 2 | 0 | 1,751 | 1,103 | 38 | Anatomy/odor evidence, not a demonstrated movement cue |
| FB4B | 2 | 5,373 | 15 | 23 | 7 | Strong anatomical lead; no calibrated independent velocity rule located |
| FB3A | 4 | 883 | 6 | 0 | 5 | Anatomical lead; same physiological gap |
| FB4R | 6 | 3,647 | 23 | 0 | 10 | Anatomy/behavior alone does not identify lateral coding |
| FB4M | 4 | 3,391 | 111 | 11 | 54 | Dopamine annotation is not a signed velocity calibration |
| FB6N | 2 | 0 | 86 | 1,173 | 59 | Anatomical lead with an unresolved prescription |
| FB6O | 2 | 0 | 89 | 660 | 12 | Same qualification |
| SpsP / IbSpsP / LNO1 / LNO2 | 4 / 31 / 4 / 2 | 0 | 0 | 0 | 0 | Each has zero direct contacts to the selected targets |

The full-population counts use the saved exact-type inventory from the same
snapshot. Per-projection source counts, target coverage, label correspondences,
transmitter annotations, and every other direct source are retained in the
machine-readable output. Counts in this table are independently checked against
that output; they are not copied from a different connectome or cell-type atlas.

## Primary evidence and the available prescriptions

**PFNa.** Ishida et al. combine heading and airflow measurements with
electrophysiology and a model distinguishing sodium spikes from calcium
oscillations. Their schematic sodium-rate fit is:

```text
r_Na = max(0, b_H*cos(H) + b_W*cos(W ± 45 degrees))^2
```

Here W is airflow direction, not body travel direction. Calcium-associated
phase inversion need not have the same fast downstream effect as sodium spikes;
transfer to hDeltaC was not established. Numerical datasets and model code are
public. This is a better specified physiological starting point than inventing a
linear lateral-velocity input, but it does not validate that substitution.
[Ishida et al., Cell 2026, Methods and limitations](https://pmc.ncbi.nlm.nih.gov/articles/PMC12758628/).

**PFNp_c.** May et al. measure transient airflow direction/airspeed responses and
fit a dynamic sensory model. Its amplitude term is:

```text
C(s, W) = (1 - exp(-g*s)) * [A*cos(W-W0) + b]
```

Temporal adaptation and offset handling are also required. The Methods use s in
m/s; the inspected MATLAB implementation uses cm/s, requiring parameter-unit
conversion. PFNp_c's weak measured bump structure also cautions against treating
a fitted sinusoidal heading bump as established physiology. Their imaging
airspeeds were 20, 45 and 60 cm/s. [May et al., 2026, Figs. 2/4 and Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC13524059/),
[PC_response_de.m](https://github.com/nagellab/Mayetal2026/blob/main/MATLAB/FBmodel/model_code/PC_response_de.m),
[PFNpc_integ2.m](https://github.com/nagellab/Mayetal2026/blob/main/MATLAB/FBmodel/model_code/PFNpc_integ2.m).

**PFR_a.** Lyu et al. report travel-direction-related population phase in walking
and flight; their drivers predominantly label PFR_a. hDeltaB is its dominant
input, and PFR phase is not identical to hDeltaB phase. This supports considering
a travel-angle cue, not assigning it an exact noiseless phase or transferring
the result to PFR_b. [Lyu et al., Methods and Extended Data Fig. 7](https://pmc.ncbi.nlm.nih.gov/articles/PMC11104186/).

**FC3 and pooled ventral PFNs.** The PFNa study provides FC3 readout evidence,
but not an interchangeable mapping for the MaleCNS FC3_a/b/c split: all three
carry the broad hemibrain/FlyWire label FC3 in this census. Earlier ventral-PFN
recordings also pool types and cannot calibrate PFNp_b or PFNm subtypes by name
alone. [Ishida et al., Fig. S6](https://pmc.ncbi.nlm.nih.gov/articles/PMC12758628/),
[Currier et al., 2020](https://elifesciences.org/articles/61510).

**Tangential inputs.** FB5AB's odor responses and navigation effects do not
provide a velocity prescription under matched odor conditions. The 2024
addendum also qualifies attribution of the earlier hDeltaC driver recordings;
it does not remove the anatomical PFNa/FB5AB -> hDeltaC routes.
[Matheson et al., 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9360402/),
[2024 addendum](https://www.nature.com/articles/s41467-024-46225-8).
The broader anatomical names and connections are useful for discovery, not
equations for locomotor activity. [Hulse et al., 2021](https://elifesciences.org/articles/66039).

Recent course-correction experiments additionally implicate PFN pathways in
walking behavior, but optogenetic effects and behavioral necessity do not
calibrate an extra signal's response to our ambiguous velocities.
[Crown et al., 2026](https://doi.org/10.1016/j.cub.2026.09.069).

## Does a candidate actually supply the missing distinction?

The current task specifies ideal heading and ground-relative body velocity,
not ambient wind, optic-flow scene geometry, or antenna mechanics. Mapping
ground velocity directly to a measured airflow response would add assumptions.
An elementary kinematic relation, before any sensory mechanics, is:

```text
relative_air_velocity_body = rotate(-heading, ambient_wind_world)
                           - body_velocity
```

The paper's airflow-source angle convention must then be reconciled with the
air-velocity direction. Zero ambient wind is a possible experimental assumption,
not a supplied task variable. Nominal training speed <=2 mm/s corresponds to
<=0.2 cm/s. Treating self-motion alone as airflow would therefore extrapolate
at least 100-fold below the lowest imaging airspeed above. Adding substantial
ambient wind changes the relation between airflow and movement and can weaken
velocity sensitivity. These calculations do not establish a sensory detection
threshold; they establish that the required calibration is missing.

Lack of speed tuning does not automatically disqualify an extra directional
signal. At fixed heading, the ambiguous PFN pattern retains a positive
projection p onto one known axis phi. If an extra signal uniquely gives body
movement direction alpha in that sector, then:

```text
speed = p / cos(alpha - phi)
```

The angle and retained projection can together distinguish the velocities.
Actual tuning must nevertheless be checked for plateaus, symmetries, heading
dependence, temporal adaptation and noise. For local velocity sensitivity:

```text
lost-change direction d = (-sin(phi), cos(phi))
extra cue z is locally informative if gradient_velocity(z) dot d != 0
```

For a population, stack all of its activity gradients with the existing PFN
gradients and test rank/singular values. Anatomical access introduces another
check: the distinct source patterns must still differ after projection through
the reported connections. Source-level independence need not survive pooling.
No published PFNa/PFNp_c prescription has been implemented or claimed to pass
this joint audit here.

## A separate PFR_a boundary experiment

A proposed idealized phase would be:

```text
travel_phase = heading + atan2(rightward_velocity, forward_velocity)
```

At nonzero speed this distinguishes the known same-heading ambiguity pair,
whose body directions are 122 and 132 degrees. It is instantaneous movement
information, not home position or accumulated displacement. However, clamping
PFR_a from this formula supplies a downstream body-to-world transformation that
the biological circuit normally computes. It cannot be described as discovering
an independent upstream sensor. A dynamic PFR population receiving only existing
PFN histories would not recover a distinction absent from those histories.

Amplitude, low-speed phase, offsets, delays, activity units, per-cell phase
mapping and synaptic sign would still need specification. In particular, this
census annotates PFR_a transmitter as unclear. Contact counts alone do not
resolve that issue. This option should be discussed as a distinct capacity
comparison, not silently substituted for the sensory-input search.

## Recommendation and stopping point

Keep **PFNa and PFNp_c** as the short list for a sensory extension, conditional
on a defensible sensory scene and task-range calibration. PFNa has the stronger
mechanistic spike-output specification; PFNp_c offers a published dynamic
direction/airspeed prescription and access to vDeltaE as well as hDeltaC.
Neither clears every requirement for the present walking task.

Keep **PFR_a** separately if the aim becomes testing the dynamic network with
an additional prescribed travel-direction representation. Do not prescribe the
same physiology to PFR_b or inject an arbitrary scalar along its edges.

FB4B/FB3A and the other strongly connected tangential populations are anatomical
leads, but no transferable independent walking-velocity rule was located in this
bounded pass. Search failure is not evidence of no physiological function.
SpsP/IbSpsP/LNO1/LNO2 do not provide a direct-route extension under the current
boundary. A one-hop census does not exclude routes through additional dynamic
intermediates.

An additional *population* is not one extra scalar or one extra neuron: full
PFNa, PFNp_c and PFR_a prescriptions would add 58, 40 or 28 clamped activities.
No candidate is selected for implementation or training. Removing the known
encoder ambiguity also remains distinct from explaining the archived
forward/rightward imbalance on already-distinguishable paths.

## Artifacts and search scope

- `results/cx-additional-input-census-20261111/`: exact queries, raw API responses,
  target IDs, all edges, metadata and manifest. Credentials are not saved.
- `results/cx-additional-input-review-20261111/`: every source-by-target-type
  summary, candidate CSV, verification checks and exact query/aggregation scripts.
- Full review of physiological details focused on PFNa, PFNp_c, PFR and FC3;
  broader searches screened strongly connected tangential types and pooled PFNs.
  All 248 source populations are inventoried, not all individually reviewed.
- Discovery used exact cell names and historical labels with movement, lateral
  velocity, airflow, calcium, electrophysiology, walking and model terms. Atlas
  and secondary speculation hits were not used as physiological evidence.
- Two published model repositories were inspected directly; no model code was
  ported. Numerical data availability for PFNa and PFNp_c is stronger than for
  the earlier Lu digitization, but does not remove the sensory/task mismatch.

The numerical screening tables and checks are reproducible offline using the
archived aggregation script with its recorded inputs. No new test suite or
simulation is needed for this research-only addition; the census checks verify
the quantities on which the shortlist relies.
