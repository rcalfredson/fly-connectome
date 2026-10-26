# Functional profiles of the logged vDeltaE inputs

Reviewed 25 October 2026. This first pass covers the **ten selected populations
in the survey report's table**, excluding the appended PFNd, PFNv and hDeltaB
comparisons. It is not a review of every source in the full census, which has
178 typed input populations plus untyped sources. The ten reviewed populations
contribute 7,749/14,583 contacts (53.14%); other inputs remain in the saved census.

All contact counts, target coverage and transmitter annotations below come from
the existing, offline-verified MaleCNS census. No new connectivity hop, dynamics,
code cleanup or commit is part of this literature review. The machine-readable
companion is `results/vdeltae-input-profiles-20261025/profiles.csv`.

Evidence labels distinguish **physiology/behavior**, **anatomy**,
**population-level association**, **dissertation**, and **our hypothesis**.
“No subtype-specific function located” describes this bounded search, not proof
that the population has never been studied. Experimental labels and cell-type
names must be reconciled before transferring results between studies/datasets.

## Table for the log

Percentages divide by all **14,583** incoming Segment contacts, including
untyped sources. Every population below reaches all 18 receiving vDeltaE cells.
Source counts mean bodies connected to vDeltaE, not necessarily the full type.
ACh/Glu are saved `consensusNt` annotations, not measurements of synaptic sign
or efficacy onto vDeltaE.

| Cell type | Connected source bodies | Contacts (% of all input) | NT | Functional profile and evidence boundary |
| --- | --- | --- | --- | --- |
| **hDeltaC** | 20 | 1,604 (10.999%) | ACh | Anatomically positioned to combine olfactory and wind-related inputs, but the often-cited physiological assignment is qualified by a published driver correction. **Anatomical candidate; function not securely established by that study.** [2022 study](https://doi.org/10.1038/s41467-022-32247-7), [2024 addendum](https://doi.org/10.1038/s41467-024-46225-8). |
| **FB6N** | 2 | 1,173 (8.044%) | Glu | Layer-6 tangential type, historical name `CRESMPFB6_1`. No isolated sensory, movement or memory function located. A broad modulatory role is a question, not an established result. [Hulse, Table 2](https://elifesciences.org/articles/66039). |
| **FB5AB** | 2 | 1,103 (7.564%) | ACh | Attractive-odor responses and activation-induced upwind movement are demonstrated. Silencing did not impair upwind orientation in the reported assay. **Olfactory/navigation evidence; not a demonstrated displacement signal.** [Matheson et al., Figs. 3, 4 and 6H](https://doi.org/10.1038/s41467-022-32247-7). |
| **FB6O** | 2 | 660 (4.526%) | Glu | Layer-6 tangential type, historical name `SIPSMPFB6_4`. No subtype-specific physiological computation located; do not assign wind, speed or reset coding from its layer/name. [Hulse, Table 2](https://elifesciences.org/articles/66039). |
| **hDeltaG** | 8 | 653 (4.478%) | ACh | Janke reports recent-distance-related bump amplitude, sugar-associated reset and a vDeltaE-dependent ramp/leak mechanism. **Dissertation evidence, assessed from its abstract.** Its input to vDeltaE is the reciprocal limb of the motif, not simply an independent upstream sensory signal. [Janke thesis](https://digitalcommons.rockefeller.edu/student_theses_and_dissertations/808/). |
| **FB6E** | 2 | 611 (4.190%) | Glu | Anatomically matched to a heterogeneous R23E10 dorsal-FB population associated with sleep regulation; that does not isolate FB6E's role. MBON-linked anatomical routes also exist. **Population association and anatomy, not a proven FB6E sleep or integration function.** [Hulse, Fig. 48 and Methods](https://elifesciences.org/articles/66039), [MB connectome, Fig. 20 supplement 1C](https://elifesciences.org/articles/62576/figures). |
| **FB5I** | 2 | 583 (3.998%) | Glu | Layer-5 tangential type with anatomical access to MBON pathways through intermediary neurons. Learned/contextual modulation is plausible, but no isolated coding function was located. Do not confuse it with dopaminergic FB5H. [MB connectome, Fig. 20 supplement 1C](https://elifesciences.org/articles/62576/figures). |
| **FB5L** | 2 | 562 (3.854%) | Glu | Layer-5 tangential type contacted by CRE042, an intermediary receiving inputs from multiple MBON types. This supports a possible learned-value/context route, not demonstrated value encoding or memory storage by FB5L. [MB connectome, Fig. 20 supplement 1A](https://elifesciences.org/articles/62576/figures). |
| **PFNp_c** | 40 | 497 (3.408%) | ACh | Direct imaging shows transient airflow direction and speed responses, with little optic-flow response. **Peer-reviewed subtype-specific physiology.** Flight-related conditions differ from our walking/body-velocity model. [May et al., Fig. 2](https://doi.org/10.1126/sciadv.aeh7220), [accessible text](https://pmc.ncbi.nlm.nih.gov/articles/PMC13524059/). |
| **PFNp_b** | 98 | 303 (2.078%) | ACh | PFN columnar type; no confidently transferable, exact-subtype physiological function located. Broad ventral-PFN airflow results are useful context, not a calibration for these 98 bodies. Cross-dataset labels require particular care. [Hulse, Fig. 29](https://elifesciences.org/articles/66039), [Currier et al.](https://elifesciences.org/articles/61510). |

## Corrections and interpretation limits that matter

**hDeltaC versus hDeltaK.** The 2024 addendum says VT062617 additionally or
perhaps predominantly labels hDeltaK. It qualifies attribution of the 2022
paper's sensory responses (Fig. 5E–I) and behavioral effects (Fig. 6) to hDeltaC.
The anatomical FB5AB/PFNa inputs to hDeltaC remain supported. This is a driver
assignment issue, not a synonym permitting us to rename the census hDeltaC cells.
The right log entry is “candidate olfactory/wind convergence; physiology
qualified,” not “established wind/odor integrator.” [Addendum](https://doi.org/10.1038/s41467-024-46225-8).

**PFNp_c publication status and stimulus.** The May work is now a peer-reviewed
2026 Science Advances article; the 2025 preprint has a different title. Its
airflow responses adapt, so it should not be modeled automatically as a
sustained speed channel. The paper's decoding models establish computational
sufficiency, not a demonstrated vDeltaE wind decoder or a displacement store.
[Final article](https://doi.org/10.1126/sciadv.aeh7220).

**hDeltaG is feedback here.** Our existing targeted data also contain 2,405
vDeltaE→hDeltaG contacts. Together with the 653 reverse contacts, this establishes
reciprocal anatomy. It does not establish that recurrence causes integration,
what the feedback signal is, or its physiological sign. The dissertation's full
methods/equations remain to be examined; no time constant or ramp rule is taken
from the abstract. [Thesis record](https://digitalcommons.rockefeller.edu/student_theses_and_dissertations/808/).

**Tangential is not synonymous with a global gain control.** The layer-5/6
tangential types differ anatomically and can carry different signals. Broad
arbors alone do not establish uniform column weights, inhibition, a ramp, a
reset, or a sleep function. Glutamate annotations do not determine receptor
effects on vDeltaE. Anatomy can motivate these hypotheses without choosing one.

**The MB routes are anatomical hypotheses.** The published MB connectome shows
MBON-linked routes to FB5L through CRE042 and to FB5I/FB6E through other
intermediaries. Calling the resulting signals learned value, context, or state
requires physiological evidence. No such additional routes were extracted in
our MaleCNS survey. [Fig. 20 supplement 1](https://elifesciences.org/articles/62576/figures).

## Names and connected-population coverage

These are observed annotation correspondences from the saved census, not a new
assertion of one-to-one functional homology. `hemibrainType` equals the MaleCNS
type for all ten reviewed populations. The following FlyWire labels occur:

| MaleCNS type | Observed `flywireType` | Connected bodies / dataset-wide exact-type count |
| --- | --- | --- |
| hDeltaC | hDeltaC | 20/20 |
| FB6N | FB6N | 2/2 |
| FB5AB | CB.FB5G0 | 2/2 |
| FB6O | CB.FB6I1 | 2/2 |
| hDeltaG | hDeltaG | 8/8 |
| FB6E | FB6E | 2/2 |
| FB5I | CB.FB5E0 | 2/2 |
| FB5L | FB5L | 2/2 |
| PFNp_c | PFNp | 40/40 |
| PFNp_b | PFNp **and** PFNm | 98/136 |

In particular, 98 is not the total PFNp_b population. A paper using “PFNm” or
“PFNp” cannot be assigned wholesale to MaleCNS PFNp_b based on its label alone.
The relevant body-level annotations are retained in the companion CSV. Older
ventral-PFN work grouped multiple anatomical types; it supports examining airflow
coding, but not copying PFNd/PFNv tuning into PFNp_b. [Currier et al., cell-type
description](https://elifesciences.org/articles/61510).

## What to investigate next

**Direction updated 26 October 2026:** the immediate next milestone is the
[source-grounded vDeltaE–hDeltaG mechanism specification](vdeltae_hdeltag_next_steps.md),
before adding dynamics or choosing an upstream population to simulate.

The best specified signals for an initial input assessment are **FB5AB's
olfactory responses** and **PFNp_c's airflow responses**. This prioritization is
about available functional evidence, not contact rank or a decision to implement
these inputs next. For each, the unresolved
question is whether and how that signal affects vDeltaE in the relevant behavior.

hDeltaC remains anatomically prominent, but needs reliable experimental access
and tuning evidence. hDeltaG must be interpreted within the reciprocal motif.
For FB6N, FB6O, FB6E, FB5I and FB5L, initial hypotheses should remain broad:
possible state/context or population modulation, tested against alternatives
such as sensory tuning. For PFNp_b, resolving subtype correspondence precedes
choosing an encoding rule. None of these findings presently supplies a complete
set of inputs for the thesis mechanism.

## Source register and search limits

| Source | Status and location used |
| --- | --- |
| [Matheson et al., 2022](https://doi.org/10.1038/s41467-022-32247-7) | Peer-reviewed primary imaging/behavior/anatomy. Figs. 3–6, read with the 2024 addendum. |
| [Matheson et al., 2024 addendum](https://doi.org/10.1038/s41467-024-46225-8) | Published correction to physiological cell-type attribution; essential for hDeltaC. |
| [May et al., 2026](https://doi.org/10.1126/sciadv.aeh7220) | Peer-reviewed primary imaging/modeling, especially Fig. 2. [Publication metadata](https://pubmed.ncbi.nlm.nih.gov/42664356/) confirms final status. |
| [Hulse et al., 2021](https://doi.org/10.7554/eLife.66039) | Peer-reviewed primary anatomy: Tables 2–3, Fig. 29, Fig. 48 and R23E10 identification methods. Group associations do not isolate subtype function. |
| [Li et al., MB connectome](https://doi.org/10.7554/eLife.62576) | Peer-reviewed primary anatomy; Fig. 20 supplement 1A/C provides the MBON-linked examples. |
| [Currier et al., 2020](https://doi.org/10.7554/eLife.61510) | Peer-reviewed primary airflow imaging/behavior; grouped ventral-PFN scope matters. |
| [Janke, 2025](https://doi.org/10.48496/947v-bw85) | Dissertation; abstract-level assessment, not a journal article or a mechanistic reproduction. |

Exact-type searches were paired with neuron/Drosophila/physiology, then relevant
older anatomical names and terms such as airflow, odor, sleep, calcium and
memory. Primary papers and their corrections were used for claims. Atlas and
ontology hits aided discovery only; their functional summaries were not treated
as experiments. The hDeltaK attractor work was encountered but not reassigned to
hDeltaC. No claim of exhaustive coverage is made, and negative search results
are explicitly labeled. Publisher/PMC access was intermittently blocked; accessible
full text, primary figure pages, indexed primary text and publication metadata
were used. The Janke full text has not been reviewed here.

The census source hashes and body IDs are recorded in
`results/vdeltae-input-profiles-20261025/manifest.json`. Counts were checked against
the saved summary, exact-type inventory and body-level annotations; generated
profiles retain the source directory unchanged. The subsequent validation and
report-generation fixes are part of the [repository checkpoint](vdeltae_hdeltag_next_steps.md#repository-checkpoint);
full-thesis model specification and new dynamics remain subsequent work.
