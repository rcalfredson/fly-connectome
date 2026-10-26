# Evidence ledger — reviewed 24 October 2026

This is a bounded primary-source review, not an exhaustive survey. Dataset
measurements are in `results/hdeltab-extension-census-final-20261024`; they are
independent of literature claims. All physiological interpretations transferred
to MaleCNS remain cross-animal and, for hemibrain anatomy, cross-dataset assumptions.

| Source and publication status | Relevant evidence | What it does not establish | Consequence |
| --- | --- | --- | --- |
| [Lu et al., Nature 2022](https://doi.org/10.1038/s41586-021-04191-x), [accessible manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC10759448/), peer-reviewed primary physiology, anatomy and modeling | PFNd/PFNv combine heading and body translation; hDeltaB represents world-relative travel velocity during walking. | The paper conjectures downstream path integration; it does not identify a demonstrated hDeltaB-driven displacement store. | Preserve the prescribed encoding as the existing model's assumption. Do not relabel its hDeltaB activity as memory. |
| [Lyu, Abbott & Maimon, Nature 2022](https://doi.org/10.1038/s41586-021-04067-0), [accessible manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC11104186/), peer-reviewed primary experiments | hDeltaB/PFR travel-direction signals; PFR used for perturbation readout because of dominant hDeltaB input. PFR has additional inputs and differs subtly from hDeltaB. Speed tuning is not uniformly vector-like across travel directions. | PFR is not established as a displacement integrator. The physiological PFR label does not resolve MaleCNS PFR_a versus PFR_b function. | PFR_a/b are relay controls, not automatic memory candidates or a guaranteed correction for upstream errors. |
| [Hulse et al., eLife 2021](https://doi.org/10.7554/eLife.66039), peer-reviewed primary hemibrain anatomy and conceptual modeling, especially Table 1, Figs. 31, 56 and 70 | hDeltaG and vDeltaE are distinct interneuron types; hDelta arbors skip columns, vDelta arbors span layers. PFR subtypes and ROB interactions are distinguished. | Recurrence and proposed vector computations do not demonstrate physiological integration. | Verify exact types and directed edges in MaleCNS. Avoid equating within-type C labels with shared functional angles. |
| [Janke, 2025 dissertation](https://digitalcommons.rockefeller.edu/student_theses_and_dissertations/808/), DOI [10.48496/947v-bw85](https://doi.org/10.48496/947v-bw85); **dissertation, abstract-level assessment** | Reports hDeltaG reset at sugar, recent-distance-dependent bump amplitude, and imaging/perturbation evidence implicating vDeltaE input. Proposed model combines a rising baseline, input-dependent slowing, and leak. | Neither a peer-reviewed demonstration of an hDeltaB-driven home vector nor grounds for replacing vDeltaE by hDeltaB. | Strongest specific motivation for examining the vDeltaE–hDeltaG motif, but insufficient detail for mechanistic reproduction here. |
| [Avritzer, 2026 dissertation](https://digitalcommons.rockefeller.edu/student_theses_and_dissertations/837/), DOI [10.48496/zdsr-5w52](https://doi.org/10.48496/zdsr-5w52); **dissertation, abstract only**, full text listed for 30 May 2027 | Reports hDeltaA memory of recent travel direction over roughly 7–10 s, context dependence and an inertia-like contribution to navigation. | Recent-direction working memory is not necessarily net displacement or a vector pointing home. | hDeltaA is a serious alternative, but larger than hDeltaG and with unresolved mechanism/inputs. |
| [Mussells Pires et al., Nature 2024](https://doi.org/10.1038/s41586-023-07006-3), peer-reviewed primary physiology and perturbations, Figs. 1–5 and Extended Data 6 | FC2 communicates goal angle; EPG and FC2 converge onto PFL3; PFL3 combines heading and goal signals for steering. FC2 includes A/B/C subtypes. | Goal signaling is not proof of goal storage or displacement integration. FC2 bump persistence does not identify the memory mechanism. | FC2/PFL3 are endpoints for anatomical route checks. Simulating steering would require separate heading, goal and motor assumptions. |

Janke's repository page lists 3 October 2026 as download availability, a date
already past at this review. The download link failed through the available web
reader, so the assessment above uses the abstract; it does **not** claim a current
embargo or inspection of equations, sample sizes, or methods. No specific leak,
ramp or reset parameter is imported. No separately citable Janke integration
preprint or peer-reviewed primary article was located in this bounded search.
That is a search limitation, not proof none exists. A dissertation is not a
preprint, and examination of a dissertation is not journal peer review.

Searches used the exact dissertation title; `Janke vDeltaE`, `hDeltaG leaky`,
`vΔE integration`; Janke/integration restricted to bioRxiv; hDeltaG/integration
restricted to PubMed; and the [Maimon lab publications/preprints list](https://maimonlab.rockefeller.edu/publications/).
That list was inspected for publication status, not treated as physiological
evidence. Secondary connectome-mining sites encountered during discovery were
not used to establish functions or to supply counts. The 2026 Maimon–Abbott
review was located but was not treated as a new primary integration experiment.

Names used in queries are literal MaleCNS `type` strings: `hDeltaG` = hΔG,
`vDeltaE` = vΔE, `hDeltaA` = hΔA. The extracted `hemibrainType`, `flywireType`,
instances and body IDs are retained for checking mappings. A type-name match
does not validate a functional coordinate transfer. In particular hDeltaG,
hDeltaA and hDeltaB column indices are not interchangeable angle bins.
