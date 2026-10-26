# vDeltaE input survey — 24 October 2026

**Direction updated 26 October 2026:** see the [current modeling plan](vdeltae_hdeltag_next_steps.md).
The census below is retained unchanged. The [input profiles](vdeltae_input_profiles.md)
provide the subsequent literature pass over ten selected populations.

The next investigation concerns possible anatomical inputs to vDeltaE and,
separately, the computation of the vDeltaE–hDeltaG motif. It is no longer
committed to a small direct extension of hDeltaB. The existing prescribed-PFN
model remains a useful baseline, but connection to the motif must be justified.

## Reading the thesis notes

The notes accurately summarize the publicly accessible abstract, with these
qualifications:

- The title is **A Neuronal Circuit Motif for Leaky Vector Integration**, by
  Abby Janke (2025), a dissertation. The [stable thesis record](https://digitalcommons.rockefeller.edu/student_theses_and_dissertations/808/)
  is preferable as a citation to the download endpoint.
- Attribute the proposed involvement of vDeltaE to the author's imaging and
  perturbation evidence. It does not establish that this is hDeltaG's only input
  or identify hDeltaB's contribution.
- Varying motif parameters across circuits is a proposed generalization,
  distinct from the specific reported experimental results.
- Keep recent-distance/displacement-related memory distinct from a calibrated
  neural home vector. In particular, distinguish net displacement from distance
  traveled along a path when reviewing the experiments and model.

The linked PDF returned HTTP 403 through the web reader in this session. These
editorial checks therefore use the abstract, not an examination of the full
results or equations. No physiological mechanism or parameter was imported.

## Extraction and validation

New outputs: `results/vdeltae-inputs-live-20261024/`. The earlier hDeltaB census,
movement/trajectory outputs and presentation exports are unchanged. No neural
dynamics were added and nothing was committed.

This is an exact-type, one-hop incoming census of all 18 selected vDeltaE cells
in `male-cns:v1.0`. The saved candidate census supplies the membership/instance
baseline and metadata. All incoming `Segment` sources are retained with standard
weight ≥1, including untyped segments, recurrence and self-connections if
reported. No reconstruction, hemisphere, ROI or cropping filters apply.

There are **1,118 source bodies**, **3,314 positive body pairs** and **14,583
incoming contacts**. Every target's incoming sum equals its recorded `post`
count. Dataset UUID/mutation metadata match the earlier census; the reference
PFNd 26913→hDeltaB 515192 remains 137. Four subsets independently match the
earlier candidate-pair queries: hDeltaB, hDeltaC, hDeltaG and vDeltaE recurrence.
These checks establish internal dataset consistency, not biological completeness.

Raw data, source annotations, cross-dataset labels, status flags, body IDs and
edge ROI JSON are preserved. ROI counts are not aggregated or interpreted as
shared functional columns. Missing annotations remain unknown; untyped fragments
are included. The untyped population contributes 408/14,583 = **2.798%**, across
267 source bodies. Source body counts can include disconnected fragments of
biological neurons and must not automatically be read as whole-neuron counts.

## Results

| Source population | Contacts | Share of all 14,583 inputs | vDeltaE targets reached |
| --- | --- | --- | --- |
| hDeltaC | 1,604 | 10.999% | 18/18 |
| FB6N | 1,173 | 8.044% | 18/18 |
| FB5AB | 1,103 | 7.564% | 18/18 |
| FB6O | 660 | 4.526% | 18/18 |
| hDeltaG | 653 | 4.478% | 18/18 |
| FB6E | 611 | 4.190% | 18/18 |
| FB5I | 583 | 3.998% | 18/18 |
| FB5L | 562 | 3.854% | 18/18 |
| PFNp_c | 497 | 3.408% | 18/18 |
| PFNp_b | 303 | 2.078% | 18/18 |
| PFNd (existing modeled 40 cells) | 1 | 0.00686% | 1/18 |
| PFNv (existing modeled 20 cells) | 109 | 0.747% | 16/18 |
| hDeltaB (existing modeled 19 cells) | 57 | 0.391% | 12/18 |

The complete rankings are in `population_inputs.csv`; `summary.md` displays the
top 25. Population share divides by all incoming Segment weights across all
selected targets. Coverage divides by all 18 targets, retaining zeros.
`population_by_target.csv` uses each target's own all-Segment incoming weight
as its fraction denominator; undefined zero-denominator fractions are blank.

Overlap with the existing model is checked by body ID, not only type name.
PFNd has 1/40 connected source cells, PFNv 17/20, and hDeltaB 13/19. Their three
matrices retain every modeled source and all 18 targets. Combined, these existing
populations contribute **167/14,583 = 1.145%** of recorded vDeltaE input. This
sum describes distinct anatomical presynaptic populations; it is not three
independent functional signals, since hDeltaB is already downstream of PFNs in
our model. The census did not extract additional paths through hDeltaC or PFNp.

## Interpretation and next step

There is direct anatomical overlap, but these data do not justify driving
vDeltaE solely with the existing prescribed-PFN→hDeltaB model. The largest
single typed population here is hDeltaC, with about 11% of input, not a majority.
The aggregate pattern includes substantial input from multiple tangential
populations, hDeltaG feedback and PFNp subtypes. Counts do not identify which
inputs determine vDeltaE tuning, physiological signs, efficacy, integration,
behavioral context or ramp/reset regulation. Small anatomical weight also does
not prove a negligible physiological effect.

Following the input profiling, the bounded next step is to extract the thesis's
vDeltaE recordings, stimulus conditions and model equations into a simulation
specification. The [current plan](vdeltae_hdeltag_next_steps.md) separates this
mechanism assessment from reconstruction of upstream drive. Whether controlled
prescribed vDeltaE activity is sufficient, and whether explicit feedback is
needed, must be decided from that assessment before implementation. Further
anatomical queries should address specific missing inputs rather than expand
every leading population or fit its activity to the desired output.

## Reproducibility

With the existing project environment activated, run from the repository root:

```bash
python -m fly_connectome.vdeltae_inputs \
  results/hdeltab-extension-census-final-20261024 \
  --movement results/pfn-hdeltab-movement-groupnorm-pfnd6-v1.0 \
  --output results/vdeltae-inputs-live-20261024
python -m fly_connectome.vdeltae_inputs \
  results/vdeltae-inputs-live-20261024 --verify-only
python -m pytest -q tests/test_vdeltae_inputs.py tests/test_circuit_extension.py tests/test_inputs.py
```

The output directory already exists; choose a new name for a new extraction.
The extractor follows the existing local `.env` authentication convention without
printing credentials. Source data hashes are checked before/after; dataset
metadata, exact queries, package/code hashes and safe failure metadata are saved.
Offline verification reconstructs summaries and modeled-population matrices,
rechecks denominators and verifies file hashes. The relevant test run passed
**42 tests**, covering zero targets, untyped sources, self-connections, malformed
data, disconnected modeled sources, reconciliation and hash tampering.

A serialization check initially compared a sorted DataFrame's incidental row
index with the fresh CSV row index. The verifier was corrected to compare rows
after resetting that unsaved index; the live census then passed offline
verification with unchanged data. A separate failed network attempt remains in
`results/vdeltae-inputs-20261024/`. Only complete, verified data should be used.
