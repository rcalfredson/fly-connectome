# Review walkthrough: directed vDeltaE → hDeltaG contact mapping

Implemented 30 October 2026. Continue after the
[full-population spatial inspection](vdeltae_population_walkthrough.md).
This stage connects the anatomical views to **actual directed contacts** between
the same 18 vDeltaE and eight hDeltaG bodies. It does not implement firing dynamics.

## 1. Sources and scope

The new `fly_connectome/vdeltae_contact_audit.py` loads:

- the verified full-population coordinate run, including all FB site IDs,
  annotations, raw skeletons, mesh and dataset provenance;
- the verified bounded connectivity census, including saved vDeltaE→hDeltaG
  body-pair weights and raw edge `roiInfo`.

`load_sources()` reuses both existing offline verifiers, checks exact body/type/
instance agreement and common dataset metadata, and hashes the source manifests
before and after inspection. The source census contains 118 positive body pairs
and 2,405 standard-weight contacts; these are references to reconcile, not assumed
counts for an unfiltered site query. All **144 possible directed body pairs**
remain present in matrices, including zero-report pairs.

Body IDs remain primary. Repeated C labels and both C0 bodies stay separate.
No anatomical angle, shared grid, sign, weight normalization or dynamic channel
is assigned. The query returns only vDeltaE→hDeltaG, not reverse, recurrent or
newly discovered pathways.

## 2. Contact query and identity

`contact_query(bodies)` constructs a bounded Cypher pattern:

```text
(vDeltaE neuron)-[:Contains]->(SynapseSet)-[:Contains]->(presynaptic site)
    -[contact:SynapsesTo]->(postsynaptic site)
    <-[:Contains]-(SynapseSet)<-[:Contains]-(hDeltaG neuron)
```

Both neuron ID lists are fixed from the audited selection. The query applies
**no confidence or ROI filter**, so it can retrieve contacts with an endpoint
outside FB or contacts below the standard confidence threshold. It also does
not require a positive standard-weight `ConnectsTo` edge: a low-confidence
contact should not disappear merely because its body pair has zero standard weight.

`WITH DISTINCT n, m, a, b, contact` removes repeated containment paths to the
same graph relationship. It does not collapse different partners sharing a
presynaptic site. The result preserves:

- source and receiving body IDs;
- the `SynapsesTo` relationship ID and both site-node IDs;
- both endpoint polarities, native XYZ coordinates and confidence scores;
- both endpoint compartment and `split` properties;
- nullable FB and FBl1–FBl9 flags at both endpoints.

Graph IDs identify records **within this dataset snapshot**, not permanent
biological identities. Raw responses are saved before validation. No compartment
is inferred from polarity, coordinates, SWC node type or proximity.

The CLI also queries the selected live neuron annotations and compares them to
the saved identities. Before/after dataset metadata and final source verification
must agree before a run is marked complete.

## 3. Confidence and ROI scopes

The recent morphology inspection retained all sites. The older census weights
use the dataset's standard connectivity convention. Conflating these counts
would make an incorrect reconciliation test.

`standard_mask()` declares the comparison explicitly: **both endpoint confidence
scores must be strictly greater than the recorded `postHighAccuracyThreshold`**,
currently 0.5. This follows the strict `>` condition used by the installed
neuprint-python `SynapseCriteria` and the repository's earlier validated spatial
workflow. It is a testable comparison convention, not an adjustable fitting rule.
Missing scores stay missing in raw data and do not pass the standard mask.

`contact_scopes()` reports all six combinations:

| Confidence | ROI scope | Meaning |
| --- | --- | --- |
| all_reported | all_ROIs | Every retrieved contact |
| all_reported | post_FB | Receiving endpoint is explicitly FB-marked |
| all_reported | both_FB | Both endpoints explicitly FB-marked |
| standard | all_ROIs | Both confidence scores pass the standard comparison |
| standard | post_FB | Standard contacts whose receiving endpoint is FB-marked |
| standard | both_FB | Standard contacts with both endpoints FB-marked |

These are overlapping views; **never sum scope totals**. Missing FB flags remain
unknown, not false. Tables separately count missing flags and confidence values.
The post-FB view is compared with edge `roiInfo.FB.post`, consistent with the
receiving-site basis of neuPrint connection ROI counts. Layer flags are retained
without summing overlapping layer counts into a contact denominator.

## 4. Validation and table construction

`validate_contacts()` checks:

1. Positive selected body IDs, valid graph IDs, pre→post polarity, finite XYZ,
   nullable confidence in [0,1], and boolean-or-missing ROI flags.
2. No duplicate relationship identities or repeated body/site-pair contacts.
   Multiple contacts sharing a presynaptic site are allowed and counted separately.
3. Consistent properties whenever a site appears in several contacts.
4. Exact agreement of FB endpoint graph IDs and properties with the saved
   full-population site inventory. This is identity-based, not nearest-point matching.
5. Standard-confidence counts for **each body pair**, including zero-report
   pairs, equal saved weights; standard post-FB counts equal saved FB edge counts.

A mismatch stops the run with a failed manifest and retained raw data. The code
does not alter the threshold, discard offending contacts or substitute zeros
to force agreement. Investigate the source convention or schema before retrying.

`contact_tables()` builds:

- `contacts_annotated.csv`: every raw contact plus source/receiver type, instance,
  literal C label and a standard-confidence indicator;
- `scope_counts.csv`: contact and unique endpoint-site counts, raw-core contact
  denominator, scope fractions and counts of missing fields;
- `body_pair_counts.csv`: 144 pairs per scope, site counts, raw body-pair contact
  denominators and fractions, with original C labels for reference;
- `compartment_counts.csv`: receiving-compartment counts and fractions with the
  scope contact total as the explicit denominator;
- total and compartment-specific matrices for every scope, with **source bodies
  as rows and receiving bodies as columns**, matching the saved census orientation;
- `compartment_inventory.csv`: numeric filename suffixes mapped to original
  receiving compartment values. Axon, dendrite and missing annotation are
  included even if absent; other observed labels remain separate.

Undefined fractions have empty CSV values, rather than invented zero fractions.
A zero matrix entry means no contact reported under that scope, not biological
absence. Counts are neither normalized model weights nor physiological efficacy.
Contact counts differ from unique-site counts because presynaptic sites can be polyadic.

`verify_contact_audit()` checks all artifact hashes, source provenance, live/saved
annotations, queries and metadata, then reconstructs the tables from raw contacts.
Changing a derived value and updating its hash therefore still fails verification.
`csv_comparable()` normalizes null spelling only for CSV comparisons; raw null
annotations remain preserved.

## 5. Matrix and contact-location figures

The new `fly_connectome/vdeltae_contact_figures.py` exports offline from a verified
contact audit, into a fresh directory:

- `body_pair_coupling.png/pdf`: unnormalized counts for all-reported contacts,
  standard contacts, and standard FB-post contacts onto annotated axons/dendrites.
  The four panels share a count scale; rows/columns retain body IDs and C labels.
  Other/missing compartments remain included in totals and tables.
- Eight `receiver_BODYID.png/pdf` views: actual standard-confidence FB receiving
  contact positions, colored by the **source vDeltaE C label**. Receiving dendritic
  sites use circles, axonal sites triangles, other/missing compartments crosses.
  Gray background shows all FB sites of that receiver and its full unhealed
  skeleton, with the FB mesh as faint context.
- `summary.md` figure index and a verifiable `plots.json`.

The receiving-site figures preserve contact multiplicity: several contacts at a
shared site can overlap. Titles report contacts and unique receiving sites
separately. Source C colors group labels only for display; they do not collapse
bodies in matrices or infer a shared column grid. In particular, two C0 bodies
can have the same display color while retaining separate matrix rows.

The transform is the saved native XYZ scale in µm, without fitted rotation,
translation, axis permutation, clipping or subsampling. Anatomical orientation
and compartment provenance remain separate questions. Figure verification checks
source integrity, expected inventories and the declared transform/scope.

## 6. Reproduce and verify

Use the project environment from the repository root. Extraction uses the locally
configured `.env` or `NEUPRINT_TOKEN` without printing credentials. Every extraction
or export requires a **new output directory**. Verification is offline and can
reuse existing outputs.

```bash
python -m fly_connectome.vdeltae_contact_audit extract \
  --coordinates results/vdeltae-hdeltag-population-coordinates-20261030 \
  --census results/hdeltab-extension-census-final-20261024 \
  --output results/vdeltae-hdeltag-contact-audit-20261030

python -m fly_connectome.vdeltae_contact_audit verify \
  --run results/vdeltae-hdeltag-contact-audit-20261030

python -m fly_connectome.vdeltae_contact_audit export \
  --run results/vdeltae-hdeltag-contact-audit-20261030 \
  --output results/vdeltae-hdeltag-contact-figures-20261030

python -m fly_connectome.vdeltae_contact_audit verify-figures \
  --run results/vdeltae-hdeltag-contact-figures-20261030
```

The CLI prints safe status/error classes only. After failure, inspect the local
manifest and raw tables, then choose a fresh output name for any retry. Previous
population anatomy, censuses, simulation outputs and presentation exports remain intact.

## 7. Regression checks and interpretation

```bash
python -m pytest -q tests/test_vdeltae_contact_audit.py
```

Synthetic cases cover directed bounds, duplicate identities, shared/polyadic
sites, invalid endpoint membership or properties, strict threshold equality,
missing confidence/ROI/compartment annotations, both endpoint ROI scopes,
per-pair weight/FB reconciliation, explicit zero pairs and denominators, complete/
failed manifests, raw retention, source-output separation, reconstruction after
hash tampering, offline CLI verification, actual matrix/overlay rendering,
source preservation and changed figure transforms.

All **20 contact-audit cases** and the full **361-test repository suite** passed,
with the same 14 previously observed Matplotlib/Pyparsing warnings.

The anatomical question is whether observed directed contacts support the
spatial relationships proposed for the reduced model. Report receiving
compartments and contact locations first. Do not label matrix diagonals
"same-column" across populations solely because their C numbers match; do not
force eight columns, omit C0, fit a permutation to expected opposite-column
wiring, or assign signs from compartment names. Synapse positions and recurrence
do not establish integration, a home vector, a goal signal or steering.

## 8. First saved run

The 30 October live extraction and offline verification completed successfully.
The query returned **2,405 contacts**, using 1,612 unique presynaptic sites and
2,405 unique receiving sites. All-reported and standard-confidence counts are
identical in this result; retaining both scopes makes the comparison explicit.

| Standard-confidence ROI scope | Contacts | Receiving axon | Receiving dendrite |
| --- | ---: | ---: | ---: |
| all_ROIs | 2,405 | 1,383 | 1,022 |
| post_FB | 2,402 | 1,381 | 1,021 |
| both_FB | 2,400 | 1,379 | 1,021 |

All body-pair standard counts match the saved census, and all FB-post counts
match the corresponding saved edge ROI counts. No receiving compartment labels
are missing in this run. Five contacts lack an FB flag at the presynaptic end;
three lack it at the receiving end. These are **missing annotations**, not
demonstrated outside-FB locations. Two contacts in the FB-post subset lack the
presynaptic FB flag, explaining the difference between post_FB and both_FB.

Saved data: `results/vdeltae-hdeltag-contact-audit-20261030`.
Saved figures: `results/vdeltae-hdeltag-contact-figures-20261030`.
Start with `body_pair_coupling.png`, then the receiver views linked in the figure
summary. The compartment matrices expose which source bodies contribute to
each receiving arbor. They provide anatomical constraints for a subsequent
model specification; they do not validate positive axonal versus negative
dendritic effects.

## 9. Source-compartment heatmaps with one color scale

The optional `--source-compartments` export adds
`source_compartment_coupling.png/pdf`: two side-by-side heatmaps split by
**vDeltaE's presynaptic-site compartment** (`compartment_pre`). This differs
from the earlier figure, which splits by the **receiving hDeltaG compartment**
(`compartment_post`). All hDeltaG receiving compartments are included in these
two new panels.

The comparison retains the standard-confidence, FB-marked-receiving-endpoint
scope. It preserves all 18 source rows and eight receiver columns, ordered by
annotation label/body ID, including both C0 bodies. Counts are not normalized.
Both images share the same `Normalize` object, limits from zero to their common
maximum body-pair count, and **one colorbar**. Zero-count panels remain visible.
Other or missing source compartments, if present, are counted in the subtitle
and remain in the underlying audit rather than being reassigned.

`source_compartment_matrices()` saves the corresponding
`source_axon_matrix.csv` and `source_dendrite_matrix.csv`.
`source_compartment_figure()` renders them. Figure verification checks both
additional matrices against the verified raw contacts. Existing export commands
and previously saved figures remain supported without the option.

```bash
python -m fly_connectome.vdeltae_contact_audit export \
  --run results/vdeltae-hdeltag-contact-audit-20261030 \
  --output results/vdeltae-hdeltag-contact-source-compartments-20261030-v2 \
  --source-compartments

python -m fly_connectome.vdeltae_contact_audit verify-figures \
  --run results/vdeltae-hdeltag-contact-source-compartments-20261030-v2
```

In the saved data, the panels contain **2,339 contacts from source sites annotated
axon** and **63 from source sites annotated dendrite** (total 2,402). A dark
dendritic panel on this shared scale reflects smaller counts, not an absent
connection or reduced physiological efficacy. These are presynaptic sites in
annotated dendritic compartments; the compartment name does not change their
recorded pre polarity.

Two additional regression cases explicitly distinguish source from receiving
compartment, check the common normalization object and single colorbar, retain
the full matrix shape and test an empty source-dendrite panel. The export
round-trip case also verifies the optional figure and its saved matrices.
All 22 contact-audit cases pass, with the previously observed dependency warnings.

## 10. Streamlined receiving-compartment comparison

For a side-by-side version of the **lower two panels of the original
`body_pair_coupling.pdf`**, use `--receiving-compartments`. This splits contacts
by **hDeltaG's receiving compartment** (`compartment_post`), includes all vDeltaE
source compartments, and retains the original standard-confidence, FB-post scope.
It is distinct from the optional source-compartment comparison above.

The figure `receiving_compartment_coupling.png/pdf` has two heatmaps and one shared
colorbar. Both use the same normalization from zero to their common maximum
body-pair count. Rows, columns and individual contact counts exactly match the
original lower two panels; only presentation and the common scale are streamlined.
`receiving_axon_matrix.csv` and `receiving_dendrite_matrix.csv` preserve the counts.

```bash
python -m fly_connectome.vdeltae_contact_audit export \
  --run results/vdeltae-hdeltag-contact-audit-20261030 \
  --output results/vdeltae-hdeltag-contact-receiving-compartments-20261030 \
  --receiving-compartments

python -m fly_connectome.vdeltae_contact_audit verify-figures \
  --run results/vdeltae-hdeltag-contact-receiving-compartments-20261030
```

The saved panels contain **1,381 contacts onto hDeltaG axonal sites** and
**1,021 onto dendritic sites**. These counts describe annotated receiving
compartments, not positive/negative effects. Other/missing receiving compartments,
if present, remain counted in the figure subtitle and underlying audit. The
other/missing title line is omitted when its count is zero, for both optional
comparisons. The updated receiving figure is saved in
`results/vdeltae-hdeltag-contact-receiving-compartments-20261030-v3`.
The paired heatmaps use one shared Y-axis label while retaining the C-label/body-ID
ticks beside both panels. X-axis labels remain repeated.

The new regression case compares both displayed arrays directly with the
original lower panels, deliberately changes source compartments to check that
classification depends on the receiving side, and checks the single shared
normalization and colorbar. The export round trip covers both optional comparisons.
All 23 contact-audit cases pass with the same known dependency warnings.
