# Review walkthrough: full vDeltaE and hDeltaG populations

Implemented 30 October 2026. Continue after the
[two-neuron FB mesh context pilot](vdeltae_fb_mesh_context_walkthrough.md).
This stage expands anatomical inspection to the **18 audited vDeltaE and eight
audited hDeltaG bodies**. It adds no neural dynamics or physical column assignment.

## 1. Expand the selection, reuse the verified pipeline

In `fly_connectome/vdeltae_spatial_audit.py`, `select_spatial_population()` retains
every row of the verified annotation audit. It checks exact population counts,
positive unique body IDs and both types, then sorts by population/body ID.
Repeated column labels, vDeltaE C0/C9, unresolved names and source fields are
retained. Membership comes from the saved audit, not geometry or connectivity.

`run_spatial_data_probe(..., full_population=True)` uses that selection and
records the scope in `capability.json`. The four original metadata/site queries
now accept an explicit 26-body mode. Their default still requires exactly two
bodies; neither mode permits an unrestricted graph expansion. Full-population
verification reconstructs the selection from the original audit and compares
every row, the recorded IDs, queries and resource inventory.

The runner fetches four query tables, **26 raw unhealed SWCs and one FB mesh**.
Before/after metadata must match the saved dataset state. A missing skeleton is
recorded by the availability stage, but coordinate extraction requires every
selected skeleton: a partial set is not presented as full-population morphology.
Failures retain raw artifacts and a failed manifest where possible.

`run_coordinate_pilot()` inherits scope from that verified source. Its existing
name and saved filenames remain for compatibility. It queries distinct FB sites
for the same 26 bodies, reconciles each body's counts and categories, copies raw
morphology byte-for-byte and builds the existing descriptive CSVs. No confidence
cutoff, skeleton healing, subsampling or connection-sign assignment is introduced.
The denominator is **distinct body/Synapse-node sites**, not synaptic contacts.

Raw coordinate CSVs load with `low_memory=False`: inference across the whole
table avoids chunk-dependent types for sparse optional layer flags in the larger
population export. All existing source hashes and count checks still apply.

## 2. Population analyses and views

New module `fly_connectome/vdeltae_population.py` supplies three additions:

1. `population_spatial_summary()` produces `column_spatial_summary.csv` with one
   row per body/compartment. It carries body ID, type, instance and annotation
   column; site count, pre/post counts, total FB sites and fraction of that body's
   FB sites; and XYZ medians and 10th/90th percentiles in µm. Axon, dendrite,
   linker and unannotated rows are included even when absent. Other observed
   compartment names remain separate. Zero-count groups have missing coordinate
   statistics; a body with no FB sites has an undefined fraction, not zero.
2. `population_projection_figure()` shows **all sites** in a compartment for one
   population against the FB mesh in native XY/XZ/YZ projections. Colors denote
   unchanged C labels, circles denote pre sites and crosses denote post sites.
   All bodies sharing a label remain in the plot; they are not collapsed into
   one neuron. Legends state the number of bodies carrying each label.
3. `column_profile_figure()` orders individual bodies by annotation label and
   compares axonal and dendritic site medians and 10–90% ranges on each native
   axis. Each row is a body, including duplicate labels. An absent compartment
   has no coordinate marker. Ranges describe site distributions, not measurement
   uncertainty or estimated physical column centers.

`population_figure_specs()` declares filenames and page membership. Histograms
are split into pages of at most six bodies, ordered by label/body ID, rather than
compressing 26 neurons into the old two-neuron canvas. The figure summary links
each population view and identifies its compartment; numeric filename suffixes
avoid inserting unchecked source strings into file paths.

`export_coordinate_pilot(..., mesh_overlay=True)` detects full-population scope
from the verified coordinate source and exports:

- 26 individual PNG/PDF body figures, with all sites and full skeletons;
- separate population/compartment overviews and axon/dendrite profile figures;
- five histogram pages (three vDeltaE, two hDeltaG), plus the FB vertex preview;
- the population summary, mesh-range diagnostics, readable summary and manifest.

Full-population exports require mesh context. Geometry uses the same explicit
native scale `[0.008, 0.008, 0.008]` µm with no fitted transform or clipping.
Source and output directories must be separate and new. Offline verification
checks the expected figure inventory and reconstructs both diagnostic CSVs.
Old two-neuron runs and exports still verify without the new scope field.

## 3. Reproduce extraction

From the repository root in the project environment. This uses locally configured
credentials without printing them; do not paste credentials into the command.
Use fresh directory names if the example outputs already exist.

```bash
python - <<'PY'
import os
from pathlib import Path
from dotenv import load_dotenv
from neuprint import Client
from fly_connectome.connectivity import SERVER, DATASET, ValidationError
from fly_connectome.vdeltae_spatial_audit import (
    run_spatial_data_probe, verify_spatial_data_probe,
)
from fly_connectome.vdeltae_spatial_pilot import (
    run_coordinate_pilot, verify_coordinate_pilot,
)

load_dotenv(Path.cwd() / ".env", override=False)
token = os.environ.get("NEUPRINT_TOKEN", "").strip()
if not token:
    raise ValidationError("Configure NEUPRINT_TOKEN locally as described in the README.")
try:
    client = Client(SERVER, dataset=DATASET, token=token)
    source = "results/vdeltae-hdeltag-population-spatial-20261030"
    coordinates = "results/vdeltae-hdeltag-population-coordinates-20261030"
    run_spatial_data_probe(
        client, "results/vdeltae-hdeltag-annotation-audit-20261029", source,
        full_population=True,
    )
    verify_spatial_data_probe(source)
    result = run_coordinate_pilot(client, source, coordinates)
    verify_coordinate_pilot(coordinates)
except Exception as error:
    print("Population extraction stopped:", type(error).__name__, "— inspect saved manifests locally.")
    raise SystemExit(1) from None
print("Verified:", len(result["selected_body_ids"]), "bodies;", result["point_count"], "FB sites")
PY
```

The live run on 30 October saved all 26 bodies and **49,149 FB sites**. Both
metadata snapshots matched the saved annotation sources. Saved outputs are
separate from the earlier pilot, census results and presentation exports.

## 4. Render and verify offline

```bash
python - <<'PY'
from fly_connectome.vdeltae_spatial_pilot import (
    export_coordinate_pilot, verify_coordinate_exports,
)
source = "results/vdeltae-hdeltag-population-coordinates-20261030"
output = "results/vdeltae-hdeltag-population-figures-20261030"
result = export_coordinate_pilot(source, output, mesh_overlay=True)
verify_coordinate_exports(output)
print("Population figures:", result["status"], output)
PY
```

To verify an existing export, run only `verify_coordinate_exports(output)`;
export intentionally refuses an existing output directory. Read `summary.md`
first, then compare both profile figures and corresponding compartment views.
Use individual body figures to investigate unusual distributions or edge labels.

## 5. Validation and interpretation

```bash
python -m pytest -q tests/test_vdeltae_population.py tests/test_vdeltae_spatial_audit.py tests/test_vdeltae_spatial_pilot.py
```

The population tests cover complete membership, repeated/edge labels, invalid
or unbounded selections, preservation of pilot defaults, count denominators,
missing compartments and zero-site bodies, all-site plotting, complete histogram
page coverage, and an end-to-end synthetic 26-body extraction/export/verification.
They also reject changed membership and incorrect summary values even when an
output hash is updated. The integration test skips repeated image encoding;
a separate real-render test and inspection of actual exports check presentation.
All 341 repository tests passed after this extension (including 11 population
cases), with the 14 previously observed Matplotlib/Pyparsing warnings. The earlier
mesh-free and mesh-context pilot exports also verified without file changes.

These plots test whether dendritic locations are consistent with the label
ordering; they do not define the labels' anatomical convention. Dendritic
indexing is documented for hemibrain hDelta neurons in
[Hulse et al., Figure 37](https://elifesciences.org/articles/66039#fig37), which
supports a hypothesis to examine in MaleCNS, rather than establishing a shared
vDeltaE/hDeltaG grid. Native XYZ axes are not labeled anatomical directions.
Dataset compartment labels remain annotations whose provenance needs checking.
Inspect C0/C9 and bodies sharing labels separately; do not wrap labels modulo
eight, force uniform angular spacing, or fit a permutation to expected wiring.

Only after reviewing the spatial evidence should the saved vDeltaE→hDeltaG
connectivity be used to challenge a proposed spatial mapping. No new connectivity
expansion, movement encoding, firing model or integration claim is part of this stage.

## 6. Initial observations from the full-population run

| Population | Bodies | FB sites | Sites labeled axon | Sites labeled dendrite | Other compartment labels |
| --- | ---: | ---: | ---: | ---: | ---: |
| vDeltaE | 18 | 16,528 | 12,069 | 4,459 | 0 |
| hDeltaG | 8 | 32,621 | 12,393 | 20,109 | 119 |

The 119 other hDeltaG sites comprise 99 labeled `cell-body-fiber` and 20 labeled
`linker`. These are graph sites, including pre and post sites within each dataset
compartment; axon sites should not be read as exclusively outputs or dendritic
sites as exclusively inputs.

For hDeltaG, dendritic median native X increases from 345.344 µm at C1 to
425.984 µm at C8, in annotation order. Axonal X locations have a shifted ordering,
with a large jump between C4 and C5. Together with the compartment views, this
is consistent with dendritic indexing and spatially offset axonal arbors. It is
not an independently verified definition of the annotation convention or a
calibrated eight-angle model.

vDeltaE needs a separate interpretation. In particular, the two C0 bodies have
different dendritic X medians: 413.328 µm (79146) and 367.388 µm (145879). Their
axonal distributions span widely separated native X locations, visible in the
individual views and broad quantile ranges. A single median or a shared `C0`
label would hide this structure. These observations do not yet tell us whether
`C0` denotes an edge-spanning class, an annotation convention, or something else.
The current data do not justify a simple shared vDeltaE/hDeltaG index grid.

The final reviewed export is
`results/vdeltae-hdeltag-population-figures-20261030-v2`; it moves profile legends
above the data rows. The first export is retained separately. Start with its
`summary.md` figure index, `hDeltaG_column_profiles.png`,
`vDeltaE_column_profiles.png`, and `body_79146.png` / `body_145879.png` for C0.

Review these against `column_spatial_summary.csv` and the individual raw site
clouds before interpreting label geometry. The next step is to resolve the
vDeltaE labeling exceptions and document a proposed spatial specification,
then challenge it with the already-saved directed connectivity.
