# fly-connectome

This repository investigates neural circuitry and computations in the Drosophila
connectome, initially focusing on MaleCNS navigation circuits, including
PFN-associated directional representations and candidate downstream integration
mechanisms. The project begins with anatomical connectivity analysis and aims to
progress toward connectome-constrained neural simulations.

The first milestone extracts and visualizes raw connectivity between PFNd and
hDeltaB neurons in **`male-cns:v1.0`**, using the official
[neuprint-python package](https://github.com/connectome-neuprint/neuprint-python)
against `neuprint.janelia.org`. Simulation infrastructure is outside this milestone.

## Setup

Use a project-specific conda environment, from the repository root:

```bash
conda env create -f environment.yml
conda activate fly-connectome
```

The environment specifies Python 3.11 and installs pinned runtime and test
dependencies with pip. Alternatively, with Python 3.11 installed:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

## Authentication

Log in to [neuPrint](https://neuprint.janelia.org/) and copy your authentication
token from the account menu. Create a local dotenv file:

```bash
cp .env.example .env
```

Edit `.env` locally to set `NEUPRINT_TOKEN` to your token. Do not commit or share
this file, paste the token into chat, or put it in a command-line argument.
`.env` and other `.env.*` files are ignored; `.env.example` contains no credential.
Avoid placing credentials in other files or force-adding ignored files.

An exported `NEUPRINT_TOKEN` also works and takes precedence over `.env`, even if
the exported value is empty. Only `.env` in the current working directory is
loaded, and only during extraction. The token is passed explicitly to `Client`;
the workflow does not rely on neuprint-python's differently named default
credential environment variable. Tokens are never written to outputs or printed.

## Extract and validate

Run from the repository root with the environment activated:

```bash
python -m fly_connectome extract --output results/pfnd-hdeltab-v1.0
```

Use a **new output directory** for each run; existing directories are refused to
avoid mixing results. The workflow:

1. Saves a dataset-wide type/count inventory and checks related annotations for
   the exact labels `PFNd` and `hDeltaB`. Related spelling or subtype variants
   cause a stop for review; instance names never determine membership.
2. Fetches the actual type and instance annotations of bodies `26913` and
   `515192`, checking that they are PFNd and hDeltaB respectively.
3. Selects all neurons with those exact `type` annotations. Prints and saves
   matched types, neuron counts, and distributions of available `status`,
   `statusLabel`, and `cropped` fields, including unknown values. No status,
   cropped, hemisphere, or ROI filters are imposed. The web-interface count of
   **40 PFNd neurons** is a sanity check: a difference is explicitly reported
   and saved, but does not itself change membership or halt the run.
4. Checks the directed reference **PFNd 26913 → hDeltaB 515192**, expecting the
   standard connection `weight` to equal **137**. Annotation or weight
   discrepancies stop the run before population connectivity extraction.
5. Queries both PFNd → hDeltaB and hDeltaB → PFNd, preserving standard integer
   synapse counts. The forward population query is checked against the reference
   again. `fetch_adjacencies(omit_rois=True, weight_props=['weight'])` retrieves
   direct totals without summing overlapping ROI counts. Every reported positive
   connection is included (`min_total_weight=1`), without normalization or
   confidence/polarity alternatives such as `weightHP`.
6. Builds matrices over the complete population lists, including disconnected
   neurons and zero-weight pairs, and saves CSVs and heatmaps.

**A zero means no connection was reported in the queried dataset. It does not
establish biological absence.** Failed queries are never converted into zero
matrices. These anatomical counts do not specify physiological efficacy.

Both matrices use presynaptic neurons as rows and postsynaptic neurons as
columns, ordered by ascending numeric body ID. Heatmaps label every body ID,
show a linear synapse-count color scale, and are saved as PNG and PDF. Each
direction has its own color scale.

## Outputs and reproducibility

Each run directory contains:

| File | Contents |
| --- | --- |
| `type_inventory.csv` | Dataset type annotations and counts |
| `PFNd_neurons.csv`, `hDeltaB_neurons.csv` | Complete populations and available reconstruction annotations |
| `population_summary.json` | Printed summary, including the 40-neuron sanity check |
| `reference_neurons.csv`, `reference_connections.csv` | Reference annotations and connection totals |
| `reference_validation.json` | Expected and observed reference weight, with pass/fail |
| `population_reference_validation.json` | Reference check within the full forward query |
| `PFNd_to_hDeltaB_connections.csv`, `hDeltaB_to_PFNd_connections.csv` | Returned positive totals, one row per pair |
| `PFNd_to_hDeltaB_matrix.csv`, `hDeltaB_to_PFNd_matrix.csv` | Full matrices with zero-filled missing pairs |
| `PFNd_to_hDeltaB.{png,pdf}`, `hDeltaB_to_PFNd.{png,pdf}` | Heatmaps |
| `metadata.json` | Run status, UTC times, server/dataset, dataset metadata, selection/query settings, axis semantics, Python/package versions, Git commit and dirty flag |

Inspect `metadata.json`: only `status: complete` denotes a successful run.
Failed runs retain available diagnostic files; later outputs may be absent.
Generated results are ignored by Git. Preserve the run directory together with
the code revision. A dirty flag signals that the commit alone does not capture
the code used; archive those local changes too. Direct dependencies are pinned,
with key installed versions recorded per run; this is not a full cross-platform
environment lock. Dataset metadata is recorded because annotations can change.

Replot saved data without a token or server query:

```bash
python -m fly_connectome plot results/pfnd-hdeltab-v1.0/PFNd_to_hDeltaB_matrix.csv \
  --output-prefix results/replot/PFNd_to_hDeltaB \
  --source-type PFNd --target-type hDeltaB
```

The Python plotting function accepts `row_order` and `column_order` as complete
permutations of integer body IDs. This supports future anatomical orderings
without changing extraction or the saved matrix:

```python
from fly_connectome.plotting import load_matrix, plot_heatmap

matrix = load_matrix("results/pfnd-hdeltab-v1.0/PFNd_to_hDeltaB_matrix.csv")
plot_heatmap(matrix, "results/reordered", "PFNd", "hDeltaB",
             row_order=list(reversed(matrix.index)),
             column_order=list(matrix.columns))
```

## Anatomical ordering from saved annotations

Generate additional artifacts offline from a complete extraction:

```bash
python -m fly_connectome anatomical results/pfnd-hdeltab-v1.0 \
  --output results/pfnd-hdeltab-anatomical-v1.0
```

The output directory must be new. The source extraction, body-ID heatmaps, and
raw matrices are preserved. Population membership still comes from exact type
annotations. Every neuron remains individually represented, including neurons
with identical instances and neurons with no reported connections.

PFNd rows are sorted by numeric FB column, PB side (L before R), then body ID.
hDeltaB columns are sorted by numeric FB column, then body ID. The reverse
heatmap reuses these population orders with axes exchanged. Body ID breaks ties
reproducibly and has no anatomical meaning; the hDeltaB instance number preceding
`_C` is retained for reference but is not a sort key. Unexpected or missing
instances, duplicate bodies, and inconsistent PFNd PB/FB mappings stop processing.
The initial parser explicitly supports the inspected MaleCNS v1.0 conventions:
`PFNd(PB04)_[LR]#_C#` and `hDeltaB_##_C#`.

**PFNd C1–C8 and hDeltaB C1–C12 are within-type annotation indices. They have
not been established as directly comparable spatial coordinates.** The plot
does not imply that matching C numbers align anatomically, assign angular
coordinates, or identify the particular hDeltaB arbor indexed by C. Parenthetical
`PB04` is a historical identifier, not a glomerulus index. See the
[MaleCNS annotator's naming clarification](https://groups.google.com/g/neuprint/c/HrafALsJ13A).

| Additional output | Contents |
| --- | --- |
| `PFNd_anatomical_order.csv`, `hDeltaB_anatomical_order.csv` | One-based order, body ID, original instance, parsed column/PB indices, source and display label |
| `*_anatomical_matrix.csv` | Complete raw-count matrix with only axes permuted |
| `*_anatomical.{png,pdf}` | Instance/body-ID labels and boundaries between anatomical column groups |
| `anatomical_ordering.json` | Completion status, sort conventions, caveat, source metadata and SHA-256 hashes, validation results |

Each heatmap retains the same linear zero-to-maximum color limits as its original
direction. Before writing artifacts, the workflow reconstructs both matrices
from saved connections and population lists, verifies every entry and axis,
checks that anatomical ordering is a complete permutation and reverses without
changing values, and verifies `26913 → 515192 = 137`. Input hashes are checked
again after plotting. Only `status: complete` in the new manifest denotes success.

Ordering uses annotations alone, without connectivity clustering, normalization,
or aggregation. This independence describes the implementation; the original
curation's independence from connectivity evidence remains unverified. Generic
body-ID permutations and body-ID-keyed plot labels/groups can also support future
morphology-derived orders without modifying extraction. The anatomical-ordering
command does not use morphology; the spatial pilot below inspects it separately.

## PFNd → hDeltaB column aggregation

Aggregate the saved forward matrix offline, using the reviewed anatomical tables:

```bash
python -m fly_connectome columns results/pfnd-hdeltab-v1.0 \
  --anatomical results/pfnd-hdeltab-anatomical-v1.0 \
  --output results/pfnd-hdeltab-columns-v1.0
```

The output directory must be new. Rows follow PFNd C1–C8 and columns follow
hDeltaB C1–C12 numerically, with no shifting, reversal, or clustering. These
type-specific annotation indices do not establish shared physical locations;
the summaries alone do not establish a directional transformation. Dynamics,
path integration, morphology, and synapse locations are outside this analysis.

The annotation caveat above the column heatmaps is hidden by default. Add
`--show-column-caveat` to display it. The caveat remains recorded in the manifest
and ranked table regardless of this plot setting.

For each of the 96 column pairs, total weight sums every directed neuron-pair
synapse count. Mean weight divides that total by **all possible pairs**
(`source_neurons × target_neurons`), including zeros. Connected pairs count
weights greater than zero, and connection density is connected / possible pairs.
Zero means no connection was reported in the queried dataset, rather than proof
of biological absence. Counts do not specify physiological efficacy.

| Output | Contents |
| --- | --- |
| `PFNd_to_hDeltaB_column_{total,mean}_matrix.csv` | Two 8 × 12 matrices |
| `PFNd_to_hDeltaB_column_{total,mean}.{png,pdf}` | Heatmaps with linear zero-to-maximum scales and explicit units |
| `PFNd_to_hDeltaB_column_summary.csv` | All 96 pairs, total/mean weight, population/pair counts, density, and both ranks |
| `PFNd_to_hDeltaB_column_rankings.md` | Top three positions for each source column by total and mean, with ties included |
| `column_aggregation.json` | Formulas, input SHA-256 hashes, dataset, caveats, and validation results |

Ranks descend within each PFNd column. Ties share competition rank (1, 1, 3),
and tied entries display in numerical hDeltaB column order. The ranked table
includes all entries at rank ≤ 3; the summary CSV retains every partner.

Before writing, validation requires exactly 40 unique PFNd and 19 unique hDeltaB
neurons, all 8/12 groups, agreement of saved order tables with parsed population
annotations, and equality of the forward matrix with both saved edges and the
anatomically ordered matrix. Aggregation must conserve total weights, possible
pairs, and connected pairs. SHA-256 hashes of CSV/JSON inputs in both source
directories (including both original matrices and ordering tables) are checked
again after output generation. Only a manifest with `status: complete` denotes
success; existing source artifacts are preserved.

## FB spatial pilot

Retrieve a bounded sample of **seven neurons and five directed pairs** from a
complete saved extraction:

```bash
python -m fly_connectome spatial results/pfnd-hdeltab-v1.0 \
  --output results/pfnd-hdeltab-spatial-pilot-v1.0
```

Pairs are `26913 → 515192`, `77360 → 515192`, `26913 → 27109`,
`26913 → 513702`, and `25982 → 11512`. The command checks live type/instance
annotations against saved populations and each full contact count against the
saved forward matrix. It fetches FB-restricted pre/post points with explicit
`FBl1`–`FBl9` flags, all-ROI contacts for each pair, individual unhealed skeletons,
and FB/FBl3/FBl4 visualization meshes. It does not fetch other connectivity or
the full connectome. Use a new output directory; failed runs retain diagnostics.

The reference must have **137 total contacts, 135 FB-post contacts, and 133 with
both endpoints in FB**. These are different scopes, not inconsistent counts.
Standard confidence uses `postHighAccuracyThreshold=0.5` with the pinned client's
strict `>` comparison on both contact endpoints; `weightHP` is not used. Changed
thresholds or reference ROI counts stop the pilot for review. Individual FB/layer
point counts are compared with neuron ROI statistics, and every FB contact
endpoint must be present in the corresponding neuron's point table.

`pre`/`post` denotes a synapse's polarity; `compartment` denotes the dataset's
axon/dendrite/linker assignment. Axonal arbors can receive postsynaptic input.
Labels are provisional anatomical assignments, not functional ground truth.
Explicit layer flags come from a body-restricted custom query to avoid losing
overlapping ROI labels in the installed client's individual-synapse row assembly.

Outputs include:

| File | Contents |
| --- | --- |
| `neurons.csv`, `live_annotations.csv` | Parsed saved column annotations and live reconstruction annotations |
| `neuron_roi_counts.csv` | Live per-neuron ROI counts used for validation |
| `fb_synapses.csv` | Unique FB sites: raw XYZ, polarity, confidence, compartment, split property and nine layer flags |
| `contacts.csv` | All five pairs' contacts, both endpoints, ROI lists encoded as JSON, FB/layer flags and compartment labels |
| `pair_counts.csv` | Total, FB-post and both-FB counts; unique sites counted separately from contacts |
| `arbor_spatial_summary.csv` | Per-body/polarity/compartment/ROI counts, XYZ mean/median, population SD, range and 10th/90th percentiles in µm |
| `skeletons/*.csv`, `meshes/*.obj` | Unhealed centerlines and visualization geometry |
| `pair_*.{png,pdf}` | Common-scale XY/XZ/YZ projections and a 3D view for each pair |
| `spatial_pilot.json` | Completion status, metadata, units, query settings, checks, code/package versions and source/artifact SHA-256 hashes |

Raw coordinates are native MaleCNS EM voxels (8 nm isotropic); summaries and
plots multiply by `0.008` to report micrometers. Axis labels remain native XYZ;
anatomical axis orientation is unverified. The plots show FB points, arbor labels,
pair post endpoints, and skeletons clipped to the view box. A clipped skeleton
segment is not necessarily inside the FB ROI. ROI meshes are visual context only,
never used for quantitative containment or distances. FB/layer summary rows
overlap: do not sum parent and child rows. No shared column coordinate, angular
conversion, or hDeltaB indexing-arbor conclusion is inferred automatically.

Replot a completed pilot offline into a new directory:

```bash
python -m fly_connectome spatial-plot results/pfnd-hdeltab-spatial-pilot-v1.0 \
  --output results/pfnd-hdeltab-spatial-replot-v1.0
```

Only `status: complete` in the spatial manifest denotes a successful extraction,
validation and plotting run. A dirty Git flag means the commit alone does not
capture the implementation; the manifest also records hashes of package source
files. Preserve local code changes along with the run.

## All connected FB column combinations

Expand the spatial pilot to every reported PFNd → hDeltaB connection in the saved
population matrix, including weight-one pairs:

```bash
python -m fly_connectome spatial-all results/pfnd-hdeltab-v1.0 \
  --cache results/pfnd-hdeltab-spatial-pilot-v1.0 \
  --output results/pfnd-hdeltab-spatial-columns-v1.0
```

The current saved matrix has **432 positive neuron pairs, 20,165 contacts and
68 connected column combinations out of 96**. All their contact coordinates are
retrieved; no representative selection limits extraction. This remains restricted
to PFNd and hDeltaB, rather than the complete connectome. A completed pilot can
optionally supply cached FB sites, skeletons and meshes after artifact hashes and
dataset metadata match. Omit `--cache` to fetch everything fresh.

The command validates all positive pairs and exact weights against the saved
matrix, checks FB/layer site counts and FB contact endpoints, and preserves the
reference's 137/135/133 scope checks. New or missing live connections stop the run.
The complete saved populations remain in the queries and coverage tables,
including neurons with no saved positive connection. These neurons contribute to
possible-pair counts, and newly reported connections involving them are checked.
Neurons with no FB synapse sites remain in the population tables and skeleton
outputs but have no FB synapse centroid.
Existing outputs are refused. Input and output hashes and metadata are recorded
in `spatial_population.json`; only `status: complete` means the whole run passed.

Open **`index.html`** in the output directory for an 8 × 12 visual index:

- `columns/C#_C#.{png,pdf}` pools **all contacts** from all connected neuron pairs
  in each positive annotated column combination, with both populations' synapse
  sites for anatomical context.
- `pairs/pair_#_#.{png,pdf}` provides a detailed skeleton/3D view of the strongest
  whole-connection pair per positive column combination. Ties use ascending body
  IDs. These 68 examples are explicitly representatives, not all 432 body pairs.
- `atlas/PFNd_C#.{png,pdf}` shows all twelve target column positions for each
  source column in common-scale XZ panels. Contact points are coloured by their
  receiving hDeltaB compartment; faint points show hDeltaB FB input sites.
- `column_locations.{png,pdf}` shows per-neuron centroids for PFNd output sites,
  hDeltaB dendritic input sites, and hDeltaB axonal output sites. These landmarks
  use sites across all partners, independent of PFNd → hDeltaB contact selection.
  `column_landmarks.csv` stores the plotted centroids; PFNd pools all FB output
  sites into one centroid per neuron, including outputs on dendritic compartments.
- `column_coverage.csv` retains all 96 cells, with neuron/pair counts, contact
  scopes, FB-post compartment counts and representative IDs. Zero cells mean
  no connection was reported in the saved matrix, not biological absence.
- `contact_arbor_summary.csv` reports contact counts and receiving positions by
  neuron pair, annotated column pair, compartment and total/FB-post scope.
- The pilot's point tables, neuron/arbor summaries, ROI counts, individual
  skeletons and visualization meshes are also retained for the full population.

Multiple target column labels do not by themselves establish that annotations are
unrelated to contact locations: spatially displaced arbors from differently labelled
neurons can overlap. These figures help compare contacts with the locations of
each labelled arbor. They do not infer an indexing rule based on a shaft, assign
normalized columns or angles, or independently validate compartment labels.
Synapse centroids summarize sites rather than shafts or anatomical boundaries.

Re-render saved population data without a token or network:

```bash
python -m fly_connectome spatial-all-plot results/pfnd-hdeltab-spatial-columns-v1.0 \
  --output results/pfnd-hdeltab-spatial-columns-replot-v1.0
```

This checks recorded data hashes against files inside the supplied run directory
and writes to a new directory, including copies of the CSV tables linked by the
gallery. New data manifests store paths relative to the run directory; legacy
manifests are mapped to that directory without reading their original paths.
Copied or relocated runs therefore do not require the original run to exist,
and modifications to the supplied copy are rejected. Validated data
are recorded before rendering, so this command can also recover figures after a
rendering failure without repeating server queries. It records a separate
`replot.json`; it does not change the original extraction manifest.

## Direct upstream inputs to PFNd

Extract a one-hop input census into a **new** directory, using the saved PFNd
population as the membership/instance baseline:

```bash
python -m fly_connectome inputs results/pfnd-hdeltab-v1.0 \
  --output results/pfnd-inputs-v1.0
```

This uses the same local `.env` authentication as `extract`. It checks the live
exact-type PFNd membership and instances against the baseline, related type
annotations, and the reference `26913 → 515192 = 137`. The existing results are
read only. No PFNv expansion, coordinates, skeletons or simulation are fetched.

The upstream query starts with the selected PFNd targets and traverses incoming
`ConnectsTo` edges from **all `Segment` nodes**, retaining standard `weight >= 1`.
There are no source-type, reconstruction-status, cropping, ROI or hemisphere
filters. Recurrent connections and any self-connections are included. The custom
query is saved verbatim in the manifest. The existing `fetch_connections` helper
requires explicit IDs; the API's `sources=None` default uses `Neuron` criteria
and would omit non-neuron segments.

**Share** is a population's total weight divided by all incoming segment weights
across the complete selected PFNd population. **Coverage** is the number of PFNd
neurons receiving positive input from that population divided by the complete
target count. **Input fraction** uses each target's own incoming segment weight
as denominator. Per-target weight distributions and coverage retain zeros;
fractions with zero denominators are undefined and saved as blank cells. Counts
are not estimates of physiological efficacy, and missing connections do not
establish biological absence.

Annotation categories and flagged statuses remain explicit. `untyped_weight`
and `flagged_weight` overlap other category totals; do not add them to those
totals. Flagged status/statusLabel values are `Glia`, `Orphan` and `Unimportant`.
All missing reconstruction values are reported as unknown, not as evidence of
complete reconstruction. These flags do not exclude edges from the baseline.

**Receiving-neuron anatomy and contact location are distinct.** Each PFNd retains
its PB side, PB glomerulus and FB column annotation. Grouping input to a neuron
with FB annotation C8 does not place that input in FB C8: the contact can be in
NO, PB or another region. Anatomical groups are joint `(PB side, PB glomerulus,
FB column)` groups. Raw totals, per-target means and input fractions accompany
the unequal group sizes. No physical column boundaries, common spatial coordinate
or functional-angle mapping are inferred.

ROI counts use the **postsynaptic endpoint** count in each edge's `roiInfo`.
Primary ROI rows are additive. Parent/child and other nonprimary ROI rows are
retained for inspection but overlap and must not be added into totals. Positive
weight outside primary ROIs becomes an explicit `NotPrimary` residual. Unknown
or unspecified *named primary* regions retain their dataset names. Malformed ROI
data, overlapping primary counts, duplicate pairs, invalid weights or changed
target annotations stop the run. Completion also requires exact reconciliation
of each PFNd neuron's recorded `post` count with its incoming segment weights;
diagnostic residuals remain saved if that check fails. This reconciles database
counts, not biological completeness.

| Output | Contents |
| --- | --- |
| `raw_connections.csv` | Raw query response, including source annotations, labels and edge ROI JSON |
| `PFNd_neurons.csv`, `PFNd_anatomical_order.csv` | Live targets, recorded post counts, reconstruction annotations and parsed anatomy |
| `upstream_neurons.csv` | One row per connected source body, original/cross-dataset annotations, category and status flag |
| `incoming_connections.csv` | Positive body pairs, standard weight and original ROI JSON |
| `connection_roi_counts.csv` | Positive pair/ROI post counts; `additive` distinguishes primary/residual rows from overlapping rows |
| `input_denominators.csv` | Every target's total input, category totals, untyped/flagged totals and post-minus-edge residual |
| `population_inputs.csv` | All populations, total share, coverage and zero-inclusive per-target distributions |
| `population_by_target.csv` | Complete population × PFNd target table, including zeros, denominators, fractions and target anatomy |
| `anatomical_group_inputs.csv` | Joint anatomical groups, target counts, total/per-target weight and fractions |
| `population_roi_inputs.csv`, `anatomical_roi_inputs.csv` | Sparse positive population/ROI and anatomical-group/ROI counts |
| `reconstruction_summary.csv` | Source-body counts and edge weights by category and reconstruction field |
| `type_inventory.csv`, `reference_*.csv` | Live type inventory and reference validation inputs |
| `population_totals`, `population_fractions`, `input_distributions`, `input_regions` (`.png`, `.pdf`) | Leading 20 populations; selection affects plots only |
| `inputs.json` | Status, UTC times, dataset/baseline metadata, exact query, definitions, reference and conservation checks, code/package versions and hashes |

Only `status: complete` denotes a successful extraction and rendering run. Failed
runs preserve available diagnostics and never turn query errors into zeros.
The manifest includes source-file hashes as well as the Git revision/dirty flag;
preserve local code changes alongside runs made from a dirty checkout.

Replot validated saved data without a token or network:

```bash
python -m fly_connectome inputs-plot results/pfnd-inputs-v1.0 \
  --output results/pfnd-inputs-replot-v1.0
```

This verifies all recorded data hashes before and after plotting, resolves data
paths inside the supplied run directory, and refuses existing output directories.
It can also recover plots after a rendering failure when the manifest already
records `data_validated: true`. It writes only plots and `replot.json` to the new
directory; the source data remains unchanged.

The population fraction heatmap hides the footer beginning "Anatomical counts"
by default. Add `--show-fraction-caveat` to `inputs` or `inputs-plot` to display it.
The caveat remains in the census metadata regardless of this plot setting.

## Initial feedforward rate model

Run the first dynamical experiments entirely offline from the validated baseline:

```bash
python -m fly_connectome simulate results/pfnd-hdeltab-v1.0 \
  --output results/pfnd-hdeltab-rate-v1.0
```

The command expects the reviewed **40 PFNd and 19 hDeltaB neurons**. It checks
the saved population annotations, reconstructs the complete forward matrix from
the saved body-pair edges, and verifies `26913 → 515192 = 137`. It never queries
neuPrint or reads a token. Existing output directories are refused.

Each neuron has one nonnegative activity value. The initial assumptions are:

| Setting | Starting value or rule |
| --- | --- |
| Coupling | PFNd → hDeltaB only; `W[pre, post] = gain × raw contact count` |
| Sign | Every included connection is assumed excitatory |
| Gain | `0.001` per contact, shared globally; no row/column normalization |
| Nonlinearity | Rectified linear, `ReLU(x) = max(0, x)` |
| Time constant | Shared `tau = 0.1 s`, illustrative rather than physiologically fitted |
| Baseline and initial activity | Zero for all neurons |
| Noise | None |
| Activity units | Arbitrary normalized rate units (a.u.), not Hz; values are not bounded at 1 |
| Solver | Simultaneous explicit Euler updates, `dt = 0.001 s` |
| Duration | 3 seconds per experiment |
| Omitted dynamics | PFNd recurrence, hDeltaB feedback/recurrence, and every other population |

The equations are:

```text
tau * d(PFNd)/dt   = -PFNd   + ReLU(external_drive)
tau * d(hDeltaB)/dt = -hDeltaB + ReLU(PFNd @ W)
```

The presynaptic activity at the **old** timestep drives the postsynaptic update;
neurons are not updated sequentially within a timestep. Raw integer counts and
floating effective weights are saved separately, with their axes unchanged.
Proportional coupling, its excitatory sign, and the shared time constant are
modeling assumptions. They do not follow uniquely from anatomy.

Six default experiments use external drive amplitude 1 per stimulated neuron:

| Experiment | Drive schedule |
| --- | --- |
| `unstimulated` | No external drive |
| `reference_pulse` | PFNd 26913, 0.2–0.3 s |
| `group_pulse` | All left-PB L2 PFNd neurons, 0.2–0.3 s |
| `sustained` | The same L2 group, 0.2–1.2 s |
| `sequence_forward` | L2 at 0.2–0.3 s, L5 at 0.3–0.4 s, L8 at 0.4–0.5 s |
| `sequence_reverse` | L8, L5, L2 at the same successive intervals |

These groups identify receiving-PB annotations of the stimulated PFNd neurons,
not contact locations or calibrated heading angles. All matching neurons are
included; actual body IDs are recorded in `stimuli.json`. In the reviewed source,
each selected group contains two neurons, so the forward/reverse sequences have
equal injected drive. Input events are active on `[start, stop)` and overlapping
events add. The final drive row describes the terminal instant, not another
integration interval.

The default groups select only left-PB neurons, whereas the earlier column
connectivity heatmaps pool both PB sides within each PFNd FB column. Their target
rankings can therefore differ (for example, the stimulated L5/C4 subset favors
hDeltaB C11 over C12). Sequential responses also overlap in time; a bright target
at a given instant can reflect an earlier stimulus.

Configure parameters with `--tau`, `--dt`, `--gain` and `--duration`. Times are in
seconds; duration and event boundaries must lie on the timestep grid. The solver
requires `dt <= tau/20`. Custom stimuli can replace the default experiments:

```json
{
  "experiments": [
    {
      "name": "custom_pulse",
      "events": [
        {"body_ids": [26913], "start_s": 0.2, "stop_s": 0.3,
         "amplitude": 1.0, "label": "reference PFNd"}
      ]
    }
  ]
}
```

Pass that file with `--stimuli path/to/stimuli.json`. Body IDs must belong to the
saved PFNd population; amplitudes must be nonnegative. An empty event list gives
an unstimulated control. The complete normalized stimulus specification is saved.

Every experiment is compared against an independent continuous-time solution for
this equal-time-constant feedforward model. A step produces the first-stage
response `1 - exp(-t/tau)` and second-stage response
`1 - (1 + t/tau) exp(-t/tau)`; finite pulses subtract shifted step responses.
Maximum absolute error is divided by the peak activity of the corresponding
population block (using a `1e-12` floor for zero responses). Both analytical and
half-timestep discrepancies must be at most 2%, and half-timestep analytical error
must not increase. Exact deterministic replay is also checked. These validate
the implementation of the stated equations, not a biological computation.

A sensitivity table varies tau by factors of 0.5 and 2 and gain by factors of 0.5
and 2, changing one parameter at a time. All sensitivity runs use half the main
timestep, including their baseline case. They test dependence on illustrative
parameters; they do not fit physiology. In this positive, unsaturated feedforward
model, doubling gain is expected to double hDeltaB activity without changing PFNd
activity or the hDeltaB response shape.

| Output | Contents |
| --- | --- |
| `model_config.json`, `stimuli.json` | Parameters, equations, assumptions, omitted dynamics and exact stimulus events |
| `raw_contact_matrix.csv`, `effective_weight_matrix.csv` | Original counts and scaled weights; source rows, target columns |
| `*_neurons.csv`, `*_anatomical_order.csv` | Population annotations and plotting orders |
| `experiments/*/drive.csv` | Time in seconds and external drive for every PFNd neuron |
| `experiments/*/activity.csv` | Time and activity of all 59 neurons; column names include population and body ID |
| `experiment_summary.csv`, `neuron_summary.csv` | Peak/final activities, timing and per-neuron activity areas |
| `numerical_validation.csv` | Analytical error, timestep comparison and replay checks for each population/experiment |
| `parameter_sensitivity.csv` | Response summaries under one-at-a-time tau/gain changes |
| `*.png`, `*.pdf` | Stimuli, mean traces, anatomical activity heatmaps and per-neuron peak comparisons |
| `rate_model.json` | Completion status, source metadata/hashes, code/package versions, validation and artifact hashes |

Peak times are undefined for zero responses. Activity area is a numerical summary
of a trace, **not** an implemented displacement integrator or stored memory.
Delayed peaks and decaying activity after stimulation are expected from the two
successive response filters. No movement inputs, angle decoder, path integration,
steering or plasticity are implemented in this milestone.

Only `status: complete` denotes a successful simulation, validation and plotting
run. Numerical failures preserve traces and validation diagnostics. Source files
are hashed before/after computation and saved model data is hashed before/after
plotting. A dirty Git revision is accompanied by package source hashes.

Replot saved validated traces without re-running the experiments:

```bash
python -m fly_connectome simulate-plot results/pfnd-hdeltab-rate-v1.0 \
  --output results/pfnd-hdeltab-rate-replot-v1.0
```

The command verifies relative data paths/hashes inside the supplied run directory,
supports moved/copied runs, and writes to a new directory. If rendering failed
after data validation, `data_validated: true` permits recovering the plots.

## Subcritical PFNd recurrence experiments

Repeat the **exact saved feedforward stimulus protocol** with PFNd recurrence:

```bash
python -m fly_connectome simulate-recurrent results/pfnd-hdeltab-rate-v1.0 \
  --census results/pfnd-inputs-v1.0 \
  --output results/pfnd-hdeltab-recurrent-v1.0
```

This is offline and requires complete, hash-validated feedforward and census
runs. Target body IDs, types and instances, dataset UUID and recorded edit/mutation
metadata must agree. The recurrent matrix comes from all census edges whose
source and target belong to the saved PFNd population, including every positive
weight and all receiving regions. In the reviewed data this is **847 pairs and
8,109 contacts**, with zero reported self-contacts. No diagonal connections are
invented and no inputs are excluded based on region or reconstruction status.

The model adds `PFNd @ R` to PFNd input, where
`R[pre, post] = recurrent_gain × recurrent_contact_count`. Both forward and
recurrent coupling are assumed excitatory. Recurrent gain is independent of the
forward gain, and no per-neuron normalization is imposed. The shared time
constant, timestep, duration, external stimuli, baseline, initial activity,
feedforward weights and no-noise setting are taken from the feedforward run.
Each neuron still has one activity value pooling its regional inputs. There is
no added saturation, explicit transmission delay, hDeltaB feedback or recurrence.

The default gain sweep is `0, 0.0005, 0.001, 0.002`; override it with
`--recurrent-gains`, including zero as a required control. The zero-gain traces
must reproduce the saved feedforward traces to CSV round-trip precision
(`rtol=1e-12`, `atol=1e-14`). The solver is separately tested for bitwise equality
between zero recurrent coupling and its original feedforward code path.

All gains are checked **before** the output directory is created. For these
nonnegative couplings, the recurrent feedback spectral radius must be below 1;
the Euler update's spectral radius must also be below 1. The raw matrix's spectral
radius is approximately `219.725`, giving a critical gain of approximately
`0.00455115` per contact. Gains at or above the stability boundary are refused.
The slowest PFNd network mode has predicted time constant
`tau / (1 - recurrent_gain * raw_spectral_radius)`, distinct from the unchanged
intrinsic neuron time constant. These checks constrain this provisional model,
not biological coupling strength.

Every condition is checked against a continuous-time matrix-exponential reference
for the piecewise-constant stimulus. Because inputs, activity and weights are
nonnegative, ReLU is the identity on the trajectories, so this linear reference
applies. It uses a scaling/squaring Taylor calculation implemented with NumPy,
independent of Euler stepping. The zero-gain reference is also checked against
the original closed-form feedforward solution. Analytical and half-timestep
errors must each be at most 2% of the population block's peak, half-timestep
analytical error must not increase, and deterministic replay must match exactly.

Comparison metrics have operational definitions saved in the manifest:

- **Recruitment:** PFNd neurons never externally stimulated during the trial
  whose peak activity exceeds `0.001 a.u.`. This is not a physiological firing
  threshold; individual peak values remain saved for alternative thresholds.
- **Decay:** time after the last stimulus offset until the largest activity in
  a population remains at or below 1% of that population's whole-trial peak.
  Unobserved crossings and zero-response cases are blank, not proof of memory.
- **Amplification:** population maximum single-neuron activity divided by its
  corresponding feedforward maximum; undefined when the reference is zero.
- **Pattern change:** cosine similarity between vectors of per-neuron temporal
  hDeltaB maxima and the feedforward maxima. These peaks need not be simultaneous
  and do not encode a calibrated heading or travel angle.

The new directory contains `raw_recurrent_matrix.csv`, `stability.csv`,
`comparison_summary.csv`, `neuron_comparison.csv`, `numerical_validation.csv`,
`stimuli.json` and `recurrent_model.json`. Each `gain_*` directory contains its
own configuration, raw/effective forward and recurrent matrices, per-experiment
drive/activity CSVs, validation tables, plots and `rate_model.json`. Root-level
`*_comparison.{png,pdf}` plots overlay the gain conditions for each experiment;
`recurrence_summary.{png,pdf}` compares amplification, recruitment and decay.
Only `status: complete` denotes a successful whole sweep. All source artifacts
and manifests are checked again at the end; existing runs are preserved.

Replot the cross-gain comparisons from saved traces:

```bash
python -m fly_connectome simulate-recurrent-plot results/pfnd-hdeltab-recurrent-v1.0 \
  --output results/pfnd-hdeltab-recurrent-replot-v1.0
```

Individual gain conditions support `simulate-plot` as well. Both replot paths
verify data hashes and support relocated runs. Increased transient persistence
in this stable network should not be interpreted as a displacement integrator
or home-vector memory.

For figures containing only the two time-dependent activity heatmaps (PFNd and
hDeltaB), add `--heatmaps-only` to either replot command:

```bash
python -m fly_connectome simulate-plot results/pfnd-hdeltab-rate-v1.0 \
  --heatmaps-only --output results/pfnd-hdeltab-rate-heatmaps-v1.0
python -m fly_connectome simulate-recurrent-plot results/pfnd-hdeltab-recurrent-v1.0 \
  --heatmaps-only --output results/pfnd-hdeltab-recurrent-heatmaps-v1.0
```

This omits the stimulus and population-mean subplots and the separate peak-profile
summary. It retains neuron labels, time axes and the original per-panel activity
scales. The recurrent command produces one set per `gain_*` condition. These
commands read saved traces, create new PNG/PDF copies, and do not rerun or modify
the simulations. The selected format is recorded in `replot.json`.

## PFNv expansion and combined feedforward experiments

Add the exact `PFNv` population to the original extraction:

```bash
python -m fly_connectome expand-pfn results/pfnd-hdeltab-v1.0 \
  --output results/pfn-hdeltab-v1.0
python -m fly_connectome columns-pfnv results/pfn-hdeltab-v1.0 \
  --output results/pfnv-hdeltab-columns-v1.0
python -m fly_connectome simulate-pfn results/pfn-hdeltab-v1.0 \
  --output results/pfn-hdeltab-rate-v1.0
python -m fly_connectome simulate-plot results/pfn-hdeltab-rate-v1.0 \
  --heatmaps-only --output results/pfn-hdeltab-rate-heatmaps-v1.0
```

Only `expand-pfn` uses the server and local credential. It checks dataset UUID
and recorded edit/mutation metadata, exact type inventory, the live PFNd and
hDeltaB annotations, and every PFNd → hDeltaB weight against the original
extraction. It saves a new extraction and preserves the baseline. Every positive
PFNv → hDeltaB connection is included, with no reconstruction, hemisphere, ROI,
or compartment filters. Full axes retain zero-weight pairs. In the reviewed
MaleCNS data:

| Source population | Neurons | hDeltaB targets | Positive pairs | Contacts |
| --- | --- | --- | --- | --- |
| PFNd | 40 | 19 | 432 | 20,165 |
| PFNv | 20 | 19 | 126 | 4,344 |

Each source population reaches all 19 target neurons, and every source neuron
has at least one reported connection. The combined matrix has 60 source rows
and 19 target columns, ordered by numeric body ID. Separate source annotations
and matrices, anatomical heatmaps, `PFNv_annotation_mapping.csv`,
`connectivity_summary.csv`, and data/artifact hashes are saved alongside it.
The offline loader reconstructs both component matrices from edge tables and
checks their combined matrix, annotations, and hashes before simulation.

`columns-pfnv` produces the same total-contact and mean-per-possible-pair heatmaps,
CSV summaries, and partner rankings as the PFNd column analysis, using PFNv
C1–C9 against hDeltaB C1–C12. Unconnected pairs remain in the mean denominator.
The annotation caveat is hidden on plots by default; `--show-column-caveat`
displays it. Each plot uses its own color scale.

PFNv instance annotations have the form `PFNv(PB05)_L2_C1`. The PB side,
glomerulus, and FB column are parsed literally and ordered by FB column, side,
then body ID. **The PFNd PB-to-FB formula is not applied to PFNv.** Examples
include L9/C9 and R2/C9, and different PB glomeruli can share an FB column label.
`PB05` is part of the instance prefix, not the receiving PB glomerulus. These
indices have not been calibrated to heading angles or aligned across types.

The combined model uses the same provisional defaults as the initial
feedforward model: zero baseline and initial activity, no noise, ReLU,
`tau = 0.1 s`, `dt = 0.001 s`, and `gain = 0.001/contact`. Its equations are:

```text
tau * d(PFNd)/dt   = -PFNd   + ReLU(I_d)
tau * d(PFNv)/dt   = -PFNv   + ReLU(I_v)
tau * d(hDeltaB)/dt = -hDeltaB + ReLU(PFNd @ W_d + PFNv @ W_v)
```

Both source populations use the same excitatory gain per anatomical contact,
without population normalization. Counts do not determine their relative
physiological efficacy. All hDeltaB input compartments are pooled into one rate
per neuron. Recurrence, cross-population PFN connections, feedback, and explicit
upstream populations are disabled. The PFNd recurrence command currently accepts
only PFNd-only feedforward runs, so it cannot silently drop PFNv from this model.

The default protocol contains 12 experiments:

- The original six PFNd experiments, with the same names and events; PFNv is
  unstimulated and stays silent.
- PFNv left-PB L2 and right-PB R2 pulses, left-PB L2 sustained stimulation, and
  left-PB L2/L5/L8 sequences in both orders. Pulse and sequence timing matches
  the PFNd protocol. These groups are anatomical labels, not movement angles.
- A simultaneous PFNd L2 + PFNv L2 pulse, which checks summation at hDeltaB.

All events apply amplitude 1 per selected neuron. The inspected PFNd L2/L5/L8
groups each contain two neurons; PFNv L2/L5/L8 and R2 each contain one. Thus these
trials are not matched for total injected drive, and the combined trial has
three stimulated neurons. Group membership and injected drive are saved.
`--stimuli`, `--tau`, `--dt`, `--gain`, and `--duration` support explicit overrides.

PFNd, PFNv, and hDeltaB have separate activity columns, summaries, numerical
checks, and heatmaps. Full plots show the external drive, three population means,
and three heatmaps; `--heatmaps-only` keeps only those three heatmaps. As before,
the solver is checked against an independent continuous-time solution, half-dt
integration, and deterministic replay. These experiments validate propagation
through the expanded feedforward network; movement-dependent input rules and
functional angle assignments remain the next modeling step.

## Movement-driven PFN encoding experiment

```bash
python -m fly_connectome simulate-movement results/pfn-hdeltab-v1.0 \
  --output results/pfn-hdeltab-movement-v1.0
```

This offline experiment prescribes PFN activity from heading and body-relative
velocity, then integrates hDeltaB rate dynamics. It tests the transformation
through the anatomical weights conditional on the supplied encoding. It does
not model upstream signal generation or assign preferred hDeltaB angles.
The input construction follows the walking model in
[Lu et al., 2022](https://doi.org/10.1038/s41586-021-04191-x),
with the assumptions and measured results written to `experiment_notes.md`.

Positive heading angles are clockwise; body velocity is positive forward and
rightward. Head and body are aligned. The PB heading map uses L5/R5 = 0°,
L7/R3 = +90°, and 45° steps, wrapping at ±180°. It is transferred to MaleCNS
using each neuron's receiving PB annotation; FB column numbers do not assign
angles. Preferred body movement directions are PFNd L/R = −31°/+31° and
PFNv L/R = +137°/−137°. These literature-based assignments are fixed before
evaluating any hDeltaB output and saved per body in `PFN_encoding_map.csv`.

```text
H_i = (1 + cos(heading - preferred_heading_i)) / 2
projection_i = forward_velocity*cos(phi_i) + rightward_velocity*sin(phi_i)
PFN_i = H_i * (b + 5*ReLU(projection_i / reference_speed))
tau * d(hDeltaB)/dt = -hDeltaB + ReLU(PFN @ W)
```

Reference speed is 1 mm/s. PFN activity is in arbitrary units, with identical
per-neuron scaling across types. It is prescribed directly, with no PFN rate
filter. The hDeltaB time constant is the previous illustrative 0.1 s, timestep
0.001 s, and coupling `W = 0.001 × raw_contacts`. There is no normalization,
recurrence, feedback, noise, angle decoder, or integration of displacement.
This is an adaptation using our raw weights, not an exact reproduction of the
published code's connectivity preprocessing.

Two PFN baseline offsets (`b=1` and `b=0`) each run 74 independent trials:

- 8 world travel directions × 8 headings at 1 mm/s, giving 64 primary trials.
- Zero translation at all 8 headings. Travel direction is undefined at rest.
- Two additional speed checks, 0.5 and 2 mm/s, at heading/world direction 0°.

Each trial lasts 2 s, with fixed heading and translation on `[0.2, 1.2)` s.
hDeltaB starts at zero. The analysis window is `[1.0, 1.2)` s, after at least
eight hDeltaB time constants of movement. A positive PFN baseline is active from
t=0 and remains after translation stops: stationary controls need not be silent.

Outputs preserve individual neuron identities. Both raw profiles and
movement-evoked profiles are saved; the latter subtract the stationary trial
with the same heading, baseline and time window. Amplitudes are reported
separately. Primary pattern similarity is the cosine after subtracting each
profile's across-neuron mean, so spatially uniform activation does not count as
directional structure. Zero/constant centered profiles (L2 norm ≤1e−12) are
undefined. Uncentered cosines are also saved. Patterns are never shifted to
align, and no output angular mapping is fitted.

The primary score is mean similarity for same-world/different-heading pairs
minus mean similarity for different-world/same-heading pairs, at matched speed.
There are 224 unordered pairs of each kind. All measurements describe this
single connectome and model, not independent flies.

Two null controls each use 199 realizations and seed 20261020:

- **Shuffled connectivity:** permute whole source rows within each PFN type ×
  PB side, preserving group-wise target totals and row profiles as a multiset.
  One permutation is held fixed across the full condition grid.
- **Shuffled movement labels:** independently permute world-direction labels
  within each heading. Simulated inputs and responses remain unchanged, while
  their associations for the comparisons are randomized. This is not temporal
  shuffling of a trajectory.

The same permutations are used for both baseline offsets. Null scores use the
linear steady-window response, checked against the integrated movement-evoked
profiles. Matched-rest subtraction removes the baseline contribution in this
linear model, so identical evoked scores across baseline offsets are expected.
The score is not a test of absolute directional accuracy or homing.

The output includes `movement_config.json`, `conditions.csv`, encoding maps,
raw/effective weights, source provenance, `comparison_summary.csv`,
`null_comparisons.csv`, `null_permutations.npz`, `activity_summary.csv`, and
`numerical_validation.csv`. Each baseline directory contains raw/evoked steady
profiles, pair comparisons, a similarity matrix, compressed individual trial
traces, and PNG/PDF plots. Trial NPZ files contain time, prescribed PFN activity,
hDeltaB activity, heading, forward velocity and rightward velocity, with axes
listed in the configuration. The manifest hashes all data and generated files.
Numerical checks use an independent continuous-time solution, half dt, and
deterministic replay; biological expectations do not gate successful execution.

`--tau`, `--dt`, `--gain`, `--baseline-levels`, `--null-repeats`, and `--seed`
support explicit overrides. Replot a saved or relocated run without a network
connection or the original extraction:

```bash
python -m fly_connectome simulate-movement-plot results/pfn-hdeltab-movement-v1.0 \
  --output results/pfn-hdeltab-movement-replot-v1.0
```

For normalized world-invariance (world travel −90°, 0°, +90°), body-direction
sweep and heading sweep plots showing only matched-rest-subtracted activity:

```bash
python -m fly_connectome simulate-movement-plot results/pfn-hdeltab-movement-v1.0 \
  --normalized-evoked-only --output results/pfn-hdeltab-movement-normalized-v1.0
```

Each condition (row) is divided by its own maximum across the 19 hDeltaB
neurons, giving a common 0–1 color scale for comparing pattern shape. This
removes amplitude differences between conditions. Rows with a peak at or below
1e-12 remain zero. PNG/PDF plots, normalized CSVs and the original row peaks
are saved for each PFN baseline.

### Relative PFNd/PFNv weight experiments

Optional source-type weight multipliers default to 1, preserving the original
model. To isolate a sixfold PFNd multiplier while keeping raw-contact weights,
PFNv gain, prescribed inputs, timing and analysis settings unchanged:

```bash
python -m fly_connectome simulate-movement results/pfn-hdeltab-v1.0 \
  --pfnd-weight-multiplier 6 --pfnv-weight-multiplier 1 \
  --output results/pfn-hdeltab-movement-pfnd6-v1.0
```

Weights are `base_gain * contacts * source_type_multiplier`. The multipliers
do not change prescribed PFN activity. They are saved in the configuration;
the original output directory remains intact. Use the same replot and decoder
commands on this new run for corresponding figures. The Lu released code also
normalizes incoming weights separately within each PFN type/side group before
applying its factor of 6. This experiment isolates relative scaling and does
not reproduce that normalization. `--pfnv-weight-multiplier` also permits
explicit PFNv scaling experiments.

To additionally use the incoming normalization from the Lu released code:

```bash
python -m fly_connectome simulate-movement results/pfn-hdeltab-v1.0 \
  --normalize-incoming-by-group --pfnd-weight-multiplier 6 \
  --output results/pfn-hdeltab-movement-groupnorm-pfnd6-v1.0
```

For every receiving hDeltaB neuron and each of the four PFN type/PB-side
groups, divide each contact count by the group's total incoming contacts to
that target, then multiply by the type multiplier and common gain. Zero-total
groups remain zero. Thus each nonempty PFNd-side group has total effective
weight `6 * gain`, and each PFNv-side group has total `1 * gain`. This matches
the released weight preprocessing up to our retained common gain (0.001);
absolute amplitudes therefore have a different scale. Other model settings,
MaleCNS connectivity and the 19-neuron heatmap display remain our own. The
normalization option defaults off and does not modify prior saved results.

### Continuous trajectory replay (preparatory milestone)

Replay explicit movement inputs through the selected prescribed-PFN model, offline:

```bash
conda activate fly-connectome
python -m fly_connectome replay-trajectory \
  results/pfn-hdeltab-movement-groupnorm-pfnd6-v1.0 \
  --output results/pfn-hdeltab-trajectory-v1.0
```

The output directory must be new. Existing movement commands, results and
presentation exports remain intact. The command verifies the saved reference
run's data hashes and recomputes its effective weights to check agreement.
It requires incoming normalization separately per receiving hDeltaB neuron
within PFNd-L/R and PFNv-L/R, PFNd multiplier 6, PFNv multiplier 1, gain 0.001,
tau 0.1 s, and uses PFN baseline b=1. Default dt is 0.001 s; `--dt` allows
convergence checks. Heading and translation preferences reuse the saved
literature-based encoding map; no decoder is fitted or applied.

The deterministic default sequence runs for 10 s: rest, straight forward
movement, a clockwise right-angle heading turn, rightward sideways movement,
backward movement reversing the earlier world-relative direction, rest,
four forward legs forming a unit square, and final rest. The first four
moving segments return to the origin at 4.5 s; the square closes at 9 s.
hDeltaB starts at zero once, at t=0, and carries state across every boundary.

Supply `--protocol path.csv` for a custom explicit time series with columns
`time_s,heading_deg,forward_mm_s,right_mm_s` and optional `label`.
There must be at least two finite rows, starting at t=0 with strictly
increasing timestamps on the dt grid. Values are held on each half-open
interval `[time_s[k], time_s[k+1])`; there is no interpolation across angles
or velocities. At a boundary the input changes while activity remains the
terminal state of the preceding interval. The last row is a terminal input
sample and contributes no subsequent time interval. No extrapolation occurs.
Instantaneous heading changes are prescribed turns, not simulated mechanics.

Heading is degrees clockwise from arbitrary world +x. Body forward and
rightward velocities are mm/s, with head and body aligned. World +y points
rightward/downward in the trajectory plot. The external geometric conversion is
`vx = vf*cos(theta) - vr*sin(theta)`,
`vy = vf*sin(theta) + vr*cos(theta)` (trigonometry uses radians).
Integrating each held world velocity over its interval gives exact geometric
displacement in mm, up to floating-point roundoff. Return-to-origin is saved
separately as minus displacement from the starting origin; its direction is
undefined at the origin. These external references never drive the network
and are not neural memory.

Exports include `protocol.csv`, sampled inputs and geometry in `trajectory.csv`,
`segment_boundaries.csv`, and all prescribed PFN rates, hDeltaB drive, raw Euler
activity and exact held-drive activity in `activity.npz`. Neuron IDs and
anatomical orders are saved. `trajectory_inputs.png/pdf` shows the path and
inputs; `raw_activity.png/pdf` shows PFNd, PFNv and hDeltaB over time, with one
unmodified amplitude scale per population over the entire sequence.
`trajectory_config.json`, `trajectory_model.json`, `numerical_validation.json`
and `experiment_notes.md` record conventions, parameters, hashes and limitations.

No independently peak-normalized rows enter temporal amplitude analysis.
No stationary subtraction is used: changing heading changes baseline PFN
activity, so the fixed-heading stationary reference from independent trials
does not apply. A future movement-evoked comparison would need continuous
zero-translation replay with the same heading history, timing and initial
conditions, explicitly described as a counterfactual input decomposition.

The run checks Euler versus exact held-drive dynamics, half-dt sensitivity,
deterministic replay, the existing fixed-condition solver, and geometric
segment endpoints. Tests additionally verify coordinate signs, known path
closure, stationary equilibrium and continuation with nonzero initial states:

```bash
python -m pytest -q tests/test_trajectory.py tests/test_movement.py
```

This establishes a numerically checked continuous feedforward replay pipeline.
Numerical correctness is separate from biological validity. Slow hDeltaB
relaxation is a leaky response to recent input, not displacement integration.
Before neural home-vector modeling, the model needs an explicit neural
accumulation mechanism, independently calibrated coordinate/readout mapping,
retention and reset rules, and validation across duration, speed, turns,
closure and perturbations. Upstream PFN generation, biological time constants,
and behavioral homing remain unvalidated; no spatial learning is implemented.

### Published hDeltaB angular decoder

Apply the decoder from the [Lu et al. model notebook](https://github.com/druckmann-lab/Translational-velocity-and-heading-model/blob/main/Translational%20Velocity%20and%20Heading%20Simulation%20Runs.ipynb)
to a saved movement run:

```bash
python -m fly_connectome decode-movement results/pfn-hdeltab-movement-v1.0 \
  --output results/pfn-hdeltab-movement-decoder-v1.0
```

The authors group 19 neurons into 14 named positions, average duplicate names,
and compute an activity-weighted circular average. The anatomical names and
multiplicities in their `NEURON_LABELS.csv` match our MaleCNS inventory exactly;
transferring their functional interpretation remains a modeling assumption.
These positions differ from the 12 FB C labels. The decoder preserves their
fitted angular offset of −10.62784572061546° and also saves the unshifted phase;
it does not fit angles to our results. That offset is not a MaleCNS physiological
calibration. This provides a Fig. 4d model analogue; Fig. 4h instead uses eight
imaging ROIs with per-fly offset correction.

Separate output includes the explicit body-ID mapping, raw and movement-evoked
angles, resultant lengths, primary-grid plots, and provenance with hashes.
Zero/cancelled vectors have undefined angles. Stationary controls have no true
travel direction. The original simulation files are verified and reused.

## Anatomical investigation and next modeling milestone

The two-body [coordinate and visualization walkthrough](docs/vdeltae_coordinate_run_walkthrough.md)
documents the implemented `fly_connectome.vdeltae_spatial_pilot` workflow. It
extracts FB-marked synapse coordinates for the audited vDeltaE/hDeltaG probe,
preserves cached unhealed morphology, checks saved counts and dataset metadata,
and writes descriptive spatial/layer tables. Offline figure exports include
native XYZ projections, site distributions and a separate mesh vertex preview,
with fresh output directories and verifiable manifests. The walkthrough gives
the live authentication block and offline run/export verification commands.
This remains a two-neuron anatomical pilot: it assigns no physical column map,
effective connection signs or new neural dynamics.
For the implemented [FB context overlay](docs/vdeltae_fb_mesh_context_walkthrough.md),
call `export_coordinate_pilot(source, fresh_output, mesh_overlay=True)` to draw
faint projected mesh triangles behind the sites and full unhealed skeletons.
The default remains mesh-free; range diagnostics and transform provenance are
saved with overlays, and the mesh is not used for containment or column mapping.

The current direction combines an [initial common rate framework](docs/common_rate_framework.md)
for controlled numerical comparisons with continued [source-grounded specification
of the vDeltaE–hDeltaG mechanism](docs/vdeltae_hdeltag_next_steps.md).
The [vDeltaE input profiles](docs/vdeltae_input_profiles.md) summarize the
anatomical and functional evidence gathered so far. These census workflows add
no dynamics. A separate [initial common rate framework](docs/common_rate_framework.md)
now supplies numerical pilot comparisons; it does not establish a biological
displacement-memory mechanism.

The [historical circuit-selection report](docs/hdeltab_extension_recommendation.md) and
[primary-source evidence ledger](docs/hdeltab_extension_evidence.md) document the
24 October 2026 anatomical investigation. Its proposed hDeltaB→hDeltaG
input-transfer experiment is superseded as the next milestone; the census and
analysis remain useful. The specific vDeltaE–hDeltaG integration evidence was
assessed from a dissertation abstract, not its full methods or equations.

### Saved data and a fresh checkout

This checkpoint includes the extraction/analysis tools, tests and research
documents together. Thesis-based model specification and new neural dynamics
belong to subsequent work; see the [checkpoint scope](docs/vdeltae_hdeltag_next_steps.md#repository-checkpoint).

`results/` is ignored by Git. The dated directories named below describe local
runs and are **not included in a checkout**. Research documents contain readable
findings; raw body pairs, matrices, annotations and run manifests remain in those
separate result directories. Existing results and presentation exports are
preserved, not regenerated or added to this checkpoint.

After [setup](#setup), either copy complete saved run directories from an
existing installation or regenerate their prerequisites using the earlier
workflows. Keep each manifest with all files it hashes; copying only a summary
or matrix is insufficient.

| Operation | Required saved inputs |
| --- | --- |
| hDeltaB downstream extraction | Original `pfnd-hdeltab-v1.0` run; see [baseline extraction](#extract-and-validate). Requires live neuPrint authentication and matching dataset metadata. |
| vDeltaE input extraction | Complete downstream census plus the saved [PFNd ×6 movement run](#relative-pfndpfnv-weight-experiments), including its PFN identity map. Requires live neuPrint authentication. |
| Downstream report generation | Complete downstream census, the same movement run, and [continuous trajectory run](#continuous-trajectory-replay-preparatory-milestone). Offline. |
| Census verification | Complete corresponding census directory. Offline; no authentication or movement/trajectory runs needed. |
| Tests | Installed development dependencies. Synthetic data and mocked queries; no saved runs or authentication needed. |

To verify existing censuses without extracting data or creating new reports:

```bash
python -c 'from fly_connectome.extension_analysis import validate_saved_census; validate_saved_census("results/hdeltab-extension-census-final-20261024")'
python -m fly_connectome.vdeltae_inputs \
  results/vdeltae-inputs-live-20261024 --verify-only
```

Extraction and report commands below require **new output directories**; use
different names if the dated outputs already exist. New live runs must pass
dataset/identity checks and need not reproduce a historical snapshot if the
server data have changed. Keep original manifests and code hashes as provenance;
do not rewrite them to match a later code revision.

### hDeltaB downstream census and analysis

Extract all direct Segment partners of the saved 19 hDeltaB neurons, plus a
fixed whitelist of 54 directed candidate-population pairs:

```bash
python -m fly_connectome.circuit_extension results/pfnd-hdeltab-v1.0 \
  --output results/hdeltab-extension-census-final-20261024
```

Generate readable summaries and verify saved counts, matrices, denominators,
hashes and model neuron identities offline:

```bash
python -m fly_connectome.extension_analysis \
  results/hdeltab-extension-census-final-20261024 \
  --movement results/pfn-hdeltab-movement-groupnorm-pfnd6-v1.0 \
  --trajectory results/pfn-hdeltab-trajectory-v1.0 \
  --output results/hdeltab-extension-analysis-20261024
```

These dated directories already exist; use new output names for subsequent
runs. Authentication follows the existing local `.env` convention. Raw body
pairs, untyped partners and annotations are retained, without status, cropping,
ROI or hemisphere filters. Output fractions divide by all outgoing Segment
contact weights, never unique presynaptic site counts. Candidate input fractions
use recorded complete-population postsynaptic counts. Exact queries and data
hashes are saved; only a complete manifest denotes success. Existing movement,
trajectory and presentation outputs are preserved.

### vDeltaE upstream survey

The [vDeltaE input survey](docs/vdeltae_input_survey.md) follows the revised
investigation of the vDeltaE–hDeltaG motif. It retains all one-hop Segment inputs
and checks overlap by body ID with the existing PFNd/PFNv/hDeltaB model:

```bash
python -m fly_connectome.vdeltae_inputs \
  results/hdeltab-extension-census-final-20261024 \
  --movement results/pfn-hdeltab-movement-groupnorm-pfnd6-v1.0 \
  --output results/vdeltae-inputs-live-20261024
python -m fly_connectome.vdeltae_inputs results/vdeltae-inputs-live-20261024 --verify-only
```

Use a new output name when repeating extraction. Raw annotations, untyped sources,
per-target denominators, full modeled-source matrices, hashes and readable
summaries are saved. Incoming weights must reconcile with every target's recorded
post count. This identifies anatomical inputs; no neural dynamics are added.

## Initial common rate framework

The [framework guide](docs/common_rate_framework.md) describes a small recurrent
rate engine with signed, ID-aligned coupling, explicit sign evidence and optional
slow excitability states. It preserves the existing PFN encoding and compares a
separate signed Janke fluorescence reference with a hypothetical rate candidate.
Both use arbitrary rate units; no Hz calibration or movement-to-vDeltaE rule is
inferred. Existing commands, saved results and presentation exports are unchanged.

```bash
python -m fly_connectome.framework_experiments pfn-reference \
  --movement results/pfn-hdeltab-movement-groupnorm-pfnd6-v1.0 \
  --trajectory results/pfn-hdeltab-trajectory-v1.0 \
  --output results/common-framework-pfn-pilot-20261026
python -m fly_connectome.framework_experiments janke-comparison \
  --output results/common-framework-janke-pilot-20261026
```

The PFN case requires the complete saved runs. The controlled Janke comparison
requires no saved data or credentials and uses idealized columns, not a mapped
MaleCNS population. Use new output names if these directories already exist.
The guide includes configuration, offline verification, outputs, a custom
recurrent example and the remaining biological assumptions.

## Tests

```bash
python -m pytest -q
```

Tests use synthetic data and mocked neuPrint responses. They require no token.
Passing these tests does **not** verify the live reference pair: that requires
an authenticated extraction run and its saved validation records.

## Related projects

- [flywire-connectomics](https://github.com/rcalfredson/flywire-connectomics):
  General anatomical connectomics analyses using the FlyWire dataset.
- [drosophila-spatial-learning-modeling](https://github.com/rcalfredson/drosophila-spatial-learning-modeling):
  Virtual spatial-learning environments, movement simulations, and configurable
  reward contingencies. Neural controllers developed here may eventually be
  integrated with that simulation framework.

## Dataset and API references

- [MaleCNS project](https://male-cns.janelia.org/), including publication and attribution information.
- [MaleCNS download page](https://male-cns.janelia.org/download/), documenting the `male-cns:v1.0` database and CC-BY licensing.
- [neuprint-python query API](https://connectome-neuprint.github.io/neuprint-python/docs/queries.html).
- [neuprint-python authentication](https://connectome-neuprint.github.io/neuprint-python/docs/quickstart.html).
