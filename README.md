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
