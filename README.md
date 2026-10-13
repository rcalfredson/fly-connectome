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
