# Hand-written investigation: annotation conventions and spatial-query design

Prepared 29 October 2026. Continue after Step 1 of the
[spatial-audit walkthrough](vdeltae_spatial_audit_walkthrough.md). That step is
implemented; this document contains suggested additions, not production code.
Transcribe and inspect each piece before continuing. No live query or resource
download has been run for this document.

## This stage's boundary

Separate two questions: what the labels are documented to mean, and what
spatial data the selected bodies actually have. This first pass prepares an
evidence ledger and testable, bounded query builders. The suggested Bash block
only verifies saved inputs, selects two probe bodies, and prints query text.

The subsequent live runner must save raw answers and provenance before we
interpret them. Do not start a spatial extraction merely by pasting queries
into an unrecorded session. No new neural dynamics, body remapping, full-core
morphology download or graph expansion belongs here.

## A. Initial primary-source evidence ledger

These are starting observations, not a completed MaleCNS convention audit.
Update the ledger as you read the linked sources, recording the specific
section, figure, dataset and any conflicting evidence.

| Question | Primary source / location | What it supports | Boundary / status |
| --- | --- | --- | --- |
| How were hΔ columns assigned in the hemibrain analysis? | [Hulse et al., Figure 37 text and caption](https://elifesciences.org/articles/66039#fig37) | The study indexes hΔ neurons by their dendritic arbor location. | Peer-reviewed hemibrain convention; transfer to MaleCNS instance tokens remains unverified. |
| Does the literature require a single eight-column grid for vΔ neurons? | [Hulse et al., vΔ morphology descriptions and Figure 31 material](https://elifesciences.org/articles/66039) | The study describes a nine-column organization and neurons spanning the edge columns. | Context for checking boundary labels; not evidence that MaleCNS C0 or C9 means a particular edge or wrap. |
| Have MaleCNS annotations changed across releases? | [Official MaleCNS release notes](https://male-cns.janelia.org/release/) | The v1.0 notes list annotation refinement. | Release provenance, not a definition of the C-token convention. |
| Where can native anatomy be inspected? | [Official MaleCNS exploration guide](https://male-cns.janelia.org/explore/) | Documents native segmentation, neuropil outlines and pre/post visualization layers. | Resource discovery; availability and coordinates for our exact bodies still need checking. |
| What distinguishes sites from connections? | [neuPrint import/data specification](https://github.com/connectome-neuprint/neuPrint/blob/master/jsonspecs.md) | Synapse records have polarity/location; one presynaptic site may appear in multiple connection records. | Schema guidance, not proof of live node-property completeness or compartment meanings. |
| How should skeleton/mesh resources be used? | [Official neuprint-python client reference](https://connectome-neuprint.github.io/neuprint-python/docs/client.html) | Targeted skeleton retrieval is supported; ROI meshes are intended for visualization. | Do not use mesh geometry as a validated quantitative FB boundary or assume every body has a retrievable skeleton. |

The key MaleCNS-specific questions are still open: whether these C tokens index
dendrites, axons or another morphological landmark; how numbers run across the
native FB; and how C0/C9 and repeated tokens were assigned. Seek dataset
annotation documentation or primary anatomical descriptions and record when
they do not answer these questions. A lack of a convention in the pages above
does not prove that no convention exists elsewhere.

Keep hDeltaG's two arbors, synapse polarity, any `compartment` property, and
our inferred anatomical interpretation in separate fields. A `post` site can
occur on an arbor annotated as axonal. Neither a schema key nor a numerical
`split` value supplies its own biological definition.

## B. Select two transparent probe bodies

Extend `fly_connectome/vdeltae_spatial_audit.py` with a pure selection function.
It operates on the table returned by `inspect_spatial_sources()`; it does not
choose an aligned or strongly connected pair. The rule is simply the smallest
body ID in each of the two audited populations.

```python
def select_spatial_probe(bodies):
    required = {"bodyId", "type", "instance", "annotation_column"}
    if not required.issubset(bodies.columns):
        raise ValidationError("Probe selection needs the audited body annotation fields.")
    if bodies["bodyId"].isna().any() or bodies["bodyId"].duplicated().any():
        raise ValidationError("Probe selection requires present, unique body IDs.")
    if not bodies["type"].isin(["vDeltaE", "hDeltaG"]).all():
        raise ValidationError("Probe selection supports only the audited vDeltaE/hDeltaG core.")

    rows = []
    for population in ["vDeltaE", "hDeltaG"]:
        selected = bodies.loc[bodies["type"].eq(population)].sort_values("bodyId")
        if selected.empty:
            raise ValidationError("Probe selection needs both audited populations.")
        rows.append(selected.iloc[[0]])
    return pd.concat(rows, ignore_index=True).copy()
```

For the current saved audit this selects vDeltaE **54907 (C6)** and hDeltaG
**19798 (C8)**. Their numerical tokens differ. Neither that difference nor this
selection establishes their relative spatial position. The pilot validates data
access and query format; it cannot resolve C0/C9 or the entire population map.

## C. Build a small read-only query bundle

Add `integers` to the existing `.inputs` import. Reusing it validates that query
IDs are positive integers before placing them in Cypher text. All graph
traversal stays inside the two selected neurons and their contained synapses.
No presynaptic/downstream partner populations are traversed.

```python
from .inputs import integers, sha256


def spatial_probe_queries(body_ids):
    ids = integers(pd.Series(list(body_ids), dtype=object), "Probe body IDs", 1)
    if len(ids) != 2 or ids.duplicated().any():
        raise ValidationError("Spatial capability probe requires exactly two distinct body IDs.")
    ids = sorted(int(value) for value in ids)
    selection = f"MATCH (n:Neuron) WHERE n.bodyId IN {ids}"
    contained_sites = """
        MATCH (n)-[:Contains]->(:SynapseSet)-[:Contains]->(s:Synapse)
        WITH DISTINCT n, s
    """
    return {
        "body_annotations": selection + """
            RETURN n.bodyId AS bodyId, n.type AS type, n.instance AS instance,
                   n.roiInfo AS roiInfo, n.status AS status,
                   n.statusLabel AS statusLabel, n.cropped AS cropped
            ORDER BY bodyId
        """,
        "site_availability": selection + """
            OPTIONAL MATCH (n)-[:Contains]->(:SynapseSet)-[:Contains]->(s:Synapse)
            WITH DISTINCT n, s
            RETURN n.bodyId AS bodyId,
                   count(s) AS all_contained_sites,
                   sum(CASE WHEN s.FB = true THEN 1 ELSE 0 END) AS sites_marked_FB,
                   sum(CASE WHEN s.location IS NOT NULL THEN 1 ELSE 0 END) AS sites_with_location,
                   sum(CASE WHEN s.confidence IS NOT NULL THEN 1 ELSE 0 END) AS sites_with_confidence
            ORDER BY bodyId
        """,
        "site_property_inventory": selection + contained_sites + """
            UNWIND keys(s) AS property
            RETURN n.bodyId AS bodyId, s.type AS synapse_type,
                   property, count(*) AS sites_with_property
            ORDER BY bodyId, synapse_type, property
        """,
        "fb_site_categories": selection + contained_sites + """
            WHERE s.FB = true
            RETURN n.bodyId AS bodyId, s.type AS synapse_type,
                   s.compartment AS compartment, s.split AS split,
                   count(*) AS site_count
            ORDER BY bodyId, synapse_type, compartment, split
        """,
    }
```

Understand the bundle before considering a live run:

- Query IDs are checked and converted to ordinary integers; text is not
  interpolated from arbitrary annotation names. The caller must obtain those
  IDs from the verified probe table, not choose unrelated bodies manually.
- `WITH DISTINCT n, s` prevents repeated membership paths from duplicating
  a site. This counts graph nodes, not contacts, partner counts or firing rates.
- The availability query uses `OPTIONAL MATCH` so a selected existing neuron
  with no contained sites can still produce a zero-count row. A missing neuron
  produces no row and must be rejected by the later runner's membership check.
- No confidence filter is applied: the purpose is to inspect which information
  exists before choosing a filter. The confidence count includes zero scores;
  it counts nonmissing values, not high-confidence sites.
- `keys(s)` inventories observed Synapse properties on these bodies, including
  ROI flags. It does not establish the schema for every neuron in MaleCNS.
  `count(*)` there counts sites carrying a property within each body/polarity.
- `s.FB = true` selects explicitly FB-marked nodes for category inspection.
  Absent FB properties are not proof that an anatomical site lies outside FB.
- Missing `compartment` or `split` values remain null groups. Do not replace
  them with zeros or infer dendrite/axon from polarity. Inspect property names
  and values first; if the data use another convention, revise the query
  explicitly and preserve the old query/version in the run record.
- These summaries are not a coordinate extraction. They deliberately do not
  retrieve all XYZ points or compute centroids yet.

### Preview from Bash without connecting to neuPrint

After transcribing the two functions:

```bash
python - <<'PY'
from fly_connectome.vdeltae_spatial_audit import (
    inspect_spatial_sources, select_spatial_probe, spatial_probe_queries,
)

bodies, inventory = inspect_spatial_sources(
    "results/vdeltae-hdeltag-annotation-audit-20261029"
)
probe = select_spatial_probe(bodies)
print(probe[["bodyId", "type", "instance", "annotation_column"]].to_string(index=False))
print("Selection rule: smallest audited body ID per population; no connectivity/geometry selection.")
for name, query in spatial_probe_queries(probe["bodyId"]).items():
    print("\nQuery:", name)
    print(query)
PY
```

This reads and verifies saved sources and prints four read-only query strings.
It neither authenticates nor executes the strings. Stop here to review the
selection and query clauses before adding the live runner.

## D. Offline tests to append

Add these imports to `tests/test_vdeltae_spatial_audit.py`:

```python
import pandas as pd

from fly_connectome.vdeltae_spatial_audit import (
    select_spatial_probe, spatial_probe_queries,
)
```

Then append:

```python
def test_probe_selection_is_independent_of_row_order_and_retains_annotations():
    bodies = pd.DataFrame({
        "bodyId": [20, 3, 10, 1],
        "type": ["vDeltaE", "hDeltaG", "vDeltaE", "hDeltaG"],
        "instance": ["vDeltaE_C0", "hDeltaG_01_C1", "vDeltaE_C9", "hDeltaG_14_C8"],
        "annotation_column": [0, 1, 9, 8],
    })
    original = bodies.copy(deep=True)
    selected = select_spatial_probe(bodies)
    assert selected["bodyId"].tolist() == [10, 1]
    assert selected["annotation_column"].tolist() == [9, 8]
    pd.testing.assert_frame_equal(bodies, original)
    pd.testing.assert_frame_equal(selected, select_spatial_probe(bodies.iloc[::-1]))


def test_probe_selection_requires_both_populations():
    bodies = pd.DataFrame({
        "bodyId": [1], "type": ["vDeltaE"], "instance": ["vDeltaE_C0"],
        "annotation_column": [0],
    })
    with pytest.raises(ValidationError, match="both audited populations"):
        select_spatial_probe(bodies)


@pytest.mark.parametrize("ids", [[1], [1, 1], [1, 2, 3], [0, 2], [None, 2], [1.5, 2]])
def test_probe_queries_reject_invalid_or_unbounded_selection(ids):
    with pytest.raises(ValidationError):
        spatial_probe_queries(ids)


def test_probe_queries_preserve_site_denominators_and_unknown_categories():
    queries = spatial_probe_queries([20, 10])
    assert set(queries) == {
        "body_annotations", "site_availability", "site_property_inventory", "fb_site_categories",
    }
    assert all("n.bodyId IN [10, 20]" in query for query in queries.values())
    assert all("ConnectsTo" not in query and "SynapsesTo" not in query for query in queries.values())
    assert "OPTIONAL MATCH" in queries["site_availability"]
    assert "WITH DISTINCT n, s" in queries["site_property_inventory"]
    assert "keys(s)" in queries["site_property_inventory"]
    assert "s.confidence >" not in "\n".join(queries.values())
    assert "s.compartment AS compartment" in queries["fb_site_categories"]
    assert "coalesce" not in queries["fb_site_categories"]
```

These add nine cases to the existing six, for **15 spatial-audit cases**. They
check the selection rule and important query boundaries without network access.
String checks do not prove Cypher syntax or server semantics; the documented
live pilot is the integration check still to come.

```bash
python -m pytest -q tests/test_vdeltae_spatial_audit.py
```

## E. Requirements for the subsequent live capability runner

Write this as the next small piece after reviewing the query builders:

1. Accept a verified audit directory and a fresh, separate output directory.
   Reuse the project's authentication setup without printing credentials.
2. Record the audit hash, selected bodies, selection rule, code/package identity,
   exact query strings and unfiltered-site denominators in a running manifest.
3. Fetch live metadata and require the recorded dataset identity to match the
   saved audit. Compare coordinate metadata separately. Stop on drift rather
   than silently mix live anatomy with an older census snapshot.
4. Execute each bounded query and immediately save its raw table. Validate exact
   body membership and type/instance equality, nonnegative integer counts, and
   subtype/property coverage consistent with the availability counts. An empty
   category table is unavailable evidence, not a missing-body row to invent.
5. Request only the two selected skeletons with `heal=False` and the FB ROI
   mesh, saving raw data and a per-resource result. Classify a documented
   not-found response separately from an empty success, authentication failure,
   timeout or unexpected error. Do not convert every exception into “no
   skeleton.” The mesh is display context, not a quantitative boundary.
6. Recheck metadata and source hashes, then write complete/failed status and
   artifact hashes. Later offline validation must be able to inspect these raw
   records without querying the server again.

Resource access is part of this capability stage; full spatial distributions
belong to the following extraction pilot. Before that pilot, decide and record
confidence filtering, coordinate units, site identifiers, polarity handling,
ROI rules, compartment definitions and count-reconciliation denominators.

The checkpoint is an evidence ledger and a reproducible capability result. It
does not require that every optional resource exists, but must distinguish
confirmed absence from untested or failed access and leave column mapping open.
