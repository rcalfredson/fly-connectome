# Hand-written implementation: interpreting vDeltaE/hDeltaG column annotations

Prepared 29 October 2026. Continue after the
[annotation-audit command walkthrough](vdeltae_column_audit_command_walkthrough.md).
This is suggested code and a staged investigation plan. The production module
and tests remain for you to write by hand. No live extraction, morphology
download, spatial assignment or simulation is performed by this document.

## Scope and stopping point

Investigate only the bodies selected by the verified annotation audit: currently
18 vDeltaE and eight hDeltaG. The existing body inventory stays unchanged.
Spatial evidence will go in separate tables keyed by body ID, retaining its
source, method, confidence and unresolved questions. Do not overwrite the literal
`annotation_column` or change `anatomical_mapping_status` in the old audit.

The questions are:

- What convention produced each population's C labels? Are there citable
  annotation instructions or a primary anatomical description?
- Where are each body's presynaptic and postsynaptic sites in the FB?
- Are independently documented arbor/compartment labels available, and what
  evidence establishes their meaning?
- How do hDeltaG's spatially separated input/output regions relate to its C
  token, and how do vDeltaE labels relate to those regions?
- What accounts for vDeltaE C0/C9 and repeated labels? They remain ordinary
  observed labels until evidence supports an interpretation.

The deliverable is a body-level evidence table plus spatial plots, not an
automatic eight-column mapping. An unresolved result is acceptable. Anatomical
alignment does not itself establish physiological signs or a home vector.

## Existing code and data to consult

- `fly_connectome/vdeltae_column_audit.py`: verified body selection and source
  provenance. Reuse its verifier rather than selecting a new population.
- `fly_connectome/spatial.py`: synapse points, contact endpoints, skeletons and
  mesh context for the earlier PFNd/hDeltaB pilot.
- `fly_connectome/spatial_population.py`: broader spatial checks and plots for
  those earlier populations.
- `tests/test_spatial.py` and `tests/test_spatial_population.py`: examples of
  validating counts, point uniqueness, units, confidence and saved sources.

Those spatial workflows have PFNd/hDeltaB selection, reference pairs, expected
counts and coordinate assumptions. Do not run them against this new core as if
they were generic. Review individual helpers before reusing them. Likewise,
the earlier PFN annotation parser does not establish a vDeltaE/hDeltaG map.

The saved source metadata currently records 8 nm isotropic voxels and FB/FBl1–9
ROI names. These are observations of these saved manifests, not defaults to
silently apply to future datasets. ROI names and voxel sizes do not supply
anatomical axis orientation, a column coordinate or compartment semantics.

## Step 1: inspect verified sources and their saved spatial metadata

Suggested new module: `fly_connectome/vdeltae_spatial_audit.py`. Start with this
read-only piece. It does not call a live client or generate final output files.

```python
import json
from pathlib import Path

import pandas as pd

from .connectivity import ValidationError
from .inputs import sha256
from .vdeltae_column_audit import verify_audit


def spatial_metadata_inventory(metadata):
    properties = metadata.get("neuronProperties", {})
    if isinstance(properties, str):
        try:
            properties = json.loads(properties)
        except json.JSONDecodeError as error:
            raise ValidationError("Unrecognized neuronProperties metadata encoding.") from error
    if not isinstance(properties, dict):
        raise ValidationError("Expected a neuronProperties metadata dictionary.")
    roi_info = metadata.get("roiInfo", {})
    if not isinstance(roi_info, dict):
        raise ValidationError("Expected an ROI metadata dictionary.")

    return {
        "dataset_identity": {
            field: metadata.get(field)
            for field in ("uuid", "latestMutationId", "lastDatabaseEdit")
        },
        "coordinate_metadata": {
            field: metadata.get(field) for field in ("voxelSize", "voxelUnits")
        },
        "recorded_post_high_accuracy_threshold": metadata.get("postHighAccuracyThreshold"),
        "fb_roi_names_in_metadata": sorted(
            name for name in roi_info if name == "FB" or name.startswith("FBl")
        ),
        "declared_neuron_properties": {
            name: properties[name]
            for name in ("instance", "type", "somaLocation", "somaSide", "roiInfo")
            if name in properties
        },
        "column_annotation_semantics": "unresolved",
        "anatomical_axis_orientation": "unverified",
        "synapse_schema_and_compartments": "not inspected by this metadata inventory",
        "skeleton_and_mesh_availability_for_selected_bodies": "not checked",
    }


def inspect_spatial_sources(audit_directory):
    audit_directory = Path(audit_directory).resolve()
    manifest_path = audit_directory / "manifest.json"
    before = sha256(manifest_path)
    manifest = verify_audit(audit_directory)
    bodies = pd.read_csv(
        audit_directory / "body_annotations.csv",
        dtype={"bodyId": "Int64", "annotation_column": "Int64", "instance_index": "Int64"},
    )
    inventories = {}
    for role, source in manifest["sources"].items():
        source_manifest = json.loads(Path(source["manifest_file"]).read_text())
        inventories[role] = {
            "source_manifest_file": source["manifest_file"],
            "source_manifest_sha256": source["manifest_sha256"],
            "spatial_metadata": spatial_metadata_inventory(
                source_manifest.get("dataset_metadata", {})
            ),
        }

    verify_audit(audit_directory)
    if sha256(manifest_path) != before:
        raise ValidationError("Annotation audit manifest changed during inspection.")
    return bodies, {
        "annotation_audit_directory": str(audit_directory),
        "annotation_audit_manifest_sha256": before,
        "sources": inventories,
    }
```

What to understand before proceeding:

- `neuronProperties` is a JSON-encoded string in the reviewed saved metadata.
  The helper accepts either that encoding or an already-decoded dictionary.
  Unexpected formats are explicit errors, not silently accepted empty schemas.
- A missing optional metadata value remains `None` and prints as JSON `null`.
  Absence from the declared property inventory is not proof that a body lacks
  a soma or that a particular live endpoint is unavailable.
- Neuron property declarations are **not the Synapse node schema**. In
  particular, they do not establish whether synapse `compartment` or `split`
  properties exist, are populated, or have verified biological meanings.
- `roiInfo` here is dataset-level metadata. Listing its FB-related keys does
  not locate any selected neuron's sites, prove nonzero counts, or certify a
  particular spatial-query ROI filter. Layer names are not column labels.
- The recorded confidence threshold is inventoried, not yet adopted as a new
  extraction filter. A later query must explicitly declare its threshold,
  comparison operator and treatment of missing confidence values.
- `verify_audit` checks output hashes and reruns the existing source verifiers.
  Repeating it after reading, plus checking the audit manifest itself, detects
  ordinary input changes over this inspection. This does not lock the files.
- The returned body table is a local copy. Its stored source annotations and
  the saved audit remain unchanged. No new membership is inferred from geometry.

### Inspect from Bash

After transcribing this first piece, run from the repository root with the
project environment active. Use the verified audit you already generated:

```bash
python - <<'PY'
import json

from fly_connectome.vdeltae_spatial_audit import inspect_spatial_sources

bodies, inventory = inspect_spatial_sources(
    "results/vdeltae-hdeltag-annotation-audit-20261029"
)
print(bodies[["bodyId", "type", "instance", "annotation_column",
              "anatomical_mapping_status"]].to_string(index=False))
print("\nSaved spatial metadata inventory:")
print(json.dumps(inventory, indent=2))
PY
```

Expected checkpoint for this audit:

- All 26 selected bodies remain present; anatomical mapping is `unreviewed`.
- The two source identity records match. Both record `voxelSize=[8, 8, 8]`,
  `voxelUnits="nanometers"`, and a post high-accuracy threshold of `0.5`.
- FB and FBl1–9 names are listed in the saved dataset ROI metadata.
- Column semantics, axis orientation, synapse compartment meanings and
  selected-body morphology availability remain unresolved/unverified.

Stop to inspect this output. Do not convert C labels into angles, download all
26 skeletons, or start a simulation as part of this first code step.

### Small metadata tests

Suggested new test file: `tests/test_vdeltae_spatial_audit.py`. Start with tests
of the pure metadata helper; source integration is exercised separately by the
read-only command above. Later tests should cover verification failure
propagation and source changes around the inspection as well.

```python
import json

import pytest

from fly_connectome.connectivity import ValidationError
from fly_connectome.vdeltae_spatial_audit import spatial_metadata_inventory


@pytest.mark.parametrize("encoding", ["dictionary", "json_string"])
def test_metadata_inventory_does_not_infer_column_geometry(encoding):
    properties = {"instance": "string", "somaLocation": "point"}
    if encoding == "json_string":
        properties = json.dumps(properties)
    result = spatial_metadata_inventory({
        "neuronProperties": properties,
        "voxelSize": [8, 8, 8], "voxelUnits": "nanometers",
        "roiInfo": {"FB": {}, "FBl6": {}, "PB": {}},
    })
    assert result["fb_roi_names_in_metadata"] == ["FB", "FBl6"]
    assert result["coordinate_metadata"]["voxelSize"] == [8, 8, 8]
    assert result["declared_neuron_properties"]["somaLocation"] == "point"
    assert result["column_annotation_semantics"] == "unresolved"
    assert result["anatomical_axis_orientation"] == "unverified"
    assert result["recorded_post_high_accuracy_threshold"] is None


def test_missing_coordinate_metadata_is_not_filled_with_defaults():
    result = spatial_metadata_inventory({})
    assert result["coordinate_metadata"] == {"voxelSize": None, "voxelUnits": None}
    assert result["fb_roi_names_in_metadata"] == []


@pytest.mark.parametrize("properties", ["invalid JSON", [], "[]"])
def test_unrecognized_property_schema_is_rejected(properties):
    with pytest.raises(ValidationError):
        spatial_metadata_inventory({"neuronProperties": properties})
```

These are six cases. Run them after transcribing the module and tests:

```bash
python -m pytest -q tests/test_vdeltae_spatial_audit.py
```

## Next pieces: agree and write them one at a time

### 2. Document annotation conventions and inspect query capabilities

Build an evidence table with source title/URL or local artifact, exact cell
type/dataset, relevant passage or field, interpretation, and uncertainty.
Distinguish primary anatomical descriptions, dataset annotation instructions,
and our hypotheses. Do not transfer a convention across datasets without
checking it. If documentation cannot explain a label, leave that question open.

Inspect the available Synapse properties and targeted body annotations before
designing the point query. Review the earlier query helpers and confidence
conventions. Test which morphology/ROI resources are actually accessible for
selected bodies; a missing skeleton download is unavailable evidence, not proof
of anatomical absence. Record native coordinate units and frame explicitly.

### 3. Design a small spatial extraction pilot

Start with one vDeltaE and one hDeltaG selected transparently from the body
table, solely to validate the extraction format. Keep their C labels visible;
do not call them aligned because their tokens match. Record the choice before
viewing the pair's spatial relation.

Preserve raw coordinates, body IDs, pre/post polarity, confidence, ROI flags,
and any available compartment/split labels, including unknown values. Keep
unique synapse sites distinct from pre-to-post contacts: an output site can
participate in more than one contact. Require explicit denominators and
thresholds for any count comparison. Do not sum overlapping layer/FB counts.

Save queries, dataset metadata before/after, source hashes, raw results, code
identity and complete/failed manifests in a fresh results directory. Check
selection membership, coordinate finiteness, uniqueness under the chosen site
identifier, units, and count reconciliation under matching filters. No maximum-
hop expansion is needed: the selected bodies remain the boundary.

### 4. Expand only after the pilot checks pass

Inventory sites for all 26 bodies and inspect spatial distributions separately
by body, polarity, layer and independently documented compartment. Use skeletons
and FB geometry for context where available. A single centroid can mix distinct
arbors; inspect distributions and morphology before reducing them to a position.

Presynaptic means output site; postsynaptic means input site. Those polarities
do not by themselves classify whole arbors as axons or dendrites. Preserve any
dataset compartment annotation separately from polarity and from our inference.

### 5. Propose and cross-check alignment

For each body, record the original C token, proposed physical input/output
regions, supporting evidence, alternative interpretations and confidence. Treat
vDeltaE C0/C9, repeated labels, layer differences and hDeltaG's two spatial
regions explicitly. Never merge bodies to force eight slots.

Only then compare the proposed spatial relationships with the saved bounded
vDeltaE-to-hDeltaG connectivity. If partner-specific contacts are needed,
restrict requests to these selected populations. Connectivity may support or
challenge the anatomical interpretation; it must not determine a permutation
chosen to make the expected same/opposite-column motif appear.

Stop before assigning effective signs, modifying rate dynamics or applying
movement tuning. A defensible coordinate relation and an explicit list of
unresolved assignments are the success criteria for this anatomical stage.
