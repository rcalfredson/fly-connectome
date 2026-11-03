# Input boundary: information sufficiency and candidate sensory mappings

Reviewed 1 November 2026 while the hDeltaC training experiment was running.
Only the saved prepared circuit and encoder were inspected; no model, training
configuration, prepared bundle or active-run artifact was changed.

## Distinguish three questions

1. Do the physical variables supplied to the encoder suffice for the synthetic
   target? Yes: ideal heading and forward/right body velocity determine planar
   world velocity and hence displacement from the initial location.
2. Does our particular neural encoding preserve those variables sufficiently?
   Not everywhere. The explicit counterexample below establishes a loss of
   velocity information in the implemented algebraic encoder.
3. Can this recurrent network learn a useful integrator from that encoding?
   That remains an empirical question. Additional input types can supply
   complementary information, redundancy, useful signal dynamics or easier
   access to already available variables. More cell types alone guarantee none
   of these benefits.

Identical entire input histories, from identical initial states, give identical
neural states and decoder outputs in the deterministic model. If the required
home vectors differ, exact recovery for both trajectories is impossible,
regardless of the number of recurrent neurons. This is stronger evidence of an
input limitation than poor performance by one fitted decoder.

## A concrete ambiguity in the current encoder

The existing prescription is

```text
rate_i = 0.5 (1 + cos(heading - preferred_heading_i))
         × (1 + 5 max(0, forward_speed cos(phi_i) + right_speed sin(phi_i)))
```

The actual saved PFNd/PFNv translation preferences contain four directions:
+31, -31, +137 and -137 degrees relative to forward. For body-motion directions
strictly between 121 and 133 degrees, only the +137-degree group has a positive
velocity projection. The other three groups retain their heading-dependent
baseline but lose all negative-projection magnitude information through
`max(0, ...)`. Therefore two velocity components are represented by only one
velocity-dependent scalar in this sector. A symmetric sector occurs between
227 and 239 degrees (equivalently -133 and -121 degrees).

For a fixed heading of zero, the following unrounded values were generated
using direction angles 122 and 132 degrees and speeds
`1 / cos(direction - 137 degrees)`:

| Walk | Forward speed (mm/s) | Rightward speed (mm/s) | Home vector after 10 s (mm) |
| --- | ---: | ---: | --- |
| A | -0.5486127918010737 | 0.8779639937928675 | (5.486127918010737, -8.779639937928675) |
| B | -0.6716865765704306 | 0.7459835179818971 | (6.7168657657043065, -7.459835179818971) |

Both have projection 1 mm/s onto +137 degrees, while the other three projections
are negative. The maximum difference across the 60 encoded PFN rates was
`8.88e-16` in NumPy, and the arrays were exactly equal after float32 conversion,
the precision used for training inputs. Repeating either constant control
produces the same complete input sequence. Both speeds are within the existing
0–2 mm/s training range. Their correct terminal vectors differ by
1.8046052891 mm, so a common prediction must have error at least 0.9023026445 mm
on one of the pair. A common stationary suffix preserves the ambiguity.

This is a property of our simplified encoder, not evidence that biological
PFNd/PFNv neurons necessarily discard these signals. The demonstration does
not explain forward-path magnitude saturation, where these ambiguous sectors
are not involved. It also does not quantify the contribution of this ambiguity
to pooled validation or test error. The current fixed-input hDeltaC comparison
remains useful for assessing the effect of adding that circuitry.

Reproduce the check with the project Python environment and a prepared bundle:

```python
import json
from pathlib import Path
import numpy as np
from fly_connectome.home_vector.data import Circuit, encode

circuit = Circuit(**json.loads(Path(
    "results/cx-home-vector-hdeltac-data-20261101/circuit.json"
).read_text()))
direction = np.array([122., 132.])
speed = 1 / np.cos(np.deg2rad(direction - 137))
velocity = speed[:, None] * np.column_stack([
    np.cos(np.deg2rad(direction)), np.sin(np.deg2rad(direction))
])
rates = encode(circuit, np.column_stack([np.zeros(2), velocity]))
print(np.max(np.abs(rates[0] - rates[1])))
print(np.array_equal(rates[0].astype(np.float32), rates[1].astype(np.float32)))
print(-10 * velocity)
```

## Candidate mapping inventory

| Input/model branch | Relevant physical signal and evidence | What must be resolved before use |
| --- | --- | --- |
| Existing PFNd/PFNv walking branch | Joint heading and body-centric translation responses; direct foundation for the current model. [Lu et al., 2022](https://www.nature.com/articles/s41586-021-04191-x). | Revisit our algebraic rectification, baseline and tuning approximation; inspect inhibitory responses and response dynamics. A simplified implementation is not the measured tuning itself. |
| PFNa | Airflow-direction responses with fitted dynamic encoding models. [May et al., 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13524059/). | These are airflow/flight-condition measurements, not a calibrated walking ground-velocity channel. |
| PFNp_c | Transient airflow-direction and airspeed responses; a published fitted encoding model. [May et al., 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13524059/). | Preserve adaptation and offset responses; establish wind and behavioral context. Do not treat airspeed as ground speed or automatically impose the PFNd heading-bump scheme. |
| Other PFNs, including PFNp_b | Anatomical candidates; some relevant LNO physiology exists. | Exact subtype correspondence and subtype-specific physiological calibration remain necessary. Upstream LNO responses cannot be assigned wholesale to each downstream PFN subtype. |
| Separately represented compass input | Direct heading access could make heading/movement separation easier for the model. | In this task ideal heading already enters PFN phase; this would change routing and representation, without supplying a new independent physical variable. |

The May study provides a particularly relevant example of fitting dynamic PFN
encoding and testing input sufficiency, but its task is wind-direction inference
in flight. Airflow depends on ambient wind and self-motion; converting optic
flow to metric ground speed requires scene-scale assumptions. A walking task
with only ideal kinematic controls does not yet specify these sensory variables.

For each candidate, a useful evidence record should include exact type/driver
and dataset correspondence, behavioral state, sensory modality, units,
direction and speed tuning, baseline and suppression, temporal response model,
recording modality (calcium, voltage or spikes), parameter/data availability,
and demonstrated anatomical access to the modeled circuit. Separate measured
parameters, transferred parameters and explicit modeling assumptions.

## Useful next work independent of the running result

- Audit whether nearby physical movements are distinguishable by the current
  encoder across direction, speed and heading; separate exact ambiguities from
  weak sensitivity and poor conditioning.
- Construct an independent current-PFN-to-world-velocity diagnostic, then
  integrate its velocity estimates in an ideal external accumulator. This
  tests input encoding with memory handled perfectly. It is a diagnostic,
  not a new biological decoder or a fair replacement for the trained CX model.
  Failure by one fitted diagnostic alone would not prove lost information.
- Compare that diagnostic with the ideal physical-variable reference to
  distinguish encoding error from accumulation error. For an auxiliary test of
  recurrent capacity, raw-variable inputs can be supplied to a separate model;
  that changes the input interface and must be labeled as such.
- Compile the physiological mapping inventory above before choosing added
  inputs. Favor well-supported complementary signals and quantify what each
  contributes rather than equating input count with information content.
- After the current experiment, vary input encoding while holding the circuit
  fixed. Any revised rectification or added sensory input is a separate
  experimental factor, so the current run should retain its original setup.
