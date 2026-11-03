# Network expansion and learned input encoding

Design discussion, 2026-11-01. This document proposes follow-up experiments;
it does not change the implemented model or report a new training result.

The subsequent [hDeltaC iteration](cx_home_vector_hdeltac_iteration.md) implements
one bounded expansion with the prescribed PFN mapping retained. Learned input
encoding remains a separate proposal.

## What can be learned without prescribing tuning?

Two different freedoms should be distinguished:

- Unknown neural tuning can be learned from a task objective. We need not
  assign each added cell type a heading preference or movement tuning curve.
- The network still needs information about the particular movement that
  occurred. From the same initial state, an autonomous deterministic network
  generates the same activity regardless of the externally specified path.
  Independent noise does not supply the missing movement information.

For example, a small learned, memoryless encoder could replace the existing
prescribed PFNd/PFNv firing curves:

```text
physical measurements s(t) = [cos(heading), sin(heading), forward_speed, rightward_speed]
PFN input rates u(t) = softplus(B s(t) + c)
CX state evolves using u(t) and its connectome-constrained recurrence
predicted home vector = D rate(t) + d
```

With 60 PFN input neurons and four measured variables, this encoder adds
60 × 4 + 60 = 300 trainable scalars. Speed units/scales must be declared.
Its weights can start randomly and be fitted jointly with the CX model and
decoder through the existing trajectory loss. This is a proposed alternative
encoder, not an implemented option.

The encoder receives no position, home vector, accumulated displacement,
trajectory identifier, elapsed-time signal or previous inputs. It has no
recurrent state. Heading and velocity remain idealized physical measurements;
learning their neural representation removes tuning assumptions, but does not
model how the animal senses them. An affine-plus-softplus encoder is a useful
restricted starting point, not a guarantee of sufficient representational
capacity. A richer encoder would require its own capacity controls: it could
perform the body-to-world coordinate transformation upstream of the CX.

Importantly, expanding the dynamic network does not require changing this
input interface. Added neurons can receive activity through anatomical edges
from PFNs and the existing network. Their responses then emerge from the
fitted dynamics. We need not directly inject movement into every new type.

## Bounded candidate pool

The saved full type inventory in
`results/vdeltae-inputs-live-20261024/type_inventory.csv` gives:

| New exact type | Full population size |
| --- | ---: |
| hDeltaC | 20 |
| FB6N | 2 |
| FB5AB | 2 |
| FB6O | 2 |
| FB6E | 2 |
| FB5I | 2 |
| FB5L | 2 |
| PFNp_c | 40 |
| PFNp_b | 136 |
| **New bodies** | **208** |

hDeltaG, hDeltaB, PFNd and PFNv in the suggested table already belong to the
current model. Adding all nine new types would produce 313 selected bodies:
253 dynamic neurons and the existing 60 clamped PFN sources. Only 98 of the
136 PFNp_b neurons occur among direct vDeltaE input partners; selecting those
98 alone would define a different, connectivity-filtered population.

The input ranking measures contacts into vDeltaE. It neither establishes all
return paths nor measures causal importance. Saved projections already show
hDeltaC → vDeltaE (1,604 contacts) and vDeltaE → hDeltaC (64 contacts), making
hDeltaC a tractable first reciprocal expansion. The entire pool requires an
inventory of all directed connections among the selected dynamic types and
all PFNd/PFNv projections into them. Missing saved blocks are unqueried, not
zero. Full-population body lists must include neurons with no selected edges.

Keep PFNd/PFNv externally clamped for the initial comparison. Making them
dynamic, allowing feedback into them, or injecting external currents into
different populations would change the input boundary and should be a
separate architecture comparison. No new live extraction has been performed
for this proposal.

## Experiments that separate the two changes

| Dynamic network | Prescribed PFN encoding | Learned PFN encoding |
| --- | --- | --- |
| Existing 45-neuron core | Repeated reference | Isolate encoding change |
| Expanded core | Isolate anatomy change | Combined experiment |

Use the same selected PFN sources, physical information, trajectory splits,
decoder class, normalization convention and training/evaluation budget across
the comparison. Repeat multiple initialization seeds. Revisit the previously
identified initialization and singleton-projection normalization issues
consistently across all arms rather than attributing those changes to anatomy.

Evaluate the home-vector challenges from the first report, including long
displacements, closure, matched world paths with different body motions, and
opposite histories followed by identical pause inputs. Include a current-input
baseline for each encoding, acute memory/recurrence ablations, and retrained
rewired controls. Additional neurons and parameters can improve capacity
without establishing a special advantage of the anatomical wiring. Query the
bounded anatomy and benchmark the resulting graph before choosing run budgets.

## What would count as physiological agreement?

A successful home-vector decoder does not uniquely specify internal firing
patterns. Keep task-only fits fixed before comparing their activity with
independent measurements. Test population-specific heading/translation tuning,
response dynamics, persistence and perturbation effects under matched sensory
and behavioral conditions. Report seed variation and use anatomical cell or
column correspondence rather than arbitrary post-hoc rearrangements.

The task must expose the relevant variables: a heading/velocity-only training
set cannot establish odor or airflow tuning. In particular, shared family
names do not justify assigning PFNp_c the existing PFNd/PFNv walking tuning.
Also, an addendum to the hDeltaC wind/odor study warns that the driver labeled
hDeltaK as well, so some earlier physiological attributions are uncertain.
[Matheson et al., addendum, 2024](https://www.nature.com/articles/s41467-024-46225-8).

There is precedent for predicting internal physiology by training
connectome-constrained networks on behaviorally relevant tasks without neural
activity targets. The fly visual-system example still supplied external visual
stimuli to photoreceptors; it did not learn without sensory input.
[Lappalainen et al., 2024](https://www.nature.com/articles/s41586-024-07939-3).

Conversely, teacher–student modeling shows that task performance plus
connectivity can leave internal activity underdetermined; activity constraints
can improve recovery. Consequently, physiological agreement should be tested
as a separate prediction, and any recordings used to fit or select models
must be distinguished from held-out validation.
[Beiran and Litwin-Kumar, 2025](https://www.nature.com/articles/s41593-025-02080-4).
