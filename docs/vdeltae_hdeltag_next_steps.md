# Current direction: specify the vDeltaE–hDeltaG mechanism

Updated 28 October 2026. A subsequent [common rate framework pilot](common_rate_framework.md)
implements numerical reference/candidate comparisons. The biological mechanism
specification and validation remain incomplete; this plan records the evidence
needed to advance beyond that pilot.

## Decision and scope

The next milestone is a source-grounded specification of the vDeltaE–hDeltaG
computation described in [Abby Janke's 2025 dissertation, *A Neuronal Circuit
Motif for Leaky Vector Integration*](https://digitalcommons.rockefeller.edu/student_theses_and_dissertations/808/).
The initial review used the abstract. A subsequent [targeted full-text reading
guide](janke_thesis_model_reading_guide.md) identifies the equations, relevant
experiments, fitting methods and unresolved parameter conventions. It is the
starting point for the specification. The separate pilot transcribes the source
recurrences under explicit conventions; it does not validate the mechanism.
The dissertation evidence must remain distinct from peer-reviewed evidence.

This supersedes the [earlier hDeltaB→hDeltaG transfer proposal](hdeltab_extension_recommendation.md)
as the next experiment. Preserve that dated proposal and its anatomical results.
The [input census](vdeltae_input_survey.md) and [ten selected input profiles](vdeltae_input_profiles.md)
remain the anatomical context; they do not supply a complete dynamical model of
vDeltaE. Those ten populations account for 7,749/14,583 incoming contacts
(53.14%), rather than all inputs.

## Information to extract before implementation

- **Observations:** vDeltaE and hDeltaG activity, stimulus and behavioral
  conditions, baseline and bump measurements, timescales, reset conditions,
  and perturbation results. Distinguish bump amplitude from baseline and peak
  activity, and identify what is known about phase.
- **Model:** actual equations, state variables, spatial mapping, initial
  conditions, parameters and units. Identify how the proposed baseline rise,
  vDeltaE-dependent slowing and leak are represented rather than replacing
  them with a generic accumulator.
- **Evidence boundaries:** measured versus assumed signs and weights, the role
  of reciprocal and within-population connections, and whether feedback was
  experimentally tested or required by the model. Anatomy alone does not
  establish that recurrence is necessary or sufficient for integration.
- **Validation:** observations used to constrain parameters versus independent
  observations the model should predict. Record missing information explicitly;
  exploratory substitutions must be labeled as our hypotheses.

The deliverable is an evidence table and a concrete simulation specification,
including required inputs, assumptions, predictions and failure criteria.

## Conditional simulation progression

Once the mechanism is specified, controlled prescribed vDeltaE activity may
allow a reduced test without modeling every upstream population. Existing pulse,
sustained-input, time-stepping and state-handling machinery can inform that work.
The existing hDeltaB response equation is a possible filtering benchmark, not
an established hDeltaG mechanism.

Do not decide to omit feedback before examining the source model. Prescribing
vDeltaE externally prevents hDeltaG from changing its activity through feedback;
if that loop is essential, use an explicitly recurrent model or a justified
reduction. Any feedforward case must state what it omits and which claim it tests.

Candidate tests include pulses followed by rest, changes in amplitude and
duration, and equal total input delivered with different timing. Direction
changes require an independently justified spatial mapping. Specify predictions
from the recovered mechanism before running these tests, and compare histories
with one parameter set. Slow decay, recurrence or persistent activity alone
does not demonstrate integration.

A movement-to-vDeltaE encoding rule is a separate question. Use a measured rule
if available; otherwise compare explicit hypotheses rather than copying
PFNd/PFNv tuning or choosing inputs to force a desired hDeltaG output.

## Relationship to the existing model

Keep the prescribed PFNd/PFNv→hDeltaB model as a separate modeling problem for
now. This is not a claim of biological independence. Sparse direct anatomical
overlap does not establish the functional interface, and PFNp_b/PFNp_c inputs
cannot automatically inherit PFNd/PFNv encoding rules.

Carry forward the observed differences between equivalent world-relative
movements produced with different heading/body-velocity combinations. A downstream
projection is not presumed to repair those differences.

Keep recent movement history, path length, net displacement, a calibrated home
vector, goal direction and steering as separate claims. The initial objective
is to reproduce the reported computation under its experimental conditions;
it does not yet establish a biological home-vector mechanism or controller.

## Repository checkpoint

The scope below records the **previous anatomical-investigation checkpoint**,
committed as `88f8f43`. The current, separate framework checkpoint contains the
shared rate solver, experiment adapters, centered/absolute-input comparison,
tests, example configuration and guides described in [the pilot documentation](common_rate_framework.md).

Keep the completed anatomical investigation in one commit: the bounded hDeltaB
downstream and vDeltaE input extraction tools, offline analysis, validation
tests, literature/evidence documents, and updated research direction. The
validation fixes cover optimization-independent count checks and reporting of
absent candidate populations. Preserve the earlier transfer proposal as
superseded history rather than presenting it as the next experiment.

Generated data under ignored `results/`, existing simulation outputs and
presentation exports remain separate. The [README](../README.md#saved-data-and-a-fresh-checkout)
documents saved-run dependencies, regeneration entry points and offline
verification. A fresh checkout can run the synthetic tests but does not contain
the census data referenced by the reports.

That anatomical commit boundary ended before new vDeltaE/hDeltaG dynamics,
movement encoding or a PFN interface. The subsequent framework pilot implements
controlled reference/candidate dynamics, while biological mechanism validation,
movement-to-vDeltaE encoding and a justified PFN interface remain future work.
