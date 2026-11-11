# Paired-state decoder row-space and null-space analysis

This is the first analysis-reboot step: compare the saved forward/lateral curved
path pairs (8/9, 10/11, 12/13 and 14/15) at matching times and accumulated world
displacements. It uses all 65 dynamic neurons and the archived best checkpoint
at update 1975, with SHA256
`2aab91f13258883819f66f12897919b2bf509617e7927120ca7f9db069d9a31f`.
No model parameters or encoder settings change. No new probes are introduced.

## What is split

The original decoder is `home = rate @ D.T + d`. Subtracting histories cancels
the intercept, so `home_B - home_A = (rate_B - rate_A) @ D.T`.

At **each matched time**, form `delta = rate_B - rate_A`. Do not subtract the
first difference or a within-trajectory baseline. With an orthonormal basis `Q`
for the decoder's row space (obtained by SVD):

```text
row_component  = (delta @ Q.T) @ Q
null_component = delta - row_component
```

The component vectors retain signed differences for every neuron. They are not
separate neuron groups or states to feed into the rate model.

The analysis checks that the components reconstruct the difference, are
orthogonal, and satisfy:

```text
row_component @ D.T  = delta @ D.T
null_component @ D.T = 0
norm(delta)^2 = norm(row_component)^2 + norm(null_component)^2
```

The decoder has rank two and nullity 63. Row/null norms use model units, not mm.
Large null norms indicate cancellation in Euclidean activity space, not a
fraction of information or a demonstration of irrelevant history. Even a small
row component can correspond to an appreciable vector error. Null differences
can affect future output through recurrence or unequal passive decay rates.

## Run

With the home-vector dependencies installed:

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fly-connectome-mpl \
python -m fly_connectome.home_vector.analysis.state_spaces \
  --activity results/cx-home-vector-balanced-u1975-activity-by-trajectory-20261106 \
  --source results/cx-home-vector-hdeltac-long-data-20261104 \
  --output results/cx-home-vector-balanced-u1975-state-spaces-20261110-v2 \
  --threads 1
```

Use a fresh output directory. Input manifests, checkpoint identity and neuron
order are verified. All eight source trajectories replay against the production
model before their states are compared. Inputs are hashed before and after
analysis, and a completion manifest hashes the output artifacts. The archived
checkpoint, metadata and path protocols are copied for provenance.

Outputs include a browser (`index.html`), written report (`report.md`), PNG/PDF
figures, full per-pair NPZ arrays, all-time CSV measurements, selected snapshots,
per-neuron snapshot components, the decoder/basis and validation diagnostics.
Snapshots are at 0, 2, 4, 6, 7.5 and 8 seconds. The terminal state is retained,
but excluded from duration-based RMS summaries because it has no following
update. Its stationary PFN input has not yet influenced the recorded rate.

Actual heading, body velocity, speed and complete PFN equality are recorded.
Displacements must agree within 5e-6 mm per component. Exact PFN equality and
equality within 1e-6 absolute encoder units are reported separately. These
curves have matching world paths but different instantaneous inputs; causal
attribution to history requires later identical-input continuations.

## First results

All eight trajectories replayed with zero maximum absolute error in rates,
PFNs, targets and predictions. The full paired activity differences are mostly
in the decoder's null space, but decoded agreement varies across the pairs:
RMS decoded differences range from 0.182 to 0.701 mm. Forward-walking vector
RMSEs range from 0.124 to 0.169 mm, versus 0.251 to 0.605 mm for lateral walking.

For paths 12/13 at 4 s, total, row and null difference norms are 39.228, 1.504
and 39.199 model units. Although 99.853% of the squared difference norm is in
the null space, decoded disagreement is 0.762 mm; individual vector errors are
0.088 mm and 0.695 mm. This demonstrates substantial cancellation alongside a
residual output disagreement, rather than uniformly accurate motor-history
invariance. No accuracy threshold or recurrent necessity is inferred.

The completed report is in
`results/cx-home-vector-balanced-u1975-state-spaces-20261110-v2/index.html`.
The earlier directory without `-v2` is the initial run before report wording
and browser presentation were expanded; the numerical analysis is unchanged.

For a flat asset pool, copy the PNGs to descriptive upload names rather than
renaming the originals. The generated report links and completion manifest use
the original filenames. Copies named, for example,
`cx-hv_u1975_paths-12-13_state-spaces_2026-11-10.png` can coexist with the
originals without changing the archived artifact hashes.
