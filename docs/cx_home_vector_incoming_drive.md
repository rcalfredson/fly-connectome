# Incoming-drive accounting for history-dependent activity

This audit follows the five [selected-trajectory cases](cx_home_vector_trajectory_tuning.md).
It asks which signed presynaptic signals accompany their activity differences,
while separating incoming drive from the target neuron's own current rate.
It changes no weights, rates or decoder parameters.

## What is measured

For a dynamic target neuron j, at each original trajectory sample:

```text
PFN drive        = sum over PFNs i (weight[i,j] × PFN_rate[i])
recurrent drive  = sum over dynamic neurons k (weight[k,j] × rate[k])
preactivation    = bias[j] + PFN drive + recurrent drive
rectified target = max(0, preactivation)

next rate change = alpha[j] × rectified target − alpha[j] × current rate[j]
alpha[j]         = 1 − exp(−dt / tau[j])
```

Drive is the signed weighted sum before rectification, not a measured biological
current or a firing rate. Individual partner contributions and population sums
retain their signs. Large positive and negative contributions can cancel.
Learned weight signs are model properties; this analysis does not establish
physiological excitation or inhibition.

The `−alpha × current rate` term explicitly accounts for the neuron's own
carried-over activity. Self-connections, if present, remain in the recurrent
drive and are flagged in the partner table. A neuron can have a high current
rate even while its instantaneous drive is declining.

Every used path is replayed against its archived checkpoint before analysis.
Summing signed partner contributions must reconstruct recurrent drive. The
reconstructed next state must match the production float32 rollout, within
floating-point tolerances; maximum errors are recorded. Accounting itself uses
float64 with the checkpoint's unchanged float32 effective weights and parameters.

## Original trajectory versus matched probes

The original-trajectory figures and snapshot table use the actual PFN inputs
at each saved moment. In cases 1 and 3 those inputs match across the paired
states. In the other cases they can differ; PFN-drive differences are shown
explicitly rather than attributing all differences to recurrence.

The separate `identical_probe_first_updates.csv` uses the same heading and body
velocity in A and B for every probe. Thus:

```text
difference in PFN drive = 0
difference in preactivation = difference in recurrent drive
difference in next rate change
    = alpha × difference in rectified target − alpha × difference in current rate
```

This directly separates presynaptic recurrent input from own-state persistence
at the branching point. It does not make either a causal necessity claim.
The static weights determine how a history-dependent difference in presynaptic
rate contributes to the target: `delta_drive = weight × (rate_B − rate_A)`.

The audit also reconstructs preactivation throughout every matched probe.
`probe_rectification.csv` records the fraction of nonpositive preactivations
over conditions and update times, separately for moving and stationary probes,
up to each sampling time. Different gate occupancy can help explain why response
shapes differ while the strongest sampled heading/body combination remains
unchanged. Fractions describe this deterministic model and grid, not repeated
trials, animals or physiological firing probabilities.

## Run

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fly-connectome-mpl \
python -m fly_connectome.home_vector.analysis.incoming_drive \
  --activity results/cx-home-vector-balanced-u1975-activity-by-trajectory-20261106 \
  --probes results/cx-home-vector-balanced-u1975-trajectory-history-checks-20261108 \
  --source results/cx-home-vector-hdeltac-long-data-20261104 \
  --output results/cx-home-vector-balanced-u1975-incoming-drive-20261108 \
  --threads 1
```

Outputs must be fresh. Input hashes, checkpoint identity, neuron ordering,
starting states, replay and next-state reconstruction are checked. The audit
accepts renamed input directories and preserves all inputs. Cases and focus
neurons come from the completed selected-probe bundle.

## Outputs

- `index.html` and `report.md`: browsable/portable report, definitions, snapshot
  table, and PNG/PDF figure links.
- `path_XX_neuron_ID_incoming_drive.png/pdf`: full natural-trajectory rate and
  rectified target, PFN/recurrent/total drive, population contributions, and six
  partners selected by largest mean absolute signed contribution. Full arrays
  retain every recurrent partner.
- `case_X_pair_Y_neuron_ID_presynaptic_difference.png/pdf`: original-trajectory
  snapshots, population drive differences, and largest individual partner
  differences. These are not plots of standardized probe inputs.
- `paired_snapshot_drives.csv`: actual rate, incoming drive, preactivation,
  rectified target and own-state/target update terms in A, B and B−A.
- `presynaptic_partner_differences.csv`: every nonzero effective recurrent
  partner, weight, presynaptic rate, signed contribution and contribution change.
  A recurrence-disabled checkpoint legitimately produces a header-only table.
- `identical_probe_first_updates.csv`: explicit first-update accounting under
  each identical moving/stationary input.
- `probe_rectification.csv`: cumulative gate occupancy and preactivation range
  for each selected state, neuron, probe group and sampling time.
- `path_XX_drives.npz`: natural-trace arrays, with target indices, population
  contributions and all-neuron recurrent partner contributions.
- `*_probe_drives.npz`: reconstructed drive/update arrays for actual probe
  update intervals. The final saved state has no subsequent recorded update
  and is excluded. The natural-trace arrays include their final state's
  instantaneous drive, which is likewise not applied to another saved state.
- `incoming_drive.json` and `manifest.json`: definitions, identity, numerical
  verification, hashes and input paths. Checkpoint, metadata, identities and
  protocols are copied into the output for provenance.

Natural arrays use time × target; population arrays insert population before
target, and partner arrays insert presynaptic neuron before target. Probe arrays
add a leading condition dimension. `target_indices` and `neurons.csv` identify
neurons. Every dynamic neuron remains a possible presynaptic partner even if
only selected targets are plotted.

## Limits

This is local mechanistic accounting, not an intervention. Large signed drive
does not imply that removing a partner would produce the same change, because
the network would subsequently evolve differently. Changed tuning shape can
arise from history-dependent recurrent state and rectification without any
weight change. An unchanged peak does not imply that a shape difference is only
an offset: the earlier offset/gain residuals test that separately. Identifying
necessary partners, or whether home-vector training created these dynamics,
requires further controlled comparisons.

## First completed audit: 8 November 2026

The default command completed for update 1975. All source paths replayed
exactly, signed partner sums reconstructed recurrent drive, and the largest
next-state reconstruction error was 7.64e-6 model units (float32 rollout versus
double-precision accounting).

At the 6 s branching point for paths 6 (A) and 7 (B), the PFN inputs match:

| Target | Actual rate A / B | PFN drive A / B | Recurrent drive A / B |
| --- | --- | --- | --- |
| vDeltaE 142911 | 49.934 / 13.061 | 42.756 / 42.756 | 7.515 / 4.893 |
| hDeltaB 17816 | 26.668 / 19.602 | 1.999 / 1.999 | 26.847 / 18.601 |
| hDeltaC 20231 | 8.926 / 1.797 | 2.079 / 2.079 | 2.512 / 0.881 |

For 17816, the largest recurrent partner contribution change is from hDeltaG
20565: −8.119 drive units (B−A), compared with the net recurrent-drive difference
of −8.247. Other partners partly reinforce or cancel that change. For 142911,
the largest changes are from hDeltaB 17816 (−1.717), hDeltaG 20565 (−1.063), and
hDeltaG 19798 (+0.301). These are exact contributions under fixed weights, not
predictions of silencing those neurons.

The table also shows why current rate cannot be equated with current drive:
142911's rate differs by 36.873 units while its instantaneous recurrent drive
differs by only 2.622 units. Its own carried-over state is part of the rate
update and must be accounted for separately.

In case 5, 59176's target is rectified to zero on 1.75% of moving-probe update
positions following the path-12 state, versus 66.84% following path 13, over the
first 250 ms. The corresponding values for 20951 are 0% and 5.41%. This directly
documents different gate occupancy accompanying history-dependent sensitivity.
These are fractions across the movement grid and time updates; a zero target
does not instantaneously force the actual rate to zero. The observations do not
establish which presynaptic partners are causally necessary.
