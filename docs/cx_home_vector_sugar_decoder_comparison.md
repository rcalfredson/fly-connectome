# Sugar-response dynamics: the next controlled comparison

## Notes for the experimental log

- Treat the sugar-response rule as a qualitative working approximation motivated by hDeltaG physiology. This experiment tests its consequences for the model's representation; it does not establish physiological calibration.
- Keep the prescribed PFN inputs, connectivity, recurrent parameters, and sugar-response constants fixed. Fit only a new linear home-vector decoder and its intercept. Preserve the original checkpoint and all earlier results.
- Build a separate derived training/evaluation dataset. Retain the original no-sugar trajectories so the readout must still work away from home, and add trajectories that return to the original anchor, receive sugar, and continue moving afterward. Include matched no-sugar controls, with identical movements and pauses.
- Vary return geometry, orientation, speed, traversal direction, pause duration, and contact timing. Keep all variants of a trajectory family in the same split. Use fresh held-out families for the principal evaluation; retain the familiar circular trajectories as explanatory diagnostics.
- Deliver sugar through an explicit environmental event. For the new examples, a finite patch around the original anchor generates a contact event after departure and reentry. The network receives the event, not position, distance to home, a loop-closure flag, or the correct home vector. Initial patch occupancy does not deliver sugar.
- Keep the target home vector relative to the original starting point throughout. Sugar neither changes that reference nor automatically zeros neural activity or decoded output.
- Fit one fixed readout across all conditions for the sugar-response network. Give the unchanged network an equivalently fitted readout, using the same trajectories, weights, regularization candidates, and split rules. Retain both networks' results under the original frozen decoder as additional references.
- Use only the 65 dynamic neurons' activities as readout features. Determine scaling and weights from training data; select regularization on validation data; evaluate the test set afterward. Do not select sugar parameters using decoder performance.
- Report whole-trajectory errors, errors at the first and final returns, and behavior before and after sugar separately. Improvement after refitting would show that some information was accessible to a different linear readout; it would not prove a physiological reset or a stable integrator over arbitrary durations.

## Commit review before starting this comparison

The pre-existing worktree changes formed one coherent measurement-inventory commit:
`feat(analysis): inventory hDeltaG sugar-response measurements`.
That commit is now present as `70b1ffa`. The user-supplied thesis PDF remains outside the proposed commit.

The new dataset builder, decoder comparison, focused tests, and this methods document form a separate proposed commit:
`feat(analysis): compare frozen sugar dynamics with refitted decoders`.
Generated result bundles remain separate archives. No automatic commit is part of this experiment.

## Derived dataset

Implementation: `fly_connectome/home_vector/analysis/sugar_decoder_data.py`.
Bundle: `results/cx-sugar-decoder-data-20261112`.

The original prepared dataset is `results/cx-home-vector-hdeltac-long-data-20261104`.
The frozen activity/checkpoint source is
`results/cx-home-vector-balanced-u1975-activity-by-trajectory-20261106`.
Checkpoint SHA256:
`2aab91f13258883819f66f12897919b2bf509617e7927120ca7f9db069d9a31f`.

| Split | Original trajectories | New return families | New variants | Total trajectories |
| --- | ---: | ---: | ---: | ---: |
| Training | 2,048 | 64 | 512 | 2,560 |
| Validation | 192 | 16 | 128 | 320 |
| Test | 320 | 24 | 192 | 512 |

Each new family contains two repeats of a regular polygon (4, 8, or 24 sides), with a random orientation, turn sign, speed of 0.5–2 mm/s, and segment duration on a 0.25-second grid. Segment-duration choices are 1.5–6 seconds for 4 sides, 0.75–2.5 seconds for 8 sides, and 0.25–0.75 seconds for 24 sides. Each family has eight variants: forward/rightward traversal, continuous/paused movement, and sugar/control. A pause lasts 0.5, 1, 2, or 4 seconds at the first closure. New examples span 12–50 seconds overall. The simulation step remains 0.0025 seconds.

The environmental patch has a 0.1 mm entry radius and 0.15 mm departure radius. An event requires departure followed by reentry; there is a 30-second minimum delivery interval. These are experimental protocol assumptions, not measured sensory thresholds or a physiological refractory period. An ineligible contact is consumed rather than delivered later while the animal remains inside. The final zero-duration state does not trigger an unused event. Contact can precede exact geometric closure. This approximates delivery on contact, without modeling taste or consumption mechanics.

In the generated bundle, every treated new example receives exactly one delivery, near the first return. Delivery times span approximately 6–24 seconds. The second loop supplies continuation after that event; this is not a repeated-feeding experiment.

Independent random streams use seed 20261112. Family identities and exact movement protocols are checked for overlap across training, validation, and test. Fresh test examples use new parameters within the same polygon classes; they do not establish generalization to arbitrary new geometry classes.

The original 12 challenge trajectories are retained separately. Sixteen familiar diagnostic conditions use paths 8, 9, 14, and 15, with continuous/paused and sugar/control arms. These diagnostics preserve the historical scheduled sugar pulse at 8 seconds, rather than the new patch detector. They do not influence fitting or selection. The original extrapolation split remains archived at source and is not evaluated in this first comparison.

## Fixed dynamics and readout fit

The fixed response is copied from `docs/data/hdeltag_sugar_probe_example.json`:

```text
elapsed_s = time_s - delivery_time_s
g_event(elapsed_s) = 0                                         if elapsed_s < 0
g_event(elapsed_s) = (1 - exp(-elapsed_s / 1.0))
                     * exp(-elapsed_s / 300.0)                 otherwise
g(time_s) = max(g_event(time_s - event_s) over delivered events)
k_reset = 1.0  # per second

dr_i/dt = (T_i - r_i) / tau_i
          - g(time_s) * k_reset * (r_i - 0)                    for hDeltaG only
```

Other populations respond through unchanged recurrent connections. Activity begins at zero once per trajectory; there is no reset between loops. The gate is an imposed effective input, not a measured neural response. The experiment is too short to assess minutes-long recovery.

Implementation: `fly_connectome/home_vector/analysis/sugar_decoder_fit.py`.
Both dynamics are replayed without gradients. Each trajectory contributes its time-averaged sufficient statistics, excluding the final zero-duration state. Training and evaluation assign half the total weight to original examples and half to new return examples, then equal weight to families and their variants within each domain. This prevents duplicated control/treatment variants or longer paths from dominating by sample count.

For each network independently, training statistics determine feature means and scales. Ridge candidates are `1e-6, 1e-4, 1e-2, 1`; validation weighted vector MSE selects one. Intercepts are unpenalized. The final raw-coordinate readout is:

```text
predicted_home_vector = activities @ decoder.T + intercept
```

The four comparisons are `original_frozen`, `original_refit`, `sugar_frozen`, and `sugar_refit`. One decoder serves all conditions within each refitted dynamics. Phase scores reuse family weights, renormalized among eligible examples; each phase is time-averaged separately. First-return and final-return errors measure individual states and are reported separately from time averages.

CPU batches are used only to accelerate independent frozen replays. Zero-gate replay must match native dynamics exactly. A treated example from each applicable split is checked against the existing individual sugar rollout, allowing small floating-point batch differences. Checkpoints, source datasets, response configuration, and derived inputs are hash-checked for preservation; network parameters are checked unchanged after replay. Decoder weights are stored separately, never written into the original checkpoint.

## Reproduction and verification

```bash
python -m fly_connectome.home_vector.analysis.sugar_decoder_data \
  --source results/cx-home-vector-hdeltac-long-data-20261104 \
  --activity results/cx-home-vector-balanced-u1975-activity-by-trajectory-20261106 \
  --config docs/data/hdeltag_sugar_probe_example.json \
  --diagnostics results/cx-hdeltag-sugar-setup-smoke-20261112 \
  --output results/cx-sugar-decoder-data-20261112

OPENBLAS_NUM_THREADS=1 python -m fly_connectome.home_vector.analysis.sugar_decoder_fit \
  --data results/cx-sugar-decoder-data-20261112 \
  --output results/cx-sugar-decoder-fit-20261112-v3 --batch-size 16
```

Use a PyTorch-enabled project environment. Outputs must be new directories. Bundles contain manifests, copied runtime scripts, protocols/configuration provenance, aggregate sufficient statistics, fitted decoders, metrics, and familiar-path prediction plots. Training activity arrays are not all retained; the saved statistics support fitting/scoring audits, and protocols plus checkpoint support replay.

The focused tests cover event detection, matched geometry, split leakage, weighting, direct versus sufficient-statistic scoring, known-readout recovery, batched dynamics, and separate closure measurements. The comparison, existing sugar probe, and measurement inventory suites passed together: 20 tests.

## First completed comparison

The final bundle is `results/cx-sugar-decoder-fit-20261112-v3`; see its `report.md`, `metrics.csv`, and four `path_*.png` diagnostics. The first completed bundle without separate closure statistics is retained at `results/cx-sugar-decoder-fit-20261112`. An interrupted attempt at adding closure statistics is retained at `results/cx-sugar-decoder-fit-20261112-v2`; it is not a completed result. Both completed runs produced exactly identical fitted decoder arrays.

| Readout | Balanced test vector RMSE (mm) | First-return RMS error (mm) | Final-return RMS error (mm) |
| --- | ---: | ---: | ---: |
| Original frozen | 1.88882 | 1.48820 | 4.19191 |
| Original refitted | 1.17749 | 0.82281 | 1.48251 |
| Sugar frozen | 2.52526 | 1.50020 | 3.97318 |
| Sugar refitted | 1.59170 | 0.91980 | 2.04596 |

The balanced test metric includes the retained original test examples and the fresh return examples with equal domain mass. The closure columns concern only the 24 fresh test return families, averaged over their eight variants, including untreated controls. They are not sugar-only endpoint measurements. Both fits selected ridge `1e-6` using validation data, the smallest candidate in the predefined grid; no further search was made after inspecting the test results.

Within fresh return examples alone, refitted original and sugar dynamics have vector RMSEs of 0.78543 and 1.69085 mm, respectively. In the treated continuous arms these are 0.77039 and 2.23786 mm; in treated pause arms, 0.80019 and 2.17764 mm. Thus the comparison does not merely reflect a change on the retained no-sugar examples.

Refitting improves decoding under both dynamics. This supports the concern that applying the old readout to changed dynamics conflates readout mismatch with representational loss. However, this particular fixed sugar-response approximation still performs worse under its newly fitted readout than unchanged dynamics under an equivalently fitted readout. It has not demonstrated a decoding advantage. This does not rule out other dynamics or identify why errors persist, and it does not establish biological inaccuracy of sensory suppression. There is no evidence here that the model has solved long-duration drift or repeated-loop stability.

All original source hashes, completed bundle manifests, and archived runtime script identities were verified. An independent calculation from saved sufficient statistics reproduced all 268 final metrics, with maximum absolute discrepancy below `1.3e-11` mm. Treated batched versus individual replay checks differed by at most `1.15e-5` activity units; disabled gates matched native dynamics exactly. No recurrent parameters, response constants, prescribed PFN inputs, or original decoder were changed.
