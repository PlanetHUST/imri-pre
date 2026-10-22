# Coupled circular gas–GW orbit: implementation contract

Status: bounded implementation completed and verified (2026-10-22). This plan governs the bounded
single-host workflow change below; it does not redefine a cosmological rate.

## Scientific question and fixed assumptions

For each object already embedded by the existing damping calculation, replace
the selected gas-only/GW-only routes and invented crossover radius with one
circular orbit whose gas and GW angular-momentum changes act together. Keep
the sampled population, damping, static pAGN disk, fixed stellar/remnant mass,
fixed SMBH mass, and flat-LambdaCDM source clock unchanged. SEVN remnant-mass
evolution, stellar tidal disruption, disk back-reaction, stochastic torques,
eccentric waveforms, and a cosmological host-weighted rate are out of scope.

Embedding currently accepts `e <= 0.01`, not exact zero. Set `e=0` explicitly
at the migration boundary and record that projection. In Peters' orbit-averaged
law `de/dt` is proportional to `e`, so its exact circular limit remains
circular. The circular Type-I gas calculation is an adiabatic closure in which
gas power is `Omega * Gamma_gas`; it is an assumption, not a deduction from
the damping threshold. Keep the existing fixed-mass torque prescription for
both species in this first comparison; record the unresolved pre-collapse
luminosity assumption rather than adding unapproved stellar-accretion physics.

For `M=m1+m2`, `mu=m1*m2/M`, and circular separation `r`:

```
L(r) = mu sqrt(G M r)
dr/dt = [Gamma_gas(r) + Gamma_GW(r)] / [dL/dr]
dr/dt |_GW = -beta/r^3
beta = (64/5) G^3 m1 m2 M / c^5
```

The code may use a one-dimensional time/radius quadrature on a monotone
segment, with a signed ODE or equivalent event-aware fallback near net-torque
zeros and outward paths. It must never divide through a net-torque zero or
replace the combined law with a route-timescale comparison. After the AGN
turns off or the orbit leaves the gas profile, use the exact circular vacuum
relation `r(t)^4 = r_off^4 - 4 beta (t-t_off)`.

## Ownership and files to touch

1. `src/physical_process/orbital_evolution.py` (new): unit-explicit circular
   gas-plus-GW evolution, first-band-entry event, disk on/off events, finite
   cosmic horizon, and typed result/status. Gas torque is evaluated only
   within its profile. The same redshift clock sets the moving observed-band
   boundary. Net-torque equilibria are physical stalls, not handoffs.
2. `src/physical_process/migration/semi_analytic.py`: expose its existing
   local positive timescale as an inward radial-drift profile without copying
   the Type-I/II formulas. `src/physical_process/migration/torque.py` and
   `evolution.py` retain ownership of the signed reference torque and its
   historical diagnostics; do not replace either prescription.
3. `src/workflows/event_work.py`, `event_rate.py`: adapt both existing scenario
   branches to one orbit interface, classify BH formation before physical
   band entry, preserve distinct evaluated zero/unavailable/error statuses,
   and count one finite BH band entry. Do not equate entry with coalescence.
4. `src/lisa/detectability.py` and `src/lisa/snr.py`: consume the orbit's
   band-entry state and continue the *same coupled radial law* through the
   detector window, without a separate gas/GW route or selected crossover.
   For gas-active entry, use the circular quadrupole amplitude with the
   actual gas+GW residence time in band. In observer time,
   `rho^2=(16/5) integral A_vac(f)^2 dot_f_GW(f) / S_n(f) dt_obs`;
   this is the stationary-phase identity
   `|h_tilde|^2=|h_tilde,vac|^2 dot_f_GW/dot_f_actual` written in time to
   remain finite near a slow chirp. Preserve the existing detector noise,
   source/detector mass and mission-window conventions. This is a first-order
   amplitude/residence-time SNR, **not** an environmental phase template or
   a claim that matched filtering with an unmodeled gas phase is available.
5. `src/workflows/models.py`, `outputs.py`, `metadata.py`, configuration and
   scenario entry points: add unambiguous orbit/disk-off/band-entry fields and
   scenario/version provenance. Retain legacy CSV columns only through a
   documented translation, never call an arbitrary crossover `gas_final`.
   The old one-event SMBH mass/radius jump is incompatible with fixed masses:
   default it off and reject finite use in the coupled path; standalone legacy
   diagnostics may remain available.
6. `tests/` and a focused `scripts/validate_coupled_orbit.py`: production-API
   physical checks, status/count/schema checks, and a reduced paired-seed run.
   Update current physics docs and mark superseded routing assertions clearly.

## Files and behavior not to touch

- Preserve the user's existing `README.md` edit and all unrelated worktree
  state. Do not rewrite archived transcripts, generated figures, or outputs.
- Do not alter population RNG, damping equations/embedding rules, pAGN disk
  construction, the Jimenez–Masset/Grishin–2024/Kanagawa torque equations, or
  the comparison branch's Type-I/II timescale law.
- Do not add a suite of new torque toy models. Keep the existing
  `torque_with_trap` and `semi_without_trap` scenario identities. A GW-only
  case is a diagnostic control, not another production migration model.
- Do not silently change detector noise, luminosity distance, mission
  duration, SNR threshold, or the per-AGN-episode rate denominator.

## Observable semantics

- The output's `n_imri` remains a BH's finite first observed LISA-band entry
  before ISCO and the `z=0` horizon; it is neither a merger count nor a
  cosmic rate. A star that has not become a BH by entry is not counted.
- Record `orbit_stop_reason`, `band_entry_time_myr`, `band_entry_radius_cm`,
  `band_entry_redshift`, the final orbit time/radius/redshift,
  `disk_off_time_myr`, `disk_off_radius_cm`, `disk_off_reason`,
  `gas_active_at_entry`, and
  `orbit_model`. Existing `detection_time_myr` remains the band-entry time.
- `r_gas_fin_cm`/`gas_final_time_myr` are legacy compatibility fields: when
  gas is still active at entry they are unavailable, and otherwise they mean
  the physical disk-off state. They never mean a model-selected crossover.
- Detector statuses remain evaluated, evaluated nondetection, model
  unavailable for genuinely unsupported inputs, invalid input/profile, and
  numerical failure as distinct states. Gas-active band entry is evaluated
  with the first-order coupled-track SNR. Scenario ID is persisted in every
  event and aggregation layer.

## Work sequence and pass gates

1. Capture current 87-test baseline, old scenario/status counts, and one
   fixed-seed reduced run. Write the typed orbit kernel and test analytic
   vacuum and gas-only limits before changing orchestration.
2. Adapt the two existing gas prescriptions to the same unit-explicit drift
   interface. Test signed aiding/opposing torques, a stable net equilibrium,
   outward motion, disk shutdown, and inner/outer profile exit. Check the
   circular energy/angular-momentum relation and resolution convergence.
3. Replace production routing and detector continuation. Test first crossing
   of the moving observed-frequency boundary, BH formation timing, inside-
   ISCO and cosmic-horizon statuses, and continuation of the same orbit law.
   Verify the time-domain SNR reduces to the existing vacuum frequency-domain
   result and yields finite gas-active SNR for both mission placements.
4. Translate outputs and metadata, run both scenarios with identical seeds
   and upstream objects, and reconcile per-event flags with repeat/average
   counts and explicit rate denominators. Verify strict CSV/JSON schemas.
5. Run the full unit suite, focused orbit diagnostic, LISA validation, code
   audit, `git diff --check`, and a reduced seeded end-to-end simulation.

Final pass requires all of these: no selected crossover or arbitrary gas-final
radius on the production path; vacuum limit agrees with Peters within the
declared numerical tolerance; gas-only monotone limit agrees with the existing
prescription; net-torque zeros and disk shutdown have correct distinct
statuses; scenario comparisons are seed-paired and not pooled; first-entry
times/radii/redshifts are self-consistent; detector continuation uses the
same coupled law and reproduces the vacuum SNR limit; all counts/statuses and serialized units reconcile; and the
worktree contains no changes to out-of-scope user files. A successful smoke
run alone does not meet this gate.

## Execution record (2026-10-22)

- The production event path now uses `evolve_circular_orbit` and, for a BH
  with gas-active entry, `continue_circular_band_track` from that same entry
  state. Neither invokes the historical gas-final-radius selector. The two
  migration scenarios share the same orbit and detector interfaces.
- The reduced seed-paired diagnostic is
  `python scripts/validate_coupled_orbit.py OUTPUT_ROOT`. It saves a
  `paired_entry_comparison.csv` and checks exact upstream object matching,
  first-entry frequency/epoch consistency, strict metadata JSON, and
  event/repeat counts and rate denominators. On the 1e5-Msun SMBH,
  5e4-Msun NSC, `z=15`, 30-Myr, seed-424242 case, both branches had eight
  embedded objects and four BH first entries; the three gas-active BH entries
  in each branch received finite first-order SNRs.
- The focused limits include Peters vacuum, gas-dominated local-timescale,
  aiding/opposing gas, exact and stable net equilibria, profile entry/exit,
  disk shutdown, moving redshift boundary, circular flux consistency, both
  mission placements, and vacuum SNR agreement. Full suite: 102 passed,
  zero failed. Legacy LISA SNR validation and `git diff --check` passed.
- Interpretation limit: a finite first-order SNR is a quadrupole signal-
  strength forecast under a known coupled trajectory. Environmental phase
  templates, matched-filter recovery, SEVN mass changes and cosmological
  host weighting remain outside this contract.
- `scripts/audit_repository.py` also ran. Its static report retains older
  repository-wide findings (including legacy scripts and classification
  heuristics), so its zero process exit is not presented as a scientific
  acceptance test for this change.
