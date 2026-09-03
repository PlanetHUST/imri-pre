# Durable Physical-Model Contract

This document condenses the physical invariants formerly spread across legacy
task prompts. It is a long-term guardrail, not a substitute for an active
specification or current source inspection.

## 1. Quantity and pipeline

The implemented local model estimates the intrinsic number of stellar-mass
black holes delivered into the modeled LISA-entry channel during one AGN
episode around one SMBH/NSC host:

```text
R_IMRI = N_IMRI / T_AGN  [Myr^-1 per host per AGN episode]
```

It is not a cosmological volumetric rate. The physical stages are:

```text
NSC population and orbits
-> disk-plane crossing
-> gas damping and embedded-state decision
-> gas migration or gas-to-GW handoff
-> intrinsic IMRI classification
-> conditional LISA detector classification
```

Each stage retains its own status and denominator so a later failure cannot
rewrite an earlier physical result.

## 2. Disk crossing and damping

For semi-major axis `a`, eccentricity `e`, and stored `cos_omega`, the disk
plane nodes are

```text
r_node,1 = a(1-e^2)/(1-e cos(omega))
r_node,2 = a(1-e^2)/(1+e cos(omega)).
```

An orbit crosses the physical disk annulus `[R_in, R_out]` only if at least one
actual node lies inside it. Interval overlap between the two node radii is not
equivalent to node membership.

During damping:

- recompute both nodes from every accepted evolved state;
- sample gas and sum orbital changes only for active nodes;
- an inactive node contributes zero and cannot set the time step;
- no node may be clamped to a disk-edge gas cell;
- no active nodes produce an outside/incomplete physical outcome, not a
  calculation error or an embedded status.

An embedded handoff is valid only when the final evolved state satisfies all
of:

```text
abs(inclination_rad) <= local aspect ratio H/R
0 <= eccentricity <= 0.01
semi_major_axis_cm * (1-eccentricity) >= disk_inner_radius_cm
semi_major_axis_cm * (1+eccentricity) <= disk_outer_radius_cm
```

The outer radius is the physical pAGN `Q=1` boundary. A predicted future
star-to-BH damping time must not fabricate an unevolved final orbit. Damping
statuses remain descriptive (`embedded_star`, `embedded_bh`,
`damping_incomplete`, `outside_disk`, `calculation_error`, and diagnostic-only
test state).

For phase-space diagnostics, periapsis/apoapsis
`a(1-e)`/`a(1+e)` are not the two node radii. Preserve censored threshold times
as unavailable rather than extrapolating a trajectory.

## 3. Shared disk and two migration scenarios

`disk_model` changes the migration prescription only. Both scenarios share the
same pAGN disk, population draw, disk crossing, damping, gap selection, entry
geometry, Peters evolution, and rate normalization.

| Behavior | `torque_with_trap` | `semi_without_trap` |
|---|---|---|
| Gap-closed branch | selected Type-I plus attenuated thermal torque | integrated local Type-I time |
| Gap-open branch | Type-II torque replaces the complete branch | integrated local Type-II time |
| Physical torque barrier | `trap_bound`, optional blocking radius | impossible by construction |
| Invalid profile/path | explicit migration error | explicit migration error |

The shared gap mask is selected by `kii` or `crida`:

```text
KII = q^2/[alpha (H/R)^2] > 11

P_Crida = 0.75/(R_H/H) + 50 alpha (H/R)^2/q < 1
and R_H/H > 1.
```

With Type II disabled, each scenario uses its gap-closed branch everywhere.

### Torque path

Composition occurs in SI:

```text
Gamma_closed = Gamma_I + (1-f_gap) Gamma_th
Gamma_selected = Gamma_II where gap_open else Gamma_closed.
```

Convert SI torque to CGS exactly once at the migration-time consumer. The
signed path integrates

```text
t_mig = integral (dL/dR)/Gamma dR,
dL/dR = 0.5 m R Omega.
```

Only a physical non-inward path or torque sign barrier returns infinite time
with `trap_bound`. Malformed, nonfinite, insufficient, or out-of-domain input
raises a migration error.

### Semi-analytic path

The local times are

```text
t_I(R)  = M^2 h(R)^2/[2 m Sigma(R) R^2 Omega(R)]
t_II(R) = 1/[2 alpha h(R)^2 Omega(R)],
t_mig   = integral t_local(R) d ln R.
```

A valid nonzero path is finite and positive. Errors never become infinity or
traps. The semi path does not construct a detailed torque profile or classify
torque roots.

## 4. Band-entry and high-mass routing

Compute the formal circular observed-frequency entry and Schwarzschild ISCO
before constructing a migration profile. Preserve finite diagnostic radii even
when an operational route is unavailable.

| `band_entry_status` | Meaning | Route |
|---|---|---|
| `entry_in_migration_domain` | formal entry lies in selected migration domain | normal migration to entry |
| `entry_below_migration_domain_but_above_isco` | physical entry is valid below the numerical profile | gas to handoff, then Peters GW |
| `band_entry_inside_isco` | formal circular entry is at/below ISCO | skip migration and detector evaluation |
| `profile_error` | selected profile/model failed | explicit unavailable error |

For a valid lower-domain mismatch:

```text
r_handoff = max(r_entry_formal, r_disk_inner, r_migration_min).
```

Gas never evaluates below `r_handoff`; Peters evolution bridges to the formal
entry. A start radius above the profile maximum remains a real domain error.
A handoff cannot bypass a physical trap.

For `band_entry_inside_isco`, do not construct or call migration, do not count
a profile error, and do not report a physical zero or non-detection. Intrinsic
plunge/merger and waveform behavior inside this unsupported circular boundary
require a separate model.

## 5. Gas/GW and event semantics

Gas continues only while its remaining delivery time is strictly shorter than
the remaining Peters GW-only time. Transport outcomes keep at least these
distinctions:

```text
already_inside_lisa_entry
entry_reached
gas_to_gw_handoff
gw_dominates_at_start
gw_dominates_after_gas
duty_cycle_limited
damping_misses_duty_cycle
migration_trap_bound
invalid_radius
```

Embedded BHs and embedded stars that form BHs before the relevant delivery
deadline remain distinct origins. A source becomes an intrinsic IMRI candidate
only through the event-rate owner; detector classification cannot create or
erase intrinsic candidates.

## 6. Detector boundary

The detector layer is governed by
[`../specs/lisa_detectability.md`](../specs/lisa_detectability.md). `lisa_snr`
is the authoritative first-order, mission-limited, sky/orientation-averaged
circular-inspiral statistic. The retained `snr_proxy` field is only a deprecated
compatibility alias.

Physical no-overlap is an evaluated zero SNR. Missing input, unsupported model,
and numerical failure are unavailable. Detector unavailability invalidates the
detector-selected rate only; it does not invalidate the intrinsic rate.

## 7. Status and denominator hierarchy

Never pool these layers:

| Layer | Canonical variable | Denominator |
|---|---|---|
| Entry geometry | `band_entry_status` | all `n_band_eval` outcomes |
| Gas/GW transport | `transport_status` | nonblank evaluated transport outcomes; inside-ISCO excluded |
| Intrinsic event | `event_outcome` | retained event rows |
| Detector result | `detectability_outcome` | retained event rows, with non-candidates as not evaluated |

For positive denominators, category counts conserve exactly and fractions sum
to one within numerical tolerance. A zero denominator is unavailable, not a
set of zero fractions. Keep valid zero, physical blockage, invalid physical
regime, numerical/domain failure, and not evaluated as separate states.

## 8. Scenario comparison

Use independent runs with paired task seeds:

```text
SeedSequence(base_seed, smbh_index, nsc_index, repeat_index)
```

Redshift and `disk_model` are deliberately excluded. Upstream stochastic
draws should match; differences may begin only at the selected migration
prescription and its consequences. Persist `disk_model` in every event,
repeat, average, summary, metadata, and figure-data identity. Never average
the two scenarios together or emit zero/infinity for an undefined ratio.
