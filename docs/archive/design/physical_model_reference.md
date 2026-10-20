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

The adaptive step controller follows the newer root-level
`cos_tde_revised/damp.py`: rejected trials reuse the same orbital terms and may
shrink below the former `1e-6` fraction floor. `max_iter` counts accepted
steps. Exhaustion of a positive step budget before the AGN deadline,
non-finite trials, and a stalled clock are numerical failures
(`calculation_error`), distinct from physical `damping_incomplete`. The
existing `max_iter=0` no-integration diagnostic behavior is retained.
This alignment preserves IMRI's corrected node equations, solved gas density,
vertical crossing speed, and embedding contract. The TDE main runner's pinned
July snapshot remains a legacy comparison, not the alignment target. See
[`../project/03_disk_embedding_dynamics.md`](../project/03_disk_embedding_dynamics.md#5-comparison-with-cos_tde_revised-and-numerical-alignment)
and `scripts/compare_damping_reference.py` for matched-input evidence.

For phase-space diagnostics, periapsis/apoapsis
`a(1-e)`/`a(1+e)` are not the two node radii. Preserve censored threshold times
as unavailable rather than extrapolating a trajectory.

## 3. Shared disk and migration models

`disk_model` selects the numerical migration path. Both paths share the same
pAGN disk, population draw, disk crossing, damping, entry geometry, Peters
evolution, and rate normalization.

| Behavior | `torque_with_trap` | `semi_without_trap` |
|---|---|---|
| Evolution | signed torque integration | integrated local timescale |
| Physical torque barrier | `trap_bound`, optional blocking radius | impossible by construction |
| Invalid profile/path | explicit migration error | explicit migration error |

The torque path has a separate `migration_model` selection:

| `migration_model` | Selected torque |
|---|---|
| `continuous_gap` (default) | continuously gap-reduced gravitational torque plus raw thermal torque |
| `classical_type2` | legacy hard switch to classical Type-II torque |
| `ungapped` | ungapped gravitational plus raw thermal torque |

The legacy `consider_type_ii_migration=False` control is retained as a master
override to `ungapped`. `gap_opening_condition` affects torque selection only
for `classical_type2`; in the default continuous model, `K > 11` is a
diagnostic deep-gap label and never a branch switch.

### Continuous torque path

Composition occurs in SI. With mass ratio `q`, aspect ratio `h=H/R`, and
viscosity `alpha`:

```text
K = q^2/[alpha h^5]
f_gap = 1/[1 + 0.04 K]
Gamma_grav,gap = f_gap Gamma_grav,ungapped
Gamma_total = Gamma_grav,gap + Gamma_thermal.
```

The implementation exposes `K`, `f_gap`, the deep-gap diagnostic, and

```text
B = 4 pi Sigma R^2/m
```

alongside the ungapped/gap-reduced gravitational, thermal, classical Type-II,
and selected total torques. `B` is diagnostic only; no secondary-dominated
Type-II law is implemented.

The gap-depth relation is the empirical Kanagawa et al. prescription for
`Sigma_gap/Sigma_0`, and Derdzinski & Mayer use the same factor as a continuous
torque reduction in an AGN/EMRI calculation:

- Kanagawa et al. (2015), *Mass Estimates of a Giant Planet in a
  Protoplanetary Disk from the Gap Structures*, ApJL 806 L15,
  [doi:10.1088/2041-8205/806/1/L15](https://doi.org/10.1088/2041-8205/806/1/L15),
  [arXiv:1505.04482](https://arxiv.org/abs/1505.04482).
- Derdzinski & Mayer (2023), *In situ extreme mass ratio inspirals via
  subparsec formation and migration of stars in thin, gravitationally unstable
  AGN discs*, MNRAS 521, 4522,
  [doi:10.1093/mnras/stad749](https://doi.org/10.1093/mnras/stad749).

This is a controlled first-order transplant: the calibrated gap-depth fit
assumes a non-self-gravitating, locally isothermal planet disk and is not
validated across every pAGN thermodynamic or high-`K` regime. Kanagawa et al.
specifically caution that the simple estimate becomes unreliable for
`K` appreciably above about `10^4`.

### Classical comparison path

`classical_type2` exactly retains the former hard-switch behavior:

```text
Gamma_total = Gamma_II where gap_open
Gamma_total = Gamma_grav,ungapped + Gamma_thermal otherwise.
```

The mask is selected by `kii` or `crida`:

```text
K = q^2/[alpha h^5] > 11

P_Crida = 0.75/(R_H/H) + 50 alpha h^2/q < 1
and R_H/H > 1.
```

The combined pressure-viscosity criterion follows Crida, Morbidelli & Masset
(2006), *On the width and shape of gaps in protoplanetary disks*, Icarus 181,
587, [arXiv:astro-ph/0511082](https://arxiv.org/abs/astro-ph/0511082).

### Signed integration

Convert SI torque to CGS exactly once at the migration-time consumer. The
signed path integrates

```text
t_mig = integral (dL/dR)/Gamma dR,
dL/dR = 0.5 m R Omega.
```

Only a physical non-inward path or torque sign barrier returns infinite time
with `trap_bound`. Malformed, nonfinite, insufficient, or out-of-domain input
raises a migration error.

### Semi-analytic comparison path

The local times are

```text
t_I(R)  = M^2 h(R)^2/[2 m Sigma(R) R^2 Omega(R)]
t_II(R) = 1/[2 alpha h(R)^2 Omega(R)],
t_mig   = integral t_local(R) d ln R.
```

A valid nonzero path is finite and positive. Errors never become infinity or
traps. The semi path does not construct a detailed torque profile or classify
torque roots. It retains its historical Type-I/Type-II timescale switch for the
`semi_without_trap` comparison and does not use the torque-path
`migration_model` selector. Accordingly, `scripts/migration_test.py` plots this
legacy semi-analytic reference only with `classical_type2`. For
`continuous_gap` and `ungapped`, it reports the reference as not applicable:
neither has an equivalent local-timescale estimate containing the signed
thermal torque and physical trap semantics.

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
