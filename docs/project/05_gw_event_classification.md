# Part 05 — GW Evolution and Event Classification

## 1. Idea

Given an embedded object and the migration machinery (part 04), this part
decides **whether an IMRI candidate is formed**. It owns:

1. the **LISA band geometry**: the formal circular entry radius from a lower
   observed GW frequency, converted to the source frame;
2. the **band-entry routing**: whether that entry is inside the migration
   domain, below it (needing a gas-to-GW handoff), inside ISCO (unsupported), or
   unclassifiable (profile error);
3. the **gas/GW transport race**: comparing, at every radius along the path,
   the remaining gas-migration time versus the remaining Peters GW-only time,
   bounded by the AGN duty cycle;
4. the **candidate classification**: embedded BH (`ebh`) or migration-formed BH
   (`mfbh`) versus no-BH.

The main idea: *an embedded object becomes an IMRI candidate when gas (and, if
necessary, vacuum GW inspiral from a handoff) can get it to the LISA-band entry
before the duty cycle runs out — with migration traps and GW dominance treated
as explicit, distinguishable outcomes.*

## 2. Doing logic

### 2.1 Band geometry

Authoritative owner: `src/physical_process/gw_frequency_radius.py`.

Frequency frame relation (fixed convention from
[`../reference/physical_model.md`](../reference/physical_model.md)):

```
f_src = (1+z) f_obs        f_obs = f_src / (1+z)
```

For a circular, nonspinning binary, the dominant quadrupole frequency gives the
formal entry radius:

```
r_entry,formal = [ G(M_smbh + m) / (π f_src,min)² ]^(1/3)
```

compared against the Schwarzschild ISCO:

```
r_ISCO = 6 G(M_smbh + m) / c²
```

`circular_band_boundary()` performs this comparison **in cm** and returns the
pre-ISCO validity flag. `peters_merger_time_myr()` implements the Peters (1964)
inspiral (analytic circular limit, numerical eccentric integral otherwise);
`peters_time_to_lisa_band_myr()` and `peters_radius_after_time_cm()` support the
GW legs.

### 2.2 Band-entry routing

Authoritative owner: `src/workflows/event_work.py`
(`resolve_band_entry_route`, `BAND_ENTRY_*` constants).

For a valid entry (above ISCO), compare `r_entry,formal` against the operational
lower migration-domain radius `max(R_disk,in, R_migration,min)` (with an explicit
numerical tolerance):

- entry inside the domain → `entry_in_migration_domain`, gas target = entry;
- entry below the domain → `entry_below_migration_domain_but_above_isco`, gas
  target = handoff radius = the lower domain bound, GW covers the rest.

`band_entry_inside_isco` (entry ≤ ISCO) is handled **before** any migration
profile is built — no torque/semi profile is constructed, no integrator is
called, and the object is retained with an explicit unsupported status.
`profile_error` is reserved for genuine profile/model failures.

### 2.3 The gas/GW transport race

Authoritative owner: `src/workflows/event_rate.py`
(`solve_gas_final_radius_cm`, `classify_event_rate`).

Available gas time after embedding: `available = duty_cycle - embed_time`.
Along the path from `r_ini` to the gas target, the solver samples radii and, at
each one, compares the remaining gas-assisted time to the remaining GW-only
time, stopping with one of these `gas_stop_reason` values:

| Reason | Meaning |
|---|---|
| `already_inside_lisa_entry` | `r_ini` already at/below entry |
| `entry_reached` | gas reaches the in-domain entry |
| `gas_to_gw_handoff` | gas reaches the handoff; GW covers the rest |
| `gw_dominates_at_start` | GW-only is faster already at `r_ini` |
| `gw_dominates_after_gas` | GW becomes faster after partial gas delivery |
| `duty_cycle_limited` | AGN lifetime exhausted first |
| `damping_misses_duty_cycle` | no positive gas time remains after embedding |
| `migration_trap_bound` | a torque trap blocks the path |
| `invalid_radius` | nonfinite/nonpositive radii |

BH formation logic: `embedded_bh` (object already a BH) or
`migration_formed_bh` (embedded star whose formation time is passed before the
gas-final time). A candidate is formed when the object is a BH candidate and the
stop reason is one of the five delivery/dominance reasons; otherwise
`no-IMRI`. `band_entry_inside_isco` yields `not evaluated`.

### 2.4 Candidate classes and origins

```
event_rate_class : "imri candi" | "no-IMRI" | "not evaluated"
bh_origin         : "ebh" | "mfbh" | "no_bh"
```

The event record (built in `event_work._imri_event_record`) stores the full
radial path bookkeeping: `r_ini`, `r_gas_fin`, `r_entry`, `r_entry_formal`,
`r_isco`, `r_disk_inner`, `r_migration_min`, `r_handoff`, plus gas/GW times and
frequencies at each point.

## 3. Key configuration

| Parameter | Default | Meaning |
|---|---|---|
| `lisa_min_gw_frequency_hz` | 1e-3 | observed lower LISA band edge |
| `lisa_max_gw_frequency_hz` | 1e-1 | observed upper LISA band edge |
| `gw_eccentricity` | 0.0 | fixed eccentricity for Peters legs |
| `duty_cycle_myr` | 10.0 | AGN episode length (rate denominator) |
| `GAS_FINAL_RADIUS_SAMPLES` | 96 | radial sampling of the transport race |

## 4. Caveats

- **Boundary comparisons happen in cm before any `rg` reporting.** Do not
  compare a total-mass ISCO expressed in one gravitational-radius convention
  with a disk radius expressed in another.
- **`band_entry_inside_isco` is a physical band-status, not a torque error and
  not a zero rate.** It must never reach the torque integrator, and it must not
  increment torque counters.
- **`np.inf` has one meaning only.** In migration outcomes it is a physical trap;
  in band entry it is never used as a sentinel (the inside-ISCO case is an
  explicit status). Do not overload one `inf` for both.
- **Gas target ≠ detector entry in the handoff case.** The gas migration time is
  measured to the handoff, not to the formal entry; the GW bridge covers
  `handoff → entry`. `migration_time_to_handoff_myr` and
  `gw_time_handoff_to_entry_myr` are logged separately.
- **Don't extrapolate below the domain.** The handoff uses the profile minimum as
  a hard floor; a start radius *above* the outer domain remains a real error.
- **Candidacy semantics are deliberately conservative.** `duty_cycle_limited`
  and `migration_trap_bound` objects are `no-IMRI` here; whether they would be
  GW-viable later is explicitly future physics
  ([`../reference/physical_model.md`](../reference/physical_model.md)).
