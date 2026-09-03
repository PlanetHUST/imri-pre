# Part 04 — Gas Migration

## 1. Idea

An embedded object (part 03) sits on a circular orbit inside the disk and is
acted on by the disk torque. Gas migration determines **how far inward the disk
can deliver the object within the AGN duty cycle**, and whether inward delivery
is physically possible at all.

Two migration prescriptions exist, selected by `--disk-model`:

- **`torque_with_trap`** (default): a detailed, signed pAGN torque profile per
  migrator mass. Torque zeros (stable traps) can *physically block* inward
  delivery → `trap_bound`.
- **`semi_without_trap`**: an integrated local semi-analytic timescale over
  `d ln R` that is always inward and positive — it cannot terminate at a torque
  trap.

The main idea: *migration is a strict radial-path integral over an explicitly
selected torque/timescale profile. Physical blockage (`trap_bound`) and
numerical/model failure are entirely different outcomes and must never be
confused.*

## 2. Doing logic

### 2.1 Shared gap-opening selection

Authoritative owner: `src/physical_process/disk_migration.py::gap_opening_mask()`.

Both scenarios and both branches use the same selector for the Type-II
(gap-open) branch:

- `kii` (default): `K_II = q² / (α (H/R)²) > 11`;
- `crida`: `P_Crida = 3H/(4R_H) + 50α(H/R)²/q < 1` and `R_H/H > 1`.

`--no-type-ii-migration` forces the mask to all-false (Type-I only).

### 2.2 Torque scenario

`select_migration_torque_si()` composes, in **SI**, one selected torque array:

```
Γ_migration(R) = Γ_I + (1 - f_gap) Γ_th          gap-closed cells
Γ_migration(R) = Γ_II                             gap-open cells (selected mask)
```

where `f_gap = torque_profile.gap_attenuation_factor`, `Γ_II` replaces the
whole branch (never added on top). This is then converted to CGS exactly once
(`torque_si_to_cgs()`) at the migration-time consumer boundary.

`torque_migration_outcome()` (`disk_migration.py`) integrates

```
t_mig = ∫_{R_ini}^{R_entry} (dL/dR) / Γ_migration(R) dR,   dL/dR = ½ m R Ω
```

with the contract:

- `r_ini ≤ r_entry` → `already_at_target` (time 0);
- continuous inward negative torque → finite positive `time_myr`;
- any nonnegative torque on the path (zero/sign crossing, outward torque) →
  `trap_bound` with `time = inf` and an optional blocking radius (the outermost
  stable root; `null` if none is resolved);
- malformed grids, nonfinite samples, or out-of-domain radii → raise a
  `TorqueMigrationError` (an error, never `inf`, never a trap).

### 2.3 Semi-analytic scenario

`semi_analytic_migration_outcome()` uses the same physical disk arrays
(`radius_cm`, `surface_density_g_cm2`, `aspect_ratio`, `torque_alpha`) and the
same shared gap mask, with local timescales

```
t_I(R) = M² h² / (2 m Σ R² Ω)
t_II(R) = 1 / (2 α h² Ω)
```

and integrates `t_mig = ∫ t_local(R) d ln R`. Both branches are positive and
inward; invalid inputs raise `SemiAnalyticMigrationError`. The outcome never has
`trap_bound`; blocking radius is always `None`.

### 2.4 Event-layer usage

Authoritative owner: `src/workflows/event_work.py::IMRIEventEvaluator`.

For each embedded object:

1. the band boundary is computed (part 05) before any migration profile;
2. for an in-domain or handoff entry, the evaluator picks exactly one profile:
   torque profile (torque scenario) or disk arrays (semi scenario), caches the
   per-object outcome callback, and resolves the route (handoff radius etc.);
3. `migration_outcome_call(start, target)` memoizes each `(start, target)` pair
   so the repeated radial samples of the gas-final-radius solver (part 05) use
   one consistent profile.

## 3. Outcome taxonomy (model-neutral)

| Status | Meaning | Which scenario |
|---|---|---|
| `already_at_target` | `r_ini ≤ target` | both |
| `finite` | continuous inward delivery | both |
| `trap_bound` | physical torque blockage (time `inf`, blocking radius optional) | torque only |
| error (`MigrationError`…) | profile/grid/domain failure | both |

Compact counters: `n_mig_eval = n_mig_finite + n_mig_trap + n_mig_error`
(torque) and `n_mig_eval = n_mig_finite + n_mig_error` (semi, `n_mig_trap = 0`).

## 4. Caveats

- **`inf` is reserved for physics, errors are separate.** `trap_bound` means the
  disk torque structure prevents inward delivery. Profile/numerical/domain
  failures are explicit errors. Neither may be converted into the other, and
  neither may become a zero rate.
- **No semi-analytic fallback.** The production torque path must never retry
  with a semi-analytic formula after a trap or error
  ([`../reference/history.md`](../reference/history.md)).
- **Unit discipline.** Torque composition happens in SI; CGS conversion happens
  once, at the migration-time consumer. Never mix.
- **Gap selection is shared.** The mask used by the production torque scenario
  must be exactly the same `gap_opening_mask()` that the semi scenario and the
  diagnostic scripts use; no caller copies the KII/Crida formulas.
- **The two scenarios are a prescription bracket, not disk uncertainty.** Both
  use the same pAGN disk for Σ, H/R, boundaries, damping, and gap decisions
  ([`../reference/physical_model.md`](../reference/physical_model.md)).
- **Profile minimum is the operational domain.** The lower migration-domain
  boundary comes from the actual solved profile/torque arrays
  (`r_migration_min_cm`), not from the `DISK_INNER_RADIUS_RG` config constant.
