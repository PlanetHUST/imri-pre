# Part 01 — Environment Build

## 1. Idea

Every realization is set inside one fixed environment: a supermassive black
hole of mass `M_smbh` surrounded by a nuclear star cluster (NSC) and an active
galactic nucleus (AGN) accretion disk. The disk is the stage on which damping
and gas migration act; the NSC is the reservoir from which the population is
drawn; the SMBH defines the gravitational potential and the LISA band geometry.

The environment part owns three things:

1. the **pAGN disk structure** — radial profiles of density, temperature,
   surface density, aspect ratio, Toomre Q, etc.;
2. the **AGN state** — SMBH mass, duty cycle, gravitational radius;
3. the **NSC scale** — the mass–radius relation that bounds orbital sampling,
   and the disk/torque **caches** so repeated realizations do not rebuild
   expensive profiles.

The main idea: *the environment is a function of SMBH mass alone (plus a few
fixed parameters), so it is constructed once per SMBH mass and shared across all
repeats and objects.* Only the migrator-mass-dependent torque profiles add a
second dimension to the cache.

## 2. Doing logic

### 2.1 pAGN disk adapter

Authoritative builder: `src/disk_quantity/pagn_disk_adapter.py::build_pagn_disk_profile()`.

Inputs: `smbh_mass_msun`, `alpha` (viscosity), and a chosen model:

- `model="sirko"` (default): accretion rate fixed by
  `DISK_ACCRETION_RATE_G_S_PER_MSUN * M_smbh` (1.44e18 g/s per Msun);
- `model="thompson"`: outer radius fixed by `thompson_rout_rg`.

Steps inside the adapter:

1. pAGN solves the disk in **SI**; the adapter converts to **CGS** at the
   boundary (`radius_cm`, `surface_density_g_cm2`, `aspect_ratio`,
   `density_g_cm3`, `sound_speed_cm_s`, `opacity_cm2_g`, …).
2. The **outer disk radius is the first outward Toomre `Q=1` boundary**
   (`TOOMRE_Q_OUTER_BOUNDARY = 1.0`). For Sirko disks the adapter coarse-searches
   outward beyond the built-in `Rmax`, re-solves with the self-gravity boundary
   as the effective outer scale, and mirrors the result onto `Rout`/`R_out`.
3. The inner radius comes from the pAGN `Rin`/`R_in` attribute (falling back to
   the profile minimum). The Schwarzschild radius `Rs` gives
   `gravitational_radius_cm = 0.5 * Rs`.

### 2.2 AGN state

Authoritative owner: `src/physical_process/agn.py::AGNProperties` (frozen
dataclass). Built once per realization via
`AGNProperties.from_mass(m_smbh_msun, duty_cycle_myr)`:

- `smbh_mass_msun`, `agn_lifetime_myr`, `gravitational_radius_cm`.

This is the only object the damping workers need to know about the AGN.

### 2.3 NSC scale

Authoritative owner: `src/physical_process/nsc_orbital.py::estimate_nsc_radius_cm()`.

```
R_nsc[cm] = reference_radius_pc * (M_nsc / reference_mass_msun)^mass_slope * pc
```

with defaults `3.31 pc` at `3.6e6 Msun`, slope `0.321`. The NSC radius is the
**upper bound of orbital sampling** (see part 02). Relaxation timescales are
also computed here (`relaxaion_timescale_s`) as diagnostics of the cluster
environment.

### 2.4 Engine caches

Authoritative owner: `src/workflows/engine.py::IMRIDynEngine`.

- `disk_cache`: one `DiskProfiles` dict per SMBH mass, keyed by `m_smbh_msun`.
  Contains the CGS radial arrays, inner/outer radius, `gravitational_radius_cm`,
  the raw pAGN `torque_disk` object, and `torque_alpha`.
- `torque_profile_cache`: one `TorqueProfile` per
  `(m_smbh_msun, migrator_mass_msun, torque_alpha)`.

`get_disk_profiles()` builds on demand and caches; `get_torque_profile()`
wraps `migration_torque.compute_torque_profile()` and raises a descriptive
`TorqueProfileError` instead of silently falling back to anything.

### 2.5 Torque profiles (environment-level ingredient)

Authoritative owner: `src/disk_quantity/migration_torque.py`.

`compute_torque_profile()` evaluates, per radial cell in **SI**:

- `gamma_0_si` — the torque normalization;
- `gamma_type_i_si` — Type-I torque from pAGN corotation + Lindblad with the
  pAGN gap attenuation;
- `gamma_th_si` — raw thermal torque;
- `gamma_type_ii_si` — Type-II branch (inward, `-alpha m h² R² Ω²`);
- `gap_attenuation_factor`, `gap_parameter_dimensionless`.

SI is preserved throughout; CGS conversion (`torque_si_to_cgs()`) happens only
at the migration-time consumer (part 04). `classify_torque_traps()` resolves
stable/anti-trap zero crossings from a composed torque for diagnostics.

## 3. Key constants (owned here)

| Constant | Value | Owner |
|---|---|---|
| `DISK_INNER_RADIUS_RG` | 6.0 | `config.py` (physical inner radius) |
| `PAGN_ALPHA` | 0.1 | `config.py` |
| `DISK_ACCRETION_RATE_G_S_PER_MSUN` | 1.44e18 | `config.py` |
| `TOOMRE_Q_OUTER_BOUNDARY` | 1.0 | `pagn_disk_adapter.py` |
| NSC reference radius / mass / slope | 3.31 pc / 3.6e6 Msun / 0.321 | `config.py` |

## 4. Caveats

- **The Q=1 boundary is the physical outer edge.** Do not use the extrema of
  padded solver arrays as physical disk boundaries; the adapter's
  `outer_radius_cm` is authoritative. The legacy `disk_solve.py` solver is only
  a method-comparison path (`disk_plot.py`); the engine's production path uses
  pAGN only and must not silently fall back to it.
- **Unit discipline.** pAGN solves in SI; the adapter returns CGS arrays. All
  torque profile internals stay SI (`*_si`) until `torque_si_to_cgs()` is called
  exactly once at the migration consumer. Never mix unit systems.
- **Disk outer radius can be huge.** For `1e3 Msun` the Q=1 radius can be
  ~6.6e7 rg. Damping containment and migration domains are evaluated against
  this physical radius, which is one reason low-SMBH bins embed very few
  objects.
- **Torque profiles are migrator-mass dependent.** A profile computed for one
  mass must never be rescaled for another; `get_torque_profile()` computes per
  migrator mass and caches by key.
- **Inner radius is a concept with several names.** The GW ISCO radius, the disk
  inner radius, and the orbital-sampling minimum are numerically equal (6 rg)
  but are physically distinct concepts and must keep separate named constants.
- **No silent fallback.** If pAGN disk construction or torque-profile
  computation fails, the error must propagate as `TorqueProfileError` /
  `MigrationProfileError`; it must not be relabeled a trap or a zero rate.
