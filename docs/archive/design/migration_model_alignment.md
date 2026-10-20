# Migration model: the fixed contract and the evidence behind it

This records **why** the migration stage has the shape it does: one fixed
torque contract with no selectable prescriptions, what was measured to get
there, and which earlier conclusions in this file turned out to be wrong.

It describes the code as committed, not a proposal. Where a claim rests on a
judgement call rather than a measurement, it says so.

Contract identifier: `MIGRATION_CONTRACT = "jm17_grishin2024_continuous_kanagawa"`
(`workflows.config`).

---

## 1. The contract

```
K        = q^2 / (alpha h^5)                     Kanagawa gap parameter
f_gap    = 1 / (1 + 0.04 K)                      Kanagawa gap depth
Gamma_sd = -(3/4) alpha h^2 m r^2 Omega^2 min(1, B)     B = Sigma r^2 / m

Gamma = f_gap Gamma_grav + Gamma_th + (1 - f_gap) Gamma_sd
```

There is **no selectable migration model** and **no classical Type-II branch**.
`f_gap` hands weight continuously from the linear gravitational torque to the
gap-limited viscous torque as the gap deepens. Gap criteria (`crida`,
`kanagawa`) survive only as **validity labels**; none of them switches a branch.

Two terms are bounded by their own validity rather than extrapolated:

* `Gamma_th` is multiplied by `thermal_torque_regime_valid` and is therefore
  zero wherever the linear thermal-torque derivation does not hold. Opt back in
  with `MigrationParameters.extrapolate_thermal_torque=True`.
* The log-slopes feeding `Gamma_grav` are repaired by interpolation across
  cells flagged by `wkb_invalid` (§2.5).

### Why each piece is shaped that way

| Piece | Reason |
|---|---|
| No classical Type-II | `B ~ 1e-3` across the population: the migrator outweighs the local disk by 3–10 orders of magnitude, so the disk cannot carry it at the viscous rate (§2.1). |
| `(1 - f_gap) Gamma_sd` term | Without it a deep gap drove the torque to zero and migration stalled. A gap stops linear torques; it does not stop viscous transport. |
| `min(1, B)` interpolation | Reproduces the classical viscous rate for a disk-dominated secondary and the inertia-limited scaling below it, without importing a fitted transition calibrated outside this mass-ratio regime. |
| Thermal torque bounded | `R_H/H > 1` in **every** cell of **every** migration path here, so the linear derivation never holds; the extrapolated term dominated the balance and produced the apparent traps (§2.3). |
| Slope repair | The disk is thermally bistable at hydrogen ionisation; differentiating across that jump reverses the sign of the Lindblad torque (§2.5). |

References: Syer & Clarke (1995); Ivanov, Papaloizou & Polnarev (1999);
Duffell et al. (2014); Dürmann & Kley (2015); Kanagawa et al. (2015, 2018);
Masset (2017); Grishin, Gilbaum & Stone (2024); Derdzinski & Mayer (2023).

---

## 2. What the measurements say

Unless stated otherwise: `alpha = 0.1`, pAGN Sirko disk, `N = 800` radial
cells. The population figures come from a **613-object run** (20 repeats ×
`M• ∈ {1e4, 1e5, 1e6}`, `M_NSC = 5e4`, seed 2026).

### 2.1 Classical Type-II is not self-consistent here

`B = Sigma r^2 / m` is the local disk mass relative to the migrator. The
classical viscous law assumes `B >> 1`.

| `M•` | N | median `B` at `r_ini` | median `B` at `r_gas_fin` | fraction `B < 1` at end |
|---|---|---|---|---|
| 1e4 | 210 | 4.1e-2 | **3.8e-10** | **100 %** |
| 1e5 | 137 | 5.4e-2 | **7.2e-4** | **100 %** |
| 1e6 | 266 | 4.9e-2 | 4.9e-2 | 76 % |

Two consequences:

1. The law is invalid for essentially the whole evaluable population, by three
   to ten orders of magnitude.
2. `B` *falls* between `r_ini` and `r_gas_fin`: the gas stage carries objects
   from a marginal regime deeper into the one where the prescription fails.
   Some objects start with `B` up to 58; none end there.

### 2.2 The population the torque acts on

| `M•` | N | mass [M⊙] min/med/max | `r_ini` [r_g] min/med/max |
|---|---|---|---|
| 1e4 | 210 | 13.6 / 40.2 / 195 | 2.5e2 / 6.1e5 / 9.9e6 |
| 1e5 | 137 | 13.7 / 31.8 / 190 | 1.9e1 / 3.2e4 / 6.4e5 |
| 1e6 | 266 | 13.7 / 26.4 / 195 | 1.2e1 / 2.0e3 / 1.0e5 |

Mass and embedding radius are **uncorrelated** (Spearman `|rho| <= 0.21`): the
torque sees a population spread over four to five decades in radius with no
mass–radius structure, so a trap could only select by radius. Masses are ZAMS
throughout (no remnant relation, an intentional simplification per
`bug_fix_plan.md` §1.2), so the 140–195 M⊙ tail should in reality leave no
remnant at all.

### 2.3 Why the gap criterion is a label and not a switch

Fraction of each object's **own** migration path that a criterion calls
gap-open:

| `M•` | criterion | mean | median | objects 100 % flagged |
|---|---|---|---|---|
| 1e4 | kanagawa | 1.000 | 1.000 | **100 %** |
| 1e4 | crida | 0.864 | 0.876 | 12.7 % |
| 1e5 | **kanagawa** | **1.000** | **1.000** | **100 %** |
| 1e5 | **crida** | 0.230 | **0.000** | 9.8 % |

Under `kanagawa` every object is flagged everywhere. `K` reaches `1e2`–`6e4` at
these radii, implying the migrators have cleared 84–99.96 % of the local disk;
Kanagawa et al. (2015) caution the fit is unreliable much above `K ~ 1e4`, and
`kanagawa_extrapolated` flags that per cell. `crida` includes the viscous
refilling term — at `alpha = 0.1`, `h = 0.014`, `q = 1e-4` that term alone is
9.8, so a 10 M⊙ object cannot hold a gap open.

Had this been used as a **branch switch**, the criterion alone would have moved
the IMRI rate by 4× and the candidate mass by 15×. That sensitivity is why the
contract does not branch on it.

### 2.4 There are no resolved migration traps — earlier claim corrected

> **Correction.** An earlier revision of this document claimed a real,
> resolution-stable trap at `1.177e5 r_g` with a 0.055 dex basin. **That was
> wrong**, and the error was in the basin measure, not the disk.
>
> The basin was measured as the **outer catchment** — from the trap outward to
> the next torque zero. For a one- or two-cell spike sitting in an otherwise
> inward disk that catchment is arbitrarily large, so the measure could not
> discriminate and the artefact passed.

The basin is now measured across the **outward-torque region the root bounds**:
inward from a stable trap, outward from an anti-trap. That is invariant under
refinement. Re-measured with the corrected definition:

| `M•` | N=400 | N=800 | N=1600 | N=3200 |
|---|---|---|---|---|
| 1e4 | 0.0338 | 0.0184 | 0.0094 | 0.0049 |
| 1e5 | 0.0169 | 0.0121 | 0.0069 | 0.0036 |
| 1e6 | 0.0019 | — | — | 0.0010 |

Every basin **halves as the grid doubles** — the definitive artefact signature.
They sit at `r ≈ 1.70–1.75e5 r_g`, i.e. on the thermal instability of §2.5.

**Under the committed contract there are no resolved traps at any SMBH mass**,
and the total torque is inward everywhere on the production disk.

The `min_separation_cells` filter that previously guarded this has been
**removed**. A cell count is not a property of the disk: it called the same
feature resolved or unresolved depending only on how finely the disk happened
to be sampled, and so found *more* traps on coarser grids — at `N = 400` it
reported a trap at 1e4 that vanishes at every finer resolution.

### 2.5 The hydrogen-ionisation instability, and what it is not

The opacity table is **not** discontinuous. `pagn.opacities.get_combined_opacity`
is a smooth bicubic spline with `|dln kappa / dln T| ~ 25` — steep, continuous.

The 2.36-dex opacity step is downstream of a **0.63-dex jump in the disk
temperature itself** (9066 K → 2094 K between adjacent cells). That jump is
resolution-independent:

| N | 400 | 800 | 1600 | 3200 | 6400 |
|---|---|---|---|---|---|
| cell width [dex] | 0.0119 | 0.0060 | 0.0030 | 0.0015 | 0.00075 |
| max `dlogT` jump | 0.632 | 0.630 | 0.636 | 0.638 | 0.638 |

A 16-fold refinement does not resolve it, so it is a genuine bistable thermal
equilibrium at hydrogen ionisation — real disk physics, not a table artefact
and not under-resolution. Checkpoint B's WARN about it is correct and should
stay.

The defect was applying linear torque theory *across* it. There
`|dlnT/dlnR|` reaches **133** against a typical **0.50**, and the Lindblad
coefficient is linear in that slope, so the torque was not merely inaccurate
but **sign-reversed** — the origin of the apparent traps.

`wkb_invalid_mask` flags the two cells bracketing the jump and
`repair_slopes_across` rebuilds their log-slopes by interpolation from the
smooth background either side. The criterion is the **jump in dex**, not the
slope: a slope threshold is resolution-dependent in exactly the way that makes
such criteria unusable (a fixed jump divided by a shrinking cell inflates the
slope without changing the physics). Measured flagged-cell count: **2 at every
resolution and every SMBH mass**.

`max_step_dex = 0.15` is a **judgement call**, placed between the smooth
background (~0.01 dex/cell) and the instability (0.63 dex).

### 2.6 A redshift-dependent SMBH ceiling

All 266 objects at `M• = 1e6` returned `band_entry_inside_isco` — not one
evaluable event. The observer-frame 1 mHz band entry falls inside the ISCO, so
`classify_unsupported_band_entry` correctly returns `None` rather than a zero.

| `z` | max evaluable `M•` [M⊙] |
|---|---|
| 0 | 4.4e6 |
| 3 | 1.1e6 |
| 10 | 4.0e5 |
| **15** (default) | **2.75e5** |

This is a **domain limit, not a physical zero**. It rises ~16× between z=15 and
z=0, so any redshift dependence extracted from a merger tree will be entangled
with it unless the ceiling is reported alongside. `docs/physics_model.md` §5
still lists a static `1e3–1e6` domain with no redshift term; that remains to be
reconciled.

---

## 3. Superseded decisions

### 3.1 The crossing-time prescription

`bug_fix_plan.md` §3.1 says "keep this crossing-time prescription" (the
inherited `t_cross = 2H / (v_i sin i)`) and §9.6 lists changing it as
stop-and-ask. **The user directed the change.** The implemented form uses the
true vertical velocity at the node:

```
v_z     = v_t sin i = sqrt(G M p) sin i / r      p = a (1 - e^2)
t_cross = 2 H / v_z
```

At a node the body lies on the line of nodes, which is *in* the disk plane, so
the radial velocity contributes nothing vertical; only the transverse component
tilts out. Verified exact against an independent 3-D state-vector calculation
at both nodes (relative error `<= 2.5e-16`).

| node | median | mean | 90th | 99th | max |
|---|---|---|---|---|---|
| 1 (sub-Keplerian) | 1.310 | 1.435 | 1.942 | 3.383 | 23.6 |
| 2 (super-Keplerian) | 1.073 | 1.098 | 1.227 | 1.347 | 1.414 |

The old form was *always* an overestimate of `v_z`, so the correction has a
definite sign. It is also the other half of a cancellation: Task A fixed the
`rho = Sigma / H` factor-of-2 alone, which left the compensation broken; this
closes it. Both are now correct independently.

### 3.2 The hard Type-II branch

`bug_fix_plan.md` §1.4 and §2.3 lock a hard branch at `K = q^2/(alpha h^5) > 11`
with the Vaccaro Eq. 15 torque. That is superseded by §1 here, on the evidence
of §2.1: the branch it switches into is invalid for ~100 % of the population.
`docs/reference/physical_model.md` already described a continuous model as the
intended default, so this brings the code into line with that description
rather than inventing a fourth one.

### 3.3 Checkpoints D and F

Both asserted properties the model does not owe — that the total torque must
change sign, and that stable traps must exist for at least one SMBH mass. Both
are **results** of the contract and are now reported as diagnostics. F gained a
check that does discriminate: **trap count must not grow when the grid is
coarsened**, which would have caught the §2.4 artefact directly.

---

## 4. Known-open items

* **`n_det` is one detection across two repeats on a single host.** First
  non-zero, not a result.
* **`docs/physics_model.md`** still carries pre-change trap tables, the removed
  attenuation factors, a "Linear model" label on an Einstein–de Sitter redshift
  relation, and the static SMBH domain of §2.6.
* **`bug_fix_plan.md` is untracked by git.** Either commit it or fold its locked
  decisions into `physics_model.md`; §3.1 and §3.2 here supersede parts of it
  and that must not live only in this file.
* **Two cosmologies remain** in the same call path: an EdS `t(z)` for redshift
  evolution and flat ΛCDM for luminosity distance. Measured cost in the default
  configuration is ~1.7 % in SNR — small, but it is a real inconsistency.
* **`extrapolate_thermal_torque`** exists so the pre-change behaviour is
  reproducible. It should not be used for production numbers.

## 5. Reproducing the measurements

```bash
# population run behind the section 2 tables
python scripts/main.py 20 output/_pop 8 \
    --smbh-masses 1e4,1e5,1e6 --nsc-masses 5e4 --base-seed 2026

# population against the fixed contract: validity occupancy and trap structure
python scripts/analyse_population_torque.py output/_pop

# contract-level invariants
python scripts/validate_migration_contract.py

# full checkpoint sweep (D and F are diagnostics; B WARNs on section 2.5)
python scripts/run_science_checkpoints.py
```

The `B`, refinement and jump tables in §2 were produced ad hoc against
`compute_torque_profile`. They should be promoted into
`scripts/run_science_checkpoints.py` so they stop being one-off measurements.
