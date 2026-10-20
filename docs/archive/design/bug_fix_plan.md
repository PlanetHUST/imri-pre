# IMRI Single-AGN Physics Consistency Repair Plan

## Purpose

This document is the implementation brief for Codex to repair the current IMRI
pipeline **without expanding the scientific scope**.

The target is a first-order, internally consistent **single-AGN IMRI simulation**:

```text
sample NSC objects
    -> select disk-crossing orbits
    -> gas-drag damping / embedding
    -> embedded radial migration
    -> gas-vs-GW handoff
    -> circular Peters inspiral
    -> observed LISA-band entry
    -> physical LISA SNR / detectability
```

The goal is **not** to make the project maximally realistic. The goal is to make
the physics that is already claimed by the model agree with its chosen
equations and with itself across module boundaries.

Implement in small stages, preserve deterministic paired seeds, and add
regression/physics tests after each stage.

---

# 1. Locked scientific decisions

These are user decisions. **Do not reinterpret or replace them.**

## 1.1 Single-AGN scope

This task models one AGN episode around one SMBH.

- No SMBH-SMBH/cosmic-merger event is active.
- The central SMBH mass remains fixed for the event:
  \[
  M_\bullet(t) = M_{\bullet,0}.
  \]
- Do not apply the old `M -> factor*M`, `r -> factor*r` cosmic-merger transform.
- Do not add recoil, binary-SMBH evolution, halo-merger evolution, or NSC
  rebuilding.

The active chain ends at post-gas GW evolution and LISA detectability.

## 1.2 Population simplifications are intentional

Do **not** change these in this task:

- stars below \(1\,M_\odot\) are omitted by the current retained-population model;
- use \(M_{\rm ZAMS}=M_{\rm BH}\) as the current first-order remnant
  approximation;
- embedded-star radial migration may be treated with the same migration model
  used for BHs;
- natal kicks are omitted;
- no detailed remnant prescription / PISN / NS branch is added;
- no relaxation-driven replenishment is added to the IMRI workflow;
- do not change the initial orbital sampling distribution in this patch.

These are model assumptions, not correctness bugs for this task.

## 1.3 Disk outer boundary is the first outward \(Q=1\) boundary

This is an explicit user choice.

Keep the current pAGN/Sirko operational disk boundary at the first outward
Toomre crossing:

\[
Q=1.
\]

The project intentionally does **not** use the full marginal-\(Q=1\) SG03 outer
branch for this first analysis.

Therefore:

- preserve the current `pagn_disk_adapter.py` search for the first outward
  \(Q=1\);
- preserve the resulting `outer_radius_cm` as the physical damping/migration
  outer boundary;
- do not replace it with the
  \(0.1\,{\rm pc}(M_\bullet/10^6M_\odot)^{1/2}\) boundary;
- do not extend a torque profile outside the already solved Q=1-truncated disk.

### Important implementation consequence

`compute_torque_profile(..., max_radius_rg=...)` must **never re-solve or extend**
the pAGN disk beyond its existing Q=1 outer boundary.

If a requested `max_radius_rg` exceeds the existing disk domain, clamp to the
existing domain. It may crop a profile inward; it must not expand it outward.

This also applies to diagnostic scripts such as `trap_plot.py`.

## 1.4 Type-II prescription is the Vaccaro et al. Eq. 15 torque

Use the supplied migration reference as the authority for the hard Type-II
branch.

Gap criterion:

\[
K = \frac{q^2}{\alpha h^5} > 11,
\qquad
q=\frac{m}{M_\bullet},
\qquad
h=\frac{H}{R}.
\]

When the criterion is true, use Eq. 15:

\[
\Gamma_{\rm II}
=
-\alpha\,m\,H^2\Omega^2
=
-\alpha\,m\,h^2R^2\Omega^2.
\]

The minus sign is the code convention for inward migration.

**Do not replace Eq. 15 by a torque reduced by a factor of two.**

Because the project's torque integrator uses the exact circular-orbit relation

\[
\frac{dL}{dR}=\frac12 mR\Omega,
\]

Eq. 15 implies the local logarithmic radial migration time

\[
t_{\ln R,{\rm II}}
=
\frac{1}{2\alpha h^2\Omega}.
\]

That factor \(1/2\) is intentional in this project because the user selected
Eq. 15 as the authoritative torque. Do not "correct" it to Eq. 14 in the
semi-analytic comparison branch.

## 1.5 Circular GW model remains the validated production model

Keep:

\[
e_{\rm GW}=0.
\]

Do not add coupled eccentric Peters evolution in this patch.

`gw_frequency_radius.py` is considered valid for the current circular
production channel. Add tests, but do not redesign it.

## 1.6 LISA band and detector assumptions are locked

Do not change:

- observed lower entry frequency: `1e-3 Hz`;
- observed upper detector frequency: `1e-1 Hz`;
- SNR threshold unless supplied by config;
- mission-duration handling;
- circular, nonspinning, quadrupole, inspiral-only detector model;
- source/detector mass-frame convention.

The existing `lisa_snr.py` is not a target for waveform-model expansion.

---

# 2. Explicit hold points / out-of-scope physics

The following are scientifically important but **must not be changed silently**.

## 2.1 AGN accretion-rate normalization

The supplied migration paper uses \(0.1\dot M_{\rm Edd}\), while the current
engine explicitly passes

```python
DISK_ACCRETION_RATE_G_S_PER_MSUN * m_smbh_msun
```

to pAGN.

Changing this changes the entire disk structure and therefore damping,
migration, traps, and rates.

**For this patch, leave the current accretion-rate normalization unchanged.**

If Codex discovers that a requested test cannot be made meaningful without
changing the accretion normalization, stop and ask the user.

Do not silently switch to pAGN's `eddington_ratio=0.1`.

## 2.2 Disk alpha

Keep the current configured/project value (`PAGN_ALPHA`) unchanged.

Do not change \(\alpha\) to reproduce a particular figure unless explicitly
asked.

## 2.3 Partial-gap/Kanagawa interpolation

The baseline model for this patch is a **hard branch**:

```text
K <= 11 : JM17 Type I + thermal torque
K > 11  : Type-II Eq. 15
```

Do not add the Appendix-B reduced-density/partial-gap model.

Remove existing partial-gap factors from the baseline torque if they are
currently mixed into Type I/thermal torque.

## 2.4 More realistic damping

Do not add in this patch:

- sound speed to the drag radius;
- Hill-radius caps;
- wake/dynamical-friction improvements;
- a new partially embedded damping regime;
- KL oscillations;
- tidal disruption;
- node-specific scale heights if doing so changes the inherited damping
  prescription beyond the equation fixes below.

Those are later model extensions.

---

# 3. Authoritative source equations

Use the supplied papers, not memory, when changing formulas.

## 3.1 Damping authority: Fan et al. 2024

The supplied star-disk paper is the authority for the current damping equations.

Important relations:

\[
\rho_d=\frac{\Sigma_d}{2H}.
\]

At the two crossings,

\[
r_{1,2}
=
\frac{a(1-e^2)}{1\mp e\cos\omega}.
\]

The code already uses the paper's crossing-time approximation

\[
t_{{\rm cross},i}
=
\frac{2H}{v_i\sin i}.
\]

**Keep this crossing-time prescription.**

The paper's Eqs. 6--8 define the sign pairing for \(da\), \(de\), and \(di\).
The current/reference Python implementation copied two wrong signs into the
second (`1 + e cos(omega)`) branch. Correct the code to follow the published
equations.

For the code branch whose denominator is

\[
1+e\cos\omega,
\]

use:

\[
1+2e\cos\omega+e^2
-
(1+e\cos\omega)^{3/2}\cos i
\]

in the `da2` bracket, and use

\[
e+e\cos^2\omega+2\cos\omega
\]

inside the corresponding `de2` numerator.

Do not change the already consistent `1-e cos(omega)` branch.

## 3.2 Migration authority: Vaccaro et al. 2026 G24-like prescription

For gap-closed cells, use:

\[
\Gamma
=
\Gamma_{\rm Type\,I}
+
\Gamma_{\rm th}.
\]

### JM17 Type-I

Use the existing Jiménez--Masset coefficient structure without gap attenuation:

\[
\Gamma_{\rm Type\,I}
=
\left(
C_L+C_C
\right)\Gamma_0,
\]

with the current sign convention for the stored log slopes.

Do **not** multiply the corotation term by `exp(-K/20)`.

Do **not** divide Type I by `1 + K/25`.

### Thermal diffusivity used in migration torque

The selected migration reference uses electron-scattering opacity

\[
\kappa_{\rm es}=0.34\ {\rm cm^2\,g^{-1}}
\]

for the thermal diffusivity entering the torque calculation.

Do not use the radial pAGN opacity array for this migration-\(\chi\) formula.

The pAGN opacity profile remains valid for constructing the pAGN disk itself;
this task only changes the opacity used in the migration torque helper.

### Thermal torque

Implement the reference equation directly:

\[
\Gamma_{\rm th}
=
1.61\frac{\gamma-1}{\gamma}
\frac{x_c}{\lambda}
\left[
\frac{L}{L_c}
\frac{4m_c}{m+4m_c}
-
\frac{2m_c}{m+2m_c}
\right]
\frac{\Gamma_0}{h}
\left(
1-e^{-\lambda\tau_V/H}
\right),
\]

with

\[
m_c=\frac{c_s\chi}{G},
\]

\[
x_c
=
-\frac{H^2}{3\gamma R}
\frac{d\ln P}{d\ln R},
\]

\[
\lambda
=
\sqrt{\frac{2\chi}{3\gamma\Omega}},
\]

\[
L=L_{\rm Edd},
\qquad
L_c=\frac{4\pi Gm\rho\chi}{\gamma}.
\]

Use `disk.tauV` for the vertical optical depth and the physical pAGN scale
height `disk.h` for \(H\).

The attenuation exponent must be dimensionless:

```python
thermal_length_m * optical_depth / scale_height_m
```

### Important thermal changes relative to current code

- remove `abs(dlog_pressure_dlog_radius)` and preserve the formula sign;
- replace the current piecewise thermal-saturation expression with the explicit
  \(m_c\) equation above;
- add optical-depth suppression
  `1 - exp(-lambda * tauV / H)`;
- remove `(1 - gap_attenuation_factor)` from downstream composition;
- do not use the gap parameter to attenuate thermal torque in the baseline model.

### Type II

Use:

\[
K=q^2/(\alpha h^5)>11
\]

and Eq. 15 torque exactly.

---

# 4. Implementation tasks by module

## Task A -- Correct the pAGN -> damping data interface

### Files

- `src/workflows/models.py`
- `src/workflows/engine.py`
- `src/workflows/single_work.py`
- `src/workflows/workers.py`
- `src/physical_process/damp.py`

### Required changes

1. Add a required `density_g_cm3` array to `DiskProfiles`.
2. In `IMRIDynEngine.get_disk_profiles()`, copy:
   ```python
   pagn_disk.density_g_cm3
   ```
   into the cached profile.
3. Pass that density profile through:
   ```text
   single_work
       -> run_damping_tasks_parallel
       -> init_damping_worker
       -> DampingCalculator
   ```
4. Add a density lookup helper analogous to `sigma_gas`.
5. In `DiskProperties.from_agn()`, obtain each active node's midplane density
   directly from the pAGN density profile.
6. **Do not reconstruct**
   ```python
   density = surface_density / scale_height
   ```
   or
   ```python
   density = surface_density / (2 * scale_height)
   ```
   when the authoritative pAGN density is already available.
7. Preserve surface density separately because it remains useful for
   diagnostics and other physics.

### Boundaries

- keep the current average-radius scale-height/crossing-time prescription;
- do not refactor damping physics beyond the direct-density interface;
- do not change the Q=1 disk boundary.

### Acceptance checks

For every valid disk grid sample:

\[
\Sigma \approx 2\rho H
\]

should hold to expected pAGN numerical precision, where
\(H=(H/R)R\).

A direct lookup of `density_g_cm3` at a grid point must return the same value
passed from `PagnDiskProfile`.

---

## Task B -- Correct the two second-node damping signs

### File

- `src/physical_process/damp.py`

### Required changes

In `DampingCalculator.compute_ode_coefficients()`:

For the second branch using `1 + e*cos_omega`:

```python
# current wrong structure
1 - 2*e*cosw + e**2 - (1 + e*cosw)**1.5 * cosi
```

becomes

```python
1 + 2*e*cosw + e**2 - (1 + e*cosw)**1.5 * cosi
```

and the second-node eccentricity term:

```python
e + e*cosw**2 - 2*cosw
```

becomes

```python
e + e*cosw**2 + 2*cosw
```

for the `1 + e*cosw` denominator branch.

### Do not change

- `da1`;
- `de1`;
- `di1`, `di2`;
- relative-velocity equations;
- crossing-time equation.

### Tests

Add a direct equation test using fixed
`a, e, i, cos_omega, tau1, tau2, tcross1, tcross2`.

Compute the expected six coefficients from the paper equations inside the test,
independently of the implementation, and compare tightly.

Also test `cos_omega=0`: the sign-dependent algebra of the two branches should
reduce consistently when drag/crossing times are set equal.

---

## Task C -- Enforce the Q=1 disk edge everywhere

### Files

- `src/disk_quantity/pagn_disk_adapter.py`
- `src/disk_quantity/migration_torque.py`
- `scripts/trap_plot.py`
- any diagnostic that requests a torque radius larger than the cached disk

### Required changes

Preserve the current first-outward-Q=1 boundary algorithm.

Change `compute_torque_profile()` so `max_radius_rg` is a **crop only**.

It must not do this when the requested max exceeds the current solved disk:

```python
disk.Rmax = larger_radius
disk.solve_disk(...)
```

A torque calculation must never mutate/re-solve the cached physical disk beyond
the Q=1 edge.

Preferred behavior:

```text
existing disk domain = authoritative
requested max < existing max -> crop
requested max >= existing max -> use full existing disk
```

### Tests

- build/mock a disk with a known outer radius;
- request a torque profile with a larger `max_radius_rg`;
- verify returned maximum radius is unchanged;
- verify the disk object's solved radius array was not extended.

---

## Task D -- Make the KII gap parameter authoritative and consistent

### Files

- `src/disk_quantity/migration_torque.py`
- `src/physical_process/disk_migration.py`
- `scripts/trap_plot.py`
- any other diagnostic implementing K directly

### Required changes

There is already a correct helper in `migration_torque.py`:

```python
comp_gap_parameter_dimensionless(mass_ratio, aspect_ratio, alpha)
```

which uses \(h^{-5}\).

Reuse that helper in `disk_migration.gap_opening_mask()` for `condition=="kii"`
rather than reimplementing a different expression.

Required formula:

```python
K = mass_ratio**2 / (alpha * aspect_ratio**5)
type_ii_mask = valid & (K > 11.0)
```

Keep:

```python
TYPE_II_GAP_PARAMETER_THRESHOLD = 11.0
```

The `crida` option may remain as an alternative configuration path; do not
change its formula in this patch.

### Tests

Use simple exact cases:

```text
q=1e-3, alpha=0.1, h=0.1
K = q^2/(alpha*h^5)
```

and test just below/above threshold.

Also assert the value from `gap_opening_mask()` agrees with
`comp_gap_parameter_dimensionless()`.

---

## Task E -- Rebuild the baseline G24 migration torque literally

### Files

- `src/disk_quantity/migration_torque.py`
- `src/physical_process/disk_migration.py`

### E1. Separate disk opacity from migration thermal opacity

Do not modify the pAGN disk solver's opacity physics.

For torque thermal diffusivity, introduce an explicit migration opacity helper
or parameter based on electron scattering.

Use the pAGN electron-scattering helper already imported by this module, with
the existing hydrogen fraction, and verify it corresponds to the intended
\(0.34\,{\rm cm^2\,g^{-1}}\) for `X=0.7`.

`thermal_diffusivity_m2_s()` used by JM17/G24 migration must use this
electron-scattering opacity rather than `disk.kappa`.

### E2. JM17 Type-I

Change:

```python
gamma_i_si = (
    (corot * gap_attenuation_factor + lind)
    / type_i_gap_denominator
    * gamma_0_si
)
```

to the baseline no-partial-gap form:

```python
gamma_i_si = (corot + lind) * gamma_0_si
```

Keep the current slope sign convention consistent with existing
`comp_corot()` and `comp_lind()`.

### E3. Thermal torque

Rewrite `comp_gamma_th_coeff()` to implement the reference equation directly.

Do not keep the old piecewise `thermal_saturation_parameter` branch merely for
backward compatibility.

Required internal quantities:

```text
chi
m_c = c_s * chi / G
x_c = -(H**2)/(3*gamma*R) * dlnP_dlnR
lambda = sqrt(2*chi/(3*gamma*Omega))
L = L_Edd(m)
L_c = 4*pi*G*m*rho*chi/gamma
thin_suppression = 1 - exp(-lambda*tauV/H)
```

Dimensionless bracket:

```text
(L/L_c) * 4*m_c/(m + 4*m_c)
    - 2*m_c/(m + 2*m_c)
```

Then combine with:

```text
1.61*(gamma-1)/gamma
* (x_c/lambda)
* bracket
* (1/h_aspect)
* thin_suppression
```

before multiplying by `gamma_0_si`.

Guard divisions explicitly. Nonfinite physical profile values should surface as
a clear torque-profile error, not silently become zero unless the helper's
existing `safe_divide` convention is scientifically justified for that exact
limit.

### E4. Simplify `TorqueProfile`

`gap_parameter_dimensionless` remains useful.

The following fields are baseline-model leftovers and should be removed if no
remaining consumer requires them:

- `gap_attenuation_factor`
- `type_i_gap_denominator`

If removing them creates unnecessary output compatibility churn, they may
temporarily remain as diagnostics, but **must not affect torque composition**.

Update docstrings accordingly.

### E5. Compose torque

In `select_migration_torque_si()`:

```python
type_i_thermal_torque_si = (
    torque_profile.gamma_type_i_si
    + torque_profile.gamma_th_si
)
```

No K attenuation.

Then:

```python
selected = np.where(
    type_ii_mask,
    torque_profile.gamma_type_ii_si,
    type_i_thermal_torque_si,
)
```

### E6. Type-II Eq. 15

Keep/verify:

```python
gamma_type_ii_si = (
    -alpha
    * migrator_mass_kg
    * aspect_ratio**2
    * radius_m**2
    * orbital_frequency_rad_s**2
)
```

This is the selected Eq. 15 implementation.

### Tests

Add synthetic-array tests for:

1. `K <= 11`: selected torque equals `gamma_type_i + gamma_th`.
2. `K > 11`: selected torque equals `gamma_type_ii`.
3. Type-II torque equals Eq. 15 numerically.
4. `x_c` preserves pressure-gradient sign.
5. thermal suppression:
   - `tauV -> 0` gives thermal torque -> 0;
   - large `tauV` gives suppression factor -> 1.
6. finite positive `m_c`.
7. no hidden use of `disk.kappa` in migration `chi`.

---

## Task F -- Preserve Eq. 15 consistency in the semi-analytic branch

### File

- `src/physical_process/disk_migration.py`

The current semi-analytic Type-II local logarithmic time is:

```python
1 / (2 * alpha * h**2 * Omega)
```

Keep it.

This is consistent with integrating the user-selected Eq. 15 torque using

\[
dL/dR = \frac12mR\Omega.
\]

Do not change it to:

```python
1 / (alpha * h**2 * Omega)
```

in this task.

Add a test that computes a constant-\(h\) Eq. 15 torque and confirms the
torque-integrated local logarithmic rate is consistent with the semi-analytic
factor \(1/(2\alpha h^2\Omega)\).

---

## Task G -- Fix trap versus direct-GW routing

### File

- `src/workflows/event_rate.py`

### Existing problem

`solve_gas_final_radius_cm()` currently terminates as
`"migration_trap_bound"` when total gas migration time is infinite before
giving a finite direct-GW path the chance to win.

A gas trap means gas delivery is blocked; it does **not** imply GW evolution is
blocked.

### Required ordering

Compute:

```text
direct GW time from current radius to entry
gas-assisted path time
```

Then apply:

1. if direct GW is finite and is no slower than the gas-assisted route, return
   `gw_dominates_at_start`;
2. only after that, if gas migration is nonfinite and direct GW cannot provide
   a valid route, return `migration_trap_bound`;
3. otherwise proceed with the existing sampled/refined gas-vs-GW crossover.

For an infinite gas time, interpret the gas-assisted route as infinity.

### Tests

Cover at least:

- finite gas, gas faster;
- finite gas, direct GW faster;
- infinite gas, finite GW -> **GW dominates**, not trap terminal;
- infinite gas, infinite GW -> trap/unevaluable according to existing status
  semantics;
- duty-cycle-limited path.

Do not change the status vocabulary unnecessarily.

---

## Task H -- Remove the cosmic-merger physics interface from the active workflow

### Files likely affected

- `src/workflows/detect_rate.py`
- `src/workflows/event_work.py`
- `src/workflows/config.py`
- `src/workflows/workflow.py`
- `src/workflows/metadata.py`
- `scripts/redshift_scan.py`
- CLI/config construction code
- tests using cosmic-merge parameters
- optionally `src/workflows/cosmic_merge.py`

### Required active behavior

`classify_detectability()` should evolve one fixed primary mass from the
post-gas state to observed LISA entry.

There should be no:

- `CosmicMergeModel` construction;
- time-to-cosmic-merge branch;
- SMBH mass jump;
- orbital-radius jump;
- second post-merge GW segment.

### Configuration

Remove these as active required parameters:

```text
cosmic_merge_time_myr
cosmic_merge_smbh_mass_factor
cosmic_merge_radius_factor
```

Update:

- `IMRIDynConfig`;
- CLI;
- `redshift_scan.DEFAULT_CONFIG_TEMPLATE`;
- `CONFIG_TEMPLATE_FIELDS`;
- metadata builders;
- workflow logging.

For loading historical metadata/config files, it is acceptable to ignore these
legacy keys with a warning rather than reject the file.

### Output compatibility

Preferred first patch:

- keep `cosmic_merge_applied` only if output consumers/tests require the column,
  but it must always be `False`;
- keep `final_smbh_mass_msun` if useful, but it must equal the original SMBH
  mass.

Do not retain user-facing cosmic-merger knobs merely to populate those legacy
columns.

Delete `cosmic_merge.py` only if no remaining import/test needs it. Otherwise
leave it unused and clearly deprecated; do not spend time on unrelated cleanup.

---

## Task I -- Replace the matter-only redshift clock with one flat-LambdaCDM cosmology

### Goal

The same cosmological parameters must determine:

\[
t(z), \quad z(t), \quad D_L(z).
\]

Use:

```text
detector_hubble_constant_km_s_mpc
detector_omega_m
```

as the authoritative pair.

Do not maintain a separate physical `hubble_constant_per_myr` trajectory.

### Preferred architecture

Create:

```text
src/workflows/cosmology.py
```

with a small immutable flat-\(\Lambda\)CDM helper used by both
`detect_rate.py` and `lisa_snr.py`.

Suggested API:

```python
@dataclass(frozen=True)
class FlatLambdaCDM:
    hubble_constant_km_s_mpc: float
    omega_m: float

    def cosmic_age_myr(self, redshift: float) -> float: ...
    def redshift_at_cosmic_age_myr(self, age_myr: float) -> float: ...
    def evolve_redshift(self, initial_redshift: float, elapsed_source_myr: float) -> float: ...
    def remaining_time_to_z0_myr(self, redshift: float) -> float: ...
    def luminosity_distance_mpc(self, redshift: float) -> float: ...
```

For a flat matter+Lambda cosmology, the cosmic-age relation may be analytic:

\[
t(z)=
\frac{2}{3H_0\sqrt{\Omega_\Lambda}}
\sinh^{-1}
\left[
\sqrt{\frac{\Omega_\Lambda}{\Omega_m}}
(1+z)^{-3/2}
\right],
\]

with \(\Omega_\Lambda=1-\Omega_m\).

Implement the inverse analytically or robustly numerically.

Use the same luminosity-distance integral currently implemented in
`lisa_snr.py`, but route it through this shared cosmology helper.

### Config cleanup

Remove `hubble_constant_per_myr` as an active parameter from new runs.

Historical configs may contain it; ignore/deprecate it on load rather than
using it.

### AGN-phase boundary

For this first single-AGN analysis, keep the AGN episode at the configured
host redshift when constructing the gas/damping disk and the initial intrinsic
LISA-entry geometry.

Do **not** implement a fully moving gas-migration target in this patch.

Post-gas GW evolution uses the proper flat-\(\Lambda\)CDM clock.

If changing the gas-stage redshift treatment becomes necessary to make a test
pass, stop and ask the user rather than introducing a moving-boundary gas
solver.

### `max_gw_time_myr`

Make this parameter active.

For post-gas evolution, the allowed elapsed time must be limited by at least:

```text
max_gw_time_myr
time remaining until z=0
time to ISCO
```

The existing `solve_time_to_observed_lisa_entry()` already limits by ISCO;
preserve that behavior and add the configured max-GW budget to the caller or
solver cleanly.

### Tests

- round trip:
  \[
  z \rightarrow t(z) \rightarrow z
  \]
  for `z = 0.1, 1, 5, 10, 15`;
- `evolve_redshift(z, 0) == z`;
- redshift decreases monotonically with positive elapsed source time;
- remaining time to `z=0` is positive;
- `D_L(z)` agrees with the existing validated LISA benchmarks;
- detectability and LISA SNR use exactly the same H0/Omega_m.

---

## Task J -- Keep the circular GW helper and add invariants

### File

- `src/physical_process/gw_frequency_radius.py`

Do not change production equations for `e=0`.

Add/retain tests for:

1. radius-frequency round trip:
   ```text
   r -> f_gw(r) -> r
   ```
2. circular Peters scaling:
   \[
   t_{\rm merge}(ka)=k^4 t_{\rm merge}(a)
   \]
3. source/observer round trip:
   \[
   f_{\rm src}=(1+z)f_{\rm obs}
   \]
4. LISA entry:
   the returned formal entry radius must reproduce configured observed
   `f_min` after redshifting;
5. ISCO:
   `band_entry_inside_isco` iff the formal entry radius is at/below
   Schwarzschild ISCO.

### Eccentric path boundary

Do not extend or certify the nonzero-eccentricity Peters path in this task.

If tests are added for it, label them helper-level only; the supported
end-to-end production configuration remains `gw_eccentricity=0`.

---

# 5. Diagnostic and script updates

## `scripts/trap_plot.py`

Update it to use the same K formula:

\[
K=q^2/(\alpha h^5).
\]

Its torque profile must respect the Q=1-truncated disk.

Do not allow `TORQUE_MAX_RADIUS_RG` to expand the disk beyond Q=1.

Also avoid silently using a different LISA lower frequency than the production
configuration. Prefer importing the production default or passing it as a CLI
argument.

## `scripts/redshift_scan.py`

Remove active cosmic-merge and matter-only-Hubble fields from the new template.

Preserve paired seeds exactly:

```text
redshift and disk model must remain excluded from derive_task_seed()
```

Do not alter the paired-comparison design.

Historical metadata may contain removed fields; ignore/deprecate them cleanly.

---

# 6. Tests and validation harness

Add one focused standalone harness consistent with project convention, e.g.

```text
scripts/imri_physics_consistency_test.py
```

Do not introduce a new testing framework unless already available and useful.

The harness should be deterministic and should separate cheap algebraic tests
from any pAGN-dependent integration test.

## Required groups

### A. pAGN/damping interface

- pAGN density is forwarded without reconstructing it;
- \(\Sigma=2\rho H\) diagnostic consistency;
- paper Eq. 6--8 coefficient test for both nodes.

### B. migration formulas

- KII uses \(h^{-5}\);
- threshold is exactly 11;
- JM17 Type I contains no K attenuation;
- thermal `x_c` sign;
- electron-scattering opacity is used in migration \(\chi\);
- optical-depth suppression limits;
- Type-II torque Eq. 15.

### C. Type-II integration consistency

For a controlled profile, verify Eq. 15 combined with the exact angular
momentum integrator gives the same local logarithmic timescale as

\[
1/(2\alpha h^2\Omega).
\]

### D. event routing

Test finite/infinite gas/GW combinations, especially:

```text
gas = inf, GW finite -> gw_dominates_at_start
```

### E. cosmology

Round trips and LISA distance benchmarks.

### F. GW helper

Circular invariants above.

### G. no cosmic merger

An end-to-end detectability case must show:

```text
final_smbh_mass_msun == initial primary mass
cosmic_merge_applied == False  # if legacy field retained
```

with no cosmic-merger configuration needed.

---

# 7. Regression strategy

Before changing equations, capture one deterministic baseline for a small grid
using the existing seed policy.

After each task, rerun the same seeds.

Track at least:

```text
N_total_sampled
N_disk_crossing
N_embedded_star
N_embedded_bh
N_migration_finite
N_trap_bound
N_gw_dominates_at_start
N_gw_dominates_after_gas
N_imri_candidate
N_lisa_detectable
```

Also compare distributions of:

```text
damping_time_myr
r_gas_fin
migration_time_to_entry_myr
gw_time_after_gas_myr
detection_redshift
lisa_snr
```

The objective is not to preserve old numbers after fixing wrong formulas.
The objective is to make every change attributable to a specific correction.

Use paired seeds so changes in counts are physics changes, not resampling noise.

---

# 8. Expected direction of changes (diagnostic only)

Do not write tests that require these directions, but use them as sanity checks.

- Direct pAGN density should remove the previous factor-of-two density excess,
  so damping will generally become slower than the old implementation.
- Correcting node-2 damping signs may change `a`/`e` evolution in a
  state-dependent direction; do not assume monotonic rate impact.
- Changing \(K\) from \(h^{-2}\) to \(h^{-5}\) can substantially enlarge the
  gap-open domain in thin disks.
- Removing artificial K attenuation from gap-closed JM17/thermal torque can
  move or create/remove torque zeroes.
- Adding optical-depth suppression should reduce thermal torque in optically
  thin regions.
- Correct trap/GW ordering can recover some sources previously terminated as
  trap-bound.
- Correct LambdaCDM time evolution will change long-delay detection redshifts
  and therefore luminosity distance/SNR.

---

# 9. Stop-and-ask conditions

Codex must stop and ask the user before doing any of the following:

1. changing the disk accretion-rate normalization / Eddington ratio;
2. changing `PAGN_ALPHA`;
3. replacing the user-selected Q=1 disk outer boundary;
4. replacing Type-II Eq. 15 or changing threshold 11;
5. adding partial-gap/Kanagawa interpolation;
6. changing the Fan et al. crossing-time model beyond the two published sign
   corrections and density-interface fix;
7. adding a moving cosmological LISA boundary during gas migration;
8. changing population/remnant assumptions;
9. changing LISA frequency bounds, SNR threshold defaults, waveform family, or
   mission model;
10. adding nonzero-eccentricity production GW evolution;
11. changing intrinsic event-rate denominator or Monte-Carlo weights;
12. changing paired-seed semantics.

If an implementation detail makes one of these appear necessary, do not infer a
new science model. Ask.

---

# 10. File-level checklist

Likely files to modify:

```text
src/workflows/models.py
src/workflows/engine.py
src/workflows/single_work.py
src/workflows/workers.py
src/physical_process/damp.py

src/disk_quantity/migration_torque.py
src/physical_process/disk_migration.py
src/disk_quantity/pagn_disk_adapter.py

src/workflows/event_rate.py
src/workflows/event_work.py
src/workflows/detect_rate.py
src/workflows/redshift_evolve.py        # replace/deprecate or wrap
src/workflows/cosmology.py              # preferred new shared helper
src/workflows/lisa_snr.py               # delegate D_L to shared cosmology

src/workflows/config.py
src/workflows/workflow.py
src/workflows/metadata.py
src/workflows/outputs.py                # only if legacy merge fields are changed
src/workflows/models.py                 # event schema if needed

scripts/redshift_scan.py
scripts/trap_plot.py
scripts/lisa_detectability_test.py      # update removed config args if needed
scripts/imri_physics_consistency_test.py  # preferred new harness
```

Search the repository for these names before editing:

```text
cosmic_merge
cosmic_merge_time_myr
cosmic_merge_smbh_mass_factor
cosmic_merge_radius_factor
hubble_constant_per_myr
gap_attenuation_factor
type_i_gap_denominator
TYPE_II_GAP_PARAMETER_THRESHOLD
local_aspect_ratio**2
DISK_ACCRETION_RATE_G_S_PER_MSUN
```

Do not assume the uploaded subset contains every consumer.

---

# 11. Verification commands

Follow repository conventions from `AGENTS.md`.

From repository root:

```bash
/home/planet/miniconda3/envs/sevnpy_env/bin/python scripts/imri_physics_consistency_test.py
/home/planet/miniconda3/envs/sevnpy_env/bin/python scripts/lisa_detectability_test.py
/home/planet/miniconda3/envs/sevnpy_env/bin/python scripts/imri_rate_visualization_test.py
/home/planet/miniconda3/envs/sevnpy_env/bin/python -m compileall src scripts
/home/planet/miniconda3/envs/sevnpy_env/bin/pyright
git diff --check
```

Run any existing migration/damping diagnostics that remain compatible:

```bash
/home/planet/miniconda3/envs/sevnpy_env/bin/python scripts/migration_test.py
```

If a pAGN-heavy full test is expensive, keep the algebraic consistency harness
fast and make the pAGN end-to-end check a separate explicit section.

---

# 12. Completion criteria

The task is complete only when all of the following are true:

1. pAGN midplane density reaches damping directly; no \(\Sigma/H\) density
   reconstruction remains in the IMRI damping path.
2. The two second-node damping signs match the supplied published equations.
3. Q=1 remains the operational outer disk boundary.
4. No torque helper can expand the disk beyond that Q=1 boundary.
5. KII is \(q^2/(\alpha h^5)\) everywhere and threshold 11 is used.
6. Gap-closed baseline torque is exactly JM17 Type I + the selected thermal
   torque, without old K attenuation factors.
7. Thermal torque uses signed \(x_c\), explicit \(m_c\), electron-scattering
   migration opacity, and optical-depth suppression.
8. Type-II uses Eq. 15 exactly.
9. Semi-analytic Type-II local logarithmic time remains
   \(1/(2\alpha h^2\Omega)\), consistent with Eq. 15 plus exact \(dL/dR\).
10. A trapped gas route does not suppress a valid finite direct-GW route.
11. No cosmic-merger physics is active or configurable in new single-AGN runs.
12. Cosmic time/redshift and luminosity distance use one flat-\(\Lambda\)CDM
    parameter pair.
13. `max_gw_time_myr` actually limits post-gas GW evolution.
14. Circular GW/LISA entry invariants pass.
15. Paired seed semantics and intrinsic rate denominator are unchanged.
16. The full compile/typecheck/diff-check suite passes.
17. Before/after differences are summarized by pipeline stage rather than only
    by final IMRI rate.

---

# 13. Final implementation note for Codex

Prefer small, reviewable commits/patch groups in this order:

```text
1. damping density + damping sign tests
2. Q=1 domain enforcement
3. KII + torque formula rewrite + torque tests
4. trap/GW event-routing fix + tests
5. remove cosmic-merger active interface
6. shared LambdaCDM cosmology + max-GW budget
7. scripts/config/output compatibility cleanup
8. deterministic before/after science diagnostic
```

Do not combine unrelated refactors with physics fixes.

When a formula changes, put the reference equation in the nearby docstring or
test comment so future maintainers can distinguish deliberate model choices
from implementation accidents.
