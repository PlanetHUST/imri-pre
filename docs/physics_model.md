# Physical model: pipeline, module ownership, assumptions and validity

This document states **which module is authoritative** for each piece of
physics, what each one assumes, and where the model's known ambiguities are.
Reviewed against the working source on 2026-10-22. It describes current
implementation choices and explicitly identified limits. Historical session
decisions are reconciled in [reference/history.md](reference/history.md);
original exports are preserved in [archive/](archive/README.md).

---

## 1. The physical pipeline

```
host (z, M_smbh, M_nsc, Z)
   │
   ├─► AGN disk structure ............ disk_quantity.pagn_disk_adapter  (pAGN Sirko/Thompson)
   │        └─ opacity ............... disk_quantity.opacity            (OPAL + Alexander-Ferguson)
   │
   ├─► compact-object population ..... physical_process.sampling        (Pop II / Pop III IMF)
   │        └─ initial orbits ........ physical_process.nsc_orbital
   │
   ├─► disk crossing ................. physical_process.nsc_orbital.disk_crossing_mask
   │
   ├─► damping / embedding ........... physical_process.damp
   │
   ├─► migration ..................... physical_process.migration
   │        ├─ torque components ..... .torque      gravitational / thermal / gap-drift
   │        ├─ gap depletion ......... .gap         Kanagawa depth + Crida diagnostics
   │        ├─ traps ................. .traps       torque zeros -> stable / anti / boundary
   │        └─ radial integration .... .evolution   effective torque -> dr/dt -> time
   │
   ├─► one circular gas + GW orbit .... physical_process.orbital_evolution
   │        └─ moving band boundary .. physical_process.redshift_evolve
   │
   ├─► event classification .......... workflows.event_rate.classify_coupled_event
   │                                   workflows.event_work (per-object orchestration)
   │
   └─► LISA .......................... lisa.snr          sensitivity, waveform, SNR
            └─ detectability ......... lisa.detectability band entry + threshold decision
```

`workflows/` coordinates the stages; `orbital_evolution.py` owns the coupled
radius trajectory and `event_rate.py` classifies its first entry. Figures in `viz/` consume
the results and do not redefine those physical decisions.

---

## 2. Module ownership

| Physics | Authoritative module | Notes |
|---|---|---|
| Physical constants | `physical_process.constants` | CGS internally, SI for the detector chain |
| Opacity | `disk_quantity.opacity` | Tables + interpolation only; no plotting |
| Disk structure (**production**) | `disk_quantity.pagn_disk_adapter` | pAGN Sirko/Thompson, solved in SI, converted to CGS |
| Disk structure (**comparison only**) | `disk_quantity.disk_solve` | Independent in-repo α-disk solver; **not** used by any run |
| IMF and stellar lifetimes | `physical_process.sampling` | Pop III below 10⁻⁴ Z⊙, else Pop II |
| NSC radius, orbits, disk crossing | `physical_process.nsc_orbital` | |
| AGN duty cycle | `physical_process.agn` | |
| Damping / embedding | `physical_process.damp` | |
| Torque components | `physical_process.migration.torque` | Fixed gravitational + bounded thermal + gap-drift composition |
| **Gap depletion and diagnostics** | `physical_process.migration.gap` | Continuous Kanagawa depth; Crida diagnostic only |
| **Migration traps** | `physical_process.migration.traps` | Derived from torque zeros; nothing hard-coded |
| Gas-only signed-torque integration | `physical_process.migration.evolution` | Historical diagnostic only; production converts the signed torque to radial drift for the coupled orbit |
| Integrated semi-analytic migration (**comparison only**) | `physical_process.migration.semi_analytic` | Archived `semi_without_trap`; programmatic only, never a production CLI option; no traps |
| Coupled circular orbit | `physical_process.orbital_evolution` | Signed gas drift and Peters GW drift act simultaneously until gas ceases; vacuum `r^4` continuation |
| Redshift-time relation | `physical_process.redshift_evolve` | Flat matter-plus-Lambda cosmic age, sharing H0 and Omega_m with detector distance |
| Simplified SMBH merger | `physical_process.cosmic_merge` | Legacy standalone diagnostic only; disabled in coupled fixed-mass production |
| Legacy gas/GW routing | `workflows.event_rate` | `solve_gas_final_radius_cm` remains for historical diagnostic tests, not production event physics |
| Event classification | `workflows.event_rate`, `workflows.event_work` | |
| LISA sensitivity / SNR | `lisa.snr` | Robson, Cornish & Liu (2019) analytic two-channel |
| LISA detectability | `lisa.detectability` | Threshold decision only |

---

## 3. Fixed torque and gap contract

Production exposes one migration prescription and no model-selection flags.
A comparison-only integrated semi-analytic scenario (the archived
`semi_without_trap`) lives in
[`migration/semi_analytic.py`](../src/physical_process/migration/semi_analytic.py);
it is constructed programmatically for controlled experiments and is never
selected by the production CLI. The production law is:

```
Gamma_total = f_gap Gamma_grav + Gamma_thermal + (1-f_gap) Gamma_gap
f_gap = 1 / (1 + 0.04 K),   K = q^2 / (alpha h^5)
Gamma_gap = -(3/4) alpha h^2 m r^2 Omega^2 min(1, B)
B = Sigma r^2 / m
```

`Gamma_grav` uses the Jimenez-Masset (2017) linear Lindblad and corotation
coefficients evaluated on the pAGN disk. The Kanagawa relation continuously
reduces this gravitational contribution; it is not converted to a classical
Type-II torque at an arbitrary threshold. The Crida pressure-plus-viscosity
parameter is retained only as a diagnostic of a deep gap.

As depletion deepens, the removed gravitational weight is handed continuously
to `Gamma_gap`, an inward viscous-drift limit slowed by the local disk mass.
For `B < 1`, its radial drift is reduced linearly by `B`; for `B >= 1`, it
saturates at the viscous drift speed. This is one continuous contract, not a
runtime Type-I/Type-II branch.

The thermal contribution follows the local Grishin, Gilbaum & Stone (2024)
form: pAGN Rosseland opacity controls radiative diffusivity, hot and cold terms
have separate saturation factors, `min(r_B,H)` is used in the Bondi-Hoyle
transition, and optically thin gas is suppressed by
`1-exp(-sqrt(alpha) tau_gap)`. `tau_gap` includes the same continuous gap
depletion. The embedded object's luminosity is Eddington capped; direct checks
show the local Bondi supply exceeds that cap throughout the physical pAGN
disks for this project's `M_bh <= 10^5 Msun` range.

The paper's analytic disk zones and global luminosity bifurcation are not
imported: pAGN remains the sole disk model. Each torque profile records the
linear-theory conditions `x_c < lambda`, `lambda < H`, and `r_H < H`, plus
whether `K > 10^4` exceeds the simple Kanagawa fit's calibrated range. The
thermal contribution is excluded where those linear checks fail; its raw
coefficient, validity mask, and applied weight remain available separately so
that exclusion cannot be mistaken for a prediction of physically zero thermal
torque. Extrapolation is diagnostic-only and must be requested explicitly.

The WKB helper flags the two cells adjacent to a background value jump above
0.15 dex and interpolates their fitted log-slopes from unflagged neighbors.
Neighboring spline ringing can survive this narrow repair. This is an
implementation approximation, not a validated torque solution through the
transition. The torque arrays can also include pAGN's padded tail outside the
physical Q=1 radius. See [migration_model_alignment.md](migration_model_alignment.md)
for the current domain and convergence limits.

Traps are derived from positive-to-negative zeros of this single signed total
torque. Their existence is an output of the disk and torque calculation, not a
configuration choice.

For sensitivity analysis, `inward_only_counterfactual_torque` supplies the
opposite extreme `Gamma_no-trap=-|Gamma_total|`. It preserves the local torque
magnitude while reflecting outward cells inward. This is labelled a
counterfactual diagnostic, never a second production model: comparison with
the signed result isolates the effect of traps without mixing in another
torque law.

## 4. Assumptions, by subsystem

### Disk
* The disk/torque approximation assumes a geometrically thin background.
  Check local aspect ratios on the profile used for each calculation.
* Vertical hydrostatic equilibrium uses `c_s = H Ω`; compare against the
  disk-provided Ω rather than mixing mass/constant conventions.
* pAGN's Sirko rotation law is Keplerian by construction. The project's CGS
  solar-mass constant differs from pAGN's value by 2.3×10⁻⁴, so a downstream
  recomputation with the project constant carries a corresponding small
  convention mismatch; torque code uses the disk-provided `Omega`.
* The outer boundary is set by Toomre `Q → 1` (self-gravity truncation).
* Opacity is **clipped**, not extrapolated, outside the tabulated range, and
  steps by up to **2.4 decades** between adjacent radial cells at its table
  transitions. This is a property of the table, not a solver failure.

### Population
* Metallicity below `10⁻⁴ Z⊙` selects the Pop III top-heavy IMF; above it, a
  Salpeter Pop II IMF with stars below 1 M⊙ discarded before the mass budget is
  accumulated (a preserved historical assumption).
* Sampling uses the **global NumPy RNG**, seeded once per repeat by the
  workflow. This is why parallel determinism requires one seed per job.

### Damping
* Two-node disk-crossing drag uses solved pAGN midplane densities, the true
  vertical crossing speed, and zero contribution from nodes outside the disk.
* Crossing integration stops at the local `i <= H/R` boundary. Migration
  eligibility additionally requires `e <= 0.01` and radial containment.
* Adaptive integration follows the newer `cos_tde_revised/damp.py` step
  controller, with cached rejected trials and no minimum fraction floor.
  Numerical exhaustion/stalling is `calculation_error`; an unfinished orbit
  at AGN shutdown is `damping_incomplete`.
* "Embedded" means `EMBEDDED_STAR` or `EMBEDDED_BH`.

The 2026-10-18 alignment targets the newer root-level TDE solver. The current
TDE `main.py` instead imports its hash-pinned `vendor/july_physics/damp.py`,
which retains the old minimum timestep floor. Both reference sources are
compared explicitly; neither TDE file is edited.

| Damping quantity/rule | TDE reference implementations | IMRI retained physics |
| --- | --- | --- |
| Density | `Sigma/H`, using one mean-node height | Solved density with `Sigma=2*rho*H` |
| Crossing speed | Total speed times `sin(i)` | Transverse speed times `sin(i)` |
| Node outside physical disk | Nearest-cell gas lookup | Zero contribution from inactive node |
| Second-node equations | Pinned source has `da` and `de` sign errors; root fixes `da` | Both pass independent drag-impulse tests |
| Capture | Fixed `i <= 8 degrees` | Local `i <= H/R`, plus `e <= 0.01` and containment for migration |
| Stellar evolution | Adds an analytical future BH damping estimate | Continues evolving the actual orbit with BH drag |

The aligned numerical method starts with a timescale fraction of `0.01`,
limits relative axis/inclination changes to `0.01`, and shrinks rejected
trials by `min(0.5, 0.9*0.01/max_change)` without a lower fraction floor.
Rejected trials reuse their orbital terms; `max_iter` counts accepted steps.
`max_iter=0` keeps the existing no-integration diagnostic behavior. Low
inclination with residual `e > 0.01` remains incomplete: no embedded eccentric
evolution model is added.

Reproduce the comparison with:

```bash
python scripts/compare_damping_reference.py \
  --reference-root /home/planet/Phy/cos_tde_revised \
  --output-dir output/diagnostics/damping_alignment_20261018
```

The shared 400-cell pAGN disk has a `3000 Msun` central object and a 10 Myr
AGN lifetime. Four initial orbits are each evaluated for a `1 Msun` star and a
`10 Msun` BH. Artifacts contain source hashes, node quantities, coefficients,
native outcomes, and the controlled numerical comparison. On the identical
stiff ODE and `0.02 rad` stopping criterion, IMRI and the newer TDE controller
agree to `1e-13` relative tolerance; the pinned controller stalls at time zero.
The eight IMRI cases keep their previous classifications and change damping
times by less than 0.2%. This validates implementation alignment on these
cases, not capture-rate accuracy or population convergence.

### Migration
* Circular, coplanar migrator; thin, viscously relaxed, axisymmetric disk.
* The only production law is the fixed Jimenez-Masset gravitational plus
  bounded Grishin-2024 thermal plus inertia-limited gap-drift torque in §3.
* Kanagawa depletion is continuous. There is no Type-I/Type-II switch in
  production and no semi-analytic production alternative. The comparison-only
  scenario integrates local Type-I timescales and switches to local Type-II
  timescales where the selected gap-opening mask is true; it is inward-only,
  has no traps, and is never selected by the CLI.
* The standalone gas-only signed-torque integrator preserves outward regions
  and may return `trap_bound`. Production instead combines this signed gas
  drift with inward GW drift; a gas-torque zero is not automatically a net
  orbital stall or a permanent block. The legacy route/trap counters are not
  a coupled-orbit trap census.

### Traps
* A gas trap is a torque zero with positive torque inside and negative outside;
  the opposite ordering is an anti-trap. Nothing is hard-coded. A stationary
  coupled orbit instead requires gas torque to balance the negative GW torque.
* Whether a trap exists is a **result**, not an assumption, and comes from the
  fixed total torque in §3.
* Trap **basins** measure the radial width of the outward-torque region. A
  particular scan may find no resolved stable zeros while retaining positive
  cells or an `outward_blocked` path. Trap convergence must be checked on the
  actual physical domain and migrator masses; it is not guaranteed by the
  current jump repair or by a historical checkpoint.

### Coupled gas and GW orbit
* After the embedding rule accepts `e <= 0.01`, the orbit is explicitly
  projected to `e=0`. This is a controlled approximation: the circular
  Peters law preserves zero eccentricity, while the gas prescription assumes
  an adiabatically circular response. The configuration rejects nonzero GW
  eccentricity; stellar/remnant masses are fixed pending SEVN integration.
* For circular `L=mu sqrt(GMr)`, the signed gas torque gives
  `dr/dt|gas=Gamma_gas/(dL/dr)` and GW gives
  `dr/dt|GW=-beta/r^3`, with both applied on the same orbit. The frequency
  crossing is found against the evolving observed-frame redshift. There is
  no selected gas/GW crossover or gas-final radius.
* Gas stops at the AGN duty deadline or radial profile edge; thereafter the
  exact vacuum `r^4` law applies. A net-torque zero can stall the gas-active
  orbit without blocking its later GW continuation.
* Legacy `band_entry_status` and migration counters are geometric/compatibility
  diagnostics; `orbit_stop_reason`, entry and disk-off fields are the physical
  result. `r_handoff` and old split-time fields are unavailable (`NaN`).

### LISA
* Component masses enter in the source frame and are redshifted once inside
  `lisa.snr`.
* Frequencies crossing the public API are **observer frame**.
* SNR scales as `1/D_L` when only the distance changes.
* SNR is **not** monotone in redshift at fixed source-frame masses and fixed
  observed frequency, because `(1+z)` raises the detector-frame chirp mass
  while the distance grows. Both effects are physical; do not "fix" this.
* The detectability threshold is a pure comparison: the SNR value itself is
  threshold-independent.

---

## 5. Validity limits

| Quantity | Supported range | What happens outside |
|---|---|---|
| SMBH mass (batch domain) | 10³–10⁶ M⊙ by default | `outside_local_model_domain`, configurable |
| Formal observer-band entry | Mass/redshift/frequency dependent, above ISCO | `band_entry_inside_isco`, unsupported rather than zero; host-domain admission alone does not guarantee this |
| Aspect ratio | `h > 0` and finite | never reported as gap-open |
| Opacity `(logT, logR)` | the tabulated grid | **clipped** to the edge, silently |
| Migration path | inside the torque radial domain | `TorqueMigrationError` |
| Eccentricity (coupled orbit) | exactly 0 after embedding projection | nonzero configuration rejected |
| LISA mission duration | 0.5, 1, 2, 4 yr | `ValueError` (confusion-noise table) |
| Redshift | finite, ≥ 0 | rejected at configuration time |

---

## 6. Status semantics

Execution state and physical outcome are kept strictly separate.

* A job with `status = success` and `n_imri = 0` is a **physical zero**.
* Any other status means the model **could not be evaluated**; its physical
  columns are `NaN`, never `0`.
* The kernel's own physical statuses (`rate_status`,
  `detector_rate_status`, `gas_stop_reason`, `migration_status`,
  `event_rate_reason`, `detect_rate_reason`) are preserved and propagate to the
  per-job summary, so a zero is always explainable.
* `migration_status="trap_bound"` requires a resolved stable torque zero;
  `migration_status="outward_blocked"` means an outward region blocks inward
  delivery without a resolved trap. Both are physical barriers, not errors.
* `classify_coupled_event` counts a BH only when it exists by the coupled
  orbit's first finite observed-band entry before ISCO and `z=0`. A stellar
  entry before BH formation is not counted. Entry is not coalescence.
* Gas-active entries use a first-order circular-quadrupole SNR integrated over
  the actual in-band coupled trajectory. In observer time the implementation
  integrates `(16/5) A_vac(f)^2 dot_f_GW(f)/S_n(f) dt_obs`, retaining the
  existing sky/inclination normalization and LISA noise. This is the
  stationary-phase amplitude/residence-time identity, not an environmental
  phase template or a matched-filter recovery guarantee. A mission window is
  placed at band entry or against the track's loud end per configuration; if
  no loud end is reached before `z=0`, the optimal request falls back to the
  band-entry window and records that case.
* After disk-off, the existing vacuum SNR remains in use. Missing input,
  numerical failure, evaluated nondetection and no overlap stay distinct.

---

## 7. Scope boundary

The project computes **host-level** single-halo results. It does not compute,
and must not be read as computing, a cosmic IMRI rate, a comoving rate density,
an observer-frame all-sky rate, or a LISA-detectable cosmic rate. Checkpoint J
asserts that no such column appears in the outputs.

The source-frame clock and detector luminosity distance now share one flat
matter-plus-Lambda cosmology with the configured `detector_hubble_constant_km_s_mpc`
and `detector_omega_m`. Radiation is absent from both. `RedshiftEvolution`
uses the analytic age relation

\[
t(z)=\frac{2}{3H_0\sqrt{\Omega_\Lambda}}
\operatorname{asinh}\!\left[
\sqrt{\frac{\Omega_\Lambda}{\Omega_m}}(1+z)^{-3/2}
\right],\qquad \Omega_\Lambda=1-\Omega_m.
\]

Elapsed Myr start at the configured source
redshift, and the GW continuation stops at the age of `z=0`. The old independent
`hubble_constant_per_myr` input was removed.

The coupled workflow rejects finite `cosmic_merge_time_myr`; its default is
infinity and component masses stay fixed. The old one-event merger remains
only in the standalone legacy detector function. `max_gw_time_myr` is a
recorded compatibility parameter, not an additional physical cutoff.
