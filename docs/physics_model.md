# Physical model: pipeline, module ownership, assumptions and validity

This document states **which module is authoritative** for each piece of
physics, what each one assumes, and where the model's known ambiguities are.
It describes the code as it is, not as one might wish it were.

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
   │        ├─ torque components ..... .torque      Lindblad / corotation / thermal / Type-II
   │        ├─ gap decision .......... .gap         every gap parameter and threshold
   │        ├─ traps ................. .traps       torque zeros -> stable / anti / boundary
   │        └─ radial integration .... .evolution   effective torque -> dr/dt -> time
   │
   ├─► gas / GW competition .......... workflows.event_rate.solve_gas_final_radius_cm
   │        └─ GW orbital evolution .. physical_process.gw_frequency_radius (Peters)
   │
   ├─► event classification .......... workflows.event_rate.classify_event_rate
   │                                   workflows.event_work (per-object orchestration)
   │
   └─► LISA .......................... lisa.snr          sensitivity, waveform, SNR
            └─ detectability ......... lisa.detectability band entry + threshold decision
```

Orchestration (`workflows/`) and figures (`viz/`) consume this chain; neither
defines any of it.

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
| Torque components | `physical_process.migration.torque` | Fixed gravitational + thermal composition |
| **Gap depletion and diagnostics** | `physical_process.migration.gap` | Continuous Kanagawa depth; Crida diagnostic only |
| **Migration traps** | `physical_process.migration.traps` | Derived from torque zeros; nothing hard-coded |
| Signed-torque integration | `physical_process.migration.evolution` | No runtime model selection |
| GW orbital evolution | `physical_process.gw_frequency_radius` | Peters, circular |
| Redshift-time relation | `physical_process.redshift_evolve` | Linear model |
| Simplified SMBH merger | `physical_process.cosmic_merge` | |
| Gas/GW handoff | `workflows.event_rate` | `solve_gas_final_radius_cm` is defined once |
| Event classification | `workflows.event_rate`, `workflows.event_work` | |
| LISA sensitivity / SNR | `lisa.snr` | Robson, Cornish & Liu (2019) analytic two-channel |
| LISA detectability | `lisa.detectability` | Threshold decision only |

---

## 3. Fixed torque and gap contract

Production exposes one migration prescription and no model-selection flags:

```
Gamma_total = f_gap Gamma_grav + Gamma_thermal
f_gap = 1 / (1 + 0.04 K),   K = q^2 / (alpha h^5)
```

`Gamma_grav` uses the Jimenez-Masset (2017) linear Lindblad and corotation
coefficients evaluated on the pAGN disk. The Kanagawa relation continuously
reduces this gravitational contribution; it is not converted to a classical
Type-II torque at an arbitrary threshold. The Crida pressure-plus-viscosity
parameter is retained only as a diagnostic of a deep gap.

The thermal contribution follows the local Grishin, Gilbaum & Stone (2024)
form: pAGN Rosseland opacity controls radiative diffusivity, hot and cold terms
have separate saturation factors, `min(r_B,H)` is used in the Bondi-Hoyle
transition, and optically thin gas is suppressed by
`1-exp(-sqrt(alpha) tau_gap)`. `tau_gap` includes the same continuous gap
depletion. The embedded object's luminosity is Eddington capped; direct checks
show the local Bondi supply exceeds that cap throughout the physical pAGN
disks for this project's `M_bh <= 10^5 Msun` range.

The paper's analytic disk zones and global luminosity bifurcation are not
imported: pAGN remains the sole disk model. This is an extrapolative demo at
low central mass, so each torque profile records the linear-theory conditions
`x_c < lambda`, `lambda < H`, and `r_H < H`, plus whether `K > 10^4` exceeds
the simple Kanagawa fit's calibrated range. These flags never silently select
another model or turn a finite torque into zero.

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
* Geometrically thin (`h < 1` everywhere; verified in Checkpoint B).
* Vertical hydrostatic equilibrium `c_s = H Ω` — verified to **4×10⁻¹⁶**
  against the disk's own Ω.
* The rotation law is **not exactly Keplerian**: pAGN's Ω differs from
  `√(GM/R³)` by up to **1.1×10⁻⁴** fractionally. Code that needs Ω should take
  it from the disk, not recompute it.
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
* Verified invariants (Checkpoint C): damping never increases inclination or
  eccentricity, final elements stay physical, and every evaluation returns an
  explicit status.
* "Embedded" means `EMBEDDED_STAR` or `EMBEDDED_BH`.

### Migration
* Circular, coplanar migrator; thin, viscously relaxed, axisymmetric disk.
* The only production law is the fixed Jimenez-Masset gravitational plus
  Grishin-2024 thermal torque described in §3.
* Kanagawa depletion is continuous. There is no Type-I/Type-II switch and no
  semi-analytic production alternative.
* The signed-torque integrator preserves outward regions. A stable zero can
  block inward delivery and returns `status="trap_bound"`, `time = inf`, and
  the blocking radius; this is a physical outcome, not a numerical failure.

### Traps
* A trap is a torque zero with positive torque inside and negative outside;
  the opposite ordering is an anti-trap. Nothing is hard-coded.
* Whether a trap exists is a **result**, not an assumption, and comes from the
  fixed total torque in §3.
* Trap **basins** are reported so a physical trap can be told from a numerical
  one: under the default, the narrowest trap at `10⁶ M⊙` has a basin of
  **0.04 decades** and should be treated as marginally resolved.

### GW and handoff
* Peters circular evolution; the `a⁴` scaling is verified to a log-slope of
  **4.02** (Checkpoint G).
* The handoff criterion is defined **exactly once**
  (`solve_gas_final_radius_cm`) and returns an explicit stop reason.
* Eccentric GW evolution is **not** supported by the SNR path: a non-zero
  eccentricity returns `snr_model_unavailable_nonzero_eccentricity`.

### LISA
* Redshift is applied **exactly once**: component masses enter in the source
  frame and are redshifted inside `lisa.snr`. Verified to 10⁻¹² (Checkpoint I).
* Frequencies crossing the public API are **observer frame**.
* SNR scales as **exactly** `1/D_L` when only the distance changes (verified to
  1 part in 10⁹ by varying `H0` alone).
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
| Aspect ratio | `h > 0` and finite | never reported as gap-open |
| Opacity `(logT, logR)` | the tabulated grid | **clipped** to the edge, silently |
| Migration path | inside the torque radial domain | `TorqueMigrationError` |
| Eccentricity (SNR) | exactly 0 | `model_unavailable` status |
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

---

## 7. Scope boundary

The project computes **host-level** single-halo results. It does not compute,
and must not be read as computing, a cosmic IMRI rate, a comoving rate density,
an observer-frame all-sky rate, or a LISA-detectable cosmic rate. Checkpoint J
asserts that no such column appears in the outputs.
