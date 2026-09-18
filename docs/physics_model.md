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
| Torque components | `physical_process.migration.torque` | Owns no threshold |
| **Gap criteria and thresholds** | `physical_process.migration.gap` | The only place a gap parameter is spelled out |
| **Migration traps** | `physical_process.migration.traps` | Derived from torque zeros; nothing hard-coded |
| Regime selection and integration | `physical_process.migration.evolution` | Owns no equation from the three above |
| GW orbital evolution | `physical_process.gw_frequency_radius` | Peters, circular |
| Redshift-time relation | `physical_process.redshift_evolve` | Linear model |
| Simplified SMBH merger | `physical_process.cosmic_merge` | |
| Gas/GW handoff | `workflows.event_rate` | `solve_gas_final_radius_cm` is defined once |
| Event classification | `workflows.event_rate`, `workflows.event_work` | |
| LISA sensitivity / SNR | `lisa.snr` | Robson, Cornish & Liu (2019) analytic two-channel |
| LISA detectability | `lisa.detectability` | Threshold decision only |

---

## 3. The gap criterion — resolved, with consequences

Two different dimensionless quantities were historically both called "the gap
parameter K", and they drove two different parts of the model:

| Name | Expression | Threshold | Historically used for |
|---|---|---|---|
| `kii` | `q² / (α h²)` | `> 11` | the Type-I → Type-II runtime switch |
| `kanagawa` | `q² / (α h⁵)` | `> 25` | the Type-I torque attenuation in `torque.py` |

They differ by `h⁻³` — six orders of magnitude at `h ≈ 0.01`.

**Resolution.** The production default is now **`kanagawa`**, so the Type-II
switch and the Type-I attenuation are driven by *one* quantity instead of two
inconsistent ones. `kii` remains selectable
(`--gap-open-condition kii`) and reproduces all pre-change results byte-for-byte.
The default is set in exactly one place,
`physical_process.migration.gap.DEFAULT_GAP_CRITERION`.

### What the change does

Measured with `m = 10 M⊙`, `α = 0.1`, 1000 radial cells. The Type-II torque is
**inward everywhere** (Checkpoint D), so moving cells to Type II erases the
outward-torque regions that create migration traps:

| `M•` [M⊙] | criterion | Type-II cells | outward-torque cells | stable traps | anti-traps |
|---|---|---|---|---|---|
| 10³ | `kii` | 26 | 704 | 1 | 2 |
| 10³ | **`kanagawa`** | 943 | **0** | **0** | **0** |
| 10⁴ | `kii` | 0 | 684 | 0 | 1 |
| 10⁴ | **`kanagawa`** | 845 | **0** | **0** | **0** |
| 10⁵ | `kii` | 0 | 507 | 0 | 1 |
| 10⁵ | **`kanagawa`** | 640 | **0** | **0** | **0** |
| 10⁶ | `kii` | 0 | 250 | 3 | 4 |
| 10⁶ | **`kanagawa`** | 189 | 69 | 3 | 3 |
| 10⁷ | either | 0 | 0 | 0 | 0 |

> **Migration traps no longer operate for `M• ≤ 10⁵ M⊙`.** Under `kanagawa` the
> torque is inward at every radius there, so nothing can be trap-bound. Traps
> survive only at `10⁶ M⊙`. Since migration trapping is a central premise of
> this project, any result that relied on trapping must be re-examined.

### Effect on a full run

Controlled single-halo run (`M• = 10⁵ M⊙`, `M_NSC = 5×10⁴ M⊙`, 2 repeats,
seed 777). Population, damping and embedding are **bit-identical**; the change
is confined to transport and everything downstream of it:

| Quantity | `kii` | `kanagawa` |
|---|---|---|
| disk-crossing objects | 26.5 | 26.5 *(unchanged)* |
| embedded BHs / BH candidates | 1.5 / 1.5 | 1.5 / 1.5 *(unchanged)* |
| migration evaluated | 3 | 3 *(unchanged)* |
| migration **finite** | 0 | **3** |
| migration **trap-bound** | 3 | **0** |
| trap-bound mass [M⊙] | 223.0 | **0.0** |
| **IMRI candidates** | 0 | **1.5** |
| **local IMRI rate** [Myr⁻¹] | 0 | **0.15** |
| detectable IMRIs | 0 | 0 *(still below SNR threshold)* |

Per-object transport reasons flip wholesale: `migration_status` goes from
`trap_bound` (6/6) to `finite` (6/6), and `gas_stop_reason` from
`migration_trap_bound` to `gw_dominates_after_gas`.

So the change converts a population that was entirely trapped in the disk into
one that migrates to the GW-driven regime and forms IMRI candidates. It does
**not** produce LISA detections in this configuration — the SNR remains below
threshold.

### Calibration of the threshold

`K > 25` is **not** an arbitrary number. `scripts/calibrate_gap_threshold.py`
establishes it from the model itself; artefacts in
`output/round2/calibration_gap_threshold/`.

**1. The threshold is a statement about gap depth.** The project's Type-I gap
denominator is `1 + K/κ_norm` with `κ_norm = 25`. That is *algebraically
identical* to the Kanagawa gap-depth law

```
Σ_gap / Σ₀ = 1 / (1 + 0.04 K)
```

verified to a relative deviation of **2.2×10⁻¹⁶**. So `K > K*` means "the gap
is cleared below a residual surface density `f = 1/(1+0.04K*)`", and any
threshold can be quoted as a depletion fraction. Helpers
`depletion_fraction()` and `threshold_for_depletion()` convert between them.

**2. `K = 25` is the value the model itself fixes.** It is the half-depth point
(`f = 0.5`) — the same `κ_norm` the Type-I torque already uses. Choosing it
means the Type-II switch happens exactly where the model says the Type-I torque
has been halved by the gap. No other value in the family is determined by the
code rather than chosen.

**3. The alternative calibration target was tested and rejected.** One could
instead place the switch where `|Γ_I| = |Γ_II|`, making the transition
continuous. Measured across the SMBH × migrator grid, that crossover spans a
factor **2.5×10⁶** in K (from `K ≈ 26` at `M• = 10⁶ M⊙`, `m = 10 M⊙` to
`K ≈ 6.4×10⁷` at `M• = 10³ M⊙`, `m = 100 M⊙`). **No single global threshold can
make the switch continuous**, so continuity is not an available calibration
target and the discontinuity is irreducible for a global-K criterion.

**4. Sensitivity.** Stable traps versus threshold (`m = 10 M⊙`):

| `f` | `K*` | 10³ M⊙ | 10⁴ M⊙ | 10⁵ M⊙ | 10⁶ M⊙ |
|---|---|---|---|---|---|
| 0.90 | 2.8 | 0 | 0 | 0 | 0 |
| **0.50** | **25** | **0** | **0** | **0** | **3** |
| 0.25 | 75 | 0 | 0 | 0 | 3 |
| 0.10 | 225 | 0 | 0 | 1 | 3 |
| 0.01 | 2475 | 0 | 0 | 1 | 3 |

End-to-end (`M• = 10⁵ M⊙`, `M_NSC = 5×10⁴ M⊙`, seed 777):

| `f` | `K*` | trap-bound | finite | IMRI | rate [Myr⁻¹] |
|---|---|---|---|---|---|
| 0.90 | 2.8 | 0.0 | 3.0 | 1.5 | 0.15 |
| **0.50** | **25** | **0.0** | **3.0** | **1.5** | **0.15** |
| 0.25 | 75 | 0.5 | 2.5 | 1.5 | 0.15 |
| 0.10 | 225 | 1.5 | 1.5 | 1.5 | 0.15 |
| 0.01 | 2475 | 2.5 | 0.5 | 0.5 | 0.05 |

The IMRI yield is **flat across two decades in `K*`** (f = 0.9 → 0.05) and only
falls at f = 0.01. The trap-bound fraction is far more sensitive: traps
reappear at `M• = 10⁵ M⊙` once `f ≤ 0.1`. So the headline yield is robust to
the threshold, while any trap-dependent conclusion is not.

### What remains a judgement call

Whether "gap open" should mean 50 % or 10 % depletion is a modelling choice
this repository cannot settle offline; no external reference could be
consulted. The calibration fixes the *family* and the *interpretation*, and
identifies the one value the model determines by itself. Revising it is one
flag: `--gap-threshold 225` (f = 0.1). Note the corotation torque — the
outward-driving one that creates traps — is already suppressed by
`exp(-25/20) = 0.29` at the calibrated threshold.

### Other caveats

* **Results from before this change are not comparable to results after it.**
  Use `--gap-open-condition kii` to reproduce them.
* **The Type-I → Type-II transition is now actually exercised**, where under
  `kii` it usually never fired. The hard per-cell switch is intentionally
  discontinuous; per point 3 above, no global threshold can remove that.

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
* Type-I uses Paardekooper-style coefficients with a thermal-torque term; the
  Type-I torque is attenuated by the **Kanagawa-form** K through
  `exp(-K/κ_t)` and `1 + K/κ_norm` with `κ_t = 20`, `κ_norm = 25`.
* The Type-I → Type-II switch is a **hard per-cell switch**, deliberately
  discontinuous: the two prescriptions are different models, not two limits of
  one expression, and no blending law is implemented. With the default `kii`
  criterion on the Checkpoint-E disk the switch never fires, so the
  discontinuity is not exercised in practice.
* Two integrators, selected by `--disk-model`:
  * `torque_with_trap` — integrates the **signed** torque, so an outward-torque
    region blocks inward delivery and returns `status="trap_bound"`,
    `time = inf` and the blocking radius. This is a **physical** outcome, not a
    numerical failure.
  * `semi_without_trap` — integrates local timescales over `d ln R`; unsigned,
    so it can never produce a trap.

### Traps
* A trap is a torque zero with positive torque inside and negative outside;
  the opposite ordering is an anti-trap. Nothing is hard-coded.
* Whether a trap exists is a **result**, not an assumption, and it depends on
  the gap criterion — see §3 for the measured table under both criteria. Under
  the current default, traps exist only at `10⁶ M⊙`.
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

---

## 7. Scope boundary

The project computes **host-level** single-halo results. It does not compute,
and must not be read as computing, a cosmic IMRI rate, a comoving rate density,
an observer-frame all-sky rate, or a LISA-detectable cosmic rate. Checkpoint J
asserts that no such column appears in the outputs.
