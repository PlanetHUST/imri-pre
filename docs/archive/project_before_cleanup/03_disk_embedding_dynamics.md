# Part 03 — Disk Embedding Dynamics (Damping)

## 1. Idea

A disk-crossing orbit is not yet a disk-embedded object. Gas drag at the
disk-plane nodes damps the orbital semi-major axis, eccentricity, and
inclination over many orbits. The damping stage decides **whether and how an
orbit becomes embedded** in the AGN disk — i.e., whether it reaches a state
compatible with the circular torque-migration model that follows.

The governing physical distinction (from
[`../reference/physical_model.md`](../reference/physical_model.md)):

- **disk crossing** is a geometric property of the initial orbit (part 02);
- **disk embedding** is a dynamical outcome of the evolved orbit.

The main idea: *damping is only a handoff filter. It must return the complete
evolved orbital state and classify it as embedded only when that state satisfies
an explicit physical contract — low inclination, sufficient circularity, and
radial containment inside the disk.*

## 2. Doing logic

### 2.1 Worker boundary

Authoritative owners: `src/workflows/workers.py` (multiprocessing boundary),
`src/physical_process/damp.py` (physics).

`run_damping_tasks_parallel()` runs each selected object in a worker pool. Each
worker (`init_damping_worker`) builds one `DampingCalculator` from the shared
disk arrays, inner/outer radius, and `use_disk_inner_radius` flag. The worker
serializes a typed `DampingResult` (status + full initial/final orbital state +
timing) back to the coordinator. Calculation exceptions become
`DampingStatus.CALCULATION_ERROR` with the error string attached, never a valid
physical status.

### 2.2 Active-node physics

`TimescaleCalculator.compute_all_timescales(mass_g, state, obj_type)`:

1. recompute both node radii from the **current evolved** state via
   `nsc_orbital.orbital_node_radii_cm`;
2. decide per-node membership in the physical annulus
   (`active_node_mask`, honoring `use_disk_inner_radius`);
3. if no node is active → return `None` (worker maps this to
   `OUTSIDE_DISK`);
4. build `DiskProperties` **only from active nodes** (aspect ratio at the mean
   active radius; gas lookups only for active nodes);
5. compute drag properties: Bondi radius for BHs, stellar radius for stars,
   relative velocity at each node, drag time;
6. compute crossing times and the orbital period.

All drag/ODE terms are evaluated **only at active nodes**. An inactive node
contributes exactly zero to the `a, e, i` changes and must not influence the
adaptive time step.

### 2.3 Integration

`DampingCalculator.damping_simulation()` adaptively integrates the per-orbit ODE
coefficients (semi-major axis, eccentricity, inclination) using a time-step
fraction that grows/shrinks to keep the relative change per step bounded. The
object type can switch mid-flight: a `Star` that has lived past its formation
time (`time ≥ formation_time`) is dragged as a `BH` thereafter
(`_object_type_at_time`).

Stopping conditions:

- inclination damped to the local criterion (see below) → final classification;
- duty cycle exhausted → `DAMPING_INCOMPLETE`;
- no active node → `OUTSIDE_DISK`;
- `e ≥ 1` → `DAMPING_INCOMPLETE`;
- positive accepted-step budget exhausted, non-finite trial, or stalled clock
  → `CALCULATION_ERROR` (a reached AGN deadline remains physical incompletion).

### 2.4 The embedded-state contract

`DampingCalculator.embedding_criteria(state)` — a state is eligible for
circular migration only if **all** of:

1. **low inclination:** `|i| ≤ (H/R)(a)` where `H/R` is the disk aspect ratio
   evaluated at the evolved `a` (not a fixed angle);
2. **sufficiently circular:** `e ≤ e_cir = 0.01`
   (`MAX_EMBEDDED_ECCENTRICITY`);
3. **radially contained:** `a(1-e) ≥ R_in` (when the inner radius is used) and
   `a(1+e) ≤ R_out` (Q=1 boundary).

Final classification (`_determine_final_state`):

- `EMBEDDED_BH` — contract satisfied and the object is a BH at that time
  (either initially a BH or `time ≥ formation_time`);
- `EMBEDDED_STAR` — contract satisfied but still a star (it may become a BH
  during migration);
- `DAMPING_INCOMPLETE` — contract not satisfied (low inclination alone is not
  enough);
- `TEST` — reserved for the deterministic diagnostic harness.

The elapsed damping time is `state.time_s - initial_state.time_s`, and the final
state (not the initial one) is what the migration stage consumes.

## 3. Key constants (owned by `damp.py`)

| Constant | Value | Meaning |
|---|---|---|
| `MAX_EMBEDDED_ECCENTRICITY` | 0.01 | circularity threshold `e_cir` |
| `INITIAL_TIME_STEP_FRACTION` | 0.01 | adaptive-step control |
| `MAX_ACCEPTED_RELATIVE_CHANGE` | 0.01 | step accept/reject threshold |
| `STELLAR_MASS_RADIUS_EXPONENT` | 0.8 | `R_star = Rsun (m/Msun)^0.8` |

## 4. Caveats

- **Embedding is not "has interacted with the disk".** It is the full contract.
  Low inclination is necessary but never sufficient.
- **Do not fabricate future states.** A semi-analytic estimate that a BH *could*
  finish damping before the duty cycle must not return the current pre-completion
  state as a completed embedded-BH state. Only actually evolved states may enter
  migration.
- **No instantaneous circularization, no radius clamping.** If the evolved
  orbit is not circular/contained, return an explicit non-embedded outcome; do
  not clamp `a` to the disk edge or extend the disk/torque domain to hide a
  damping failure.
- **Outside nodes must not be assigned the nearest disk-edge gas cell.** Inactive
  nodes are inactive; they get `nan`/`inf` drag terms and zero ODE contribution.
- **The inclination limit is local.** `(H/R)(a)` is recomputed from the solved
  pAGN profile as `a` evolves; it is not a constant opening angle.
- **Counter semantics.** Only `EMBEDDED_STAR` / `EMBEDDED_BH` results reach the
  event classifier; `DAMPING_INCOMPLETE`, `OUTSIDE_DISK`, and
  `CALCULATION_ERROR` are counted together as "outside or failed" and never sent
  to migration.

## 5. Comparison with cos_tde_revised and numerical alignment

The 2026-10-18 alignment uses the adaptive integration method in
`/home/planet/Phy/cos_tde_revised/damp.py` and retains IMRI's corrected disk
crossing physics and migration eligibility. Only the IMRI repository is edited.

### Which TDE source is used?

There are two active contexts in `cos_tde_revised`. Its present `main.py`
delegates to `scripts/run_simulation.py`, which imports a hash-verified July
snapshot from `vendor/july_physics/`. The root-level `damp.py`, used by the
previous scenario and focused diagnostics, contains newer adaptive-step fixes.
Changing that root file does not change the present TDE main runner.

Both are compared directly by `scripts/compare_damping_reference.py`. Every
implementation receives the same solved pAGN disk arrays and initial orbits;
the diagnostic records source hashes. Reference modules run in isolated
subprocesses to avoid collisions with IMRI's modules.

### Physical comparison

All three implementations use CGS units, the same two node radii
`r = a(1-e^2)/(1 +/- e*cos(omega))`, relative motion against circular disk gas,
an effective drag radius `max(R_star, 2*G*m/v_rel^2)`, and a crossing-weighted
orbital-element update. Their native outcomes differ for specific reasons:

| Quantity or rule | TDE pinned snapshot | TDE root file | IMRI, retained |
| --- | --- | --- | --- |
| Midplane density | `Sigma/H`, one mean-node height | Same | Solved pAGN density, `Sigma=2*rho*H` |
| Crossing speed | Total orbital speed times `sin(i)` | Same | Transverse orbital speed times `sin(i)` |
| Nodes outside the disk | Nearest profile sample | Same | Inactive node contributes zero drag |
| Second-node `da` bracket | Contains `-2e*cos(omega)` | Corrected to `+2e*cos(omega)` | Corrected |
| Second-node `de` numerator | Contains `-2cos(omega)` | Same | Corrected to `+2cos(omega)` |
| End of crossing integration | `i <= 8 degrees` | Same | `i <= local H/R` |
| Circular-migration handoff | No eccentricity/annulus eligibility check in damping | Same | Requires `e <= 0.01` and the whole orbit inside the physical disk |
| Star becoming a BH | Adds a local analytical BH damping estimate | Same | Continues numerical evolution using BH drag |
| Returned state | Legacy tuple of time, axis and flag | Same, plus diagnostic attributes | Typed status and full evolved orbit |

IMRI stops the crossing approximation at `i <= H/R`. A low-inclination orbit
that still has `e > 0.01` remains `damping_incomplete`; this change does not add
an embedded eccentric-orbit evolution model. The two projects therefore do not
promise identical capture fractions or migration starting radii.

The node corrections are checked independently in
`tests/test_damping_equations.py` by applying a drag impulse to Cartesian state
vectors and recomputing the osculating elements. They are not inferred from
agreement with the reference code.

### Numerical alignment

IMRI now uses the newer TDE root solver's accepted-step algorithm:

- Start with a drag-to-crossing timescale fraction of `0.01`, limit relative
  changes in axis/inclination to `0.01`, and grow sufficiently small steps up
  to a fraction of `0.1`.
- On rejection, shrink by the smaller of `0.5` and `0.9*0.01/max_change`.
  There is no `1e-6` floor that can trap the solver in repeated rejection.
- Reuse the unchanged state's timescales and derivatives on rejected trials.
  Count accepted steps toward `max_iter`.
- Reject non-finite trial states and detect a clock that cannot advance.
  Exhausting a positive accepted-step budget before reaching the physical
  deadline is `calculation_error`, distinct from `damping_incomplete` at AGN
  shutdown. `max_iter=0` retains the existing no-integration diagnostic behavior.

The local disk, node masks, stellar evolution rule, embedding conditions,
workflow APIs, population draws, and migration model remain unchanged by this
alignment. The pinned TDE snapshot is left intact.

### Reproduction and evidence

From the IMRI root, with the scientific Python environment:

```bash
PYTHONPATH=src python tests/run_tests.py
python scripts/compare_damping_reference.py \
  --reference-root /home/planet/Phy/cos_tde_revised \
  --output-dir output/diagnostics/damping_alignment_20261018
```

The diagnostic writes `shared_disk.npz`, `damping_comparison.csv`, and
`validation.json`. It compares a `3000 Msun` central mass, 10 Myr AGN lifetime,
400-cell disk, four initial orbits, and both a `1 Msun` star and `10 Msun` BH.
The table includes node densities, crossing times, per-orbit coefficients,
analytical estimates and native numerical return values. Native integrations
have a 5000 accepted-step budget (or trial budget in the pinned snapshot).
Legacy return flags are recorded literally: they cannot certify convergence.

A separate stiff constant-derivative experiment sets the same `0.02 rad`
stopping threshold in both solvers. IMRI and the newer TDE controller agree
to relative tolerance `1e-13`; the pinned floor stalls at the initial state.
The shared orbital periods agree, and the symmetric-node case exposes the
legacy density normalization factor of two, with the small nearest-cell
radius correction recorded explicitly.

The eight native IMRI cases retain their pre-alignment statuses; the largest
relative damping-time change is below 0.2%. These controlled comparisons and
impulse tests validate the implementation and its stated differences. They do
not establish the physical accuracy of disk capture rates or a production
population convergence result.
