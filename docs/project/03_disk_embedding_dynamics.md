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
- `e ≥ 1` or `max_iter` → `DAMPING_INCOMPLETE`.

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
