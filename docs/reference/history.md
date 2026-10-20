# Decisions consolidated from the reference sessions

Reviewed against the working source on 2026-10-18. This is a decision ledger,
not a list of instructions to replay. Current contracts are
[physics_model.md](../physics_model.md) and
[engineering_contract.md](../engineering_contract.md).

## Original evidence

Both former root text files are preserved byte-for-byte in
[archive/transcripts/](../archive/transcripts/). The
[archive index](../archive/README.md) records their SHA-256 checksums.
Line numbers below refer to those unchanged originals.

| Session | What it records | Durable destination |
| --- | --- | --- |
| [2026-09-18 export](../archive/transcripts/2026-09-18-163756-read-instructionsmd-carefully-and-complete-t.txt), lines 6-1116 | Merger-tree adapter, host mapping, local-run reuse, deterministic job identity, failure isolation and interface checkpoints | [Merger-tree interface](../merger_tree_interface.md) |
| Same file, lines 1117-1710 | Round-2 module ownership, moving physics out of plotting/orchestration, gap and trap audit | [Source map](../project/README.md), [engineering contract](../engineering_contract.md) |
| Same file, lines 1711-2241 | Hard gap switches, Kanagawa threshold scans and their effects on traps | Historical experiments; superseded by [the fixed continuous contract](../migration_model_alignment.md) |
| [2026-09-21 export](../archive/transcripts/2026-09-21-204204-hello.txt), lines 12-733 | Seventeen initial audit findings, hypotheses and suggested repairs | Findings reconciled below; not all initial claims survived verification |
| Same file, lines 734-808 | User's single-host scope and gas-phase duty-cycle clarification | [Scope and event semantics](../physics_model.md) |
| Same file, lines 1013-1928 | Density, node equations, analytical impulse checks and correction of the energy argument | [Damping](../physics_model.md#damping), [equation tests](../../tests/test_damping_equations.py) |
| Same file, lines 1929-3352 | Torque composition, gas/GW routing and finite-entry accounting repairs | [Migration contract](../migration_model_alignment.md), [event_rate.py](../../src/workflows/event_rate.py) |
| Same file, lines 3353-4182 | Disk dominance, validity occupancy and user-directed vertical crossing speed | [Damping](../physics_model.md#damping), [migration limits](../migration_model_alignment.md) |
| Same file, lines 4183-6183 | Continuous-gap alternatives, trap basin filtering and unresolved model-selection discussion | Historical comparisons; current production exposes one signed contract |

The exports include shell output, patches and earlier errors. An old success
message, trap count or threshold scan applies only to that code and input set.

## Decisions retained in the current model

- The immediate scientific object is one SMBH/NSC host during an AGN episode.
  Tree rows can call that kernel; they do not by themselves define a cosmic rate.
- Disk crossing requires an actual orbital node in the physical annulus.
  Sampling gas at an outside node by clamping it to an edge is not allowed.
- Damping must return the evolved orbit. Circular migration requires local
  low inclination, eccentricity at most 0.01 and radial containment.
- pAGN owns the disk and physical first outward Q=1 edge. Density is read from
  its solved profile. The user later directed the true vertical crossing speed.
- The gas phase is limited by the AGN episode. A later purely GW band entry is
  not automatically rejected merely because its time exceeds the AGN duration.
  The counted numerator also requires a finite physical band-entry time.
- Retained-population sampling and the first-order equal stellar/remnant mass
  approximation remain deliberate model assumptions. Detailed remnant masses,
  natal kicks, isotropic angular resampling and replenishment were not silently
  added during these repairs.
- Physical stops, unsupported entry, profile failures, unavailable SNR and
  valid zero remain different outcomes.
- SI torque is converted once at the CGS evolution boundary. Paired inputs and
  seeds are required when comparing scientific treatments.

## Corrections and superseded conclusions

| Earlier statement or implementation | Current interpretation |
| --- | --- |
| Any increase in semi-major axis implies energy creation | Retracted in the September 21 export, lines 1843-1870. Gas can accelerate a sub-Keplerian body. The corrected node equations are tested with Cartesian impulses. |
| A generic eccentricity clamp proves damping correctness | A clamp can hide an equation error. Independent node-by-node value comparisons are required. |
| Classical Type-II plus several gap thresholds are production choices | Removed. Production has continuous Kanagawa depth, bounded Grishin-style thermal torque and the stated inertia-limited gap-drift interpolation. |
| A low-inclination orbit is captured at 8 degrees | IMRI uses local H/R and also checks eccentricity and radial containment. The TDE legacy comparison retains its older rule. |
| Outward torque always identifies a stable trap | A resolved stable zero is trap_bound; outward delivery blockage without one is outward_blocked. |
| Zero resolved traps proves the profile has no positive cells | False. Unresolved islands and outward-blocked paths can remain. Jump repair and physical-grid edges still need validation. |
| A checkpoint PASS or matching scalar/batch result establishes physical validity | It establishes the tested implementation behavior only. Regime applicability and convergence require additional evidence. |
| Default main runs have fixed seed 20260913 and automatic plots | This was a proposal. The live CLI uses optional --base-seed, default None; single-halo main saves data without automatically calling run_figures. |

## Later alignment and remaining limits

The October torque cleanup fixes the production composition and removes its
selector zoo while retaining the no-trap extreme as a diagnostic. The
2026-10-18 damping alignment adopts the newer root TDE step controller while
retaining IMRI's corrected equations, density, crossing speed and handoff.
The TDE main runner actually imports a separate pinned July snapshot; the
[comparison script](../../scripts/compare_damping_reference.py) tests both.

Three limits must remain visible:

1. Thermal linear validity and the Kanagawa calibration can fail for these
   low central masses. A finite computed torque is not a calibration result.
2. The two-cell WKB mask may leave neighboring spline ringing; padded pAGN
   arrays can extend past the physical Q=1 boundary. No cleanup claim resolves
   either issue. See [migration_model_alignment.md](../migration_model_alignment.md).
3. The historical repair plan requested no central merger, but the live
   detector continuation still enables its legacy single-merger model whenever
   cosmic_merge_time_myr is finite (default 1000 Myr). Its factors and time are
   explicit configuration. This source/intent difference is documented rather
   than changed by a documentation cleanup.

Older pre-transcript history is preserved in
[archive/design/history_reference.md](../archive/design/history_reference.md).
The archived repair and reorganization plans are not automatic authorization
for unimplemented physics or CLI changes.
