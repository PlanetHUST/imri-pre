# 03 — Damping and embedding

Current source guide, reviewed 2026-10-18. [Part index](README.md).

## Idea

A disk-crossing orbit is evolved under gas drag before it is eligible for circular migration.

## Source and data flow

| Owner | Responsibility |
| --- | --- |
| [physical_process/damp.py](../../src/physical_process/damp.py) | Owns OrbitalState, active-node timescales, per-orbit a/e/i changes, the adaptive loop and DampingOutcome. |
| [workflows/workers.py](../../src/workflows/workers.py) | Creates a calculator per worker and serializes the complete final state and status. |
| [physical_process/nsc_orbital.py](../../src/physical_process/nsc_orbital.py) | Owns the two node radii and membership predicate. |

## Boundaries

Only active nodes interact. IMRI reads solved density and uses transverse speed times sin(i) for crossing time. Integration stops at local i<=H/R; handoff additionally requires e<=0.01 and radial containment. Low inclination with remaining eccentricity is damping_incomplete, not instantaneous circularization.

The newer root-level cos_tde_revised/damp.py supplies the aligned adaptive method: rejected-trial caching, no minimum fraction floor and accepted-step iteration counting. Non-finite trials, a stalled clock and positive step-budget exhaustion are calculation_error. AGN shutdown is physical incompletion. The TDE main runner uses a separate pinned July snapshot. The comparison script tests both without changing either reference.

The node equations are independently checked by Cartesian drag impulses in tests/test_damping_equations.py; agreement with legacy code is not their physical justification.

## Contracts and evidence

- [Current physical model](../physics_model.md)
- [Fixed migration contract](../migration_model_alignment.md)
- [Execution and verification](../engineering_contract.md)
- [Consolidated decisions](../reference/history.md)
- [Preserved pre-cleanup explanation](../archive/project_before_cleanup/03_disk_embedding_dynamics.md)
