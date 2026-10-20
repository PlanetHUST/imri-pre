# 04 — Signed migration, gaps and traps

Current source guide, reviewed 2026-10-18. [Part index](README.md).

## Idea

The production torque is one mass-dependent signed profile. Its local sign and resolved zeros determine whether inward gas delivery succeeds.

## Source and data flow

| Owner | Responsibility |
| --- | --- |
| [physical_process/migration/torque.py](../../src/physical_process/migration/torque.py) | Builds gravitational, bounded thermal and inertia-limited gap-drift components in SI. |
| [physical_process/migration/gap.py](../../src/physical_process/migration/gap.py) | Owns continuous Kanagawa depth, Crida diagnostics and applicability flags. |
| [physical_process/migration/traps.py](../../src/physical_process/migration/traps.py) | Classifies stable, anti, boundary and unresolved zeros on the full profile. |
| [physical_process/migration/evolution.py](../../src/physical_process/migration/evolution.py) | Integrates signed torque in CGS and returns finite/trap/outward outcomes or an explicit error. |
| [physical_process/migration/diagnostics.py](../../src/physical_process/migration/diagnostics.py) | Supplies the diagnostic inward-only -abs(Gamma) extreme. |

## Boundaries

There is no runtime classical Type-II, ungapped, semi-without-trap or gap-threshold selector. Crida is diagnostic. A trap must be a resolved stable zero; outward blockage without one remains outward_blocked. Thermal/calibration flags, incomplete WKB ringing repair and the padded disk tail limit interpretation. Read the dedicated migration contract before interpreting a trap count.

## Contracts and evidence

- [Current physical model](../physics_model.md)
- [Fixed migration contract](../migration_model_alignment.md)
- [Execution and verification](../engineering_contract.md)
- [Consolidated decisions](../reference/history.md)
- [Preserved pre-cleanup explanation](../archive/project_before_cleanup/04_gas_migration.md)
