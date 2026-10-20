# 10 — Verification and evidence

Current source guide, reviewed 2026-10-18. [Part index](README.md).

## Idea

A diagnostic calls the production owner on controlled inputs and checks independent expectations or explicitly stated differences.

## Source and data flow

| Owner | Responsibility |
| --- | --- |
| [../tests/run_tests.py](../../tests/run_tests.py) | Runs automated tests without requiring pytest. |
| [../scripts/run_science_checkpoints.py](../../scripts/run_science_checkpoints.py) | Provides the physical stage and reduced integration checkpoints. |
| [../scripts/run_interface_checkpoints.py](../../scripts/run_interface_checkpoints.py) | Checks tree parsing, mapping, reuse, failure isolation and reproducibility. |
| [../scripts/compare_damping_reference.py](../../scripts/compare_damping_reference.py) | Compares shared inputs with both TDE damping sources and isolates controller parity. |
| [../scripts/validate_migration_contract.py](../../scripts/validate_migration_contract.py) | Checks composition and reports validity, trap/no-trap and transport diagnostics. |

## Boundaries

Start with focused equation, boundary and status checks, then affected-stage diagnostics and a reduced seeded run. Compilation and finite arrays do not establish model applicability. Historical checkpoint PASS/WARN values are evidence for their snapshot only. The full damping scan and full population grid were not rerun merely to reorganize documentation.

## Contracts and evidence

- [Current physical model](../physics_model.md)
- [Fixed migration contract](../migration_model_alignment.md)
- [Execution and verification](../engineering_contract.md)
- [Consolidated decisions](../reference/history.md)
- [Preserved pre-cleanup explanation](../archive/project_before_cleanup/10_validation_diagnostics.md)
