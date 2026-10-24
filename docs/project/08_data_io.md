# 08 — Schemas, metadata and persistence

Current source guide, reviewed 2026-10-18. [Part index](README.md).

## Idea

Persistence records the physical outcomes already computed by the stage owners.

## Source and data flow

| Owner | Responsibility |
| --- | --- |
| [workflows/models.py](../../src/workflows/models.py) | Owns typed boundary records and per-realization/averaged result containers. |
| [workflows/outputs.py](../../src/workflows/outputs.py) | Owns CSV schemas, unit/display translations, rate completeness and strict diagnostic JSON. |
| [workflows/metadata.py](../../src/workflows/metadata.py) | Records resolved configuration and the fixed migration contract. |
| [workflows/single_halo.py](../../src/workflows/single_halo.py) | Returns a typed host-execution result without turning a failed run into a physical zero. |
| [workflows/tree_batch.py](../../src/workflows/tree_batch.py) | Writes host selection, job summaries, source/input provenance and per-job output namespaces. |

## Boundaries

Canonical local files are imri_events.csv, imri_repeat_results.csv, imri_averaged_results.csv, imri_replenishment_ledger.csv, simulation_metadata.json and intermediate/migration_*.json. The replenishment ledger links each in-episode ISCO consumption to one donor injection or source exhaustion; repeat and averaged rows carry finite source balances. A successful job can still have unavailable rates; inspect rate_status and detector_rate_status. Keep null/unavailable, valid zero, numerical failure and physical stopping distinct. Read schemas from source rather than annotation order or old sample CSVs.

## Contracts and evidence

- [Current physical model](../physics_model.md)
- [Fixed migration contract](../migration_model_alignment.md)
- [Execution and verification](../engineering_contract.md)
- [Consolidated decisions](../reference/history.md)
- [Preserved pre-cleanup explanation](../archive/project_before_cleanup/08_data_io.md)
