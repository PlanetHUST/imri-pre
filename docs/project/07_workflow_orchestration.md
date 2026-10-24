# 07 — Entry points, caches and workers

Current source guide, reviewed 2026-10-18. [Part index](README.md).

## Idea

Both single-halo and tree execution call the same local physics kernel.

## Source and data flow

| Owner | Responsibility |
| --- | --- |
| [workflows/cli.py](../../src/workflows/cli.py) | Parses positional repeats/output/damping workers and current scientific options; --tree enables batch mode. |
| [workflows/workflow.py](../../src/workflows/workflow.py) | Runs the mass grid, derives optional per-repeat seeds, persists results and averages repeats. |
| [workflows/single_work.py](../../src/workflows/single_work.py) | Coordinates one sampled realization through disk, population, damping and event stages. |
| [workflows/replenishment.py](../../src/workflows/replenishment.py) | Orders completed in-episode ISCO events and supplies one distinct donor per vacated channel. |
| [workflows/outputs.py](../../src/workflows/outputs.py) | Persists per-object events, repeat summaries and the consumption/injection ledger. |
| [workflows/engine.py](../../src/workflows/engine.py) | Caches disk and mass-dependent torque resources. |
| [workflows/tree_batch.py](../../src/workflows/tree_batch.py) | Streams host jobs, derives host seeds, runs the outer pool and writes provenance. |

## Boundaries

The single-halo flag is --base-seed with default None. Explicit task seeds determine repeat draws and the independent replenishment donor stream; host seed identity is stable across worker order. `--no-consumption-replenishment` disables replacement injections. No fixed default main seed or --plots flag is present. The kernel writes data without automatically invoking run_figures. Batch mode defaults an unspecified inner damping pool to one to limit nested parallelism.

## Contracts and evidence

- [Current physical model](../physics_model.md)
- [Fixed migration contract](../migration_model_alignment.md)
- [Execution and verification](../engineering_contract.md)
- [Consolidated decisions](../reference/history.md)
- [Preserved pre-cleanup explanation](../archive/project_before_cleanup/07_workflow_orchestration.md)
