# 01 — Environment and disk

Current source guide, reviewed 2026-10-18. [Part index](README.md).

## Idea

The production environment is a solved pAGN disk around the configured central object. The adapter returns CGS profiles and a physical outer boundary; an array endpoint is not automatically that boundary.

## Source and data flow

| Owner | Responsibility |
| --- | --- |
| [disk_quantity/pagn_disk_adapter.py](../../src/disk_quantity/pagn_disk_adapter.py) | Builds Sirko/Thompson profiles, converts SI to CGS, finds the first outward Q=1 edge and stores the disk object. |
| [disk_quantity/opacity.py](../../src/disk_quantity/opacity.py) | Owns the in-repository opacity tables/interpolation. |
| [disk_quantity/disk_solve.py](../../src/disk_quantity/disk_solve.py) | Independent disk solver for comparisons; not a production fallback. |
| [workflows/engine.py](../../src/workflows/engine.py) | Caches one disk per central mass and torque profiles per physical mass/alpha identity. |

## Boundaries

Disk height, density, surface density and radius must come from the same solved profile. pAGN arrays can include a padded tail beyond Q=1. Keep the physical edge distinct from the numerical domain.

## Contracts and evidence

- [Current physical model](../physics_model.md)
- [Fixed migration contract](../migration_model_alignment.md)
- [Execution and verification](../engineering_contract.md)
- [Consolidated decisions](../reference/history.md)
- [Preserved pre-cleanup explanation](../archive/project_before_cleanup/01_environment_build.md)
