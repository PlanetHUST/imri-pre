# Part-by-part source map

Reviewed against the live files on 2026-10-18. These are explanatory guides;
[physics_model.md](../physics_model.md) and the actual source determine current
behavior. Earlier generated explanations are preserved in the
[archive](../archive/project_before_cleanup/).

| Part | Guide | Main boundary |
| --- | --- | --- |
| 01 | [Environment and disk](01_environment_build.md) | disk_quantity/pagn_disk_adapter.py |
| 02 | [Population, orbits, crossing and finite NSC supply](02_population_sampling.md) | physical_process/sampling.py, workflows/replenishment.py |
| 03 | [Damping and embedding](03_disk_embedding_dynamics.md) | physical_process/damp.py |
| 04 | [Signed migration, gaps and traps](04_gas_migration.md) | physical_process/migration/torque.py |
| 05 | [Gas/GW routing and intrinsic events](05_gw_event_classification.md) | physical_process/gw_frequency_radius.py |
| 06 | [Post-gas evolution and LISA](06_cosmology_detectability.md) | lisa/snr.py |
| 07 | [Entry points, caches and workers](07_workflow_orchestration.md) | workflows/cli.py |
| 08 | [Schemas, metadata and persistence](08_data_io.md) | workflows/models.py |
| 09 | [Saved-result analysis and figures](09_analysis_visualization.md) | workflows/run_figures.py |
| 10 | [Verification and evidence](10_validation_diagnostics.md) | ../tests/run_tests.py |

Start with [overview.md](overview.md), then follow the pipeline order. The
current script list and reproducible commands are in
[the engineering contract](../engineering_contract.md). The normal entry is
[scripts/main.py](../../scripts/main.py); the redshift driver is
[scripts/run_redshift_scan.py](../../scripts/run_redshift_scan.py).
