# Project Reconstruction — Index

This directory reconstructs the main idea of the IMRI rate-computation project
from the live source in `src/` and `scripts/` plus the durable contracts in
`docs/reference/`. It is a *conceptual* reconstruction: it explains what each
part is for, how the code carries it out, and where the sharp edges are. It is
not a task contract; current specifications and plans continue to govern
individual changes. See [`../README.md`](../README.md) for document authority.

Start with **`overview.md`** — the macroscopic statement of the project, the
end-to-end pipeline, and how the parts connect.

## Part documents

| # | Document | Layer | Primary code owners |
|---|---|---|---|
| — | [`overview.md`](overview.md) | macroscopic main idea | — |
| 01 | [`01_environment_build.md`](01_environment_build.md) | AGN disk + NSC + torque environment | `engine.py`, `pagn_disk_adapter.py`, `migration_torque.py`, `agn.py`, `nsc_orbital.py` |
| 02 | [`02_population_sampling.md`](02_population_sampling.md) | stellar population + orbital sampling + disk-crossing selection | `sampling.py`, `nsc_orbital.py`, `population.py` |
| 03 | [`03_disk_embedding_dynamics.md`](03_disk_embedding_dynamics.md) | damping / orbit evolution / embedded-state contract | `damp.py`, `workers.py` |
| 04 | [`04_gas_migration.md`](04_gas_migration.md) | torque & semi-analytic gas migration, traps | `disk_migration.py`, `event_work.py` |
| 05 | [`05_gw_event_classification.md`](05_gw_event_classification.md) | LISA band, band routing, gas/GW transport, candidates | `gw_frequency_radius.py`, `event_rate.py`, `event_work.py` |
| 06 | [`06_cosmology_detectability.md`](06_cosmology_detectability.md) | redshift evolution, cosmic merger, physical LISA SNR | `redshift_evolve.py`, `cosmic_merge.py`, `lisa_snr.py`, `detect_rate.py` |
| 07 | [`07_workflow_orchestration.md`](07_workflow_orchestration.md) | config/CLI, grid driver, single realization, workers, seeds | `config.py`, `cli.py`, `workflow.py`, `single_work.py`, `workers.py`, `redshift_scan.py` |
| 08 | [`08_data_io.md`](08_data_io.md) | persistence, units, strict JSON, diagnostics, metadata | `models.py`, `outputs.py`, `metadata.py` |
| 09 | [`09_analysis_visualization.md`](09_analysis_visualization.md) | rate figures, status taxonomy, paired comparison | `rate_plots.py`, `final_analysis.py`, `redshift_scan.py` |
| 10 | [`10_validation_diagnostics.md`](10_validation_diagnostics.md) | deterministic harnesses, invariants, parity, verification | `scripts/*_test.py`, `migration_test.py`, `trap_plot.py`, `disk_plot.py`, `damp_test.py` |

## Source layout map

```
scripts/                    entry points + diagnostic harnesses (self-inject src/)
  main.py                   → cli.run_cli (single configured run)
  redshift_scan.py          → paired disk-model × redshift scan
  final_analysis.py         → regenerate figures from persisted results
  migration_test.py         → migration timescale comparison
  trap_plot.py              → trap/anti-trap radii
  disk_plot.py              → disk method comparison + torque plots
  damp_test.py              → damping diagnostics
  imri_rate_visualization_test.py → deterministic figure harness

src/physical_process/       physics (imported as top-level physical_process.*)
  constants.py              cgs constants
  agn.py                    AGNProperties
  sampling.py               IMF + lifetime sampling
  nsc_orbital.py            orbital sampling, node radii, NSC radius, relaxation
  damp.py                   orbital damping + embedded contract
  disk_migration.py         gap mask, torque composition, migration outcomes
  gw_frequency_radius.py    GW frequency/radius, Peters, LISA band, ISCO preflight

src/disk_quantity/          disk structure + torques
  pagn_disk_adapter.py      Sirko/Thompson pAGN → CGS profiles, Q=1 outer radius
  migration_torque.py       TorqueProfile construction, SI torque, trap finder
  disk_solve.py             legacy in-repo solver (comparison/fallback)
  opacity.py                opacity tables (deliberately excluded from pyright)

src/workflows/              orchestration + analysis + persistence
  config.py, cli.py         configuration
  engine.py                 cached disk/torque resources
  population.py             sampling orchestration
  workers.py                damping worker pool
  single_work.py            one realization
  workflow.py               grid driver + seed derivation
  event_work.py             event evaluator / band routing
  event_rate.py             gas/GW transport + candidate classification
  lisa_snr.py               physical LISA sensitivity, waveform, and SNR kernel
  detect_rate.py            detector classification
  redshift_evolve.py        cosmology model
  cosmic_merge.py           simplified merger model
  models.py                 typed records / result containers
  outputs.py, metadata.py   persistence
  rate_plots.py             figures + compact outcome data
  reporting.py, runtime.py  terminal output
```

## How to read

1. `overview.md` for the big picture;
2. the part document for the layer you are working on;
3. the relevant specification, plan, or `docs/reference/` contract for exact
   required behavior.

Each part document follows the same structure: **Idea** (what it is for),
**Doing logic** (how the code does it), and **Caveats** (boundaries and
recurring pitfalls). Where a current specification conflicts with this
reconstruction, the specification governs.
