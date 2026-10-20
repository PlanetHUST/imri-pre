# 09 — Saved-result analysis and figures

Current source guide, reviewed 2026-10-18. [Part index](README.md).

## Idea

Presentation consumes calculated or persisted scientific results and must not redefine intrinsic candidacy or detector selection.

## Source and data flow

| Owner | Responsibility |
| --- | --- |
| [workflows/run_figures.py](../../src/workflows/run_figures.py) | Provides saved-result run figure functions; single-halo main does not call them automatically. |
| [viz/rate_plots.py](../../src/viz/rate_plots.py) | Loads run tables, maps status categories and produces rate/scan figures and plotted data. |
| [viz/disk_plots.py](../../src/viz/disk_plots.py) | Plots disk profiles. |
| [viz/batch_plots.py](../../src/viz/batch_plots.py) | Produces merger-tree interface figures. |
| [viz/opacity_plots.py](../../src/viz/opacity_plots.py) | Plots opacity diagnostics. |

## Boundaries

The maintained scan entry is scripts/run_redshift_scan.py and it delegates each point to run_simulation. The maintained summary entry is scripts/report_event_summary.py. Removed final_analysis.py/redshift_scan.py command names in old documents are historical. Preserve source units and raw status fields beneath presentation labels. A plotted strain crossing is not the integrated SNR selection.

## Contracts and evidence

- [Current physical model](../physics_model.md)
- [Fixed migration contract](../migration_model_alignment.md)
- [Execution and verification](../engineering_contract.md)
- [Consolidated decisions](../reference/history.md)
- [Preserved pre-cleanup explanation](../archive/project_before_cleanup/09_analysis_visualization.md)
