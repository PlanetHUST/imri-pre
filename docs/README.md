# Documentation map

Reviewed against the working source on 2026-10-18. Start with the root
[README](../README.md) for commands and this page for document ownership.

## Current guides

| Question | Read | Source owner |
| --- | --- | --- |
| What does one host simulation calculate? | [Physical model](physics_model.md) | [workflow.py](../src/workflows/workflow.py), [single_work.py](../src/workflows/single_work.py) |
| How are damping, gaps and traps defined? | [Physical model](physics_model.md), [migration contract](migration_model_alignment.md) | [damp.py](../src/physical_process/damp.py), [migration/](../src/physical_process/migration/) |
| When is a damped orbit replenished? | [Consumption replenishment plan](plans/active/imri_consumption_replenishment.md), [physical model](physics_model.md) | [replenishment.py](../src/workflows/replenishment.py), [event_work.py](../src/workflows/event_work.py) |
| What are the CLI, units, outputs and checks? | [Engineering contract](engineering_contract.md) | [cli.py](../src/workflows/cli.py), [outputs.py](../src/workflows/outputs.py) |
| How are tree hosts mapped and seeded? | [Merger-tree interface](merger_tree_interface.md) | [host_mapping.py](../src/workflows/host_mapping.py), [tree_batch.py](../src/workflows/tree_batch.py) |
| Where is each stage implemented? | [Part-by-part source map](project/README.md) | Current source |
| Why did the design change? | [Decision history](reference/history.md) | [Archived transcripts](archive/README.md) |
| What remains outside the local model? | [Research roadmap](reference/research_roadmap.md) | Future work, not implemented behavior |

## Authority and history

The current user instruction governs the task. Implemented specifications
apply within their stated subsystem; current source and schemas determine
what the program actually does. A difference between an old plan and source
must be stated explicitly, not silently resolved by rewriting the physics.

[The LISA specification](specs/lisa_detectability.md) records the detector
contract. [Its completed plan](plans/active/lisa_detectability_plan.md) is
implementation history. Old repair/reorganization proposals and checkpoint
reports are historical; their former selectors, filenames and numerical
results are not current defaults.

The root session exports have been preserved byte-for-byte under
[archive/transcripts/](archive/transcripts/). Their durable decisions and
later corrections are combined in [reference/history.md](reference/history.md).
They are reference material, not executable instructions.

## Commands and figures

The normal entry is [scripts/main.py](../scripts/main.py), which delegates to
[workflows.cli.run_cli](../src/workflows/cli.py). The live single-halo parser
uses positional repeat/output/worker arguments and the optional
--base-seed flag. Omitting that seed does not promise reproducibility.

The main workflow saves CSV/JSON. It does not automatically call the
[saved-result figure layer](../src/workflows/run_figures.py). Tree mode can
produce its batch-interface figures; the
[redshift scan](../scripts/run_redshift_scan.py) generates scan figures.
There are no --plots/--no-plots or torque/gap selection flags in the main CLI.

See [engineering_contract.md](engineering_contract.md) for actual diagnostic
commands. Use --help to inspect an entry point before launching a large run.
