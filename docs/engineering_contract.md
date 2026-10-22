# Execution, ownership and verification contract

Reviewed against the current files on 2026-10-22. This guide describes present
behavior; archived plans record requested or earlier behavior.

## Execution path

    scripts/main.py -> workflows.cli.run_cli
      single-halo: workflows.workflow.run_simulation
        -> IMRISingleWorkflow.run_single_simulation
        -> cached pAGN disk -> population -> damping workers
        -> IMRIEventEvaluator -> CSV/JSON persistence
      --tree: workflows.tree_batch.run_tree_batch
        -> reader -> host mapping -> run_single_halo -> same kernel

Each grid point runs num_repeats and averages its own realizations. A tree job
is a full local run for one host, not a single sampled object. The outer job
pool and inner damping pool are distinct; batch mode defaults the inner pool
to one when it was not explicitly set.

## Scientific ownership

| Part | Authoritative files |
| --- | --- |
| Disk, units and physical outer boundary | [pagn_disk_adapter.py](../src/disk_quantity/pagn_disk_adapter.py) |
| Population and orbital geometry | [sampling.py](../src/physical_process/sampling.py), [nsc_orbital.py](../src/physical_process/nsc_orbital.py) |
| Damping and typed outcome | [damp.py](../src/physical_process/damp.py) |
| Signed torque, gap diagnostics, trap classification, integration | [migration/](../src/physical_process/migration/) |
| Coupled circular gas/GW orbit | [orbital_evolution.py](../src/physical_process/orbital_evolution.py) |
| Intrinsic first-entry candidacy | [event_rate.py](../src/workflows/event_rate.py) |
| Per-object routing and final event assembly | [event_work.py](../src/workflows/event_work.py) |
| LISA PSD, vacuum waveform/SNR, and first-order coupled-track SNR | [lisa/snr.py](../src/lisa/snr.py) |
| Supplied-entry detector decision | [lisa/detectability.py](../src/lisa/detectability.py) |
| Schema, persistence and metadata | [models.py](../src/workflows/models.py), [outputs.py](../src/workflows/outputs.py), [metadata.py](../src/workflows/metadata.py) |
| Figures and saved-result analysis | [viz/](../src/viz/), [run_figures.py](../src/workflows/run_figures.py) |

Scripts and tests consume these owners. No diagnostic should copy the full
production model and then claim its agreement as independent validation.

## Units and states

The local workflow uses grams, centimeters, seconds and explicitly named
Myr/Msun interfaces. pAGN solves in SI; its adapter converts disk quantities
to CGS. Torque composition stays SI and converts once at the migration-time
consumer. The LISA kernel has its own explicit SI boundary; frequencies passed
to its public interface are observer-frame Hz and masses are source-frame.
Cosmic age and luminosity distance share the configured flat-LambdaCDM H0 and
Omega_m. Source-frame elapsed Myr begin at the host's specified redshift and
the continuation ends at z=0; no independent Hubble-time setting remains.

Use descriptive statuses rather than numeric sentinels. Preserve physical
zero, unavailable, unsupported regime, numerical error, outward blockage and
resolved trap as different states. JSON is strict: non-finite diagnostics are
nullable with a status, never bare NaN/Infinity.

A successful host execution can still have unavailable intrinsic or detector
rates. Read rate_status and detector_rate_status as well as the job status.
A failed host must not be represented as a physical zero.

## Reproducibility

The single-halo flag is --base-seed; its default is None. If supplied,
workflows.workflow.derive_task_seed uses
SeedSequence([base_seed, smbh_index, nsc_index, repeat_index]) and seeds the
global NumPy stream once per repeat. Redshift is excluded from that identity.
Tree mode derives a host base seed from master seed and stable tree identity,
then uses the same per-repeat rule. Match all other inputs for comparisons.

The main CLI uses the signed torque/trap prescription. The comparison-only
`semi_without_trap` branch uses its existing inward Type-I/II timescale law.
Both now feed one coupled orbit interface and must be compared with identical
seed and upstream object ledger. The older sign-reflection no-trap diagnostic
is not a selectable production model. See
[migration_model_alignment.md](migration_model_alignment.md).

## CLI and artifacts

From the repository root, after installing dependencies:

```bash
PYTHONPATH=src python scripts/main.py 1 output/example 1 \
  --smbh-masses 1e5 --nsc-masses 5e4 --base-seed 777
python scripts/main.py --help
```

The main kernel saves:

| File | Content |
| --- | --- |
| imri_averaged_results.csv | Per-grid-point averages and rate validity |
| imri_repeat_results.csv | Per-repeat seeds, counts, status partitions and rates |
| imri_events.csv | Per-object intrinsic and detector outcomes |
| simulation_metadata.json | Resolved parameters, migration contract and runtime |
| intermediate/migration_*.json | Per-repeat migration diagnostics |

Tree output adds config/, hosts/, jobs/, simulations/, diagnostics/, optional
batch figures/ and summary.json. The current single-halo main workflow saves
data without automatically generating run figures. Earlier plans for default
plots and a fixed default seed are not implemented by this CLI.

## Diagnostics

| Command | What it checks or produces |
| --- | --- |
| PYTHONPATH=src python tests/run_tests.py | Automated regression and equation checks |
| python scripts/run_science_checkpoints.py --only C | Controlled damping checkpoint |
| python scripts/run_science_checkpoints.py --only H | Seeded stage integration |
| python scripts/run_interface_checkpoints.py | Tree interface, mapping, failure isolation and reproducibility |
| python scripts/validate_damping_model.py --output-dir output/diagnostics/damping | Boundary cases, scans, trajectories and figures |
| python scripts/compare_damping_reference.py --output-dir output/diagnostics/damping_reference | Matched inputs against both TDE sources |
| python scripts/validate_migration_contract.py --output-csv output/diagnostics/migration.csv | Fixed torque composition, validity and signed/no-trap comparison |
| python scripts/validate_gas_gw_handoff.py --help | Gas/GW routing diagnostic options |
| python scripts/validate_coupled_orbit.py /tmp/imri-coupled-check | Seed-paired reduced runs and upstream-ledger reconciliation for both scenarios |
| python scripts/validate_lisa_detectability.py --help | LISA detector diagnostic options |
| python scripts/plot_disk_structure.py | Generates the configured torque/disk plots under figs/; constants are in the script and there is no argument parser |
| python scripts/analyse_population_torque.py --help | Saved-population torque regime analysis |
| python scripts/report_event_summary.py --help | Saved event summary options |
| python scripts/compare_baseline.py --help | Scientific artifact comparison |

Set MPLCONFIGDIR=/tmp/matplotlib in restricted environments. The working
interpreter used for recent checks is
/home/planet/miniconda3/envs/sevnpy_env/bin/python.
Checkpoint reports belong to the exact source/inputs that produced them; a
historical PASS is not a current baseline. Full diagnostic sweeps can be more
expensive than unit tests. Start with the affected stage.

For a documentation-only cleanup, check local links, CLI examples, archived
file hashes, source invariance and git diff --check. Scientific verification
requires the relevant physical tests when the implementation changes.
