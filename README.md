# IMRI in nuclear star clusters and AGN disks

Simulates the formation and LISA detectability of **intermediate-mass-ratio
inspirals (IMRIs)**: stellar-mass compact objects in a nuclear star cluster
(NSC) that cross an AGN disk, are damped into it, migrate through it, and
inspiral into the central massive black hole.

The project answers two questions:

1. **one halo** — for a given host (redshift, SMBH mass, NSC mass,
   metallicity), how many IMRI candidates form and how many would LISA see?
2. **many halos** — given a merger-tree CSV of hosts, run that simulation for
   each selected host, reproducibly and in parallel.

It deliberately stops at host-level results. **No cosmic rate is computed.**

---

## What one simulation means

One *single-halo run* is one complete Monte Carlo experiment for one host:

```
sample an NSC population  →  select disk-crossing orbits  →  damp them into the disk
   →  migrate the embedded objects  →  resolve gas vs GW competition
   →  classify IMRI candidates  →  evaluate LISA SNR and detectability
```

It runs the configured number of Monte Carlo repeats internally and reports
their average. One merger-tree *job* is exactly one such run.

---

## The physical pipeline

```
host  →  AGN disk  →  compact-object population  →  disk crossing  →  damping
      →  migration (torque → gap → traps → radial evolution)
      →  gas/GW handoff  →  IMRI event classification  →  LISA SNR → detectability
```

Each arrow is owned by exactly one module. See
[`docs/physics_model.md`](docs/physics_model.md) for the authoritative
module-ownership table and the assumptions behind each stage.

> **Migration contract.** Production uses one signed torque:
> `Gamma = f_gap Gamma_grav + Gamma_thermal`, with continuous Kanagawa depth
> `f_gap = 1/(1+0.04K)` and `K=q²/(alpha h⁵)`. The thermal term follows the
> local Grishin-2024 prescription on the pAGN disk. There is no classical
> Type-II branch or runtime gap/model selector. Regime-limit flags are saved as
> diagnostics rather than silently switching prescriptions. See
> [`docs/physics_model.md`](docs/physics_model.md) §3.
> The validation diagnostic also reports a clearly labelled no-trap extreme,
> `Gamma_no-trap=-|Gamma|`, alongside the signed trap-aware outcome; it is not
> used by production workflows.

---

## Repository layout

```
src/                      scientific truth — reusable, testable, no plotting
├── physical_process/     constants, AGN, IMF, orbits, damping, GW, redshift
│   └── migration/        torque.py · gap.py · traps.py · evolution.py
├── disk_quantity/        opacity tables, pAGN disk adapter, comparison solver
├── lisa/                 snr.py · detectability.py   (detector physics only)
├── workflows/            orchestration: config, engine, event pipeline, CLI,
│                         merger-tree reader, host mapping, batch executor
└── viz/                  every figure — consumes src/, defines no physics

scripts/                  thin user-facing entry points (see below)
tests/                    automated correctness assertions
docs/                     physics model and merger-tree interface
output/                   generated results, checkpoints and figures
galaxy_1e12.csv           supplied merger-tree input (read-only)
lisa_ref/                 supplied LISA reference code (not used in production)
```

The rule is one line long: **`src/` is scientific truth, everything else
consumes it.** No script, test or figure contains a unique physical
prescription; `tests/test_architecture.py` enforces this.

---

## Running it

### One halo

```bash
python scripts/main.py 2 output/dyn_results 4 \
    --smbh-masses 1e5 --nsc-masses 5e4 --base-seed 777
```

Positional arguments are `num_repeats`, `output_dir`, `damping_cpu_count`.

### A merger-tree batch

```bash
python scripts/main.py \
    --tree galaxy_1e12.csv \
    --output output/tree_run \
    --workers 16 \
    --master-seed 2026 \
    --z-min 6.005 --z-max 6.03 --node 40,80,120 --max-hosts 12 \
    --smbh-mass-source stellar-scaling \
    --nsc-mass-source stellar-scaling
```

Every single-halo option above becomes the *base* configuration; the tree
supplies the host-dependent parameters. Host selectors: `--node`, `--branch`,
`--z-min/--z-max`, `--mbh-min/--mbh-max`, `--mhalo-min/--mhalo-max`,
`--max-hosts`.

> **Note on the supplied CSV:** its `MBH` column is identically zero, so the
> default `--smbh-mass-source tree` rejects every host with an explicit error.
> Choose `config` or `stellar-scaling` to run it. See
> [`docs/merger_tree_interface.md`](docs/merger_tree_interface.md).

### Programmatically, without the CLI

```python
from workflows import IMRIDynConfig, HaloHost, run_single_halo

result = run_single_halo(config=base_config, host=host, seed=seed,
                         output_dir="output/one_host")
print(result.status, result.summary["n_imri"])
```

---

## Deterministic seeds

```
master seed  →  host base seed = SeedSequence([master, node, branch, record_index])
             →  per-repeat task seed = derive_task_seed(base_seed, 0, 0, repeat)
```

Seeds derive only from stable identities, never from process order or Python's
`hash`. Consequently **results do not depend on the worker count**: Checkpoint K
verifies that `--workers 1` and `--workers 4` produce byte-identical per-host
results.

---

## Where outputs go

| Path | Contents |
|---|---|
| `<output_dir>/imri_averaged_results.csv` | one row per grid point, averaged over repeats |
| `<output_dir>/imri_repeat_results.csv` | one row per Monte Carlo repeat, with its seed |
| `<output_dir>/imri_events.csv` | one row per evaluated object, with its classification reason |
| `<output_dir>/simulation_metadata.json` | resolved configuration and model identifiers |
| `<output_dir>/intermediate/` | per-repeat migration diagnostics |

A merger-tree batch adds `config/` (resolved config + provenance), `hosts/`,
`jobs/job_summary.csv`, `simulations/<job_id>/`, `diagnostics/`, `figures/`
and `summary.json`.

**Reading a zero correctly:** a job with `status = success` and `n_imri = 0` is
a *physical zero*. Any other status means the model could not be evaluated, and
its physical columns are `NaN`, never `0`.

---

## Validation

```bash
python tests/run_tests.py                    # automated correctness (no pytest needed)
python scripts/run_science_checkpoints.py    # scientific checkpoints B–L
python scripts/run_interface_checkpoints.py  # merger-tree interface checkpoints A–H
```

The four scripts most worth running first:

| Script | Question it answers |
|---|---|
| `tests/run_tests.py` | Do the canonical modules still behave correctly? |
| `scripts/run_science_checkpoints.py` | Is each physical subsystem sane, and does the end-to-end pipeline still work? |
| `scripts/audit_repository.py` | Has duplicated or script-local physics crept back in? |
| `scripts/compare_baseline.py` | Did a refactor change any scientific output? |

`compare_baseline.py capture <run> <file>` then `check <run> <file>` compares
runs while ignoring timings and paths.

---

## Scripts

| Script | Purpose |
|---|---|
| `main.py` | CLI entry point and programmatic API |
| `run_science_checkpoints.py` | Round-2 scientific checkpoints B–L |
| `run_interface_checkpoints.py` | merger-tree interface checkpoints A–H |
| `run_redshift_scan.py` | fixed-contract redshift scan driver |
| `audit_repository.py` | dependency / duplicated-physics audit |
| `compare_baseline.py` | scientific-output regression comparison |
| `validate_damping_model.py` | damping invariants, scans and phase diagrams |
| `validate_gas_gw_handoff.py` | gas/GW handoff regression checks |
| `validate_lisa_detectability.py` | LISA PSD, cosmology, SNR and classification checks |
| `validate_migration_contract.py` | fixed torque contract, validity flags, and trap/no-trap bounds |
| `plot_disk_structure.py` | disk profiles and pAGN-vs-in-repo comparison |
| `analyse_population_torque.py` | population occupancy of torque validity regimes and traps |
| `report_event_summary.py` | print an event summary from a results directory |

---

## Environment

Python ≥ 3.10 with `numpy`, `scipy`, `pandas`, `matplotlib` and `pagn`.
Install for development with `pip install -e .`; the package puts `src/` on the
path, so `physical_process`, `disk_quantity`, `lisa`, `viz` and `workflows` are
importable directly.
