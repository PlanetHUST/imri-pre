# Merger-tree interface and parallel single-halo batch execution

This document describes the interface added between a large external
merger-tree CSV and the repository's existing single-halo IMRI simulation.

The existing local physics is unchanged. The new layer only decides *which
host parameters* each single-halo run receives, *how runs are identified and
seeded*, and *where their results are written*.

```
galaxy_1e12.csv ──► MergerTreeReader ──► HaloHost ──► HaloSimulationJob
                       (usecols,          (compact,      (job_id, seed,
                        chunked)           typed)         output_dir)
                                                              │
                                              resolve_host_parameters
                                                              │
                                                     run_single_halo
                                                              │
                                              run_simulation (unchanged)
                                                              │
                                        output/<run>/simulations/<job_id>/
```

The pipeline stops at host-level single-halo results. No cosmic rate, comoving
density, `dV_c/dz` weighting or all-sky integration is computed anywhere.

---

## 1. Original execution architecture

| Module | Original role |
|---|---|
| `scripts/main.py` | path setup, then `run_cli()` |
| `src/workflows/cli.py` | argparse → one `IMRIDynConfig` → `run_simulation(config)` |
| `src/workflows/workflow.py` | `run_simulation`: loops the SMBH × NSC mass grid × `num_repeats`, seeds NumPy globally via `derive_task_seed`, averages and saves |
| `src/workflows/single_work.py` | `IMRISingleWorkflow.run_single_simulation(m_smbh, m_nsc, repeat_idx, task_seed)` — **one Monte Carlo realization** |
| `src/workflows/engine.py` | cached pAGN disk and torque profiles per SMBH mass |
| `src/workflows/workers.py` | `multiprocessing.Pool` over per-object orbital damping tasks |
| `src/workflows/outputs.py` | `imri_averaged_results.csv`, `imri_events.csv`, `imri_repeat_results.csv`, `simulation_metadata.json`, per-repeat migration diagnostics |

Key properties established by inspection before any change:

* **Randomness.** The physics uses the *global* NumPy RNG (`np.random.rand`,
  `np.random.random`, `np.random.uniform` in `sampling.py` and
  `nsc_orbital.py`). `run_simulation` calls `np.random.seed(task_seed)` once
  per repeat, where
  `task_seed = SeedSequence([base_seed, smbh_index, nsc_index, repeat_index])`.
  Results are therefore reproducible **within one process** and were verified
  to be identical across repeated runs and across `damping_cpu_count` values.
* **Existing parallelism.** Only the *inner* damping loop is parallel. It is
  order-independent: results come back through `imap_unordered` and are
  immediately re-sorted by `idx`. Nothing in the physics depends on completion
  order.
* **Global state.** `IMRIDynConfig` is a plain object constructed per run and
  passed explicitly; the only true module-level mutable state is
  `workflows.runtime.VERBOSE` and the per-engine caches, neither of which is
  shared across processes.

---

## 2. Host-dependent versus global parameters

The kernel consumes exactly **four** quantities that can legitimately vary from
host to host:

| Local input | Where it lives | Effect |
|---|---|---|
| `redshift` | `IMRIDynConfig.redshift` | source/observer frequency conversion, LISA time budget, redshift evolution |
| `smbh_mass_msun` | one entry of `smbh_masses_msun` | AGN properties, pAGN disk profile, gravitational radius |
| `nsc_mass_msun` | one entry of `nsc_masses_msun` | sampled NSC population mass, NSC radius |
| `metallicity` | `IMRIDynConfig.metallicity` (solar units) | Pop II vs Pop III IMF branch |

Everything else in `IMRIDynConfig` — disk model, gap-opening condition,
migration switches, LISA detector model, duty cycle, repeat count, tolerances,
checkpoint interval — is a **global model parameter** and is inherited
unchanged from the base configuration. Nothing was moved out of `config.py`.

---

## 3. Merger-tree reader (`src/workflows/merger_tree.py`)

The only module that knows CSV column names.

### Required columns (7 of 30)

| Column | Verified meaning | Use |
|---|---|---|
| `z` | host redshift | → local `redshift`, selection |
| `m` | halo mass [Msun] (file `galaxy_1e12.csv` has `m ≈ 1e12`, matching its name) | selection and provenance only |
| `Mstar` | host stellar mass [Msun] | only when a `stellar-scaling` mapping is selected |
| `Z` | absolute metal **mass fraction** | only when `--metallicity-source tree` |
| `MBH` | central BH mass [Msun] | → local `smbh_mass_msun` when present |
| `inode` | tree node index (320 distinct values) | host identity, `--node` |
| `br` | branch index (constant `1.0` in this file) | host identity, `--branch` |

**How `Z` was verified.** `Z == mZ / (Mhot + Mcold)` holds to a relative
accuracy of 5e-13 over all 9999 rows (against 0.72 for `mZ/Mcold` and 0.03 for
`mZ/(Mhot+Mcold+Mstar)`). That fixes `Z` as a dimensionless mass fraction, not
a solar-normalised value — hence the `/ 0.0145` conversion in the mapping.

### Deliberately ignored columns

`plc`, `br_b4`, `fc`, `sf`, `bh`, `brf`, `Mhot`, `Mcold`, `Mout`, `mZ`, `dM`,
`Q`, `mlist`, `dt`, `gen`, `MSF`, `MBH2`, `Mgas`, `Mgas2`, `Mtde`, `Mtde2`,
`Macc2`. None is an input of the existing local model, and their semantics are
not established by the supplied materials. They are never parsed — checkpoint B
proves this by putting non-numeric junk in them and reading the file anyway.

### Large-file strategy

* `pd.read_csv(..., usecols=REQUIRED_TREE_COLUMNS, chunksize=N)` — 7 columns,
  one chunk resident at a time.
* Filtering and rejection happen **during** the stream; hosts are yielded from
  a generator, so jobs can be submitted before the file is fully read.
* The input file is opened read-only, never rewritten, copied or reordered.
* `record_index` comes from a sequential counter, not from a chunk offset, so
  identities are chunk-size independent.

### Compact counters

`rows_seen`, `rows_selected`, `missing_required_field`, `invalid_redshift`,
`invalid_halo_mass`, `invalid_bh_mass`, `outside_selection`,
`truncated_by_max_hosts`, plus `jobs_created` in the batch summary. No
per-rejected-row diagnostic table is produced.

---

## 4. Host-to-run mapping (`src/workflows/host_mapping.py`)

```python
resolved = resolve_host_parameters(host, base_config, policy)
```

Pure, cheap, and independent of the expensive simulation. Every resolved value
carries a provenance tag so a **direct tree value** is never confused with a
**derived local quantity**:

| Tag | Meaning |
|---|---|
| `tree:<column>` | read straight from the CSV |
| `tree:Z/Zsun` | read from the CSV with a verified unit conversion |
| `derived:<rule>` | computed from a tree field by an explicit configured rule |
| `config` | not host dependent; from the base configuration |

### Mapping sources

| Local input | Sources (CLI option) |
|---|---|
| redshift | always `tree:z` |
| SMBH mass | `tree` (default, uses `MBH`), `config`, `stellar-scaling` (`f × Mstar`) |
| NSC mass | `config` (default), `stellar-scaling` (`f × Mstar`) |
| metallicity | `config` (default), `tree` (`Z / 0.0145`) |

The `stellar-scaling` fractions (default `1e-4`) are **user-controlled
modelling placeholders**, not measurements — see Limitations.

### Validity boundaries

Before any expensive run: finite non-negative redshift, positive SMBH mass,
positive NSC mass, finite metallicity, and configurable domain bounds
(`--domain-smbh-min-msun` / `--domain-smbh-max-msun`, default 1e3–1e6 Msun,
matching the mass range the repository itself explores by default). Violations
raise `HostMappingError` carrying a batch status category.

---

## 5. Refactored single-halo interface (`src/workflows/single_halo.py`)

```python
result = run_single_halo(
    config=base_config,     # global model parameters
    host=host,              # HaloHost | ResolvedHostParameters | None
    seed=seed,              # deterministic base seed
    output_dir=output_dir,  # explicit output namespace
    policy=policy,
)
```

* Builds one resolved `IMRIDynConfig` (a 1 × 1 grid at the host's redshift and
  metallicity) via `config_with_overrides`, which **copies** rather than mutates
  the base configuration — so the base object is safe to ship to workers.
* Calls the unchanged `run_simulation`. **No physics was reimplemented.**
* Contains no CSV-specific code; the physical kernel never sees a CSV row.
* Redirects kernel stdout/stderr into `<output_dir>/run.log` so parallel
  workers do not interleave.
* Never raises for a per-host failure — every failure maps to a status.
* Returns `HaloSimulationResult(job_id, host_id, status, seed, output_path,
  summary, resolved, error, runtime_s)`.

`host=None` reproduces the ordinary single-halo mode exactly.

### What one invocation means

One call is one **complete** existing single-halo workflow: the kernel's own
`num_repeats` Monte Carlo realizations run inside the call and are averaged as
before. Repeats are *not* expanded into separate jobs.

### Status vocabulary

`success`, `host_not_selected`, `missing_required_input`, `invalid_host_input`,
`outside_local_model_domain`, `mapping_unavailable`, `simulation_failure`,
`numerical_failure`, `output_failure`.

Only `success` means the physics was evaluated. A `success` row with
`n_imri == 0` is a **physical zero**; any other status means the model **could
not be evaluated** and its physical columns are left `NaN`, never `0`. The
kernel's own physical statuses (`rate_status`, `detector_rate_status`) are
preserved untouched in the per-job summary.

---

## 6. Parallel execution (`src/workflows/tree_batch.py`)

### Job identity

`job_id == host_id == n<node>_b<branch>_r<record_index>` — unique,
deterministic, chunk-size independent, filesystem safe. Duplicate identities
raise rather than silently overwrite.

### Seed derivation

```
host base seed = SeedSequence([master_seed, node_key, branch_key, record_index])[0]
per-repeat seed = kernel derive_task_seed(base_seed, 0, 0, repeat_index)
```

Python's process-dependent `hash` is never used; no RNG stream is shared
between workers. Each worker seeds the global NumPy RNG from its own job's seed
inside the unchanged kernel.

### Worker behaviour

`run_jobs` runs serially at `workers == 1` and otherwise submits to a
`ProcessPoolExecutor` on a **spawn** context with a bounded in-flight window
(`2 × workers`), so a long host stream is never materialised in memory.
Results are collected by submission index and returned in submission order
regardless of completion order. Each worker receives one explicit picklable
`(job, base_config, policy)` payload, writes only to its own
`simulations/<job_id>/` directory, and returns a structured result.

Nested pools work: `ProcessPoolExecutor` workers are non-daemonic, so each host
run can still use the kernel's inner damping pool. In batch mode
`damping_cpu_count` defaults to `1` to avoid oversubscribing the machine
`workers`-fold.

### Failure isolation

A host that cannot be mapped never reaches a worker. A host that fails at
runtime is caught inside `run_single_halo`, written to
`simulations/<job_id>/error.txt`, and reported with a status — other jobs are
unaffected. Verified in checkpoint G, including a runtime `output_failure`
sandwiched between two successful jobs.

---

## 7. CLI

### Old single-halo mode — unchanged

```bash
python scripts/main.py 2 output/dyn_results 4 \
    --smbh-masses 1e5 --nsc-masses 5e4 --base-seed 777
```

Byte-identical to the pre-refactor implementation (verified by regression diff).
Tree-only selectors used without `--tree` produce a clear error.

### New merger-tree batch mode

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

Every option of the old mode becomes the **base** configuration.

| Group | Options |
|---|---|
| input/output | `--tree`, `--output` |
| selectors | `--node`, `--branch`, `--z-min`, `--z-max`, `--mbh-min`, `--mbh-max`, `--mhalo-min`, `--mhalo-max`, `--max-hosts` |
| mapping | `--smbh-mass-source`, `--nsc-mass-source`, `--metallicity-source`, `--smbh-stellar-mass-fraction`, `--nsc-stellar-mass-fraction`, `--domain-smbh-min-msun`, `--domain-smbh-max-msun` |
| execution | `--workers`, `--master-seed`, `--tree-chunk-size`, `--no-batch-figures` |

No `config.py` editing and no manual transfer of tree rows is required.

---

## 8. Output layout

```
output/tree_run/
├── config/
│   ├── resolved_config.json    base config, mapping policy, selection, settings
│   └── provenance.json         tree fingerprint, columns used/ignored, seeds,
│                               job list, code versions, scope note
├── hosts/selected_hosts.csv    compact selected-host table (streamed)
├── jobs/job_summary.csv        one row per job: identity, seed, status, error,
│                               resolved inputs + provenance tags, kernel results
├── simulations/<job_id>/       unchanged kernel outputs + run.log
├── diagnostics/batch_summary.json  reader counters, status counts
├── figures/                    selected_hosts_domain.png, batch_status.png
└── summary.json                compact top-level summary
```

Provenance identifies the input tree by absolute path, size, mtime and a SHA-256
of its first 64 KiB — the raw CSV is never copied.

---

## 9. Validation

`python scripts/run_interface_checkpoints.py` (results under
`output/checkpoints/`). The following are historical interface-run results,
not a rerun against the current physics: **A–C, E–H PASS; D WARN** (formatting
only). Run the command again to establish a current baseline.

| Checkpoint | What it proves | Result |
|---|---|---|
| **A** Baseline | original CLI path produces recorded outputs and seeds | PASS |
| **B** Reader | required-column validation, `usecols`, numeric parsing, identity, all filters, chunk-size invariance (fixture sizes 1/2/3/4/7/1000; real tree 50/333/1000/50000 — identical host sets), bounded full-file stream over 9999 rows | PASS |
| **C** Mapping | raw tree values vs resolved local inputs per policy, unit conversions, provenance tags, domain rejection, edge cases | PASS |
| **D** Refactor equivalence | `run_single_halo` vs original CLI, matched seed — averaged/repeat/event tables agree **value for value**; byte-identical for the no-host path | WARN¹ |
| **E** Tree equivalence | manual CLI run vs tree-derived run at the same resolved parameters — **byte-identical** outputs | PASS |
| **F** Parallel reproducibility | workers 1/2/4: identical job set, order, seeds, per-repeat task seeds, statuses and **byte-identical per-job result files**; no collisions or duplicates | PASS |
| **G** Failure isolation | valid hosts complete beside a missing-input host, an out-of-domain host, a reader-rejected row and a runtime `output_failure`; failed jobs carry `NaN`, not `0` | PASS |
| **H** Real-tree batch | 12 hosts from `galaxy_1e12.csv` through the **actual CLI path**, 4 workers: selection respected, seeds reconstructible, per-host outputs present, provenance complete, no cosmic-rate fields | PASS |

¹ **The D warning.** For the explicit-host path the `redshift` column is written
as `1.500000e+01` instead of `15`, because `config.py`'s default
`IMRI_REDSHIFT = 15` is an `int` while a host-resolved redshift is a `float`.
Every value in every table is equal; only the CSV text formatting of that one
column differs. This is reported rather than normalised away.

Timing: 4 hosts complete in 34.8 s serially, 20.7 s on 2 workers and 12.8 s on
4 workers (`figures/figure4_runtime_vs_workers.png`).

---

## 10. Limitations

**`MBH` is empty in the supplied tree.** Every one of the 9999 rows has
`MBH == 0`, as do `MBH2`, `Mgas`, `Mgas2`, `Mtde`, `Mtde2`. The default
`--smbh-mass-source tree` therefore rejects every host in this file with
`invalid_host_input` and an explicit message. This is reported faithfully rather
than silently substituted. To run this file you must choose `config` or
`stellar-scaling` explicitly.

**The `stellar-scaling` fractions are placeholders.** `M_smbh = 1e-4 × Mstar`
and `M_nsc = 1e-4 × Mstar` are *configurable conveniences* for exercising the
interface, not calibrated scaling relations. The supplied materials establish no
M–σ or M_NSC relation, so no such relation is claimed. Any scientific use must
replace these fractions deliberately.

**Uncertain field semantics.** `m` is treated as halo mass in Msun on the
strength of the filename and its ≈1e12 value; `br`/`br_b4` are constant in this
file so branch filtering is untested against a genuinely branching tree; `gen`,
`dM`, `Q`, `mlist`, `plc`, `dt` were not interpreted. `dt` is not used because
the local interface does not require it.

**Narrow host diversity.** This file spans `z ∈ [6.000, 6.038]` and
`Mstar ∈ [3.076, 3.138] × 10⁹` — one halo's history, not a population. The
interface is exercised, but the host domain it covers is tiny.

**Domain bounds are policy, not solver limits.** The default 1e3–1e6 Msun SMBH
window mirrors the repository's own default grid. The kernel itself ran without
error at 1e2 and 1e9 Msun in probing; the bounds are a configurable guard, not a
discovered failure boundary.

**Parallel and performance.** Determinism relies on the kernel seeding the
global NumPy RNG per repeat; any future use of thread-level parallelism inside
the physics would need re-verification. Nested pooling means effective
concurrency is `workers × damping_cpu_count`, which the user must size to the
machine. Scaling in checkpoint F is sub-ideal (2.7× on 4 workers) because
per-host runtimes are uneven and each spawned worker re-imports the stack.

**Local-simulation limitations are inherited unchanged.** Nothing in the
existing physics — the pAGN disk model, the damping prescription, the migration
and trap treatment, the simplified Peters threshold, the LISA SNR model — was
modified, reviewed or revalidated by this work.

**Out of scope, by design.** No cosmic IMRI rate, comoving rate density,
observer-frame all-sky rate, `dV_c/dz` weighting, merger-tree population
normalisation or LISA-detectable cosmic rate is computed. The batch output is a
collection of host-level single-halo results and must not be read as an
absolute cosmic rate.

---

## 11. Files

**New**

| File | Purpose |
|---|---|
| `src/workflows/merger_tree.py` | CSV → `HaloHost`; selection; counters |
| `src/workflows/host_mapping.py` | `HaloHost` → resolved local inputs; policy; domain |
| `src/workflows/single_halo.py` | `run_single_halo`; `HaloSimulationResult`; statuses |
| `src/workflows/tree_batch.py` | jobs, seeds, parallel executor, outputs, provenance |
| `src/viz/batch_plots.py` | two batch interface figures |
| `scripts/run_interface_checkpoints.py` | checkpoints A–H |
| `docs/merger_tree_interface.md` | this document |

**Modified**

| File | Change |
|---|---|
| `src/workflows/cli.py` | tree-mode argument group and dispatch; config construction factored into `build_config_from_arguments`; old path preserved |
| `scripts/main.py` | now also a programmatic entry point re-exporting the public API |
| `src/workflows/__init__.py` | export the new public names |

No existing physical module was modified.
