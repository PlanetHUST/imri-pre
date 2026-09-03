# Part 07 — Workflow Orchestration

## 1. Idea

The physics parts (01–06) are staged by an orchestration layer that turns a
configuration into a finished set of result tables. It owns:

1. **configuration** — one validated object (`IMRIDynConfig`) that carries every
   scientific and runtime choice;
2. **the CLI** — the human interface that builds that config;
3. **the grid driver** — loops over (SMBH mass × NSC mass × repeat) and
   checkpoint-saves partial results;
4. **the single realization** — one repeat, staged as a short readable sequence;
5. **workers and scheduling** — multiprocessing for damping, CPU discovery;
6. **seeding** — deterministic, paired Monte Carlo tasks.

The main idea: *the workflow coordinates stages; it never reimplements their
physics.* Each file has one responsibility, and the execution order of one
realization is readable top to bottom.

## 2. Doing logic

### 2.1 Configuration

Authoritative owner: `src/workflows/config.py::IMRIDynConfig`.

Keyword-only constructor, explicit validation of every field (positive masses,
`0 < f_min < f_max`, finite redshift, gap condition in
`GAP_OPENING_CONDITIONS`, disk model in `DISK_MODELS`, …). Internal NSC model
parameters keep defaults; CLI-owned runtime values are explicit. `disk_model`
defaults to `torque_with_trap`; `gap_opening_condition` defaults to `kii`.

### 2.2 CLI

Authoritative owner: `src/workflows/cli.py::run_cli` → `build_parser`.

`scripts/main.py` is a short forwarding entry point (path injection +
`run_cli()`). One invocation runs one concrete config into one output directory.
The scan-level multi-redshift, multi-scenario experiment is owned by
`scripts/redshift_scan.py` (not the CLI).

### 2.3 Grid driver

Authoritative owner: `src/workflows/workflow.py::run_simulation`.

- builds one `IMRISingleWorkflow(IMRIDynEngine(config))`;
- loops `smbh_index × nsc_index`, then repeats;
- per repeat: derive seed, seed numpy, run realization, print summary, save
  migration diagnostics;
- averages repeats into `IMRIDynAveragedResult`;
- checkpoints averaged/event/repeat CSVs every `checkpoint_grid_interval`;
- writes `simulation_metadata.json` at the end.

### 2.4 Single realization

Authoritative owner: `src/workflows/single_work.py::IMRISingleWorkflow`.

`run_single_simulation()` reads as explicit stages:

```
AGN state  →  disk profiles  →  population sampling  →  disk-crossing
  →  damping workers  →  event classification  →  result
```

The file does not contain torque composition, KII/Crida formulas, frequency
unit scaling, or CSV schema logic.

### 2.5 Workers and scheduling

Authoritative owners: `src/workflows/workers.py` (pool + task serialization),
`src/workflows/runtime.py` (verbose printing).

`get_hpc_cpu_count()` reads scheduler variables (`SLURM_CPUS_PER_TASK`,
`PBS_NP`, `NSLOTS`) then falls back to local CPUs. Damping tasks run in a `Pool`
whose workers each hold one `DampingCalculator`. Multiprocessing/CPU concerns
live only here — not in the orbital physics module.

### 2.6 Seeding (paired tasks)

Authoritative owner: `workflow.derive_task_seed`.

```
task_seed = SeedSequence(base_seed, smbh_index, nsc_index, repeat_index)
```

Redshift and disk model are deliberately **excluded**, so the same
`(SMBH, NSC, repeat)` task is paired across all redshifts and both migration
scenarios. This is what makes the paired migration-prescription comparison and
the redshift scan scientifically clean.

### 2.7 Redshift scan

Authoritative owner: `scripts/redshift_scan.py::run_scan`.

Loops `disk_model × redshift`, each into a collision-free directory, reuses one
config template, keeps paired seeds, then builds combined long-form repeat/event
frames and hands them to the analysis layer. Refuses to overwrite a non-empty
output dir without confirmation, and validates the target is not a repo/system
dir.

## 3. Key invariants

- `run_simulation` is the only grid loop; `run_single_simulation` is the only
  realization loop.
- One main invocation = one disk model into one output directory.
- Physics modules never import workflow/output/CLI/plotting code.
- `workflows/__init__.py` exports only the supported public application surface;
  scripts import `run_cli` explicitly (no wildcard).

## 4. Caveats

- **Do not reset or rewrite the working tree.** The tree is usually mid-task;
  uncommitted/untracked files are user-owned baseline state
  (see `AGENTS.md`).
- **Pairedness requires identical upstream draws.** Task seeds and sampled
  populations must be exactly equal across scenarios; differences may begin only
  at the selected migration profile and its consequences.
- **Independent seeded runs, not shared mutable objects.** Do not optimize the
  first implementation by sharing realization objects between scenario runs.
- **Checkpointing matters.** A long grid should retain completed migration JSON
  diagnostics even if a later grid point fails — that is why per-repeat
  diagnostics are written immediately after each repeat.
- **CLI defaults are CLI-owned.** `config.py` should not silently invent runtime
  defaults that the CLI already supplies; a new user-authored config must not
  receive a silent scientific default.
