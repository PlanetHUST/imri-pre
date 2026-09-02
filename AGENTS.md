# AGENTS.md

Python research code for dynamical IMRI (intermediate-mass-ratio inspiral)
event-rate calculations in AGN disks around SMBHs. No packaging, no CI, no
formal test framework: `scripts/*_test.py` are standalone assertion harnesses.
Active contracts live in `docs/specs/` and `docs/plans/`; durable guidance lives
in `docs/reference/`. Start at `docs/README.md`.

## Environment

- `python` is not on PATH. Use the conda env interpreter
  `/home/planet/miniconda3/envs/sevnpy_env/bin/python` (also holds `pyright`,
  `pytest`, `ipython`). `sevnpy_env` is fixed by `.vscode/settings.json` and
  `pyrightconfig.json`.
- No `requirements.txt`/`pyproject.toml`; assume packages already exist in
  `sevnpy_env`.
- Typecheck with `pyright` at the repo root (uses `pyrightconfig.json`:
  includes `src` + `scripts`, basic mode, Python 3.12). `src/disk_quantity/opacity.py`
  is deliberately excluded. `scripts/imri_rate_visualization_test.py` carries a
  top-of-file pyright suppression comment.

## Layout

- `src/workflows/` — orchestration: `cli.py` (args → `IMRIDynConfig`),
  `workflow.py` (`run_simulation` grid driver), `engine.py` (cached
  disk/torque resources), `single_work.py` (one realization),
  `event_rate.py`/`detect_rate.py` (classification), `outputs.py` (CSV
  formatting), `rate_plots.py` (all scan figures).
- `src/physical_process/` — physics, imported as top-level `physical_process.*`:
  `disk_migration.py`, `damp.py`, `gw_frequency_radius.py`, `nsc_orbital.py`,
  `constants.py`.
- `src/disk_quantity/` — disk models: `pagn_disk_adapter.py` (Sirko/Thompson
  pAGN), `migration_torque.py`, `disk_solve.py` (legacy fallback).
- `scripts/` — entry points and diagnostic harnesses. They self-inject `src/`
  into `sys.path`; run them from the repo root.
- `docs/` — `README.md` navigation; active `specs/` and `plans/`; compact
  long-term `reference/`; and the AI-generated source-oriented `project/`
  explanation. Read the relevant specification or plan before a task.

## Running

- Single grid run: `scripts/main.py [repeats] [output_dir] [workers]` plus
  flags (`--disk-model`, `--smbh-masses`, `--gap-open-condition`, ...). See
  `--help`.
- Paired redshift × disk-model scan: `scripts/redshift_scan.py`. Defaults:
  redshifts `10 13 15`, both disk models, `base_seed 12345`, output
  `imri_redshift_scan_results/`. It refuses to overwrite a non-empty output dir
  unless you type `yes`, and validates the target is not a repo/system dir.
- Deterministic figure harness: `scripts/imri_rate_visualization_test.py
  --output-dir figs/`.
- Diagnostics: `scripts/migration_test.py`, `scripts/trap_plot.py`,
  `scripts/disk_plot.py`, `scripts/damp_test.py`, `scripts/final_analysis.py`.

## Verification (established convention)

- `.../bin/python scripts/imri_rate_visualization_test.py`
- `.../bin/python -m compileall src scripts`
- `.../bin/pyright`
- `git diff --check`

## Conventions and gotchas

- Units are baked into variable/column/arg names: `_msun`, `_myr`, `_cm`,
  `_rg`, `_pc`, `_g_cm2`, `_si`. Keep this everywhere.
- `migration_torque.py` computes torques in SI; convert to CGS only at the
  consumer boundary via `torque_si_to_cgs()`. Never mix unit systems.
- `--disk-model` selects the migration prescription only
  (`torque_with_trap` default, `semi_without_trap`). It must NOT change the pAGN
  structure model (`sirko`/`thompson`); both scenarios share the same disk for
  boundaries, crossing, damping, and gap decisions.
- Gap opening: `--gap-open-condition {kii,crida}`, default `kii`. The shared
  selector is `disk_migration.gap_opening_mask()`.
- Torque zero-crossings (stable trap or anti-trap) → migration time `inf` with
  `migration_trap_bound` status; profile/domain failures are separate explicit
  errors, never zeros.
- Deterministic paired seeds: `numpy SeedSequence(base_seed, smbh_index,
  nsc_index, repeat_index)` `workflow.derive_task_seed`. Redshift and disk
  model are intentionally excluded so matched tasks share draws.
- Output CSVs are formatted at save time: `workflows/outputs.py` hides cm
  columns and renames frequency columns per chosen unit, and writes
  `float_format="%.6e"`; `redshift_scan.py` scan-level CSVs use `"%.17g"`.
  Compare CSVs with care.
- The working tree is usually mid-task: treat all uncommitted/untracked files
  as user-owned baseline and do not reset or rewrite unrelated changes.
- Git: `figs/`, `docs/`, `imri_dyn_results/`, and all `*.md` (including this
  file) are gitignored. `imri_redshift_scan_results/` is tracked.
