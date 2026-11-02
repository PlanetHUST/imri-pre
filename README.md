# IMRI in nuclear star clusters and AGN disks

Simulates the formation and LISA detectability of **intermediate-mass-ratio inspirals (IMRIs)**: stellar-mass compact objects in a nuclear star cluster (NSC) that cross an AGN disk, are damped into it, migrate through it, and inspiral into the central massive black hole.

The project answers two questions:

1. **One halo** — for a given host (redshift, SMBH mass, NSC mass, metallicity), how many IMRI candidates form and how many would LISA see?
2. **Many halos** — given a merger-tree CSV of hosts, run that simulation for each selected host, reproducibly and in parallel.

It deliberately stops at host-level results now. **No cosmic rate is computed.** It is left for future updates

Start with the [single-halo workflow](docs/main_workflow.md) for the action and condition path from sampling to final rate, or the [documentation map](docs/README.md) for current contracts and the part-by-part source guide. The former root text exports are preserved in [the reference archive](docs/archive/README.md); their durable decisions and later corrections are combined in [the decision history](docs/reference/history.md).

---

## What one simulation means

One *single-halo run* is one complete Monte Carlo experiment for one host:

```
sample an NSC population  →  select disk-crossing orbits  →  damp them into the disk
   →  migrate the embedded objects  →  resolve gas vs GW competition
   →  classify IMRI candidates  →  evaluate LISA SNR and detectability
   →  at ISCO consumption, refill the vacated orbit from the finite NSC pool
   →  repeat damping and classification until no channel or donor remains
   →  finalize rates and write per-stage checkpoints
```

It runs the configured number of Monte Carlo repeats internally and reports
their average. One merger-tree *job* is exactly one such run.
The control loop is in `src/workflows/single_work.py`; the physical actions
and detector calculations stay in their respective modules. Its additive
`imri_workflow_checkpoints.csv` and `imri_source_transitions.csv` outputs make
the stage counts and each source's action path inspectable.

---

## The physical pipeline

```
host  →  AGN disk  →  stellar population  →  disk crossing  →  damping
      →  migration (torque → gap → traps → radial evolution)
      →  gas/GW handoff  →  IMRI event classification  →  LISA SNR → detectability
                                              └→ ISCO consumption → replenishment
```

Each arrow is owned by exactly one module. See
[`docs/physics_model.md`](docs/physics_model.md) for the authoritative
module-ownership table and the assumptions behind each stage.

> **Migration contract.** Production uses one signed torque:
> `Gamma = f_gap Gamma_grav + Gamma_thermal + (1-f_gap) Gamma_gap`, with
> continuous Kanagawa depth `f_gap = 1/(1+0.04K)` and
> `K=q²/(alpha h⁵)`. `Gamma_gap` is the inward viscous-drift limit slowed by
> the local disk-to-migrator mass ratio. The thermal term follows the local
> Grishin-2024 prescription on the pAGN disk and is applied only where its
> linear validity checks pass. There is no runtime torque/gap selector.
> Regime-limit flags are persisted with the profile. See
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
The optional `--base-seed` defaults to `None`; pass it explicitly for paired,
reproducible runs. The single-halo main workflow writes scientific CSV/JSON
without automatically invoking the saved-result figure functions.
Consumption-driven replenishment is enabled by default. Use
`--no-consumption-replenishment` to run the initial disk-crossing cohort only.
The [replenishment plan](docs/plans/active/imri_consumption_replenishment.md)
records the finite-source and event-time assumptions.

### SEVN remnant masses and natal kicks

The opt-in stellar mode keeps the same sampled NSC, disk, and finite donor
supply. It gets collapse age, remnant type/mass, and kick speed from the
installed `sevnpy` backend. The kick-to-orbit operation is implemented locally in
`physical_process.natal_kick`. In this workspace, the `sevnpy_env` environment
contains SEVN:

```bash
export PYTHONPATH="$PWD/src"
SEVN_PY=/home/planet/miniconda3/envs/sevnpy_env/bin/python
SEVN_TABLES=/home/planet/Phy/pre-codes/sevn/tables
"$SEVN_PY" scripts/main.py 20 output/sevn_example 4 \
    --smbh-masses 1e5 --nsc-masses 2e5 --redshift 15 --base-seed 12345 \
    --stellar-evolution-model sevn --natal-kick-model sevn \
    --sevn-hydrogen-tables "$SEVN_TABLES/SEVN_parsec_ov05_AGB_popIII" \
    --sevn-helium-tables "$SEVN_TABLES/SEVNtracks_parsec_pureHe36"
```

Use `--natal-kick-model none` for the paired remnant-mass control. IMRI's
`--metallicity` is in solar units; SEVN receives the absolute mass fraction.
The run writes `imri_stellar_ledger.csv` with one row per evaluated source,
including sources that cannot become BHs. The metadata records SEVN options,
versions, and table hashes. Pre-collapse drag still uses fixed ZAMS mass; the
mass change occurs at collapse. If a bound eccentric post-kick orbit could
reach the LISA band but cannot be evaluated by the circular detector model,
its status makes the rate incomplete. Sources outside the selected SEVN track
coverage also make the rate incomplete; there is no silent mass or metallicity
clamp. See the
[integration plan](docs/plans/active/sevn_remnant_kick_integration.md).

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

| Path                                     | Contents                                                     |
| ---------------------------------------- | ------------------------------------------------------------ |
| `<output_dir>/imri_averaged_results.csv` | one row per grid point, averaged over repeats                |
| `<output_dir>/imri_repeat_results.csv`   | one row per Monte Carlo repeat, with its seed                |
| `<output_dir>/imri_events.csv`           | one row per evaluated object, with its classification reason |
| `<output_dir>/imri_replenishment_ledger.csv` | one row per in-episode ISCO consumption and resulting donor injection or source exhaustion |
| `<output_dir>/simulation_metadata.json`  | resolved configuration and model identifiers                 |
| `<output_dir>/intermediate/`             | per-repeat migration diagnostics                             |

A merger-tree batch adds `config/` (resolved config + provenance), `hosts/`,
`jobs/job_summary.csv`, `simulations/<job_id>/`, `diagnostics/`, `figures/`
and `summary.json`.

**Reading a zero correctly:** a job with `status = success` and `n_imri = 0` is
an evaluated zero only when its rate validity is complete. Check `rate_status`
and `detector_rate_status` too: successful execution does not guarantee that
every physical/detector result is available. Failed jobs retain unavailable
physical columns instead of zeros.

---

## Validation

```bash
PYTHONPATH=src python tests/run_tests.py       # automated correctness (no pytest needed)
python scripts/run_science_checkpoints.py    # scientific checkpoints B–L
python scripts/run_interface_checkpoints.py  # merger-tree interface checkpoints A–H
```

The four scripts most worth running first:

| Script                               | Question it answers                                                           |
| ------------------------------------ | ----------------------------------------------------------------------------- |
| `tests/run_tests.py`                 | Do the canonical modules still behave correctly?                              |
| `scripts/run_science_checkpoints.py` | Is each physical subsystem sane, and does the end-to-end pipeline still work? |
| `scripts/audit_repository.py`        | Has duplicated or script-local physics crept back in?                         |
| `scripts/compare_baseline.py`        | Did a refactor change any scientific output?                                  |

`compare_baseline.py capture <run> <file>` then `check <run> <file>` compares
runs while ignoring timings and paths.

---

## Scripts

| Script                           | Purpose                                                        |
| -------------------------------- | -------------------------------------------------------------- |
| `main.py`                        | CLI entry point and programmatic API                           |
| `run_science_checkpoints.py`     | Round-2 scientific checkpoints B–L                             |
| `run_interface_checkpoints.py`   | merger-tree interface checkpoints A–H                          |
| `run_redshift_scan.py`           | fixed-contract redshift scan driver                            |
| `audit_repository.py`            | dependency / duplicated-physics audit                          |
| `compare_baseline.py`            | scientific-output regression comparison                        |
| `validate_damping_model.py`      | damping invariants, scans and phase diagrams                   |
| `compare_damping_reference.py`   | shared-input comparison with both TDE damping sources          |
| `validate_gas_gw_handoff.py`     | gas/GW handoff regression checks                               |
| `validate_lisa_detectability.py` | LISA PSD, cosmology, SNR and classification checks             |
| `validate_migration_contract.py` | fixed torque contract, validity flags, and trap/no-trap bounds |
| `plot_disk_structure.py`         | disk profiles and pAGN-vs-in-repo comparison                   |
| `analyse_population_torque.py`   | population occupancy of torque validity regimes and traps      |
| `report_event_summary.py`        | print an event summary from a results directory                |

---

## Environment

Python ≥ 3.10 with `numpy`, `scipy`, `pandas`, `matplotlib` and `pagn`.
Install for development with `pip install -e .`; the package puts `src/` on the
path, so `physical_process`, `disk_quantity`, `lisa`, `viz` and `workflows` are
importable directly.

This cleanup changes documentation and reference locations only. Existing
scientific outputs under `output/` and `figs/` remain evidence for the source
and inputs that produced them, rather than current numerical baselines.

### Pop III IMF sensitivity study

`--popiii-max-mass-msun` and `--popiii-imf-slope` control the upper mass and
power-law exponent `dN/dm ∝ m^-alpha` for the Pop III branch. Their defaults
remain 197 Msun and 1.77. These options do not alter the Pop II sampler. SEVN
track coverage remains authoritative and sources outside it are unavailable.

Run the reproducible endpoint map and checkpointed single-halo study with:

```bash
PYTHONPATH=src /home/planet/miniconda3/envs/sevnpy_env/bin/python \
  scripts/run_imf_sensitivity.py --stage all --repeats 20 --cpus 2
```

Each realization has its own `checkpoint.json`, log, initial population, and
workflow outputs. Re-running the command resumes completed checkpoints. The
study uses a fixed NSC mass and paired base seeds; changing the IMF changes the
population size and RNG history, so equal source IDs across scenarios do not
identify the same object. `repeat_summary.csv` and `scenario_summary.csv`
are refreshed as repeats complete. To evolve all sampled sources above the
diagnostic mass cut, including those that did not cross the disk, run
`PYTHONPATH=src python scripts/census_high_mass_sevn.py output/imf_sensitivity_20261029`.
The optional all-source endpoint census can be run with
`PYTHONPATH=src python scripts/census_initial_sevn.py output/imf_sensitivity_20261029 --follow`.
The mass threshold 240.5 Msun is a diagnostic
initial-mass cut for the specified SEVN table and metallicity, not a universal
black-hole mass-gap boundary. The stellar ledger includes evaluated disk
crossers and donors, not the full population of stellar endpoints.

After the realization and census checkpoints complete, refresh the derived
artifacts with:

```bash
PYTHONPATH=src python scripts/enrich_sevn_endpoint_map.py output/imf_sensitivity_20261029
PYTHONPATH=src python scripts/run_imf_sensitivity.py --stage summarize
PYTHONPATH=src python scripts/report_imf_sensitivity.py output/imf_sensitivity_20261029
PYTHONPATH=src python scripts/summarize_imf_masses.py output/imf_sensitivity_20261029
MPLBACKEND=Agg python scripts/plot_imf_endpoints.py output/imf_sensitivity_20261029
MPLBACKEND=Agg python scripts/plot_imf_study.py output/imf_sensitivity_20261029
PYTHONPATH=src python scripts/validate_imf_study.py output/imf_sensitivity_20261029
```

Use the same SEVN environment as the simulation. Full-NSC census endpoints are
eventual stellar outcomes; they do not imply that every remnant formed during
the modeled AGN episode.

To extend the IMF study to the archived redshift, duty-cycle, and migration
cases, first complete `output/imf_sensitivity_20261029`. The torque
`z15_duty10` results are reused from that study. Run the other cases with:

```bash
for case in z10_duty10 z15_duty100; do
  PYTHONPATH=src python scripts/run_imf_redshift_lifetime.py --model torque --case "$case"
done
for case in z10_duty10 z15_duty10 z15_duty100; do
  PYTHONPATH=src python scripts/run_imf_redshift_lifetime.py --model semi --case "$case"
done
PYTHONPATH=src python scripts/report_imf_redshift_lifetime.py
python scripts/summarize_archived_imri.py output/imf_redshift_lifetime_20261029
python scripts/write_pan_comparison_report.py output/imf_redshift_lifetime_20261029
PYTHONPATH=src:scripts python scripts/validate_imf_redshift_lifetime.py output/imf_redshift_lifetime_20261029 --require-complete
MPLBACKEND=Agg python scripts/plot_imf_redshift_lifetime.py output/imf_redshift_lifetime_20261029
```

Run those commands in the SEVN environment. Each completed realization writes
its own checkpoint and `run.log`; the analysis scripts read only saved results.
The previous fixed-mass archive uses a different repeat-seed schedule from this
IMF study, so its aggregate comparison is not a matched-source experiment.
