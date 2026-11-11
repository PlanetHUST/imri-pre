# Run guide and single-host workflow

The core idea is a **single-host IMRI simulation**: initialize a specified stellar/BH population (default: original analytic IMF and power-law orbital sampler), numerically solve damping, follow inward Type-I semi-analytic migration, classify inspirals, and calculate LISA detectability. SEVN and alternative migration models are opt-in; tree batch is no longer supported.

## Setup and run

Requires Python 3.10+ and the dependencies in `pyproject.toml`:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python scripts/main.py 1 outputs/pan2021_small_run 1 \
  --smbh-masses 1e5 --nsc-masses 5e4 --base-seed 777
```

Use an explicit seed for repeatable sampling. Keep the same dependency versions and configuration when comparing numerical results. `outputs/pan2021_small_run/` is generated locally, not source-controlled; inspect `simulation_metadata.json`, `imri_averaged_results.csv`, `imri_repeat_results.csv`, and `imri_events.csv` there. A zero event count is a valid result. Do not compare output directories byte-for-byte: metadata and diagnostics may contain run-specific information.

## Small checks

```bash
python scripts/check.py
```

Run `python scripts/check.py --run-dir outputs/pan2021_small_run` to also validate saved output. Install the test extra as above; some optional checks use pytest helpers.

The default is restored to `legacy_popiii`, your original IMF/lifetime model. Pan/Xue component populations require an explicit `--population-model` option. The published mass, radial and normalization definitions are described in [population prescriptions](published_populations.md).

## Execution order

[`scripts/main.py`](../scripts/main.py) calls [`run_simulation`](../src/workflows/workflow.py) over the requested host grid and derived repeat seeds. [`IMRISingleWorkflow`](../src/workflows/single_work.py) processes each host episode:

```text
host + seed → disk and population → disk-crossing sources
            → numerical orbital damping → optional SEVN backend
            → simultaneous gas/GW migration → first selected-band entry
            → physical BH classification and LISA selection
            → eligible ISCO consumption → unused donor on that channel
            → repeat until deadline/pool exhaustion → counts and rates
```

A replacement donor is eligible only after a prior source reaches ISCO during the AGN episode. Primary and replacement sources use the same evolution/classification path; the donor pool is finite. `--no-consumption-replenishment` disables replacements, not gas fueling. SEVN requires external tables and an installed backend.

## Outputs and status handling

Use a new directory for each distinct experiment. The three positional CLI arguments are repeat count, output directory and damping worker count. The [output index](../outputs/README.md) separates `core/` science results, `supporting/` uncertainty/validation studies and `archive/` exploratory/historical runs. The unchanged CLI default `outputs/single_host/` is scratch, not a designated science result; use explicit directories for new studies.

| File | Purpose |
|---|---|
| `simulation_metadata.json` | Resolved configuration, units and provenance |
| `imri_averaged_results.csv` | Host-level means |
| `imri_repeat_results.csv` | Counts, seeds and evaluation statuses per repeat |
| `imri_events.csv` | Object-level physical/detector classifications |
| `imri_workflow_checkpoints.csv` | Stage bookkeeping |
| `imri_source_transitions.csv` | Source action boundaries |
| `imri_replenishment_ledger.csv` | ISCO consumption and donor accounting |
| `intermediate/` | Per-repeat migration diagnostics; SEVN may also write a stellar ledger |

Rates count successful embeddings, physical BH band entries, or the detector-selected subset, divided by source-frame AGN duration. Embeddings include stars; entries are not ISCO merger counts. Delayed entries may be included. Definitions and scientific interpretation are centralized in [results and uncertainties](simulation_results_synthesis.md).

Inspect `rate_status`, `detector_rate_status`, and event-level flags: unavailable evaluations must not be treated as zero. The saved-run validator checks stage order, source uniqueness, donor accounting and rate bookkeeping without rerunning the simulation.

## Additional controls and tests

Independent dimensional controls include `--disk-alpha`, `--disk-accretion-rate-g-s`, `--disk-outer-radius-cm`, `--nsc-radius-pc-at-reference-mass` and `--nsc-radius-mass-slope`. Leaving dimensional overrides unset retains the standard accretion-rate scaling and Q-boundary choice. An explicit disk radius changes the solver domain; extending it can include unstable gas.

The default migration is inward analytical Type-I. `--migration-model torque_with_trap` selects signed torques with continuous gap/viscous transport; add `--torque-no-gap-transport` for the ungapped comparison. These alternatives do not change the default. Exact archived comparisons should use their runners, which copy benchmark metadata rather than generic CLI defaults; commands are in the [results reproduction section](simulation_results_synthesis.md#9-reproduce-and-inspect-the-experiments).

`PYTHONPATH=src python -m pytest -q` runs the retained tests. Focused parameter-control regressions are in `tests/core/test_parameter_dependence.py`; detector checks are in `tests/core/test_detect_process.py`. Tests do not replace physical validation or a convergence study.

For equations and assumptions see [physical mechanisms](physical_mechanisms.md); for component choices see [population prescriptions](published_populations.md).
