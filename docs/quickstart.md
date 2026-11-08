# Minimal reproducible run

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

For stage definitions and assumptions, see [main workflow](main_workflow.md) and [physical model](physics_model.md).
