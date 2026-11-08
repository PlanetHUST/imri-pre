# Single-host IMRI simulation

This project simulates compact objects in one nuclear star cluster around a massive black hole. The CLI uses your original analytic power-law IMF and stellar formation-time prescription by default (13.6–197 Msun, slope 1.77; custom bounds including 1–400 Msun remain available). It numerically damps disk-crossing orbits, uses inward semi-analytic **Type-I-only** migration with coupled gas/GW inspiral, and estimates LISA detectability. The original radial sampler with gamma=1.75 is restored; published population presets are opt-in only. Results are Monte Carlo estimates for the **specified host**, not a cosmic event rate.

## Reproduce a small run

Requires Python 3.10+. From the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python scripts/main.py 1 outputs/pan2021_small_run 1 \
  --smbh-masses 1e5 --nsc-masses 5e4 --base-seed 777
python scripts/check.py --run-dir outputs/pan2021_small_run
```

The three positional arguments are Monte Carlo repeats, output directory and damping worker count. An explicit seed makes sampling reproducible; keep the same inputs and dependency versions when comparing results. A zero IMRI count can be a valid scientific result. For a more detailed guide, see [quickstart](docs/quickstart.md).

## Where to look

| Path | Purpose |
| --- | --- |
| `scripts/main.py` | Single-host CLI only |
| `src/workflows/` | Configuration, orchestration and saved-result schemas |
| `src/physical_process/`, `src/disk_quantity/`, `src/lisa/` | Portable physical and detector models; `migration/type_i.py` is the default drift |
| `src/optional_models/` | Opt-in SEVN, damping benchmarks and Type-I/II comparison |
| `src/viz/` | Optional figures; not required for the main run |
| `scripts/check.py`, `tests/core/`, `tests/optional/` | One-command checks, default-path and optional-model assertions |
| `outputs/` | Generated runs; use separate directories for each population |

See [published populations](docs/published_populations.md) for mass normalization, radial-law definitions and limitations. Explicit `--population-model legacy_popiii` retains the old analytic IMF; custom Pop III IMF controls or SEVN auto-select that legacy path unless a population is explicitly chosen. Pan paper normalization and the Xue BH-only control do not have an independent NSC-mass axis.

The run writes `simulation_metadata.json` (configuration), `imri_averaged_results.csv` (host-level summary), `imri_repeat_results.csv` (per-repeat counts and seeds), and `imri_events.csv` (event classifications). It also writes source-transition, stage-checkpoint, replenishment and intermediate diagnostic files in the same run directory. These are generated artifacts, not checked-in fixtures. Use a **new output directory** for each distinct experiment; don't mix runs in one folder. The validator above checks saved stage and source bookkeeping.

Run the focused checks with:

```bash
python scripts/check.py
```

The wrapper invokes the retained tests; `PYTHONPATH=src python -m pytest -q` also works if pytest is installed. See [workflow](docs/main_workflow.md) and [model](docs/physics_model.md) for the action path and assumptions.

Tree batch is no longer supported. The moved `input/galaxy_1e12.csv` is preserved locally but is not used by the single-host command. Optional SEVN and alternative migration prescriptions remain available programmatically; the default uses no SEVN backend. LISA SNR is implemented in `src/lisa/snr.py`; the unused external GitHub clone formerly at `lisa_ref/` has been removed.
