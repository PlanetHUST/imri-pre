# Part 08 — Data I/O

## 1. Idea

Persistence turns in-memory physics results into auditable, human-readable, and
strictly machine-readable artifacts. The data layer owns:

1. **canonical in-memory units** — all event frequencies stored in Hz, radii in
   cm, times in Myr;
2. **output formatting at save time** — data-dependent frequency units, hidden
   cm columns, renamed compact columns;
3. **strict JSON** — no `Infinity`/`NaN` tokens; unavailable values are `null`;
4. **per-repeat diagnostics** — intermediate migration JSON written immediately
   after each repeat;
5. **metadata** — configuration, derived quantities, runtime statistics.

The main idea: *models are typed domain data; outputs.py is the only owner of
presentation and persistence. Physics must never depend on how a value is
displayed, and a missing/unavailable value must never be serialized as a
physical zero.*

## 2. Doing logic

### 2.1 Canonical units in memory

- `IMRIEventRecord` (in `models.py`) stores frequencies as `*_hz`, radii as
  `*_cm` (with parallel `*_rg` diagnostics), times as `*_myr`.
- Physical detector diagnostics retain canonical `lisa_snr`, source/detector
  chirp masses, `luminosity_distance_mpc`, observer-frame SNR bounds and ISCO,
  mission/effective duration, status, reason, and observation case.
- Result containers (`IMRIDynSingleResult`, `IMRIDynAveragedResult`) carry typed
  counters and unit-explicit fields. `to_dict()` is the single serialization
  surface.

### 2.2 Output formatting (outputs.py)

- `save_results` → `imri_averaged_results.csv` via an explicit
  `IMRI_RESULT_COLUMN_MAP` (source → compact output name, e.g. `n_imri`,
  `rate_imri_per_myr`, `n_mig_eval` …). Column order is defined explicitly, not
  derived from annotation order.
- `save_events` → `imri_events.csv`: `EVENT_HIDDEN_COLUMNS` drop the cm radius
  columns; `EVENT_FREQUENCY_COLUMNS` are rescaled into a data-dependent unit
  chosen from `hz / milli_hz / micro_hz / nano_hz / pico_hz` by the median
  finite nonzero value, and the chosen unit is baked into the column name
  (`f_entry_src_milli_hz`, …).
- `save_repeat_results` → `imri_repeat_results.csv` with long-form identity
  columns (`disk_model`, `redshift`, masses, `repeat`, seeds) and every compact
  counter. Intrinsic `rate_status` and `detector_rate_status` are separate, as
  are detector-evaluated and detector-unavailable candidate counts.
- All CSV writing uses `float_format="%.6e"` (scan-level summaries use
  `"%.17g"`; plot data `"%.10g"`).

### 2.3 Strict JSON

- `save_metadata` writes `simulation_metadata.json` with `allow_nan=False`,
  recursively converting nonfinite floats to `null`.
- `save_migration_diagnostics` writes one `intermediate/migration_*.json` per
  repeat immediately after the repeat, also `allow_nan=False`. A trap-bound
  object serializes `"migration_time_to_entry_myr": null` with an explicit
  `"migration_status": "trap_bound"` — the status carries the physics.

### 2.4 Conservation checks at save time

Before writing diagnostics, `save_migration_diagnostics` verifies:

```
n_band_eval = n_entry_in_domain + n_entry_handoff + n_entry_inside_isco + n_profile_error
n_mig_eval  = n_entry_in_domain + n_entry_handoff + n_profile_error
            = n_mig_finite + n_mig_trap + n_mig_error
```

plus per-status event-record agreement, and that `semi_without_trap` produces no
trap-bound outcomes. Failures raise instead of writing inconsistent data.

### 2.5 Scan-level products (redshift_scan.py)

- per-model run directories under `<scan>/<disk_model>/zNNN_<z>/`;
- `disk_model_scan_summary.csv` (long form, one row per
  `(disk_model, redshift, SMBH, NSC, repeat)`);
- `disk_model_rate_comparison.csv` (paired semi-minus-torque guarded comparison);
- `figures/data/*.csv` and `figures/**/*.png|.pdf`;
- `disk_model_scan_metadata.json` recording every artifact path.

## 3. Canonical output files (one `main.py` run)

| File | Content |
|---|---|
| `imri_averaged_results.csv` | one row per (SMBH, NSC) bin, averaged over repeats |
| `imri_events.csv` | one row per evaluated embedded object |
| `imri_repeat_results.csv` | one row per Monte Carlo repeat |
| `simulation_metadata.json` | config + derived quantities + runtime stats |
| `intermediate/migration_*.json` | per-repeat migration diagnostics |

## 4. Caveats

- **Formatting is a save-time concern only.** In-memory frequencies are always
  Hz; `outputs.py` renames/scales columns at save time. Physics modules must not
  emit display-formatted values.
- **Unavailable ≠ zero, `null` ≠ a number.** Inside-ISCO, profile-error, and
  incomplete repeats yield `null`/NaN rate fields (and `rate_status` explaining
  why), never a serialized `0.0`.
- **Physical detector no-overlap is the exception:** it is a successfully
  evaluated zero SNR with an explicit no-overlap status. Detector model/input
  unavailability remains blank/NaN and makes only the detector rate incomplete.
- **No `Infinity`/`NaN` in JSON.** Strict readers must be able to parse every
  JSON artifact; nonfinite values become `null` with an explicit status.
- **Comparison CSVs differ in float precision.** `main.py` output uses `%.6e`;
  scan-level summaries use `%.17g`. Compare with care.
- **Column schemas are stable.** Do not reorder `IMRI_EVENT_COLUMNS` or rename
  compact counters casually; downstream consumers (final analysis, rate plots,
  status fractions) validate against them.
- **Never recompute physics in the output layer.** `outputs.py` consumes typed
  records; it does not recompute disk, torque, migration, GW, event-rate, or
  detectability quantities.
