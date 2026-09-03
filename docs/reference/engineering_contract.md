# Engineering and Verification Contract

This document condenses the durable software, unit, persistence, and validation
rules from the superseded reconstruction and diagnostic tasks.

## 1. Ownership and dependency direction

```text
scripts and CLI
-> workflow orchestration
   -> disk/environment resources
   -> population and orbital stages
   -> damping stage
   -> migration stage
   -> event-rate stage
   -> redshift/cosmic-merger/detector stage
-> reporting and persistence

domain stages -> physical constants and typed domain records
```

Rules:

- one module owns each scientific decision;
- diagnostics and plots call production functions rather than copy equations;
- workflows coordinate named stages but do not implement formulas;
- output code translates units and schemas but does not recompute physics;
- physics modules do not import workflow, CLI, plotting, persistence, or worker
  concerns;
- importing a library module must not print, plot, create files, or build an
  expensive model;
- a script has a short `main()` and explicit `if __name__ == "__main__"` guard;
- avoid forwarding helpers and abstraction layers that only rename one call.

The detailed current source map is in [`../project/README.md`](../project/README.md).

## 2. Boundary and naming rules

Use typed records across stages. A record should carry descriptive status,
canonical-unit values, and optional error/boundary detail; do not use magic
integers, positional tuples, or an infinite value for unrelated outcomes.

Public physical names include units:

```text
mass_msun, mass_g
radius_cm, radius_rg, radius_pc
time_s, time_myr, duration_years
frequency_hz
surface_density_g_cm2
torque_si or a concrete torque unit
hubble_constant_per_myr
hubble_constant_km_s_mpc
rate_per_myr
```

Put frame and meaning before the unit suffix, for example
`chirp_mass_source_msun`, `frequency_observer_hz`, and `r_handoff_cm`.
Dimensionless names describe meaning: `mass_ratio`, `aspect_ratio`,
`efficiency_fraction`, `gap_open_mask`, `event_count`, `migration_status`.

Use one canonical in-memory unit per quantity family. Convert at the external
library boundary or output boundary exactly once and store the result under a
new unit-explicit name. Display scaling never leaks back into domain physics.

## 3. Ownership of constants and controls

Classify each behavior-affecting literal into exactly one category:

1. universal physical constant;
2. constant of one named physical model;
3. validated workflow/scenario configuration;
4. visible diagnostic experiment input;
5. output-format/presentation constant;
6. internal numerical control local to one algorithm.

One concept has one owner. Do not centralize unrelated concepts merely because
their values match. User-selected or comparison-sensitive parameters must be
visible in configuration or diagnostic constants and passed explicitly.

## 4. Orchestration and reproducibility

`IMRIDynConfig` is the validated configuration boundary. CLI and scan entry
points construct it explicitly. `run_simulation()` owns the SMBH x NSC x repeat
grid; a single-realization workflow owns the physical stage order; only damping
uses per-object multiprocessing.

For sampled work:

- derive the task seed from stable task identity;
- record base and task seeds in repeat output;
- preserve random-call order during behavior-preserving refactors;
- compare paired scenarios through independent seeded runs, not shared mutable
  realization objects.

Long runs checkpoint completed tables and write per-repeat diagnostics as soon
as a repeat finishes.

## 5. Persistence contract

Models contain typed canonical data; `src/workflows/outputs.py` owns all
presentation and persistence.

Canonical run products are:

| Artifact | Purpose |
|---|---|
| `imri_averaged_results.csv` | one row per SMBH/NSC bin averaged over repeats |
| `imri_events.csv` | retained evaluated/event-status ledger |
| `imri_repeat_results.csv` | long-form scenario, redshift, grid, seed, counters, and rates |
| `simulation_metadata.json` | configuration, derived values, provenance, runtime summary |
| `intermediate/migration_*.json` | strict per-repeat migration diagnostics |

Schema order is explicit. Output-friendly frequency scaling occurs only at
save time, and the chosen unit appears in the column name. Existing compact
column names remain compatibility surfaces even if internal typed names improve.

JSON uses `allow_nan=False`. A nonfinite physical sentinel becomes `null` plus
an explicit status. Never serialize bare `NaN`/`Infinity`, and never replace an
unavailable value with numeric zero.

## 6. Conservation invariants

Before persistence, verify:

```text
n_band_eval
  = n_entry_in_domain + n_entry_handoff
  + n_entry_inside_isco + n_profile_error

n_mig_eval
  = n_entry_in_domain + n_entry_handoff + n_profile_error
  = n_mig_finite + n_mig_trap + n_mig_error
```

For `semi_without_trap`, `n_mig_trap = 0`. Event statuses must reproduce repeat
counters, event-to-repeat keys must be unique and resolvable, and rates must
match counts divided by the configured duty cycle for complete repeats.

Detector counters have their own completeness gate. Detector unavailability
does not invalidate intrinsic conservation or the intrinsic rate.

## 7. Diagnostic-script contract

A diagnostic is a controlled scientific experiment:

1. visible constants/CLI choices with units;
2. controlled or fixed-seed inputs;
3. production adapters and production functions;
4. compact typed results;
5. independent analytical, boundary, conservation, and status assertions;
6. numerical artifacts saved beside figures when useful;
7. concise terminal report and nonzero exit on failure.

Let unexpected calculation errors surface. Catch only errors that are the
expected assertion target. Plot-only smoothing or masks operate on copies and
must not alter raw arrays, root classification, or production results.

Use exact equality for discrete state, identities, schemas, counters, and
artifacts promised unchanged. Use documented tolerances for continuous
quantities only. A diagnostic that duplicates the production equation is not
independent evidence.

## 8. Change sequence

For a behavior-preserving reconstruction:

1. inventory dirty files and capture focused baselines;
2. introduce typed/canonical-unit boundaries and adapt callers;
3. centralize one duplicated physical decision;
4. reduce orchestration after the boundary is stable;
5. update output adapters while preserving public schemas;
6. compare stage arrays, statuses, counters, and artifacts before proceeding.

Do not combine structural movement with new physics, retuned thresholds, output
schema changes, or unrelated bug repair. For an intended physics change, keep
unchanged-case regressions and validate the new cases against analytical limits,
independent references, or explicitly accepted expectations.

## 9. Verification ladder

Run checks from cheapest and narrowest to most representative:

1. direct synthetic/analytical probes;
2. affected diagnostic section or script;
3. boundary/status and conservation cases;
4. fixed-seed stage parity or expected-difference checks;
5. schema and strict JSON round trips;
6. deterministic visualization/data harness when plots change;
7. compileall, configured Pyright, and `git diff --check`;
8. reduced real pAGN workflow for workflow-level changes;
9. full grid/scan only when its cost is justified.

Canonical project commands:

```bash
/home/planet/miniconda3/envs/sevnpy_env/bin/python \
  scripts/imri_rate_visualization_test.py

/home/planet/miniconda3/envs/sevnpy_env/bin/python -m compileall src scripts
/home/planet/miniconda3/envs/sevnpy_env/bin/pyright
git diff --check
```

Set `PYTHONPATH=src` and `MPLCONFIGDIR=/tmp/matplotlib` for focused runtime or
plot checks when required. Compilation/static checks are not proof of physical
equivalence; visual similarity is not artifact parity.

## 10. Evidence handoff

For every completed part, report:

```text
Changed: source ownership or intended physical behavior
Held fixed: formulas, upstream inputs, seeds, schemas, or presentation
Checked: exact commands/harnesses and observable results
Limits: expensive or unsupported cases not exercised
```
