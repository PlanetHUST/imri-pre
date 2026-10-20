# Active Plan: Physical LISA Detectability

This completed implementation record remains at its original path for
compatibility. Older `detect_rate.py`/`lisa_snr.py` names in the milestone log
refer to the detector now owned by `src/lisa/detectability.py` and
`src/lisa/snr.py`. See [the current source map](../../project/README.md).

Status: completed on 2026-09-02

Governing specification: `docs/specs/lisa_detectability.md`

Created after repository inspection: 2026-09-02

## Goal

Replace the threshold-valued `snr_proxy` fallback with an auditable,
mission-limited, sky-averaged circular-inspiral LISA SNR while preserving the
intrinsic IMRI formation, migration, and rate contract.

The implementation followed the later continuous strategy: validate one full
source-to-SNR demonstration first, then integrate the same kernel.

## Governing constraints

- `event_rate.py` remains the intrinsic formation owner.
- `detect_rate.py` remains the detector-classification owner.
- Existing event-entry frequency defaults remain unchanged during this task.
- Band-entry, profile-error, trap, and detector statuses remain distinct.
- Intrinsic rates remain available when only detector SNR is unavailable.
- Source masses and source orbital radii remain canonical.
- `lisa_snr` becomes authoritative.
- The dirty worktree is user-owned; unrelated changes are not edited, reset, or
  absorbed.

## Milestone summary

| Milestone | Objective | Status | Depends on |
|---:|---|---|---|
| 1 | Establish detector interfaces and resolve semantics | complete | specification |
| 2 | Implement and validate LISA PSD | complete | 1 |
| 3 | Implement and validate luminosity distance | complete | 1 |
| 4 | Implement source/detector mass-frequency handling | complete | 1, 3 |
| 5 | Implement inspiral waveform and SNR kernel | complete | 2, 3, 4 |
| 6 | Implement mission-limited frequency window | complete | 4, 5 |
| 7 | Integrate with event classification | complete | 5, 6 |
| 8 | Propagate configuration, schema, outputs, and metadata | complete | 1, 7 |
| 9 | Run physical validation cases | complete | 2-8 |
| 10 | Run repository regression verification | complete | 9 |

## Milestone 1 — Establish detector-model interfaces and semantics

### Objective

Freeze numerical interfaces, statuses, cosmology ownership, mission-window
semantics, circular-waveform behavior, and compatibility policy before coding.

### Files likely to change

- `docs/specs/lisa_detectability.md`
- this active plan
- no production files until the decisions are complete

### Interfaces affected

- proposed physical-kernel result, such as `LISASNRResult`;
- `DetectRateDecision` additions;
- `classify_detectability()` detector arguments;
- detector statuses and completeness counters;
- configuration names/defaults;
- `snr_proxy` compatibility policy.

### Acceptance criteria

- Every unresolved specification item has a recorded decision.
- The `16/5` convention and matching RCL effective PSD are fixed together.
- No status maps unavailable/failure to numeric zero.
- Intrinsic and detector completeness have separate conservation rules.
- The source-aligned mission window is explicit.

### Validation command

```bash
rg -n "Unresolved|Decision|lisa_snr|snr_proxy|16/5" \
  docs/specs/lisa_detectability.md \
  docs/plans/active/lisa_detectability_plan.md
```

### Dependencies

The authoritative specification. Every later milestone depends on this one.

## Milestone 2 — Implement and validate the LISA PSD

### Objective

Implement RCL instrument noise, finite-arm response, mission-duration Galactic
foreground, and total two-channel effective strain PSD with explicit units and
provenance.

### Files likely to change

- new `src/physical_process/lisa_sensitivity.py`;
- `src/physical_process/constants.py` only for genuinely shared constants;
- new `scripts/lisa_detectability_test.py`.

### Interfaces affected

- instrument-noise PSD;
- Galactic-confusion PSD;
- total effective strain PSD;
- supported mission-duration lookup;
- sensitivity model identifier.

### Acceptance criteria

- Inputs are observed Hz; outputs are `Hz^-1`.
- PSD and ASD terminology is correct.
- Supported curves are positive and finite over the configured band.
- Unsupported durations fail; no silent interpolation occurs.
- The 4-year 1 mHz benchmark passes independent review and tolerance.

### Validation command

```bash
/home/planet/miniconda3/envs/sevnpy_env/bin/python \
  scripts/lisa_detectability_test.py --section psd
```

### Dependencies

Milestone 1.

## Milestone 3 — Implement and validate luminosity distance

### Objective

Provide unit-explicit flat-LambdaCDM luminosity distance and, if chosen in
Milestone 1, shared cosmic-age and inverse-redshift functions.

### Files likely to change

- new `src/physical_process/cosmology.py`, or deliberately expanded
  `src/workflows/redshift_evolve.py` if it remains the single owner;
- `scripts/lisa_detectability_test.py`.

### Interfaces affected

- `luminosity_distance_mpc(redshift, ...)`;
- possibly cosmic age, inverse redshift, and remaining time;
- detector cosmology configuration/provenance.

### Acceptance criteria

- `H0` units cannot be confused with the current per-Myr parameter.
- `D_L(0)=0`; positive-redshift distances are finite, positive, and monotone.
- Benchmarks at `z=0.1`, `1`, `10`, and `15` agree with an independent
  reference under identical assumptions.
- If time functions are implemented, `z -> t -> z` round trips pass.
- Radiation and flatness choices are persisted.

### Validation command

```bash
/home/planet/miniconda3/envs/sevnpy_env/bin/python \
  scripts/lisa_detectability_test.py --section cosmology
```

### Dependencies

Milestone 1 cosmology decisions.

## Milestone 4 — Implement source/detector mass-frequency handling

### Objective

Add unit-explicit chirp-mass, redshifted-mass, observed-ISCO, and frame-
consistency helpers without changing source-frame orbital helpers.

### Files likely to change

- `src/physical_process/gw_frequency_radius.py` for general GW helpers, or the
  detector module for detector-only helpers;
- `scripts/lisa_detectability_test.py`.

### Interfaces affected

- source chirp mass;
- detector chirp and total mass;
- observed Schwarzschild-ISCO frequency;
- frame round-trip diagnostics.

### Acceptance criteria

- Source masses remain canonical and detector masses are derived once.
- Both observed-ISCO paths agree.
- A targeted regression fails under double redshifting.
- Existing `circular_band_boundary()` behavior is unchanged.

### Validation command

```bash
/home/planet/miniconda3/envs/sevnpy_env/bin/python \
  scripts/lisa_detectability_test.py --section frames
```

### Dependencies

Milestones 1 and 3.

## Milestone 5 — Implement the inspiral waveform and SNR kernel

### Objective

Implement the leading stationary-phase circular inspiral amplitude and
orientation-averaged SNR integral independently of workflow classification.

### Files likely to change

- detector module from Milestone 2;
- `scripts/lisa_detectability_test.py`.

### Interfaces affected

- Fourier amplitude `A(f)`;
- SNR integrand;
- typed `LISASNRResult` with value, status, distance, bounds, and provenance;
- bounded numerical-failure behavior.

### Acceptance criteria

- Amplitude uses detector chirp mass, observed frequency, and distance once.
- SNR uses `16/5` with the matching PSD and no duplicate response factor.
- Fixed-frame SNR scales as `1 / D_L`.
- The squared integrand is nonnegative and meets the selected convergence goal.
- Invalid inputs are unavailable; numerical failures are not zero.

### Validation command

```bash
/home/planet/miniconda3/envs/sevnpy_env/bin/python \
  scripts/lisa_detectability_test.py --section snr
```

### Dependencies

Milestones 2-4.

## Milestone 6 — Implement the mission-limited frequency window

### Objective

Calculate ordered observed-frequency bounds from the configured band, actual
source state, observer mission duration, and observed ISCO.

### Files likely to change

- detector module;
- `src/workflows/detect_rate.py` only to supply actual source state;
- `scripts/lisa_detectability_test.py`.

### Interfaces affected

- time-frequency and inverse-frequency helpers;
- window result/status;
- current observed frequency input;
- effective observation duration.

### Acceptance criteria

- Full windows reproduce the configured observer duration.
- Already-in-band sources begin at actual current frequency; no outward reset
  to formal entry occurs.
- Entry-during-window and early-ISCO cases use shorter durations.
- `f_lower >= f_upper` maps only to physical no-overlap and SNR zero.
- Bounds are observed-frame and within the configured band.

### Validation command

```bash
/home/planet/miniconda3/envs/sevnpy_env/bin/python \
  scripts/lisa_detectability_test.py --section window
```

### Dependencies

Milestones 4 and 5.

## Milestone 7 — Integrate with event classification

### Objective

Replace `single_frequency_snr_proxy()` in the positive-candidate path with the
validated kernel while preserving gas/GW/cosmic-merger routing and upstream
unavailability.

### Files likely to change

- `src/workflows/detect_rate.py`;
- `src/workflows/event_work.py`;
- classification cases in `scripts/lisa_detectability_test.py`.

### Interfaces affected

- `classify_detectability()`;
- `DetectRateDecision`;
- raw detector statuses/reasons;
- `unsupported_band_entry_decision()`;
- non-candidate compatibility behavior.

### Acceptance criteria

- Above/below threshold use finite `lisa_snr`.
- No-overlap, unavailable, failure, and upstream-not-evaluated stay distinct.
- No candidate, migration, or event-rate rule changes.
- Inside-ISCO entry still bypasses migration and detector kernels.
- `snr_proxy`, if retained, only aliases `lisa_snr`.
- Detection redshift and actual available source state remain consistent after
  a simplified cosmic merger.

### Validation command

```bash
/home/planet/miniconda3/envs/sevnpy_env/bin/python \
  scripts/lisa_detectability_test.py --section classification
```

### Dependencies

Milestones 5 and 6.

## Milestone 8 — Propagate configuration, schema, outputs, and metadata

### Objective

Make the detector model reproducible in every layer and separate intrinsic-rate
completeness from detector completeness.

### Files likely to change

- `src/workflows/config.py`
- `src/workflows/cli.py`
- `src/workflows/models.py`
- `src/workflows/event_work.py`
- `src/workflows/outputs.py`
- `src/workflows/metadata.py`
- `src/workflows/reporting.py`
- `scripts/redshift_scan.py`
- `scripts/final_analysis.py` if it reports SNR/status diagnostics
- `scripts/imri_rate_visualization_test.py` where fixtures need new fields
- `docs/project/06_cosmology_detectability.md`
- `docs/project/08_data_io.md`

### Interfaces affected

- `IMRIDynConfig` and CLI;
- strict redshift-scan template;
- event model and result counters/rates;
- event/repeat/averaged CSVs;
- diagnostic/metadata JSON;
- runtime reporting and downstream analysis.

### Acceptance criteria

- Every detector parameter and model identifier is persisted.
- Event records contain authoritative SNR and audit diagnostics.
- Intrinsic and detector completeness are independent and conserved.
- Unavailable detector rates serialize as NaN/blank and JSON `null`, not zero.
- Frequency diagnostics use save-time unit formatting.
- Historical metadata compatibility is explicit.
- Downstream consumers do not use old proxy semantics.

### Validation commands

```bash
/home/planet/miniconda3/envs/sevnpy_env/bin/python scripts/main.py --help
/home/planet/miniconda3/envs/sevnpy_env/bin/python scripts/redshift_scan.py --help
/home/planet/miniconda3/envs/sevnpy_env/bin/python \
  scripts/lisa_detectability_test.py --section schema
```

### Dependencies

Milestones 1 and 7.

## Milestone 9 — Run physical validation cases

### Objective

Validate the complete detector path across mass, redshift, threshold, and
status regimes before interpreting rate changes.

### Files likely to change

- `scripts/lisa_detectability_test.py`;
- small deterministic fixtures only if justified;
- no production changes unless a validation failure demonstrates a defect.

### Interfaces affected

All detector interfaces and persisted diagnostics.

### Acceptance criteria

- PSD, cosmology, frame, window, SNR, status, and schema sections pass.
- Representative IMRIs are evaluated at low redshift and `z=10`, `13`, `15`.
- Controlled increasing-distance cases suppress SNR.
- ISCO truncation and upstream inside-ISCO censorship remain explicit.
- Paired-seed before/after comparison leaves intrinsic outcomes unchanged.
- Numerical convergence is demonstrated.

### Validation command

```bash
/home/planet/miniconda3/envs/sevnpy_env/bin/python \
  scripts/lisa_detectability_test.py
```

Also run a one-point seeded smoke below `/tmp`; select its mass/redshift only
after unit tests establish a known positive or negative branch.

### Dependencies

Milestones 2-8.

## Milestone 10 — Run repository regression verification

### Objective

Prove clean integration without altering unrelated physics or user-owned work.

### Files likely to change

None. Any required fix returns to the responsible earlier milestone.

### Interfaces affected

Repository-wide import, type, output, and visualization contracts.

### Acceptance criteria

- Focused detector and existing visualization tests pass.
- Source and scripts compile; Pyright passes.
- CLI help exposes explicit detector defaults and units.
- Strict JSON/CSV smoke artifacts pass schema checks.
- `git diff --check` passes.
- Final diff contains no unrelated changes or generated artifacts.

### Validation commands

```bash
MPLCONFIGDIR=/tmp/matplotlib \
  /home/planet/miniconda3/envs/sevnpy_env/bin/python \
  scripts/imri_rate_visualization_test.py
/home/planet/miniconda3/envs/sevnpy_env/bin/python -m compileall src scripts
/home/planet/miniconda3/envs/sevnpy_env/bin/pyright
git diff --check
```

### Dependencies

Milestone 9.

## Completed milestones

- [x] Inspected the current detectability call path and behavior.
- [x] Inspected configuration, CLI, metadata, models, outputs, reporting, and
  redshift-scan propagation.
- [x] Searched production references to `snr_proxy`, threshold, LISA entry
  frequencies, detector labels, and detector booleans.
- [x] Confirmed there is no focused detector-physics harness.
- [x] Recorded the authoritative mathematical/status specification.
- [x] Created this dependency-ordered implementation plan.

All production implementation milestones are complete. The checklist above is
retained as the audit trail from the original planning pass.

## Scientific and architectural decisions

- Detector selection remains downstream of intrinsic candidate formation.
- Existing entry-frequency defaults remain unchanged.
- RCL two-channel effective PSD and `16/5` SNR convention are paired.
- `lisa_snr` is authoritative.
- Source masses remain canonical and detector masses are derived once.
- SNR frequencies and mission duration are observer-frame.
- The first waveform is circular, nonspinning, quadrupole, inspiral-only.
- The window is conditional/source-aligned, not calendar-aligned.
- Physical no-overlap is zero; unavailability/failure is unavailable.
- Detector numerics are helpers; `detect_rate.py` owns classification.

## Resolved questions

The event-time model remains separate from detector flat-LambdaCDM distance;
radiation is omitted; nonzero eccentricity is detector-unavailable;
non-candidates are not evaluated; `snr_proxy` is a deprecated alias; and the
adaptive quadrature tolerances are `1e-11` for distance and `1e-8` for SNR.

These decisions are implemented and recorded in detector metadata. They are
not open gates for the current circular, nonspinning population model.

## Validation results

Implementation validation:

- Standalone demo SNR: `58.07069322689072` for `1e5+100 Msun`, `z=1`, 4 yr.
- RCL 4-year PSD at 1 mHz: `8.296514847664091e-38 Hz^-1`.
- Log-grid and adaptive SNR integrations agree to better than `2e-9` relative.
- Fixed-frame inverse-distance SNR ratio: `2.000000000000` for 1 versus 2 Gpc.
- Controlled 96-case mass-redshift grid through `z=15`: no
  numerical/model failures.
- Seeded `z=1` workflow: two intrinsic candidates, two physical detections.
- Paired-seed `z=10` workflow: one intrinsic candidate, finite SNR `12.43405`,
  below threshold, detector schema complete.
- Visualization, compileall, Pyright, strict JSON/CSV, and diff checks are part
  of the final verification record.

## Pre-implementation issues discovered during planning

All detector-specific items below were resolved by this implementation unless
they are explicitly retained as an upstream model limitation.

1. `snr_proxy` is a threshold-valued band flag, not an SNR.
2. Production has no luminosity distance, mission duration, LISA PSD, waveform,
   or integrated SNR.
3. `lisa_min_gw_frequency_hz` controls both intrinsic entry geometry and
   detector band membership; changing it alters intrinsic rates.
4. `band_entry_inside_isco` bypasses detectability upstream and cannot be
   reinterpreted without a separate entry-contract change.
5. When Peters time is zero for an already-in-band source, `detect_rate.py`
   later resets `final_radius_cm` to formal entry; SNR must retain actual state.
6. Current completeness combines intrinsic and detectable rates and has no
   detector-unavailable/failure accounting.
7. Non-candidates carry `non-detectable`, `False`, and proxy zero, although
   counters avoid adding them to non-detectable IMRIs.
8. `gw_eccentricity` permits nonzero values while the target waveform is
   circular.
9. Current redshift time uses `(1+z)=(H0 t)^(-2/3)`, not target flat-LambdaCDM
   distance/time physics.
10. Several typed detector radii/frequencies are hidden or commented out of the
    event CSV, so diagnostics must enter through `outputs.py` deliberately.
11. The visualization harness validates status presentation, not detector
    physics.
12. `final_analysis.py` consumes the detector boolean rather than SNR.

## Discrepancies from the original conversational plan

1. The current redshift owner is a closed-form time-redshift model, not simple
   linear subtraction in the present repository. Retain/replace must be a
   decision rather than an assumption.
2. The change cannot live only in `detect_rate.py`; configuration,
   `event_work.py`, models, counters, outputs, metadata, reporting, the strict
   redshift-scan template, analysis, and validation fixtures are affected.
3. Reusable PSD/SNR numerics belong in a testable physical helper rather than
   making the workflow classifier own every equation.
4. The lower frequency also controls intrinsic entry and migration routing, so
   its default remains unchanged here.
5. Upstream inside-ISCO events never reach `classify_detectability`; they remain
   unavailable under this task rather than receiving low SNR.
6. Output work includes separating detector completeness from intrinsic
   completeness, not only adding event diagnostics.
7. Mission-window semantics need actual current frequency for already-in-band
   sources, which the current final-radius assignment does not preserve.
8. There is no focused detector test; a standalone harness is required.
9. The repository had no `docs/specs/` or `docs/plans/active/` structure. This
   task introduces them; the older empty `docs/plan/` is not an execution-plan
   system.

## Next action

The immediate priority is population-level validation of the existing
circular, nonspinning selection: repeated-seed convergence, threshold-edge
counts, and the distinction between conditional per-host yield and a
cosmological detection rate. Eccentricity and spin are not the next automatic
development step because the current Monte Carlo does not evolve either
quantity. Any waveform extension should follow evidence from that population
audit and corresponding source-model variables.
