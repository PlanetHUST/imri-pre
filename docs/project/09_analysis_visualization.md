# Part 09 — Analysis and Visualization

## 1. Idea

The analysis layer turns persisted tables into scholar-facing figures and
compact, reusable outcome data. It answers three single-sentence questions:

1. **When** during one AGN episode are IMRI candidates produced?
2. **How** does the final intrinsic rate depend on redshift?
3. **How** does the final intrinsic rate depend on SMBH mass?

It also produces the diagnostic **status-fraction** taxonomy that keeps the
four classification layers (band geometry → transport → event → detectability)
visually and semantically separate.

The main idea: *one figure = one question. Status dashboards and internal
workflow terms are diagnostic, not primary presentation.*

## 2. Doing logic

### 2.1 Primary quantity

The primary plotted quantity everywhere is the intrinsic IMRI candidate rate:

```
R_IMRI = N_IMRI / T_AGN    [Myr^-1]
```

per repeat, summarized over complete repeats (mean + 16th/84th percentiles).
The cumulative AGN-time curve is

```
R_IMRI(<t) = #{candidates with gas_final_time_myr ≤ t} / (N_repeat × T_AGN)
```

using `gas_final_time_myr` as the production clock — never detection time,
damping time, or a later entry clock.

### 2.2 Figure families (rate_plots.py)

Three single-panel families, identical style, ≤ 4 curves, colorblind-safe
(Okabe-Ito), shared y-limits across paired scenarios:

- `imri_rate_vs_agn_time` — step curves per SMBH mass, endpoint must equal the
  averaged final rate;
- `imri_rate_vs_redshift` — lines per SMBH mass with markers at simulated z;
- `imri_rate_vs_smbh_mass` — log-x lines per redshift.

Valid zeros are filled markers at zero; unavailable points are `x` markers and
never connected across. Uncertainty bands require ≥ 10 valid repeats
(`MIN_REPEATS_FOR_BAND`). The physical sky-averaged LISA-selected rate is kept
in the summary data as clearly labeled secondary information but is **not**
plotted as the primary intrinsic rate.

Companion artifacts:

- `figures/data/imri_rate_time_series.csv`, `imri_rate_summary.csv`;
- `figures/figure_captions.md` with ready-to-use scholarly captions
  (NSC mass, duty cycle, repeat count, uncertainty semantics).

### 2.3 Status-fraction taxonomy

Four layers (`label_construct.md`), each with its own denominator:

| variable | raw source | denominator |
|---|---|---|
| `band_entry_status` | repeat counters | `n_band_eval` |
| `transport_status` | event `gas_stop_reason` (normalized) | persisted transport outcomes, excluding inside-ISCO and blank |
| `event_outcome` | `event_rate_class` | all event rows |
| `detectability_outcome` | `detect_rate_class` | all event rows |

Raw → presentation mapping is explicit (`STATUS_CATEGORY_LABELS`), ordering is
scientific not alphabetical (`STATUS_CATEGORY_ORDER`), colors are stable across
figures and runs (`STATUS_CATEGORY_COLORS`). Unknown raw categories raise, never
create silent legend entries. Aggregation is explicitly scan-grid (redshift
bars aggregate SMBH masses; SMBH-mass bars aggregate redshifts) and captions say
so.

### 2.4 Paired scenario comparison

`build_paired_rate_comparison` produces, per `(redshift, SMBH, NSC)` bin, both
scenario rates plus `semi_minus_torque_rate_per_myr` and a guarded ratio with a
`comparison_status` (`valid`, `partial_repeats`, `unavailable:…`,
`ratio_undefined_nonpositive_torque_rate`, `missing_scenario`). Never pool the
two models into one mean; never turn an undefined ratio into zero/infinity.

### 2.5 Validation inside the layer

- `validate_repeat_results` — identity uniqueness, `rate = count/duty_cycle`,
  NaN rate for non-complete repeats;
- `validate_event_keys` — every event resolves to one repeat;
- `validate_agn_time_endpoints` — time-curve endpoint equals averaged rate;
- `validate_status_fraction_data` — denominators, conservation, fraction sums,
  valid/unavailable states, category completeness.

## 3. Key artifacts

| Product | Contents |
|---|---|
| `figures/imri_rate_vs_agn_time.{png,pdf}` | cumulative production curve |
| `figures/imri_rate_vs_redshift.{png,pdf}` | rate vs z |
| `figures/imri_rate_vs_smbh_mass.{png,pdf}` | rate vs SMBH mass |
| `figures/data/imri_rate_summary.csv` | one row per (model, z, SMBH, NSC) bin |
| `figures/data/imri_rate_time_series.csv` | plotted cumulative curves |
| `figures/data/imri_status_fraction_summary.csv` | four-layer taxonomy by redshift |
| `figures/imri_status_fractions_vs_redshift.{png,pdf}` | stacked bars per model |
| `figures/figure_captions.md` | scholarly captions |

## 4. Caveats

- **Do not reinterpret the conditional sky-averaged LISA-selected rate as a
  cosmological detector rate** and do not plot it as a primary curve.
- **Status fractions are diagnostic.** They may include outcomes from repeats
  excluded from rate estimation; captions must state this.
- **`not evaluated` is never folded into zero/non-detectable/no-IMRI.**
- **No physics in the plot layer.** Plots consume persisted data; they never
  recompute torque, gap, Toomre-Q, GW, or classification formulas.
- **One readable axis, ≤ 4 curves, ≤ 4 legend entries.** If the scan contains
  more than four redshifts/masses, select four representative values for the
  figure and keep the full scan in the summary CSV.
- **A flat result is a valid result.** Do not claim a redshift/mass trend when
  points are identical within Monte Carlo resolution.
