# High-redshift IMRI scenario: synthesis of saved simulations

## Scientific conclusion

The saved simulations support **a conditional high-redshift IMRI formation channel**, not yet a realistic, calibrated prediction of a detectable high-redshift source or a cosmological event rate.

For the main z=10 benchmark, the analytical Type-I model produces 53 compact-object band entries across three 10-Myr realizations, but none passes the adopted orientation-averaged LISA SNR threshold of 20. Changing migration substantially changes entry yields and delays. It does not produce detections during the high-z AGN episode in the completed four-scenario comparison.

The most important next task is not to increase the apparent rate: it is to establish a physically defensible compact-object population, validate capture and stellar endpoints, and evaluate detection probability with an appropriate waveform and orientation selection.

This is the central results guide for the host/redshift, migration, disk/cluster, IMF and detector studies. It uses existing CSVs and reports; it introduces no new simulation, corrected rate, or fitted uncertainty distribution. Paths below are relative to the repository root. Equations and physical derivations are maintained in [physical mechanisms](physical_mechanisms.md), population definitions in [population prescriptions](published_populations.md), and execution details in the [run guide](quickstart.md).

## 1. What is the proposed source environment?

The archived high-redshift reference is:

| Quantity | Value |
|---|---:|
| Initial redshift | 10 |
| Central black-hole mass | 1e5 Msun |
| Total NSC mass | 3e6 Msun |
| NSC radius | 3.12184 pc |
| Density index | 1.75 |
| AGN duration | 10 Myr |
| Disk alpha | 0.1 |
| Absolute accretion rate | 1.44e23 g/s |
| Outer disk radius | 0.00322555 pc |
| Initial stellar mass prescription | 13.6–197 Msun, dN/dm proportional to m^-1.77 |
| Detector | 4-year LISA model; 1–100 mHz; averaged SNR >=20 |

This is a specific, testable scenario, but **specific does not mean observationally calibrated**. In particular, the young top-heavy population, its association with the host at z=10, remnant masses, disk structure and cluster potential need justification. The current equal-mass stellar-to-BH conversion is not realistic stellar evolution. The global NSC/central-BH mass ratio also motivates checking the enclosed cluster contribution along sampled orbits rather than assuming the Kepler approximation everywhere.

A physically motivated example source should eventually specify a BH secondary mass and origin (stellar remnant or grown BH), age, spin, embedding radius/time, entry redshift and waveform. A stellar progenitor mass alone does not determine the BH mass.

## 2. Formation budget: where are objects lost?

From `outputs/supporting/population_budget_audit/report.md`, means per reference realization:

| Stage | Objects per 10 Myr |
|---|---:|
| Initial population | 67,659 |
| Initial disk crossers | 420 |
| Primary complete embeddings | 81 |
| Total embeddings including replacements | 83 |
| Embedded stars / BHs | 65.33 / 17.67 |
| BH band entries | 17.67 |
| Detected entries | 0 |

Only 0.621% of the initial population crosses the small disk annulus, and 19.29% of crossers completes primary embedding. Thus geometry, orbital distribution and full capture are major bottlenecks. Damping is not synonymous with BH supply: most embeddings here are stars.

There are two unresolved interpretation limits:

- **Capture handoff:** 101 of the 1,260 primary tasks across the three repeats stop before the AGN deadline after inclination damping but fail the final eccentricity condition. They are correctly not counted as captures, but their rejection does not demonstrate that continued physical damping would fail. A validated continuation is needed; simply adding them to the yield would be wrong.
- **Stellar endpoints:** all 196 embedded-star non-IMRI entries precede their assigned collapse. Their estimated tidal radii exceed their nominal band-entry radii. They cannot be relabeled BH IMRIs or used to infer an additive replacement rate without disruption/survival physics.

These are structural uncertainties, not Monte Carlo error bars.

## 3. Migration uncertainty: entry numbers and delays must be separated

From `outputs/core/physical_scenario_rates/scenario_rates.csv` and `scenario_entry_diagnostics.csv`. Rates are source-frame Myr^-1 per simulated host. The uncertainties shown are three-repeat sampling SEM.

| Prescription | All entries, mean ± SEM | Entries within AGN | Detections within AGN |
|---|---:|---:|---:|
| Analytical Type-I, refill on | 1.767 ± 0.273 | 1.767 | 0 |
| Signed Type-I + thermal, ungapped | 0.667 ± 0.233 | 0.667 | 0 |
| Signed torque + continuous gap/viscous transport | 2.067 ± 0.318 | 1.833 | 0 |
| Analytical Type-I, refill off | 1.567 ± 0.273 | 1.567 | 0 |

The tested migration prescriptions span a factor 3.1 in total entry yield, or 2.75 restricting to the AGN episode. Relative to baseline, ungapped signed torques reduce the sampled yield by about 62%; gap/viscous transport increases the in-episode yield by only about 4%. This is a **scenario envelope**, not a confidence interval or proof that one prescription is correct. Trapping, stellar collapse during migration and changed replacement histories contribute; the difference cannot be assigned to a Type-II timescale alone.

Primary damping is identical in all four cases (8.1 Myr^-1). Total damping changes because replacement histories change. Turning off this finite-object refill reduces baseline entries by about 11%; it does not test external gas supply or general relaxation-driven NSC replenishment.

**Critical result:** the gap/viscous run has four detections, yielding 0.133 Myr^-1 when divided by the original AGN duration. All occur after gas shutdown, 441–9,101 Myr after initialization, at actual entry z=6.08 down to 0.38. They are not z=10 detections. Even the z≈6 event requires separate delay/host-evolution validation. Fixed-mass Gyr vacuum continuations are conditional yields, not a calibrated evolving-host prediction.

## 4. Disk and cluster sensitivity

`outputs/supporting/parameter_dependence/` is a **different benchmark**: M_NSC=2e5 Msun and R_NSC=1.30884 pc, with baseline entry rate 0.200 ± 0.115 Myr^-1. Do not splice its values into the 3e6-Msun reference as if all inputs were identical.

One-at-a-time results from its CSV:

| Varied input | 0.5 times reference | Reference | 1.5 times reference |
|---|---:|---:|---:|
| alpha | 0.167 ± 0.067 | 0.200 ± 0.115 | 0.200 ± 0.058 |
| Central BH mass | 0.300 ± 0.058 | 0.200 ± 0.115 | 0.100 ± 0.058 |
| Mdot | 0.167 ± 0.120 | 0.200 ± 0.115 | 0.233 ± 0.067 |
| Outer disk radius | 0.100 ± 0.058 | 0.200 ± 0.115 | 0.467 ± 0.176 |
| Density index gamma_n | 0.033 ± 0.033 | 0.200 ± 0.115 | 3.867 ± 0.410 |
| IMF slope gamma_m | 0.367 ± 0.203 | 0.200 ± 0.115 | 0.200 ± 0.100 |
| Maximum stellar mass | 0.267 ± 0.033 | 0.200 ± 0.115 | 0.267 ± 0.120 |
| Metallicity | 0.200 ± 0.115 | 0.200 ± 0.115 | 0.200 ± 0.115 |
| NSC radius | 0.700 ± 0.265 | 0.200 ± 0.115 | 0.100 ± 0.100 |
| NSC mass | 0.067 ± 0.067 | 0.200 ± 0.115 | 0.433 ± 0.088 |

All entries are intrinsic rates in Myr^-1 with sampling SEM. This z=15 experiment has 21 distinct one-at-a-time points plus four off-axis alpha–Mdot points completing a measured 3×3 slice: 25 configurations and 75 realizations. All saved default-model evaluation statuses are complete and all sampled detectable rates are zero. Optional migration models were not part of this completed run. These sparse, three-repeat experiments do not establish precise derivatives. Small alpha/Mdot changes are not resolved against sampling scatter. The pronounced responses to concentration and disk size identify **supply geometry as a priority**, not a validated power-law rate scaling. Changing gamma_n from 0.875 to 2.625 is a broad structural change, not a small perturbation.

The extended disk can include Q<1 gas; it is a boundary sensitivity experiment, not necessarily a self-consistent stable disk. A physical disk uncertainty study should recompute Sigma, H/R, temperature, opacity and the allowed outer edge together for each alpha/Mdot/central-mass combination, recording stability and migration validity along trajectories. Independently moving the edge is useful diagnostically but insufficient as the disk-structure error budget.

Metallicity variations here give identical outputs because all tested values remain below the model's Pop III/II switch. This is **not evidence that metallicity is physically unimportant**.

In the larger high-z host scan, increasing NSC mass from 1e6 to 1e7 Msun raises entries from 0.467 to 5.433 Myr^-1 at fixed radius. That tests increased density at fixed size, not an empirical NSC mass-radius sequence. Central-mass scans likewise keep absolute Mdot and disk radius fixed; low-mass cases can leave the thin-disk regime. The 1e6-Msun endpoint is unavailable under the selected band/entry definition, not a measured physical zero.

## 5. Why high-z detections vanish in the reference

`outputs/supporting/detectability_audit/report.md` independently checks the stored SNR integrals and replays tracks. The detector is not producing zero signal:

| Initial z | Median SNR | Maximum SNR |
|---|---:|---:|
| 8 | 12.36 | 17.40 |
| 10 | 9.78 | 13.74 |
| 12 | 7.69 | 10.78 |
| 15 | 5.26 | 7.41 |

All 53 paired reference candidates remain below 20. Independent SNR integration agrees to about 0.02%; gas-induced in-band SNR suppression is negligible in these tracks. The main limitation is the allowed secondary masses and detector band/distance, not a detector numerical failure. Increasing cluster counts does not increase an individual source's SNR.

Detector-only controls at fixed central mass 1e5 Msun require approximately 259 Msun secondaries at z=8, 418 Msun at z=10 and 1,583 Msun at z=15 to reach averaged SNR 20 under the same vacuum model. These are **conditional detector targets**, not demonstrated remnant masses or formation rates. A 1e5+1000 Msun system at z=10 is a useful detectable control, but needs a separately supported secondary-growth channel before becoming the project's realistic example source.

### IMF-interval comparison

At the shared z=10 reference, the saved IMF tests give:

| Stellar interval [Msun] | Damping | BH entry | Detectable |
|---|---:|---:|---:|
| 13.6–197 | 8.300 ± 0.458 | 1.767 ± 0.273 | 0 |
| 13.6–400, upper-only control | 7.733 ± 0.291 | 2.100 ± 0.351 | 0 |
| 1–400 | 18.733 ± 0.273 | 1.233 ± 0.176 | 0 |

Units are Myr^-1; errors are three-repeat SEM. At fixed total mass, the corresponding expected initial object counts are about 67,675, 51,860 and 299,017. Lowering the minimum adds many objects but dilutes the very massive tail relative to the upper-only extension. Shared seeds no longer give object-by-object pairing.

The broad 1–400 Msun experiment produces five detections at z=8 across three realizations: 0.167 ± 0.067 Myr^-1, with detected secondary masses about 255–355 Msun and maximum SNR 23.65. There are none at z=10–15. Across the broad grid, eight candidate evaluations are below the original lower mass bound and nine entries are delayed beyond the AGN interval. Lifetimes are extrapolated outside 13.6–197 Msun and remnant/disruption endpoints are missing. This demonstrates mass-tail sensitivity, not realistic BH production. Raising the stellar cutoff is not a substitute for modeling BH growth; the checked Pan/Xue papers do not prescribe this stellar IMF.

### Detector validation details

The audit inspects 873 candidate evaluations across configurations, including paired-source reevaluations rather than 873 independent binaries. All are finite, positive, evaluated SNRs below threshold. Explicit frequency-unit decoding, independent distance/noise/amplitude integration, and rebuilt tracks check the result; maximum track-replay discrepancy is about 2.8e-7 relative. A 1e5+100 Msun vacuum source at z=1 gives SNR 58.07, and 1e5+1000 Msun at z=8 gives 39.11: positive controls work.

Lowering only the band cutoff for a 1e5+197 Msun z=8 control changes SNR from 17.469 (1 mHz) to 17.539 (0.1 mHz), still below 20. Threshold replay also validates the classifier without changing production rates. A historical difference in solar-mass/G constants between orbit and detector modules explains a small normalization discrepancy, not the SNR deficit; no correction is applied here.

Finally, thresholding an orientation-averaged SNR is not an orientation-marginalized detection probability. The current zero count does not establish that every orientation is undetectable. Spin and waveform changes may also affect band overlap; they require explicit tests, not an assumed SNR boost.

## 6. Population alternatives are not an uncertainty distribution

Saved single-repeat pilots in `outputs/archive/populations/published_population_validation/*/host_rates.csv` give:

| Population control | Total damping | Primary damping | BH entry | Detectable |
|---|---:|---:|---:|---:|
| Pan conditional NSC normalization | 86.2 | 86.2 | 0.1 | 0 |
| Pan paper normalization | 61.3 | 61.3 | 0.1 | 0 |
| Xue BH-only proxy | 41.8 | 3.8 | 41.6 | 0 |

Units are Myr^-1. There is no estimated repeat scatter from one realization. These populations change component masses, BH abundance, radial/angular laws and normalization together, and do not reproduce the full published physics. They must not be pooled with the old IMF runs to quote a confidence interval.

The large Xue total/primary difference especially motivates inspecting donor-channel recycling before interpreting its high entry yield as fresh physical supply. The Pan examples show that many stellar embeddings need not imply many BH entries. None of these low-mass-BH controls solves the high-z SNR deficit by population size alone.

## 7. What uncertainty can actually be reported?

Keep four distinct categories:

1. **Sampling:** three-repeat SEM for a fixed model. Baseline intrinsic SEM is about 15% of its mean; more repeats and paired difference statistics are needed for precise comparisons.
2. **Numerics:** damping continuation, deadline overshoot, ODE tolerances/events and resolution. Validate before claiming physical sensitivity. Radau resolves torque-trap stiffness; it does not validate the torque law itself.
3. **Physical scenarios:** population/remnants, cluster geometry/potential, disk structure, migration/traps, stellar disruption and supply. Current comparisons sample alternatives without assigning probabilities to them.
4. **Selection/cosmology:** orientation, spin/waveforms, mission window, source-to-observer time, host abundance and delays. These are not encoded in the plotted SEM.

A zero sampled detection mean with zero SEM is not a zero upper confidence limit. Under a simple stationary Poisson approximation only, zero detections over 30 host-Myr would imply a one-sided 95% limit of -ln(0.05)/30 ≈0.100 Myr^-1. Finite-stock simulations need not be Poisson, and this illustrative limit describes the adopted selection/model, not a physical all-orientation or cosmological bound.

The archived quantity N_entry/T_AGN is best called a **per-episode band-entry yield divided by source-frame duty duration**, especially when delayed entries are included. A cosmic detection rate requires host/episode abundance, duty-cycle and delay distributions, comoving volume, observer-time conversion by 1/(1+z_entry), and a selection probability. A mission detection probability also needs occupancy/observation-window modeling; the present rate alone is insufficient.

## 8. Recommended project focus

A defensible near-term claim is:

> A specified z≈10 AGN/NSC benchmark can deliver compact objects into the LISA band. Its yield is sensitive to orbital supply and signed migration physics, while the tested <=197-Msun secondaries remain below the adopted averaged-SNR threshold. Delayed detections and optimistic massive-secondary controls do not yet establish detectable IMRIs during the z≈10 episode.

Priority sequence:

1. Select and justify one host, epoch, age distribution and compact-object origin. Include a validated remnant mapping or explicit pre-existing BH reservoir; do not equate all progenitors with BHs.
2. Validate full capture continuation and finite-star disruption/consumption before rerunning rates.
3. Recompute self-consistent disk families and compare migration prescriptions on the **same** population, with trajectory-weighted applicability diagnostics.
4. Save distributions of secondary mass, embedding radius/time, entry time/redshift and SNR, separating in-episode from delayed entries. Increase repeats and report paired differences.
5. For a realistic detectable example, test a justified massive-secondary channel or justified spin/waveform model, then integrate over orientations. Do not lower the threshold merely to manufacture detections.
6. Only after that, attach a host-population/delay model to estimate a cosmic detection rate or mission probability.

## 9. Reproduce and inspect the experiments

Use distinct output directories and check metadata before rerunning. Historical CSVs, figure notes and full audit reports remain in `outputs/`; they are the detailed evidence, not additional competing narrative guides.

### Host/redshift comparison

```bash
python scripts/compare_host_rates.py --population-model legacy_popiii \
  --output-dir outputs/core/high_redshift_rates
python scripts/compare_host_rates.py --plot-only --output-dir outputs/core/high_redshift_rates
```

The design has 13 distinct configurations and three repeats: z=8,10,12,14,15; central masses 1e4,3e4,1e5,3e5,1e6 Msun; NSC masses 1e6,2e6,3e6,5e6,1e7 Msun. Only the panel's horizontal parameter changes. Physical disk/NSC radii and absolute Mdot are fixed at the reference. `host_rates.csv`, `repeat_rates.csv`, `experiment.json`, `ppt_notes.md` and `rate_dependence.{png,pdf,svg}` preserve means, scatter, provenance and presentation settings. Shared symlog axes are linear below 0.1 Myr^-1; lines are guides, not fits.

At the 1e6-Msun endpoint, event flags say `isco_before_entry` even though historical core aggregates report zero/complete. The comparison runner preserves the original core statuses separately and marks the event-derived rates unavailable. Do not infer a physical zero from the raw aggregate alone. Earlier low-z outputs remain under `outputs/archive/historical/host_rate_comparison/`.

### Second central-mass benchmark: 1e4 Msun

```bash
python scripts/compare_low_mass_redshift.py
python scripts/compare_low_mass_redshift.py --plot-only
```

`outputs/core/low_mass_redshift/` adds three-repeat simulations at z=8,10,12,14,15 for a 1e4-Msun central BH and the same total 3e6-Msun NSC. Population, NSC radius, duty time, migration/refill and LISA settings are copied from the original reference. Absolute Mdot is reduced to 1.44e22 g/s following the standard mass scaling, and the disk edge is recalibrated from that host's disk before being held fixed across redshift. This is a second host benchmark, **not** the old central-mass scan's fixed-Mdot/fixed-edge point or a pure central-mass derivative.

`rate_dependence.{png,pdf,svg}` compares both central masses versus redshift in shared-scale embedding, IMRI-entry and detectable-rate panels. `snr_vs_redshift.{png,pdf,svg}` compares median/10–90% candidate SNR and maximum SNR, retaining the threshold 20. `host_rates.csv`, `repeat_rates.csv`, `snr_summary.csv`, `candidate_snr.csv` and `report.md` retain measured values, scatter and delayed-entry counts. Full new runs are under `simulations/z_*/`; the old 1e5-Msun runs are reused, not overwritten.

At z=10 the new benchmark gives embedding rate 12.93 ± 0.73 Myr^-1 and BH-entry rate 5.40 ± 0.40 Myr^-1, compared with 8.30 ± 0.46 and 1.77 ± 0.27 in the old host. Its candidate median/max SNR is 6.52/10.14, compared with 9.78/13.74. Thus higher entry yield does not imply higher individual SNR or threshold-passing detections. These differences include the recalibrated disk, not central mass alone.

For candidate-population properties, `snr_population_scatter.{png,pdf,svg}` in the same folder compares SNR versus secondary mass and versus actual observation redshift. Blue circles mark 1e4-Msun central BHs; orange triangles mark 1e5-Msun central BHs; SNR 20 is a dashed reference, not a sample cut. Regenerate without rerunning physics:

```bash
python scripts/compare_low_mass_redshift.py --scatter-only
```

All 1,075 evaluated points are shown: 810 from the 1e4 benchmark and 265 from the 1e5 benchmark. They represent 162 and 53 paired (repeat, source_id) groups evaluated across five initial redshifts, not 1,075 independent binaries. The redshift coordinate is `snr_observation_redshift`, not the initial host label; subthreshold points are candidates, not claimed detections. The mass panel mixes redshifts, so a mass-only trend should be checked within initial-z groups. Counts/selection are saved in `snr_population_scatter_summary.json`.

The larger NSC/central-mass ratio and secondary mass ratios up to about 0.02 make the Kepler potential, linear migration and first-order waveform assumptions especially important limitations. The new run does not validate those approximations or introduce realistic stellar remnants.

### Third central-mass benchmark: 1e6 Msun

```bash
python scripts/compare_low_mass_redshift.py --mass 1e6 \
  --comparison-dir outputs/core/low_mass_redshift
python scripts/compare_low_mass_redshift.py --plot-only \
  --output-dir outputs/core/high_mass_redshift
```

`outputs/core/high_mass_redshift/` adds the same three-repeat, five-redshift study for a 1e6-Msun central BH and 3e6-Msun NSC. Its mass-scaled Mdot is 1.44e24 g/s and its independently recalibrated disk edge is 1.61380e16 cm. Across the five redshifts the measured embedding rate is 17.03 ± 1.04 Myr^-1 (sampling SEM), including stars and BHs. The comparison CSVs and figures include all three central masses, reusing the earlier 1e4/1e5 runs unchanged. The original fixed-disk mass-axis point is not reused as this new host.

**Band limitation:** the nonspinning observed ISCO frequency is approximately 0.49–0.28 mHz over z=8–15, below the retained 1-mHz lower edge. The workflow's `isco_before_entry` classifications are preserved as **N/A** for intrinsic entry rate, detectable rate and SNR; the embedding rate remains measurable. The figures annotate these missing curves rather than plotting false zeros, and `snr_summary.csv` records unavailable-event counts. The original raw aggregate zero/complete exports are not sufficient evidence of an evaluated physical zero. A lower-frequency-band or spinning-BH study requires a separate experiment, not a silent change here.

### Migration and refill comparison

```bash
python scripts/compare_physical_scenarios.py
python scripts/compare_physical_scenarios.py --plot-only
```

The runner copies the archived reference disk/population/detector controls and reuses the compatible baseline rather than rescaling it. `scenario_rates.csv` and `scenario_repeats.csv` retain scatter and statuses; `scenario_entry_diagnostics.csv` separates delayed/in-episode entries; `representative_torque_profiles.csv` records an **unweighted** radial/mass diagnostic grid, not event-weighted validity fractions. Per-case simulations/logs and `experiment.json` preserve provenance. The initial slow explicit torque attempt is retained as provenance, not a zero-yield result.

### Four-scenario SNR population grid

```bash
python scripts/compare_scenario_populations.py --jobs 4 --workers 1
python scripts/compare_scenario_populations.py --plot-only
```

`outputs/core/scenario_populations/` extends the same four prescriptions to both 1e4/1e5-Msun hosts and all five initial redshifts: 40 configurations, three repeats each. Compatible baselines and the existing 1e5-Msun z=10 alternatives are reused; other configurations receive new full simulations. Each alternative copies its exact host/redshift baseline disk, population and detector controls, with identical primary damping records checked. No mass/frequency/threshold adjustment is made to produce detections.

Each scenario subfolder has the same two-panel `snr_population_scatter` (SNR–secondary mass and SNR–actual observation redshift), plus `rate_dependence`, `snr_vs_redshift` and underlying candidate/repeat tables. Blue circles and orange triangles distinguish central masses. `scenario_rates.csv` keeps total and within-AGN detectable rates, delayed counts and entry-redshift ranges; loud delayed events must not be interpreted as detections during the initial high-z episode. `experiment.json` records each run's completion/reuse/failure status and original source path. Read its status before treating this grid as complete; pending/failed work is never an evaluated zero.

The per-scenario point clouds are alternative prescriptions for paired populations, not additive physical populations. Their visible densities are not normalized detection probabilities, and redshift/scenario reevaluations are not independent binaries. Detailed results and the z=10 comparison table are in `outputs/core/scenario_populations/report.md`.

### Disk/cluster sensitivity

```bash
python scripts/test_parameter_dependence.py --repeats 3 --workers 1
python scripts/test_parameter_dependence.py --plot-only
# Optional new multi-prescription experiment, not the completed default run:
python scripts/test_parameter_dependence.py \
  --models semi_type_i torque_with_trap semi_without_trap \
  --output-dir outputs/parameter_models
```

`outputs/supporting/parameter_dependence/{experiment.json,parameter_dependence.csv}` records all dimensional inputs/factors and statuses; per-point outputs retain logs. The historical heatmap was removed from the saved presentation but can be regenerated. ±50% denotes a parameter range, not a Gaussian prior or variance; only alpha–Mdot interactions were sampled.

### IMF intervals

```bash
python scripts/compare_host_rates.py --population-model legacy_popiii \
  --popiii-min-mass-msun 1 --popiii-max-mass-msun 400 \
  --output-dir outputs/archive/imf/imf_1_400_rates
python scripts/compare_host_rates.py --population-model legacy_popiii \
  --popiii-min-mass-msun 13.6 --popiii-max-mass-msun 400 \
  --reference-only --output-dir outputs/archive/imf/imf_13p6_400_reference
python scripts/report_imf_comparison.py
```

The broad grid has 39 realizations and the upper-only reference has three. The report and event diagnostics retain mass-bound, delayed-entry and remnant-interpretation flags. Cache identity includes the minimum mass. For published-population control commands and source references, see [population prescriptions](published_populations.md).

### Detector and population audits

```bash
python scripts/check_detect_process.py
PYTHONPATH=src python -m pytest tests/core/test_detect_process.py -q
python scripts/check_population_budget.py
```

The detector audit reclassifies/replays saved candidates without changing production outputs or threshold. Artifacts include `candidate_audit.csv`, `paired_redshift_snr.csv`, `track_replay.csv`, `detector_controls.csv`, `threshold_sensitivity.csv` and `checks.json` under `outputs/supporting/detectability_audit/`. Threshold sensitivity is conditional reclassification, not new physical simulations.

The population audit reconstructs the seeded populations and reruns reference primary damping tasks, not a corrected full-rate simulation. `population_census.csv`, `primary_damping_replay.csv`, `embedded_stellar_timing.csv`, `geometry_only_sensitivity.csv` and `checks.json` under `outputs/supporting/population_budget_audit/` preserve the budget and handoff checks. Geometry-only changes are supply diagnostics, not re-solved disks or capture rates.

### Principal evidence

- `outputs/core/high_redshift_rates/{host_rates.csv,ppt_notes.md}`
- `outputs/core/physical_scenario_rates/{scenario_rates.csv,scenario_entry_diagnostics.csv,ppt_notes.md}`
- `outputs/supporting/parameter_dependence/{experiment.json,parameter_dependence.csv}`
- `outputs/supporting/population_budget_audit/report.md`
- `outputs/supporting/detectability_audit/report.md`
- `outputs/archive/imf/imf_1_400_rates/imf_comparison_report.md`
- `outputs/archive/populations/published_population_check/report.md`
- `outputs/archive/populations/published_population_validation/*/host_rates.csv`

The [output index](../outputs/README.md) documents the core/supporting/archive layout and cross-folder dependencies. Archived experiments are retained evidence, not calibrated predictions or expendable test fixtures.
