# Early-Universe rate comparison (archived legacy population)

This page describes the earlier `legacy_popiii` benchmark, not the revised CLI default. See [published populations](published_populations.md) for the new Pan/Xue choices.

Run the controlled single-system experiment and generate one clean three-panel figure:

```bash
python scripts/compare_host_rates.py --population-model legacy_popiii
# Replot existing data without new simulations:
python scripts/compare_host_rates.py --plot-only --output-dir outputs/high_redshift_rates
```

Outputs: `outputs/high_redshift_rates/rate_dependence.{png,pdf,svg}`. The figure has only axes, three rate curves, Monte Carlo error bars and a shared legend. Definitions, assumptions and numerical rates for the PPT main text are saved separately in `outputs/high_redshift_rates/ppt_notes.md`.

## Design

- Redshift: 8, 10, 12, 14, 15.
- SMBH mass: 1e4, 3e4, 1e5, 3e5, 1e6 Msun (logarithmic horizontal axis).
- NSC mass: 1e6, 2e6, 3e6, 5e6, 1e7 Msun (logarithmic horizontal axis).
- Reference: z=10, M_bullet=1e5 Msun, M_NSC=3e6 Msun.
- Vary only the horizontal parameter in each panel. There are 13 distinct configurations and three paired-seed repeats per configuration.
- Calibrate the physical NSC radius from the default empirical relation at the new reference NSC mass, then hold this radius fixed throughout the scan. This is not a scan along the mass-radius relation.
- Calibrate the physical disk edge from the reference Q=1 disk, then hold this radius and absolute accretion rate fixed. Other defaults (IMF, metallicity, viscosity, AGN duration, migration prescription and detector) stay fixed.

The NSC range is deliberately an uncertain massive-NSC scenario, not an observationally calibrated population prior. These comparisons do not predict cosmological host abundances.

## Rates and presentation

Damping is the number of successful stellar/BH embeddings, including replacements, divided by the 10-Myr source-frame AGN duration. IMRI is the number of physical first entries into the specified observed GW band divided by the same duration, not the ISCO merger rate. Detectable is the SNR-selected subset. All units are simply Myr^-1 within the simulated system; no volume or host-number normalization is applied.

All panels share a vertical range and symlog scale (linear below 0.1 Myr^-1, logarithmic above). True zeros are shown. Missing evaluations are gaps, not zeros. Means and sampling SEM are saved in `host_rates.csv`, with repeat data in `repeat_rates.csv`; lines are guides through measured points, not fits. Three repeats are exploratory, not convergence evidence.

## Important model boundaries

The existing 1-mHz observed lower frequency cutoff is retained. For a nonspinning 1e6-Msun SMBH, the observed ISCO frequency is only approximately 0.28–0.49 mHz at z=15–8. Such inspirals cannot reach the selected band before ISCO. The per-event workflow flags these entries as unevaluated (`isco_before_entry`), although the core aggregate CSV currently reports complete/zero for this case. The comparison script independently checks event classifications, marks the corresponding repeat rates unavailable, and saves the original aggregate statuses in `core_rate_status` and `core_detector_rate_status`. Thus this report preserves the distinction instead of plotting a fabricated evaluated zero. The original simulation exports are not modified. Damping remains evaluable. A lower observed frequency cutoff would be a separate detector-band experiment.

Fixed absolute accretion rate means the Eddington-scaled accretion rate changes inversely with SMBH mass. At 1e4 Msun it is ten times its reference value and may be outside the calibrated thin-disk regime. Likewise, a fixed outer disk radius need not remain at Q=1 when mass changes. These are conditional sensitivity calculations, not validated predictions throughout all disk regimes.

Redshift changes the observed/source frequency mapping, cosmic-time horizon, detectability and possibly replenishment eligibility; it does not evolve the assumed stellar population or disk cosmologically. Primary-only embedding rates are retained in the table to diagnose replenishment feedback.

The earlier low-redshift runs and their tables remain under `outputs/host_rate_comparison/`; the broad ten-input study remains under `outputs/parameter_dependence/`. The high-redshift experiment does not overwrite either simulation set.
