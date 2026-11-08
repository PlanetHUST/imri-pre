# IMF mass-interval benchmark: 1–400 Msun (archived sensitivity model)

This is not the revised default population. See [published populations](published_populations.md) for the Pan/Xue-aligned component prescriptions and radial laws.

## Experiment

The new experiment changes the Pop III sampling interval from **13.6–197** to **1–400 Msun**, retaining dN/dm proportional to m^-1.77 and the same **total NSC mass**. It does not hold the number of objects fixed. All other benchmark inputs and the three-panel layout are unchanged:

- z=8–15, M_bullet=1e4–1e6 Msun, M_NSC=1e6–1e7 Msun;
- reference z=10, M_bullet=1e5 Msun, M_NSC=3e6 Msun;
- fixed physical NSC radius, absolute disk accretion rate and disk outer edge;
- three repeats, the same base seed, 10-Myr AGN duration;
- unchanged damping/handoff, replenishment, analytic lifetimes and fixed-mass prescription;
- unchanged nonspinning detector model, SNR threshold 20 and 1-mHz observed lower cutoff.

The minimum mass is now an explicit configuration/CLI/metadata input. The legacy IMF default remains **13.6 Msun**, so earlier seeded simulations can be reproduced explicitly; the CLI default has been restored to your original IMF model; published component presets remain opt-in. Cache checks include the lower bound and cannot reuse a default-range run for the new range.

```bash
python scripts/compare_host_rates.py --population-model legacy_popiii \
  --popiii-min-mass-msun 1 --popiii-max-mass-msun 400 \
  --output-dir outputs/imf_1_400_rates

# Isolate the upper-cutoff effect with an inexpensive reference-only control:
python scripts/compare_host_rates.py --population-model legacy_popiii \
  --popiii-min-mass-msun 13.6 --popiii-max-mass-msun 400 \
  --reference-only --output-dir outputs/imf_13p6_400_reference

python scripts/report_imf_comparison.py

# Regenerate the broad-range figure from existing results:
python scripts/compare_host_rates.py --plot-only --output-dir outputs/imf_1_400_rates
```

New figure: `outputs/imf_1_400_rates/rate_dependence.{png,pdf,svg}`. Details remain outside the figure, in `ppt_notes.md` and `imf_comparison_report.md`. Original default simulation exports and figures are retained. Each figure shares one vertical range across its three panels; when putting the default and broad figures side by side, use matched vertical limits rather than comparing apparent slopes on different ranges.

## Why an upper-only control matters

Changing both IMF bounds confounds the high-mass tail with low-mass normalization. For a fixed 3e6-Msun mass budget and slope 1.77:

| Interval [Msun] | Expected mean mass | Expected initial objects | Expected initial objects above 258.46 Msun |
|---|---:|---:|---:|
| 13.6–197 | 44.330 | 67,675 | 0 |
| 13.6–400 | 57.848 | 51,860 | 1,657 |
| 1–400 | 10.033 | 299,017 | 1,197 |

The last column is a whole-cluster expectation, **not** a crossing, capture or detection count. The broader interval adds many small objects, but has fewer very heavy objects than the upper-only extension at the same total mass. Shared seeds do not preserve object-by-object pairing after population sizes change.

## Completed benchmark results

The broad grid completed 39 realizations (13 points, three repeats), plus three upper-only reference controls. At the shared z=10, M_bullet=1e5, M_NSC=3e6 Msun reference:

| Interval [Msun] | Damping [Myr^-1] | IMRI [Myr^-1] | Detectable [Myr^-1] |
|---|---:|---:|---:|
| Default 13.6–197 | 8.300 ± 0.458 | 1.767 ± 0.273 | 0 |
| Upper-only 13.6–400 | 7.733 ± 0.291 | 2.100 ± 0.351 | 0 |
| Broad 1–400 | 18.733 ± 0.273 | 1.233 ± 0.176 | 0 |

Errors are sampling SEM, not confidence intervals. The upper-only IMRI difference is not statistically resolved by three repeats. Broadening toward low masses raises the total embedding count but does not automatically raise BH supply.

At z=8 in the broad reference-mass scan, five detections across the three realizations give **0.1667 ± 0.0667 Myr^-1**, versus zero in the default sample. The detected secondary masses are approximately 255–355 Msun; maximum SNR is 23.65. At z=10–15, the evaluated reference-mass detection counts remain zero, although maximum SNR changes with redshift.

Eight candidate evaluations across the broad grid fall below the original 13.6-Msun lower bound, and nine entries occur after the AGN interval. They are flagged in the diagnostics: the unchanged lifetime-to-BH prescription cannot establish their realistic remnant identities. All five positive detections involve secondaries above 197 Msun. These are fixed-mass numerical outcomes, not validated stellar-remnant-population rates.

See `outputs/imf_1_400_rates/imf_comparison_report.md` and the event-level diagnostic CSV for the full comparison.

## What Pan (2021) and Xue (2026) actually use

The relevant local PDFs were inspected, rather than inferring their assumptions from a figure or secondary citation:

- **Pan & Yang (2021), Formation Rate of Extreme Mass Ratio Inspirals in Active Galactic Nuclei**, PRD 103, 103018; [DOI](https://doi.org/10.1103/PhysRevD.103.103018), [arXiv:2101.09146](https://arxiv.org/abs/2101.09146), local v6, Sec. II.C.3, p. 4: two components, 1-Msun stars and 10-Msun BHs; total stellar mass 20 M_bullet and relative BH abundance delta=1e-3.
- **Pan, Lyu & Yang (2021), Wet Extreme Mass Ratio Inspirals May Be More Common For Spaceborne Gravitational Wave Detection**, [arXiv:2104.01208](https://arxiv.org/abs/2104.01208), local v2, Sec. V: the detector population uses 10-Msun secondaries and SMBH spin a=0.98.
- **Xue, Bartos, Haiman & Tagawa (2026), Recoil-regulated extreme mass-ratio inspirals in AGN disks**, [arXiv:2605.29305](https://arxiv.org/abs/2605.29305), local v1, Sec. II.A / Table I, p. 3: pre-existing BHs initially span 5–15 Msun with slope 2.35, normalized to N_BH=M_bullet/(20 Msun). Heavier BHs arise through accretion and mergers. Sec. II.F uses FastEMRIWaveforms with Kerr spin a=0.9, circular aligned orbits and sky averaging; its inner boundary corresponds to 1e-4 Hz.

These are **not** prescriptions for a stellar IMF spanning 1–400 Msun. The requested interval is therefore a user-defined sensitivity benchmark, not a reproduction of those papers. Their population normalization, growth, waveform/spin, frequency boundaries and cosmological rate normalizations differ from this calculation.

## Essential interpretation limits

The existing fixed-mass model effectively retains the sampled stellar mass and treats its analytic lifetime as a BH-formation clock. Extending its input range does not add a calibrated stellar-remnant mapping. The lifetime fit is extrapolated below 13.6 and above 197 Msun. WD/NS outcomes, winds and pair-instability disruption are not introduced. A 400-Msun progenitor is **not automatically a 400-Msun BH** in a realistic population.

The diagnostic table flags candidates below the original lower bound, candidates above the old upper bound, late entries and the masses responsible for positive detections. These outputs must be read before interpreting a formal increase in candidate/detection rate as astrophysical. Previously identified post-alignment eccentric-damping, tidal-disruption and supply/refilling limitations are unchanged.

## Suggestions

1. Present the new figure as **fixed-mass IMF-interval sensitivity**, with the upper-only reference control to separate two effects.
2. For realistic stellar-IMF predictions, validate a metallicity-dependent remnant mapping over the entire 1–400 Msun interval, including WD/NS and pair-instability outcomes.
3. For a Pan/Xue-style comparison, use separate stellar and existing-BH populations with a specified BH abundance. Produce massive BHs through a validated growth channel rather than interpreting a large initial stellar cutoff as merger growth.
4. Test slope changes separately. Xue's 2.35 is an initial **BH** mass-function slope, not automatic justification for changing the present stellar IMF slope.
5. Treat Kerr/spin waveform effects as a separate detector-model comparison. They can change band overlap substantially and should not be conflated with an IMF upper-bound effect.
6. Keep the known damping/handoff limitations visible in the main text; increasing an upper bound does not repair them.
