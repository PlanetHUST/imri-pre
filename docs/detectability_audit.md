# Audit of the archived zero high-redshift detection rates

This report concerns the original `legacy_popiii` runs. It is not a rate prediction for the revised [published population presets](published_populations.md).

## Result

There is a real redshift response in the detector calculation, hidden by the hard SNR threshold. The audit finds no detector failure explaining the zero counts.

For the 53 matched candidate sources in the reference-mass redshift scan:

| Initial redshift | Median SNR | Maximum SNR | Above threshold 20 |
|---:|---:|---:|---:|
| 8 | 12.364 | 17.397 | 0 |
| 10 | 9.780 | 13.740 | 0 |
| 12 | 7.691 | 10.783 | 0 |
| 14 | 5.983 | 8.399 | 0 |
| 15 | 5.258 | 7.407 | 0 |

All matched sources' SNRs decrease with redshift. Across all configurations, all **873 candidate evaluations** have finite positive SNR, `evaluated` status, and below-threshold rejection reasons. This count includes repeated evaluations of paired sources, not 873 independent binaries. The unevaluated 1e6-Msun case is excluded, not treated as a detector zero.

A fixed population can therefore have identical damping and intrinsic-entry counts but different SNR distributions. Applying a hard threshold can yield a flat zero detection-rate curve despite that response. Increasing NSC mass supplies more sources, not a larger individual-source SNR.

## Tests performed

```bash
python scripts/check_detect_process.py
PYTHONPATH=src python -m pytest tests/core/test_detect_process.py -q
```

The diagnostic does not rerun the sampled population or change simulation outputs, thresholds or rates.

1. Inspect every saved candidate's rejection reason, SNR status, secondary mass, observation duration, distance, redshift, detector-frame chirp mass and frequency bounds. Decode the output formatter's Hz/mHz/micro-Hz suffixes explicitly.
2. Check cosmological distances with an independent Gauss-Legendre integral and detector-frame masses with explicit single redshifting.
3. Independently implement the four-year RCL noise curve and SPA amplitude, integrating SNR with log-frequency Simpson quadrature. Compare the same measured frequency bounds, with the gas/GW residence-time ratio included where gas is active. Maximum relative SNR discrepancy is about **1.90e-4 (0.019%)**.
4. Rebuild the brightest candidate's in-band track at each of five redshifts and rerun the production detector classifier. Maximum replay discrepancy is approximately **2.8e-7** relative, consistent with exported-data rounding.
5. Positive detector-only controls: a 1e5+100 Msun vacuum system at z=1 gives SNR **58.07** and is detected; 1e5+1000 Msun at z=8 gives **39.11**, also detected. These are controls, not simulated rate predictions.
6. Change only the threshold in replay tests. The brightest z=8 event is detected at threshold 15 but not 20, with unchanged SNR and observation bounds. This validates the decision branch without changing production results.
7. Compare coupled and vacuum limits, inverse-distance scaling, frequency-unit decoding, redshifted mass/frequency conversions, and finite observation windows in regression tests.

## Why these sources remain below threshold

- The fixed-mass population has secondary masses limited to approximately 197 Msun. Increasing distance and decreasing observed ISCO frequency reduce SNR over z=8–15.
- A four-year mission does not imply four years of signal: these sources generally cross the chosen band in months. The detector uses the available duration, not the AGN lifetime.
- In-band gas residence-time suppression is negligible for these particular candidate tracks; it is not the explanation for the missing detections.
- Lowering the frequency cutoff alone does not rescue the reference source: the detector-only 1e5+197 Msun binary at z=8 has SNR 17.469 at a 1-mHz cutoff and 17.539 at 0.1 mHz, still below 20.
- SNRs are orientation-averaged. A hard threshold on an averaged SNR is not an orientation-marginalized detection probability. The result does not imply that every favorable physical orientation would be undetectable.

A small historical normalization inconsistency exists: the orbital module uses 1.98892e33 g per solar mass, while the detector uses 1.98847e30 kg; their G constants also differ slightly. This is consistent with the small coupled/vacuum reference discrepancy, not the roughly 15% SNR increase needed to push the brightest simulated z=8 event over 20. Constants are not changed by this audit, and no physical rate correction is claimed.

## Artifacts

Under `outputs/detectability_audit/`:

- `snr_vs_redshift.png`, `.pdf`, `.svg`: clean SNR diagnostic figure;
- `report.md`: detailed main-text findings;
- `candidate_audit.csv`, `snr_summary.csv`, `paired_redshift_snr.csv`: event-level checks and paired-source redshift responses;
- `track_replay.csv`, `detector_controls.csv`, `checks.json`: reproducible validation results;
- `threshold_sensitivity.csv`: conditional threshold-reclassification tests, **not new physical simulations**. For example, threshold 10 gives reference-mass rates 1.133, 0.833, 0.200, 0, 0 Myr^-1 at z=8, 10, 12, 14, 15, respectively. Production threshold remains 20.

![SNR diagnostic](../outputs/detectability_audit/snr_vs_redshift.png)
