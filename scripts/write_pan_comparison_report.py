"""Write a sourced, status-aware report for the IMF redshift/lifetime study."""
from pathlib import Path
import argparse
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
s=pd.read_csv(a.directory/'scenario_comparison.csv');m=pd.read_csv(a.directory/'primary_membership_summary.csv')
matched=pd.read_csv(a.directory/'matched_primary_sources.csv')
order=['max197_slope177','max250_slope177','max300_slope177','max600_slope177','max600_slope100','max600_slope230']
cases=['z10_duty10','z15_duty10','z15_duty100']
lines=['# High-redshift IMF study and Pan & Yang (2021)', '',
'This is a single-halo study at M_SMBH=1e5 Msun, M_NSC=2e5 Msun, Z=1e-5 Zsun. Each case uses a fixed stellar-mass budget and 20 planned repeat seeds; completed counts are shown below.',
'The quoted rates are source-frame per Myr, conditional on the selected halo and AGN episode; they are not cosmological volumetric or observer-frame rates.',
'IMRI entry means first observed LISA-band crossing under the current model, including entries after the AGN ends. ISCO consumption counts physical IMRIs reaching ISCO within the active episode; division by its full duty time is a closer Pan-style proxy, but is not Pan’s flux definition.',
'Rate averages condition on complete status and can be biased if the incomplete repeats are not random. Detections use SNR >=20. SEM quantifies Monte Carlo repeat scatter only.', '',
'[Pan & Yang (2021), Table 1](https://arxiv.org/html/2101.09146) gives, for M_SMBH=1e5 Msun, gamma=1.5, delta=0.001, alpha=0.1, accretion=0.1 Eddington, wind on, capture parameter 1, 4200 and 1300 Gyr^-1 at 10 and 100 Myr: 4.2 and 1.3 Myr^-1. Their capture-flux prescription is not our ISCO count.',
'Their two-component 1 and 10 Msun population is evolved in a Fokker-Planck model for 5 Gyr before disk activation. Our initial populations are sampled at the episode start and evolved with SEVN. In our adopted cosmology the Universe is ~472 Myr old at z=10 and ~269 Myr at z=15.',
'With H0=67.4 km/s/Mpc and Omega_m=0.315, an episode starting at z=15 ends at z≈14.62 after 10 Myr or z≈11.96 after 100 Myr; one starting at z=10 ends at z≈9.85 after 10 Myr. Event redshifts are evolved individually.',
'Their NSC stellar mass is 20 M_SMBH and initial sBH number fraction delta=0.001; our NSC mass is 2 M_SMBH and our stellar/remnant mass function is explicitly modeled. Their scattering, supply and capture prescriptions differ from our finite, consumption-triggered donor pool.',
'Their alpha/TQM disk prescriptions and wind torque differ from our pAGN Sirko disk plus signed torque or integrated comparison model; neither local model is a parameter-matched reproduction of Pan. At 1e5 Msun our explicit Mdot=1.44e23 g/s corresponds to pAGN le≈0.88 for its eps=0.1 and composition defaults; Pan uses 0.1 of its Eddington accretion rate. These normalizations need not be identical. The IMRI waveform/detector cut is an additional distinction.', '',
'## Completed scenario values', '',
'| Model | IMF | z / duty Myr | Completed / 20 | Complete intrinsic rates | Band-entry rate / Myr | ISCO proxy / Myr | Detections |',
'| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
def fmt(x):return 'unavailable' if pd.isna(x) else f'{x:.3f}'
for model in ['torque','semi']:
 for imf in order:
  for case in cases:
   q=s[(s.model==model)&(s.imf==imf)&(s.case==case)]
   if q.empty:continue
   x=q.iloc[0];lines.append(f'| {model} | {imf} | {int(x.redshift)} / {int(x.duty_myr)} | {int(x.repeats_completed)} | {int(x.rates_complete)} | {fmt(x.intrinsic_rate_mean_per_myr)} | {fmt(x.isco_proxy_rate_mean_per_myr)} | {int(x.detected_observed)} |')
archive_path=a.directory/'archived_fixed_mass_comparison.csv'
if archive_path.exists():
 archive=pd.read_csv(archive_path)
 lines += ['', '## Previous fixed-mass archive (unpaired)', '',
           '| Model | Case | Fixed-mass archive intrinsic / Myr | SEVN + kicks intrinsic / Myr | New complete / completed |',
           '| --- | --- | ---: | ---: | ---: |']
 for row in archive.itertuples(index=False):
  q=s[(s.model==row.model)&(s.case==row.case)&(s.imf=='max197_slope177')]
  if q.empty:continue
  x=q.iloc[0]
  lines.append(f'| {row.model} | {row.case} | {fmt(row.mean_intrinsic_per_myr)} | {fmt(x.intrinsic_rate_mean_per_myr)} | {int(x.rates_complete)}/{int(x.repeats_completed)} |')
 lines.append('Archive uses a different repeat seed schedule after repeat zero. Both stellar physics and sampling history differ, so this is descriptive and not an isolated causal effect.')
redshift_path=a.directory/'matched_redshift_summary.csv'
if redshift_path.exists():
 zsum=pd.read_csv(redshift_path)
 lines += ['', '## Paired z=10 and z=15, 10 Myr episodes', '',
           '| Model | IMF | Shared IMRIs | Only z=10 / z=15 | Shared evaluated SNR median z=10 / z=15 | Shared detections z=10 / z=15 | Incomplete-rate rows |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
 for row in zsum.itertuples(index=False):
  lines.append(f'| {row.model} | {row.imf} | {row.shared_imri} | {row.only_z10} / {row.only_z15} | {fmt(row.median_snr_z10)} / {fmt(row.median_snr_z15)} | {row.shared_detected_z10} / {row.shared_detected_z15} | {row.n_rows_in_incomplete_intrinsic_repeats} |')
 lines.append('Pairing includes donor events only when their source and channel identity agree; injection times for shared pairs are retained in the detailed table. Detection totals are not cosmological rates.')
lines += ['', '## Matched original-IMF comparison', '']
for model in ['torque','semi']:
 short=s[(s.model==model)&(s.imf=='max197_slope177')&(s.case=='z15_duty10')]
 long=s[(s.model==model)&(s.imf=='max197_slope177')&(s.case=='z15_duty100')]
 if len(short) and len(long):
  x=short.iloc[0];y=long.iloc[0]
  lines.append(f"- {model}: ISCO proxy at 10 Myr = {fmt(x.isco_proxy_rate_mean_per_myr)} Myr^-1 ({int(x.rates_complete)}/{int(x.repeats_completed)} complete); at 100 Myr = {fmt(y.isco_proxy_rate_mean_per_myr)} Myr^-1 ({int(y.rates_complete)}/{int(y.repeats_completed)} complete). Pan alpha-disk estimates are 4.2 and 1.3 Myr^-1, respectively, for different assumptions and a different flux observable.")
  comp=m[(m.contrast=='redshift_or_lifetime')&(m.model==model)&(m.imf=='max197_slope177')&(m.case=='z15_duty100')]
  if len(comp):
   c=comp.iloc[0];lines.append(f"  Shared primary IMRIs: {int(c.shared_primary_imris)}; added primaries: {int(c.added_primary_imris)}; shared/added median BH masses: {fmt(c.shared_mass_median_msun)}/{fmt(c.added_mass_median_msun)} Msun. Shared SNR median changes from {fmt(c.shared_snr_base_median)} to {fmt(c.shared_snr_case_median)}; added SNR median is {fmt(c.added_snr_median)}.")
zcomp=m[(m.contrast=='redshift_or_lifetime')&(m.model=='torque')&(m.imf=='max197_slope177')&(m.case=='z10_duty10')]
if len(zcomp):
 c=zcomp.iloc[0];lines.append(f"- For the torque 10 Myr runs, {int(c.shared_primary_imris)} primary IMRIs are shared between z=15 and z=10; their evaluated SNR medians are {fmt(c.shared_snr_base_median)} and {fmt(c.shared_snr_case_median)}, respectively.")
example=matched[(matched.contrast=='redshift_or_lifetime')&(matched.model=='torque')&(matched.imf=='max197_slope177')&(matched.case=='z15_duty100')&matched.is_imri_candidate_base.eq(True)&matched.is_imri_candidate_case.eq(True)].copy()
example=example[example.snr_status_base.eq('evaluated')&example.snr_status_case.eq('evaluated')&example.lisa_snr_base.notna()&example.lisa_snr_case.notna()]
if len(example):
 example['snr_change']=(example.lisa_snr_case-example.lisa_snr_base).abs();x=example.loc[example.snr_change.idxmax()]
 lines.append(f"- Largest shared-primary SNR change in this comparison: repeat {int(x['repeat'])}, source {int(x.source_id)}, BH mass {fmt(x.mass_msun_base)} Msun. Short/long entry times are {fmt(x.band_entry_time_myr_base)}/{fmt(x.band_entry_time_myr_case)} Myr, entry redshifts {fmt(x.band_entry_redshift_base)}/{fmt(x.band_entry_redshift_case)}, and SNRs {fmt(x.lisa_snr_base)}/{fmt(x.lisa_snr_case)}. Gas stopped because of {x.gas_stop_reason_base}/{x.gas_stop_reason_case}. This is an example, not a population-level attribution.")
lines+=['','Detection counts are totals across completed realizations, not rates. Rate SEMs and missing-status counts are in `scenario_comparison.csv`; event counts and after-duty entries are in `repeat_comparison.csv`.',
'`matched_primary_sources.csv` contains source-ID joins only within the same IMF and repeat. Primaries are compared on matching IDs and mass; donor chains can diverge after different consumption events.',
'`primary_membership_summary.csv` separates shared, added, and removed primary IMRIs and their mass medians, including within-model redshift/lifetime contrasts and semi-minus-torque contrasts. Large shared-object SNR differences require tracing their entry times and redshifts; an aggregate median alone cannot identify that cause.',
'`archived_fixed_mass_comparison.csv` records the previous six fixed-mass/no-kick runs. Their root seed is fixed at 12345 with the repeat index inside SeedSequence; the new IMF study uses base seed 12345 + repeat, so only repeat zero has the same task seed. Compare those archives as unpaired model controls.',
'Pan’s values are reference numbers rather than a high-redshift empirical measurement. A quantitative causal attribution to redshift requires a matched population, disk, capture prescription, and rate definition.']
(a.directory/'comparison_with_pan_2021.md').write_text('\n'.join(lines)+'\n')
