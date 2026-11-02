"""Validate persisted redshift/lifetime checkpoints and matched source inputs."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from check_single_halo_workflow import check_run

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--require-complete',action='store_true');a=p.parse_args()
df=pd.read_csv(a.directory/'repeat_comparison.csv')
if a.require_complete and len(df)!=720:raise AssertionError(f'{len(df)} of 720 completed')
for row in df.itertuples(index=False):
 d=Path(row.directory);metadata=json.loads((d/'simulation_metadata.json').read_text());meta=metadata['parameters']
 assert check_run(d)==1
 assert np.isclose(meta['redshift'],row.redshift) and np.isclose(meta['duty_cycle_myr'],row.duty_myr)
 assert np.isclose(meta['popiii_max_mass_msun'],row.max_mass_msun) and np.isclose(meta['popiii_imf_slope'],row.slope)
 assert meta['stellar_evolution_model']=='sevn' and meta['natal_kick_model']=='sevn'
 assert meta['disk_model']==('torque_with_trap' if row.model=='torque' else 'semi_without_trap')
 checkpoint=json.loads((d/'checkpoint.json').read_text());assert checkpoint['base_seed']==row.seed and checkpoint['task_seed']==row.task_seed
 if 'baseline' in checkpoint:
  baseline=Path(checkpoint['baseline']);before=baseline/'initial_population.csv';after=d/'initial_population.csv'
  base_metadata=json.loads((baseline/'simulation_metadata.json').read_text())
  for key in ['hydrogen_tables_sha256','helium_tables_sha256','sevnpy_version','kick_orbit_sha256']:
   assert metadata['stellar_model'].get(key)==base_metadata['stellar_model'].get(key)
  for key in ['metallicity','initial_stellar_density_gamma','snr_threshold','lisa_mission_duration_years','snr_window_placement','detector_hubble_constant_km_s_mpc','detector_omega_m']:
   assert meta[key]==base_metadata['parameters'][key]
  assert hashlib.sha256(before.read_bytes()).digest()==hashlib.sha256(after.read_bytes()).digest()
  population=pd.read_csv(after);stellar=pd.read_csv(d/'imri_stellar_ledger.csv')
  match=stellar.merge(population,on='source_id',suffixes=('_ledger','_initial'),validate='many_to_one')
  assert len(match)==len(stellar) and np.allclose(match.zams_mass_msun_ledger,match.zams_mass_msun_initial,rtol=1e-8)
 assert int(row.n_entries_after_duty)+int(row.n_entries_within_duty)==int(row.n_imri_observed)
matched=pd.read_csv(a.directory/'matched_primary_sources.csv')
assert matched['_merge'].eq('both').all()
assert np.allclose(matched.zams_mass_msun_base,matched.zams_mass_msun_case,rtol=0,atol=0)
redshift_pairs=pd.read_csv(a.directory/'matched_redshift_imri_events.csv')
shared=redshift_pairs[redshift_pairs['_merge']=='both']
assert np.allclose(shared.mass_msun_z10,shared.mass_msun_z15,rtol=1e-8,atol=1e-8)
status={'check':'passed','completed_repeats':len(df),'matched_primary_rows':len(matched), 'matched_redshift_imri_rows':len(shared),
        'incomplete_intrinsic_rates':int(df.rate_status.ne('complete').sum()),
        'incomplete_detector_rates':int(df.detector_rate_status.ne('complete').sum()),
        'rates_incomplete_are_unavailable_not_zero':True}
(a.directory/'validation.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status,indent=2))
