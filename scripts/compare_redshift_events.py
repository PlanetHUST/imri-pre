"""Join actual primary/donor IMRIs across z=10 and z=15 10 Myr episodes."""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
d=pd.read_csv(a.directory/'repeat_comparison.csv');rows=[];summaries=[]
for (model,imf),g in d.groupby(['model','imf'],sort=False):
 z10=g[g.case=='z10_duty10'].set_index('repeat');z15=g[g.case=='z15_duty10'].set_index('repeat')
 for repeat in z10.index.intersection(z15.index):
  x=z15.loc[repeat];y=z10.loc[repeat]
  def read(row):
   e=pd.read_csv(Path(row.directory)/'imri_events.csv');e=e[e.is_imri_candidate.eq(True)]
   keys=['task_kind','source_id','origin_primary_idx','copy_number']
   assert not e.duplicated(keys).any()
   return e[keys+['mass_msun','injection_time_myr','band_entry_time_myr','band_entry_redshift','lisa_snr','snr_status','is_lisa_detectable']]
  z=read(x).merge(read(y),on=['task_kind','source_id','origin_primary_idx','copy_number'],how='outer',suffixes=('_z15','_z10'),indicator=True)
  z['model']=model;z['imf']=imf;z['repeat']=repeat
  z['both_intrinsic_rates_complete']=x.rate_status=='complete' and y.rate_status=='complete'
  z['both_detector_rates_complete']=x.detector_rate_status=='complete' and y.detector_rate_status=='complete'
  rows.extend(z.to_dict('records'))
if rows:
 frame=pd.DataFrame(rows);frame.to_csv(a.directory/'matched_redshift_imri_events.csv',index=False)
 for (model,imf),g in frame.groupby(['model','imf'],sort=False):
  shared=g[g['_merge']=='both'];valid=shared[(shared.snr_status_z15=='evaluated')&(shared.snr_status_z10=='evaluated')]
  s={'model':model,'imf':imf,'shared_imri':len(shared),'only_z10':int((g['_merge']=='right_only').sum()),
     'only_z15':int((g['_merge']=='left_only').sum()),'shared_evaluated_snr':len(valid),
     'median_snr_z15':valid.lisa_snr_z15.median(),'median_snr_z10':valid.lisa_snr_z10.median(),
     'shared_detected_z15':int(valid.is_lisa_detectable_z15.eq(True).sum()),
     'shared_detected_z10':int(valid.is_lisa_detectable_z10.eq(True).sum()),
     'injection_time_max_difference_myr':(shared.injection_time_myr_z10-shared.injection_time_myr_z15).abs().max(),
     'remnant_mass_max_difference_msun':(shared.mass_msun_z10-shared.mass_msun_z15).abs().max(),
     'n_rows_in_incomplete_intrinsic_repeats':int((~g.both_intrinsic_rates_complete).sum()),
     'n_rows_in_incomplete_detector_repeats':int((~g.both_detector_rates_complete).sum())}
  summaries.append(s)
 pd.DataFrame(summaries).to_csv(a.directory/'matched_redshift_summary.csv',index=False)
