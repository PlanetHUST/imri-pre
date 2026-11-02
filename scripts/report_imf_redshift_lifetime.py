"""Summarize checkpointed IMF/redshift/lifetime runs and compare Pan & Yang 2021."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_imf_sensitivity import SCENARIOS
from run_imf_redshift_lifetime import CASES

p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'output/imf_redshift_lifetime_20261029')
p.add_argument('--baseline-dir',type=Path,default=ROOT/'output/imf_sensitivity_20261029');a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
rows=[]; primary={}
for model in ('torque','semi'):
 for case,(z,duty) in CASES.items():
  for name,maximum,slope in SCENARIOS:
   for i in range(20):
    d=(a.baseline_dir/name/f'repeat_{i:03d}' if model=='torque' and case=='z15_duty10' else a.output_dir/model/case/name/f'repeat_{i:03d}')
    cp=d/'checkpoint.json'
    if not cp.exists() or json.loads(cp.read_text()).get('status')!='complete':continue
    info=json.loads(cp.read_text());rep=pd.read_csv(d/'imri_repeat_results.csv').iloc[0];events=pd.read_csv(d/'imri_events.csv');stellar=pd.read_csv(d/'imri_stellar_ledger.csv')
    candidates=events[events.is_imri_candidate.eq(True)];high=candidates[candidates.zams_mass_msun>=240.5]
    complete=rep.rate_status=='complete';det_complete=rep.detector_rate_status=='complete'
    rows.append({'model':model,'case':case,'imf':name,'repeat':i,'seed':info['base_seed'],'task_seed':info['task_seed'],
     'redshift':z,'duty_myr':duty,'max_mass_msun':maximum,'slope':slope,'n_sampled':int(rep.n_sampled),'n_disk_cross':int(rep.n_disk_cross),
     'n_emb_star':int(rep.n_emb_star),'n_emb_bh':int(rep.n_emb_bh),'n_migration_formed_bh':int(rep.n_mfbh),
     'n_replenished':int(rep.n_replenished),'n_isco_consumed':int(rep.n_isco_consumed),
     'n_no_imri':int(rep.n_no_imri),'n_trapped':int(rep.n_mig_trap),'n_outward_blocked':int(rep.n_mig_outward_blocked),
     'n_detector_unavailable':int(rep.n_detector_unavailable),
     'n_replenishment_unavailable':int(rep.n_replenishment_unavailable),'n_imri_observed':int(rep.n_imri),'n_high_imri_observed':len(high),
     'n_entries_after_duty':int((pd.to_numeric(candidates.band_entry_time_myr,errors='coerce')>duty).sum()),
     'n_entries_within_duty':int((pd.to_numeric(candidates.band_entry_time_myr,errors='coerce')<=duty).sum()),
     'n_detected_observed':int(rep.n_det),'n_stellar_unavailable':int(rep.n_stellar_unavailable),
     'intrinsic_rate_per_myr':float(rep.rate_imri_per_myr) if complete else np.nan,
     'isco_proxy_rate_per_myr':float(rep.n_isco_consumed)/duty if complete and int(rep.n_replenishment_unavailable)==0 else np.nan,
     'detector_rate_per_myr':float(rep.rate_det_per_myr) if det_complete else np.nan,
     'rate_status':rep.rate_status,'detector_rate_status':rep.detector_rate_status,'directory':str(d)})
    stars=stellar[stellar.task_kind=='primary'][['source_id','zams_mass_msun','damping_status']].copy()
    events=events[events.task_kind=='primary'][['source_id','is_imri_candidate','is_lisa_detectable','mass_msun','lisa_snr','snr_status','band_entry_time_myr','band_entry_redshift','gas_active_at_entry','gas_stop_reason','gas_elapsed_time_myr']].copy()
    assert not stars.source_id.duplicated().any() and not events.source_id.duplicated().any()
    primary[(model,case,name,i)]=stars.merge(events,on='source_id',how='left',validate='one_to_one')
if not rows:raise SystemExit('No complete run checkpoints available')
df=pd.DataFrame(rows);df.to_csv(a.output_dir/'repeat_comparison.csv',index=False)
paired=[];matched=[]
for (model,case,name,i),stars in primary.items():
 key=(model,'z15_duty10',name,i)
 if case=='z15_duty10' or key not in primary:continue
 control=primary[key]
 join=control.merge(stars,on='source_id',how='outer',suffixes=('_base','_case'),indicator=True)
 for rec in join.to_dict('records'):
  rec.update(model=model,case=case,imf=name,repeat=i,contrast='redshift_or_lifetime');matched.append(rec)
for (model,case,name,i),stars in primary.items():
 if model!='semi' or ('torque',case,name,i) not in primary:continue
 control=primary[('torque',case,name,i)]
 join=control.merge(stars,on='source_id',how='outer',suffixes=('_base','_case'),indicator=True)
 for rec in join.to_dict('records'):
  rec.update(model='semi',case=case,imf=name,repeat=i,contrast='semi_minus_torque');matched.append(rec)
for (model,imf,case),g in df.groupby(['model','imf','case'],sort=False):
 valid=g[g.rate_status.eq('complete')];det=g[g.detector_rate_status.eq('complete')];proxy=g[g.isco_proxy_rate_per_myr.notna()]
 base=df[(df.model==model)&(df.imf==imf)&(df.case=='z15_duty10')]
 pair=base.merge(g,on='repeat',suffixes=('_base','_case'));pair=pair[pair.rate_status_base.eq('complete')&pair.rate_status_case.eq('complete')]
 diff=pair.intrinsic_rate_per_myr_case-pair.intrinsic_rate_per_myr_base
 def sem(x):return float(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else np.nan
 paired.append({'model':model,'imf':imf,'case':case,'redshift':g.redshift.iloc[0],'duty_myr':g.duty_myr.iloc[0],
  'repeats_completed':len(g),'rates_complete':len(valid),'detector_rates_complete':len(det),
  'intrinsic_rate_mean_per_myr':valid.intrinsic_rate_per_myr.mean(),'intrinsic_rate_sem_per_myr':sem(valid.intrinsic_rate_per_myr),
  'isco_proxy_rate_mean_per_myr':proxy.isco_proxy_rate_per_myr.mean(),'isco_proxy_rate_sem_per_myr':sem(proxy.isco_proxy_rate_per_myr),
  'detect_rate_mean_per_myr':det.detector_rate_per_myr.mean(),'detect_rate_sem_per_myr':sem(det.detector_rate_per_myr),
  'mean_n_entries_after_duty':g.n_entries_after_duty.mean(), 'mean_n_entries_within_duty':g.n_entries_within_duty.mean(),
  'mean_n_sampled':g.n_sampled.mean(),'mean_n_disk_cross':g.n_disk_cross.mean(),
  'mean_n_isco_consumed':g.n_isco_consumed.mean(),'mean_n_replenished':g.n_replenished.mean(),'mean_n_imri_observed':g.n_imri_observed.mean(),
  'mean_n_high_imri_observed':g.n_high_imri_observed.mean(),'detected_observed':int(g.n_detected_observed.sum()),
  'paired_rate_difference_per_myr':diff.mean(),'paired_rate_difference_sem':sem(diff),'paired_complete':len(pair),
  'n_rate_incomplete':len(g)-len(valid),'n_detector_incomplete':len(g)-len(det)})
summary=pd.DataFrame(paired);summary.to_csv(a.output_dir/'scenario_comparison.csv',index=False)
matched_df=pd.DataFrame(matched);matched_df.to_csv(a.output_dir/'matched_primary_sources.csv',index=False)
membership=[]
if not matched_df.empty:
 for (contrast,model,case,imf),g in matched_df.groupby(['contrast','model','case','imf'],sort=False):
    b=g.is_imri_candidate_base.eq(True);c=g.is_imri_candidate_case.eq(True)
    shared=b&c;added=(~b)&c;removed=b&(~c)
    same=shared&g.snr_status_base.eq('evaluated')&g.snr_status_case.eq('evaluated')&g.lisa_snr_base.notna()&g.lisa_snr_case.notna()
    delta=(g.loc[same,'lisa_snr_case']-g.loc[same,'lisa_snr_base']).abs()
    membership.append({'contrast':contrast,'model':model,'case':case,'imf':imf,'paired_primary_sources':len(g),
      'unmatched_primary_ids':int(g['_merge'].ne('both').sum()),'shared_primary_imris':int(shared.sum()),
      'added_primary_imris':int(added.sum()),'removed_primary_imris':int(removed.sum()),
      'shared_mass_median_msun':g.loc[shared,'mass_msun_case'].median(),
      'added_mass_median_msun':g.loc[added,'mass_msun_case'].median(),
      'shared_snr_pairs':int(same.sum()),'shared_max_abs_snr_change':delta.max(),
      'shared_snr_base_median':g.loc[same,'lisa_snr_base'].median(),
      'shared_snr_case_median':g.loc[same,'lisa_snr_case'].median(),
      'added_snr_median':g.loc[added & g.snr_status_case.eq('evaluated'),'lisa_snr_case'].median(),
      'removed_snr_median':g.loc[removed & g.snr_status_base.eq('evaluated'),'lisa_snr_base'].median(),
      'added_entry_time_median_myr':g.loc[added,'band_entry_time_myr_case'].median()})
pd.DataFrame(membership).to_csv(a.output_dir/'primary_membership_summary.csv',index=False)
redshift_pairs=[]
for (model,imf,case),g in df.groupby(['model','imf','case'],sort=False):
 if case=='z15_duty10':continue
 base=df[(df.model==model)&(df.imf==imf)&(df.case=='z15_duty10')]
 pair=base.merge(g,on='repeat',suffixes=('_base','_case'))
 for row in pair.itertuples(index=False):
  redshift_pairs.append({'model':model,'imf':imf,'case':case,'repeat':row.repeat,
    'both_rates_complete':row.rate_status_base=='complete' and row.rate_status_case=='complete',
    'intrinsic_rate_difference_per_myr':row.intrinsic_rate_per_myr_case-row.intrinsic_rate_per_myr_base,
    'isco_proxy_difference_per_myr':row.isco_proxy_rate_per_myr_case-row.isco_proxy_rate_per_myr_base,
    'observed_event_count_difference':row.n_imri_observed_case-row.n_imri_observed_base})
pd.DataFrame(redshift_pairs).to_csv(a.output_dir/'paired_redshift_lifetime_differences.csv',index=False)
migration_pairs=[]
for (imf,case),g in df.groupby(['imf','case'],sort=False):
 t=g[g.model=='torque'];q=g[g.model=='semi']
 paired_models=t.merge(q,on='repeat',suffixes=('_torque','_semi'))
 for row in paired_models.itertuples(index=False):
  migration_pairs.append({'imf':imf,'case':case,'repeat':row.repeat,
    'both_rates_complete':row.rate_status_torque=='complete' and row.rate_status_semi=='complete',
    'intrinsic_rate_semi_minus_torque_per_myr':row.intrinsic_rate_per_myr_semi-row.intrinsic_rate_per_myr_torque,
    'isco_proxy_semi_minus_torque_per_myr':row.isco_proxy_rate_per_myr_semi-row.isco_proxy_rate_per_myr_torque})
pd.DataFrame(migration_pairs).to_csv(a.output_dir/'paired_migration_differences.csv',index=False)
pan=pd.DataFrame([{'model':'alpha_wind_on','capture_parameter':1.0,'duty_myr':10,'rate_per_myr':4.2},
 {'model':'alpha_wind_on','capture_parameter':1.0,'duty_myr':100,'rate_per_myr':1.3},
 {'model':'alpha_wind_on','capture_parameter':0.1,'duty_myr':10,'rate_per_myr':5.7},
 {'model':'alpha_wind_on','capture_parameter':0.1,'duty_myr':100,'rate_per_myr':1.6},
 {'model':'TQM_wind_on','capture_parameter':1.0,'duty_myr':10,'rate_per_myr':14.0},
 {'model':'TQM_wind_on','capture_parameter':1.0,'duty_myr':100,'rate_per_myr':5.0}])
pan['smbh_mass_msun']=1e5;pan['initial_density_slope']=1.5;pan['relative_bh_number_fraction']=0.001
pan['disk_parameter_alpha_or_X']=0.1;pan['accretion_rate_in_eddington_units']=0.1;pan['pre_disk_relaxation_gyr']=5.0
pan['source_rate_unit']='Gyr^-1 converted /1000 to Myr^-1';pan['source']='https://arxiv.org/html/2101.09146 Table 1';pan.to_csv(a.output_dir/'pan_2021_reference.csv',index=False)
status={'completed_new_repeats':int(len(df)-len(df[(df.model=='torque')&(df.case=='z15_duty10')])),
 'included_baseline_repeats':int(len(df[(df.model=='torque')&(df.case=='z15_duty10')])),
 'expected_repeats':720,'complete_run_checkpoints_by_case':df.groupby(['model','case']).size().to_dict()}
status['complete_run_checkpoints_by_case']={f'{k[0]}:{k[1]}':int(v) for k,v in status['complete_run_checkpoints_by_case'].items()}
(a.output_dir/'checkpoint_summary.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status,indent=2))
