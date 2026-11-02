"""Summarize completed IMF-study checkpoints and matched repeat differences."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
df=pd.read_csv(a.directory/'repeat_summary.csv')
metrics=['n_initial','n_initial_zams_ge240p5','n_primary_crossing','n_crossing_zams_ge240p5',
         'n_evaluated_bh_zams_ge240p5','n_primary_embedded_bh_zams_ge240p5','n_imri','n_imri_zams_ge240p5','n_detectable']
metrics += [x for x in ['n_all_bh_endpoints','n_all_upper_branch_bh','n_all_no_remnant','n_all_stellar_unavailable'] if x in df.columns]
base=df[df.scenario=='max197_slope177'].set_index('repeat')
rows=[]
for scenario,g in df.groupby('scenario',sort=False):
    if scenario=='max197_slope177':continue
    matched=base.join(g.set_index('repeat'),lsuffix='_control',rsuffix='_scenario',how='inner')
    for metric in metrics:
        for repeat,r in matched.iterrows():
            rows.append({'scenario':scenario,'repeat':repeat,'metric':metric,
                         'difference':r[metric+'_scenario']-r[metric+'_control'],
                         'both_rates_complete':r['rate_status_control']=='complete' and r['rate_status_scenario']=='complete'})
paired=pd.DataFrame(rows,columns=['scenario','repeat','metric','difference','both_rates_complete'])
paired.to_csv(a.directory/'paired_differences.csv',index=False)
text=['# Pop III IMF sensitivity study','','Each repeat is one 10 Myr AGN episode; no cosmological weighting.',
      'Counts and rates below are descriptive; complete-repeat rate means condition on successful evaluation and may be biased when incomplete repeats are present.',
      'The ZAMS threshold 240.5 Msun is a scenario-specific diagnostic, not a general BH mass-gap edge.',
      'SEVN endpoint counts in the stellar ledger cover evaluated primary crossers and replacement donors only.',
      'The full-population census evolves all sampled stars to their eventual endpoints even if collapse occurs after the AGN ends; those endpoints are predictions, not all active BHs at a given time.',
      'Initial population sampling stops at the first object that takes total mass above 200000 Msun; the overshoot is at most one sampled stellar mass.',
      'Same-seed repeats are paired statistically; source IDs and donor chains are not assumed identical.','']
for scenario,g in df.groupby('scenario',sort=False):
    text += [f'## {scenario}',f'- Completed repeats: {len(g)}.',
             f'- Intrinsic rate status counts: {g.rate_status.value_counts().to_dict()}.',
             f'- Detector rate status counts: {g.detector_rate_status.value_counts().to_dict()}.']
    for status, metric, label in [('rate_status','n_imri','intrinsic'), ('detector_rate_status','n_detectable','detector')]:
        valid=g.loc[g[status]=='complete',metric] / 10.0
        if len(valid):
            sem=valid.std(ddof=1)/np.sqrt(len(valid)) if len(valid)>1 else float('nan')
            text.append(f'- {label} rate for complete repeats: {valid.mean():.4f} ± {sem:.4f} per Myr (n={len(valid)}).')
    for metric in metrics:
        vals=g[metric].dropna();sem=vals.std(ddof=1)/np.sqrt(len(vals)) if len(vals)>1 else float('nan')
        text.append(f'- {metric}: {vals.mean():.3f} ± {sem:.3f} per episode; total {int(vals.sum())}; n={len(vals)}.')
    if scenario!='max197_slope177':
        match=paired[(paired.scenario==scenario)&(paired.metric=='n_imri') & paired.both_rates_complete].difference
        if len(match):text.append(f'- Paired change in n_imri versus control for complete rates: {match.mean():+.3f} ± {match.std(ddof=1)/np.sqrt(len(match)) if len(match)>1 else float("nan"):.3f} per episode (n={len(match)}).')
    text.append('')
(a.directory/'report.md').write_text('\n'.join(text)+'\n')
