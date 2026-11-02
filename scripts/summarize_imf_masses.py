"""Summarize actual BH masses at production, selection, and IMRI stages."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
rows=[]
histograms=[]
edges=np.array([0.,10.,25.,50.,80.,100.,150.,200.,250.,300.,400.,500.,600.,np.inf])
for scenario in sorted(x for x in a.directory.iterdir() if x.is_dir()):
    stage={'all_sampled_bh':[], 'crossing_bh_endpoint':[], 'primary_embedded_bh':[], 'intrinsic_imri':[], 'detectable':[]}
    census_count=0;repeat_count=0
    for d in sorted(scenario.glob('repeat_*')):
        if not (d/'checkpoint.json').exists():continue
        import json
        if json.loads((d/'checkpoint.json').read_text())['status']!='complete':continue
        repeat_count+=1
        full=d/'initial_stellar_endpoints.csv'
        if full.exists():
            x=pd.read_csv(full);stage['all_sampled_bh'].extend(x.loc[x.remnant_type==6,'remnant_mass_msun'].tolist());census_count+=1
        s=pd.read_csv(d/'imri_stellar_ledger.csv');s=s[s.task_kind=='primary']
        stage['crossing_bh_endpoint'].extend(s.loc[s.remnant_type==6,'remnant_mass_msun'].tolist())
        stage['primary_embedded_bh'].extend(s.loc[(s.remnant_type==6)&(s.damping_status=='embedded_bh'),'remnant_mass_msun'].tolist())
        e=pd.read_csv(d/'imri_events.csv')
        stage['intrinsic_imri'].extend(e.loc[e.is_imri_candidate==True,'remnant_mass_msun'].tolist())
        stage['detectable'].extend(e.loc[e.is_lisa_detectable==True,'remnant_mass_msun'].tolist())
    for name,values in stage.items():
        v=np.asarray(values,dtype=float);v=v[np.isfinite(v)]
        exposure=census_count if name=='all_sampled_bh' else repeat_count
        counts,_=np.histogram(v,bins=edges)
        for i,n in enumerate(counts):
            histograms.append({'scenario':scenario.name,'stage':name,'bin_left_msun':edges[i],
                'bin_right_msun':edges[i+1], 'count':int(n), 'completed_repeats':exposure,
                'mean_per_episode':float(n/exposure) if exposure else np.nan})
        rows.append({'scenario':scenario.name,'stage':name,'completed_repeats':census_count if name=='all_sampled_bh' else repeat_count,
                     'count':len(v),'mass_min_msun':v.min() if len(v) else np.nan,
                     'mass_median_msun':np.median(v) if len(v) else np.nan,'mass_max_msun':v.max() if len(v) else np.nan,
                     'count_bh_mass_ge200_msun':int((v>=200).sum())})
pd.DataFrame(rows).to_csv(a.directory/'bh_mass_stage_summary.csv',index=False)
pd.DataFrame(histograms).to_csv(a.directory/'bh_mass_histograms.csv',index=False)
