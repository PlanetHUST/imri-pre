"""Read-only summary of the previous fixed-mass six-case reference."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('output_dir',type=Path);a=p.parse_args()
root=Path('output/coupled_six_replenishment_20261023');rows=[]
for model in ('torque','semi'):
 for case,duty in [('z10_duty10',10),('z15_duty10',10),('z15_duty100',100)]:
  dirname=f'torque_{case}' if model=='torque' else f'single_host_example_semi_{case}_r20'
  f=root/dirname/'imri_repeat_results.csv';d=pd.read_csv(f);v=d[d.rate_status=='complete'];det=d[d.detector_rate_status=='complete']
  rows.append({'model':model,'case':case,'repeats':len(d),'complete_rates':len(v),'stellar_model':'fixed_mass','kick_model':'none',
    'mean_intrinsic_per_myr':v.rate_imri_per_myr.mean(),'sem_intrinsic_per_myr':v.rate_imri_per_myr.std(ddof=1)/np.sqrt(len(v)),
    'mean_isco_proxy_per_myr':(v.n_isco_consumed/duty).mean(),'detections_total':int(det.n_det.sum()),
    'seed_policy':'SeedSequence(12345,0,0,repeat); differs from new base_seed=12345+repeat', 'source':str(f)})
pd.DataFrame(rows).to_csv(a.output_dir/'archived_fixed_mass_comparison.csv',index=False)
