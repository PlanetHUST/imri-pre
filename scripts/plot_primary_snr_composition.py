"""Visualize matched primary SNR and mass composition across episode lifetimes."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
d=pd.read_csv(a.directory/'matched_primary_sources.csv')
d=d[(d.contrast=='redshift_or_lifetime')&(d.model=='torque')&(d.case=='z15_duty100')&(d.imf=='max197_slope177')]
b=d.is_imri_candidate_base.eq(True);c=d.is_imri_candidate_case.eq(True);shared=d[b&c];added=d[~b&c]
fig,axes=plt.subplots(1,2,figsize=(10,4))
for x,label,color in [(shared,'Shared primaries','tab:blue'),(added,'Added in 100 Myr','tab:orange')]:
 mass=x.mass_msun_case.dropna();axes[0].hist(mass,bins=np.linspace(0,80,17),histtype='step',linewidth=1.8,label=f'{label} (n={len(mass)})',color=color)
 y=x.loc[x.snr_status_case.eq('evaluated'),'lisa_snr_case'].dropna();axes[1].hist(y,bins=np.linspace(0,20,21),histtype='step',linewidth=1.8,label=f'{label} (n={len(y)})',color=color)
axes[0].set(xlabel=r'BH mass [$M_\odot$]',ylabel='Primary IMRI count',title='100 Myr sample: mass')
axes[1].set(xlabel='Evaluated LISA SNR',ylabel='Primary IMRI count',title='100 Myr sample: SNR')
axes[0].legend(frameon=False,fontsize=8);axes[1].legend(frameon=False,fontsize=8)
fig.suptitle(r'z=15, torque model, IMF max 197 $M_\odot$; 20 paired repeats')
fig.text(.5,.01,'Primary sources only; two 100 Myr repeats have incomplete rates. Counts are observed entries.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.05,1,.94));fig.savefig(a.directory/'primary_snr_mass_composition.png',dpi=180);plt.close(fig)
