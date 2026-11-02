"""Show source-matched SNR shifts for the two high-redshift episodes."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
x=pd.read_csv(a.directory/'matched_redshift_imri_events.csv');x=x[(x.imf=='max600_slope177')&(x['_merge']=='both')&(x.snr_status_z10=='evaluated')&(x.snr_status_z15=='evaluated')]
fig,axes=plt.subplots(1,2,figsize=(10,4.6),sharex=True,sharey=True)
limit=max(25,float(x.lisa_snr_z15.max()) if len(x) else 0,float(x.lisa_snr_z10.max()) if len(x) else 0)*1.05
for ax,model in zip(axes,['torque','semi']):
 d=x[x.model==model]
 for kind,color in [('primary','tab:blue'),('replacement','tab:orange')]:
  t=d[d.task_kind==kind];ax.scatter(t.lisa_snr_z15,t.lisa_snr_z10,label=f'{kind}: {len(t)}',alpha=.75,s=24,color=color)
 ax.plot([0,limit],[0,limit],color='grey',linestyle='--',linewidth=1);ax.axvline(20,color='red',linewidth=1);ax.axhline(20,color='red',linewidth=1)
 ax.set(xlabel='SNR, z=15',title=f'{model}: {len(d)} paired entries',xlim=(0,limit),ylim=(0,limit));ax.legend(frameon=False,fontsize=8)
axes[0].set_ylabel('SNR, z=10');fig.suptitle(r'10 Myr episodes, same IMF and source/channel identity; max $600\,M_\odot$, $\alpha=1.77$')
fig.text(.5,.01,'Red lines mark SNR=20; only evaluated shared IMRIs are shown.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(a.directory/'paired_redshift_snr.png',dpi=180);plt.close(fig)
