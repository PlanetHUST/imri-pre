"""Plot finite-run means against, but not as equivalent to, Pan 2021 capture estimates."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
s=pd.read_csv(a.directory/'scenario_comparison.csv')
cases=['z10_duty10','z15_duty10','z15_duty100'];labels=['z=10, 10 Myr','z=15, 10 Myr','z=15, 100 Myr']
fig,axes=plt.subplots(1,2,figsize=(11,5),sharey=True)
for ax,model in zip(axes,['torque','semi']):
 for imf,color,label in [('max197_slope177','tab:blue',r'197 $M_\odot$'),('max600_slope177','tab:orange',r'600 $M_\odot$')]:
  d=s[(s.model==model)&(s.imf==imf)].set_index('case').reindex(cases)
  ax.errorbar(range(3),d.isco_proxy_rate_mean_per_myr,yerr=d.isco_proxy_rate_sem_per_myr,marker='o',color=color,label=label,capsize=3)
  incomplete=d.n_rate_incomplete.fillna(0)>0
  ax.scatter(np.arange(3)[incomplete],d.loc[incomplete,'isco_proxy_rate_mean_per_myr'],marker='x',s=90,color='red',zorder=4)
 ax.plot(range(3),[4.2,4.2,1.3],marker='s',linestyle='--',color='black',label='Pan 2021 capture estimate')
 ax.set_yscale('log');ax.set_xticks(range(3),labels,rotation=25,ha='right');ax.set_title(model+' migration');ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel(r'Per-halo episode-average rate [$\mathrm{Myr}^{-1}$]');axes[0].legend(frameon=False,fontsize=8)
fig.suptitle('ISCO consumption proxy compared with Pan & Yang (2021)')
fig.text(.5,.01,'Pan: 10⁵ Msun, α-disk, μcap=1; no matched redshift; capture estimate differs from ISCO. Red ×: incomplete rates.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(a.directory/'pan_isco_comparison.png',dpi=180);plt.close(fig)
