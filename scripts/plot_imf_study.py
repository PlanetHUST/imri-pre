"""Plot completed-repeat above-branch production and IMRI counts."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
d=pd.read_csv(a.directory/'repeat_summary.csv')
names=['max197_slope177','max250_slope177','max300_slope177','max600_slope177','max600_slope100','max600_slope230']
labels=['197 / 1.77','250 / 1.77','300 / 1.77','600 / 1.77','600 / 1.00','600 / 2.30']
fig,ax=plt.subplots(1,2,figsize=(11,4.4))
for i,(column,title) in enumerate([('n_all_upper_branch_bh','Predicted NSC BH endpoints'),('n_imri_zams_ge240p5','Intrinsic IMRI entries')]):
    for j,name in enumerate(names):
        s=d.loc[d.scenario==name,column].dropna();mean=s.mean() if len(s) else np.nan
        sem=s.std(ddof=1)/np.sqrt(len(s)) if len(s)>1 else 0
        ax[i].bar(j,mean,yerr=sem,color='tab:blue',capsize=3)
        if len(s):ax[i].text(j,mean+sem,f'n={len(s)}',ha='center',va='bottom',fontsize=8)
    ax[i].set_xticks(range(len(names)),labels,rotation=35,ha='right')
    ax[i].set(xlabel=r'IMF maximum [$M_\odot$] / slope $\alpha$',ylabel='Count per AGN episode',title=title)
fig.suptitle(r'Upper-branch diagnostic: $M_{\rm ZAMS}\geq240.5\,M_\odot$')
fig.text(0.5, 0.01, 'Error bars: SEM. The 300 / 1.77 case includes one incomplete repeat.', ha='center', fontsize=8)
fig.tight_layout(rect=(0, 0.05, 1, 1));fig.savefig(a.directory/'imf_upper_branch_counts.png',dpi=180);plt.close(fig)
