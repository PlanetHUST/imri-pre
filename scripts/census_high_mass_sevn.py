"""Evaluate sampled high-mass stellar endpoints, including non-crossing sources."""
from pathlib import Path
import argparse
import json
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from workflows.stellar_evolution import evolve_stellar_endpoint

parser=argparse.ArgumentParser()
parser.add_argument('directory',type=Path)
args=parser.parse_args()
tables=Path('/home/planet/Phy/pre-codes/sevn/tables')
h=tables/'SEVN_parsec_ov05_AGB_popIII'
he=tables/'SEVNtracks_parsec_pureHe36'
for checkpoint in sorted(args.directory.glob('*/repeat_*/checkpoint.json')):
    info=json.loads(checkpoint.read_text())
    if info['status']!='complete':continue
    dest=checkpoint.parent
    out=dest/'initial_upper_branch_endpoints.csv'
    if out.exists():continue
    initial=pd.read_csv(dest/'initial_population.csv')
    selected=initial[initial.zams_mass_msun>=240.5]
    stellar=pd.read_csv(dest/'imri_stellar_ledger.csv')
    primary=set(stellar.loc[stellar.task_kind=='primary','source_id'])
    rows=[]
    for item in selected.itertuples(index=False):
        e=evolve_stellar_endpoint(source_id=int(item.source_id),zams_mass_msun=float(item.zams_mass_msun),
          metallicity_zsun=1e-5,task_seed=int(info['task_seed']),hydrogen_tables=h,helium_tables=he)
        row=vars(e).copy();row['disk_crossing_primary']=int(item.source_id) in primary;rows.append(row)
    columns=['source_id','zams_mass_msun','metallicity_absolute_z','collapse_time_myr','presupernova_mass_msun',
             'presupernova_phase','remnant_type','remnant_mass_msun','kick_speed_km_s','seed','disk_crossing_primary']
    tmp=out.with_suffix('.tmp');pd.DataFrame(rows,columns=columns).to_csv(tmp,index=False);tmp.replace(out)
    print(f'{dest}: {len(rows)} high-mass endpoints',flush=True)
