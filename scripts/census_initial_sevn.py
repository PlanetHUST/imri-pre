"""Checkpoint SEVN endpoints for every sampled NSC source, without orbit evolution."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import argparse
import json
import os
import sys
import time

import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from workflows.stellar_evolution import evolve_stellar_endpoint
from run_imf_sensitivity import SCENARIOS


def evaluate(checkpoint: Path) -> tuple[str,int]:
    info=json.loads(checkpoint.read_text());d=checkpoint.parent
    initial=pd.read_csv(d/'initial_population.csv')
    ledger=pd.read_csv(d/'imri_stellar_ledger.csv')
    primary=set(ledger.loc[ledger.task_kind=='primary','source_id'])
    tables=Path('/home/planet/Phy/pre-codes/sevn/tables')
    h=tables/'SEVN_parsec_ov05_AGB_popIII';he=tables/'SEVNtracks_parsec_pureHe36'
    rows=[]
    for row in initial.itertuples(index=False):
        try:
            e=evolve_stellar_endpoint(source_id=int(row.source_id),zams_mass_msun=float(row.zams_mass_msun),
                metallicity_zsun=1e-5,task_seed=int(info['task_seed']),hydrogen_tables=h,helium_tables=he)
            item=vars(e).copy();item['endpoint_status']='complete';item['error']=''
        except Exception as exc:
            item={'source_id':row.source_id,'zams_mass_msun':row.zams_mass_msun,
                  'endpoint_status':'unavailable','error':f'{type(exc).__name__}: {exc}'}
        item['disk_crossing_primary']=int(row.source_id) in primary
        rows.append(item)
    frame=pd.DataFrame(rows);target=d/'initial_stellar_endpoints.csv';temp=target.with_suffix('.tmp')
    frame.to_csv(temp,index=False);temp.replace(target)
    done={'status':'complete','source_count':len(rows),'unavailable_count':int((frame.endpoint_status!='complete').sum()),
          'task_seed':info['task_seed']}
    tmp=d/'checkpoint_initial_sevn.tmp';tmp.write_text(json.dumps(done,indent=2)+'\n');tmp.replace(d/'checkpoint_initial_sevn.json')
    return str(d),len(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--repeats',type=int,default=20)
    parser.add_argument('--follow',action='store_true');args=parser.parse_args()
    if args.workers<1 or args.repeats<1:parser.error('workers and repeats must be positive')
    pending={};submitted=set()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while True:
            for name,_,_ in SCENARIOS:
                for index in range(args.repeats):
                    cp=args.directory/name/f'repeat_{index:03d}'/'checkpoint.json'
                    out=cp.parent/'checkpoint_initial_sevn.json'
                    if cp in submitted or out.exists() or not cp.exists():continue
                    if json.loads(cp.read_text()).get('status')!='complete':continue
                    submitted.add(cp);pending[pool.submit(evaluate,cp)]=cp
            if pending:
                finished,_=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
                for future in finished:
                    cp=pending.pop(future);name,n=future.result()
                    print(f'{name}: {n} initial stellar endpoints complete',flush=True)
            elif not args.follow:break
            else:time.sleep(5)
            expected=len(SCENARIOS)*args.repeats
            completed=sum((args.directory/name/f'repeat_{i:03d}'/'checkpoint_initial_sevn.json').exists()
                          for name,_,_ in SCENARIOS for i in range(args.repeats))
            if completed==expected and not pending:break
if __name__=='__main__':main()
