"""Refresh study progress without interpreting incomplete scientific outcomes."""
import argparse
from pathlib import Path
import json
import time

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--follow',action='store_true');a=p.parse_args()
while True:
 counts={};failures=[]
 for cp in a.directory.glob('*/*/*/repeat_*/checkpoint.json'):
  v=json.loads(cp.read_text());key=f"{v['model']}:{v['case']}";item=counts.setdefault(key,{'complete':0,'running':0,'failed':0})
  item[v['status']]+=1
  if v['status']=='failed':failures.append(str(cp))
 result={'expected_new_repeats':600,'completed_new_repeats':sum(x['complete'] for x in counts.values()),'by_model_case':counts,'failures':failures,'unix_time':time.time()}
 temp=a.directory/'checkpoint_progress.tmp';temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(a.directory/'checkpoint_progress.json')
 if not a.follow or failures or result['completed_new_repeats']==600:break
 time.sleep(10)
