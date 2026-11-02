"""Run checkpointed redshift/lifetime comparisons on the existing IMF seeds."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from run_imf_sensitivity import SCENARIOS, H, HE, write_json

CASES={'z10_duty10':(10,10),'z15_duty10':(15,10),'z15_duty100':(15,100)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=list(CASES),required=True)
    p.add_argument('--output-dir',type=Path,default=ROOT/'output/imf_redshift_lifetime_20261029')
    p.add_argument('--baseline-dir',type=Path,default=ROOT/'output/imf_sensitivity_20261029')
    p.add_argument('--repeats',type=int,default=20);p.add_argument('--cpus',type=int,default=2)
    p.add_argument('--model',choices=['torque','semi'],default='torque')
    a=p.parse_args();redshift,duty=CASES[a.case]
    if a.repeats<1 or a.cpus<1:p.error('positive repeats and cpus required')
    if a.model=='torque' and a.case=='z15_duty10':p.error('baseline already exists in imf_sensitivity_20261029')
    env=os.environ.copy();env.update(PYTHONPATH=str(ROOT/'src'),MPLCONFIGDIR='/tmp/mpl-imf-study',MPLBACKEND='Agg',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for name,maximum,slope in SCENARIOS:
        for i in range(a.repeats):
            baseline=a.baseline_dir/name/f'repeat_{i:03d}';old=json.loads((baseline/'checkpoint.json').read_text())
            if old['status']!='complete':raise RuntimeError(f'incomplete baseline {baseline}')
            dest=a.output_dir/a.model/a.case/name/f'repeat_{i:03d}';cp=dest/'checkpoint.json'
            if cp.exists() and json.loads(cp.read_text()).get('status')=='complete':continue
            dest.mkdir(parents=True,exist_ok=True);shutil.copyfile(baseline/'initial_population.csv',dest/'initial_population.csv')
            args=['1',str(dest),str(a.cpus),'--smbh-masses','1e5','--nsc-masses','2e5','--redshift',str(redshift),
                '--metallicity','1e-5','--duty-cycle-myr',str(duty),'--base-seed',str(old['base_seed']),
                '--stellar-evolution-model','sevn','--natal-kick-model','sevn',
                '--sevn-hydrogen-tables',str(H),'--sevn-helium-tables',str(HE),
                '--popiii-max-mass-msun',str(maximum),'--popiii-imf-slope',str(slope)]
            if a.model=='torque': cmd=[sys.executable,str(ROOT/'scripts/main.py'),*args]
            else:cmd=[sys.executable,str(ROOT/'scripts/run_imf_semi_case.py'),*args]
            state={'status':'running','scenario':name,'case':a.case,'model':a.model,'repeat':i,
                   'base_seed':old['base_seed'],'task_seed':old['task_seed'],'command':cmd,'baseline':str(baseline)}
            write_json(cp,state)
            with (dest/'run.log').open('w') as f:result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            if result.returncode:
                state.update(status='failed',returncode=result.returncode);write_json(cp,state)
                raise RuntimeError(f'run failed: {dest}/run.log')
            state['status']='complete';write_json(cp,state)
            print(f'{a.model} {a.case} {name} repeat {i+1}/{a.repeats} complete',flush=True)
if __name__=='__main__':main()
