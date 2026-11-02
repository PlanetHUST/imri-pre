"""Checkpointed SEVN endpoint map and single-halo Pop III IMF experiment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from physical_process.sampling import sample_efficient
from workflows.stellar_evolution import evolve_stellar_endpoint
from workflows.workflow import derive_task_seed

TABLES = Path('/home/planet/Phy/pre-codes/sevn/tables')
H = TABLES / 'SEVN_parsec_ov05_AGB_popIII'
HE = TABLES / 'SEVNtracks_parsec_pureHe36'
SCENARIOS = [('max197_slope177', 197., 1.77), ('max250_slope177', 250., 1.77),
             ('max300_slope177', 300., 1.77), ('max600_slope177', 600., 1.77),
             ('max600_slope100', 600., 1.), ('max600_slope230', 600., 2.30)]


def write_json(path: Path, value: dict) -> None:
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def endpoint_map(root: Path) -> None:
    path = root / 'sevn_endpoints.csv'
    masses = np.unique(np.concatenate([np.arange(14., 601., 2.), np.arange(118., 131., 0.5),
                                       np.arange(236., 251., 0.5), [13.6, 197., 250., 300., 600.]]))
    existing = pd.read_csv(path).to_dict('records') if path.exists() else []
    done = {float(x['zams_mass_msun']) for x in existing}
    for mass in masses:
        if float(mass) in done:
            continue
        ep = evolve_stellar_endpoint(source_id=int(round(mass * 10)), zams_mass_msun=float(mass),
            metallicity_zsun=1e-5, task_seed=12345, hydrogen_tables=H, helium_tables=HE)
        existing.append(vars(ep))
        pd.DataFrame(existing).sort_values('zams_mass_msun').to_csv(path, index=False)
        write_json(root / 'checkpoint_endpoints.json', {'completed': len(existing), 'total': len(masses),
                   'last_mass_msun': mass, 'status': 'complete' if len(existing) == len(masses) else 'running'})


def simulate(root: Path, repeats: int, cpu: int) -> None:
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT / 'src'), MPLCONFIGDIR='/tmp/mpl-imf-study', MPLBACKEND='Agg',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    for name, maximum, slope in SCENARIOS:
        for index in range(repeats):
            dest = root / name / f'repeat_{index:03d}'
            done = dest / 'checkpoint.json'
            if done.exists() and json.loads(done.read_text()).get('status') == 'complete':
                continue
            dest.mkdir(parents=True, exist_ok=True)
            base_seed = 12345 + index
            task_seed = derive_task_seed(base_seed, 0, 0, 0)
            np.random.seed(task_seed)
            masses, _ = sample_efficient(2e5, 1e-5, popiii_max_mass_msun=maximum, popiii_imf_slope=slope)
            pd.DataFrame({'source_id': np.arange(len(masses)), 'zams_mass_msun': masses}).to_csv(dest / 'initial_population.csv', index=False)
            cmd = [sys.executable, str(ROOT/'scripts/main.py'), '1', str(dest), str(cpu),
                   '--smbh-masses','1e5','--nsc-masses','2e5','--redshift','15','--metallicity','1e-5',
                   '--duty-cycle-myr','10','--base-seed',str(base_seed), '--stellar-evolution-model','sevn',
                   '--natal-kick-model','sevn','--sevn-hydrogen-tables',str(H),'--sevn-helium-tables',str(HE),
                   '--popiii-max-mass-msun',str(maximum),'--popiii-imf-slope',str(slope)]
            write_json(done, {'status':'running','scenario':name,'repeat':index,'base_seed':base_seed,'task_seed':task_seed,'command':cmd})
            with (dest/'run.log').open('w') as log:
                result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                write_json(done, {'status':'failed','returncode':result.returncode,'scenario':name,'repeat':index,'base_seed':base_seed,'task_seed':task_seed,'command':cmd})
                raise RuntimeError(f'{name} repeat {index} failed: {dest / "run.log"}')
            write_json(done, {'status':'complete','scenario':name,'repeat':index,'base_seed':base_seed,'task_seed':task_seed,'command':cmd})
            print(f'{name} repeat {index + 1}/{repeats} complete', flush=True)
            summarize(root)


def summarize(root: Path) -> None:
    rows = []
    for name, maximum, slope in SCENARIOS:
        for dest in sorted((root/name).glob('repeat_*')):
            cp = dest/'checkpoint.json'
            if not cp.exists() or json.loads(cp.read_text()).get('status') != 'complete':
                continue
            info = json.loads(cp.read_text())
            init = pd.read_csv(dest/'initial_population.csv')
            stellar = pd.read_csv(dest/'imri_stellar_ledger.csv')
            events = pd.read_csv(dest/'imri_events.csv')
            rep = pd.read_csv(dest/'imri_repeat_results.csv').iloc[0]
            primary = stellar[stellar.task_kind == 'primary']
            candidate = events[events.is_imri_candidate == True]
            detectable = events[events.is_lisa_detectable == True]
            # 240.5 Msun is a diagnostic ZAMS threshold from a fine single-star
            # probe, not a universal pair-instability or BH mass-gap boundary.
            row = {'scenario':name,'repeat':info['repeat'],'base_seed':info['base_seed'],
                'max_mass_msun':maximum,'imf_slope':slope,'n_initial':len(init),
                'initial_mass_msun':init.zams_mass_msun.sum(), 'n_initial_zams_ge240p5':int((init.zams_mass_msun >= 240.5).sum()),
                'n_primary_crossing':len(primary), 'n_crossing_zams_ge240p5':int((primary.zams_mass_msun >= 240.5).sum()),
                'n_evaluated_bh_endpoints':int((stellar.remnant_type==6).sum()),
                'n_evaluated_bh_zams_ge240p5':int(((stellar.remnant_type==6)&(stellar.zams_mass_msun>=240.5)).sum()),
                'n_primary_bh_zams_ge240p5':int(((primary.remnant_type==6)&(primary.zams_mass_msun>=240.5)).sum()),
                'n_primary_embedded_bh_zams_ge240p5':int(((primary.damping_status=='embedded_bh')&(primary.zams_mass_msun>=240.5)).sum()),
                'n_imri':len(candidate),'n_imri_zams_ge240p5':int((candidate.zams_mass_msun>=240.5).sum()),
                'n_detectable':len(detectable),'n_detectable_zams_ge240p5':int((detectable.zams_mass_msun>=240.5).sum()),
                'rate_status':rep['rate_status'],'detector_rate_status':rep['detector_rate_status'],
                'n_stellar_unavailable':int((stellar.damping_status=='stellar_unavailable').sum())}
            census_path = dest / 'initial_stellar_endpoints.csv'
            if census_path.exists():
                census = pd.read_csv(census_path)
                bh = census.remnant_type.eq(6)
                upper = census.zams_mass_msun.ge(240.5)
                row.update(n_all_bh_endpoints=int(bh.sum()),
                           n_all_upper_branch_bh=int((bh & upper).sum()),
                           n_all_no_remnant=int(census.remnant_type.eq(-1).sum()),
                           n_all_stellar_unavailable=int(census.endpoint_status.ne('complete').sum()))
            rows.append(row)
    if rows:
        df=pd.DataFrame(rows); df.to_csv(root/'repeat_summary.csv',index=False)
        numeric=['n_initial','n_initial_zams_ge240p5','n_primary_crossing','n_crossing_zams_ge240p5',
                 'n_evaluated_bh_zams_ge240p5','n_primary_bh_zams_ge240p5','n_primary_embedded_bh_zams_ge240p5',
                 'n_imri','n_imri_zams_ge240p5','n_detectable','n_detectable_zams_ge240p5']
        numeric += [c for c in ['n_all_bh_endpoints','n_all_upper_branch_bh', 'n_all_no_remnant','n_all_stellar_unavailable'] if c in df.columns]
        summary=df.groupby('scenario',sort=False)[numeric].agg(['mean','sum','count'])
        summary.columns=['_'.join(x) for x in summary.columns];summary.to_csv(root/'scenario_summary.csv')
        write_json(root/'checkpoint_study.json',{'completed_repeats':len(df), 'scenario_completed':df.groupby('scenario').size().to_dict(), 'updated_unix':time.time()})


def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir', type=Path, default=ROOT/'output/imf_sensitivity_20261029')
    ap.add_argument('--repeats',type=int,default=20);ap.add_argument('--cpus',type=int,default=2)
    ap.add_argument('--stage',choices=['all','endpoints','simulate','summarize'],default='all')
    a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    if a.repeats<1 or a.cpus<1:ap.error('repeats and cpus must be positive')
    if a.stage in ('all','endpoints'):endpoint_map(a.output_dir)
    if a.stage in ('all','simulate'):simulate(a.output_dir,a.repeats,a.cpus)
    if a.stage in ('all','summarize'):summarize(a.output_dir)
if __name__=='__main__':main()
