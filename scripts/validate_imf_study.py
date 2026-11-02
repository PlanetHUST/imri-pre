"""Check population identity, finite mass budgets, and endpoint ledgers."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
count=0
for cp in sorted(a.directory.glob('*/repeat_*/checkpoint.json')):
    info=json.loads(cp.read_text())
    if info['status']!='complete':continue
    d=cp.parent; initial=pd.read_csv(d/'initial_population.csv');stellar=pd.read_csv(d/'imri_stellar_ledger.csv')
    assert np.array_equal(initial.source_id.to_numpy(),np.arange(len(initial)))
    assert initial.zams_mass_msun.sum()>=2e5 and initial.zams_mass_msun.iloc[:-1].sum()<2e5
    m=stellar.merge(initial,on='source_id',suffixes=('_ledger','_initial'),validate='many_to_one')
    assert len(m)==len(stellar) and np.allclose(m.zams_mass_msun_ledger,m.zams_mass_msun_initial,rtol=1e-8)
    high=d/'initial_upper_branch_endpoints.csv'
    if high.exists():
        endpoints=pd.read_csv(high);expected=initial[initial.zams_mass_msun>=240.5]
        assert set(endpoints.source_id)==set(expected.source_id)
        same=stellar.merge(endpoints,on='source_id',suffixes=('_ledger','_all'))
        if len(same):
            assert (same.remnant_type_ledger==same.remnant_type_all).all()
            assert np.allclose(same.remnant_mass_msun_ledger,same.remnant_mass_msun_all,rtol=1e-8,atol=1e-8)
            assert np.allclose(same.kick_speed_km_s_ledger,same.kick_speed_km_s_all,rtol=1e-8,atol=1e-8)
    census=d/'initial_stellar_endpoints.csv'
    if census.exists():
        full=pd.read_csv(census)
        assert len(full)==len(initial) and set(full.source_id)==set(initial.source_id)
        assert full.source_id.is_unique
        matched=stellar.merge(full,on='source_id',suffixes=('_ledger','_all'))
        available=matched[matched.endpoint_status=='complete']
        assert len(available)==len(stellar)
        assert (available.remnant_type_ledger==available.remnant_type_all).all()
        assert np.allclose(available.remnant_mass_msun_ledger,available.remnant_mass_msun_all,rtol=1e-8,atol=1e-8)
        assert np.allclose(available.kick_speed_km_s_ledger,available.kick_speed_km_s_all,rtol=1e-8,atol=1e-8)
    count+=1
print(f'validated {count} completed repeat populations and all available SEVN endpoint censuses')
