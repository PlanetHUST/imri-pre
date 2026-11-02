"""Add pre-collapse core masses to the checkpointed SEVN endpoint map."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sevnpy.sevn import SEVNmanager, Star

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
t=Path('/home/planet/Phy/pre-codes/sevn/tables')
SEVNmanager.init({'tables':str(t/'SEVN_parsec_ov05_AGB_popIII'),'tables_HE':str(t/'SEVNtracks_parsec_pureHe36')})
df=pd.read_csv(a.directory/'sevn_endpoints.csv');rows=[]
for row in df.itertuples(index=False):
    star=Star(Mzams=row.zams_mass_msun,Z=row.metallicity_absolute_z,rseed=int(row.seed));star.evolve('end')
    pre=star.evolve_table.iloc[-2];end=star.evolve_table.iloc[-1]
    assert int(end['RemnantType'])==row.remnant_type
    if row.remnant_type!=-1:assert np.isclose(float(end['Mass']),row.remnant_mass_msun,rtol=1e-12)
    rows.append({'presupernova_helium_core_msun':float(pre['MHE']), 'presupernova_co_core_msun':float(pre['MCO'])})
out=pd.concat([df.reset_index(drop=True),pd.DataFrame(rows)],axis=1)
out.to_csv(a.directory/'sevn_endpoint_core_map.csv',index=False)
print(f'validated and enriched {len(out)} endpoints')
