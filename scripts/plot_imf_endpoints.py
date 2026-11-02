"""Plot a checkpointed SEVN endpoint grid without running stellar evolution."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=Path)
args = parser.parse_args()
df = pd.read_csv(args.directory / 'sevn_endpoints.csv').sort_values('zams_mass_msun')
fig, ax = plt.subplots(figsize=(8, 5))
for remtype, label, color in [(6, 'BH remnant', 'tab:blue'), (5, 'neutron star', 'tab:orange'), (-1, 'no remnant', 'tab:red')]:
    selected = df[df.remnant_type == remtype]
    ax.scatter(selected.zams_mass_msun, selected.remnant_mass_msun, s=10, label=label, color=color)
ax.axvline(197, color='black', linestyle='--', label='Current IMF maximum')
ax.set(xlabel=r'Initial stellar mass [$M_\odot$]', ylabel=r'Remnant mass [$M_\odot$]',
       title=r'SEVN endpoints, $Z=10^{-5} Z_\odot$')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(args.directory / 'sevn_endpoints.png', dpi=180)
plt.close(fig)
