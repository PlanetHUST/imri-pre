# Physical model and its limits

The implemented model follows one host through an AGN episode. The [workflow](main_workflow.md) describes the action order; the source modules below contain the equations. `scripts/` and `src/viz/` must not define alternative physics.

| Stage | Source owner | Important boundary |
| --- | --- | --- |
| Disk and opacity | `src/disk_quantity/pagn_disk_adapter.py`, `opacity.py` | pAGN disk; comparison solver is not the production disk |
| Population, orbits and crossing | `src/physical_process/population_models.py`, `sampling.py`, `nsc_orbital.py` | explicit component budgets; spatial density is not an a-CDF |
| Damping into disk | `src/physical_process/damp.py` | failed or unavailable embedding has its own status |
| Migration | `src/physical_process/migration/type_i.py` | default local Type-I timescale, no Type-II branch or traps; signed torque/gaps/traps remain optional comparisons |
| Coupled gas/GW orbit | `src/physical_process/orbital_evolution.py` | gas during the AGN episode; GW can continue afterwards |
| Band entry and event classification | `src/workflows/event_rate.py`, `event_work.py` | first entry, rather than every crossing |
| Finite-source replenishment | `src/workflows/single_work.py`, `replenishment.py` | donor only after eligible ISCO consumption during duty time |
| LISA SNR and detectability | `src/lisa/snr.py`, `detectability.py` | circular detector model; unavailable evaluation is not a zero SNR |

The default CLI population is restored to the original analytic IMF (`legacy_popiii`), with gamma=1.75 and bounds 13.6–197 Msun at slope 1.77. Custom bounds remain available. The opt-in experimental `pan2021_nsc` population uses: 1-Msun stars plus pre-existing 10-Msun BHs with number ratio 1e-3 and total mass fixed to M_NSC. Its full Tremaine spatial tracer shape is obtained by isotropic Eddington inversion in the Kepler potential, with fiducial inner gamma=1.5. It does not include Pan's stellar potential or Fokker-Planck evolution. Xue's dN/dr exponent 0 corresponds to a spherical-average density exponent -2, not uniform spatial density. See [published population prescriptions](published_populations.md); exact paper number normalizations and the conditional NSC-mass experiment are distinct options. The older IMF/lifetime model remains explicitly available.

The default gas drift uses the inward Type-I local timescale on the pAGN disk:

```text
t_I(R) = M_SMBH² h(R)² / (2 m Σ(R) R² Ω(R))
dr/dt = -R / t_I(R)
```

The orbit solver integrates this drift simultaneously with GW losses. The default never switches to Type-II migration, even in cells satisfying a gap criterion, and has no torque traps. The optional Type-I/II comparison is in `src/optional_models/semi_analytic.py`; signed gravitational/thermal/gap torque with derived traps is reusable physics under `src/physical_process/migration/`. Neither is selectable in `scripts/main.py`. `src/optional_models/sevn.py` and the damping benchmark adapters are also opt-in.

The main event path uses a coupled circular gas and GW orbit and an analytic LISA sensitivity model. Opt-in SEVN stellar remnants and natal kicks require external SEVN tables and backend; eccentric post-kick sources that the circular detector model cannot evaluate are reported as unavailable. A host-level rate cannot be interpreted as a cosmological rate. Numerical approximations, disk-domain restrictions and detector/model-unavailable flags should be respected when interpreting any result; inspect the event rows and repeat statuses, not just the averaged count.

See [`src/workflows/config.py`](../src/workflows/config.py) for current defaults and [`src/workflows/outputs.py`](../src/workflows/outputs.py) for saved schemas. Tests cover the core contracts, but do not replace convergence studies.
