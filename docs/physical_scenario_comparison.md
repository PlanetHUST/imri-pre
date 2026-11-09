# Physical-scenario rate comparison

The four approved cases use the restored benchmark: z=10, M_bullet=1e5 Msun, total M_NSC=3e6 Msun, original 13.6–197 Msun IMF with slope 1.77, density index 1.75, three original paired seeds, 10-Myr AGN duration. All disk, population, numerical damping and detector parameters are copied from the archived reference metadata.

| Case | Migration | Finite NSC-object refill |
|---|---|---|
| Baseline | Existing analytical Type-I timescale | On |
| Torque I + thermal | Gamma_I + Gamma_th, no gap corrections | On |
| Torque + gap/viscous | f_gap Gamma_I + Gamma_th + (1-f_gap) Gamma_viscous | On |
| No refill | Same analytical Type-I as baseline | Off |

The ungapped case uses the unperturbed optical depth in its thermal term as well as disabling gravitational gap attenuation and viscous transport. Both torque cases retain signed torques and the existing thermal applicability mask; neither enables thermal extrapolation. Diagnostic gap quantities are still computed for both modes. The continuous gap/viscous prescription is **not** a sharp classical Type-II switch. Ungapped linear torques in a gap-forming regime are a controlled counterfactual, not a claim of physical validity there.

Hydrodynamically calibrated damping and its final-radius closure are **pending**. They have not been substituted or extrapolated. “Refill off” does not switch off disk gas fueling, change Mdot or introduce/remove external gas inflow; it disables only the existing finite-object donor replacement.

## Run / replot

```bash
python scripts/compare_physical_scenarios.py
python scripts/compare_physical_scenarios.py --plot-only
```

Outputs under `outputs/physical_scenario_rates/`:

- `physical_scenario_rates.png`, `.pdf`, `.svg`: three clean panels, categorical scenario axis, common y scale and units Myr^-1;
- `scenario_rates.csv`, `scenario_repeats.csv`: means, sampling SEM, individual repeat rates and evaluation statuses;
- `scenario_entry_diagnostics.csv`: delayed entries/detections, actual entry-redshift ranges and migration-formed BH counts;
- `representative_torque_profiles.csv`: unweighted radial-cell diagnostics on a small representative mass grid; these are not event-weighted validity fractions;
- `ppt_notes.md`, `experiment.json`: definitions, assumptions, provenance and model settings outside the figure;
- per-case standard simulation outputs and logs. The baseline is reused from its compatible saved run, never rescaled.

The runner compares every case's **primary** damping records with the benchmark. Total damping counts can differ because migration changes subsequent consumption/refill histories, not because the primary damping law was changed.

## Completed three-repeat result

All four cases have complete intrinsic and detector evaluations, and identical primary damping records. Mean rates in Myr^-1:

| Case | Damping | IMRI | Detectable |
|---|---:|---:|---:|
| Baseline | 8.300 | 1.767 | 0 |
| Torque I + thermal | 8.133 | 0.667 | 0 |
| Torque + gap/viscous | 8.400 | 2.067 | 0.133 |
| No refill | 8.100 | 1.567 | 0 |

Primary-only damping is 8.1 Myr^-1 in every case. Total differences reflect the changed consumption/refill histories. The ungapped signed-torque case yields fewer IMRI entries; the continuous gap/viscous case also has 11 migration-formed BH entries across the three realizations, versus none in the baseline. These are finite-sample differences, not convergence or statistical-significance claims.

**Crucial detection caveat:** all four positive detections in the gap/viscous case occur AFTER the 10-Myr AGN episode: about 441, 6,408, 7,387 and 9,101 Myr after initial z=10, at actual entry redshifts about 6.08, 0.78, 0.61 and 0.38. They do not show a nonzero detectable rate during the z=10 AGN episode. Restricting to entries within that episode gives detectable rate **zero in all four cases**. The gap/viscous in-episode intrinsic rate is 1.833 Myr^-1; the quoted 2.067 includes seven delayed entries across three repeats. Gyr-delayed fixed-mass vacuum continuations are conditional benchmark yields, not a calibrated model of long-term host/SMBH growth.

Sampling SEM, counts, entry delays and redshifts are saved in the CSVs and `ppt_notes.md`; these details are deliberately outside the figure.

## Why Radau is used only for torque cases

A signed torque can create a stable equilibrium in the net radial drift v(r). For a small displacement near r_*:

```text
v(r_*) = 0
 d(delta r)/dt ≈ v'(r_*) delta r
 tau_restore = -1/v'(r_*)   for v'(r_*) < 0
```

This is the orbital **restoring timescale**, not the NSC two-body relaxation time. Representative ungapped profiles contain stable roots near 1.82e5 and 4.21e5 gravitational radii, with restoring times about 0.06–0.85 years. An explicit solver must take small steps for numerical stability even after settling near the trap, potentially millions of steps over a Myr-scale gas episode. This is stiffness, not necessarily rapid long-distance migration.

After user approval, torque cases use implicit Radau; analytical Type-I cases retain DOP853. The same signed gas+GW drift, tolerances, boundaries, entry events, disk-off rules and analytic vacuum branch remain in place. There is no new trap-freezing rule, force clipping or altered migration physics. Solver selection propagates through initial evolution, in-band continuation, ISCO consumption and disk-contact recursion; metadata records it and caches reject old torque runs with a different method.

Focused tests compare nonstiff entry/ISCO outcomes between the two methods and verify a stiff restoring trap and event deadline. The initial slow explicit attempt is retained as provenance rather than treated as a zero-rate result.

## Interpretation limits

Rates retain the benchmark's definition: finite physical band-entry counts divided by source-frame AGN duration, including entries after the disk episode. They are not instantaneous rates at the initial z. Delayed entry redshifts must be inspected before interpreting a change in detectability.

The existing torque prescription includes an Eddington-capped compact-object thermal-luminosity model; applying the same migration path to a star awaiting collapse does not add a calibrated stellar radiative-feedback model. Post-alignment eccentric damping, finite-star disruption, remnant mapping, Kepler-potential assumptions and the donor-channel approximation remain unchanged. No model difference here repairs those limitations.

N/A values remain unavailable rather than becoming zero. Zero sampled detections are not zero upper confidence limits. Three-repeat SEM is not a convergence study or a physical-systematics uncertainty.

## Direct CLI controls

The default remains analytical Type-I. Explicit torque controls are:

```bash
python scripts/main.py 3 outputs/torque_ungapped 6 \
  --migration-model torque_with_trap --torque-no-gap-transport \
  --smbh-masses 1e5 --nsc-masses 3e6 --redshift 10 --base-seed 20261107
```

Omit `--torque-no-gap-transport` for the existing continuous gap/viscous branch. `--no-consumption-replenishment` independently disables finite-object refill. To reproduce the figure exactly, use the comparison runner, which additionally copies the saved disk edge and other benchmark controls instead of falling back to generic CLI defaults.
