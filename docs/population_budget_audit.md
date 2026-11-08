# Why was the legacy damping / embedding rate only about 10 per Myr?

This audits the archived analytic-IMF model. The revised component masses, initial BH reservoir and radial laws are described in [published populations](published_populations.md); their population census is separate.

Reproduce the audit without changing the saved simulations:

```bash
python scripts/check_population_budget.py
```

The audit checks all saved high-redshift population budgets, reconstructs the reference initial populations from their original seeds, and reruns only the reference primary damping tasks to recover final eccentricity/inclination/containment predicates. It does not change capture criteria, remnant prescriptions, detector thresholds or production rates.

## Reference budget

For z=10, M_bullet=1e5 Msun, M_NSC=3e6 Msun, the mean counts per **10-Myr realization** are:

| Stage | Mean objects |
|---|---:|
| Initial population | 67,658.67 |
| Initial disk crossers | 420.00 |
| Primary successful embeddings | 81.00 |
| Primary embedded stars | 65.33 |
| Primary embedded BHs | 15.67 |
| Primary damping incomplete | 337.67 |
| Primary outside disk | 1.33 |
| Replacement tasks | 17.67 |
| Replacement successful embeddings | 2.00 |
| All successful embeddings | 83.00 |
| All embedded stars | 65.33 |
| All embedded BHs / BH IMRI entries | 17.67 |
| BHs formed during migration | 0 |
| Detectable entries | 0 |

Thus the plotted 8.3 Myr^-1 is **(stars + BHs) successfully embedded / 10 Myr**. It is not BH-only, not the number of disk-plane passages, and not a steady-state supply flux. The initial geometric pool divided by duration would be 42 Myr^-1, but those objects do not all finish damping.

## Main supply restrictions

1. **Population assumption differs from a general NSC.** Z=1e-5 Zsun selects the 13.6–197 Msun top-heavy IMF. Mean mass is 44.34 Msun, so a 3e6-Msun cluster contains about 67,659 sampled objects, not millions of low-mass stars. There is no independently initialized old-BH reservoir. The default fixed-mass stellar model is coeval at AGN start and switches to BH drag after an assigned lifetime, with no calibrated remnant-mass mapping or endpoint filter for BH/NS/pair-instability outcomes. Adding low-mass stars at fixed total NSC mass would also change the high-mass mass budget; it cannot simply multiply BH supply.
2. **The disk is tiny compared with the NSC.** The fixed reference edge is 0.003226 pc, versus an NSC radius of 3.122 pc. Only 0.621% of initial objects have an orbital node inside the annulus. These 420 crossers, not the full NSC, are initial damping tasks.
3. **Complete capture has several conditions.** Successful primary capture among crossers is 19.29%. Handoff requires inclination within the local disk thickness, eccentricity <=0.01, radial containment and completion within the AGN lifetime.
4. **The supply is not continuously replenished from a relaxed cluster.** Existing replacements are unique donors on channel orbits after eligible BH IMRI/ISCO consumptions. Relaxation times are diagnostic, not a stochastic angular-momentum-diffusion inflow. Noncandidate stellar entries do not trigger refilling; external supply is absent.

## A verified post-alignment handoff limitation

All **1,260** primary outcomes across the three reference repeats were reproduced. **101 incomplete outcomes stop before the AGN deadline**. Every one of these is already inclination-damped but still fails the eccentricity criterion; one also fails radial containment.

The loop in `DampingCalculator.damping_simulation` evolves while inclination is not damped. Once inclination qualifies, it stops and checks the stricter full embedding predicate. An eccentric object can therefore be returned as incomplete with time remaining. This is an unresolved post-alignment handoff/evolution phase, not necessarily physically slow damping all the way to the deadline.

It would be unsafe to fix this by relabeling such objects as fully embedded. Simply applying the disk-crossing impulse approximation indefinitely inside the disk also needs validation. The eccentric, low-inclination embedded phase should be modeled and tested explicitly before recalculating rates. Integration timesteps can additionally overshoot the AGN end; those late states remain rejected. The audit reports existing behavior, not a corrected capture rate.

## Why no BHs form during migration in this reference run

Across the three repeats, 196 successful stellar embeddings reach the *nominal* selected band before their assigned collapse time. Their median remaining lifetime at entry is 2.489 Myr; median embedding-to-entry time is only 0.01726 Myr. The migration-formed-BH classifier is active and covered by a timing test, but none of these particular tracks select it.

A point-mass vacuum diagnostic gives only 0.008–0.115 source years from nominal entry to ISCO, much shorter than their residual stellar lifetimes. This is **not a physical prediction for finite stars**: using the same stellar radius scaling as the drag model, the estimated tidal radius exceeds the nominal band-entry radius for all 196. Stellar disruption, survival and consumption/refilling require explicit treatment. Those objects must not simply be added to BH candidate counts or treated as a validated extra replenishment rate.

## Does too few candidates cause low SNR?

Not directly. Individual SNR depends on binary masses, distance, frequency evolution and observing window, not the number of other candidates. Fewer candidates reduce the number of draws from a loud tail. More candidates cannot create a tail excluded by the population model.

For a detector-only 1e5-Msun primary at fixed z, with the same four-year, orientation-averaged model and 1-mHz cutoff:

| z | SNR for a 197-Msun secondary | Secondary mass giving SNR 20 |
|---:|---:|---:|
| 8 | 17.47 | 258.5 Msun |
| 10 | 13.76 | 418.1 Msun |
| 15 | 7.25 | 1,582.8 Msun |

These are detector controls, not new populations or corrected physical rates. Actual entry redshifts evolve slightly, so they are not exact universal bounds. Nevertheless they explain why increasing supply within the same capped mass model does not automatically restore detectable events. Orientation averaging versus orientation-marginalized selection is a separate issue, discussed in the [detector audit](detectability_audit.md).

## Recommended order of further work

1. Specify the intended NSC population: young coeval Pop III cluster versus an age-distributed cluster with low-mass stars and a pre-existing BH component. Validate mass normalization and remnant formation.
2. Validate the low-inclination eccentric damping/handoff phase and time-boundary handling.
3. Add or validate finite-star disruption/consumption and associated supply/refill rules; distinguish these from BH IMRIs.
4. Then rerun the full population, capture, migration and detector selection. Do not rescale existing rates or change thresholds merely to force detections.

Detailed artifacts are under `outputs/population_budget_audit/`: `report.md`, `population_census.csv`, `primary_damping_replay.csv`, `reconstructed_reference_stock.csv`, `embedded_stellar_timing.csv`, `geometry_only_sensitivity.csv`, `secondary_mass_snr_controls.csv` and `checks.json`. Geometry-only edge variations are supply diagnostics, not re-solved stable disks or capture-rate predictions.
