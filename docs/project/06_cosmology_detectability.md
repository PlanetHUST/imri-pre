# 06 — Post-gas evolution and LISA

Current source guide, reviewed 2026-10-22. [Part index](README.md).

## Idea

One circular gas-plus-GW trajectory supplies the physical first band entry.
The detector model is conditional on BH formation by that entry.

## Source and data flow

| Owner | Responsibility |
| --- | --- |
| [lisa/snr.py](../../src/lisa/snr.py) | Owns LISA PSD, source-to-detector mass conversion, luminosity distance, mission window and physical inspiral SNR. |
| [physical_process/orbital_evolution.py](../../src/physical_process/orbital_evolution.py) | Owns simultaneous gas/GW radius evolution, disk-off and first moving-band crossing. |
| [lisa/detectability.py](../../src/lisa/detectability.py) | Consumes the supplied entry and decides detector status/threshold; no second production entry orbit. |
| [physical_process/redshift_evolve.py](../../src/physical_process/redshift_evolve.py) | Owns flat-LambdaCDM cosmic age and the source-frame time/redshift mapping. |
| [physical_process/cosmic_merge.py](../../src/physical_process/cosmic_merge.py) | Legacy standalone diagnostic; disabled in the coupled fixed-mass workflow. |
| [workflows/event_work.py](../../src/workflows/event_work.py) | Records detector values and applies the finite-entry counting predicate. |

## Boundaries

Masses enter as source-frame values and frequencies as observer-frame Hz.
The coupled orbit is circular, with `e=0` projected at embedding after the
existing `e<=0.01` check; nonzero configured GW eccentricity is rejected.
Gas-active band entry gets a first-order SNR from the same coupled in-band
trajectory: the circular quadrupole amplitude is weighted by the actual time
spent at each observed frequency and the existing LISA PSD. It does not
include a gas-dependent phase template, so this is a signal-strength forecast,
not a full matched-filter detection guarantee. The saved
`snr_observation_redshift` identifies the SNR window epoch; it can differ from
the first-entry `detection_redshift`. No frequency overlap is an evaluated
zero.
`snr_proxy` is a compatibility alias, not a threshold-valued surrogate.

The cosmic clock and detector luminosity distance use the same configured H0
and Omega_m; the old independent inverse-Myr Hubble input is gone. Elapsed
event time is measured from the host's initial redshift, and the remaining
budget ends at z=0. The AGN duty time uses the same elapsed clock. The
coupled workflow has no mass/radius jump: `cosmic_merge_time_myr` defaults to
infinity and finite values are rejected. Legacy standalone detector tests
retain the old optional merger function.

## Contracts and evidence

- [Current physical model](../physics_model.md)
- [Fixed migration contract](../migration_model_alignment.md)
- [Execution and verification](../engineering_contract.md)
- [Consolidated decisions](../reference/history.md)
- [Preserved pre-cleanup explanation](../archive/project_before_cleanup/06_cosmology_detectability.md)
