# 06 — Post-gas evolution and LISA

Current source guide, reviewed 2026-10-18. [Part index](README.md).

## Idea

The detector model is conditional on the intrinsic event and its physical band entry.

## Source and data flow

| Owner | Responsibility |
| --- | --- |
| [lisa/snr.py](../../src/lisa/snr.py) | Owns LISA PSD, source-to-detector mass conversion, luminosity distance, mission window and physical inspiral SNR. |
| [lisa/detectability.py](../../src/lisa/detectability.py) | Owns post-gas entry evolution and detector status/threshold decisions. |
| [physical_process/redshift_evolve.py](../../src/physical_process/redshift_evolve.py) | Owns the legacy power-law time/redshift relation. |
| [physical_process/cosmic_merge.py](../../src/physical_process/cosmic_merge.py) | Owns the optional one-event mass/radius transformation. |
| [workflows/event_work.py](../../src/workflows/event_work.py) | Records detector values and applies the finite-entry counting predicate. |

## Boundaries

Masses enter as source-frame values and frequencies as observer-frame Hz. Unsupported eccentricity and missing inputs remain unavailable; no frequency overlap is an evaluated zero. snr_proxy is a compatibility alias, not a threshold-valued surrogate.

The legacy cosmic-merger continuation is active for a finite cosmic_merge_time_myr (default 1000 Myr). This differs from the fixed-host intent of the old repair plan. A documentation cleanup does not disable it. The redshift-time law used for evolution is also distinct from the detector luminosity-distance cosmology; record both.

## Contracts and evidence

- [Current physical model](../physics_model.md)
- [Fixed migration contract](../migration_model_alignment.md)
- [Execution and verification](../engineering_contract.md)
- [Consolidated decisions](../reference/history.md)
- [Preserved pre-cleanup explanation](../archive/project_before_cleanup/06_cosmology_detectability.md)
