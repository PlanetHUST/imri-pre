# 05 — Gas/GW routing and intrinsic events

Current source guide, reviewed 2026-10-18. [Part index](README.md).

## Idea

The per-object workflow combines gas transport with circular Peters evolution and distinguishes candidacy from an actual finite band entry.

## Source and data flow

| Owner | Responsibility |
| --- | --- |
| [physical_process/gw_frequency_radius.py](../../src/physical_process/gw_frequency_radius.py) | Owns circular GW times, frequency/radius conversion and ISCO helpers. |
| [workflows/event_rate.py](../../src/workflows/event_rate.py) | Owns gas/GW competition, EventRateDecision and is_physical_imri_entry. |
| [workflows/event_work.py](../../src/workflows/event_work.py) | Coordinates band-entry preflight, migration, detector evaluation and the final event ledger. |

## Boundaries

Formal observer-frequency entry is checked against ISCO before profile construction. A valid entry below the gas grid uses a gas-to-GW handoff; entry inside ISCO is unsupported, not a zero or profile failure. Direct finite GW delivery can remain available despite a blocked gas path. Gas-phase shutdown does not itself count an event: the headline numerator also requires finite physical band entry. Post-gas GW entry may occur after the AGN episode.

## Contracts and evidence

- [Current physical model](../physics_model.md)
- [Fixed migration contract](../migration_model_alignment.md)
- [Execution and verification](../engineering_contract.md)
- [Consolidated decisions](../reference/history.md)
- [Preserved pre-cleanup explanation](../archive/project_before_cleanup/05_gw_event_classification.md)
