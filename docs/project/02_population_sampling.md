# 02 — Population, orbits and crossing

Current source guide, reviewed 2026-10-18. [Part index](README.md).

## Idea

The initial population and node geometry supply the objects that damping actually evolves.

## Source and data flow

| Owner | Responsibility |
| --- | --- |
| [physical_process/sampling.py](../../src/physical_process/sampling.py) | Samples Pop II/Pop III masses and stellar lifetimes under the retained-population assumptions. |
| [physical_process/nsc_orbital.py](../../src/physical_process/nsc_orbital.py) | Samples axes, eccentricity and historical angular distributions; computes node radii and annulus membership. |
| [workflows/population.py](../../src/workflows/population.py) | Coordinates sampling and selects actual disk-crossing nodes. |
| [workflows/workers.py](../../src/workflows/workers.py) | Serializes the selected population into deterministic damping tasks. |
| [workflows/replenishment.py](../../src/workflows/replenishment.py) | Draws each unused non-crossing NSC donor only when an eligible channel reaches ISCO during the AGN episode. |

## Boundaries

The preserved assumptions include the Pop II subsolar discard, first-order equal stellar/remnant masses and the historical angular sampler. Uniform inclination is not isotropic sampling. A replacement keeps its sampled mass and stellar lifetime but adopts the vacated primary orbit as a phase-space channel approximation. Each donor is drawn once; no new NSC population is generated during the episode.

## Contracts and evidence

- [Current physical model](../physics_model.md)
- [Fixed migration contract](../migration_model_alignment.md)
- [Execution and verification](../engineering_contract.md)
- [Consolidated decisions](../reference/history.md)
- [Preserved pre-cleanup explanation](../archive/project_before_cleanup/02_population_sampling.md)
