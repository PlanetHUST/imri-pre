# Project overview

Reviewed against the live source on 2026-10-18.

## Question and flow

Given a central mass, NSC mass, metallicity, redshift and AGN episode, sample
one local system and calculate intrinsic IMRI band-entry outcomes plus
conditional LISA detectability.

    host -> pAGN disk -> population/orbits -> actual node crossing
         -> damping -> signed migration -> gas/GW handoff
         -> physical band entry -> intrinsic count -> detector selection
         -> ISCO consumption -> finite NSC donor -> vacated orbit -> damping

A tree batch maps each selected row onto the same local kernel. It does not
supply host-abundance weighting, ancestry evolution or an all-sky cosmic rate.

## Execution

[scripts/main.py](../../scripts/main.py) delegates to
[workflows/cli.py](../../src/workflows/cli.py). Single-halo mode calls the
grid driver; --tree selects the outer batch layer. The disk and torque engine
caches physical resources. Per-object damping can run in an inner process
pool. Persistence happens through outputs.py and metadata.py.

The live main CLI uses optional --base-seed (default None), one fixed migration
contract and positional repeat/output/worker arguments. It saves data without
automatically running the figure API. Earlier proposed defaults are archived.
The single realization also processes replacement donors in ISCO-consumption
time order. The finite source and each injection are recorded separately.

## Model boundaries

The low-mass disk and linear torque applicability remain approximations.
The thermal validity mask, Kanagawa extrapolation flag, WKB jump repair and
physical versus numerical outer edge must be visible in analyses. An absent
resolved trap does not guarantee everywhere inward torque.

Damping alignment follows the newer TDE numerical controller while preserving
IMRI node physics and the stricter migration handoff. The detector supports
circular inspirals and explicitly rejects unsupported cases. Its legacy
finite-time cosmic-merger option remains in the implementation; it is not a
cosmological population model.

## Read next

- [Part map](README.md)
- [Physical model](../physics_model.md)
- [Migration contract and open limits](../migration_model_alignment.md)
- [Execution, units and artifacts](../engineering_contract.md)
- [Decision history and source references](../reference/history.md)
