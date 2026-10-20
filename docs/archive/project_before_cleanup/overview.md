# IMRI Rate Computation in High-Redshift AGN Disks — Project Overview

This document is the macroscopic reconstruction of the project main idea. It is
written from the durable contracts in `docs/reference/`, active specifications,
and the live source in `src/` and `scripts/`. It describes what the project is,
why the pipeline is shaped the way it is, and how the part documents
(`docs/project/0N_*.md`) fit together.

---

## 1. One-sentence statement

> Compute the intrinsic rate at which stellar-mass black holes are delivered,
> by gas migration inside the AGN disk of a high-redshift central black hole,
> into the LISA band, and turn that local yield into a scientific measurement of
> how the rate depends on SMBH mass, NSC mass, redshift, and the gas-migration
> prescription.

The broader, long-range goal (recorded in
[`../reference/research_roadmap.md`](../reference/research_roadmap.md)) is a
cosmological IMRI population model built on EPS merger trees. This repository
currently implements and validates the **local response model** — one SMBH, one
NSC, one AGN episode — that the cosmological layer is designed to query later.

---

## 2. What "IMRI" means in this pipeline

**Working definition:** an IMRI (intermediate-mass-ratio inspiral) is a compact
binary with a very unequal mass ratio inspiralling toward a massive black hole.

In the *local workflow implemented here*, the pair is:

- **primary:** the central SMBH, `M_smbh ~ 1e3–1e6 Msun`;
- **secondary:** a stellar-mass black hole, `m ~ 10–200 Msun`, initially a star
  in the nuclear star cluster (NSC) that surrounds the SMBH;
- the channel is `agn_disk_sbh_to_central_bh` (see the channel list in
  [`../reference/research_roadmap.md`](../reference/research_roadmap.md)).

The secondary is not "intermediate-mass" itself; the ratio is intermediate
because the primary is a *low-mass* SMBH / high-mass IMBH. Cluster-delivered
IMBH–SBH channels are explicitly future work and must not be folded into this
local result.

The scientific quantity computed is the **intrinsic IMRI candidate rate per AGN
episode**:

```
R_IMRI = N_IMRI / T_AGN     [Myr^-1]
```

where `N_IMRI` is the number of embedded black holes that the gas migration
delivers into the LISA band (or to a gas-to-GW handoff that then reaches it)
within one duty-cycle of AGN activity, and `T_AGN` is the duty cycle
(`--duty-cycle-myr`, default 10 Myr). This is a *per-host, per-episode yield*,
not a volumetric cosmological rate and not a detector rate.

---

## 3. The scientific question, decomposed

The pipeline answers, in order, four nested questions:

1. **Supply** — How many stellar-mass remnants of the NSC cross and become
   embedded in the AGN disk? (geometric + damping selection)
2. **Delivery** — How far inward does gas-assisted migration carry each embedded
   object within the duty cycle, before duty-cycle exhaustion, a migration trap,
   or GW dominance stops it?
3. **Band entry** — Does the object reach the formal circular LISA-band entry
   radius (defined by a lower observed GW frequency), either by gas directly or
   by gas to a handoff plus Peters GW inspiral?
4. **Conditional detectability** — Given redshift evolution and a simplified
   cosmic merger, what mission-limited, sky/orientation-averaged circular-
   inspiral LISA SNR does the resulting candidate produce?

Every stage is a *conditional efficiency*, so a low final rate can be traced to
the responsible physical layer (supply, damping, trapping, boundary censoring,
or detector selection).

---

## 4. End-to-end pipeline

```text
 configuration (CLI / config / redshift scan)
        │
        ▼
 grid driver (workflow.py) ──► per (SMBH, NSC, repeat): single_work.py
        │                                │
        │   environment (engine.py)      │
        │     pAGN disk (Sirko/Thompson) │
        │     Q=1 outer radius           │
        │     AGN properties             │
        │     torque profiles (cached)   │
        │                                ▼
        │   population (population.py + sampling.py + nsc_orbital.py)
        │     IMF + stellar lifetime ─► masses, formation times
        │     NSC orbital sampling ──► a, e, i, cos(ω)
        │     node-based disk-crossing selection
        │                                ▼
        │   damping (workers.py ──► damp.py)
        │     active-node gas drag on a, e, i
        │     embedded-state contract
        │                                ▼
        │   event classification (event_work.py + event_rate.py + detect_rate.py)
        │     band boundary routing (in-domain / handoff / inside-ISCO / error)
        │     gas migration to target (torque or semi-analytic)
        │     gas/GW transport comparison
        │     conditional physical LISA SNR
        │                                ▼
        │   repeat result + event ledger ──► migration diagnostics (JSON)
        ▼
 persistence (outputs.py): averaged CSV, event CSV, repeat CSV, metadata JSON
        ▼
 analysis (rate_plots.py + final_analysis.py + redshift_scan.py)
     rate figures, status fractions, paired scenario comparison
```

Dependencies always point downward:

```text
scripts / CLI
   → workflow orchestration
       → population / orbital stage
       → disk resource stage
       → damping stage
       → migration stage
       → event-rate stage
       → redshift / cosmic-merge / detectability stage
   → reporting and persistence
domain stages → shared physical constants + typed records
```

Physics modules never import workflow, output, CLI, plotting, or worker code;
plots never recompute production formulas; output code never recomputes physics.

---

## 5. The five physical layers

| Part document | Layer | Code owner | Question it answers |
|---|---|---|---|
| `01_environment_build.md` | AGN disk + NSC + torque environment | `engine.py`, `pagn_disk_adapter.py`, `migration_torque.py`, `agn.py`, `nsc_orbital.py` (mass–radius) | What disk/NSC does the object live in? |
| `02_population_sampling.md` | Stellar population + orbits | `sampling.py`, `nsc_orbital.py`, `population.py` | What objects exist and which can touch the disk? |
| `03_disk_embedding_dynamics.md` | Damping / orbit evolution | `damp.py`, `workers.py` | Which orbits become embedded, and to what final state? |
| `04_gas_migration.md` | Gas migration | `disk_migration.py`, `event_work.py` (outcome callbacks) | How far can gas deliver an embedded object? |
| `05_gw_event_classification.md` | GW + transport classification | `gw_frequency_radius.py`, `event_rate.py`, `event_work.py` | Is an IMRI candidate formed? |
| `06_cosmology_detectability.md` | Cosmology + detectability | `redshift_evolve.py`, `cosmic_merge.py`, `detect_rate.py` | Is the candidate "detectable"? |

The remaining documents cover the non-physical layers that hold the physics
together:

| Part document | Layer |
|---|---|
| `07_workflow_orchestration.md` | configuration, grid driver, workers, seeding, checkpointing |
| `08_data_io.md` | CSV/JSON persistence, unit handling, metadata, strict JSON |
| `09_analysis_visualization.md` | rate figures, status taxonomy, paired comparison |
| `10_validation_diagnostics.md` | deterministic harnesses, counter invariants, verification |

---

## 6. The two migration scenarios (a controlled comparison)

The single most important scientific axis of the current project is the
**paired migration-prescription bracket** (`--disk-model`):

- `torque_with_trap` (default): detailed signed pAGN torque profile per migrator
  mass; Type-I + thermal in gap-closed cells, Type-II in gap-open cells;
  stable traps/anti-traps can physically stop inward delivery (`trap_bound`).
- `semi_without_trap`: integrated local semi-analytic Type-I/Type-II timescale
  over `d ln R`; always inward and positive, never trap-bound.

Both scenarios share the *same* pAGN disk structure for boundaries, disk
crossing, damping, `Σ(R)`, `H/R`, and the gap-opening decision
([`../reference/physical_model.md`](../reference/physical_model.md)). The comparison is therefore a bracket
on the **migration prescription**, not a full high-redshift disk uncertainty.

This is why paired tasks deliberately use the same seed, with redshift and disk
model **excluded** from the seed derivation:

```
task_seed = SeedSequence(base_seed, smbh_index, nsc_index, repeat_index)
```

Matching tasks across redshift and scenario draw identical sampled populations,
so any rate difference is attributable to the selected prescription (and
redshift), not Monte Carlo noise.

---

## 7. High-mass boundaries must never look like zeros

A recurring theme across the durable contracts is that **censoring is not
astrophysics**. The formal lower LISA-band radius (`--lisa-min-gw-frequency-hz`,
default 1e-3 Hz, observed frame) is compared, in cm, against the migration
profile's lower domain and against the nonspinning Schwarzschild ISCO. This
produces four mutually exclusive band-entry statuses:

| Status | Meaning | Route |
|---|---|---|
| `entry_in_migration_domain` | entry inside the migration profile | normal gas migration |
| `entry_below_migration_domain_but_above_isco` | entry valid but below the profile domain | gas to handoff, then Peters GW |
| `band_entry_inside_isco` | entry at/below ISCO | no torque call; explicit unsupported status |
| `profile_error` | profile/model failure | real error, recorded separately |

For the default configuration this matters at `M_smbh = 1e5 Msun` (handoff, the
formal entry sits ~0.3 rg below the torque grid) and `M_smbh = 1e6 Msun`
(entry inside ISCO). Neither may be counted as a physical zero rate, and an
inside-ISCO object must never reach the torque integrator.

---

## 8. Statuses form a strict hierarchy

The workflow stores several distinct concepts that must never be pooled
([`../reference/physical_model.md`](../reference/physical_model.md)):

```text
band-entry geometry
   → gas/GW transport outcome
   → IMRI event classification
   → detectability classification
```

Each plotted fraction has its own explicit denominator; `not evaluated` and
`unavailable` are never folded into zero/failure. This discipline is enforced by
counter-conservation invariants (see `10_validation_diagnostics.md`) and by the
visualization taxonomy in `09_analysis_visualization.md`.

---

## 9. What the current result does and does not establish

Established (from the historical corrected-frequency audit summarized in
[`../reference/history.md`](../reference/history.md)):

- For the supported bins (`1e3`, `1e4 Msun`), migration traps are the dominant
  suppressor of candidate embedded BHs, and the yield is highly stochastic.
- The result is a local, per-host, per-episode source yield.

Not established, and explicitly out of scope for the current code:

- A cosmological volumetric or all-sky LISA detection rate — requires host
  weighting and merger-tree carrier histories beyond the conditional SNR.
- A complete relativistic IMRI selection function — the current detector model
  is circular, nonspinning, inspiral-only, and sky/orientation averaged.
- The cluster/IMBH channels — those are separate labels and future kernels.

---

## 10. The road beyond the local model

[`../reference/research_roadmap.md`](../reference/research_roadmap.md) defines
the planned experiment ladder:

| Model | Added physics |
|---|---|
| M0 | correct boundaries, fixed host, paired seeds (≈ current local model) |
| M1 | KII vs Crida gap criteria |
| M2 | remnant formation/retention |
| M3 | improved waveform/detector selection |
| M4 | redshift-dependent host weighting |
| M5 | merger-tree carrier histories (galaxy/cluster transport) |
| M6 | cluster / dual-SMBH nuclear kernels |

The current repository implements M0/M1-style local paired comparisons and a
first-order physical detector model. The local response surface is meant to be
queried or interpolated by a future tree layer, not recomputed at every tree
timestep.

---

## 11. How to use the part documents

Each part document follows a fixed structure:

- **Idea** — the scientific concept the part owns, in physics terms;
- **Doing logic** — how the live code implements it, stage by stage, with the
  authoritative function/module names and the equations;
- **Caveats** — known limitations, boundary rules, and recurring pitfalls.

Read `README.md` in this directory for the index and the module→part map, then
read the specific part relevant to the change you are making. When a current
specification conflicts with this reconstruction, the specification governs;
these documents explain the project rather than define the current task.
