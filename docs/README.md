# IMRI Documentation Map

This directory separates **current authority**, **durable reference**, and
**generated explanation**. Start here instead of searching root-level task
prompts.

## Authority order

When documents disagree, use this order:

1. the user's current request and repository instructions;
2. an implemented specification in [`specs/`](specs/);
3. the matching plan in [`plans/`](plans/), for execution status and checks;
4. current source code and persisted schemas;
5. durable contracts in [`reference/`](reference/);
6. the generated project explanation in [`project/`](project/).

The `project/` documents explain the repository but do not override a current
specification. Historical notes explain why a decision exists but never define
current behavior by themselves.

## Directory structure

```text
docs/
├── README.md                         navigation and document authority
├── plans/
│   └── active/                       current/recent execution plans
├── specs/                            authoritative implemented specifications
├── reference/                        compact long-term contracts
│   ├── physical_model.md             physics, scenarios, boundaries, statuses
│   ├── engineering_contract.md       ownership, units, I/O, verification
│   ├── research_roadmap.md           cosmological extension and model ladder
│   └── history.md                    compressed decision and migration history
└── project/                          AI-generated source-oriented explanation
    ├── README.md                     part index
    ├── overview.md                   end-to-end conceptual overview
    └── 01_...10_*.md                 detailed explanation by source layer
```

## Read by task

| Need | Read first | Then |
|---|---|---|
| Implement or review LISA SNR | [`specs/lisa_detectability.md`](specs/lisa_detectability.md) | [`plans/active/lisa_detectability_plan.md`](plans/active/lisa_detectability_plan.md), [`project/06_cosmology_detectability.md`](project/06_cosmology_detectability.md) |
| Change damping or disk embedding | [`reference/physical_model.md`](reference/physical_model.md) | [`project/03_disk_embedding_dynamics.md`](project/03_disk_embedding_dynamics.md) |
| Change migration scenarios or traps | [`reference/physical_model.md`](reference/physical_model.md) | [`project/04_gas_migration.md`](project/04_gas_migration.md) |
| Change GW entry or high-mass routing | [`reference/physical_model.md`](reference/physical_model.md) | [`project/05_gw_event_classification.md`](project/05_gw_event_classification.md) |
| Refactor modules, units, or outputs | [`reference/engineering_contract.md`](reference/engineering_contract.md) | the affected `project/0N_*.md` file |
| Change diagnostics or figures | [`reference/engineering_contract.md`](reference/engineering_contract.md) | [`project/09_analysis_visualization.md`](project/09_analysis_visualization.md), [`project/10_validation_diagnostics.md`](project/10_validation_diagnostics.md) |
| Design merger-tree/cosmological work | [`reference/research_roadmap.md`](reference/research_roadmap.md) | [`project/overview.md`](project/overview.md) |
| Understand why an interface exists | [`reference/history.md`](reference/history.md) | current source and the relevant specification |

## Document lifecycle

- Put an exact, reviewable scientific or interface contract in `specs/`.
- Put ordered implementation milestones and validation gates in `plans/`.
- Update `reference/` only for decisions expected to remain useful across
  tasks. Keep it compact and source-oriented.
- Regenerate or revise `project/` when the architecture changes materially;
  label it as explanatory rather than governing.
- Do not create new root-level prompt, log, or memory files. Task conversation
  history belongs outside the repository; durable conclusions belong in one of
  the four reference documents.
- A completed plan may remain under `plans/active/` while it is the current
  implementation record. Move it to a future `plans/completed/` only when a
  newer plan supersedes it and all links are updated.

## Consolidation note

On 2026-09-02, legacy root prompts, one-off repair harness documents, the old
`previous_part/` log/memory/prompt set, and long-form future-work notes were
consolidated into `reference/`. The consolidation changes documentation
ownership only; it does not change source code, physics, output schemas, or
validation results.
