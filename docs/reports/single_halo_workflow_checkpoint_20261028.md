# Single-halo workflow refactor checkpoint

The 2026-10-28 structural change puts source actions, the ISCO donor loop,
and one end-of-episode rate finalization in `src/workflows/single_work.py`.
`replenishment.ConsumptionCycle` retains the finite donor pool and stable
chronological priority. The equations, model controls, seed policy, existing
scientific CSV schemas, and output precision were not changed by this part.

## Deterministic comparison

Host: `M_SMBH=1e5 Msun`, `M_NSC=2e5 Msun`, `z=15`, `Z=1e-5 Zsun`, 10 Myr
AGN episode, base seed `12345`, four damping workers for the 20-repeat
comparison. The SEVN H/He tables and optional `adv_damp` source are those
recorded in the [stellar integration baseline](sevn_remnant_kick_baseline_20261027.md).

| Checkpoint | Result |
| --- | --- |
| One seeded fixed-mass repeat, before versus after | Existing averaged, event, repeat, and donor CSVs byte-identical; migration diagnostic JSON byte-identical. |
| One seeded SEVN kick repeat, before versus after | The same files plus stellar ledger byte-identical; migration diagnostic JSON byte-identical. |
| 20 SEVN no-kick repeats against saved 2026-10-27 run | Averaged, event, repeat, donor, and stellar CSVs and all 20 migration diagnostic JSON files byte-identical. |
| 20 SEVN kick repeats against saved 2026-10-27 run | Averaged, event, repeat, donor, and stellar CSVs and all 20 migration diagnostic JSON files byte-identical. |
| New saved workflow checkpoint checker | Passed all 20 repeats in each SEVN mode. |
| Full Python test suite and source hygiene | `pytest -q`, compilation, `git diff --check`, and local links in six updated documentation pages passed. |

The new `imri_workflow_checkpoints.csv` and
`imri_source_transitions.csv` files are additive diagnostics. Run
`python scripts/check_single_halo_workflow.py <output_dir>` to check their
stage, source, donor, and rate consistency. See the
[main workflow](../main_workflow.md) for their grain and clock meanings.

The comparison does not validate a new physical prescription. It verifies
that the structural change preserved these controlled runs. The remaining
scientific limits of circular post-kick evolution and SEVN pre-collapse mass
are stated in the [stellar integration plan](../plans/active/sevn_remnant_kick_integration.md).
