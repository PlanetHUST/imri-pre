# One halo episode: source actions, conditions, and rates

This is the navigation page for the **implemented** single-halo workflow. Its
control flow is in [`IMRISingleWorkflow.run_single_simulation`](../src/workflows/single_work.py).
The entry script and grid runner choose hosts, seeds, repeats, and output paths;
they do not decide an individual source's physical route. One repeat models one
AGN episode for one host. Rates below are conditional per halo per Myr, not a
cosmic population rate.

## Read the control flow in one place

```text
host + seed
  -> build/cached pAGN disk and sample NSC sources
  -> select primary disk crossers; leave other unique IDs in the donor pool
  -> damp primaries in a parallel batch, then restore object-index order
  -> for each primary: stellar handoff if needed -> entry/SNR classification
  -> queue eligible ISCO consumptions by (time, primary channel, object index)
  -> while the queue is nonempty:
       consume its earliest occupied channel
       draw one unused NSC donor, if available
       give that donor the primary channel's initial orbit at injection time
       run the same damping -> stellar handoff -> entry/SNR actions
       queue its later ISCO consumption, if eligible
  -> finalize counts and rates once; persist results and checkpoints
```

The primary batch must finish before donor injection: the first ISCO agenda is
built from all primary events. A donor can be drawn only after its predecessor
has reached ISCO within the AGN episode. `replenishment.ConsumptionCycle`
owns the finite donor pool, stable queue order, and channel accounting; the
`while` loop and source actions are visible in `single_work.py`.

The source action chain is a physical state machine with fixed guards, not an
autonomous planner. `SourceState` holds the current `SourcePhase`, optional
replacement task, damping outcome, and event as it passes through the loop.
`SourceOutcome` returns the final damping state and optional IMRI event.
`DampingResult` and `IMRIEventRecord` retain the existing published fields,
while the new transition ledger exposes their handoffs.

## Actions and guards

| Action in the episode driver | Guard / stop condition | Physical owner and result |
| --- | --- | --- |
| Sample and select | Host, disk domain, and sampled orbital geometry | `population.py` supplies selected primary IDs and the unused NSC reservoir. |
| Damping | Each primary; each injected donor at its absolute injection time | `workers.py` calls `DampingCalculator`. Its result includes embedding, outside-disk, collapse, kick, or failure status. |
| Stellar handoff | SEVN mode **and** damping ended with `EMBEDDED_STAR` | `event_work.resolve_embedded_stellar_event` migrates the star to collapse, applies its SEVN remnant and optional kick, then checks BH embedding or redamping. |
| Entry and detector | Every processed source; only eligible embedded outcomes can produce an event | `event_work` advances the circular gas+GW orbit, `event_rate` decides BH candidacy at first band entry, and `lisa.detectability` evaluates SNR and the threshold. |
| ISCO consumption | Physical IMRI entry with a finite ISCO time inside the duty cycle | `ConsumptionCycle` reopens that occupied channel once and draws one unused donor. Exhaustion is a recorded outcome. |
| Rate finalization | After the donor agenda is empty | `event_work.finalize_rates` applies the existing validity and duty denominator once; `outputs.py` validates persisted physical-entry and detector statuses. |

In SEVN mode, the damping action itself has a time boundary: SEVN provides one
collapse time, remnant type, gravitational remnant mass, and kick draw for each
source ID. If collapse occurs before embedding, `workers.py` stops stellar
damping there, applies the remnant/kick, and may resume BH damping. If the star
embeds first, the later stellar-handoff action migrates it until collapse.
Thus the source driver has one route for primaries and replacements, while the
physical actions can contain the numerical substeps they own.

## Clocks, identity, and terminal outcomes

- Every `*_time_myr` in the source and event ledgers is absolute elapsed Myr
  since this AGN episode began. `injection_time_myr` is zero for a primary and
  the predecessor's ISCO consumption time for a replacement.
- `formation_time_myr` is the legacy source field for the stellar lifetime or
  SEVN collapse time. It is **not** the replacement's injection time. The
  existing `damping_time_myr` event field can include pre-collapse migration
  for a BH formed during migration. Use the transition times and the stellar
  ledger to resolve that history; a future schema revision should give those
  elapsed intervals separate names.
- Gas operates only while the AGN is active. Circular GW continuation can
  reach band entry after AGN shutdown, up to the configured cosmological
  `z=0` boundary. The ISCO donor trigger must occur within the AGN duty cycle.
- `source_id` identifies a unique sampled NSC object. `object_index` identifies
  its evaluated primary or replacement occurrence. `origin_primary_idx` is
  the fixed channel template; `copy_number` counts successive donors there.
- Non-BH remnant, unbound kick, outside-disk, trap/blocked, inside-ISCO,
  unavailable eccentric detector treatment, and numerical failure remain
  distinct outcomes. Unavailable results are not zero-rate observations.

The worker may finish an action with a SEVN collapse/kick already resolved.
The source transition row records that action's **outcome**, and
`imri_stellar_ledger.csv` holds its endpoint/kick details. Transition rows are
high-level action boundaries, not the numerical integration steps inside a
physical solver.

## From events to a rate

An eligible embedded source is evolved with the chosen gas torque and GW loss.
The first observed LISA-band crossing is tested for BH candidacy. The detector
decision then reports evaluated SNR, no frequency overlap, or an explicit
unavailable reason. For a complete repeat, the intrinsic and detectable rates
are respectively `N_physical_entries / duty_cycle_myr` and
`N_detectable_entries / duty_cycle_myr`. The latter count requires a valid SNR
above the configured threshold. Read `rate_status` and
`detector_rate_status` in `imri_repeat_results.csv` before using either rate.
`outputs.py` reconciles first-entry event rows with the in-memory counters;
this validation is intentionally retained during the structural refactor.

## Checkpoint and trace files

Two additive CSV files are written beside the existing outputs:
They are refreshed at the configured grid checkpoint interval and when the
run completes.

| File | Grain and use |
| --- | --- |
| `imri_workflow_checkpoints.csv` | Five cumulative snapshots per repeat: population selected, primary damping complete, primary sources complete, consumption cycle complete, rates finalized. Each row records the host, seed, model controls, duty cycle, counts, and unavailable categories. Rate columns are populated only at finalization. |
| `imri_source_transitions.csv` | One row per source action boundary: damping, optional embedded-star handoff, and entry/detection classification. It records injection, action, collapse, band-entry, and ISCO clocks separately. Join `event_id` to `imri_events.csv` for SNR and orbit details, or `source_id`/`object_index` to the stellar and donor ledgers. |

The existing event, repeat, averaged, stellar, donor, and migration diagnostic
schemas remain unchanged. Run the persisted-artifact check after a simulation:

```bash
python scripts/check_single_halo_workflow.py output/dyn_results
```

It checks the stage sequence, primary/donor counts, source uniqueness, ISCO
time order, ledger totals, and the final rate/status agreement. It reads saved
files and does not consume random numbers or rerun the model.

## Adding a mechanism

State its input units, physical stop condition, returned status, and which
clock it advances. Add its guard and action call in the source chain or the
episode consumption loop. Record the new boundary in the source trace and,
when it changes a population count, in a workflow checkpoint. Keep the
mechanism's equations in its domain module. Compare fixed-seed source IDs,
transitions, event fields, donor order, SNR, and rates against a saved control
before changing the physical prescription.
