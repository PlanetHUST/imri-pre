# How one host run works

Entry point: [`scripts/main.py`](../scripts/main.py). [`workflows.workflow.run_simulation`](../src/workflows/workflow.py) loops over the requested mass grid and Monte Carlo repeats; [`IMRISingleWorkflow.run_single_simulation`](../src/workflows/single_work.py) processes one host episode. Every repeat has its own derived seed. A run uses an explicit AGN duty cycle; its rates are measured within the simulated system, not a cosmic rate.

```text
host + seed -> disk and selected stellar/BH population -> disk-crossing sources
            -> numerical damping of orbital equations -> optional SEVN backend
            -> inward Type-I timescale migration with simultaneous gas and GW evolution
            -> first LISA-band entry -> physical IMRI and detector classification
            -> eligible ISCO consumption -> unused NSC donor in that channel
            -> repeat donor actions until exhausted -> counts and rates
```

A donor can enter only after a prior source reaches ISCO within the AGN duty cycle. The same damping and event-classification path handles primary and replacement sources; the NSC source pool is finite. `--no-consumption-replenishment` limits the calculation to the initial cohort. The opt-in SEVN mode needs external tables and an installed backend; the default CLI is restored to `legacy_popiii`: your original analytic IMF, formation clocks and radial power-law sampler. Published stellar/BH component presets are opt-in only. See [population prescriptions](published_populations.md) for mass normalization, radial distributions and approximation boundaries.

For a complete repeat, the intrinsic rate is the number of physical IMRI band entries divided by the AGN duty time in Myr. The detectable rate counts only eligible entries with evaluated SNR above the threshold. Do not treat unavailable physics or detector evaluations as zero detections: read `rate_status` and `detector_rate_status` in `imri_repeat_results.csv`.

Each run directory under `outputs/` contains:

- `simulation_metadata.json`: configuration and provenance;
- `imri_averaged_results.csv` and `imri_repeat_results.csv`: aggregate and per-repeat results;
- `imri_events.csv`: object-level classification and detection status;
- `imri_workflow_checkpoints.csv` and `imri_source_transitions.csv`: stage counts and action boundaries;
- `imri_replenishment_ledger.csv`: ISCO/donor accounting;
- `intermediate/`: per-repeat migration diagnostics; an optional stellar ledger is written in SEVN mode.

Run `python scripts/check.py --run-dir outputs/pan2021_small_run` after the [quickstart](quickstart.md) example to check saved stage order, source uniqueness, donor counts and rate bookkeeping without rerunning the simulation. Equations and limits are in the [physical model](physics_model.md).
