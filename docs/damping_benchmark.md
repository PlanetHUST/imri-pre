# Damping benchmark

Run only the first model-upgrade stage (damping physics and comparison):

```bash
PYTHONPATH=src /home/planet/miniconda3/envs/sevnpy_env/bin/python scripts/compare_damping_models.py \
  --output-dir outputs/damping_benchmark --sources 64 --repeats 8 \
  --seed 260529305 --lifetime-myr .1 --bootstrap-draws 2000
```

Append `--resume` to replay completed atomic source checkpoints. Changing source
inputs, physics/presentation code, runtime versions or settings requires a new
output directory. The runner never changes production defaults or existing
workflow CSV/checkpoint contracts.

Primary outputs are `damping_flux_comparison.csv`, `damping_parameter_scan.csv`
and `damping_model_comparison_report.md`. Also inspect `damping_source_outcomes.csv`
for every non-event and censored/unsupported state, `damping_trajectories.csv`,
`damping_joint_radius_time.csv`, `damping_sensitivities.csv`,
`damping_lifetime_boundaries.csv`, `damping_pagn_audit.csv`, the PNG/PDF figures,
and `damping_benchmark.html`. Open the HTML beside its PNG files. Full resolved
inputs and provenance live in `matched_inputs.json` and `manifest.json`.

`physical_process.damping_models` exposes immutable CGS/radian `DampingInput`,
common `DampingOutcome`, `RepoDampingAdapter`, and `XueDampingAdapter`. The latter
provides `inclination_damping_rate`, `effective_crossing_probability`,
`damping_time`, and `validity_mask`. Local scalar results include units and status;
missing quantities are `None`, distinct from evaluated zero. `damping_time` is a
duration; `t_embed_myr` is absolute time since disk turn-on. The Xue duration is
uncensored; `evolve` applies shutdown censoring.

The benchmark intentionally uses a declared circular compact-object fixture,
shared identically between models, rather than altering the production IMF or
inventing a remnant mapping. A list of serialized `DampingInput` records can be
supplied with `--input-json PATH`; those IDs, masses, weights, epochs and lifetimes
are preserved exactly. A separate audit uses the existing solved pAGN Sirko disk.
This audit is a domain check, not a full production NSC population experiment.

The Xue adapter evaluates the base-10 combined fit, with two disk crossings per
orbit, at a fixed reference radius. Its domain is deliberately restricted to the
rounded Rowan reference environment and the overlapping Hill-mass span verified
from Whitehead Fig 5. Density interpolation and transfer of the low-inclination
branch are explicitly flagged assumptions. Arbitrary eccentric, retrograde,
other-host/mass/disk extensions are unavailable. The existing solver evolves
radius; Xue's fixed radius is a closure assumption. Xue total orbital energy and
angular-momentum losses are unavailable, so no full orbital-loss agreement can
be claimed. The geometric occupancy interface is not a scattering probability
and does not further reduce the fitted damping rate.

`direct_plus_effective_scattering` returns unavailable: the cited papers supply
no independently calibrated additive encounter term. Diagnostic relaxation
times in the production code do not imply dynamical two-body relaxation.

Weighted first-event flux is summed event weight divided by bin width, with
paired bootstrap intervals over realization means. Fractions retain all input
weight in their denominator. Unsupported inputs suppress full-population
estimates, while observed lower bounds remain separate. Ratios require complete
coverage, positive flux and Kish effective count at least five in both models.
The intervals describe population sampling, not hydrodynamic calibration error.

Validation:

```bash
PYTHONPATH=src /home/planet/miniconda3/envs/sevnpy_env/bin/python -m pytest
```

This command includes independent native impulse checks, coefficient/sign/cadence
checks, analytic low-inclination damping, timestep convergence, lifetime boundary
and initial-embedding checks, weighted count conservation, invalid/zero
separation, common-random-number sampling and exact checkpoint replay.
