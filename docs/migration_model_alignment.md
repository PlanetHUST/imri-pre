# Fixed migration contract and validation limits

Reviewed against the current working source on 2026-10-18. Production uses
one signed migration contract, identified by
MIGRATION_CONTRACT = "jm17_grishin2024_continuous_kanagawa" in
[workflows/config.py](../src/workflows/config.py).
The earlier investigation and numerical tables are preserved in
[archive/design/migration_model_alignment.md](archive/design/migration_model_alignment.md).
They are historical measurements rather than current defaults.

## Production equations and owners

With mass ratio q, aspect ratio h=H/r and viscosity alpha:

    K = q^2 / (alpha h^5)
    f_gap = 1 / (1 + 0.04 K)
    B = Sigma r^2 / m
    Gamma_gap = -(3/4) alpha h^2 m r^2 Omega^2 min(1, B)
    Gamma_total = f_gap Gamma_grav + Gamma_thermal + (1-f_gap) Gamma_gap

[gap.py](../src/physical_process/migration/gap.py) owns K, the Kanagawa depth
and Crida deep-gap diagnostics. [torque.py](../src/physical_process/migration/torque.py)
owns torque construction and composition. The inward gap-drift expression is
an explicit interpolation assumption, not a separately selected classical
Type-II model. It follows from dL/dr = (1/2)m*r*Omega and
v_r = -(3/2)alpha*h^2*r*Omega*min(1,B).

Production has no migration-model, gap-criterion or gap-threshold selector.
The disk_model field remains a compatibility label for the two explicitly
separated scenarios: the default `torque_with_trap` production contract and
the comparison-only `semi_without_trap` implementation in
[semi_analytic.py](../src/physical_process/migration/semi_analytic.py).
Crida is diagnostic only in production. Malformed profiles do not fall back to
a second transport prescription.

The comparison scenario is assembled programmatically for controlled tests; it
is not exposed by the CLI and is not a production alternative. It integrates
local Type-I and Type-II timescales over `d ln R`, is always inward, and has no
trap or outward-blocking outcome.

## Thermal treatment

The gravitational coefficient is the local Jimenez-Masset linear treatment.
The thermal term follows the local Grishin-2024-style prescription on the
pAGN solution: Rosseland opacity sets radiative diffusivity, hot/cold terms
have separate saturation factors, min(r_B,H) sets the saturation length, and
1-exp(-sqrt(alpha)*tau_gap) suppresses optically thin gas. Gap depletion enters
tau_gap. The current embedded-object luminosity is Eddington capped.

The paper's analytic disk zones and global population conclusions are not
imported as assumptions for this lower-mass disk model. The profile records
x_c/lambda, lambda/H, r_H/H, r_B/H and the calibration diagnostic K>1e4.
Production excludes the thermal contribution where its linear-regime checks
fail; the raw coefficient and applied validity weight remain separately
available. The diagnostic extrapolation parameter is not exposed by the
production CLI. Excluding an unsupported term is a modeling choice, not a
prediction that the true thermal torque is physically zero.

## Trap and no-trap comparison

[traps.py](../src/physical_process/migration/traps.py) classifies torque zeros
on the full profile. A stable trap has positive torque inside and negative
outside; the opposite ordering is an anti-trap. Basin widths help distinguish
resolved features from narrow numerical structure.

[evolution.py](../src/physical_process/migration/evolution.py) integrates
(dL/dr)/Gamma in CGS. A resolved stable zero on the path gives trap_bound;
blocking outward torque without a resolved stable zero gives outward_blocked.
A non-finite or out-of-domain profile is an error, not a physical trap.

The diagnostic no-trap extreme is Gamma_no_trap = -abs(Gamma_total), provided
by [diagnostics.py](../src/physical_process/migration/diagnostics.py). It holds
local torque magnitudes and all upstream inputs fixed, changing only outward
signs. Exact zero cells are rejected rather than hidden behind an arbitrary
floor. This counterfactual bounds delivery sensitivity; it is not a second
production disk or an assertion that the system must have traps.

## Open physical and numerical limits

- The low-mass central objects and large perturber-to-disk ratios can exceed
  the applicability of the thermal theory and empirical gap fit. Report
  validity occupancy on the actual migrator paths.
- The WKB helper currently flags the two cells adjacent to value jumps above
  0.15 dex and interpolates their fitted log-slopes. Neighboring spline ringing
  can survive. This is not proof of a converged physical torque through the
  ionisation transition.
- The adapter stores the physical Q=1 outer edge, but pAGN arrays include a
  padded tail. The current torque view retains that array. Interpret features
  relative to both the physical edge and numerical grid.
- The current contract validation path uses the second-to-last torque cell as
  its start. That can lie outside the physical edge. Its whole-grid delivery
  time must not be presented as a physical-edge measurement.
- A scan with zero resolved stable traps can still contain positive cells and
  an outward_blocked path. Earlier claims that every total torque is inward,
  or that every apparent trap has been removed, overstate the available check.
- The band-entry/ISCO limit depends on mass, redshift and observed frequency;
  the configurable host mass domain is not a guarantee of detector support.

These are documented limits, not algorithm changes in this cleanup.

## Reproduce the checks

```bash
PYTHONPATH=src python tests/run_tests.py
python scripts/validate_migration_contract.py \
  --output-csv output/diagnostics/migration_contract.csv
python scripts/run_science_checkpoints.py --only D E F
python scripts/analyse_population_torque.py --help
```

The CSV records signed and no-trap times/statuses, trap counts, gap depletion,
thermal validity/applied fractions, WKB flags and disk dominance. Persist
inputs and source identity when using these diagnostics to make scientific
claims. Checkpoints and finite arrays establish implementation consistency;
they do not validate the physical model across the low-mass regime.
