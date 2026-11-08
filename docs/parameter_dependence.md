# Initial-parameter IMRI sensitivity experiment (archived legacy model)

This records the earlier analytic-IMF benchmark. Its script explicitly selects `legacy_popiii`; the revised CLI default and published radial prescriptions are documented in [published populations](published_populations.md).

## Audit of the existing simulation

The existing workflow could **partially**, but not completely, test the screenshot's inputs. A mass-grid run alone is not a controlled ten-parameter sensitivity experiment.

| Physical input | Previously available? | Interpretation / limitation |
|---|---|---|
| Disk viscosity alpha | No | Workflow hard-coded 0.1. |
| SMBH mass M | Yes | CLI mass grid; changing M also changed Mdot and the Q=1 edge. |
| Accretion rate Mdot | No | Hard-coded 1.44e18 (M/Msun) g/s. |
| Outer disk radius r_out | No, derived | First Q=1 radius; no independent boundary experiment. |
| NSC density slope gamma_n | Yes | Initial orbital distribution, default 1.75. |
| IMF slope gamma_m | Yes, Pop III only | `popiii_imf_slope`, default 1.77. Not a remnant-mass-function slope. |
| Maximum mass m_max | Yes, Pop III only | Default 197 Msun; sampled stellar/fixed secondary mass, not a SEVN remnant cutoff. |
| Metallicity Z | Partly | Switches Pop III/II at 1e-4 Zsun; no continuous Z dependence within either branch in the default backend. |
| NSC radius R_NSC | Programmatic | Empirical mass-radius normalization, not previously exposed by CLI. |
| NSC mass M_NSC | Yes | Also changed R_NSC via the empirical mass-radius relation. |

Thus the answer is not simply “the simulator cannot do sensitivity tests”: the physics already accepts several inputs, but independent controls, experimental design, and comprehensive output were missing. Multi-prescription comparisons were programmatically possible but not a controlled initial-parameter scan.

## Added controls

`IMRIDynConfig`, the engine, metadata, and the CLI now support:

- `--disk-alpha` (default 0.1)
- `--disk-accretion-rate-g-s` (default unset: original Mdot proportional to M)
- `--disk-outer-radius-cm` (default unset: original Q=1 boundary)
- `--nsc-radius-pc-at-reference-mass` (default 3.31 pc at 3.6e6 Msun)
- `--nsc-radius-mass-slope` (default 0.321)

The Sirko solver actually resolves the disk on the explicit radial domain; this is not a post-hoc multiplication of rates or a selection-mask-only change. Normal simulation defaults are unchanged. Disk-profile figures also read these controls from run metadata.

**Caution:** allowing r_out above the reference Q=1 radius can include self-gravitating gas. This is an explicit phenomenological boundary sensitivity, not a claim that such an extended disk is stable. Different alpha/Mdot values can also move Q=1 inside the fixed outer edge. Disk metallicity-dependent opacities and continuous remnant-metallicity evolution have not been introduced.

## Experiment

```bash
python scripts/test_parameter_dependence.py --repeats 3 --workers 1
# Optional identical initial-condition design for the other migration prescriptions:
python scripts/test_parameter_dependence.py --models semi_type_i torque_with_trap semi_without_trap --output-dir outputs/parameter_models
# Replot without rerunning physics:
python scripts/test_parameter_dependence.py --plot-only
```

Default experiment reference:

- M=1e5 Msun, one explicitly selected member of the existing CLI default grid (1e3, 1e4, 1e5, 1e6), **not an average over that grid**;
- M_NSC=2e5 Msun, AGN duration 10 Myr, redshift 15;
- other physical inputs retain current CLI defaults, including replenishment;
- calibrate r_out once from the unmodified reference Q=1 disk, then hold its dimensional value fixed;
- calibrate R_NSC from the reference empirical relation, then hold its physical value fixed, compensating its normalization when M_NSC changes;
- hold absolute Mdot fixed when varying M, rather than inadvertently holding an Eddington ratio fixed;
- ten one-at-a-time scans at factors **0.5, 1, 1.5** (21 distinct points), plus four off-axis points completing an actual measured alpha–Mdot 3x3 phase-space slice: **25 configurations per model**;
- reuse base seed 20261107 across points/models; derive the usual independent repeat seeds. Population-size changes prevent exact object-by-object pairing, so this reduces rather than eliminates Monte Carlo noise.

“50% variance” is interpreted as a **±50% parameter range**, not a statistical variance of 0.5 or a Gaussian prior. The two-dimensional slice is not a ten-dimensional interaction study. Only alpha–Mdot interactions are sampled; all other sensitivities are conditional on the reference inputs.

## Outputs and interpretation

Under `outputs/parameter_dependence/`:

- `experiment.json`: physical reference values/units, seeds, design, caveats;
- `parameter_dependence.csv`: all actual input values and factors, intrinsic/detected rates, evaluation statuses, intrinsic Monte Carlo standard errors;
- `imri_parameter_phase_space.png` and `.pdf`: ten-input OAT rate heatmap and measured alpha–Mdot phase-space slice; no interpolated surrogate rates;
- `<model>/point_*/`: standard simulation outputs, per-repeat/event tables, metadata, and log for every configuration.

The plotted quantity is intrinsic first entry into the configured **observed** GW band, divided by source-frame AGN lifetime, per host, in Myr^-1. It is not a merger count, observer-time rate, cosmological volumetric rate, or LISA detection rate. Detector-selected rates are retained separately in the table. Unevaluated rates remain NaN / N/A, never converted to zero.

For the default Z=1e-5 Zsun, the entire ±50% interval stays below the Pop III/II threshold. **Exactly identical paired-seed outputs for this Z scan are expected by construction**. A flat metallicity row demonstrates a missing continuous metallicity response in this prescription, not astrophysical independence of IMRI rates from metallicity. Testing that physics needs a metallicity-dependent stellar/remnant prescription (e.g. a separately validated SEVN setup).

Three repeats are exploratory. The CSV standard error estimates repeat-to-repeat sampling uncertainty; it is not a physical-systematics error bar, and a zero-event sample does not imply a zero upper confidence bound. Increase `--repeats` before interpreting differences as statistically resolved.

## Completed default-model run

The 25-point, three-repeat `semi_type_i` experiment completed with **all intrinsic and detector evaluation statuses complete** (75 realizations). Rates below are per host per source-frame Myr; ± denotes repeat Monte Carlo SEM, not a confidence interval.

| Input varied | -50% | Reference | +50% |
|---|---:|---:|---:|
| alpha | 0.167 ± 0.067 | 0.200 ± 0.115 | 0.200 ± 0.058 |
| M | 0.300 ± 0.058 | 0.200 ± 0.115 | 0.100 ± 0.058 |
| Mdot | 0.167 ± 0.120 | 0.200 ± 0.115 | 0.233 ± 0.067 |
| r_out | 0.100 ± 0.058 | 0.200 ± 0.115 | 0.467 ± 0.176 |
| gamma_n | 0.033 ± 0.033 | 0.200 ± 0.115 | 3.867 ± 0.410 |
| gamma_m | 0.367 ± 0.203 | 0.200 ± 0.115 | 0.200 ± 0.100 |
| m_max | 0.267 ± 0.033 | 0.200 ± 0.115 | 0.267 ± 0.120 |
| Z | 0.200 ± 0.115 | 0.200 ± 0.115 | 0.200 ± 0.115 |
| R_NSC | 0.700 ± 0.265 | 0.200 ± 0.115 | 0.100 ± 0.100 |
| M_NSC | 0.067 ± 0.067 | 0.200 ± 0.115 | 0.433 ± 0.088 |

The strongest response in this exploratory scan is the NSC density slope: a steeper profile places many more objects near the small disk. Compactness and NSC mass also affect supply. Most disk/IMF differences are small relative to the sampling errors, so these runs do **not** establish precise trends or interaction coefficients. Metallicity is exactly degenerate, as predicted by the audit. All sampled detector-selected rates are zero at z=15 and SNR threshold 20; that is a finite-sample result, not a universal non-detectability statement. The optional migration prescriptions have not been run in this default-model result.

The original heatmap figures were removed on request; their simulation tables and logs are retained. The replacement presentation is the [three-panel high-redshift rate comparison](high_redshift_rates.md). The historical heatmap can still be regenerated with `--plot-only`.

## Automated regression coverage

`tests/core/test_parameter_dependence.py` checks propagation to solver and metadata, explicit-radius solving instead of Q-boundary selection, invalid inputs, design completeness, independent NSC radius/mass controls, and exact metallicity degeneracy of the default sampler.
