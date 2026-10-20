# Compact Project History and Superseded Interfaces

This is a high-density decision history distilled from the former
`previous_part/` prompt, memory, and chronological Codex log. It helps explain
current interfaces; current source, specifications, and plans remain authoritative.

## 1. Evolution by milestone

| Period | Durable result |
|---|---|
| 2026-07-25 to 07-26 | Established pAGN/legacy disk comparisons, mass-dependent SI torque profiles, Type-II ingredients, and signed radius-to-radius torque migration. |
| 2026-07-27 | Added the first simplified IMRI demo, Peters/LISA helpers, event CSV, and averaged results. |
| 2026-07-28 | Removed the TDE-facing workflow and renamed the public workflow/result surface around IMRI. |
| 2026-07-31 | Split the one-realization workflow from resource caching, split physical helpers by domain, made runtime variables unit-explicit, added redshift/time bookkeeping, and included delayed star-to-BH candidates. |
| 2026-08-01 | Separated intrinsic event classification from detector classification, corrected the torque-time integrand to `(dL/dR)/Gamma`, moved readable unit scaling to output time, and made the pAGN `Q=1` radius the physical disk edge. |
| 2026-08-02 | Added gas-versus-GW stopping, simplified cosmic-merger/redshift interfaces, and corrected the strict ordering of the gas/GW decision. |
| 2026-08-08 | Added integrated migration comparison and stable/anti-trap diagnostics; filtered unresolved adjacent opposite-type roots by grid separation. |
| 2026-08-09 to 08-10 | Corrected the pAGN Type-I/thermal prescription against the upstream notebook, fixed the SI/CGS boundary, kept plot smoothing diagnostic-only, and distinguished physical `Q=1` from extended numerical plot grids. |
| 2026-08-11 | Centralized KII/Crida gap selection, made Type-II replace rather than add to the selected branch, made production torque-only at that time, introduced typed migration outcomes, and reserved infinity for physical blockage. |
| 2026-08-12 | Reconstructed the project into domain owners, typed damping/results, explicit output schemas, short orchestration stages, and behavior-preserving validation. |
| 2026-08-17 to 08-19 | Diagnosed disk-crossing versus migration-domain mismatch and implemented four-way band-entry preflight, gas-to-GW handoff, inside-ISCO unsupported routing, strict counters, and retained status records. |
| 2026-08-29 to 08-31 | Added paired `torque_with_trap` versus `semi_without_trap` scenarios, deterministic cross-scenario seeds, scan-level scenario identity, publication-oriented intrinsic-rate figures, and four-layer status visualizations. |
| 2026-09-02 | Replaced the threshold-valued detector heuristic with an implemented mission-limited, sky/orientation-averaged circular-inspiral LISA SNR while retaining intrinsic/detector separation. |

## 2. Historical corrected-frequency audit

The 2026-08-18 pre-boundary-repair run used redshift 15, ten repeats, a
`10 Myr` duty cycle, KII with Type II enabled, NSC mass `2e5 Msun`, and SMBH
masses `1e3`, `1e4`, `1e5`, and `1e6 Msun`. It did not persist seeds, so it is
not a paired baseline.

| SMBH mass | Pre-repair event interpretation | Finite/trap/error |
|---:|---|---:|
| `1e3 Msun` | 12 candidates; local mean `0.12/Myr` | 137 / 43 / 0 |
| `1e4 Msun` | 2 candidates; local mean `0.02/Myr` | 10 / 177 / 0 |
| `1e5 Msun` | entry about `0.30 rg` below the torque minimum; not a valid zero | 0 / 0 / 121 |
| `1e6 Msun` | 1 mHz circular entry inside ISCO; not a valid zero | 0 / 0 / 214 |

Across supported bins, 48 of 62 embedded-BH candidates were trap-bound; trap
fractions of evaluated paths were 23.9% (`1e3 Msun`) and 94.7%
(`1e4 Msun`). The 14 old detector values were threshold-valued proxies and are
not evidence about the current physical LISA SNR. Boundary routing and detector
modeling have since changed, so re-run current code before quoting any result.

## 3. Corrections that must not regress

| Old behavior or ambiguity | Current durable decision |
|---|---|
| TDE and IMRI shared one public workflow surface | The repository is IMRI-centered; removed TDE APIs are not compatibility requirements. |
| One cached reference-mass torque was rescaled for all migrators | Torque profiles are mass dependent and cached with physical identity. |
| Torque time integrated `L/Gamma` over radius | Use `(dL/dR)/Gamma`, with `dL/dR = 0.5 m R Omega`. |
| SI and CGS torque arrays were mixed or generically named | Construct/compose in SI and convert once at the migration consumer. |
| Type-II torque was added on top of the closed branch | Type-II replaces the complete selected branch in gap-open cells. |
| Profile failure could silently fall back to another migration method | Profile/domain/numerical failures remain explicit; scenario selection is configured, never an error fallback. |
| A torque sign crossing and an invalid grid both produced infinity | Infinity is physical `trap_bound`; invalid inputs raise/record errors. |
| Disk crossing used interval overlap or sampled outside nodes at disk edges | Crossing is actual per-node membership; only active nodes interact with gas. |
| Low inclination alone implied successful embedding | Final inclination, `e <= 0.01`, and radial containment are all required. |
| The formal LISA entry below a torque grid produced a domain error | Valid entry above ISCO uses a gas-to-GW handoff at the migration-domain boundary. |
| Entry inside ISCO looked like a zero or profile error | It is retained as unsupported/not evaluated and skips migration/detector calls. |
| Event frequencies were stored in presentation units | In-memory frequencies are canonical Hz; output code scales and renames. |
| Output schemas depended on model annotation order | CSV columns are explicit compatibility contracts. |
| `NaN`/`Infinity` appeared in JSON | Strict JSON uses `null` with an explicit status. |
| Scenario runs could differ because of random draws | Task seeds exclude redshift and disk model so comparable tasks are paired. |
| Raw status labels were renamed for figures | Presentation categories are deterministic mappings; raw persisted fields remain reproducible. |
| Threshold membership set `snr_proxy` directly to the threshold | Physical `lisa_snr` is authoritative; `snr_proxy` is a deprecated compatibility alias. |

## 4. Superseded names and assumptions

Do not design new work around these historical surfaces:

- `run_imri_demo()` and `--imri-demo` as the primary workflow;
- removed `TDEDyn*` configuration/results and TDE CSVs;
- monolithic `helper_function.py` as the physical owner;
- misspelled `disk_plot_utilts.py`;
- `r_imri_*`, `start_radius_*`, threshold-valued `snr_proxy`, or annotation-
  derived event schemas;
- an implicit semi-analytic fallback after torque/profile failure;
- source-mass helpers that apply redshift twice;
- a fixed disk endpoint used in place of the pAGN physical `Q=1` boundary;
- unpaired historical runs as evidence for a scenario-rate difference.

Compatibility fields or wrappers may remain in source, but new code should use
the current typed, unit-explicit, model-neutral interfaces.

## 5. Historical evidence limits

Recorded smoke values, candidate totals, trap counts, and plots belonged to
specific inputs, dependencies, resolutions, and random seeds. They are evidence
that a past change was exercised, not evergreen numerical baselines. Re-run the
current diagnostic before quoting a number.

The most reusable historical lessons are methodological:

- inspect the literal current contract and live schema before scripting checks;
- preserve unrelated dirty worktree changes;
- expose diagnostic science inputs instead of relying on hidden defaults;
- compare persisted results and stage arrays for behavior-preserving work;
- distinguish unavailable/censored data from physical zero;
- report full-sweep or external-validation limits separately from checks that
  actually ran.

## 6. Where current details live

- Physical/status invariants: [`physical_model.md`](physical_model.md)
- Architecture, I/O, and checks: [`engineering_contract.md`](engineering_contract.md)
- Cosmological extension: [`research_roadmap.md`](research_roadmap.md)
- Current source explanation: [`../project/README.md`](../project/README.md)
- Implemented detector specification: [`../specs/lisa_detectability.md`](../specs/lisa_detectability.md)
- Detector implementation record: [`../plans/active/lisa_detectability_plan.md`](../plans/active/lisa_detectability_plan.md)
