# Round-2 report: scientific refactor, physical transparency, repository consolidation

---

## A. Original architecture

| Layer | Modules |
|---|---|
| Entry | `scripts/main.py` → `workflows/cli.py` → `workflows/workflow.run_simulation` |
| One realization | `workflows/single_work.py` → `workflows/event_work.py` |
| Shared resources | `workflows/engine.py` (cached pAGN disk + torque profiles) |
| Parallelism | `workflows/workers.py` (inner damping `multiprocessing.Pool`) |
| Physics | `physical_process/` (damping, orbits, IMF, GW), `disk_quantity/` (opacity, disk, torque) |
| Output | `workflows/outputs.py`, `metadata.py`, `reporting.py` |
| Round-1 batch | `merger_tree.py`, `host_mapping.py`, `single_halo.py`, `tree_batch.py` |

Randomness uses the **global NumPy RNG**, seeded once per repeat; only the inner
damping loop is parallel and it is order-independent.

---

## B. Physical audit (Checkpoint A)

57 files, 26 086 lines audited. Artefacts:
`output/round2/A_repository_audit/{audit.csv,audit.json,findings.json}`.

| ID | Sev | Finding |
|---|---|---|
| **F1** | high | Two different quantities both called the gap parameter `K` |
| **F2** | high | `trap_plot.py` and `migration_test.py` reimplemented the gap diagnostics |
| F3 | med | Migration physics split across `disk_quantity/` and `physical_process/` |
| F4 | med | LISA instrument physics living inside the workflow package |
| F5 | med | 2 565 lines of plotting inside physics/workflow packages |
| F6 | med | `disk_solve.py` is a second, non-production disk solver |
| F7 | low | `engine.py` imported `disk_solve` without using it |
| F8 | low | Six scripts named `*_test` that are diagnostics, not tests |
| F9 | med | Figure scripts hard-coded `z=10`, `f_min=1e-4` against workflow defaults `z=15`, `1e-3` |
| F10 | med | `pyproject.toml` declared `testpaths=["tests"]` but no tests existed |

**F1 in detail.** `migration_torque.comp_gap_parameter_dimensionless` used
`K = q²/(α h⁵)` for the Type-I torque attenuation, while
`disk_migration.gap_opening_mask("kii")` used `K = q²/(α h²) > 11` for the
Type-II switch. They differ by `h⁻³` — six orders of magnitude for `h ≈ 0.01`.
Two scripts had copy-pasted the second form inline.

---

## C. New architecture

```
src/
├── physical_process/        constants · agn · sampling · nsc_orbital · damp
│   │                        gw_frequency_radius · redshift_evolve · cosmic_merge
│   └── migration/           torque.py · gap.py · traps.py · evolution.py
├── disk_quantity/           opacity.py · pagn_disk_adapter.py · disk_solve.py
├── lisa/                    snr.py · detectability.py
├── workflows/               config · engine · population · single_work · event_work
│                            event_rate · outputs · metadata · cli · merger_tree
│                            host_mapping · single_halo · tree_batch
└── viz/                     rate_plots · disk_plots · batch_plots · opacity_plots

tests/     4 modules, 48 automated assertions, runnable without pytest
scripts/   14 thin entry points, verb-first names
docs/      physics_model.md · merger_tree_interface.md · round2_report.md
```

`src/` is scientific truth; `scripts/`, `tests/` and `viz/` consume it.
`tests/test_architecture.py` enforces this mechanically.

---

## D. Disk

Canonical production structure is **pAGN** via `disk_quantity/pagn_disk_adapter`;
opacity is `disk_quantity/opacity` (OPAL + Alexander-Ferguson tables).
`disk_solve.py` is retained and explicitly labelled a **diagnostic alternative**
— it is not reached by any run, only by the pAGN-vs-in-repo comparison figure.

`opacity.py` no longer imports matplotlib: its verification plot moved to
`viz/opacity_plots.py`, leaving a pure data + interpolation module.

Checkpoint B results: all quantities finite and positive, `h < 1` everywhere,
`c_s = H Ω` verified to **4×10⁻¹⁶**, Toomre `Q → 1` at the outer boundary.

---

## E. Damping

`physical_process/damp.py` remains the single implementation; unchanged.
Checkpoint C scans 27 configurations (3 masses × 3 radii × 3 inclinations) and
verifies: every evaluation returns a status, damping times are finite and
non-negative, damping **never increases** inclination or eccentricity, final
elements stay physical, and both embedding and non-embedding outcomes occur.

---

## F. Migration

Split into four modules with disjoint ownership:

* **`torque.py`** — components only, owns no threshold. Its gap parameter now
  *delegates* to `gap.kanagawa_gap_parameter`, so attenuation and criteria can
  never drift apart.
* **`gap.py`** — every gap parameter and threshold, all named and configurable
  through `GapCriterion`; `gap_diagnostics()` returns all four diagnostics plus
  the decision.
* **`traps.py`** — trap classification, extended to report **boundary** and
  **unresolved** roots so a physical trap is distinguishable from a numerical
  one.
* **`evolution.py`** — regime selection and integration, owning no equation
  from the other three.

Checkpoint D verifies the decomposition identity to 10⁻¹² and confirms the
total torque changes sign. Checkpoint F treats trap existence as a **result**,
not an assumption: it scans SMBH mass, classifies every torque zero, measures
each trap's basin width, and tabulates the outcome under *all three* gap
criteria so the criterion's effect on trapping is visible. Under the current
default, traps survive only at 10⁶ M⊙ (narrowest basin 0.04 decades, flagged);
under `kii` they also appear at 10³ M⊙. See §J.1.

**Regime transition.** The Type-I → Type-II switch is a hard per-cell switch,
intentionally discontinuous; no blending was added. Under the previous `kii`
default it almost never fired; under the current `kanagawa` default it fires
across most of the disk, so the discontinuity is now genuinely exercised.

---

## G. Gas/GW

The handoff criterion `solve_gas_final_radius_cm` is verified to be defined
**exactly once**, and the production chain is
`event_work → classify_event_rate → solve_gas_final_radius_cm`. Checkpoint G
calls that same solver rather than reimplementing it (an earlier draft of the
checkpoint did reimplement the gas timescale; that was the bug, and it was
fixed). Peters `a⁴` scaling verified at log-slope **4.02**.

---

## H. Event formation

Checkpoint H traces one controlled population through every stage and asserts
the funnel is monotone: `disk_crossing ≤ sampled`,
`bh_candidates ≤ embedded + migration_formed`, `imri ≤ bh_candidates`,
`detectable ≤ imri`, `finite + trap_bound ≤ evaluated`. Every evaluated object
carries an `event_rate_reason` and a `detect_rate_reason`, so a zero is always
explainable. Execution failure and physical zero remain strictly separate.

---

## I. LISA

Moved out of `workflows/` into `src/lisa/` with an explicit boundary documented
in the package docstring: **this package never decides whether an object becomes
an IMRI.** Checkpoint I verifies redshift is applied exactly once
(`M_det = (1+z) M_src` to 10⁻¹²), SNR scales as **exactly** `1/D_L` when only
`H0` changes (to 1 part in 10⁹), the PSD has an interior sensitivity minimum,
and the SNR value is threshold-independent while the decision is not.

It also records that SNR is **not** monotone in redshift at fixed source-frame
masses and fixed observed frequency — `(1+z)` raises the detector-frame chirp
mass while distance grows. An earlier draft asserted monotonicity; that
assertion was wrong and was replaced by the unambiguous `1/D_L` test.

---

## J. Scientific changes

The refactor itself changed nothing: every scientific output stayed
byte-identical through all of §C–§I (32 files, verified after each step).

**One deliberate scientific decision was then taken on top of it**, at the
model authors' instruction.

### J.1 Default gap criterion: `kii` → `kanagawa`

| | |
|---|---|
| **Old behaviour** | Type-II switch used `K_II = q²/(α h²) > 11`, while the Type-I torque attenuation in the same pipeline used `K = q²/(α h⁵)`. Two inconsistent quantities. |
| **New behaviour** | Both use `K = q²/(α h⁵) > 25`. `DEFAULT_GAP_CRITERION = "kanagawa"`, set in one place (`migration/gap.py`). |
| **Reason** | Resolves finding F1. The switch and the attenuation are now driven by one quantity instead of two differing by `h⁻³`. |
| **Physical interpretation** | Far more of the disk is Type II (64–94 % for `M• = 10³–10⁵ M⊙`). The Type-II torque is inward everywhere, so the outward-torque regions that create migration traps are erased: **traps stop operating for `M• ≤ 10⁵ M⊙`**, surviving only at `10⁶ M⊙`. Objects that were trap-bound now migrate to the GW-driven regime and become IMRI candidates. |
| **Measured effect** | Controlled run (`M• = 10⁵`, `M_NSC = 5×10⁴`, seed 777): population/damping/embedding **bit-identical**; migration `trap_bound` 3→0 and `finite` 0→3; trap-bound mass 223→0 M⊙; IMRI candidates 0→1.5; local rate 0→0.15 Myr⁻¹; detectable IMRIs 0→0 (still sub-threshold). 29 of 32 baseline files changed. |
| **Validation** | `--gap-open-condition kii` reproduces the *entire* pre-change baseline byte-for-byte (32/32). Checkpoints B–L re-run: 9 PASS, 2 WARN, 0 FAIL. Tests extended to 43, including assertions that the default is single-sourced, that it equals the torque model's own gap parameter, and that `kii` remains reachable. |
| **Risk** | The threshold is load-bearing; it was subsequently calibrated (§J.2). |

### J.2 Calibration of the `kanagawa` threshold

`scripts/calibrate_gap_threshold.py`; artefacts in
`output/round2/calibration_gap_threshold/`.

| | |
|---|---|
| **Old behaviour** | `K > 25` chosen by analogy with `κ_norm`, with no stated meaning and no sensitivity study. Not reachable from the CLI. |
| **New behaviour** | Same value, now *derived*: the Type-I gap denominator `1 + K/κ_norm` is proven identical to the Kanagawa gap-depth law `Σ_gap/Σ₀ = 1/(1+0.04K)` to 2.2×10⁻¹⁶, so `K = 25` is the half-depth point. Exposed as `--gap-threshold`, convertible via `depletion_fraction()` / `threshold_for_depletion()`, recorded in provenance. |
| **Reason** | A load-bearing threshold must be interpretable, reproducible and testable. |
| **Physical interpretation** | `K > K*` means "the gap is cleared below residual surface density `f = 1/(1+0.04K*)`". `K = 25` ⇔ `f = 0.5`. |
| **Alternative tested** | Placing the switch at `\|Γ_I\| = \|Γ_II\|` (continuous transition) was measured and **rejected**: the crossover spans 2.5×10⁶ in K across the mass grid, so no global threshold achieves continuity. |
| **Sensitivity** | IMRI yield is flat across two decades in `K*` (f = 0.9→0.05); trap-bound fraction is not — traps reappear at `M• = 10⁵ M⊙` once `f ≤ 0.1`. |
| **Validation** | 5 new tests including the gap-depth identity, the inverse-helper round trip, and CLI→config→worker plumbing of the override. Numerical results unchanged (same value). |
| **Residual risk** | Whether "gapped" means 50 % or 10 % depletion is a modelling choice not settleable offline. |

### J.3 Smaller, opt-in changes

| Change | Old | New | Reason | Validation |
|---|---|---|---|---|
| Gap thresholds/exponents | literals inside expressions | named constants, configurable via `GapCriterion` | Round-2 §10 | `test_thresholds_are_configurable` |
| `kii` criterion | the only option | retained as an explicit selection | pre-change reproducibility | byte-identical baseline replay |
| Trap root categories | stable / anti only | + boundary + unresolved | distinguish physical from numerical traps | Checkpoint F |
| `trap_plot` `z`, `f_min` | hard-coded 10, 1e-4 | default to workflow values (15, 1e-3), overridable | F9: figure and run described different models | `--help` output |

The last one **changes the default figure**, deliberately: it previously
depicted a model the simulation does not run. The old values remain reachable
via `--redshift 10 --lisa-min-gw-frequency-hz 1e-4`.

---

## K. Removed and relocated code

| Old | Reason | Replacement |
|---|---|---|
| `disk_quantity/migration_torque.py` | migration split across packages (F3) | `physical_process/migration/torque.py` + `traps.py` |
| `physical_process/disk_migration.py` | mixed gap, torque selection, traps, integration | `migration/gap.py` + `traps.py` + `evolution.py` |
| `workflows/lisa_snr.py` | detector physics in the workflow package (F4) | `lisa/snr.py` |
| `workflows/detect_rate.py` | detector physics in the workflow package (F4) | `lisa/detectability.py` |
| `workflows/cosmic_merge.py`, `redshift_evolve.py` | physical models in the workflow package; caused an import cycle | `physical_process/` |
| `workflows/rate_plots.py` (1 898 lines) | plotting inside a workflow package (F5) | `viz/rate_plots.py` |
| `disk_quantity/disk_plot_utils.py` (667 lines) | plotting inside a physics package (F5) | `viz/disk_plots.py` |
| `workflows/tree_batch_plots.py` | plotting inside a workflow package | `viz/batch_plots.py` |
| `opacity.plot_opacity_verification` | forced matplotlib into a data module | `viz/opacity_plots.py` |
| `engine.py: import disk_solve` | dead import (F7) | removed |
| `scripts/trap_plot.py` | duplicated gap physics (F2), hard-coded assumptions (F9) | `scripts/plot_migration_traps.py` |
| `scripts/migration_test.py` | duplicated gap physics (F2) | `scripts/plot_migration_timescales.py` |
| `scripts/{damp,gas_gw_handoff,lisa_detectability,imri_rate_visualization}_test.py` | `*_test` names imply pytest (F8) | `validate_*.py` |
| `scripts/final_analysis.py` | ambiguous name (F8) | `report_event_summary.py` |
| `scripts/{disk_plot,redshift_scan,interface_checkpoints}.py` | non-descriptive names | `plot_disk_structure.py`, `run_redshift_scan.py`, `run_interface_checkpoints.py` |

*Nothing was deleted before its replacement was validated.* The repository is
**not** under version control, so git history is not available as an archive —
this table is the record.

---

## L. Regression results

| Check | Result |
|---|---|
| Scientific baseline (32 files) after every **refactor** step | **identical** |
| Scientific baseline after the **criterion change** | 29/32 changed, as intended (§J.1) |
| Pre-change baseline replayed with `--gap-open-condition kii` | **32/32 identical** |
| Single-halo CLI | runs; outputs reflect the new criterion |
| Merger-tree batch CLI | runs; provenance preserved |
| Serial vs parallel (`--workers` 1 vs 4) | identical job set, seeds, statuses and results |
| Round-1 checkpoints A–H | A–C, E–H PASS; D WARN (CSV int/float formatting only) |
| Round-2 checkpoints B–L | 9 PASS, 2 WARN (both documented physics), 0 FAIL |
| Automated tests | **48 passed, 0 failed** |
| Existing validation scripts | all 4 pass |
| Compilation and imports | clean, no cycles |

The two Round-2 WARNs are deliberate: the opacity table's 2.4-decade step at a
table transition, and the 0.008-decade trap basin at 10⁶ M⊙.

---

## M. Remaining limitations

**The gap criterion was resolved in favour of `kanagawa`, and the consequence
is large.** Migration traps no longer operate for `M• ≤ 10⁵ M⊙` — the entire
torque profile is inward there. Because migration trapping is a central premise
of this project, every earlier trap-dependent conclusion needs re-examination.
The threshold `K > 25` has been calibrated against the model's own gap-depth
law (§J.2) and means "gap cleared below half the unperturbed surface density".
What no offline study can settle is whether 50 % is the right depletion
fraction; the IMRI yield is insensitive to it across two decades, but
trap-dependent conclusions are not.

Other limitations:

* **Physics was re-homed, not re-derived.** `damp.py` (980 lines) and the pAGN
  torque coefficients were moved and documented but not independently
  re-validated against their source literature.
* **`disk_solve.py` disagrees with pAGN by construction** and is unused in
  production; the comparison figure is the only place the two meet.
* **Opacity is silently clipped** outside its table; callers get no signal.
* **Eccentric evolution is unsupported** in the SNR path (`e ≠ 0` →
  `model_unavailable`); the Peters threshold uses a fixed eccentricity.
* **The rotation law is not exactly Keplerian** (1.1×10⁻⁴); code that
  recomputes `Ω = √(GM/R³)` instead of reading the disk's own `Ω` incurs that
  error.
* **Narrow trap basins** can be numerical. Checkpoint F reports basin widths so
  the caller can judge, but the production integrator does not filter on them.
* **`checkpoint_grid_interval`, `kappa_t`, `kappa_norm`** and the pAGN
  `MigrationParameters` defaults remain unexplained inherited values.
* **The supplied merger tree is one halo's history** (`z ∈ [6.000, 6.038]`)
  with an empty `MBH` column; the `stellar-scaling` mass fractions are
  user-controlled placeholders, not calibrated relations.
* **No pytest in the supplied environment**, so `tests/run_tests.py` is a
  bespoke runner; the tests are pytest-compatible if it is ever installed.
* **Performance was not optimised**; spawned batch workers re-import the stack,
  giving ~2.7× speedup on 4 workers.

---

## N. Execution-boundary compliance

1. All work was performed inside the supplied local project path
   (`.../task_003/run/attachment`).
2. No internet or network access was used at any point.
3. No dependencies were installed, downloaded, upgraded or downgraded.
4. No new virtual environment was created; the supplied `./.venv` was used
   exactly as provided (it contains no pytest, which is why the test runner is
   dependency-free).
5. No AI subagents, delegated agents, coding-agent CLIs or agent-swarm
   mechanisms were used. All inspection, reasoning, editing, testing and
   validation were performed directly in this session. The only parallelism is
   `multiprocessing` / `ProcessPoolExecutor` **inside the simulation code**.
6. Only local repository code, the supplied `galaxy_1e12.csv`, the supplied
   `lisa_ref/` reference material and the supplied environment were used.

One incidental note: a `git init` was run by accident while staging file moves
and was immediately removed — the repository was not under version control
before and is not now.
