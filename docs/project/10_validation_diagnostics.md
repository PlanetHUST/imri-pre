# Part 10 — Validation and Diagnostics

## 1. Idea

The physics and orchestration parts are held together by a set of deterministic
harnesses and invariants. Validation exists so that a structural change, a
prescription change, or a label change can be proven **not** to have altered
physics, and so that censoring/error states never masquerade as zeros.

The main idea: *every diagnostic calls production functions — it never copies
their equations — and every number that is plotted or persisted is reproducible
from a saved artifact.*

## 2. Doing logic

### 2.1 Deterministic harness scripts

| Script | Validates |
|---|---|
| `scripts/migration_test.py` | semi-analytic vs torque migration timescales, shared gap mask, trap differences |
| `scripts/trap_plot.py` | stable/anti-trap radii vs SMBH mass, LISA entry, KII vs Crida |
| `scripts/disk_plot.py` | pAGN vs legacy disk-solver profile comparison, torque components |
| `scripts/damp_test.py` | node membership, active-node damping, embedded contract, phase-space trajectory, first-passage times |
| `scripts/imri_rate_visualization_test.py` | plotted numbers: monotonic curves, endpoint = final rate, valid zero vs unavailable, paired seeds, compact CSVs reproduce plotted arrays |
| `scripts/high_mass_boundary_test.py` (planned; contract in [`../reference/physical_model.md`](../reference/physical_model.md), not yet in the tree) | 1e5 handoff and 1e6 inside-ISCO routing, torque-call spies, counter conservation |
| `scripts/final_analysis.py` | load persisted run, validate tables, regenerate rate figures |

Each script is a short explicit pipeline (visible constants → typed results →
production calls → reporting/plotting → `main()`), fails loudly on invariant
violations, and exits nonzero.

### 2.2 Counter conservation invariants

Enforced at save time (`outputs.save_migration_diagnostics`) and by the plot
layer:

```
n_band_eval = n_entry_in_domain + n_entry_handoff + n_entry_inside_isco + n_profile_error
n_mig_eval  = n_entry_in_domain + n_entry_handoff + n_profile_error
            = n_mig_finite + n_mig_trap + n_mig_error
```

plus: trap-bound count matches event records; semi scenario has zero traps;
profile-error count matches migration-error count; each `band_entry_status` in
the event ledger matches its repeat counter; status fractions sum to their
denominator and to one (valid bins).

### 2.3 Status semantics invariants

- `band_entry_inside_isco` never appears in the transport denominator and never
  increments torque counters.
- `not evaluated` stays distinct from `non-detectable` and `no-IMRI`.
- valid zero (`0.0` with `result_state=valid`) stays distinct from unavailable
  (NaN/null with an explicit status).
- Unknown raw categories raise rather than creating silent legend entries.

### 2.4 Parity checks for refactors

The reconstruction contract
([`../reference/engineering_contract.md`](../reference/engineering_contract.md)) requires exact parity
after any structural change:

- selected-torque array equality across production, `migration_test.py`,
  `trap_plot.py`, and disk torque plotting (element-for-element, both KII and
  Crida);
- migration outcome parity for every supported case (already-at-target, finite,
  trap with/without blocking radius, errors);
- seeded single-workflow parity of all sampled arrays, counters, statuses,
  rates, and reason strings;
- CSV/JSON schema and value parity;
- diagnostic artifact parity (figures, printed diagnostics).

### 2.5 Canonical verification commands

From the repo root, with the project interpreter:

```bash
# deterministic figure/number harness
.../bin/python scripts/imri_rate_visualization_test.py

# bytecode + import integrity
.../bin/python -m compileall src scripts

# configured type checking (basic mode, includes src + scripts)
.../bin/pyright

# whitespace/conflict hygiene
git diff --check
```

plus the physics-focused harnesses (`migration_test.py`, `trap_plot.py`,
`disk_plot.py`, `damp_test.py`) and, for workflow-level changes, a reduced
seeded `main.py` smoke run through the real pAGN path.

## 3. Caveats

- **Never copy a production formula into a diagnostic.** A matching figure made
  from a copied formula proves nothing about the production code.
- **Do not update a baseline to accept an unexplained difference.** If a stage
  changes a scientific array, classification, random sequence, or persisted
  value, stop and find the ownership/refactor error.
- **Preserve user-owned dirty changes.** The working tree is usually mid-task;
  freeze current behavior around unrelated edits rather than resetting them.
- **Determinism is fragile.** Any reordering of random draws, even with the same
  seed, breaks paired parity. Do not reorder sampling/damping/migration calls
  casually.
- **Seeds and populations are not persisted in old outputs.** Earlier reruns are
  not paired experiments; do not compare candidate totals across unseeded runs.
- **Ten repeats is a pilot.** Percentile bands need ≥ 10 valid repeats; paper
  figures should prefer ≥ 30 and state `N` in the caption.
