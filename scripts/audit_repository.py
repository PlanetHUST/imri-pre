"""Checkpoint A - repository, dependency and duplicated-physics audit.

Analysis only: this script never deletes or modifies repository code.  It
reconstructs the import graph mechanically and merges it with a curated
classification of every module and script, then writes

    output/round2/A_repository_audit/audit.json
    output/round2/A_repository_audit/audit.csv
    output/round2/A_repository_audit/findings.json

Run with::

    python scripts/audit_repository.py
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output" / "round2" / "A_repository_audit"
LOCAL_PACKAGES = {
    "physical_process",
    "disk_quantity",
    "workflows",
    "lisa",
    "viz",
    "src",
    "lisa_ref",
}

# ---------------------------------------------------------------------------
# Curated classification.  "canonical" means: this file is the authoritative
# implementation of the listed scientific content for production runs.
# ---------------------------------------------------------------------------
CLASSIFICATION: dict[str, dict[str, Any]] = {
    # ---- physics: constants, population, host-independent models ----------
    "src/physical_process/constants.py": dict(
        purpose="CGS/SI physical constants and unit aliases",
        role="canonical", physics=True, duplicate=False),
    "src/physical_process/agn.py": dict(
        purpose="AGN duty-cycle properties from SMBH mass",
        role="canonical", physics=True, duplicate=False),
    "src/physical_process/sampling.py": dict(
        purpose="Pop II / Pop III IMF sampling and stellar lifetimes",
        role="canonical", physics=True, duplicate=False,
        note="Uses the global NumPy RNG; seeded once per repeat by the workflow."),
    "src/physical_process/nsc_orbital.py": dict(
        purpose="NSC radius, initial orbits, disk-crossing mask, relaxation time",
        role="canonical", physics=True, duplicate=False),
    "src/physical_process/damp.py": dict(
        purpose="Orbital damping / embedding of disk-crossing objects",
        role="canonical", physics=True, duplicate=False),
    "src/physical_process/gw_frequency_radius.py": dict(
        purpose="GW frequency <-> radius, Peters timescales, LISA entry radius",
        role="canonical", physics=True, duplicate=False),
    "src/physical_process/redshift_evolve.py": dict(
        purpose="Linear redshift-time evolution model",
        role="canonical", physics=True, duplicate=False),
    "src/physical_process/cosmic_merge.py": dict(
        purpose="Simplified central-SMBH merger applied during evolution",
        role="canonical", physics=True, duplicate=False),
    # ---- physics: migration (Round-2 canonical package) -------------------
    "src/physical_process/migration/torque.py": dict(
        purpose="Lindblad / corotation / thermal / Type-II torque components",
        role="canonical", physics=True, duplicate=False,
        note="Owns no threshold; its gap parameter delegates to gap.py."),
    "src/physical_process/migration/gap.py": dict(
        purpose="Every gap parameter, threshold and gap-open decision",
        role="canonical", physics=True, duplicate=False,
        note="Sole definition of kii / kanagawa / crida. See F1."),
    "src/physical_process/migration/traps.py": dict(
        purpose="Trap identification from torque zeros (stable/anti/boundary/unresolved)",
        role="canonical", physics=True, duplicate=False),
    "src/physical_process/migration/evolution.py": dict(
        purpose="Regime selection and radial migration integration",
        role="canonical", physics=True, duplicate=False),
    # ---- physics: disk ----------------------------------------------------
    "src/disk_quantity/opacity.py": dict(
        purpose="OPAL + Alexander-Ferguson Rosseland opacity tables and lookup",
        role="canonical", physics=True, duplicate=False,
        note="Data and interpolation only; clips outside the tabulated range."),
    "src/disk_quantity/pagn_disk_adapter.py": dict(
        purpose="pAGN Sirko/Thompson disk solution -> CGS profile arrays",
        role="canonical", physics=True, duplicate=False,
        note="PRODUCTION disk structure."),
    "src/disk_quantity/disk_solve.py": dict(
        purpose="In-repo alpha-disk structure solver",
        role="diagnostic-alternative", physics=True, duplicate=True,
        note="NOT canonical. Reached only by the pAGN-vs-in-repo comparison figure."),
    # ---- physics: LISA ----------------------------------------------------
    "src/lisa/snr.py": dict(
        purpose="LISA sensitivity, waveform, cosmology and integrated SNR",
        role="canonical", physics=True, duplicate=False),
    "src/lisa/detectability.py": dict(
        purpose="Time-to-band-entry solving and the SNR threshold decision",
        role="canonical", physics=True, duplicate=False),
    # ---- workflow ---------------------------------------------------------
    "src/workflows/config.py": dict(purpose="IMRIDynConfig and model defaults", role="canonical", physics=False, duplicate=False),
    "src/workflows/engine.py": dict(purpose="Cached disk and torque profiles per SMBH mass", role="canonical", physics=False, duplicate=False),
    "src/workflows/models.py": dict(purpose="Typed payloads and result containers", role="canonical", physics=False, duplicate=False),
    "src/workflows/population.py": dict(purpose="Population sampling + disk-crossing stage", role="canonical", physics=False, duplicate=False),
    "src/workflows/single_work.py": dict(purpose="One Monte Carlo realization", role="canonical", physics=False, duplicate=False),
    "src/workflows/workflow.py": dict(purpose="Grid loop, seeding, saving", role="canonical", physics=False, duplicate=False),
    "src/workflows/workers.py": dict(purpose="Damping multiprocessing boundary", role="canonical", physics=False, duplicate=False),
    "src/workflows/event_rate.py": dict(purpose="Gas/GW handoff and event-rate classification", role="canonical", physics=True, duplicate=False,
        note="Sole definition of solve_gas_final_radius_cm."),
    "src/workflows/event_work.py": dict(purpose="Per-object pipeline orchestration", role="canonical", physics=True, duplicate=False),
    "src/workflows/outputs.py": dict(purpose="Result serialization and validation", role="canonical", physics=False, duplicate=False),
    "src/workflows/metadata.py": dict(purpose="Run metadata assembly", role="canonical", physics=False, duplicate=False),
    "src/workflows/reporting.py": dict(purpose="Terminal reporting", role="canonical", physics=False, duplicate=False),
    "src/workflows/runtime.py": dict(purpose="Verbosity switch", role="canonical", physics=False, duplicate=False),
    "src/workflows/cli.py": dict(purpose="CLI dispatcher (single-halo + tree batch)", role="canonical", physics=False, duplicate=False),
    "src/workflows/merger_tree.py": dict(purpose="Merger-tree CSV reader", role="canonical", physics=False, duplicate=False),
    "src/workflows/host_mapping.py": dict(purpose="Host -> local parameter mapping", role="canonical", physics=False, duplicate=False),
    "src/workflows/single_halo.py": dict(purpose="Programmatic single-halo interface", role="canonical", physics=False, duplicate=False),
    "src/workflows/tree_batch.py": dict(purpose="Deterministic parallel batch executor", role="canonical", physics=False, duplicate=False),
    # ---- visualization ----------------------------------------------------
    "src/viz/rate_plots.py": dict(purpose="Rate/time/status tables and figures", role="plotting", physics=False, duplicate=False),
    "src/viz/disk_plots.py": dict(purpose="Disk structure and torque figures", role="plotting", physics=False, duplicate=False),
    "src/viz/batch_plots.py": dict(purpose="Merger-tree batch interface figures", role="plotting", physics=False, duplicate=False),
    "src/viz/opacity_plots.py": dict(purpose="Opacity-table verification figure", role="plotting", physics=False, duplicate=False),
    # ---- scripts ----------------------------------------------------------
    "scripts/main.py": dict(purpose="CLI entry point + programmatic API", role="entry-point", physics=False, duplicate=False),
    "scripts/audit_repository.py": dict(purpose="This audit (Checkpoint A)", role="checkpoint", physics=False, duplicate=False),
    "scripts/compare_baseline.py": dict(purpose="Scientific-output regression comparison", role="checkpoint", physics=False, duplicate=False),
    "scripts/run_science_checkpoints.py": dict(purpose="Round-2 scientific checkpoints B-L", role="checkpoint", physics=False, duplicate=False),
    "scripts/run_interface_checkpoints.py": dict(purpose="Merger-tree interface checkpoints A-H", role="checkpoint", physics=False, duplicate=False),
    "scripts/run_redshift_scan.py": dict(purpose="Paired redshift scan driver", role="diagnostic", physics=False, duplicate=False),
    "scripts/validate_damping_model.py": dict(purpose="Damping invariants, scans, phase diagrams", role="diagnostic", physics=False, duplicate=False),
    "scripts/validate_gas_gw_handoff.py": dict(purpose="Gas/GW handoff regression checks", role="diagnostic", physics=False, duplicate=False,
        note="Local helpers are synthetic fixtures probing the canonical solver."),
    "scripts/validate_lisa_detectability.py": dict(purpose="LISA PSD/cosmology/SNR/classification checks", role="diagnostic", physics=False, duplicate=False),
    "scripts/validate_rate_plot_layer.py": dict(purpose="Rate table/figure layer harness", role="diagnostic", physics=False, duplicate=False),
    "scripts/plot_disk_structure.py": dict(purpose="Disk profile and solver-comparison figures", role="diagnostic", physics=False, duplicate=False),
    "scripts/plot_migration_timescales.py": dict(purpose="Migration timescale comparison figure", role="diagnostic", physics=False, duplicate=False,
        note="Round-2: now a thin consumer of physical_process.migration."),
    "scripts/plot_migration_traps.py": dict(purpose="Trap / anti-trap / LISA-entry radius figure", role="diagnostic", physics=False, duplicate=False,
        note="Round-2: now a thin consumer of physical_process.migration."),
    "scripts/report_event_summary.py": dict(purpose="Print an event summary from a results directory", role="diagnostic", physics=False, duplicate=False),
    # ---- tests ------------------------------------------------------------
    "tests/run_tests.py": dict(purpose="Dependency-free test runner", role="test", physics=False, duplicate=False),
    "tests/test_architecture.py": dict(purpose="Module-boundary and layering assertions", role="test", physics=False, duplicate=False),
    "tests/test_migration_gap.py": dict(purpose="Gap criteria correctness", role="test", physics=False, duplicate=False),
    "tests/test_migration_traps.py": dict(purpose="Trap identification correctness", role="test", physics=False, duplicate=False),
    "tests/test_merger_tree_interface.py": dict(purpose="Merger-tree interface invariants", role="test", physics=False, duplicate=False),
    # ---- supplied reference ----------------------------------------------
    "lisa_ref/LISA.py": dict(purpose="Supplied LISA reference implementation", role="reference", physics=True, duplicate=True),
    "lisa_ref/LISA_tools.py": dict(purpose="Supplied LISA reference tools", role="reference", physics=True, duplicate=True),
    "lisa_ref/LISAPhenomA.py": dict(purpose="Supplied PhenomA reference", role="reference", physics=True, duplicate=True),
    "lisa_ref/PhenomA.py": dict(purpose="Supplied PhenomA reference", role="reference", physics=True, duplicate=True),
    "lisa_ref/WaveformTools.py": dict(purpose="Supplied waveform reference tools", role="reference", physics=True, duplicate=True),
}

# Findings that drive the Round-2 refactor.
FINDINGS: list[dict[str, Any]] = [
    dict(
        id="F1-gap-parameter-inconsistency",
        resolution="EXPOSED: both parameters named, documented and configurable in migration/gap.py; 'kanagawa' added as an opt-in criterion; default unchanged. The physical question remains OPEN.",
        severity="high",
        title="Two different quantities are both called the gap parameter K",
        detail=(
            "disk_quantity/migration_torque.comp_gap_parameter_dimensionless uses "
            "K = q^2 / (alpha * h^5) (the pAGN/Kanagawa form) to build the Type-I "
            "gap attenuation and denominator, while "
            "physical_process/disk_migration.gap_opening_mask('kii') uses "
            "K = q^2 / (alpha * h^2) with threshold 11 to decide the Type-II "
            "switch. The exponent differs by h^3, so the two disagree by many "
            "orders of magnitude for thin disks."
        ),
        action=(
            "Make both explicit and separately named in a canonical gap module, "
            "expose the exponent and threshold as named documented parameters, "
            "add an opt-in 'kanagawa' criterion, and keep 'kii' as the default so "
            "no production result changes. Report the discrepancy as an "
            "unresolved model ambiguity."
        ),
    ),
    dict(
        id="F2-script-local-gap-physics",
        resolution='FIXED: both scripts call gap_diagnostics(); enforced by tests/test_architecture.py.',
        severity="high",
        title="trap_plot.py and migration_test.py reimplement the gap diagnostics",
        detail=(
            "Both scripts compute K_II, the Crida parameter and R_H/H inline "
            "rather than calling disk_migration, so a change to the canonical "
            "criterion would silently not propagate to the figures."
        ),
        action="Extract canonical gap_diagnostics() into src and make both scripts thin consumers.",
    ),
    dict(
        id="F3-migration-split-across-packages",
        resolution='FIXED: physical_process/migration/{torque,gap,traps,evolution}.py.',
        severity="medium",
        title="Migration physics is split between disk_quantity and physical_process",
        detail=(
            "Torque components live in disk_quantity/migration_torque.py while "
            "gap criteria, trap blocking and time integration live in "
            "physical_process/disk_migration.py, and trap classification is in "
            "the torque module. The scientific dependency graph is not visible "
            "from the package layout."
        ),
        action="Create physical_process/migration/ with torque.py, gap.py, traps.py, evolution.py.",
    ),
    dict(
        id="F4-lisa-physics-in-workflows",
        resolution='FIXED: src/lisa/ package with a documented boundary.',
        severity="medium",
        title="LISA instrument physics lives inside the workflow package",
        detail=(
            "workflows/lisa_snr.py (sensitivity, waveform, cosmology, SNR) and "
            "workflows/detect_rate.py (detectability) are detector physics, not "
            "orchestration."
        ),
        action="Move to a dedicated src/lisa package with an explicit public API.",
    ),
    dict(
        id="F5-plotting-inside-science-packages",
        resolution='FIXED: src/viz/; enforced by test_physics_packages_do_not_import_plotting.',
        severity="medium",
        title="Plotting code lives inside physics and workflow packages",
        detail=(
            "workflows/rate_plots.py (1898 lines), disk_quantity/disk_plot_utils.py "
            "(667 lines) and opacity.py's plot_opacity_verification are figure code "
            "inside scientific packages; opacity.py imports matplotlib at module "
            "scope purely for that plot."
        ),
        action="Create src/viz/ and move figure code there.",
    ),
    dict(
        id="F6-second-disk-solver",
        resolution='DOCUMENTED: kept and labelled a diagnostic alternative; not reached by any run.',
        severity="medium",
        title="disk_solve.py is a second, non-production disk-structure solver",
        detail=(
            "Production uses pAGN via pagn_disk_adapter. disk_solve.radial_structure "
            "is reached only by the pAGN-vs-disk_solve comparison figure, and it "
            "redefines sigma_SB, k_B, m_p, alpha and the inner radius locally."
        ),
        action=(
            "Keep it as an explicitly labelled diagnostic alternative (it backs a "
            "real comparison figure), document that it is NOT canonical, and source "
            "its shared constants from constants.py."
        ),
    ),
    dict(
        id="F7-dead-import",
        resolution='FIXED: import removed; checked by Checkpoint L.',
        severity="low",
        title="workflows/engine.py imports disk_solve but never uses it",
        detail="'import disk_quantity.disk_solve as disk' at engine.py:5 is unused.",
        action="Remove the import.",
    ),
    dict(
        id="F8-ambiguous-script-names",
        resolution='FIXED: all scripts renamed verb-first; checked by Checkpoint L.',
        severity="low",
        title="Script names do not describe what the scripts do",
        detail=(
            "final_analysis.py, damp_test.py, migration_test.py, "
            "gas_gw_handoff_test.py, lisa_detectability_test.py and "
            "imri_rate_visualization_test.py are diagnostics, not automated tests, "
            "and '*_test' wrongly suggests a pytest suite."
        ),
        action="Rename to verb-first names describing the action.",
    ),
    dict(
        id="F9-hardcoded-script-assumptions",
        resolution='FIXED: exposed as CLI options defaulting to the workflow values.',
        severity="medium",
        title="Figure scripts hard-code physical assumptions that differ from config",
        detail=(
            "trap_plot.py fixes REDSHIFT=10 and LISA_MIN_GW_FREQUENCY_HZ=1e-4 "
            "while the workflow defaults are z=15 and 1e-3; a reader comparing the "
            "figure to a run would silently compare different models."
        ),
        action="Expose them as CLI options defaulting to the workflow configuration values.",
    ),
    dict(
        id="F10-no-automated-tests",
        resolution='FIXED: tests/ with 40 assertions and a dependency-free runner.',
        severity="medium",
        title="The repository declares a pytest testpath but has no tests",
        detail=(
            "pyproject.toml sets testpaths=['tests'] and a pytest optional "
            "dependency, but no tests/ directory exists and pytest is not "
            "installed in the supplied environment."
        ),
        action=(
            "Add tests/ with dependency-free automated correctness checks and a "
            "runner that works without pytest in this environment."
        ),
    ),
]


def local_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                found.add("." + (node.module or ""))
            elif node.module and node.module.split(".")[0] in LOCAL_PACKAGES:
                found.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in LOCAL_PACKAGES:
                    found.add(alias.name)
    return sorted(found)


def build_audit() -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    files = sorted(
        p for p in PROJECT_ROOT.rglob("*.py")
        if "__pycache__" not in str(p)
        and ".venv" not in str(p)
        and "output" not in p.relative_to(PROJECT_ROOT).parts
    )
    for path in files:
        rel = str(path.relative_to(PROJECT_ROOT))
        source = path.read_text()
        tree = ast.parse(source)
        meta = CLASSIFICATION.get(rel, None)
        if meta is None and path.name == "__init__.py":
            meta = dict(purpose="package marker / public API re-export",
                        role="package-init", physics=False, duplicate=False)
        if meta is None:
            meta = dict(purpose="", role="unclassified", physics=False, duplicate=False)
        rows.append(
            {
                "file": rel,
                "lines": len(source.splitlines()),
                "functions": sum(isinstance(n, ast.FunctionDef) for n in ast.walk(tree)),
                "classes": sum(isinstance(n, ast.ClassDef) for n in ast.walk(tree)),
                "purpose": meta["purpose"],
                "role": meta["role"],
                "contains_physics": meta["physics"],
                "duplicate_or_alternative": meta["duplicate"],
                "note": meta.get("note", ""),
                "local_imports": ";".join(local_imports(path)),
            }
        )
    frame = pd.DataFrame(rows)

    importers: dict[str, list[str]] = {}
    for row in rows:
        stem = Path(row["file"]).stem
        importers[row["file"]] = sorted(
            other["file"]
            for other in rows
            if other["file"] != row["file"] and stem in other["local_imports"].split(";")
            or (other["file"] != row["file"] and any(
                part.split(".")[-1] == stem for part in other["local_imports"].split(";") if part
            ))
        )
    summary = {
        "files": len(rows),
        "total_lines": int(frame["lines"].sum()),
        "by_role": frame["role"].value_counts().to_dict(),
        "physics_files": int(frame["contains_physics"].sum()),
        "duplicate_or_alternative_files": int(frame["duplicate_or_alternative"].sum()),
        "unclassified": frame.loc[frame["role"] == "unclassified", "file"].tolist(),
        "importers": importers,
    }
    return frame, summary


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame, summary = build_audit()
    frame.to_csv(OUTPUT_DIR / "audit.csv", index=False)
    with (OUTPUT_DIR / "audit.json").open("w") as handle:
        json.dump({"summary": summary, "files": frame.to_dict(orient="records")}, handle, indent=2)
    with (OUTPUT_DIR / "findings.json").open("w") as handle:
        json.dump(FINDINGS, handle, indent=2)

    print(f"Checkpoint A - repository audit ({summary['files']} files, {summary['total_lines']} lines)")
    print("  roles:", summary["by_role"])
    if summary["unclassified"]:
        print("  UNCLASSIFIED:", summary["unclassified"])
    print(f"  physics files: {summary['physics_files']}, "
          f"duplicate/alternative: {summary['duplicate_or_alternative_files']}")
    print("  findings:")
    for finding in FINDINGS:
        print(f"    [{finding['severity']:>6}] {finding['id']}: {finding['title']}")
    print(f"  => {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
