#!/usr/bin/env python3
"""One-time, lossless output layout migration; no physics or src changes.

Default: print the plan. --apply moves artifacts and rewrites embedded paths.
--check validates current CSV run_dir and JSON benchmark_run dependencies.
The generated migration manifest preserves original hashes and path changes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
MOVES = {
    "outputs/physical_scenario_rates/torque_i_thermal_pre_optical_toggle": "outputs/archive/solver_attempts/torque_i_thermal_pre_optical_toggle",
    "outputs/physical_scenario_rates/explicit_solver_pause.json": "outputs/archive/solver_attempts/explicit_solver_pause.json",
    "outputs/parameter_dependence/.imri_parameter_phase_space.png.swp": "outputs/archive/editor_residue/.imri_parameter_phase_space.png.swp",
    "outputs/high_redshift_rates": "outputs/core/high_redshift_rates",
    "outputs/physical_scenario_rates": "outputs/core/physical_scenario_rates",
    "outputs/parameter_dependence": "outputs/supporting/parameter_dependence",
    "outputs/detectability_audit": "outputs/supporting/detectability_audit",
    "outputs/population_budget_audit": "outputs/supporting/population_budget_audit",
    "outputs/imf_1_400_rates": "outputs/archive/imf/imf_1_400_rates",
    "outputs/imf_13p6_400_reference": "outputs/archive/imf/imf_13p6_400_reference",
    "outputs/published_population_validation": "outputs/archive/populations/published_population_validation",
    "outputs/published_population_check": "outputs/archive/populations/published_population_check",
    "outputs/damping_benchmark": "outputs/archive/calibration/damping_benchmark",
    "outputs/host_rate_comparison": "outputs/archive/historical/host_rate_comparison",
}
TEXT_SUFFIXES = {".csv", ".json", ".md", ".log", ".html", ".txt", ".svg"}
MANIFEST = ROOT / "outputs/archive/output_migration_manifest.json"


def translate(text: str, mapping: dict[str, str]) -> str:
    # One substitution pass: nested source paths must beat their parents,
    # and replacements must never be processed as new inputs.
    pattern = "|".join(re.escape(key) for key in sorted(mapping, key=len, reverse=True))
    return re.sub(pattern, lambda match: mapping[match.group()], text)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_dependencies() -> int:
    missing = []
    references = 0
    for path in (ROOT / "outputs").rglob("*.csv"):
        with path.open(newline="") as stream:
            rows = csv.DictReader(stream)
            if not rows.fieldnames or "run_dir" not in rows.fieldnames:
                continue
            for ref in {row["run_dir"] for row in rows if row.get("run_dir")}:
                references += 1
                if not (ROOT / ref).is_dir():
                    missing.append((str(path.relative_to(ROOT)), ref))
    for path in (ROOT / "outputs").rglob("experiment.json"):
        ref = json.loads(path.read_text()).get("benchmark_run")
        if ref:
            references += 1
            if not (ROOT / ref).is_dir():
                missing.append((str(path.relative_to(ROOT)), ref))
    print(f"Checked {references} saved run/benchmark references; {len(missing)} missing.")
    for item in missing:
        print("MISSING:", *item)
    integrity_errors = []
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
        inverse = {target: source for source, target in manifest["moves"].items()}
        for record in manifest["files"]:
            path = ROOT / record["path"]
            if not path.is_file():
                integrity_errors.append((record["path"], "missing file"))
                continue
            data = path.read_bytes()
            if digest(data) != record["sha256"]:
                integrity_errors.append((record["path"], "changed since migration"))
            if record["path_text_updated"]:
                data = translate(data.decode("utf-8"), inverse).encode("utf-8")
            if digest(data) != record["original_sha256"]:
                integrity_errors.append((record["path"], "original bytes not recoverable"))
        print(f"Verified {len(manifest['files'])} migrated files; {len(integrity_errors)} integrity errors.")
        for item in integrity_errors:
            print("INTEGRITY:", *item)
    return bool(missing or integrity_errors)


def apply() -> int:
    if MANIFEST.exists():
        raise SystemExit("Migration already recorded; use --check, not --apply.")
    for source, target in MOVES.items():
        if not (ROOT / source).exists() or (ROOT / target).exists():
            raise SystemExit(f"Preflight failed: source absent or target exists: {source} -> {target}")
    if check_dependencies():
        raise SystemExit("Repair missing input references before migration.")
    # Inventory before changing anything, including original provenance text hashes.
    inventory = []
    for path in sorted((ROOT / "outputs").rglob("*")):
        if path.is_file():
            original = str(path.relative_to(ROOT))
            inventory.append(dict(original_path=original, path=translate(original, MOVES),
                                  original_sha256=digest(path.read_bytes())))
    for source, target in MOVES.items():
        destination = ROOT / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        (ROOT / source).rename(destination)
    changed = 0
    for record in inventory:
        path = ROOT / record["path"]
        data = path.read_bytes()
        if path.suffix in TEXT_SUFFIXES:
            original_text = data.decode("utf-8")
            updated = translate(original_text, MOVES).encode("utf-8")
            if updated != data:
                # Exact reversible check: only output path strings may change.
                inverse = {target: source for source, target in MOVES.items()}
                if translate(updated.decode("utf-8"), inverse).encode("utf-8") != data:
                    raise RuntimeError(f"Non-reversible path rewrite: {path}")
                path.write_bytes(updated)
                changed += 1
                data = updated
        record["sha256"] = digest(data)
        record["path_text_updated"] = record["sha256"] != record["original_sha256"]
    manifest = dict(schema_version=1, purpose="Output organization only; no simulation rerun or numeric changes",
                    moves=MOVES, retained_runtime_default="outputs/single_host",
                    files=inventory, files_with_path_text_updates=changed,
                    provenance_note="Historical paths in text were relocated; original hashes and inverse path map are preserved here.")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Preserved {len(inventory)} files; updated paths in {changed}; no files deleted.")
    return check_dependencies()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        return check_dependencies()
    if args.apply:
        return apply()
    for source, target in MOVES.items():
        print(f"{source} -> {target}")
    print("Runtime scratch outputs/single_host remains unchanged. No moves without --apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
