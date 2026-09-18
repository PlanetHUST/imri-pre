"""Scientific-baseline comparison for the Round-2 refactor.

Compares two complete run trees (single-halo outputs and/or a merger-tree batch
directory) and reports whether the *scientific* content is identical, ignoring
fields that legitimately vary between two runs of the same code:

* wall-clock timings (``runtime_s``, ``*_runtime_*``, ``*_time_per_*``);
* absolute paths and the directory a run was written to;
* the run timestamp in ``simulation_metadata.json``.

Usage::

    python scripts/compare_baseline.py capture <run_dir> <baseline.json>
    python scripts/compare_baseline.py check   <run_dir> <baseline.json>

``capture`` records the scientific content; ``check`` re-derives it and exits
non-zero if anything scientific differs, printing exactly what changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

#: Keys whose values are timings, paths or timestamps rather than physics.
VOLATILE_KEYS = {
    "runtime_s",
    "total_runtime_seconds",
    "total_runtime_minutes",
    "average_time_per_simulation_seconds",
    "timestamp_s_since_epoch",
    "date",
    "created_iso",
    "modified_iso",
    "output_path",
    "output_root",
    "output_dir",
    "tree_path",
    "path",
    "config_source",
    "runtime_statistics",
    "simulation_info",
}
VOLATILE_COLUMNS = {"runtime_s", "output_path"}
CSV_SUFFIX = ".csv"
JSON_SUFFIX = ".json"


def _strip(value: Any) -> Any:
    """Recursively drop volatile keys from a JSON-like structure."""
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [_strip(item) for item in value]
    return value


def _csv_signature(path: Path) -> str:
    frame = pd.read_csv(path)
    columns = [c for c in frame.columns if c not in VOLATILE_COLUMNS]
    payload = frame[columns].to_csv(index=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _json_signature(path: Path) -> str:
    payload = json.dumps(_strip(json.loads(path.read_text())), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def scientific_signature(run_dir: Path) -> dict[str, str]:
    """Return {relative path: content hash} over the scientific outputs."""
    signature: dict[str, str] = {}
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(run_dir))
        try:
            if path.suffix == CSV_SUFFIX:
                signature[relative] = _csv_signature(path)
            elif path.suffix == JSON_SUFFIX:
                signature[relative] = _json_signature(path)
        except Exception as error:  # noqa: BLE001 - a broken file is a real diff
            signature[relative] = f"UNREADABLE:{type(error).__name__}"
    return signature


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("capture", "check"))
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("baseline", type=Path)
    arguments = parser.parse_args(argv)

    signature = scientific_signature(arguments.run_dir)
    if arguments.action == "capture":
        arguments.baseline.parent.mkdir(parents=True, exist_ok=True)
        with arguments.baseline.open("w") as handle:
            json.dump(signature, handle, indent=2, sort_keys=True)
        print(f"captured {len(signature)} scientific files -> {arguments.baseline}")
        return 0

    reference = json.loads(arguments.baseline.read_text())
    missing = sorted(set(reference) - set(signature))
    added = sorted(set(signature) - set(reference))
    changed = sorted(
        name for name in set(reference) & set(signature)
        if reference[name] != signature[name]
    )
    for label, names in (("MISSING", missing), ("ADDED", added), ("CHANGED", changed)):
        for name in names:
            print(f"  {label}: {name}")
    if missing or added or changed:
        print(
            f"BASELINE MISMATCH: {len(missing)} missing, {len(added)} added, "
            f"{len(changed)} changed (of {len(reference)} files)"
        )
        return 1
    print(f"BASELINE OK: {len(reference)} scientific files identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
