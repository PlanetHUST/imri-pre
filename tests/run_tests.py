"""Dependency-free test runner.

This collects the no-argument ``test_*`` functions from ``tests/core/`` and
``tests/optional/`` without using pytest's runner, reporting
pass/fail counts and a non-zero exit status on failure.  Some optional tests import pytest helpers; install the `test` extra as in the
quickstart. The same files are also collected by pytest.

    python tests/run_tests.py            # run everything
    python tests/run_tests.py gap traps  # run modules matching a substring
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path
from unittest import SkipTest

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
SOURCE_ROOT = PROJECT_ROOT / "src"
for path in (SOURCE_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"tests.{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    patterns = argv if argv is not None else sys.argv[1:]
    modules = sorted(TESTS_DIR.rglob("test_*.py"))
    if patterns:
        modules = [m for m in modules if any(p in m.stem for p in patterns)]
    if not modules:
        print("no test modules matched")
        return 1

    passed = failed = skipped = 0
    failures: list[tuple[str, str]] = []
    for path in modules:
        before = passed
        module = _load(path)
        names = sorted(n for n in dir(module) if n.startswith("test_"))
        for name in names:
            function = getattr(module, name)
            if not callable(function):
                continue
            try:
                function()
            except SkipTest:
                skipped += 1
            except Exception:
                failed += 1
                failures.append((f"{path.stem}.{name}", traceback.format_exc()))
                print(f"FAIL {path.relative_to(TESTS_DIR)}::{name}")
            else:
                passed += 1
        print(f"{path.relative_to(TESTS_DIR)}: {passed - before} passed")

    print(f"\n{'=' * 60}")
    for name, trace in failures:
        print(f"\nFAILED {name}\n{trace}")
    print(f"{passed} passed, {skipped} skipped, {failed} failed, {len(modules)} modules")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
