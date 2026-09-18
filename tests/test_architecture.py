"""Correctness tests for module boundaries and architectural invariants.

These encode the Round-2 architecture rules so that a future change which
reintroduces script-local physics, a duplicated gap parameter, or a
physics->plotting dependency fails a test rather than passing silently.
"""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
SCRIPTS = PROJECT_ROOT / "scripts"

#: Expressions that may only appear in the canonical gap module.
GAP_FORMULA_FRAGMENTS = (
    "np.cbrt(mass_ratio / 3.0)",
    "50.0 * alpha * local_aspect_ratio**2",
    "0.75 / hill_ratio",
)


def _python_files(root: Path) -> list[Path]:
    return [p for p in sorted(root.rglob("*.py")) if "__pycache__" not in str(p)]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def test_no_script_reimplements_the_gap_formulas() -> None:
    """Finding F2 must stay fixed: gap physics lives in one module only."""
    offenders = []
    for path in _python_files(SCRIPTS):
        text = path.read_text()
        for fragment in GAP_FORMULA_FRAGMENTS:
            if fragment in text:
                offenders.append(f"{path.name}: {fragment}")
    assert not offenders, f"script-local gap physics reappeared: {offenders}"


def test_gap_parameter_is_defined_exactly_once() -> None:
    """Only gap.py may spell out the q^2/(alpha h^n) closed forms."""
    definitions = []
    for path in _python_files(SRC):
        text = path.read_text()
        if "mass_ratio**2 / (" in text or "mass_ratio**2 / (\n" in text:
            definitions.append(path.relative_to(SRC).as_posix())
    assert definitions == ["physical_process/migration/gap.py"], definitions


def test_physics_packages_do_not_import_plotting() -> None:
    """Scientific modules must not depend on matplotlib or on viz."""
    offenders = []
    for package in ("physical_process", "disk_quantity", "lisa"):
        for path in _python_files(SRC / package):
            for module in _imports(path):
                root = module.split(".")[0]
                if root in {"matplotlib", "viz", "pylab", "seaborn"}:
                    offenders.append(f"{path.relative_to(SRC)} imports {module}")
    assert not offenders, f"physics depends on plotting: {offenders}"


def test_physics_packages_do_not_import_workflows() -> None:
    """Physics must not depend on orchestration (keeps the layering acyclic)."""
    offenders = []
    for package in ("physical_process", "disk_quantity"):
        for path in _python_files(SRC / package):
            for module in _imports(path):
                if module.split(".")[0] == "workflows":
                    offenders.append(f"{path.relative_to(SRC)} imports {module}")
    assert not offenders, f"physics depends on workflows: {offenders}"


def test_physics_modules_do_not_read_merger_tree_column_names() -> None:
    """Round-1 rule: CSV field names never leak into the physical modules."""
    tree_columns = ('"Mstar"', '"inode"', '"Mhot"', '"Mcold"', '"MBH"', '"br_b4"')
    offenders = []
    for package in ("physical_process", "disk_quantity", "lisa"):
        for path in _python_files(SRC / package):
            text = path.read_text()
            for column in tree_columns:
                if column in text:
                    offenders.append(f"{path.relative_to(SRC)}: {column}")
    assert not offenders, f"tree schema leaked into physics: {offenders}"


def test_viz_modules_select_a_headless_backend() -> None:
    """Figure modules must be importable inside worker processes."""
    offenders = []
    for path in _python_files(SRC / "viz"):
        if path.name == "__init__.py":
            continue
        text = path.read_text()
        if "matplotlib" in text and 'use("Agg")' not in text:
            offenders.append(path.name)
    assert not offenders, f"viz modules without an Agg backend: {offenders}"


def test_superseded_modules_are_gone() -> None:
    """The replaced modules must not linger beside their replacements."""
    removed = [
        SRC / "disk_quantity" / "migration_torque.py",
        SRC / "physical_process" / "disk_migration.py",
        SRC / "disk_quantity" / "disk_plot_utils.py",
        SRC / "workflows" / "lisa_snr.py",
        SRC / "workflows" / "detect_rate.py",
        SRC / "workflows" / "rate_plots.py",
        SRC / "workflows" / "tree_batch_plots.py",
        SRC / "workflows" / "cosmic_merge.py",
        SRC / "workflows" / "redshift_evolve.py",
    ]
    still_present = [p.relative_to(SRC).as_posix() for p in removed if p.exists()]
    assert not still_present, f"superseded modules still present: {still_present}"


def test_canonical_packages_expose_a_public_api() -> None:
    for package in ("physical_process.migration", "lisa", "workflows", "viz"):
        module = __import__(package, fromlist=["__all__"])
        assert getattr(module, "__all__", None), f"{package} has no __all__"


def test_no_circular_imports_between_top_level_packages() -> None:
    """Import every package in isolation; a cycle raises ImportError."""
    import importlib
    import subprocess
    import sys

    for package in ("physical_process", "disk_quantity", "lisa", "viz", "workflows"):
        result = subprocess.run(
            [sys.executable, "-c", f"import {package}"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{package}: {result.stderr[-400:]}"
        importlib.import_module(package)
