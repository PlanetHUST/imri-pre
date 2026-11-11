"""Output organization must preserve artifacts and cross-folder dependencies."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "output_layout", Path(__file__).resolve().parents[2] / "scripts/reorganize_outputs.py"
)
layout = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(layout)


def test_path_translation_prefers_nested_moves_and_is_single_pass():
    mapping = {"outputs/old": "outputs/new", "outputs/old/attempt": "outputs/archive/attempt"}
    original = "outputs/old/run outputs/old/attempt/log outputs/unrelated"
    updated = layout.translate(original, mapping)
    assert updated == "outputs/new/run outputs/archive/attempt/log outputs/unrelated"
    assert layout.translate(updated, {v: k for k, v in mapping.items()}) == original


def test_lossless_migration_and_dependency_check(tmp_path, monkeypatch):
    old = tmp_path / "outputs/old"
    (old / "run").mkdir(parents=True)
    csv_text = "run_dir,rate\noutputs/old/run,1.23456789\n"
    (old / "rates.csv").write_text(csv_text)
    (old / "experiment.json").write_text(json.dumps({"benchmark_run": "outputs/old/run"}))
    (old / "run/data.bin").write_bytes(b"\x00\xff\x01")
    manifest = tmp_path / "outputs/archive/output_migration_manifest.json"
    monkeypatch.setattr(layout, "ROOT", tmp_path)
    monkeypatch.setattr(layout, "MOVES", {"outputs/old": "outputs/core/new"})
    monkeypatch.setattr(layout, "MANIFEST", manifest)
    assert layout.apply() == 0
    new = tmp_path / "outputs/core/new"
    assert not old.exists()
    assert (new / "rates.csv").read_text() == csv_text.replace("outputs/old", "outputs/core/new")
    assert (new / "run/data.bin").read_bytes() == b"\x00\xff\x01"
    assert len(json.loads(manifest.read_text())["files"]) == 3
    assert layout.check_dependencies() == 0
    with pytest.raises(SystemExit, match="already recorded"):
        layout.apply()
    (new / "run/data.bin").write_bytes(b"changed")
    assert layout.check_dependencies() == 1


def test_missing_dependency_fails(tmp_path, monkeypatch):
    output = tmp_path / "outputs"
    output.mkdir()
    (output / "rates.csv").write_text("run_dir\noutputs/missing\n")
    monkeypatch.setattr(layout, "ROOT", tmp_path)
    monkeypatch.setattr(layout, "MANIFEST", output / "absent.json")
    assert layout.check_dependencies() == 1
