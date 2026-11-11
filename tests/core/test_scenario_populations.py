"""Scenario grids copy each host exactly and do not turn missing work into zero."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


@pytest.fixture
def experiment(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("scenario_populations", scripts / "compare_scenario_populations.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("case,model,gap,refill", [
    ("baseline", "semi_type_i", True, True),
    ("torque_i_thermal", "torque_with_trap", False, True),
    ("torque_gap_viscous", "torque_with_trap", True, True),
    ("no_replenishment", "semi_type_i", True, False),
])
def test_only_scenario_controls_change(experiment, monkeypatch, case, model, gap, refill):
    original = SimpleNamespace(disk_alpha=.1, disk_accretion_rate_g_s=1.44e22,
                               disk_outer_radius_cm=1.586e16, redshift=12., base_seed=20261107)
    monkeypatch.setattr(experiment, "config_from_saved_run", lambda *args: original)
    cfg = experiment.scenario_config(dict(baseline_run="baseline", target="target", case=case), 1)
    assert (cfg.disk_model, cfg.torque_gap_transport, cfg.consumption_replenishment) == (model, gap, refill)
    assert cfg.disk_accretion_rate_g_s == 1.44e22
    assert cfg.disk_outer_radius_cm == 1.586e16
    assert cfg.redshift == 12.
    assert cfg.base_seed == 20261107


def test_reuse_requires_all_repeats(experiment, monkeypatch, tmp_path):
    monkeypatch.setattr(experiment.helpers, "compatible_run", lambda *args: True)
    cfg = SimpleNamespace(num_repeats=3)
    path = tmp_path / "imri_repeat_results.csv"
    pd.DataFrame({"repeat": [0, 1]}).to_csv(path, index=False)
    assert not experiment.complete(tmp_path, cfg)
    pd.DataFrame({"repeat": [0, 1, 2]}).to_csv(path, index=False)
    assert experiment.complete(tmp_path, cfg)


def test_overview_plots_all_four_cases_with_loud_delayed_sources(experiment, tmp_path):
    for case, *_ in experiment.CASES:
        folder = tmp_path / case
        folder.mkdir()
        pd.DataFrame([
            dict(M=1e4, mass_msun=50., snr_observation_redshift=9.9, lisa_snr=8., snr_status="evaluated"),
            dict(M=1e5, mass_msun=100., snr_observation_redshift=1., lisa_snr=150., snr_status="evaluated"),
        ]).to_csv(folder / "candidate_snr.csv", index=False)
    experiment.plot_overview(tmp_path)
    for suffix in ("png", "pdf", "svg"):
        assert (tmp_path / f"snr_population_scenarios.{suffix}").stat().st_size > 0


def test_collect_ignores_pending_or_failed_runs(experiment, monkeypatch, tmp_path):
    monkeypatch.setattr(experiment.helpers, "summarize_run", lambda path: pytest.fail("must not read incomplete runs"))
    plans = [dict(case="baseline", status="pending"), dict(case="torque_i_thermal", status="failed")]
    experiment.collect(plans, tmp_path)
    assert (tmp_path / "scenario_rates.csv").read_text().strip() == ""
