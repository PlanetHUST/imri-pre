"""Explicit IMF bounds, fixed total mass, and cache compatibility."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from physical_process.sampling import sample_efficient
from workflows.cli import build_parser, build_config_from_arguments
from workflows.metadata import build_metadata
from workflows.population import sample_disk_crossing_population

spec = importlib.util.spec_from_file_location("imf_host_comparison", Path(__file__).resolve().parents[2] / "scripts/compare_host_rates.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def config(*args):
    return build_config_from_arguments(build_parser().parse_args(["--population-model", "legacy_popiii", *args]))


def test_explicit_default_bounds_preserve_seeded_population_exactly():
    state = np.random.get_state()
    try:
        np.random.seed(14)
        original = sample_efficient(1e5, 1e-5)
        np.random.seed(14)
        explicit = sample_efficient(1e5, 1e-5, popiii_min_mass_msun=13.6, popiii_max_mass_msun=197)
    finally:
        np.random.set_state(state)
    for actual, expected in zip(explicit, original):
        np.testing.assert_array_equal(actual, expected)


def test_one_to_four_hundred_is_mass_normalized_not_number_normalized():
    state = np.random.get_state()
    try:
        np.random.seed(24)
        broad, lifetimes = sample_efficient(1e5, 1e-5, popiii_min_mass_msun=1, popiii_max_mass_msun=400)
        np.random.seed(24)
        reference, _ = sample_efficient(1e5, 1e-5)
    finally:
        np.random.set_state(state)
    assert broad.min() >= 1
    assert broad.max() <= 400
    assert (broad < 13.6).any()
    assert (broad > 197).any()
    assert 1e5 <= broad.sum() < 1e5 + 400
    assert len(broad) > len(reference)
    assert broad.mean() < reference.mean()
    assert (np.isfinite(lifetimes) & (lifetimes > 0)).all()


def test_bounds_reach_population_sampler_and_metadata():
    cfg = config("--popiii-min-mass-msun", "1", "--popiii-max-mass-msun", "400")
    calls = []
    def fake_sample(mass, **kwargs):
        calls.append(kwargs)
        return np.array([1., 200., 400.]), np.array([1e10, 3e6, 2e6])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("workflows.population.stellar_sampling.sample_efficient", fake_sample)
        population = sample_disk_crossing_population(1e5, 1e3, {"inner_radius_cm": 1e12, "outer_radius_cm": 1e16}, cfg)
    assert calls[0]["popiii_min_mass_msun"] == 1
    assert calls[0]["popiii_max_mass_msun"] == 400
    np.testing.assert_array_equal(population.initial["mass_msun"], [1, 200, 400])
    parameters = build_metadata(cfg, {})["parameters"]
    assert parameters["popiii_min_mass_msun"] == 1
    assert parameters["popiii_max_mass_msun"] == 400


def test_invalid_imf_bounds_rejected():
    for low, high in ((0, 400), (-1, 400), (1, 1), (2, 1), (float("nan"), 400), (1, float("inf"))):
        with pytest.raises(ValueError):
            config("--popiii-min-mass-msun", str(low), "--popiii-max-mass-msun", str(high))


def test_analytic_mass_budget_separates_upper_tail_from_low_mass_dilution():
    report_spec = importlib.util.spec_from_file_location("imf_comparison_report", Path(__file__).resolve().parents[2] / "scripts/report_imf_comparison.py")
    report = importlib.util.module_from_spec(report_spec)
    report_spec.loader.exec_module(report)
    broad = report.imf_expectation(1, 400)
    upper = report.imf_expectation(13.6, 400)
    default = report.imf_expectation(13.6, 197)
    assert broad["mean_mass_msun"] == pytest.approx(10.032858188)
    assert broad["expected_objects"] > upper["expected_objects"]
    assert broad["expected_objects_above_258_msun"] < upper["expected_objects_above_258_msun"]
    assert default["expected_objects_above_258_msun"] == 0
    assert broad["mean_mass_msun"] * broad["expected_objects"] == pytest.approx(3e6)


def test_legacy_default_cache_cannot_be_reused_for_a_different_lower_bound():
    cfg = config()
    metadata = build_metadata(cfg, {})
    metadata["parameters"].pop("popiii_min_mass_msun")
    with TemporaryDirectory() as directory:
        path = Path(directory)
        (path / "simulation_metadata.json").write_text(json.dumps(metadata))
        for filename in ("imri_repeat_results.csv", "imri_source_transitions.csv", "imri_events.csv"):
            (path / filename).write_text("dummy\n")
        assert comparison.compatible_run(path, cfg)
        cfg.popiii_min_mass_msun = 1
        assert not comparison.compatible_run(path, cfg)
