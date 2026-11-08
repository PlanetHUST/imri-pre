"""Regression tests for controlled initial-condition sensitivity inputs."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from workflows.cli import build_config_from_arguments, build_parser
from workflows.engine import IMRIDynEngine
from workflows.metadata import build_metadata
from physical_process.sampling import sample_efficient

spec = importlib.util.spec_from_file_location("parameter_experiment", Path(__file__).resolve().parents[2] / "scripts/test_parameter_dependence.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def config(*args):
    return build_config_from_arguments(build_parser().parse_args(["--population-model", "legacy_popiii", *args]))


def test_disk_controls_reach_solver_and_metadata():
    with pytest.MonkeyPatch.context() as monkeypatch:
        _check_disk_controls(monkeypatch)


def _check_disk_controls(monkeypatch):
    cfg = config("--disk-alpha", "0.15", "--disk-accretion-rate-g-s", "1e23", "--disk-outer-radius-cm", "1e17")
    calls = []
    def fake_disk(mass, alpha, **kwargs):
        calls.append((mass, alpha, kwargs))
        array = np.array([1e13, 1e17])
        return SimpleNamespace(inner_radius_cm=1e13, outer_radius_cm=1e17,
            radius_cm=array, surface_density_g_cm2=array, aspect_ratio=array,
            density_g_cm3=array, gravitational_radius_cm=1e11, disk=object())
    monkeypatch.setattr("workflows.engine.build_pagn_disk_profile", fake_disk)
    engine = IMRIDynEngine(cfg)
    assert engine.get_disk_profiles(1e5) is engine.get_disk_profiles(1e5)
    assert calls == [(1e5, 0.15, {"accretion_rate_g_s": 1e23, "outer_radius_cm": 1e17})]
    parameters = build_metadata(cfg, {})["parameters"]
    assert parameters["disk_alpha"] == 0.15
    assert parameters["migration_contract"]["alpha"] == 0.15
    assert parameters["disk_outer_radius_cm"] == 1e17


def test_invalid_controls_rejected():
    for flag, value in [("--disk-alpha", "0"), ("--disk-accretion-rate-g-s", "nan"), ("--disk-outer-radius-cm", "-1"), ("--nsc-radius-pc-at-reference-mass", "0"), ("--metallicity", "nan")]:
        with pytest.raises(ValueError):
            config(flag, value)


def test_design_has_all_ten_parameters_and_measured_phase_plane():
    points = experiment.design()
    assert len(points) == len(set(points)) == 25
    for i in range(10):
        for factor in experiment.FACTORS:
            point = [1.0] * 10
            point[i] = factor
            assert tuple(point) in points
    for alpha in experiment.FACTORS:
        for mdot in experiment.FACTORS:
            point = [1.0] * 10
            point[0], point[2] = alpha, mdot
            assert tuple(point) in points


def test_mass_and_radius_are_independent():
    cfg = config()
    values = dict(zip(experiment.PARAMETERS, (0.1, 1e5, 1.44e23, 1e17, 1.75, 1.77, 197, 1e-5, 1.0, 3e5)))
    experiment.apply_values(cfg, values)
    radius = cfg.nsc_radius_pc_at_reference_mass * (cfg.nsc_masses_msun[0] / cfg.nsc_radius_reference_mass_msun) ** cfg.nsc_radius_mass_slope
    assert radius == pytest.approx(1.0)
    assert cfg.disk_accretion_rate_g_s == 1.44e23


def test_default_metallicity_scan_is_exactly_degenerate():
    samples = []
    for metallicity in (0.5e-5, 1e-5, 1.5e-5):
        np.random.seed(12)
        samples.append(sample_efficient(2e5, metallicity))
    for mass, lifetime in samples[1:]:
        np.testing.assert_array_equal(mass, samples[0][0])
        np.testing.assert_array_equal(lifetime, samples[0][1])


def test_phase_plot_uses_measured_grid_and_handles_unavailable():
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as directory:
        _check_phase_plot(Path(directory))


def _check_phase_plot(tmp_path):
    import json
    import pandas as pd
    rows = []
    for index, point in enumerate(experiment.design()):
        row = {f"factor_{name}": factor for name, factor in zip(experiment.PARAMETERS, point)}
        row.update(model="semi_type_i", rate_imri_per_myr=np.nan if index == 0 else 1.0,
                   intrinsic_rate_sem=0.1)
        rows.append(row)
    (tmp_path / "experiment.json").write_text(json.dumps({
        "baseline": {"M": 1e5, "M_NSC": 2e5, "Z": 1e-5}, "repeats": 3,
    }))
    experiment.plot_results(pd.DataFrame(rows), tmp_path)
    assert (tmp_path / "imri_parameter_phase_space.png").stat().st_size > 1000
    assert (tmp_path / "imri_parameter_phase_space.pdf").stat().st_size > 1000


def test_explicit_radius_bypasses_q_search():
    with pytest.MonkeyPatch.context() as monkeypatch:
        _check_explicit_radius(monkeypatch)


def _check_explicit_radius(monkeypatch):
    from disk_quantity import pagn_disk_adapter as adapter
    class Disk:
        def __init__(self, **kwargs):
            self.Rs, self.Rmin, self.Rmax = 2.0, 5.0, 1e7
        def solve_disk(self, N):
            self.R = np.geomspace(12, self.Rmax, N)
            for name in ("h", "rho", "T", "tauV", "cs", "kappa", "Omega"):
                setattr(self, name, np.ones(N))
            self.Q = np.linspace(2, 0.5, N)
    monkeypatch.setattr(adapter, "_require_pagn", lambda: (SimpleNamespace(SirkoAGN=Disk), None, SimpleNamespace(MSun=1)))
    def forbidden(*args):
        raise AssertionError("explicit boundary must bypass Q search")
    monkeypatch.setattr(adapter, "_solve_sirko_disk_with_q_outer_radius", forbidden)
    result = adapter.build_pagn_disk_profile(1e5, 0.1, n_resolution=20, outer_radius_cm=1e5)
    assert result.outer_radius_cm == 1e5
    assert result.radius_cm[-1] == pytest.approx(1e5)
    assert result.disk.Rout == 1e3
