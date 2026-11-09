"""Small tests of the approved migration/refill comparison controls."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pytest
from physical_process.migration.torque import MigrationParameters, compose_migration_torque_si
from workflows.cli import build_parser, build_config_from_arguments
from workflows.engine import IMRIDynEngine
from workflows.metadata import build_metadata

spec = importlib.util.spec_from_file_location("scenario_experiment", Path(__file__).resolve().parents[2] / "scripts/compare_physical_scenarios.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def config(*args):
    return build_config_from_arguments(build_parser().parse_args(list(args)))


def test_ungapped_torque_keeps_sign_and_does_not_apply_gap_attenuation():
    grav, thermal = np.array([-3., 2., -1.]), np.array([1., -.5, 2.])
    gap, viscous = np.array([1., .1, .001]), np.array([-10., -5., -2.])
    no_gap = compose_migration_torque_si(grav, thermal, gap, viscous, include_gap_transport=False)
    full = compose_migration_torque_si(grav, thermal, gap, viscous)
    np.testing.assert_array_equal(no_gap, grav + thermal)
    assert no_gap[1] > 0  # No abs() or forced inward migration.
    np.testing.assert_allclose(full, gap * grav + thermal + (1-gap) * viscous)
    assert MigrationParameters().include_gap_transport is True
    assert MigrationParameters().extrapolate_thermal_torque is False
    with pytest.raises(TypeError):
        compose_migration_torque_si(grav, thermal, gap, viscous, include_gap_transport="no")


def test_ungapped_thermal_uses_unperturbed_optical_depth_but_keeps_hill_mask():
    from physical_process.migration import torque
    disk = SimpleNamespace(R=np.array([10., 20.]), h=np.array([1., 2.]),
        cs=np.ones(2), Omega=np.array([.1, .05]), rho=np.ones(2), tauV=np.array([1000., 2000.]))
    gap = SimpleNamespace(gap_depth_fraction=np.array([.1, .2]), hill_radius_over_scale_height=np.array([.1, 2.]))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torque, "_require_pagn_bits", lambda: (SimpleNamespace(G=1., c=1.), lambda X: 1.))
        patch.setattr(torque, "thermal_diffusivity_m2_s", lambda *args: np.full(2, .01))
        full = torque.compute_thermal_torque_terms(np.array([-2., -2.]), disk, .001, .001, gap, MigrationParameters())
        no_gap = torque.compute_thermal_torque_terms(np.array([-2., -2.]), disk, .001, .001, gap, MigrationParameters(include_gap_transport=False))
    np.testing.assert_array_equal(full.effective_optical_depth, disk.tauV * gap.gap_depth_fraction)
    np.testing.assert_array_equal(no_gap.effective_optical_depth, disk.tauV)
    assert not full.thermal_torque_regime_valid[1]
    assert not no_gap.thermal_torque_regime_valid[1]


def test_cli_defaults_unchanged_and_comparison_flags_are_independent():
    default = config()
    assert default.disk_model == "semi_type_i"
    assert default.torque_gap_transport is True
    assert default.consumption_replenishment is True
    torque = config("--migration-model", "torque_with_trap", "--torque-no-gap-transport")
    assert torque.disk_model == "torque_with_trap"
    assert torque.torque_gap_transport is False
    assert torque.consumption_replenishment is True
    off = config("--no-consumption-replenishment")
    assert off.disk_model == default.disk_model
    assert off.disk_accretion_rate_g_s == default.disk_accretion_rate_g_s
    assert off.consumption_replenishment is False
    contract = build_metadata(torque, {})["parameters"]["migration_contract"]
    assert contract["torque_composition"] == "Gamma_I+Gamma_th"
    assert contract["gap_transport"] == "disabled"
    assert contract["thermal_extrapolation_enabled"] is False


def test_engine_cache_separates_gap_modes_and_disallows_thermal_extrapolation():
    cfg = config()
    engine = IMRIDynEngine(cfg)
    calls = []
    def fake_profile(disk, **kwargs):
        calls.append(kwargs["params"])
        return SimpleNamespace(mode=kwargs["params"].include_gap_transport)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("workflows.engine.compute_torque_profile", fake_profile)
        data = dict(torque_disk=object(), torque_alpha=.1)
        full = engine.get_torque_profile(data, 1e5, 100)
        cfg.torque_gap_transport = False
        no_gap = engine.get_torque_profile(data, 1e5, 100)
        assert full is not no_gap
        assert engine.get_torque_profile(data, 1e5, 100) is no_gap
        cfg.torque_gap_transport = True
        assert engine.get_torque_profile(data, 1e5, 100) is full
    assert len(calls) == 2
    assert all(not params.extrapolate_thermal_torque for params in calls)


def test_radau_and_default_match_on_nonstiff_entry_and_continuation():
    from physical_process.orbital_evolution import evolve_circular_orbit, continue_circular_band_track
    from physical_process.gw_frequency_radius import gw_radius_for_frequency_cm
    from physical_process.redshift_evolve import RedshiftEvolution
    entry_radius = gw_radius_for_frequency_cm(100100., .002)
    clock = RedshiftEvolution(67.4, .315)
    grid = np.linspace(.5 * entry_radius, 2.5 * entry_radius, 100)
    for speed in (0., -100.):
        common = dict(primary_mass_msun=1e5, secondary_mass_msun=100.,
            initial_radius_cm=2 * entry_radius, initial_time_myr=0., initial_redshift=1.,
            duty_cycle_myr=1e-4, disk_inner_radius_cm=grid[0], gas_radius_cm=grid,
            gas_drift_cm_s=np.full(len(grid), speed), min_observed_frequency_hz=.001,
            redshift_model=clock)
        explicit = evolve_circular_orbit(**common)
        implicit = evolve_circular_orbit(**common, solver_method="Radau")
        assert explicit.status == implicit.status == "band_entry"
        assert implicit.entry_time_myr == pytest.approx(explicit.entry_time_myr, rel=2e-6)
        assert implicit.entry_radius_cm == pytest.approx(explicit.entry_radius_cm, rel=1e-7)
        tracks = [continue_circular_band_track(entry=entry, primary_mass_msun=1e5,
            secondary_mass_msun=100., initial_redshift=1., duty_cycle_myr=1e-4,
            disk_inner_radius_cm=grid[0], gas_radius_cm=grid,
            gas_drift_cm_s=np.full(len(grid), speed), max_observed_frequency_hz=.1,
            redshift_model=clock, solver_method=method)
            for entry, method in ((explicit, "DOP853"), (implicit, "Radau"))]
        assert tracks[0].status == tracks[1].status == "isco"
        assert tracks[0].end_time_myr == pytest.approx(tracks[1].end_time_myr, rel=2e-6)
        assert tracks[0].end_radius_cm == pytest.approx(tracks[1].end_radius_cm, rel=1e-7)


def test_radau_preserves_a_stiff_restoring_trap_and_event_deadline():
    from physical_process import constants as c
    from physical_process.orbital_evolution import evolve_circular_orbit, peters_beta_cm4_s
    from physical_process.redshift_evolve import RedshiftEvolution
    trap = 1e16
    grid = np.linspace(.8 * trap, 1.2 * trap, 501)
    beta = peters_beta_cm4_s(1e5, 100.)
    # Same signed gas+GW law; nearly linear net restoring drift with tau=1 yr.
    gas = -(grid-trap) / c.yr + beta / grid**3
    outcome = evolve_circular_orbit(primary_mass_msun=1e5, secondary_mass_msun=100.,
        initial_radius_cm=1.05 * trap, initial_time_myr=0., initial_redshift=10.,
        duty_cycle_myr=1., disk_inner_radius_cm=grid[0], gas_radius_cm=grid,
        gas_drift_cm_s=gas, min_observed_frequency_hz=.001,
        redshift_model=RedshiftEvolution(67.4, .315), stop_time_myr=.2,
        solver_method="Radau")
    assert outcome.status == "stopped_at_event"
    assert outcome.final_time_myr == pytest.approx(.2)
    assert outcome.final_radius_cm == pytest.approx(trap, rel=1e-7)
    assert not np.isfinite(outcome.entry_time_myr)


def test_solver_policy_changes_only_torque_cases_and_is_saved():
    from workflows.event_work import IMRIEventEvaluator
    baseline = config()
    torque = config("--migration-model", "torque_with_trap")
    assert IMRIEventEvaluator(IMRIDynEngine(baseline)).orbit_solver_method == "DOP853"
    assert IMRIEventEvaluator(IMRIDynEngine(torque)).orbit_solver_method == "Radau"
    assert build_metadata(torque, {})["parameters"]["coupled_orbit_integrator"] == "Radau"
    assert build_metadata(baseline, {})["parameters"]["coupled_orbit_integrator"] == "DOP853"


def test_benchmark_loader_preserves_saved_physics_and_handles_json_infinity():
    cfg = config("3", "unused", "1", "--smbh-masses", "1e5", "--nsc-masses", "3e6", "--redshift", "10", "--base-seed", "20261107", "--disk-accretion-rate-g-s", "1.44e23", "--disk-outer-radius-cm", "9.953e15")
    metadata = build_metadata(cfg, {})
    with TemporaryDirectory() as directory:
        path = Path(directory)
        (path / "simulation_metadata.json").write_text(json.dumps(metadata))
        loaded = experiment.config_from_benchmark(path, path / "new_run", 6)
    assert loaded.disk_accretion_rate_g_s == cfg.disk_accretion_rate_g_s
    assert loaded.disk_outer_radius_cm == cfg.disk_outer_radius_cm
    assert loaded.initial_stellar_density_gamma == cfg.initial_stellar_density_gamma
    assert loaded.base_seed == cfg.base_seed
    assert loaded.num_repeats == 3
    assert loaded.cosmic_merge_time_myr == np.inf
    assert loaded.damping_cpu_count == 6
