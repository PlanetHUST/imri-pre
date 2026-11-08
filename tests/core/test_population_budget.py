"""Population-budget conservation and diagnostics of current model limits."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from physical_process import constants as const
from physical_process.agn import AGNProperties
from physical_process.damp import DampingCalculator, DampingStatus, OrbitalState
from physical_process.sampling import sample_efficient
from workflows.event_rate import classify_coupled_event

spec = importlib.util.spec_from_file_location("population_budget_audit", Path(__file__).resolve().parents[2] / "scripts/check_population_budget.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_budget_counts_embedded_stars_and_distinguishes_crossings():
    with TemporaryDirectory() as directory:
        path = Path(directory)
        pd.DataFrame([dict(repeat=0, task_seed=1, redshift=10, m_smbh_msun=1e5,
            m_nsc_msun=3e6, duty_cycle_myr=10, n_sampled=1000, n_disk_cross=4,
            n_replenished=1, n_emb_star=2, n_emb_bh=1, n_mfbh=0,
            n_imri=1, n_det=0, rate_imri_per_myr=.1, rate_det_per_myr=0)]).to_csv(path / "imri_repeat_results.csv", index=False)
        pd.DataFrame(dict(stage=["damping"]*5, repeat=[0]*5,
            task_kind=["primary"]*4 + ["replacement"],
            status=["embedded_star", "embedded_star", "embedded_bh", "damping_incomplete", "damping_incomplete"],
            time_myr=[1, 2, 3, 4, 11])).to_csv(path / "imri_source_transitions.csv", index=False)
        pd.DataFrame({"repeat": [0]*3, "event_rate_reason": ["bh_not_formed_by_band_entry"]*2 + ["band_entry"]}).to_csv(path / "imri_events.csv", index=False)
        rows, _, _ = audit.census_run(path)
        row = rows.iloc[0]
        assert row.total_embedded == 3
        assert row.embedded_star == 2
        assert row.initial_crossers == 4
        assert row.damping_rate == .3
        assert row.primary_incomplete_before_deadline == 1
        assert row.geometric_selection_averaged_over_duration == .4
        assert row.crossing_fraction == .004


def test_default_popiii_population_has_no_low_mass_stars():
    previous = np.random.get_state()
    try:
        np.random.seed(123)
        masses, lifetimes = sample_efficient(1e5, 1e-5)
    finally:
        np.random.set_state(previous)
    assert masses.min() >= 13.6
    assert masses.max() <= 197
    assert masses.sum() >= 1e5
    assert masses.sum() < 1e5 + 197
    assert len(masses) < 1e5 / 13.6 + 1
    assert (lifetimes > 0).all()


def test_migration_formed_bh_branch_is_active_when_timing_allows():
    orbit = SimpleNamespace(status="band_entry", entry_time_myr=4,
        disk_off_time_myr=np.nan, disk_off_radius_cm=np.nan,
        disk_off_reason="gas_active_at_entry")
    formed = classify_coupled_event(object_status=DampingStatus.EMBEDDED_STAR,
        formation_time_myr=3, embed_time_myr=1, orbit=orbit)
    unformed = classify_coupled_event(object_status=DampingStatus.EMBEDDED_STAR,
        formation_time_myr=6, embed_time_myr=1, orbit=orbit)
    assert formed.is_imri_candidate is True
    assert formed.bh_origin == "mfbh"
    assert unformed.is_imri_candidate is False
    assert unformed.event_rate_reason == "bh_not_formed_by_band_entry"


def test_diagnostic_exposes_current_postalignment_handoff_gap():
    # This documents the PRESENT limitation, not desired future capture physics:
    # aligned but eccentric objects are returned early as incomplete.
    calculator = DampingCalculator(AGNProperties.from_mass(1e5, 10), verbose=False,
        radius_grid_cm=np.array([1e13, 1e18]),
        surface_density_g_cm2=np.array([1e5, 1e5]),
        aspect_ratio=np.array([.01, .01]), disk_inner_radius_cm=1e13,
        disk_outer_radius_cm=1e18, use_disk_inner_radius=True,
        density_g_cm3=np.array([1e-10, 1e-10]))
    state = OrbitalState(semi_major_axis_cm=1e16, eccentricity=.1,
        inclination_rad=.001, cos_omega=.5, time_s=const.Myr)
    criteria = calculator.embedding_criteria(state)
    assert criteria.low_inclination
    assert not criteria.sufficiently_circular
    assert criteria.radially_contained
    outcome = calculator.damping_simulation(100 * const.msun, state, 3, "Star")
    assert outcome.status is DampingStatus.DAMPING_INCOMPLETE
    assert outcome.final_state.time_s == state.time_s
    assert outcome.elapsed_time_s == 0


def test_stellar_timing_extrapolation_does_not_create_bh_events():
    events = pd.DataFrame([dict(repeat=0, source_id=3, task_kind="primary", copy_number=0,
        event_rate_reason="bh_not_formed_by_band_entry", band_entry_time_myr=1,
        formation_time_myr=3, m_smbh_msun=1e5, mass_msun=100,
        band_entry_radius_cm=2e11)])
    damping = pd.DataFrame([dict(repeat=0, source_id=3, task_kind="primary", copy_number=0, time_myr=.8)])
    result = audit.stellar_timing(events, damping).iloc[0]
    assert result.migration_to_entry_myr == pytest.approx(.2)
    assert result.collapse_after_entry_myr == 2
    assert 0 < result.point_mass_vacuum_isco_delay_myr < 1e-6
    assert result.tidal_radius_over_band_radius > 1


def test_more_capped_candidates_cannot_change_individual_snr():
    controls = audit.critical_secondary_masses()
    assert (controls.snr_at_population_cap < 20).all()
    assert (controls.secondary_mass_for_snr20 > 197).all()
    assert controls.iloc[0].secondary_mass_for_snr20 == pytest.approx(258.46, rel=1e-4)
    assert np.all(np.diff(controls.secondary_mass_for_snr20) > 0)
