"""Analytical limits and physical events of the combined circular orbit."""

from unittest.mock import patch

import numpy as np

from physical_process import constants as const
from physical_process.gw_frequency_radius import gw_radius_for_frequency_cm
from physical_process.gw_frequency_radius import circular_band_boundary
from physical_process.orbital_evolution import (
    continue_circular_band_track,
    evolve_circular_orbit,
    peters_beta_cm4_s,
)
from physical_process.redshift_evolve import RedshiftEvolution
from physical_process.damp import DampingStatus
from workflows.event_rate import classify_coupled_event
from lisa.detectability import classify_coupled_entry, classify_detectability
from lisa.snr import evaluate_lisa_coupled_track_snr, evaluate_lisa_inspiral_snr
from workflows.cli import build_config_from_arguments, build_parser

PRIMARY_MASS_MSUN = 1.0e5
SECONDARY_MASS_MSUN = 100.0
INITIAL_REDSHIFT = 1.0
MIN_FREQUENCY_HZ = 1.0e-3
ENTRY_RADIUS_CM = gw_radius_for_frequency_cm(
    PRIMARY_MASS_MSUN + SECONDARY_MASS_MSUN,
    MIN_FREQUENCY_HZ * (1.0 + INITIAL_REDSHIFT),
)
INITIAL_RADIUS_CM = 2.0 * ENTRY_RADIUS_CM
CLOCK = RedshiftEvolution(67.4, 0.315)


def _orbit(gas_speed_cm_s: float, duty_cycle_myr: float = 1.0e-4):
    return evolve_circular_orbit(
        primary_mass_msun=PRIMARY_MASS_MSUN,
        secondary_mass_msun=SECONDARY_MASS_MSUN,
        initial_radius_cm=INITIAL_RADIUS_CM,
        initial_time_myr=0.0,
        initial_redshift=INITIAL_REDSHIFT,
        duty_cycle_myr=duty_cycle_myr,
        disk_inner_radius_cm=0.5 * ENTRY_RADIUS_CM,
        gas_radius_cm=np.linspace(
            0.5 * ENTRY_RADIUS_CM, 1.2 * INITIAL_RADIUS_CM, 200
        ),
        gas_drift_cm_s=np.full(200, gas_speed_cm_s),
        min_observed_frequency_hz=MIN_FREQUENCY_HZ,
        redshift_model=CLOCK,
    )


def test_candidate_eligibility_preserves_valid_imri_snr() -> None:
    orbit = _orbit(0.0, duty_cycle_myr=1.0e-8)
    decision = classify_coupled_entry(
        orbit=orbit,
        bh_candidate=True,
        is_imri_candidate=True,
        primary_mass_msun=PRIMARY_MASS_MSUN,
        secondary_mass_msun=SECONDARY_MASS_MSUN,
        min_observed_frequency_hz=MIN_FREQUENCY_HZ,
        max_observed_frequency_hz=0.1,
        snr_threshold=20.0,
        mission_duration_years=4.0,
        detector_hubble_constant_km_s_mpc=67.4,
        detector_omega_m=0.315,
    )

    assert decision.detect_rate_class == "imri data"
    assert decision.is_lisa_detectable is True
    assert decision.snr_status == "evaluated"
    assert decision.detect_rate_reason == "lisa_snr_above_threshold"
    assert np.isclose(decision.lisa_snr, 58.07069349420182, rtol=2.0e-10)


def test_candidate_eligibility_excludes_false_before_snr_in_both_paths() -> None:
    orbit = _orbit(0.0, duty_cycle_myr=1.0e-8)
    coupled_kwargs = dict(
        orbit=orbit,
        bh_candidate=True,
        is_imri_candidate=False,
        primary_mass_msun=PRIMARY_MASS_MSUN,
        secondary_mass_msun=SECONDARY_MASS_MSUN,
        min_observed_frequency_hz=MIN_FREQUENCY_HZ,
        max_observed_frequency_hz=0.1,
        snr_threshold=20.0,
        mission_duration_years=4.0,
        detector_hubble_constant_km_s_mpc=67.4,
        detector_omega_m=0.315,
    )
    legacy_kwargs = dict(
        is_imri_candidate=False,
        primary_mass_msun=PRIMARY_MASS_MSUN,
        secondary_mass_msun=SECONDARY_MASS_MSUN,
        r_gas_fin_cm=orbit.entry_radius_cm,
        gas_final_time_myr=0.0,
        initial_redshift=INITIAL_REDSHIFT,
        min_observed_frequency_hz=MIN_FREQUENCY_HZ,
        max_observed_frequency_hz=0.1,
        eccentricity=0.0,
        cosmic_merge_time_myr=np.inf,
        cosmic_merge_smbh_mass_factor=2.0,
        cosmic_merge_radius_factor=2.0,
        snr_threshold=20.0,
        mission_duration_years=4.0,
        detector_hubble_constant_km_s_mpc=67.4,
        detector_omega_m=0.315,
    )
    with patch(
        "lisa.detectability.evaluate_lisa_inspiral_snr",
        side_effect=AssertionError("excluded candidate reached SNR evaluation"),
    ) as snr_evaluator:
        coupled = classify_coupled_entry(**coupled_kwargs)
        legacy = classify_detectability(**legacy_kwargs)

    snr_evaluator.assert_not_called()
    for decision in (coupled, legacy):
        assert decision.detect_rate_class == "not evaluated"
        assert decision.is_lisa_detectable is None
        assert decision.snr_status == "not_evaluated"
        assert decision.detect_rate_reason == "not_imri_candidate"
        assert np.isnan(decision.lisa_snr)
        assert np.isinf(decision.detection_time_myr)


def test_candidate_eligibility_preserves_unavailable_status_in_both_paths() -> None:
    orbit = _orbit(0.0, duty_cycle_myr=1.0e-8)
    coupled_kwargs = dict(
        orbit=orbit,
        bh_candidate=True,
        is_imri_candidate=None,
        primary_mass_msun=PRIMARY_MASS_MSUN,
        secondary_mass_msun=SECONDARY_MASS_MSUN,
        min_observed_frequency_hz=MIN_FREQUENCY_HZ,
        max_observed_frequency_hz=0.1,
        snr_threshold=20.0,
        mission_duration_years=4.0,
        detector_hubble_constant_km_s_mpc=67.4,
        detector_omega_m=0.315,
    )
    legacy_kwargs = dict(
        is_imri_candidate=None,
        primary_mass_msun=PRIMARY_MASS_MSUN,
        secondary_mass_msun=SECONDARY_MASS_MSUN,
        r_gas_fin_cm=orbit.entry_radius_cm,
        gas_final_time_myr=0.0,
        initial_redshift=INITIAL_REDSHIFT,
        min_observed_frequency_hz=MIN_FREQUENCY_HZ,
        max_observed_frequency_hz=0.1,
        eccentricity=0.0,
        cosmic_merge_time_myr=np.inf,
        cosmic_merge_smbh_mass_factor=2.0,
        cosmic_merge_radius_factor=2.0,
        snr_threshold=20.0,
        mission_duration_years=4.0,
        detector_hubble_constant_km_s_mpc=67.4,
        detector_omega_m=0.315,
    )
    with patch(
        "lisa.detectability.evaluate_lisa_inspiral_snr",
        side_effect=AssertionError("unavailable candidate reached SNR evaluation"),
    ) as snr_evaluator:
        coupled = classify_coupled_entry(**coupled_kwargs)
        legacy = classify_detectability(**legacy_kwargs)

    snr_evaluator.assert_not_called()
    for decision in (coupled, legacy):
        assert decision.detect_rate_class == "not evaluated"
        assert decision.is_lisa_detectable is None
        assert decision.snr_status == "not_evaluated"
        assert decision.detect_rate_reason == "band_entry_inside_isco"
        assert np.isnan(decision.lisa_snr)
        assert np.isnan(decision.detection_time_myr)


def test_vacuum_limit_obeys_peters_radius_and_moving_band() -> None:
    outcome = _orbit(0.0)
    assert outcome.status == "band_entry"
    beta = peters_beta_cm4_s(PRIMARY_MASS_MSUN, SECONDARY_MASS_MSUN)
    expected_myr = (
        INITIAL_RADIUS_CM**4 - outcome.entry_radius_cm**4
    ) / (4.0 * beta * const.Myr)
    assert np.isclose(outcome.entry_time_myr, expected_myr, rtol=2.0e-7)
    observed_frequency_hz = (
        const.G
        * (PRIMARY_MASS_MSUN + SECONDARY_MASS_MSUN)
        * const.msun
        / outcome.entry_radius_cm**3
    ) ** 0.5 / (np.pi * (1.0 + outcome.entry_redshift))
    assert np.isclose(observed_frequency_hz, MIN_FREQUENCY_HZ, rtol=1.0e-8)


def test_collapse_stop_and_same_mass_restart_recover_direct_orbit() -> None:
    """An inserted event boundary must not itself change the orbit."""
    radius_grid = np.linspace(0.5 * ENTRY_RADIUS_CM, 1.2 * INITIAL_RADIUS_CM, 200)
    drift = np.full(radius_grid.size, -1.0e3)
    common = dict(
        primary_mass_msun=PRIMARY_MASS_MSUN,
        secondary_mass_msun=SECONDARY_MASS_MSUN,
        initial_redshift=INITIAL_REDSHIFT,
        duty_cycle_myr=1.0e-4,
        disk_inner_radius_cm=radius_grid[0],
        gas_radius_cm=radius_grid,
        gas_drift_cm_s=drift,
        min_observed_frequency_hz=MIN_FREQUENCY_HZ,
        redshift_model=CLOCK,
    )
    direct = evolve_circular_orbit(
        **common, initial_radius_cm=INITIAL_RADIUS_CM, initial_time_myr=0.0
    )
    stopped = evolve_circular_orbit(
        **common, initial_radius_cm=INITIAL_RADIUS_CM, initial_time_myr=0.0,
        stop_time_myr=1.0e-6,
    )
    assert stopped.status == "stopped_at_event"
    assert stopped.final_time_myr == 1.0e-6
    resumed = evolve_circular_orbit(
        **common, initial_radius_cm=stopped.final_radius_cm,
        initial_time_myr=stopped.final_time_myr,
    )
    assert direct.status == resumed.status == "band_entry"
    assert np.isclose(resumed.entry_time_myr, direct.entry_time_myr, rtol=2.0e-6)
    assert np.isclose(resumed.entry_radius_cm, direct.entry_radius_cm, rtol=2.0e-6)


def test_aiding_and_opposing_gas_change_the_same_orbit() -> None:
    vacuum = _orbit(0.0)
    aiding = _orbit(-1.0e3)
    opposing = _orbit(1.0e3)
    assert aiding.entry_time_myr < vacuum.entry_time_myr
    assert opposing.entry_time_myr > vacuum.entry_time_myr
    assert aiding.gas_active_at_entry is True


def test_net_torque_balance_stalls_until_disk_shuts_off() -> None:
    beta = peters_beta_cm4_s(PRIMARY_MASS_MSUN, SECONDARY_MASS_MSUN)
    balancing_gas_speed_cm_s = beta / INITIAL_RADIUS_CM**3
    outcome = _orbit(balancing_gas_speed_cm_s, duty_cycle_myr=5.0e-5)
    assert outcome.status == "band_entry"
    assert outcome.gas_active_at_entry is False
    assert outcome.disk_off_reason == "agn_duty_end"
    assert np.isclose(outcome.disk_off_time_myr, 5.0e-5)
    assert np.isclose(outcome.disk_off_radius_cm, INITIAL_RADIUS_CM, rtol=1.0e-8)
    assert outcome.entry_time_myr > outcome.disk_off_time_myr


def test_stable_net_torque_equilibrium_is_not_a_handoff() -> None:
    beta = peters_beta_cm4_s(PRIMARY_MASS_MSUN, SECONDARY_MASS_MSUN)
    equilibrium_cm = 1.5 * ENTRY_RADIUS_CM
    grid_cm = np.linspace(0.5 * ENTRY_RADIUS_CM, 1.2 * INITIAL_RADIUS_CM, 401)
    relaxation_time_myr = 1.0e-6
    gas_drift_cm_s = (
        beta / grid_cm**3
        - (grid_cm - equilibrium_cm) / (relaxation_time_myr * const.Myr)
    )
    outcome = evolve_circular_orbit(
        primary_mass_msun=PRIMARY_MASS_MSUN,
        secondary_mass_msun=SECONDARY_MASS_MSUN,
        initial_radius_cm=INITIAL_RADIUS_CM,
        initial_time_myr=0.0,
        initial_redshift=INITIAL_REDSHIFT,
        duty_cycle_myr=1.0e-4,
        disk_inner_radius_cm=grid_cm[0],
        gas_radius_cm=grid_cm,
        gas_drift_cm_s=gas_drift_cm_s,
        min_observed_frequency_hz=MIN_FREQUENCY_HZ,
        redshift_model=CLOCK,
    )
    assert outcome.status == "band_entry"
    assert outcome.gas_active_at_entry is False
    assert outcome.disk_off_reason == "agn_duty_end"
    assert np.isclose(outcome.disk_off_radius_cm, equilibrium_cm, rtol=1.0e-6)
    assert np.isclose(outcome.disk_off_time_myr, 1.0e-4)


def test_profile_resolution_converges_for_smooth_signed_drift() -> None:
    def solve(samples: int) -> float:
        grid_cm = np.linspace(0.5 * ENTRY_RADIUS_CM, 1.2 * INITIAL_RADIUS_CM, samples)
        outcome = evolve_circular_orbit(
            primary_mass_msun=PRIMARY_MASS_MSUN,
            secondary_mass_msun=SECONDARY_MASS_MSUN,
            initial_radius_cm=INITIAL_RADIUS_CM,
            initial_time_myr=0.0,
            initial_redshift=INITIAL_REDSHIFT,
            duty_cycle_myr=1.0e-3,
            disk_inner_radius_cm=grid_cm[0],
            gas_radius_cm=grid_cm,
            gas_drift_cm_s=-300.0 * np.sqrt(grid_cm / ENTRY_RADIUS_CM),
            min_observed_frequency_hz=MIN_FREQUENCY_HZ,
            redshift_model=CLOCK,
        )
        assert outcome.status == "band_entry"
        return outcome.entry_time_myr

    assert np.isclose(solve(81), solve(641), rtol=2.0e-5)


def test_circular_energy_and_angular_momentum_fluxes_are_consistent() -> None:
    primary_g = PRIMARY_MASS_MSUN * const.msun
    secondary_g = SECONDARY_MASS_MSUN * const.msun
    total_g = primary_g + secondary_g
    reduced_g = primary_g * secondary_g / total_g
    radius_cm = INITIAL_RADIUS_CM
    d_energy_dr = const.G * primary_g * secondary_g / (2.0 * radius_cm**2)
    d_angular_momentum_dr = (
        0.5 * reduced_g * np.sqrt(const.G * total_g / radius_cm)
    )
    omega_s_inverse = np.sqrt(const.G * total_g / radius_cm**3)
    assert np.isclose(d_energy_dr / d_angular_momentum_dr, omega_s_inverse)
    gw_radius_drift = -peters_beta_cm4_s(
        PRIMARY_MASS_MSUN, SECONDARY_MASS_MSUN
    ) / radius_cm**3
    assert np.isclose(
        d_energy_dr * gw_radius_drift,
        omega_s_inverse * d_angular_momentum_dr * gw_radius_drift,
    )


def test_vacuum_object_entering_outer_disk_activates_gas() -> None:
    outer_cm = 1.2 * ENTRY_RADIUS_CM
    grid_cm = np.linspace(0.5 * ENTRY_RADIUS_CM, outer_cm, 200)
    outcome = evolve_circular_orbit(
        primary_mass_msun=PRIMARY_MASS_MSUN,
        secondary_mass_msun=SECONDARY_MASS_MSUN,
        initial_radius_cm=INITIAL_RADIUS_CM,
        initial_time_myr=0.0,
        initial_redshift=INITIAL_REDSHIFT,
        duty_cycle_myr=1.0e-3,
        disk_inner_radius_cm=grid_cm[0],
        gas_radius_cm=grid_cm,
        gas_drift_cm_s=np.full(grid_cm.size, -1.0e3),
        min_observed_frequency_hz=MIN_FREQUENCY_HZ,
        redshift_model=CLOCK,
    )
    vacuum = _orbit(0.0, duty_cycle_myr=1.0e-3)
    assert outcome.status == "band_entry"
    assert outcome.gas_active_at_entry is True
    assert outcome.entry_time_myr < vacuum.entry_time_myr


def test_inward_local_timescale_limit_matches_log_radius_integral() -> None:
    tiny_secondary_msun = 1.0e-10  # suppress GW solely for this analytic limit
    local_time_myr = 1.0e-4
    grid_cm = np.linspace(0.5 * ENTRY_RADIUS_CM, 1.2 * INITIAL_RADIUS_CM, 200)
    outcome = evolve_circular_orbit(
        primary_mass_msun=PRIMARY_MASS_MSUN,
        secondary_mass_msun=tiny_secondary_msun,
        initial_radius_cm=INITIAL_RADIUS_CM,
        initial_time_myr=0.0,
        initial_redshift=INITIAL_REDSHIFT,
        duty_cycle_myr=1.0,
        disk_inner_radius_cm=grid_cm[0],
        gas_radius_cm=grid_cm,
        gas_drift_cm_s=-grid_cm / (local_time_myr * const.Myr),
        min_observed_frequency_hz=MIN_FREQUENCY_HZ,
        redshift_model=CLOCK,
    )
    assert outcome.status == "band_entry"
    expected_myr = local_time_myr * np.log(
        INITIAL_RADIUS_CM / outcome.entry_radius_cm
    )
    assert np.isclose(outcome.entry_time_myr, expected_myr, rtol=3.0e-6)


def test_inner_and_outward_profile_exits_continue_in_vacuum() -> None:
    inner_cm = 1.3 * ENTRY_RADIUS_CM
    inward_grid_cm = np.linspace(inner_cm, 1.2 * INITIAL_RADIUS_CM, 200)
    inward = evolve_circular_orbit(
        primary_mass_msun=PRIMARY_MASS_MSUN,
        secondary_mass_msun=SECONDARY_MASS_MSUN,
        initial_radius_cm=INITIAL_RADIUS_CM,
        initial_time_myr=0.0,
        initial_redshift=INITIAL_REDSHIFT,
        duty_cycle_myr=1.0e-3,
        disk_inner_radius_cm=inner_cm,
        gas_radius_cm=inward_grid_cm,
        gas_drift_cm_s=np.full(inward_grid_cm.size, -1.0e3),
        min_observed_frequency_hz=MIN_FREQUENCY_HZ,
        redshift_model=CLOCK,
    )
    assert inward.status == "band_entry"
    assert inward.gas_active_at_entry is False
    assert inward.disk_off_reason == "inner_profile_edge"
    assert np.isclose(inward.disk_off_radius_cm, inner_cm, rtol=1.0e-8)
    assert inward.entry_time_myr > inward.disk_off_time_myr

    outer_cm = 1.2 * INITIAL_RADIUS_CM
    outward_grid_cm = np.linspace(0.5 * ENTRY_RADIUS_CM, outer_cm, 200)
    outward = evolve_circular_orbit(
        primary_mass_msun=PRIMARY_MASS_MSUN,
        secondary_mass_msun=SECONDARY_MASS_MSUN,
        initial_radius_cm=INITIAL_RADIUS_CM,
        initial_time_myr=0.0,
        initial_redshift=INITIAL_REDSHIFT,
        duty_cycle_myr=1.0e-3,
        disk_inner_radius_cm=outward_grid_cm[0],
        gas_radius_cm=outward_grid_cm,
        gas_drift_cm_s=np.full(outward_grid_cm.size, 1.0e3),
        min_observed_frequency_hz=MIN_FREQUENCY_HZ,
        redshift_model=CLOCK,
    )
    assert outward.status == "band_entry"
    assert outward.gas_active_at_entry is False
    assert outward.disk_off_reason == "outer_profile_edge"
    assert np.isclose(outward.disk_off_radius_cm, outer_cm, rtol=1.0e-8)
    assert outward.entry_time_myr > outward.disk_off_time_myr


def test_classification_uses_one_entry_and_evaluates_gas_active_snr() -> None:
    active = _orbit(-1.0e3, duty_cycle_myr=1.0e-3)
    gas_grid_cm = np.linspace(
        0.5 * ENTRY_RADIUS_CM, 1.2 * INITIAL_RADIUS_CM, 200
    )
    band_track = continue_circular_band_track(
        entry=active,
        primary_mass_msun=PRIMARY_MASS_MSUN,
        secondary_mass_msun=SECONDARY_MASS_MSUN,
        initial_redshift=INITIAL_REDSHIFT,
        duty_cycle_myr=1.0e-3,
        disk_inner_radius_cm=gas_grid_cm[0],
        gas_radius_cm=gas_grid_cm,
        gas_drift_cm_s=np.full(gas_grid_cm.size, -1.0e3),
        max_observed_frequency_hz=0.1,
        redshift_model=CLOCK,
    )
    before_formation = classify_coupled_event(
        object_status=DampingStatus.EMBEDDED_STAR,
        formation_time_myr=active.entry_time_myr + 0.1,
        embed_time_myr=0.0,
        orbit=active,
    )
    assert before_formation.is_imri_candidate is False
    embedded_bh = classify_coupled_event(
        object_status=DampingStatus.EMBEDDED_BH,
        formation_time_myr=np.inf,
        embed_time_myr=0.0,
        orbit=active,
    )
    assert embedded_bh.is_imri_candidate is True
    decision = classify_coupled_entry(
        orbit=active,
        bh_candidate=embedded_bh.bh_candidate,
        is_imri_candidate=embedded_bh.is_imri_candidate,
        primary_mass_msun=PRIMARY_MASS_MSUN,
        secondary_mass_msun=SECONDARY_MASS_MSUN,
        min_observed_frequency_hz=MIN_FREQUENCY_HZ,
        max_observed_frequency_hz=0.1,
        snr_threshold=20.0,
        mission_duration_years=4.0,
        detector_hubble_constant_km_s_mpc=67.4,
        detector_omega_m=0.315,
        band_track=band_track,
    )
    assert decision.detection_time_myr == active.entry_time_myr
    assert decision.detection_redshift == active.entry_redshift
    assert decision.snr_status == "evaluated"
    assert np.isfinite(decision.lisa_snr)
    assert decision.is_lisa_detectable is not None

    vacuum_entry = _orbit(0.0, duty_cycle_myr=1.0e-8)
    vacuum_decision = classify_coupled_entry(
        orbit=vacuum_entry,
        bh_candidate=True,
        is_imri_candidate=True,
        primary_mass_msun=PRIMARY_MASS_MSUN,
        secondary_mass_msun=SECONDARY_MASS_MSUN,
        min_observed_frequency_hz=MIN_FREQUENCY_HZ,
        max_observed_frequency_hz=0.1,
        snr_threshold=20.0,
        mission_duration_years=4.0,
        detector_hubble_constant_km_s_mpc=67.4,
        detector_omega_m=0.315,
    )
    assert vacuum_entry.gas_active_at_entry is False
    assert vacuum_decision.detection_time_myr == vacuum_entry.entry_time_myr
    assert vacuum_decision.snr_status == "evaluated"
    assert np.isfinite(vacuum_decision.lisa_snr)
    assert decision.lisa_snr < vacuum_decision.lisa_snr


def test_coupled_track_snr_reduces_to_vacuum_result_when_gas_drift_is_zero() -> None:
    entry = _orbit(0.0, duty_cycle_myr=1.0e-3)
    grid_cm = np.linspace(0.5 * ENTRY_RADIUS_CM, 1.2 * INITIAL_RADIUS_CM, 200)
    track = continue_circular_band_track(
        entry=entry,
        primary_mass_msun=PRIMARY_MASS_MSUN,
        secondary_mass_msun=SECONDARY_MASS_MSUN,
        initial_redshift=INITIAL_REDSHIFT,
        duty_cycle_myr=1.0e-3,
        disk_inner_radius_cm=grid_cm[0],
        gas_radius_cm=grid_cm,
        gas_drift_cm_s=np.zeros(grid_cm.size),
        max_observed_frequency_hz=0.1,
        redshift_model=CLOCK,
    )
    coupled = evaluate_lisa_coupled_track_snr(
        track=track,
        m1_source_msun=PRIMARY_MASS_MSUN,
        m2_source_msun=SECONDARY_MASS_MSUN,
        mission_duration_years=4.0,
        detector_min_frequency_hz=MIN_FREQUENCY_HZ,
        detector_max_frequency_hz=0.1,
        snr_threshold=20.0,
        window_placement="optimal",
    )
    vacuum = evaluate_lisa_inspiral_snr(
        m1_source_msun=PRIMARY_MASS_MSUN,
        m2_source_msun=SECONDARY_MASS_MSUN,
        redshift=entry.entry_redshift,
        current_observed_frequency_hz=(
            MIN_FREQUENCY_HZ
        ),
        mission_duration_years=4.0,
        detector_min_frequency_hz=MIN_FREQUENCY_HZ,
        detector_max_frequency_hz=0.1,
        snr_threshold=20.0,
        window_placement="optimal",
    )
    assert coupled.status == vacuum.status == "evaluated"
    assert np.isclose(coupled.lisa_snr, vacuum.lisa_snr, rtol=2.0e-3)

    coupled_entry_window = evaluate_lisa_coupled_track_snr(
        track=track,
        m1_source_msun=PRIMARY_MASS_MSUN,
        m2_source_msun=SECONDARY_MASS_MSUN,
        mission_duration_years=4.0,
        detector_min_frequency_hz=MIN_FREQUENCY_HZ,
        detector_max_frequency_hz=0.1,
        snr_threshold=20.0,
        window_placement="band_entry",
    )
    vacuum_entry_window = evaluate_lisa_inspiral_snr(
        m1_source_msun=PRIMARY_MASS_MSUN,
        m2_source_msun=SECONDARY_MASS_MSUN,
        redshift=entry.entry_redshift,
        current_observed_frequency_hz=MIN_FREQUENCY_HZ,
        mission_duration_years=4.0,
        detector_min_frequency_hz=MIN_FREQUENCY_HZ,
        detector_max_frequency_hz=0.1,
        snr_threshold=20.0,
        window_placement="band_entry",
    )
    assert coupled_entry_window.status == vacuum_entry_window.status == "evaluated"
    assert np.isclose(
        coupled_entry_window.lisa_snr,
        vacuum_entry_window.lisa_snr,
        rtol=2.0e-3,
    )


def test_gas_active_track_remains_finite_through_disk_shutdown() -> None:
    entry = _orbit(-200.0, duty_cycle_myr=1.0e-3)
    grid_cm = np.linspace(0.5 * ENTRY_RADIUS_CM, 1.2 * INITIAL_RADIUS_CM, 200)
    disk_off_myr = entry.entry_time_myr + 1.0e-8
    track = continue_circular_band_track(
        entry=entry,
        primary_mass_msun=PRIMARY_MASS_MSUN,
        secondary_mass_msun=SECONDARY_MASS_MSUN,
        initial_redshift=INITIAL_REDSHIFT,
        duty_cycle_myr=disk_off_myr,
        disk_inner_radius_cm=grid_cm[0],
        gas_radius_cm=grid_cm,
        gas_drift_cm_s=np.full(grid_cm.size, -200.0),
        max_observed_frequency_hz=0.1,
        redshift_model=CLOCK,
    )
    assert track.gas_log_radius_solution is not None
    assert np.isclose(track.gas_end_time_myr, disk_off_myr)
    assert np.isclose(
        track.radius_cm_at(track.gas_end_time_myr),
        track.gas_end_radius_cm,
        rtol=1.0e-9,
    )
    result = evaluate_lisa_coupled_track_snr(
        track=track,
        m1_source_msun=PRIMARY_MASS_MSUN,
        m2_source_msun=SECONDARY_MASS_MSUN,
        mission_duration_years=4.0,
        detector_min_frequency_hz=MIN_FREQUENCY_HZ,
        detector_max_frequency_hz=0.1,
        snr_threshold=20.0,
        window_placement="band_entry",
    )
    assert result.status == "evaluated"
    assert np.isfinite(result.lisa_snr) and result.lisa_snr > 0.0


def test_balanced_in_band_chirp_has_finite_first_order_snr() -> None:
    radius_cm = 0.99 * ENTRY_RADIUS_CM
    beta = peters_beta_cm4_s(PRIMARY_MASS_MSUN, SECONDARY_MASS_MSUN)
    grid_cm = np.unique(
        np.concatenate(
            (np.linspace(0.5 * radius_cm, radius_cm, 64),
             np.linspace(radius_cm, 2.0 * radius_cm, 64))
        )
    )
    gas_drift_cm_s = beta / grid_cm**3
    entry = evolve_circular_orbit(
        primary_mass_msun=PRIMARY_MASS_MSUN,
        secondary_mass_msun=SECONDARY_MASS_MSUN,
        initial_radius_cm=radius_cm,
        initial_time_myr=0.0,
        initial_redshift=INITIAL_REDSHIFT,
        duty_cycle_myr=1.0e-3,
        disk_inner_radius_cm=grid_cm[0],
        gas_radius_cm=grid_cm,
        gas_drift_cm_s=gas_drift_cm_s,
        min_observed_frequency_hz=MIN_FREQUENCY_HZ,
        redshift_model=CLOCK,
    )
    assert entry.status == "already_inside_band"
    track = continue_circular_band_track(
        entry=entry,
        primary_mass_msun=PRIMARY_MASS_MSUN,
        secondary_mass_msun=SECONDARY_MASS_MSUN,
        initial_redshift=INITIAL_REDSHIFT,
        duty_cycle_myr=1.0e-3,
        disk_inner_radius_cm=grid_cm[0],
        gas_radius_cm=grid_cm,
        gas_drift_cm_s=gas_drift_cm_s,
        max_observed_frequency_hz=0.1,
        redshift_model=CLOCK,
    )
    result = evaluate_lisa_coupled_track_snr(
        track=track,
        m1_source_msun=PRIMARY_MASS_MSUN,
        m2_source_msun=SECONDARY_MASS_MSUN,
        mission_duration_years=4.0,
        detector_min_frequency_hz=MIN_FREQUENCY_HZ,
        detector_max_frequency_hz=0.1,
        snr_threshold=20.0,
        window_placement="band_entry",
    )
    assert result.status == "evaluated"
    assert np.isfinite(result.lisa_snr) and result.lisa_snr > 0.0

    never_reaches_upper = continue_circular_band_track(
        entry=entry,
        primary_mass_msun=PRIMARY_MASS_MSUN,
        secondary_mass_msun=SECONDARY_MASS_MSUN,
        initial_redshift=INITIAL_REDSHIFT,
        duty_cycle_myr=1.0e4,
        disk_inner_radius_cm=grid_cm[0],
        gas_radius_cm=grid_cm,
        gas_drift_cm_s=gas_drift_cm_s,
        max_observed_frequency_hz=0.1,
        redshift_model=CLOCK,
    )
    assert never_reaches_upper.status == "cosmic_time_limit"
    fallback = evaluate_lisa_coupled_track_snr(
        track=never_reaches_upper,
        m1_source_msun=PRIMARY_MASS_MSUN,
        m2_source_msun=SECONDARY_MASS_MSUN,
        mission_duration_years=4.0,
        detector_min_frequency_hz=MIN_FREQUENCY_HZ,
        detector_max_frequency_hz=0.1,
        snr_threshold=20.0,
        window_placement="optimal",
    )
    assert fallback.status == "evaluated"
    assert fallback.observation_case.startswith(
        "coupled_optimal_upper_unreached_entry_fallback"
    )


def test_moving_band_can_become_available_after_initial_inside_isco_boundary() -> None:
    primary_msun = 1.0e6
    secondary_msun = 1.0
    total_msun = primary_msun + secondary_msun
    isco_cm = 6.0 * const.G * total_msun * const.msun / const.c**2
    radius_cm = 1.5 * isco_cm
    min_frequency_hz = 1.0e-3
    boundary = circular_band_boundary(
        primary_msun, secondary_msun, min_frequency_hz, 15.0
    )
    assert boundary.status == "band_entry_inside_isco"
    beta = peters_beta_cm4_s(primary_msun, secondary_msun)
    grid_cm = np.unique(
        np.concatenate(
            (np.linspace(isco_cm, radius_cm, 64),
             np.linspace(radius_cm, 2.0 * radius_cm, 64))
        )
    )
    outcome = evolve_circular_orbit(
        primary_mass_msun=primary_msun,
        secondary_mass_msun=secondary_msun,
        initial_radius_cm=radius_cm,
        initial_time_myr=0.0,
        initial_redshift=15.0,
        duty_cycle_myr=1.5e4,
        disk_inner_radius_cm=isco_cm,
        gas_radius_cm=grid_cm,
        gas_drift_cm_s=beta / grid_cm**3,
        min_observed_frequency_hz=min_frequency_hz,
        redshift_model=CLOCK,
    )
    assert outcome.status == "band_entry"
    assert outcome.entry_time_myr > 0.0
    assert outcome.entry_redshift < 15.0
    assert outcome.gas_active_at_entry is True


def test_workflow_rejects_non_circular_or_mass_jump_configuration() -> None:
    base = build_parser().parse_args(["1", "/tmp/imri-coupled-config-check", "1"])
    base.gw_eccentricity = 0.01
    try:
        build_config_from_arguments(base)
    except ValueError as error:
        assert "gw_eccentricity=0" in str(error)
    else:
        raise AssertionError("non-circular coupled configuration was accepted")
    base.gw_eccentricity = 0.0
    base.cosmic_merge_time_myr = 1000.0
    try:
        build_config_from_arguments(base)
    except ValueError as error:
        assert "cosmic merger disabled" in str(error)
    else:
        raise AssertionError("finite mass-jump configuration was accepted")
