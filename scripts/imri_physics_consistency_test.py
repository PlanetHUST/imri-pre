#!/usr/bin/env python3
"""Deterministic checks for the single-AGN physics consistency repair."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from disk_quantity.migration_torque import (  # noqa: E402
    MigrationParameters,
    TorqueProfile,
    comp_gamma_th_coeff,
    comp_gamma_type_ii_si,
    comp_gap_parameter_dimensionless,
    compute_torque_profile,
    migration_electron_scattering_opacity_m2_kg,
    thermal_diffusivity_m2_s,
)
from disk_quantity.pagn_disk_adapter import build_pagn_disk_profile  # noqa: E402
from physical_process import constants as const  # noqa: E402
from physical_process.damp import (  # noqa: E402
    DampingCalculator,
    DiskProperties,
    OrbitalState,
)
from physical_process.disk_migration import (  # noqa: E402
    CLASSICAL_TYPE2,
    CONTINUOUS_GAP,
    GAP_DEPTH_COEFFICIENT,
    MIGRATION_MODELS,
    TYPE_II_GAP_PARAMETER_THRESHOLD,
    UNGAPPED,
    gap_depth_factor,
    gap_opening_mask,
    migration_torque_diagnostics,
    select_migration_torque_si,
)
from physical_process import gw_frequency_radius as gw_helpers  # noqa: E402
from workflows.config import (  # noqa: E402
    DISK_ACCRETION_RATE_G_S_PER_MSUN,
    PAGN_ALPHA,
)
from workflows.cosmology import FlatLambdaCDM  # noqa: E402
from workflows.detect_rate import classify_detectability  # noqa: E402
from workflows.event_rate import solve_gas_final_radius_cm  # noqa: E402
import workflows.engine as engine_module  # noqa: E402

SMBH_MASS_MSUN = 1.0e3
MIGRATOR_MASS_MSUN = 30.0
HUBBLE_CONSTANT_KM_S_MPC = 67.4
OMEGA_M = 0.315
LISA_MIN_FREQUENCY_HZ = 1.0e-3
LISA_MAX_FREQUENCY_HZ = 1.0e-1
MISSION_DURATION_YEARS = 4.0
SNR_THRESHOLD = 20.0


def _torque_profile_for_selection() -> TorqueProfile:
    values = np.asarray([1.0, 2.0], dtype=float)
    return TorqueProfile(
        radius_m=values,
        radius_cm=100.0 * values,
        radius_rg=values,
        radius_rs=0.5 * values,
        surface_density_kg_m2=values,
        pressure_pa=values,
        dlog_surface_density_dlog_radius=values,
        dlog_temperature_dlog_radius=values,
        dlog_pressure_dlog_radius=values,
        corotation_coeff=values,
        lindblad_coeff=values,
        thermal_coeff=values,
        gap_parameter_dimensionless=values,
        gamma_0_si=values,
        gamma_type_i_si=np.asarray([-2.0, -3.0]),
        gamma_th_si=np.asarray([0.5, 0.75]),
        gamma_type_ii_si=np.asarray([-20.0, -30.0]),
    )


def validate_damping_equations() -> None:
    state = OrbitalState(
        semi_major_axis_cm=7.0,
        eccentricity=0.3,
        inclination_rad=0.4,
        cos_omega=0.25,
    )
    tau1_s, tau2_s = 11.0, 13.0
    crossing1_s, crossing2_s = 0.7, 0.9
    e = state.eccentricity
    cosw = state.cos_omega
    cosi = state.cos_inclination
    sini = state.sin_inclination
    expected = (
        -state.semi_major_axis_cm
        / tau1_s
        * 2.0
        / (1.0 - e**2)
        * (1.0 - 2.0 * e * cosw + e**2 - (1.0 - e * cosw) ** 1.5 * cosi)
        * crossing1_s,
        -state.semi_major_axis_cm
        / tau2_s
        * 2.0
        / (1.0 - e**2)
        * (1.0 + 2.0 * e * cosw + e**2 - (1.0 + e * cosw) ** 1.5 * cosi)
        * crossing2_s,
        -sini / (tau1_s * np.sqrt(1.0 - e * cosw)) * crossing1_s,
        -sini / (tau2_s * np.sqrt(1.0 + e * cosw)) * crossing2_s,
        -1.0
        / tau1_s
        * (
            2.0 * e
            - 2.0 * cosw
            - (e + e * cosw**2 - 2.0 * cosw)
            / np.sqrt(1.0 - e * cosw)
            * cosi
        )
        * crossing1_s,
        -1.0
        / tau2_s
        * (
            2.0 * e
            + 2.0 * cosw
            - (e + e * cosw**2 + 2.0 * cosw)
            / np.sqrt(1.0 + e * cosw)
            * cosi
        )
        * crossing2_s,
    )
    calculator = object.__new__(DampingCalculator)
    actual = calculator.compute_ode_coefficients(
        state,
        tau1_s,
        tau2_s,
        crossing1_s,
        crossing2_s,
        (True, True),
    )
    assert np.allclose(actual, expected, rtol=1.0e-14, atol=1.0e-14)

    symmetric_state = state.update(cos_omega=0.0)
    symmetric = calculator.compute_ode_coefficients(
        symmetric_state,
        12.0,
        12.0,
        0.8,
        0.8,
        (True, True),
    )
    assert np.isclose(symmetric[0], symmetric[1])
    assert np.isclose(symmetric[2], symmetric[3])
    assert np.isclose(symmetric[4], symmetric[5])

    disk_properties = DiskProperties.from_agn(
        1.0,
        2.0,
        (True, True),
        np.asarray([100.0, 200.0]),
        np.asarray([3.0, 5.0]),
        np.asarray([0.1, 0.2]),
        np.asarray([1.0, 2.0]),
    )
    assert disk_properties.density_g_cm3 == (3.0, 5.0)


def validate_migration_equations() -> None:
    mass_ratio = 1.0e-3
    alpha = 0.1
    aspect_ratio = np.asarray([0.1, 0.03])
    gap_parameter = comp_gap_parameter_dimensionless(
        mass_ratio,
        aspect_ratio,
        alpha,
    )
    assert np.allclose(gap_parameter, mass_ratio**2 / (alpha * aspect_ratio**5))
    assert TYPE_II_GAP_PARAMETER_THRESHOLD == 11.0
    assert np.array_equal(
        gap_opening_mask(mass_ratio, aspect_ratio, alpha, "kii"),
        gap_parameter > TYPE_II_GAP_PARAMETER_THRESHOLD,
    )
    expected_gap_depth_factor = 1.0 / (
        1.0 + GAP_DEPTH_COEFFICIENT * gap_parameter
    )
    assert np.allclose(
        gap_depth_factor(gap_parameter),
        expected_gap_depth_factor,
    )
    assert MIGRATION_MODELS == (CONTINUOUS_GAP, CLASSICAL_TYPE2, UNGAPPED)

    profile = _torque_profile_for_selection()
    migrator_mass_kg = 2.0
    diagnostics = migration_torque_diagnostics(
        profile,
        mass_ratio,
        aspect_ratio,
        alpha,
        "kii",
        True,
        migrator_mass_kg=migrator_mass_kg,
    )
    assert np.allclose(
        diagnostics.gap_depth_factor,
        expected_gap_depth_factor,
    )
    assert np.allclose(
        diagnostics.gamma_grav_gap_si,
        expected_gap_depth_factor * profile.gamma_type_i_si,
    )
    assert np.allclose(
        diagnostics.gamma_total_si,
        diagnostics.gamma_grav_gap_si + profile.gamma_th_si,
    )
    assert np.allclose(
        diagnostics.local_disk_mass_ratio,
        4.0
        * np.pi
        * profile.surface_density_kg_m2
        * profile.radius_m**2
        / migrator_mass_kg,
    )
    assert np.array_equal(
        diagnostics.deep_gap_mask,
        gap_parameter > TYPE_II_GAP_PARAMETER_THRESHOLD,
    )
    assert not diagnostics.type_ii_mask.any()
    for component in (
        diagnostics.gap_parameter_dimensionless,
        diagnostics.gap_depth_factor,
        diagnostics.local_disk_mass_ratio,
        diagnostics.gamma_grav_ungapped_si,
        diagnostics.gamma_grav_gap_si,
        diagnostics.gamma_thermal_si,
        diagnostics.gamma_type_ii_si,
        diagnostics.gamma_total_si,
    ):
        assert np.all(np.isfinite(component))

    # Legacy six-positional-argument call: same return type and array shape,
    # with the new continuous model supplied by the optional default.
    selection = select_migration_torque_si(
        profile,
        mass_ratio,
        aspect_ratio,
        alpha,
        "kii",
        True,
    )
    assert selection.torque_si.shape == profile.gamma_type_i_si.shape
    assert np.allclose(selection.torque_si, diagnostics.gamma_total_si)
    assert not selection.type_ii_mask.any()

    classical = select_migration_torque_si(
        profile,
        mass_ratio,
        aspect_ratio,
        alpha,
        "kii",
        True,
        migration_model=CLASSICAL_TYPE2,
    )
    assert np.isclose(classical.torque_si[0], -1.5)
    assert np.isclose(classical.torque_si[1], -30.0)
    assert np.array_equal(classical.type_ii_mask, np.asarray([False, True]))
    assert np.all(np.isfinite(classical.torque_si))

    ungapped = select_migration_torque_si(
        profile,
        mass_ratio,
        aspect_ratio,
        alpha,
        "kii",
        True,
        migration_model=UNGAPPED,
    )
    expected_ungapped = profile.gamma_type_i_si + profile.gamma_th_si
    assert np.allclose(ungapped.torque_si, expected_ungapped)
    assert not ungapped.type_ii_mask.any()
    assert np.all(np.isfinite(ungapped.torque_si))

    legacy_disabled = select_migration_torque_si(
        profile,
        mass_ratio,
        aspect_ratio,
        alpha,
        "kii",
        False,
        migration_model=CLASSICAL_TYPE2,
    )
    assert np.allclose(legacy_disabled.torque_si, expected_ungapped)

    # The continuous torque has no branch or jump at the old K=11 threshold.
    near_threshold_k = np.asarray([11.0 - 1.0e-8, 11.0, 11.0 + 1.0e-8])
    near_threshold_aspect_ratio = (
        mass_ratio**2 / (alpha * near_threshold_k)
    ) ** (1.0 / 5.0)
    continuity_profile = replace(
        profile,
        gamma_type_i_si=np.full(3, -2.0),
        gamma_th_si=np.full(3, 0.5),
        gamma_type_ii_si=np.full(3, -20.0),
    )
    continuous = select_migration_torque_si(
        continuity_profile,
        mass_ratio,
        near_threshold_aspect_ratio,
        alpha,
        "kii",
        True,
        migration_model=CONTINUOUS_GAP,
    )
    expected_continuous = (
        gap_depth_factor(near_threshold_k) * continuity_profile.gamma_type_i_si
        + continuity_profile.gamma_th_si
    )
    assert np.allclose(continuous.torque_si, expected_continuous)
    assert np.all(np.isfinite(continuous.torque_si))
    assert abs(continuous.torque_si[2] - continuous.torque_si[0]) < 1.0e-9

    migrator_mass_kg = 2.0e31
    radius_m = np.asarray([3.0e9])
    omega_rad_s = np.asarray([4.0e-4])
    h_aspect = np.asarray([0.02])
    type_ii = comp_gamma_type_ii_si(
        alpha,
        migrator_mass_kg,
        h_aspect,
        radius_m,
        omega_rad_s,
    )
    expected_type_ii = (
        -alpha * migrator_mass_kg * h_aspect**2 * radius_m**2 * omega_rad_s**2
    )
    assert np.allclose(type_ii, expected_type_ii)
    local_log_time_s = (
        0.5 * migrator_mass_kg * radius_m * omega_rad_s * radius_m / -type_ii
    )
    assert np.allclose(
        local_log_time_s,
        1.0 / (2.0 * alpha * h_aspect**2 * omega_rad_s),
    )


def _thermal_disk(*, pressure_tau: float, opacity: float) -> SimpleNamespace:
    return SimpleNamespace(
        T=np.asarray([1.0e5]),
        rho=np.asarray([1.0e-7]),
        h=np.asarray([2.0e8]),
        Omega=np.asarray([1.0e-4]),
        cs=np.asarray([2.0e4]),
        R=np.asarray([1.0e10]),
        tauV=np.asarray([pressure_tau]),
        kappa=np.asarray([opacity]),
    )


def validate_thermal_torque() -> None:
    params = MigrationParameters(alpha=0.1)
    opacity_m2_kg = migration_electron_scattering_opacity_m2_kg(0.7)
    assert np.isclose(opacity_m2_kg, 0.034, rtol=1.0e-12)

    disk_low_kappa = _thermal_disk(pressure_tau=1.0e12, opacity=1.0e-30)
    disk_high_kappa = _thermal_disk(pressure_tau=1.0e12, opacity=1.0e30)
    chi_low = thermal_diffusivity_m2_s(disk_low_kappa, params.gamma_adiabat)
    chi_high = thermal_diffusivity_m2_s(disk_high_kappa, params.gamma_adiabat)
    assert np.array_equal(chi_low, chi_high)
    critical_mass_kg = disk_low_kappa.cs * chi_low / const.GRAVITATIONAL_CONSTANT_SI
    assert np.all(np.isfinite(critical_mass_kg) & (critical_mass_kg > 0.0))

    slope = np.asarray([-2.0])
    zero_tau = comp_gamma_th_coeff(
        slope,
        _thermal_disk(pressure_tau=0.0, opacity=1.0),
        2.0e31,
        params,
    )
    thick_negative_slope = comp_gamma_th_coeff(
        slope,
        disk_low_kappa,
        2.0e31,
        params,
    )
    thick_positive_slope = comp_gamma_th_coeff(
        -slope,
        disk_low_kappa,
        2.0e31,
        params,
    )
    very_thick_negative_slope = comp_gamma_th_coeff(
        slope,
        _thermal_disk(pressure_tau=1.0e15, opacity=1.0),
        2.0e31,
        params,
    )
    assert np.allclose(zero_tau, 0.0, atol=0.0)
    assert np.all(np.isfinite(thick_negative_slope))
    assert np.allclose(thick_negative_slope, -thick_positive_slope)
    assert np.allclose(thick_negative_slope, very_thick_negative_slope)


def validate_event_routing() -> None:
    def gas_time(start_cm: float, target_cm: float) -> float:
        return 0.1 * max(0.0, start_cm - target_cm)

    def solve_case(
        *,
        available_gas_time_myr: float,
        migration_time_myr_call: Callable[[float, float], float],
        gw_time_to_entry_myr: Callable[[float, float], float],
    ) -> tuple[float, float, float, float, float, str]:
        return solve_gas_final_radius_cm(
            r_ini_cm=100.0,
            r_gas_target_cm=10.0,
            r_entry_cm=10.0,
            handoff_required=False,
            gas_start_time_myr=0.0,
            num_radius_samples=32,
            available_gas_time_myr=available_gas_time_myr,
            migration_time_myr_call=migration_time_myr_call,
            gw_time_to_entry_myr=gw_time_to_entry_myr,
        )

    gas_faster = solve_case(
        available_gas_time_myr=10.0,
        migration_time_myr_call=gas_time,
        gw_time_to_entry_myr=lambda radius_cm, epoch_myr: 0.2
        * max(0.0, radius_cm - 10.0),
    )
    assert gas_faster[-1] == "entry_reached"
    direct_faster = solve_case(
        available_gas_time_myr=10.0,
        migration_time_myr_call=gas_time,
        gw_time_to_entry_myr=lambda radius_cm, epoch_myr: 1.0,
    )
    assert direct_faster[-1] == "gw_dominates_at_start"
    trapped_but_gw_finite = solve_case(
        available_gas_time_myr=10.0,
        migration_time_myr_call=lambda start_cm, target_cm: np.inf,
        gw_time_to_entry_myr=lambda radius_cm, epoch_myr: 1.0,
    )
    assert trapped_but_gw_finite[-1] == "gw_dominates_at_start"
    trapped_without_gw = solve_case(
        available_gas_time_myr=10.0,
        migration_time_myr_call=lambda start_cm, target_cm: np.inf,
        gw_time_to_entry_myr=lambda radius_cm, epoch_myr: np.inf,
    )
    assert trapped_without_gw[-1] == "migration_trap_bound"
    duty_limited = solve_case(
        available_gas_time_myr=2.0,
        migration_time_myr_call=gas_time,
        gw_time_to_entry_myr=lambda radius_cm, epoch_myr: 0.2
        * max(0.0, radius_cm - 10.0),
    )
    assert duty_limited[-1] == "duty_cycle_limited"


def validate_cosmology_and_gw() -> None:
    cosmology = FlatLambdaCDM(HUBBLE_CONSTANT_KM_S_MPC, OMEGA_M)
    for redshift in (0.1, 1.0, 5.0, 10.0, 15.0):
        age_myr = cosmology.cosmic_age_myr(redshift)
        assert np.isclose(
            cosmology.redshift_at_cosmic_age_myr(age_myr),
            redshift,
            rtol=2.0e-14,
        )
        assert cosmology.evolve_redshift(redshift, 0.0) == redshift
        assert cosmology.evolve_redshift(redshift, 1.0) < redshift
        assert cosmology.remaining_time_to_z0_myr(redshift) > 0.0
    expected_distances_mpc = {
        0.1: 477.522094099,
        1.0: 6802.525834694,
        10.0: 105971.081494102,
        15.0: 167196.706346107,
    }
    for redshift, expected_mpc in expected_distances_mpc.items():
        assert np.isclose(
            cosmology.luminosity_distance_mpc(redshift),
            expected_mpc,
            rtol=2.0e-12,
        )

    total_mass_msun = 1.001e5
    radius_cm = 2.0e12
    frequency_hz = gw_helpers.gw_frequency_hz(total_mass_msun, radius_cm)
    assert np.isclose(
        gw_helpers.gw_radius_for_frequency_cm(total_mass_msun, frequency_hz),
        radius_cm,
        rtol=2.0e-14,
    )
    base_time_myr = gw_helpers.peters_merger_time_myr(
        1.0e5,
        100.0,
        radius_cm,
        0.0,
    )
    assert np.isclose(
        gw_helpers.peters_merger_time_myr(1.0e5, 100.0, 3.0 * radius_cm, 0.0),
        3.0**4 * base_time_myr,
    )
    source_frequency_hz = gw_helpers.source_frame_gw_frequency_hz(1.0e-3, 10.0)
    assert gw_helpers.observed_frame_gw_frequency_hz(source_frequency_hz, 10.0) == 1.0e-3

    for primary_mass_msun in (1.0e5, 1.0e7):
        boundary = gw_helpers.circular_band_boundary(
            primary_mass_msun,
            100.0,
            LISA_MIN_FREQUENCY_HZ,
            1.0,
        )
        entry_frequency_hz = gw_helpers.observed_frame_gw_frequency_hz(
            gw_helpers.gw_frequency_hz(
                primary_mass_msun + 100.0,
                boundary.formal_entry_radius_cm,
            ),
            1.0,
        )
        assert np.isclose(entry_frequency_hz, LISA_MIN_FREQUENCY_HZ)
        assert (boundary.status == "band_entry_inside_isco") == (
            boundary.formal_entry_radius_cm <= boundary.isco_radius_cm
        )

    entry_radius_cm = gw_helpers.lisa_entry_radius_cm(
        1.0e5,
        100.0,
        LISA_MIN_FREQUENCY_HZ,
        1.0,
    )
    fixed_primary = classify_detectability(
        is_imri_candidate=True,
        primary_mass_msun=1.0e5,
        secondary_mass_msun=100.0,
        r_gas_fin_cm=entry_radius_cm,
        gas_final_time_myr=0.0,
        initial_redshift=1.0,
        min_observed_frequency_hz=LISA_MIN_FREQUENCY_HZ,
        max_observed_frequency_hz=LISA_MAX_FREQUENCY_HZ,
        max_gw_time_myr=1.0e4,
        eccentricity=0.0,
        snr_threshold=SNR_THRESHOLD,
        mission_duration_years=MISSION_DURATION_YEARS,
        detector_hubble_constant_km_s_mpc=HUBBLE_CONSTANT_KM_S_MPC,
        detector_omega_m=OMEGA_M,
    )
    assert fixed_primary.final_smbh_mass_msun == 1.0e5
    assert fixed_primary.cosmic_merge_applied is False

    time_limited = classify_detectability(
        is_imri_candidate=True,
        primary_mass_msun=1.0e5,
        secondary_mass_msun=100.0,
        r_gas_fin_cm=2.0 * entry_radius_cm,
        gas_final_time_myr=0.0,
        initial_redshift=1.0,
        min_observed_frequency_hz=LISA_MIN_FREQUENCY_HZ,
        max_observed_frequency_hz=LISA_MAX_FREQUENCY_HZ,
        max_gw_time_myr=1.0e-12,
        eccentricity=0.0,
        snr_threshold=SNR_THRESHOLD,
        mission_duration_years=MISSION_DURATION_YEARS,
        detector_hubble_constant_km_s_mpc=HUBBLE_CONSTANT_KM_S_MPC,
        detector_omega_m=OMEGA_M,
    )
    assert time_limited.detect_rate_reason.endswith("misses_gw_time_budget")


def validate_pagn_interface_and_boundary() -> None:
    profile = build_pagn_disk_profile(
        SMBH_MASS_MSUN,
        PAGN_ALPHA,
        n_resolution=240,
        accretion_rate_g_s=(
            DISK_ACCRETION_RATE_G_S_PER_MSUN * SMBH_MASS_MSUN
        ),
        suppress_pagn_output=True,
    )
    scale_height_cm = np.asarray(profile.aspect_ratio) * np.asarray(profile.radius_cm)
    assert np.allclose(
        profile.surface_density_g_cm2,
        2.0 * np.asarray(profile.density_g_cm3) * scale_height_cm,
        rtol=2.0e-13,
    )

    engine = object.__new__(engine_module.IMRIDynEngine)
    engine.disk_cache = {}
    original_builder = engine_module.build_pagn_disk_profile
    engine_module.build_pagn_disk_profile = lambda *args, **kwargs: profile
    try:
        cached = engine.get_disk_profiles(SMBH_MASS_MSUN)
    finally:
        engine_module.build_pagn_disk_profile = original_builder
    assert np.array_equal(cached["density_g_cm3"], profile.density_g_cm3)

    pagn_disk = cast(Any, profile.disk)
    original_radius_m = np.asarray(pagn_disk.R, dtype=float).copy()
    requested_max_rg = 10.0 * float(np.max(original_radius_m)) / (
        0.5 * float(pagn_disk.Rs)
    )
    torque_profile = compute_torque_profile(
        pagn_disk,
        migrator_mass_msun=MIGRATOR_MASS_MSUN,
        max_radius_rg=requested_max_rg,
        params=MigrationParameters(alpha=PAGN_ALPHA),
    )
    assert np.array_equal(np.asarray(pagn_disk.R, dtype=float), original_radius_m)
    assert np.max(torque_profile.radius_cm) <= profile.outer_radius_cm

    crop_radius_rg = float(np.median(torque_profile.radius_rg))
    cropped = compute_torque_profile(
        pagn_disk,
        migrator_mass_msun=MIGRATOR_MASS_MSUN,
        max_radius_rg=crop_radius_rg,
        params=MigrationParameters(alpha=PAGN_ALPHA),
    )
    assert np.max(cropped.radius_rg) <= crop_radius_rg
    assert np.array_equal(np.asarray(pagn_disk.R, dtype=float), original_radius_m)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--section",
        choices=("all", "algebra", "pagn"),
        default="all",
        help="run cheap algebraic checks, the pAGN integration check, or both",
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.section in ("all", "algebra"):
        validate_damping_equations()
        validate_migration_equations()
        validate_thermal_torque()
        validate_event_routing()
        validate_cosmology_and_gw()
        print("single-AGN algebraic physics checks passed")
    if arguments.section in ("all", "pagn"):
        validate_pagn_interface_and_boundary()
        print("pAGN density and Q=1 torque-domain checks passed")


if __name__ == "__main__":
    main()
