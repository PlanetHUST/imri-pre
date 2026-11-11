"""Flat-Lambda-CDM clock checks for the gas, GW and detector stages."""

import numpy as np
from scipy.integrate import quad

from lisa.detectability import classify_detectability
from lisa.snr import luminosity_distance_mpc
from physical_process.constants import MEGAYEAR_S, MPC_M
from physical_process.gw_frequency_radius import gw_radius_for_frequency_cm
from physical_process.redshift_evolve import RedshiftEvolution


def test_cosmic_age_matches_independent_friedmann_integral() -> None:
    clock = RedshiftEvolution(67.4, 0.315)
    hubble_per_myr = 67.4 * 1000.0 / MPC_M * MEGAYEAR_S
    for redshift in (0.0, 1.0, 15.0, 30.0):
        age_integral, _ = quad(
            lambda scale_factor: np.sqrt(scale_factor)
            / np.sqrt(0.315 + 0.685 * scale_factor**3),
            0.0,
            1.0 / (1.0 + redshift),
            epsabs=0.0,
            epsrel=1.0e-11,
        )
        assert np.isclose(
            clock.redshift_to_cosmic_time(redshift),
            age_integral / hubble_per_myr,
            rtol=1.0e-10,
        )


def test_clock_roundtrip_and_present_day_budget() -> None:
    clock = RedshiftEvolution(67.4, 0.315)
    for redshift in (0.0, 0.5, 1.0, 15.0, 30.0):
        age_myr = clock.redshift_to_cosmic_time(redshift)
        assert np.isclose(
            clock.cosmic_time_to_redshift(age_myr), redshift, atol=1.0e-12
        )
        assert np.isclose(
            age_myr + clock.remaining_time_to_zero_redshift_myr(redshift),
            clock.redshift_to_cosmic_time(0.0),
            atol=1.0e-9,
        )
        assert np.isclose(clock.redshift_evlove_with_time(redshift, 0.0), redshift)
    assert clock.redshift_evlove_with_time(1.0, 1.0e5) == 0.0
    assert clock.remaining_time_to_zero_redshift_myr(0.0) == 0.0


def test_detector_epoch_and_distance_use_the_same_cosmology() -> None:
    initial_redshift = 1.0
    elapsed_myr = 100.0
    source_frequency_hz = 1.0e-3 * (1.0 + initial_redshift)
    radius_cm = gw_radius_for_frequency_cm(1.0e4 + 100.0, source_frequency_hz)

    def detect(hubble_km_s_mpc: float, omega_m: float):
        return classify_detectability(
            is_imri_candidate=True,
            primary_mass_msun=1.0e4,
            secondary_mass_msun=100.0,
            r_gas_fin_cm=radius_cm,
            gas_final_time_myr=elapsed_myr,
            initial_redshift=initial_redshift,
            min_observed_frequency_hz=1.0e-3,
            max_observed_frequency_hz=1.0e-1,
            eccentricity=0.0,
            cosmic_merge_time_myr=np.inf,
            cosmic_merge_smbh_mass_factor=2.0,
            cosmic_merge_radius_factor=2.0,
            snr_threshold=20.0,
            mission_duration_years=4.0,
            detector_hubble_constant_km_s_mpc=hubble_km_s_mpc,
            detector_omega_m=omega_m,
        )

    for hubble_km_s_mpc, omega_m in ((67.4, 0.315), (74.0, 0.27)):
        result = detect(hubble_km_s_mpc, omega_m)
        expected_redshift = RedshiftEvolution(
            hubble_km_s_mpc, omega_m
        ).redshift_evlove_with_time(initial_redshift, elapsed_myr)
        assert result.detection_time_myr == elapsed_myr
        assert np.isclose(result.detection_redshift, expected_redshift)
        assert np.isclose(
            result.luminosity_distance_mpc,
            luminosity_distance_mpc(
                expected_redshift,
                hubble_constant_km_s_mpc=hubble_km_s_mpc,
                omega_m=omega_m,
            ),
        )
    assert (
        detect(67.4, 0.315).detection_redshift
        != detect(74.0, 0.27).detection_redshift
    )
