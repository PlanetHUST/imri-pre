#!/usr/bin/env python3
"""Standalone validation harness for first-order physical LISA detectability."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Sequence
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from workflows.lisa_snr import (  # noqa: E402
    JULIAN_YEAR_S,
    SNR_STATUS_EVALUATED,
    SNR_STATUS_NUMERICAL_FAILURE,
    SNR_STATUS_NO_OVERLAP,
    SNR_STATUS_REQUIRED_INPUT_UNAVAILABLE,
    evaluate_lisa_inspiral_snr,
    fourier_inspiral_amplitude_hz_inverse,
    inspiral_time_to_coalescence_s,
    integrated_lisa_snr,
    lisa_confusion_strain_psd_hz,
    lisa_effective_strain_psd_hz,
    lisa_instrument_strain_psd_hz,
    luminosity_distance_mpc,
    mission_observation_window,
    observed_frequency_from_time_to_coalescence_hz,
    source_chirp_mass_msun,
)
import physical_process.gw_frequency_radius as gw_helpers  # noqa: E402
from workflows.detect_rate import (  # noqa: E402
    DETECTION_CLASS_DETECTABLE,
    DETECTION_CLASS_NOT_EVALUATED,
    classify_detectability,
)
from workflows.models import IMRIDynAveragedResult, IMRIDynSingleResult  # noqa: E402
from workflows.outputs import (  # noqa: E402
    IMRI_EVENT_COLUMNS,
    IMRI_REPEAT_RESULT_COLUMNS,
    IMRI_RESULT_COLUMNS,
)

DEMO_M1_SOURCE_MSUN = 1.0e5
DEMO_M2_SOURCE_MSUN = 100.0
DEMO_REDSHIFT = 1.0
DEMO_MISSION_YEARS = 4.0
DEMO_F_MIN_HZ = 1.0e-3
DEMO_F_MAX_HZ = 1.0e-1
DEMO_SNR_THRESHOLD = 20.0

PSD_FREQUENCIES_HZ = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
PSD_4_YEAR_BENCHMARKS_HZ_INV = (
    2.1418352557641668e-33,
    6.471724317934254e-36,
    8.296514847664091e-38,
    3.1256997223140400e-40,
    1.4431694831552352e-40,
    3.0165755631253280e-40,
    2.1295830794035954e-39,
)
COSMOLOGY_BENCHMARKS_MPC = {
    0.1: 477.5220940990449,
    1.0: 6802.525834693799,
    5.0: 47676.04090629676,
    10.0: 105971.08149410246,
    15.0: 167196.70634610672,
}


def demo_result(redshift: float = DEMO_REDSHIFT):
    return evaluate_lisa_inspiral_snr(
        m1_source_msun=DEMO_M1_SOURCE_MSUN,
        m2_source_msun=DEMO_M2_SOURCE_MSUN,
        redshift=redshift,
        current_observed_frequency_hz=DEMO_F_MIN_HZ,
        mission_duration_years=DEMO_MISSION_YEARS,
        detector_min_frequency_hz=DEMO_F_MIN_HZ,
        detector_max_frequency_hz=DEMO_F_MAX_HZ,
        snr_threshold=DEMO_SNR_THRESHOLD,
    )


def validate_demo() -> None:
    result = demo_result()
    assert result.status == SNR_STATUS_EVALUATED
    assert result.is_detectable is True
    assert result.reason == "lisa_snr_above_threshold"
    assert np.isclose(result.chirp_mass_source_msun, 1584.5764038704444)
    assert np.isclose(result.chirp_mass_detector_msun, 3169.152807740889)
    assert np.isclose(result.luminosity_distance_mpc, 6802.525834693799)
    assert np.isclose(result.f_isco_observer_hz, 0.021963245727087584)
    assert np.isclose(result.lisa_snr, 58.07069322689072, rtol=2e-10)

    psd_1_mhz_hz_inv = lisa_effective_strain_psd_hz(1.0e-3, 4.0)
    amplitude_at_f_low_hz_inverse = fourier_inspiral_amplitude_hz_inverse(
        result.f_low_hz,
        result.chirp_mass_detector_msun,
        result.luminosity_distance_mpc,
    )
    print("\nEnd-to-end demonstration")
    for key in (
        "m1_source_msun",
        "m2_source_msun",
        "redshift",
        "chirp_mass_source_msun",
        "chirp_mass_detector_msun",
        "luminosity_distance_mpc",
        "f_isco_source_hz",
        "f_isco_observer_hz",
        "current_observed_frequency_hz",
        "f_low_hz",
        "f_high_hz",
        "mission_duration_years",
        "observation_duration_years",
        "observation_case",
        "lisa_snr",
        "snr_threshold",
        "reason",
    ):
        print(f"{key:36s} {asdict(result)[key]}")
    print(f"{'lisa_psd_at_1e-3_hz_hz_inv':36s} {psd_1_mhz_hz_inv}")
    print(f"{'lisa_asd_at_1e-3_hz_hz_mhalf':36s} {np.sqrt(psd_1_mhz_hz_inv)}")
    print(f"{'fourier_amplitude_at_f_low_hz_inv':36s} {amplitude_at_f_low_hz_inverse}")


def validate_psd() -> None:
    values = [lisa_effective_strain_psd_hz(frequency_hz, 4.0) for frequency_hz in PSD_FREQUENCIES_HZ]
    assert np.all(np.isfinite(values))
    assert np.all(np.asarray(values) > 0.0)
    assert np.allclose(values, PSD_4_YEAR_BENCHMARKS_HZ_INV, rtol=2e-12)
    minimum_index = int(np.argmin(values))
    assert PSD_FREQUENCIES_HZ[minimum_index] == 1.0e-2
    for mission_years in (0.5, 1.0, 2.0, 4.0):
        for frequency_hz in PSD_FREQUENCIES_HZ:
            instrument = lisa_instrument_strain_psd_hz(frequency_hz)
            foreground = lisa_confusion_strain_psd_hz(
                frequency_hz, mission_years
            )
            total = lisa_effective_strain_psd_hz(frequency_hz, mission_years)
            assert instrument > 0.0
            assert foreground >= 0.0
            assert np.isclose(total, instrument + foreground)
    unavailable_duration = evaluate_lisa_inspiral_snr(
        m1_source_msun=DEMO_M1_SOURCE_MSUN,
        m2_source_msun=DEMO_M2_SOURCE_MSUN,
        redshift=DEMO_REDSHIFT,
        current_observed_frequency_hz=DEMO_F_MIN_HZ,
        mission_duration_years=3.0,
        detector_min_frequency_hz=DEMO_F_MIN_HZ,
        detector_max_frequency_hz=DEMO_F_MAX_HZ,
        snr_threshold=DEMO_SNR_THRESHOLD,
    )
    assert unavailable_duration.status == "model_unavailable"
    assert np.isnan(unavailable_duration.lisa_snr)

    print("\nRCL four-year total effective strain sensitivity")
    print("f_obs_hz       psd_hz_inv       asd_hz_mhalf")
    for frequency_hz, psd_hz_inv in zip(PSD_FREQUENCIES_HZ, values, strict=True):
        print(f"{frequency_hz:10.3e}  {psd_hz_inv:14.7e}  {np.sqrt(psd_hz_inv):14.7e}")


def validate_cosmology() -> None:
    print("\nFlat-LambdaCDM luminosity-distance benchmarks")
    print("z             d_l_mpc")
    assert luminosity_distance_mpc(0.0) == 0.0
    previous_distance_mpc = 0.0
    for redshift, benchmark_mpc in COSMOLOGY_BENCHMARKS_MPC.items():
        distance_mpc = luminosity_distance_mpc(redshift)
        assert np.isclose(distance_mpc, benchmark_mpc, rtol=2e-11)
        assert distance_mpc > previous_distance_mpc
        previous_distance_mpc = distance_mpc
        print(f"{redshift:5.1f}  {distance_mpc:18.9f}")
    with np.testing.assert_raises(ValueError):
        luminosity_distance_mpc(-0.1)


def validate_windows() -> None:
    result = demo_result()
    detector_chirp_mass_msun = result.chirp_mass_detector_msun
    tau_at_entry_s = inspiral_time_to_coalescence_s(
        DEMO_F_MIN_HZ, detector_chirp_mass_msun
    )

    one_year_before_entry_hz = observed_frequency_from_time_to_coalescence_hz(
        tau_at_entry_s + JULIAN_YEAR_S, detector_chirp_mass_msun
    )
    enters = mission_observation_window(
        current_observed_frequency_hz=one_year_before_entry_hz,
        f_isco_observer_hz=result.f_isco_observer_hz,
        chirp_mass_detector_msun=detector_chirp_mass_msun,
        detector_min_frequency_hz=DEMO_F_MIN_HZ,
        detector_max_frequency_hz=DEMO_F_MAX_HZ,
        mission_duration_years=DEMO_MISSION_YEARS,
    )
    assert enters.status == SNR_STATUS_EVALUATED
    assert enters.observation_case.startswith("enters_band_during_mission:")
    assert np.isclose(enters.time_to_band_years, 1.0)

    five_years_before_entry_hz = observed_frequency_from_time_to_coalescence_hz(
        tau_at_entry_s + 5.0 * JULIAN_YEAR_S, detector_chirp_mass_msun
    )
    misses = mission_observation_window(
        current_observed_frequency_hz=five_years_before_entry_hz,
        f_isco_observer_hz=result.f_isco_observer_hz,
        chirp_mass_detector_msun=detector_chirp_mass_msun,
        detector_min_frequency_hz=DEMO_F_MIN_HZ,
        detector_max_frequency_hz=DEMO_F_MAX_HZ,
        mission_duration_years=DEMO_MISSION_YEARS,
    )
    assert misses.status == SNR_STATUS_NO_OVERLAP
    assert misses.reason == "source_does_not_enter_band_during_mission"

    slow_chirp_mass_msun = (1.0 + 0.1) * source_chirp_mass_msun(10.0, 10.0)
    slow = mission_observation_window(
        current_observed_frequency_hz=DEMO_F_MIN_HZ,
        f_isco_observer_hz=0.1,
        chirp_mass_detector_msun=slow_chirp_mass_msun,
        detector_min_frequency_hz=DEMO_F_MIN_HZ,
        detector_max_frequency_hz=DEMO_F_MAX_HZ,
        mission_duration_years=DEMO_MISSION_YEARS,
    )
    assert slow.observation_case == "already_in_band:mission_limited_slow_inspiral"
    assert slow.f_high_hz < 1.001 * slow.f_low_hz

    detector_limited = mission_observation_window(
        current_observed_frequency_hz=DEMO_F_MIN_HZ,
        f_isco_observer_hz=1.0,
        chirp_mass_detector_msun=detector_chirp_mass_msun,
        detector_min_frequency_hz=DEMO_F_MIN_HZ,
        detector_max_frequency_hz=DEMO_F_MAX_HZ,
        mission_duration_years=DEMO_MISSION_YEARS,
    )
    assert detector_limited.observation_case == (
        "already_in_band:reaches_detector_max_during_mission"
    )
    assert detector_limited.f_high_hz == DEMO_F_MAX_HZ

    no_overlap = mission_observation_window(
        current_observed_frequency_hz=DEMO_F_MIN_HZ,
        f_isco_observer_hz=5.0e-4,
        chirp_mass_detector_msun=detector_chirp_mass_msun,
        detector_min_frequency_hz=DEMO_F_MIN_HZ,
        detector_max_frequency_hz=DEMO_F_MAX_HZ,
        mission_duration_years=DEMO_MISSION_YEARS,
    )
    assert no_overlap.status == SNR_STATUS_NO_OVERLAP

    print("\nObservation-window cases")
    for name, window in (
        ("already_in_band_reaches_isco", mission_observation_window(
            current_observed_frequency_hz=DEMO_F_MIN_HZ,
            f_isco_observer_hz=result.f_isco_observer_hz,
            chirp_mass_detector_msun=detector_chirp_mass_msun,
            detector_min_frequency_hz=DEMO_F_MIN_HZ,
            detector_max_frequency_hz=DEMO_F_MAX_HZ,
            mission_duration_years=DEMO_MISSION_YEARS,
        )),
        ("enters_during_mission", enters),
        ("slowly_evolving", slow),
        ("detector_upper_limited", detector_limited),
        ("never_overlaps", misses),
        ("isco_below_band", no_overlap),
    ):
        print(
            f"{name:28s} {window.observation_case:55s} "
            f"[{window.f_low_hz:.6e}, {window.f_high_hz:.6e}]"
        )


def validate_snr() -> None:
    result = demo_result()
    snr_near = integrated_lisa_snr(
        chirp_mass_detector_msun=result.chirp_mass_detector_msun,
        luminosity_distance_mpc_value=1000.0,
        f_low_hz=result.f_low_hz,
        f_high_hz=result.f_high_hz,
        mission_duration_years=result.mission_duration_years,
    )
    snr_far = integrated_lisa_snr(
        chirp_mass_detector_msun=result.chirp_mass_detector_msun,
        luminosity_distance_mpc_value=2000.0,
        f_low_hz=result.f_low_hz,
        f_high_hz=result.f_high_hz,
        mission_duration_years=result.mission_duration_years,
    )
    assert np.isclose(snr_near / snr_far, 2.0, rtol=2e-12)

    log_frequency = np.linspace(np.log(result.f_low_hz), np.log(result.f_high_hz), 200001)
    frequency_hz = np.exp(log_frequency)
    integrand = np.asarray([
        frequency
        * fourier_inspiral_amplitude_hz_inverse(
            frequency,
            result.chirp_mass_detector_msun,
            result.luminosity_distance_mpc,
        )
        ** 2
        / lisa_effective_strain_psd_hz(frequency, result.mission_duration_years)
        for frequency in frequency_hz
    ])
    grid_snr = np.sqrt(16.0 / 5.0 * np.trapezoid(integrand, log_frequency))
    assert np.isclose(grid_snr, result.lisa_snr, rtol=2e-9)

    required_input_unavailable = evaluate_lisa_inspiral_snr(
        m1_source_msun=DEMO_M1_SOURCE_MSUN,
        m2_source_msun=DEMO_M2_SOURCE_MSUN,
        redshift=0.0,
        current_observed_frequency_hz=DEMO_F_MIN_HZ,
        mission_duration_years=DEMO_MISSION_YEARS,
        detector_min_frequency_hz=DEMO_F_MIN_HZ,
        detector_max_frequency_hz=DEMO_F_MAX_HZ,
        snr_threshold=DEMO_SNR_THRESHOLD,
    )
    assert required_input_unavailable.status == SNR_STATUS_REQUIRED_INPUT_UNAVAILABLE
    assert required_input_unavailable.is_detectable is None
    assert np.isnan(required_input_unavailable.lisa_snr)

    with patch(
        "workflows.lisa_snr.integrated_lisa_snr",
        side_effect=ArithmeticError("controlled quadrature failure"),
    ):
        numerical_failure = demo_result()
    assert numerical_failure.status == SNR_STATUS_NUMERICAL_FAILURE
    assert numerical_failure.is_detectable is None
    assert np.isnan(numerical_failure.lisa_snr)
    print("\nSNR numerical checks")
    print(f"fixed-frame distance ratio rho(1000 Mpc)/rho(2000 Mpc): {snr_near/snr_far:.12f}")
    print(f"adaptive-quadrature SNR: {result.lisa_snr:.12f}")
    print(f"200001-point log-grid SNR: {grid_snr:.12f}")


def validate_redshift_grid() -> None:
    print("\nFixed source-mass redshift behavior")
    print("z       d_l_mpc       mchirp_z     f_low_hz    f_high_hz   snr       status")
    for redshift in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 13.0, 15.0):
        result = demo_result(redshift)
        assert result.status in {SNR_STATUS_EVALUATED, SNR_STATUS_NO_OVERLAP}
        print(
            f"{redshift:4.1f}  {result.luminosity_distance_mpc:12.3f}  "
            f"{result.chirp_mass_detector_msun:11.3f}  {result.f_low_hz:10.3e}  "
            f"{result.f_high_hz:10.3e}  {result.lisa_snr:8.3f}  {result.status}"
        )


def validate_mass_redshift_grid() -> None:
    print("\nControlled mass-redshift selection grid")
    print("m1       m2       z     d_l_mpc     f_isco_obs  f_low       f_high      snr       class")
    for m1_source_msun in (1e4, 1e5, 1e6, 1e7):
        for m2_source_msun in (10.0, 100.0, 1000.0):
            for redshift in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 13.0, 15.0):
                result = evaluate_lisa_inspiral_snr(
                    m1_source_msun=m1_source_msun,
                    m2_source_msun=m2_source_msun,
                    redshift=redshift,
                    current_observed_frequency_hz=DEMO_F_MIN_HZ,
                    mission_duration_years=DEMO_MISSION_YEARS,
                    detector_min_frequency_hz=DEMO_F_MIN_HZ,
                    detector_max_frequency_hz=DEMO_F_MAX_HZ,
                    snr_threshold=DEMO_SNR_THRESHOLD,
                )
                assert result.status in {SNR_STATUS_EVALUATED, SNR_STATUS_NO_OVERLAP}
                classification = (
                    "detectable"
                    if result.is_detectable is True
                    else "below"
                    if result.status == SNR_STATUS_EVALUATED
                    else "no_overlap"
                )
                print(
                    f"{m1_source_msun:7.0e}  {m2_source_msun:7.0e}  {redshift:4.1f}  "
                    f"{result.luminosity_distance_mpc:10.1f}  "
                    f"{result.f_isco_observer_hz:10.3e}  {result.f_low_hz:10.3e}  "
                    f"{result.f_high_hz:10.3e}  {result.lisa_snr:8.2f}  {classification}"
                )


def validate_classification() -> None:
    entry_radius_cm = gw_helpers.lisa_entry_radius_cm(
        DEMO_M1_SOURCE_MSUN,
        DEMO_M2_SOURCE_MSUN,
        DEMO_F_MIN_HZ,
        DEMO_REDSHIFT,
    )

    def classify(
        *,
        is_candidate: bool = True,
        eccentricity: float = 0.0,
        gas_final_radius_cm: float = entry_radius_cm,
    ):
        return classify_detectability(
            is_imri_candidate=is_candidate,
            primary_mass_msun=DEMO_M1_SOURCE_MSUN,
            secondary_mass_msun=DEMO_M2_SOURCE_MSUN,
            r_gas_fin_cm=gas_final_radius_cm,
            gas_final_time_myr=0.0,
            initial_redshift=DEMO_REDSHIFT,
            hubble_constant_per_myr=1.0 / 13.6e3,
            min_observed_frequency_hz=DEMO_F_MIN_HZ,
            max_observed_frequency_hz=DEMO_F_MAX_HZ,
            eccentricity=eccentricity,
            cosmic_merge_time_myr=np.inf,
            cosmic_merge_smbh_mass_factor=2.0,
            cosmic_merge_radius_factor=2.0,
            snr_threshold=DEMO_SNR_THRESHOLD,
            mission_duration_years=DEMO_MISSION_YEARS,
            detector_hubble_constant_km_s_mpc=67.4,
            detector_omega_m=0.315,
        )

    decision = classify()
    assert decision.detect_rate_class == DETECTION_CLASS_DETECTABLE
    assert decision.is_lisa_detectable is True
    assert decision.detect_rate_reason == "lisa_snr_above_threshold"
    assert np.isclose(decision.lisa_snr, demo_result().lisa_snr, rtol=2e-10)
    assert decision.snr_proxy == decision.lisa_snr
    assert decision.snr_status == SNR_STATUS_EVALUATED

    assert decision.lisa_snr < 100.0
    below_threshold = classify_detectability(
        is_imri_candidate=True,
        primary_mass_msun=DEMO_M1_SOURCE_MSUN,
        secondary_mass_msun=DEMO_M2_SOURCE_MSUN,
        r_gas_fin_cm=entry_radius_cm,
        gas_final_time_myr=0.0,
        initial_redshift=DEMO_REDSHIFT,
        hubble_constant_per_myr=1.0 / 13.6e3,
        min_observed_frequency_hz=DEMO_F_MIN_HZ,
        max_observed_frequency_hz=DEMO_F_MAX_HZ,
        eccentricity=0.0,
        cosmic_merge_time_myr=np.inf,
        cosmic_merge_smbh_mass_factor=2.0,
        cosmic_merge_radius_factor=2.0,
        snr_threshold=100.0,
        mission_duration_years=DEMO_MISSION_YEARS,
        detector_hubble_constant_km_s_mpc=67.4,
        detector_omega_m=0.315,
    )
    assert below_threshold.snr_status == SNR_STATUS_EVALUATED
    assert below_threshold.is_lisa_detectable is False
    assert below_threshold.detect_rate_reason == "lisa_snr_below_threshold"

    already_in_band = classify(gas_final_radius_cm=0.5 * entry_radius_cm)
    assert np.isclose(already_in_band.final_radius_cm, 0.5 * entry_radius_cm)
    assert already_in_band.observed_frequency_hz > DEMO_F_MIN_HZ
    assert np.isclose(
        already_in_band.snr_f_low_hz, already_in_band.observed_frequency_hz
    )

    noncandidate = classify(is_candidate=False)
    assert noncandidate.detect_rate_class == DETECTION_CLASS_NOT_EVALUATED
    assert noncandidate.is_lisa_detectable is None
    assert np.isnan(noncandidate.lisa_snr)

    eccentric = classify(eccentricity=0.1)
    assert eccentric.detect_rate_class == DETECTION_CLASS_NOT_EVALUATED
    assert eccentric.is_lisa_detectable is None
    assert eccentric.snr_status == "model_unavailable"
    assert eccentric.detect_rate_reason == "snr_model_unavailable_nonzero_eccentricity"
    assert np.isnan(eccentric.lisa_snr)

    print("\nWorkflow classification integration")
    print(
        f"candidate={decision.detect_rate_class}, snr={decision.lisa_snr:.6f}, "
        f"reason={decision.detect_rate_reason}"
    )
    print(
        f"noncandidate={noncandidate.detect_rate_class}, "
        f"reason={noncandidate.detect_rate_reason}"
    )
    print(
        f"eccentric={eccentric.detect_rate_class}, "
        f"reason={eccentric.detect_rate_reason}"
    )


def validate_schema() -> None:
    expected_event_fields = {
        "event_id",
        "mass_ratio_secondary_to_primary",
        "lisa_snr",
        "snr_status",
        "chirp_mass_source_msun",
        "chirp_mass_detector_msun",
        "luminosity_distance_mpc",
        "snr_f_low_hz",
        "snr_f_high_hz",
        "f_isco_observer_hz",
        "mission_duration_years",
        "snr_observation_duration_years",
        "snr_observation_case",
        "snr_input_observed_frequency_hz",
        "snr_threshold",
        "monte_carlo_event_weight",
        "intrinsic_rate_contribution_per_myr",
        "detected_rate_contribution_per_myr",
        "snr_proxy",
    }
    assert expected_event_fields.issubset(IMRI_EVENT_COLUMNS)
    assert {
        "detector_rate_status",
        "n_detector_eval",
        "n_detector_unavailable",
        "n_snr_below",
        "n_no_overlap",
        "n_detector_input_unavailable",
        "n_detector_model_unavailable",
        "n_detector_numerical_failure",
    }.issubset(IMRI_REPEAT_RESULT_COLUMNS)
    assert {
        "detector_rate_status",
        "n_detector_eval",
        "n_detector_unavailable",
        "n_snr_below",
        "n_no_overlap",
        "n_detector_input_unavailable",
        "n_detector_model_unavailable",
        "n_detector_numerical_failure",
    }.issubset(IMRI_RESULT_COLUMNS)

    result = IMRIDynSingleResult(1.0e5, 2.0e5, 0, "semi_without_trap", 123)
    result.imri_count = 1
    result.imri_mass_msun = 100.0
    result.imri_rate_per_myr = 0.1
    result.imri_mass_rate_msun_per_myr = 10.0
    result.detector_unavailable_imri_count = 1
    result.detectable_imri_rate_per_myr = np.nan
    result.detectable_imri_mass_rate_msun_per_myr = np.nan
    averaged = IMRIDynAveragedResult(
        1.0e5, 2.0e5, "semi_without_trap", [result]
    ).to_dict()
    assert averaged["rate_evaluation_status"] == "complete"
    assert averaged["detector_rate_evaluation_status"] == (
        "incomplete_detector_unavailable"
    )
    assert averaged["imri_candidate_rate_mean_per_myr"] == 0.1
    assert np.isnan(float(averaged["detectable_imri_rate_mean_per_myr"]))
    print("\nSchema and completeness checks")
    print("physical event diagnostics: present")
    print("intrinsic complete + detector unavailable: preserved separately")


SECTIONS: dict[str, Callable[[], None]] = {
    "demo": validate_demo,
    "psd": validate_psd,
    "cosmology": validate_cosmology,
    "windows": validate_windows,
    "snr": validate_snr,
    "redshift": validate_redshift_grid,
    "grid": validate_mass_redshift_grid,
    "classification": validate_classification,
    "schema": validate_schema,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate first-order physical LISA inspiral detectability."
    )
    parser.add_argument(
        "--section",
        choices=("all", *SECTIONS),
        default="all",
        help="validation section to run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    section = build_parser().parse_args(argv).section
    selected = SECTIONS.items() if section == "all" else ((section, SECTIONS[section]),)
    for _, validation in selected:
        validation()
    print("\nLISA detectability validation passed.")


if __name__ == "__main__":
    main()
