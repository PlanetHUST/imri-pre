#!/usr/bin/env python3
"""Standalone validation harness for first-order physical LISA detectability."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Sequence, cast
from unittest.mock import patch

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from lisa.snr import (  # noqa: E402
    SNR_STATUS_EVALUATED,
    SNR_STATUS_MODEL_UNAVAILABLE,
    SNR_STATUS_NUMERICAL_FAILURE,
    SNR_STATUS_NO_OVERLAP,
    SNR_STATUS_REQUIRED_INPUT_UNAVAILABLE,
    SNR_WINDOW_PLACEMENT_BAND_ENTRY,
    SNR_WINDOW_PLACEMENT_OPTIMAL,
    evaluate_lisa_inspiral_snr,
    fourier_inspiral_amplitude_hz_inverse,
    inspiral_time_to_coalescence_s,
    integrated_lisa_snr,
    lisa_confusion_strain_psd_hz,
    lisa_effective_strain_psd_hz,
    lisa_instrument_strain_psd_hz,
    luminosity_distance_mpc,
    mission_observation_window as _mission_observation_window,
    observed_frequency_from_time_to_coalescence_hz,
    source_chirp_mass_msun,
)
import physical_process.gw_frequency_radius as gw_helpers  # noqa: E402
from physical_process.constants import JULIAN_YEAR_S  # noqa: E402
from lisa.detectability import (  # noqa: E402
    DETECTION_CLASS_DETECTABLE,
    DETECTION_CLASS_NOT_EVALUATED,
    classify_detectability,
)
from workflows.models import IMRIDynAveragedResult, IMRIDynSingleResult  # noqa: E402
from workflows.config import IMRIDynConfig  # noqa: E402
from workflows.outputs import (  # noqa: E402
    IMRI_EVENT_COLUMNS,
    IMRI_REPEAT_RESULT_COLUMNS,
    IMRI_RESULT_COLUMNS,
    save_events,
    save_repeat_results,
    save_results,
)
from viz.rate_plots import new_figure, save_figure, save_plot_data  # noqa: E402

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
REDSHIFT_GRID = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 13.0, 15.0)
PRIMARY_MASS_GRID_MSUN = (1e4, 1e5, 1e6, 1e7)
SECONDARY_MASS_GRID_MSUN = (10.0, 100.0, 1000.0)


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


@cache
def build_psd_benchmark_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frequency_hz": PSD_FREQUENCIES_HZ,
            "instrument_psd_hz_inv": [
                lisa_instrument_strain_psd_hz(frequency_hz)
                for frequency_hz in PSD_FREQUENCIES_HZ
            ],
            "confusion_psd_hz_inv": [
                lisa_confusion_strain_psd_hz(frequency_hz, DEMO_MISSION_YEARS)
                for frequency_hz in PSD_FREQUENCIES_HZ
            ],
            "total_psd_hz_inv": [
                lisa_effective_strain_psd_hz(frequency_hz, DEMO_MISSION_YEARS)
                for frequency_hz in PSD_FREQUENCIES_HZ
            ],
        }
    )


@cache
def build_sensitivity_snr_density_data() -> tuple[pd.DataFrame, float]:
    result = demo_result()
    snr_frequency_hz = np.exp(
        np.linspace(np.log(result.f_low_hz), np.log(result.f_high_hz), 200001)
    )
    display_frequency_hz = np.geomspace(1.0e-4, DEMO_F_MAX_HZ, 1200)
    frequency_hz = np.unique(
        np.concatenate(
            (
                display_frequency_hz,
                snr_frequency_hz,
                np.asarray([result.f_low_hz, result.f_high_hz]),
            )
        )
    )
    instrument_psd = np.asarray(
        [lisa_instrument_strain_psd_hz(frequency) for frequency in frequency_hz]
    )
    confusion_psd = np.asarray(
        [
            lisa_confusion_strain_psd_hz(frequency, result.mission_duration_years)
            for frequency in frequency_hz
        ]
    )
    total_psd = instrument_psd + confusion_psd
    in_snr_window = (frequency_hz >= result.f_low_hz) & (
        frequency_hz <= result.f_high_hz
    )
    snr_density = np.full(frequency_hz.shape, np.nan)
    amplitudes = np.asarray(
        [
            fourier_inspiral_amplitude_hz_inverse(
                frequency,
                result.chirp_mass_detector_msun,
                result.luminosity_distance_mpc,
            )
            for frequency in frequency_hz[in_snr_window]
        ]
    )
    snr_density[in_snr_window] = (
        16.0
        / 5.0
        * frequency_hz[in_snr_window]
        * amplitudes**2
        / total_psd[in_snr_window]
    )
    snr_rows = in_snr_window
    grid_snr = float(
        np.sqrt(
            np.trapezoid(
                snr_density[snr_rows],
                np.log(frequency_hz[snr_rows]),
            )
        )
    )
    plot_rows = ~in_snr_window
    window_indices = np.flatnonzero(in_snr_window)
    plot_rows[window_indices[::50]] = True
    plot_rows[window_indices[-1]] = True
    return (
        pd.DataFrame(
            {
                "frequency_hz": frequency_hz[plot_rows],
                "instrument_asd_hz_mhalf": np.sqrt(instrument_psd[plot_rows]),
                "confusion_asd_hz_mhalf": np.sqrt(confusion_psd[plot_rows]),
                "total_asd_hz_mhalf": np.sqrt(total_psd[plot_rows]),
                "snr_squared_density_per_ln_frequency": snr_density[plot_rows],
                "in_detector_band": (
                    frequency_hz[plot_rows] >= DEMO_F_MIN_HZ
                )
                & (frequency_hz[plot_rows] <= DEMO_F_MAX_HZ),
                "in_snr_window": in_snr_window[plot_rows],
                "f_low_hz": result.f_low_hz,
                "f_high_hz": result.f_high_hz,
                "f_isco_observer_hz": result.f_isco_observer_hz,
                "integrated_lisa_snr": result.lisa_snr,
                "snr_threshold": result.snr_threshold,
                "observation_case": result.observation_case,
            }
        ),
        grid_snr,
    )


@cache
def build_redshift_grid_data() -> pd.DataFrame:
    records = []
    for redshift in REDSHIFT_GRID:
        result = demo_result(redshift)
        records.append(
            {
                "redshift": redshift,
                "luminosity_distance_mpc": result.luminosity_distance_mpc,
                "chirp_mass_detector_msun": result.chirp_mass_detector_msun,
                "f_isco_observer_hz": result.f_isco_observer_hz,
                "f_low_hz": result.f_low_hz,
                "f_high_hz": result.f_high_hz,
                "lisa_snr": (
                    result.lisa_snr
                    if result.status == SNR_STATUS_EVALUATED
                    else np.nan
                ),
                "snr_status": result.status,
                "is_detectable": result.is_detectable,
            }
        )
    return pd.DataFrame.from_records(records)


@cache
def build_mass_redshift_grid_data() -> pd.DataFrame:
    records = []
    for m1_source_msun in PRIMARY_MASS_GRID_MSUN:
        for m2_source_msun in SECONDARY_MASS_GRID_MSUN:
            for redshift in REDSHIFT_GRID:
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
                records.append(
                    {
                        "m1_source_msun": m1_source_msun,
                        "m2_source_msun": m2_source_msun,
                        "redshift": redshift,
                        "luminosity_distance_mpc": result.luminosity_distance_mpc,
                        "chirp_mass_detector_msun": (
                            result.chirp_mass_detector_msun
                        ),
                        "f_isco_observer_hz": result.f_isco_observer_hz,
                        "f_low_hz": result.f_low_hz,
                        "f_high_hz": result.f_high_hz,
                        "lisa_snr": (
                            result.lisa_snr
                            if result.status == SNR_STATUS_EVALUATED
                            else np.nan
                        ),
                        "snr_status": result.status,
                        "is_detectable": result.is_detectable,
                    }
                )
    return pd.DataFrame.from_records(records)


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
    diagnostic = build_psd_benchmark_data()
    values = diagnostic["total_psd_hz_inv"].to_numpy(dtype=float)
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
    def mission_observation_window(**kwargs):
        """Pin these cases to the legacy band-entry start semantics."""
        return _mission_observation_window(
            window_placement=SNR_WINDOW_PLACEMENT_BAND_ENTRY, **kwargs
        )

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

    validate_optimal_placement_windows()


def validate_optimal_placement_windows() -> None:
    """Optimal placement must close the window at the loud end of the track."""
    result = demo_result()
    detector_chirp_mass_msun = result.chirp_mass_detector_msun

    # A source sitting at band entry, still far from merger: the mission is
    # moved forward to the observed ISCO instead of sampling the faint start.
    slow_chirp_mass_msun = (1.0 + 0.1) * source_chirp_mass_msun(10.0, 10.0)
    optimal_slow = _mission_observation_window(
        current_observed_frequency_hz=DEMO_F_MIN_HZ,
        f_isco_observer_hz=0.1,
        chirp_mass_detector_msun=slow_chirp_mass_msun,
        detector_min_frequency_hz=DEMO_F_MIN_HZ,
        detector_max_frequency_hz=DEMO_F_MAX_HZ,
        mission_duration_years=DEMO_MISSION_YEARS,
        window_placement=SNR_WINDOW_PLACEMENT_OPTIMAL,
    )
    band_entry_slow = _mission_observation_window(
        current_observed_frequency_hz=DEMO_F_MIN_HZ,
        f_isco_observer_hz=0.1,
        chirp_mass_detector_msun=slow_chirp_mass_msun,
        detector_min_frequency_hz=DEMO_F_MIN_HZ,
        detector_max_frequency_hz=DEMO_F_MAX_HZ,
        mission_duration_years=DEMO_MISSION_YEARS,
        window_placement=SNR_WINDOW_PLACEMENT_BAND_ENTRY,
    )
    assert optimal_slow.status == SNR_STATUS_EVALUATED
    assert optimal_slow.observation_case == (
        "optimal_mission_limited_pre_merger:reaches_isco_during_mission"
    )
    # Window closes at the top of the available band, not just above f_min.
    assert np.isclose(optimal_slow.f_high_hz, DEMO_F_MAX_HZ)
    assert optimal_slow.f_low_hz > band_entry_slow.f_high_hz
    assert np.isclose(
        optimal_slow.observation_duration_years, DEMO_MISSION_YEARS
    )

    # The mission never starts below the band or before the source is there.
    assert optimal_slow.f_low_hz >= DEMO_F_MIN_HZ

    # A source whose whole residual inspiral is shorter than the mission keeps
    # the full track and reports the true (shorter) observation duration.
    tau_to_isco_s = inspiral_time_to_coalescence_s(
        DEMO_F_MIN_HZ, detector_chirp_mass_msun
    ) - inspiral_time_to_coalescence_s(
        result.f_isco_observer_hz, detector_chirp_mass_msun
    )
    assert tau_to_isco_s < DEMO_MISSION_YEARS * JULIAN_YEAR_S
    optimal_full = _mission_observation_window(
        current_observed_frequency_hz=DEMO_F_MIN_HZ,
        f_isco_observer_hz=result.f_isco_observer_hz,
        chirp_mass_detector_msun=detector_chirp_mass_msun,
        detector_min_frequency_hz=DEMO_F_MIN_HZ,
        detector_max_frequency_hz=DEMO_F_MAX_HZ,
        mission_duration_years=DEMO_MISSION_YEARS,
        window_placement=SNR_WINDOW_PLACEMENT_OPTIMAL,
    )
    assert optimal_full.observation_case == (
        "optimal_full_residual_inspiral:reaches_isco_during_mission"
    )
    assert np.isclose(optimal_full.f_low_hz, DEMO_F_MIN_HZ)
    assert np.isclose(optimal_full.f_high_hz, result.f_isco_observer_hz)
    assert optimal_full.observation_duration_years < DEMO_MISSION_YEARS

    # No-overlap guards are shared with band-entry placement.
    optimal_no_overlap = _mission_observation_window(
        current_observed_frequency_hz=DEMO_F_MIN_HZ,
        f_isco_observer_hz=5.0e-4,
        chirp_mass_detector_msun=detector_chirp_mass_msun,
        detector_min_frequency_hz=DEMO_F_MIN_HZ,
        detector_max_frequency_hz=DEMO_F_MAX_HZ,
        mission_duration_years=DEMO_MISSION_YEARS,
        window_placement=SNR_WINDOW_PLACEMENT_OPTIMAL,
    )
    assert optimal_no_overlap.status == SNR_STATUS_NO_OVERLAP

    with np.testing.assert_raises(ValueError):
        _mission_observation_window(
            current_observed_frequency_hz=DEMO_F_MIN_HZ,
            f_isco_observer_hz=result.f_isco_observer_hz,
            chirp_mass_detector_msun=detector_chirp_mass_msun,
            detector_min_frequency_hz=DEMO_F_MIN_HZ,
            detector_max_frequency_hz=DEMO_F_MAX_HZ,
            mission_duration_years=DEMO_MISSION_YEARS,
            window_placement="not_a_placement",
        )

    # Optimal placement can only ever help: same source, never a lower SNR.
    for redshift in (1.0, 5.0, 10.0, 15.0):
        common = dict(
            m1_source_msun=1.0e4,
            m2_source_msun=50.0,
            redshift=redshift,
            current_observed_frequency_hz=DEMO_F_MIN_HZ,
            mission_duration_years=DEMO_MISSION_YEARS,
            detector_min_frequency_hz=DEMO_F_MIN_HZ,
            detector_max_frequency_hz=DEMO_F_MAX_HZ,
            snr_threshold=20.0,
        )
        entry_snr = evaluate_lisa_inspiral_snr(
            **common, window_placement=SNR_WINDOW_PLACEMENT_BAND_ENTRY
        )
        optimal_snr = evaluate_lisa_inspiral_snr(
            **common, window_placement=SNR_WINDOW_PLACEMENT_OPTIMAL
        )
        assert optimal_snr.status == SNR_STATUS_EVALUATED
        assert optimal_snr.window_placement == SNR_WINDOW_PLACEMENT_OPTIMAL
        assert optimal_snr.lisa_snr >= entry_snr.lisa_snr

    print("\nOptimal-placement observation windows")
    for name, window in (
        ("slow_source_moved_to_isco", optimal_slow),
        ("short_residual_keeps_track", optimal_full),
    ):
        print(
            f"{name:28s} {window.observation_case:60s} "
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

    _, grid_snr = build_sensitivity_snr_density_data()
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
        "lisa.snr.integrated_lisa_snr",
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
    for row in build_redshift_grid_data().to_dict(orient="records"):
        assert row["snr_status"] in {SNR_STATUS_EVALUATED, SNR_STATUS_NO_OVERLAP}
        print(
            f"{row['redshift']:4.1f}  {row['luminosity_distance_mpc']:12.3f}  "
            f"{row['chirp_mass_detector_msun']:11.3f}  "
            f"{row['f_low_hz']:10.3e}  {row['f_high_hz']:10.3e}  "
            f"{row['lisa_snr']:8.3f}  {row['snr_status']}"
        )


def validate_mass_redshift_grid() -> None:
    print("\nControlled mass-redshift selection grid")
    print("m1       m2       z     d_l_mpc     f_isco_obs  f_low       f_high      snr       class")
    for row in build_mass_redshift_grid_data().to_dict(orient="records"):
        assert row["snr_status"] in {SNR_STATUS_EVALUATED, SNR_STATUS_NO_OVERLAP}
        classification = (
            "detectable"
            if row["is_detectable"] is True
            else "below"
            if row["snr_status"] == SNR_STATUS_EVALUATED
            else "no_overlap"
        )
        print(
            f"{row['m1_source_msun']:7.0e}  {row['m2_source_msun']:7.0e}  "
            f"{row['redshift']:4.1f}  {row['luminosity_distance_mpc']:10.1f}  "
            f"{row['f_isco_observer_hz']:10.3e}  {row['f_low_hz']:10.3e}  "
            f"{row['f_high_hz']:10.3e}  {row['lisa_snr']:8.2f}  {classification}"
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
        "detection_time_myr",
        "detection_redshift",
        "r_det_entry_cm",
        "r_det_entry_rg",
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

    result = IMRIDynSingleResult(1.0e5, 2.0e5, 0, "torque_with_trap", 123)
    result.imri_count = 1
    result.imri_mass_msun = 100.0
    result.imri_rate_per_myr = 0.1
    result.imri_mass_rate_msun_per_myr = 10.0
    result.detector_unavailable_imri_count = 1
    result.detectable_imri_rate_per_myr = np.nan
    result.detectable_imri_mass_rate_msun_per_myr = np.nan
    averaged = IMRIDynAveragedResult(
        1.0e5, 2.0e5, "torque_with_trap", [result]
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


def validate_output_semantics() -> None:
    """Check that public CSV writers count finite physical entries."""
    result = IMRIDynSingleResult(1.0e5, 2.0e5, 0, "torque_with_trap", 123)
    result.imri_count = 2
    result.imri_mass_msun = 200.0
    result.imri_rate_per_myr = 0.2
    result.detectable_imri_count = 1
    result.no_detectable_imri_count = 1
    result.detector_evaluated_imri_count = 2
    result.detector_no_overlap_imri_count = 1
    result.detectable_imri_rate_per_myr = 0.1
    result.imri_events = cast(Any, [
        {
            "is_imri_candidate": True,
            "detection_time_myr": 3.0,
            "detection_redshift": 1.0,
            "r_det_entry_cm": 2.0e12,
            "r_det_entry_rg": 100.0,
            "f_entry_src_hz": 2.0e-3,
            "f_entry_obs_hz": 1.0e-3,
            "is_lisa_detectable": True,
            "snr_status": SNR_STATUS_EVALUATED,
            "mass_msun": 100.0,
            "intrinsic_rate_contribution_per_myr": 0.1,
            "detected_rate_contribution_per_myr": 0.1,
        },
        {
            "is_imri_candidate": True,
            "detection_time_myr": np.inf,
            "detection_redshift": np.nan,
            "r_det_entry_cm": np.nan,
            "r_det_entry_rg": np.nan,
            "f_entry_src_hz": np.nan,
            "f_entry_obs_hz": np.nan,
            "is_lisa_detectable": False,
            "snr_status": SNR_STATUS_NO_OVERLAP,
            "mass_msun": 100.0,
            "intrinsic_rate_contribution_per_myr": 0.1,
            "detected_rate_contribution_per_myr": 0.0,
        },
    ])
    averaged = IMRIDynAveragedResult(
        1.0e5, 2.0e5, "torque_with_trap", [result]
    )
    unavailable_result = IMRIDynSingleResult(
        2.0e5, 2.0e5, 0, "torque_with_trap", 456
    )
    unavailable_result.imri_count = 1
    unavailable_result.imri_mass_msun = 100.0
    unavailable_result.imri_rate_per_myr = 0.1
    unavailable_result.detector_unavailable_imri_count = 1
    unavailable_result.imri_events = cast(
        Any,
        [
            {
                "is_imri_candidate": True,
                "detection_time_myr": np.inf,
                "is_lisa_detectable": None,
                "snr_status": SNR_STATUS_MODEL_UNAVAILABLE,
                "mass_msun": 100.0,
                "intrinsic_rate_contribution_per_myr": 0.1,
                "detected_rate_contribution_per_myr": np.nan,
            }
        ],
    )
    unavailable_averaged = IMRIDynAveragedResult(
        2.0e5, 2.0e5, "torque_with_trap", [unavailable_result]
    )
    config = object.__new__(IMRIDynConfig)
    config.redshift = 1.0
    config.base_seed = 123
    config.duty_cycle_myr = 10.0
    with TemporaryDirectory(prefix="imri-output-test-") as temporary_dir:
        output_dir = Path(temporary_dir)
        output_results = [averaged, unavailable_averaged]
        repeat_frame = save_repeat_results(output_results, output_dir, config)
        averaged_frame = save_results(output_results, output_dir)
        event_frame = save_events(output_results, output_dir)
    for frame in (repeat_frame, averaged_frame):
        row = frame[np.isclose(frame["m_smbh_msun"], 1.0e5)].iloc[0]
        assert row["n_imri"] == 1
        assert row["n_det"] == 1
        assert np.isclose(row["rate_imri_per_myr"], 0.1)
        assert np.isclose(row["rate_det_per_myr"], 0.1)
    assert "r_det_entry_cm" in event_frame
    assert np.allclose(
        event_frame["intrinsic_rate_contribution_per_myr"].iloc[:2], [0.1, 0.0]
    )
    for frame in (repeat_frame, averaged_frame):
        unavailable_row = frame[np.isclose(frame["m_smbh_msun"], 2.0e5)].iloc[0]
        assert unavailable_row["rate_status"] == (
            "incomplete_physical_entry_unavailable"
        )
        assert np.isnan(unavailable_row["rate_imri_per_myr"])
    assert np.isnan(event_frame["intrinsic_rate_contribution_per_myr"].iloc[2])
    print("\nOutput semantics")
    print("finite physical entry: n_imri=1; downstream-only eligibility excluded")


def _mark_non_evaluated_states(
    axis,
    data: pd.DataFrame,
    used_labels: set[str] | None = None,
) -> None:
    styles = (
        (SNR_STATUS_NO_OVERLAP, "no overlap", "x", "#D55E00", 0.05),
        (
            SNR_STATUS_REQUIRED_INPUT_UNAVAILABLE,
            "unavailable",
            "s",
            "0.35",
            0.10,
        ),
        (SNR_STATUS_MODEL_UNAVAILABLE, "unavailable", "s", "0.35", 0.10),
        (SNR_STATUS_NUMERICAL_FAILURE, "numerical failure", "P", "0.35", 0.15),
    )
    labels = used_labels if used_labels is not None else set()
    for status, label, marker, color, axis_height in styles:
        rows = data[data["snr_status"].eq(status)]
        if rows.empty:
            continue
        axis.scatter(
            rows["redshift"],
            np.full(len(rows), axis_height),
            transform=axis.get_xaxis_transform(),
            marker=marker,
            color=color,
            s=48,
            clip_on=False,
            label=label if label not in labels else None,
        )
        labels.add(label)


def plot_sensitivity_snr_density(output_dir: Path) -> None:
    data, grid_snr = build_sensitivity_snr_density_data()
    save_plot_data(
        data, output_dir / "data" / "lisa_sensitivity_snr_density.csv"
    )
    result = demo_result()
    fig, axes = new_figure(
        "LISA sensitivity and demo-source SNR accumulation",
        nrows=2,
        ncols=1,
        figsize=(7.0, 8.0),
    )
    sensitivity_axis = axes[0, 0]
    density_axis = axes[1, 0]
    sensitivity_axis.loglog(
        data["frequency_hz"],
        data["instrument_asd_hz_mhalf"],
        linewidth=1.7,
        label=r"$\sqrt{S_{\rm instrument}}$",
    )
    sensitivity_axis.loglog(
        data["frequency_hz"],
        data["confusion_asd_hz_mhalf"],
        linewidth=1.7,
        label=r"$\sqrt{S_{\rm confusion}}$",
    )
    sensitivity_axis.loglog(
        data["frequency_hz"],
        data["total_asd_hz_mhalf"],
        color="black",
        linewidth=2.1,
        label=r"$\sqrt{S_{\rm total}}$",
    )
    sensitivity_axis.axvspan(
        DEMO_F_MIN_HZ,
        DEMO_F_MAX_HZ,
        color="#56B4E9",
        alpha=0.10,
        label="detector band",
    )
    for frequency, label, color in (
        (result.f_low_hz, r"$f_{\rm low}$", "#009E73"),
        (result.f_high_hz, r"$f_{\rm high}$", "#D55E00"),
        (result.f_isco_observer_hz, r"$f_{\rm ISCO,obs}$", "#CC79A7"),
    ):
        sensitivity_axis.axvline(
            frequency, color=color, linestyle="--", linewidth=1.1, label=label
        )
    sensitivity_axis.set_ylabel(r"ASD [Hz$^{-1/2}$]")
    sensitivity_axis.legend(frameon=False, fontsize=8, ncol=2)

    window = data["in_snr_window"]
    density_axis.semilogx(
        data.loc[window, "frequency_hz"],
        data.loc[window, "snr_squared_density_per_ln_frequency"],
        color="#0072B2",
        linewidth=2.0,
    )
    density_axis.axvspan(
        result.f_low_hz, result.f_high_hz, color="#009E73", alpha=0.10
    )
    density_axis.set_xlabel("Observed frequency [Hz]")
    density_axis.set_ylabel(r"$d\rho^2/d\ln f$")
    density_axis.text(
        0.03,
        0.96,
        (
            f"integrated $\\rho$ = {grid_snr:.2f}\n"
            f"threshold = {result.snr_threshold:g}\n{result.observation_case}"
        ),
        transform=density_axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    save_figure(fig, output_dir / "lisa_sensitivity_snr_density")


def plot_snr_vs_redshift(output_dir: Path) -> None:
    data = build_redshift_grid_data()
    save_plot_data(data, output_dir / "data" / "lisa_snr_vs_redshift.csv")
    fig, axis = new_figure("LISA SNR versus redshift: demo source")
    evaluated = cast(
        pd.DataFrame,
        data.loc[
            cast(pd.Series, data["snr_status"]).eq(SNR_STATUS_EVALUATED)
        ],
    )
    detectable = cast(
        pd.DataFrame,
        evaluated.loc[cast(pd.Series, evaluated["is_detectable"]).eq(True)],
    )
    below = cast(
        pd.DataFrame,
        evaluated.loc[cast(pd.Series, evaluated["is_detectable"]).eq(False)],
    )
    axis.plot(
        evaluated["redshift"],
        evaluated["lisa_snr"],
        color="0.55",
        linewidth=1.2,
    )
    axis.scatter(
        detectable["redshift"],
        detectable["lisa_snr"],
        color="#0072B2",
        marker="o",
        label="evaluated: detectable",
    )
    axis.scatter(
        below["redshift"],
        below["lisa_snr"],
        facecolors="white",
        edgecolors="#E69F00",
        marker="o",
        label="evaluated: below threshold",
    )
    _mark_non_evaluated_states(axis, data)
    axis.axhline(DEMO_SNR_THRESHOLD, color="black", linestyle="--", label="threshold")
    axis.set_xlabel("Redshift $z$")
    axis.set_ylabel(r"LISA SNR $\rho$")
    axis.set_yscale("log")
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_dir / "lisa_snr_vs_redshift")


def plot_mass_redshift_grid(output_dir: Path) -> None:
    data = build_mass_redshift_grid_data()
    save_plot_data(data, output_dir / "data" / "lisa_mass_redshift_grid.csv")
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    for m2_source_msun in SECONDARY_MASS_GRID_MSUN:
        selected = cast(
            pd.DataFrame,
            data.loc[
                np.isclose(
                    np.asarray(data["m2_source_msun"], dtype=float),
                    m2_source_msun,
                )
            ],
        )
        fig, axis = new_figure(
            f"LISA detectability: secondary mass {m2_source_msun:g} $M_\\odot$"
        )
        used_status_labels: set[str] = set()
        for m1_source_msun, color in zip(
            PRIMARY_MASS_GRID_MSUN, colors, strict=True
        ):
            curve = cast(
                pd.DataFrame,
                selected.loc[
                    np.isclose(
                        np.asarray(selected["m1_source_msun"], dtype=float),
                        m1_source_msun,
                    )
                ],
            ).sort_values(by="redshift")
            evaluated = cast(
                pd.DataFrame,
                curve.loc[
                    cast(pd.Series, curve["snr_status"]).eq(SNR_STATUS_EVALUATED)
                ],
            )
            axis.plot(
                evaluated["redshift"],
                evaluated["lisa_snr"],
                color=color,
                marker="o",
                linewidth=1.8,
                label=rf"$M_1={m1_source_msun:.0e}\,M_\odot$",
            )
            _mark_non_evaluated_states(axis, curve, used_status_labels)
        axis.axhline(
            DEMO_SNR_THRESHOLD, color="black", linestyle="--", label="threshold"
        )
        axis.set_xlabel("Redshift $z$")
        axis.set_ylabel(r"LISA SNR $\rho$")
        axis.set_yscale("log")
        axis.legend(frameon=False, fontsize=8, ncol=2)
        fig.tight_layout()
        mass_token = int(m2_source_msun)
        save_figure(
            fig, output_dir / f"lisa_detectability_m2_{mass_token}_msun"
        )


def generate_plots(output_dir: Path) -> None:
    plot_sensitivity_snr_density(output_dir)
    plot_snr_vs_redshift(output_dir)
    plot_mass_redshift_grid(output_dir)


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
    "output": validate_output_semantics,
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
    parser.add_argument(
        "--plots",
        action="store_true",
        help="write diagnostic plots and their CSV data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diagnostics/lisa_detectability"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    section = arguments.section
    selected = SECTIONS.items() if section == "all" else ((section, SECTIONS[section]),)
    for _, validation in selected:
        validation()
    if arguments.plots:
        generate_plots(arguments.output_dir)
    print("\nLISA detectability validation passed.")


if __name__ == "__main__":
    main()
