#!/usr/bin/env python3
"""Focused regression harness for gas delivery and moving LISA entry."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import physical_process.gw_frequency_radius as gw_helpers  # noqa: E402
from physical_process.damp import DampingStatus  # noqa: E402
from workflows.detect_rate import (  # noqa: E402
    classify_detectability,
    solve_time_to_observed_lisa_entry,
)
from workflows.cosmology import FlatLambdaCDM  # noqa: E402
from workflows.event_rate import (  # noqa: E402
    GAS_FINAL_RADIUS_SAMPLES,
    classify_event_rate,
    solve_gas_final_radius_cm,
)
from workflows.rate_plots import new_figure, save_figure, save_plot_data  # noqa: E402

PRIMARY_MASS_MSUN = 1.0e5
SECONDARY_MASS_MSUN = 100.0
LISA_MIN_FREQUENCY_HZ = 1.0e-3
LISA_MAX_FREQUENCY_HZ = 1.0e-1
INITIAL_REDSHIFT = 10.0
HUBBLE_CONSTANT_KM_S_MPC = 67.4
OMEGA_M = 0.315
MAX_GW_TIME_MYR = 13.6e3
ECCENTRICITY = 0.0
MISSION_DURATION_YEARS = 4.0
SNR_THRESHOLD = 20.0


@dataclass(frozen=True)
class CrossoverDiagnostic:
    reached: tuple[Any, ...]
    crossover: tuple[Any, ...]
    duty_limited: tuple[Any, ...]
    cosmic_time_unreachable: tuple[Any, ...]
    trap_bound: tuple[Any, ...]
    trap_event: Any
    callback_epochs: tuple[tuple[float, float], ...]
    plot_data: pd.DataFrame


@dataclass(frozen=True)
class DutyHandoffDiagnostic:
    entry_radius_cm: float
    event: Any
    reaches_lisa: Any
    misses_event: Any
    misses_cosmic_time: Any
    plot_data: pd.DataFrame


@dataclass(frozen=True)
class MovingEntryDiagnostic:
    high_z_entry_radius_cm: float
    high_z_start_radius_cm: float
    frozen_high_z_time_myr: float
    moving_high_z: Any
    frozen_low_z_time_myr: float
    moving_low_z: Any
    plot_data: pd.DataFrame


def _linear_migration_time_myr(
    start_radius_cm: float,
    target_radius_cm: float,
) -> float:
    return 0.1 * max(0.0, start_radius_cm - target_radius_cm)


def _gas_faster_gw_time_myr(radius_cm: float, absolute_time_myr: float) -> float:
    del absolute_time_myr
    return 0.2 * max(0.0, radius_cm - 10.0)


def _crossover_gw_time_myr(radius_cm: float, absolute_time_myr: float) -> float:
    del absolute_time_myr
    radial_distance_cm = max(0.0, radius_cm - 10.0)
    return 0.002 * radial_distance_cm**2


def _solve_synthetic_gas(
    *,
    available_gas_time_myr: float,
    gw_time_call,
):
    return solve_gas_final_radius_cm(
        r_ini_cm=100.0,
        r_gas_target_cm=10.0,
        r_entry_cm=10.0,
        handoff_required=False,
        gas_start_time_myr=0.0,
        available_gas_time_myr=available_gas_time_myr,
        migration_time_myr_call=_linear_migration_time_myr,
        gw_time_to_entry_myr=gw_time_call,
        num_radius_samples=GAS_FINAL_RADIUS_SAMPLES,
    )


@cache
def build_crossover_diagnostic() -> CrossoverDiagnostic:
    reached = _solve_synthetic_gas(
        available_gas_time_myr=10.0,
        gw_time_call=_gas_faster_gw_time_myr,
    )
    callback_epochs: list[tuple[float, float]] = []

    def record_epoch_gw_time_myr(
        radius_cm: float,
        absolute_time_myr: float,
    ) -> float:
        callback_epochs.append((radius_cm, absolute_time_myr))
        return _gas_faster_gw_time_myr(radius_cm, absolute_time_myr)

    solve_gas_final_radius_cm(
        r_ini_cm=100.0,
        r_gas_target_cm=10.0,
        r_entry_cm=10.0,
        handoff_required=False,
        gas_start_time_myr=3.0,
        available_gas_time_myr=10.0,
        migration_time_myr_call=_linear_migration_time_myr,
        gw_time_to_entry_myr=record_epoch_gw_time_myr,
        num_radius_samples=GAS_FINAL_RADIUS_SAMPLES,
    )
    crossover = _solve_synthetic_gas(
        available_gas_time_myr=10.0,
        gw_time_call=_crossover_gw_time_myr,
    )
    duty_limited = _solve_synthetic_gas(
        available_gas_time_myr=2.0,
        gw_time_call=_gas_faster_gw_time_myr,
    )
    cosmic_time_unreachable = _solve_synthetic_gas(
        available_gas_time_myr=2.0,
        gw_time_call=lambda radius_cm, absolute_time_myr: np.inf,
    )
    trap_bound = solve_gas_final_radius_cm(
        r_ini_cm=100.0,
        r_gas_target_cm=10.0,
        r_entry_cm=10.0,
        handoff_required=False,
        gas_start_time_myr=0.0,
        available_gas_time_myr=10.0,
        migration_time_myr_call=lambda start_cm, target_cm: np.inf,
        gw_time_to_entry_myr=_gas_faster_gw_time_myr,
        num_radius_samples=GAS_FINAL_RADIUS_SAMPLES,
    )
    trap_event = classify_event_rate(
        object_status=DampingStatus.EMBEDDED_BH,
        formation_time_myr=np.nan,
        damping_time_myr=1.0,
        embed_time_myr=1.0,
        r_ini_cm=100.0,
        r_gas_target_cm=10.0,
        r_entry_cm=10.0,
        handoff_required=False,
        migration_time_myr_call=lambda start_cm, target_cm: np.inf,
        gw_time_to_entry_myr_call=_gas_faster_gw_time_myr,
        duty_cycle_myr=10.0,
        num_radius_samples=GAS_FINAL_RADIUS_SAMPLES,
    )
    radii = np.linspace(10.0, 100.0, 361)
    gas_assisted_remaining = np.asarray(
        [
            _linear_migration_time_myr(radius, 10.0)
            + _crossover_gw_time_myr(10.0, _linear_migration_time_myr(100.0, radius))
            for radius in radii
        ]
    )
    plot_data = pd.DataFrame(
        {
            "radius_over_target": radii / 10.0,
            "controlled_radius_units": radii,
            "gw_remaining_myr": [
                _crossover_gw_time_myr(radius, 0.0) for radius in radii
            ],
            "gas_assisted_remaining_myr": gas_assisted_remaining,
            "is_initial_radius": np.isclose(radii, 100.0),
            "is_crossover_radius": np.isclose(radii, float(crossover[0])),
            "is_target_radius": np.isclose(radii, 10.0),
        }
    )
    return CrossoverDiagnostic(
        reached=reached,
        crossover=crossover,
        duty_limited=duty_limited,
        cosmic_time_unreachable=cosmic_time_unreachable,
        trap_bound=trap_bound,
        trap_event=trap_event,
        callback_epochs=tuple(callback_epochs),
        plot_data=plot_data,
    )


def validate_gas_boundaries() -> None:
    diagnostic = build_crossover_diagnostic()
    reached = diagnostic.reached
    assert reached[0] == 10.0
    assert reached[1] == 9.0
    assert reached[-1] == "entry_reached"

    direct_sample_radius_cm, direct_sample_time_myr = next(
        (radius_cm, time_myr)
        for radius_cm, time_myr in diagnostic.callback_epochs
        if 10.0 < radius_cm < 100.0
    )
    assert np.isclose(
        direct_sample_time_myr,
        3.0 + _linear_migration_time_myr(100.0, direct_sample_radius_cm),
    )
    assert all(
        np.isclose(time_myr, 12.0)
        for radius_cm, time_myr in diagnostic.callback_epochs
        if radius_cm == 10.0
    )

    crossover = diagnostic.crossover
    assert crossover[-1] == "gw_dominates_after_gas"
    assert np.isclose(crossover[0], 60.0, rtol=2.0e-8)
    assert np.isclose(crossover[1], 4.0, rtol=2.0e-8)
    assert crossover[0] > 10.0

    duty_limited = diagnostic.duty_limited
    assert duty_limited[-1] == "duty_cycle_limited"
    assert np.isclose(duty_limited[0], 80.0, rtol=2.0e-8)
    assert np.isclose(duty_limited[1], 2.0, rtol=2.0e-8)
    assert diagnostic.cosmic_time_unreachable[-1] == "duty_cycle_limited"
    assert diagnostic.trap_bound[0] == 100.0
    assert diagnostic.trap_bound[-1] == "gw_dominates_at_start"
    trap_event = diagnostic.trap_event
    assert trap_event.gas_stop_reason == "gw_dominates_at_start"
    assert trap_event.is_imri_candidate is True

    no_route = solve_gas_final_radius_cm(
        r_ini_cm=100.0,
        r_gas_target_cm=10.0,
        r_entry_cm=10.0,
        handoff_required=False,
        gas_start_time_myr=0.0,
        available_gas_time_myr=10.0,
        migration_time_myr_call=lambda start_cm, target_cm: np.inf,
        gw_time_to_entry_myr=lambda radius_cm, absolute_time_myr: np.inf,
        num_radius_samples=GAS_FINAL_RADIUS_SAMPLES,
    )
    assert no_route[-1] == "migration_trap_bound"


def _detect(
    *,
    radius_cm: float,
    initial_redshift: float,
    gas_final_time_myr: float = 0.0,
    max_gw_time_myr: float = MAX_GW_TIME_MYR,
):
    return classify_detectability(
        is_imri_candidate=True,
        primary_mass_msun=PRIMARY_MASS_MSUN,
        secondary_mass_msun=SECONDARY_MASS_MSUN,
        r_gas_fin_cm=radius_cm,
        gas_final_time_myr=gas_final_time_myr,
        initial_redshift=initial_redshift,
        min_observed_frequency_hz=LISA_MIN_FREQUENCY_HZ,
        max_observed_frequency_hz=LISA_MAX_FREQUENCY_HZ,
        max_gw_time_myr=max_gw_time_myr,
        eccentricity=ECCENTRICITY,
        snr_threshold=SNR_THRESHOLD,
        mission_duration_years=MISSION_DURATION_YEARS,
        detector_hubble_constant_km_s_mpc=HUBBLE_CONSTANT_KM_S_MPC,
        detector_omega_m=OMEGA_M,
    )


def _duty_limited_event(entry_radius_cm: float, radius_multiplier: float = 20.0):
    r_ini_cm = radius_multiplier * entry_radius_cm
    r_target_cm = entry_radius_cm
    migration_rate_myr_per_cm = 10.0 / (r_ini_cm - r_target_cm)

    def migration_time_myr(start_radius_cm: float, target_radius_cm: float) -> float:
        return migration_rate_myr_per_cm * max(
            0.0, start_radius_cm - target_radius_cm
        )

    def gas_faster_time_myr(radius_cm: float, absolute_time_myr: float) -> float:
        del absolute_time_myr
        return 100.0 + 2.0 * migration_time_myr(radius_cm, r_target_cm)

    return classify_event_rate(
        object_status=DampingStatus.EMBEDDED_BH,
        formation_time_myr=np.nan,
        damping_time_myr=1.0,
        embed_time_myr=1.0,
        r_ini_cm=r_ini_cm,
        r_gas_target_cm=r_target_cm,
        r_entry_cm=r_target_cm,
        handoff_required=False,
        migration_time_myr_call=migration_time_myr,
        gw_time_to_entry_myr_call=gas_faster_time_myr,
        duty_cycle_myr=3.0,
        num_radius_samples=GAS_FINAL_RADIUS_SAMPLES,
    )


@cache
def build_duty_handoff_diagnostic() -> DutyHandoffDiagnostic:
    entry_radius_cm = gw_helpers.lisa_entry_radius_cm(
        PRIMARY_MASS_MSUN,
        SECONDARY_MASS_MSUN,
        LISA_MIN_FREQUENCY_HZ,
        1.0,
    )
    event = _duty_limited_event(entry_radius_cm)
    reaches_lisa = _detect(
        radius_cm=event.r_gas_fin_cm,
        initial_redshift=1.0,
        gas_final_time_myr=event.gas_final_time_myr,
    )
    misses_event = _duty_limited_event(entry_radius_cm, radius_multiplier=1.0e3)
    misses_cosmic_time = _detect(
        radius_cm=misses_event.r_gas_fin_cm,
        initial_redshift=1.0,
        gas_final_time_myr=misses_event.gas_final_time_myr,
    )
    embedding_time_myr = 1.0
    initial_radius_cm = 20.0 * entry_radius_cm
    times = np.unique(
        np.concatenate(
            (
                np.linspace(0.0, reaches_lisa.detection_time_myr, 500),
                np.asarray(
                    [
                        embedding_time_myr,
                        event.gas_final_time_myr,
                        reaches_lisa.detection_time_myr,
                    ]
                ),
            )
        )
    )
    redshift_model = FlatLambdaCDM(HUBBLE_CONSTANT_KM_S_MPC, OMEGA_M)
    orbit_radius_cm: list[float] = []
    entry_radius_at_time_cm: list[float] = []
    redshifts: list[float] = []
    phases: list[str] = []
    for time_myr in times:
        redshift = redshift_model.evolve_redshift(1.0, float(time_myr))
        redshifts.append(redshift)
        entry_radius_at_time_cm.append(
            gw_helpers.lisa_entry_radius_cm(
                PRIMARY_MASS_MSUN,
                SECONDARY_MASS_MSUN,
                LISA_MIN_FREQUENCY_HZ,
                redshift,
            )
        )
        if time_myr <= embedding_time_myr:
            orbit_radius_cm.append(initial_radius_cm)
            phases.append("pre_embedding")
        elif time_myr <= event.gas_final_time_myr:
            gas_fraction = (time_myr - embedding_time_myr) / (
                event.gas_final_time_myr - embedding_time_myr
            )
            orbit_radius_cm.append(
                initial_radius_cm
                + gas_fraction * (event.r_gas_fin_cm - initial_radius_cm)
            )
            phases.append("gas_active")
        else:
            orbit_radius_cm.append(
                gw_helpers.peters_radius_after_time_cm(
                    PRIMARY_MASS_MSUN,
                    SECONDARY_MASS_MSUN,
                    event.r_gas_fin_cm,
                    float(time_myr - event.gas_final_time_myr),
                    ECCENTRICITY,
                    reaches_lisa.final_entry_radius_cm,
                )
            )
            phases.append("gw_only_after_duty")
    plot_data = pd.DataFrame(
        {
            "elapsed_time_myr": times,
            "phase": phases,
            "orbital_radius_over_initial_entry": (
                np.asarray(orbit_radius_cm) / entry_radius_cm
            ),
            "moving_entry_radius_over_initial_entry": (
                np.asarray(entry_radius_at_time_cm) / entry_radius_cm
            ),
            "redshift": redshifts,
            "gas_stop_reason": event.gas_stop_reason,
            "final_detection_redshift": reaches_lisa.detection_redshift,
            "is_embedding_time": np.isclose(times, embedding_time_myr),
            "is_duty_end": np.isclose(times, event.gas_final_time_myr),
            "is_physical_entry": np.isclose(
                times, reaches_lisa.detection_time_myr
            ),
        }
    )
    return DutyHandoffDiagnostic(
        entry_radius_cm=entry_radius_cm,
        event=event,
        reaches_lisa=reaches_lisa,
        misses_event=misses_event,
        misses_cosmic_time=misses_cosmic_time,
        plot_data=plot_data,
    )


def validate_duty_limited_handoff() -> None:
    diagnostic = build_duty_handoff_diagnostic()
    entry_radius_cm = diagnostic.entry_radius_cm
    event = diagnostic.event
    reaches_lisa = diagnostic.reaches_lisa
    misses_event = diagnostic.misses_event
    misses_cosmic_time = diagnostic.misses_cosmic_time
    assert event.gas_stop_reason == "duty_cycle_limited"
    assert event.is_imri_candidate is True
    assert event.gas_final_time_myr == 3.0
    assert np.isfinite(reaches_lisa.gw_time_after_gas_myr)
    assert np.isfinite(reaches_lisa.detection_time_myr)
    assert np.isfinite(reaches_lisa.detection_redshift)
    assert reaches_lisa.detect_rate_reason != "not_imri_candidate"
    assert misses_event.gas_stop_reason == "duty_cycle_limited"
    assert misses_event.is_imri_candidate is True
    assert misses_cosmic_time.is_lisa_detectable is False
    assert "misses_cosmic_time" in misses_cosmic_time.detect_rate_reason
    assert misses_cosmic_time.detect_rate_reason != "not_imri_candidate"

    redshift_at_gas_fin = FlatLambdaCDM(
        HUBBLE_CONSTANT_KM_S_MPC,
        OMEGA_M,
    ).evolve_redshift(1.0, event.gas_final_time_myr)
    print("\nRepresentative duty-limited gas -> GW case")
    print(f"initial_z={1.0:.9g}")
    print(f"gas_final_time_myr={event.gas_final_time_myr:.9g}")
    print(f"r_ini_cm={20.0 * entry_radius_cm:.9e}")
    print(f"r_gas_fin_cm={event.r_gas_fin_cm:.9e}")
    print(f"gas_stop_reason={event.gas_stop_reason}")
    print(f"z_at_gas_fin={redshift_at_gas_fin:.9g}")
    print(f"gw_time_after_gas_myr={reaches_lisa.gw_time_after_gas_myr:.9g}")
    print(f"detection_redshift={reaches_lisa.detection_redshift:.9g}")
    print(f"final_entry_radius_cm={reaches_lisa.final_entry_radius_cm:.9e}")
    print(f"final_observed_frequency_hz={reaches_lisa.observed_frequency_hz:.9e}")
    print(f"lisa_snr={reaches_lisa.lisa_snr:.9g}")
    print(f"snr_status={reaches_lisa.snr_status}")


@cache
def build_moving_entry_diagnostic() -> MovingEntryDiagnostic:
    redshift_model = FlatLambdaCDM(HUBBLE_CONSTANT_KM_S_MPC, OMEGA_M)
    high_z_entry_radius_cm = gw_helpers.lisa_entry_radius_cm(
        PRIMARY_MASS_MSUN,
        SECONDARY_MASS_MSUN,
        LISA_MIN_FREQUENCY_HZ,
        INITIAL_REDSHIFT,
    )
    high_z_start_radius_cm = 500.0 * high_z_entry_radius_cm
    frozen_high_z_time_myr = gw_helpers.peters_time_to_lisa_band_myr(
        PRIMARY_MASS_MSUN,
        SECONDARY_MASS_MSUN,
        high_z_start_radius_cm,
        LISA_MIN_FREQUENCY_HZ,
        ECCENTRICITY,
        INITIAL_REDSHIFT,
    )
    moving_high_z = solve_time_to_observed_lisa_entry(
        primary_mass_msun=PRIMARY_MASS_MSUN,
        secondary_mass_msun=SECONDARY_MASS_MSUN,
        initial_radius_cm=high_z_start_radius_cm,
        initial_redshift=INITIAL_REDSHIFT,
        start_time_myr=0.0,
        redshift_model=redshift_model,
        min_observed_frequency_hz=LISA_MIN_FREQUENCY_HZ,
        eccentricity=ECCENTRICITY,
        max_elapsed_time_myr=redshift_model.remaining_time_to_z0_myr(
            INITIAL_REDSHIFT
        ),
    )
    low_redshift = 0.1
    low_z_entry_radius_cm = gw_helpers.lisa_entry_radius_cm(
        PRIMARY_MASS_MSUN,
        SECONDARY_MASS_MSUN,
        LISA_MIN_FREQUENCY_HZ,
        low_redshift,
    )
    low_z_start_radius_cm = 2.0 * low_z_entry_radius_cm
    frozen_low_z_time_myr = gw_helpers.peters_time_to_lisa_band_myr(
        PRIMARY_MASS_MSUN,
        SECONDARY_MASS_MSUN,
        low_z_start_radius_cm,
        LISA_MIN_FREQUENCY_HZ,
        ECCENTRICITY,
        low_redshift,
    )
    moving_low_z = solve_time_to_observed_lisa_entry(
        primary_mass_msun=PRIMARY_MASS_MSUN,
        secondary_mass_msun=SECONDARY_MASS_MSUN,
        initial_radius_cm=low_z_start_radius_cm,
        initial_redshift=low_redshift,
        start_time_myr=0.0,
        redshift_model=redshift_model,
        min_observed_frequency_hz=LISA_MIN_FREQUENCY_HZ,
        eccentricity=ECCENTRICITY,
        max_elapsed_time_myr=redshift_model.remaining_time_to_z0_myr(
            low_redshift
        ),
    )
    times = np.unique(
        np.concatenate(
            (
                np.linspace(0.0, frozen_high_z_time_myr, 600),
                np.linspace(
                    moving_high_z.time_to_entry_myr,
                    frozen_high_z_time_myr,
                    200,
                ),
                np.asarray(
                    [moving_high_z.time_to_entry_myr, frozen_high_z_time_myr]
                ),
            )
        )
    )
    isco_radius_cm = gw_helpers.circular_band_boundary(
        PRIMARY_MASS_MSUN,
        SECONDARY_MASS_MSUN,
        LISA_MIN_FREQUENCY_HZ,
        INITIAL_REDSHIFT,
    ).isco_radius_cm
    orbit_radius_cm = [
        gw_helpers.peters_radius_after_time_cm(
            PRIMARY_MASS_MSUN,
            SECONDARY_MASS_MSUN,
            high_z_start_radius_cm,
            float(time_myr),
            ECCENTRICITY,
            isco_radius_cm,
        )
        for time_myr in times
    ]
    redshifts = [
        redshift_model.evolve_redshift(INITIAL_REDSHIFT, float(time_myr))
        for time_myr in times
    ]
    moving_entry_radius_cm = [
        gw_helpers.lisa_entry_radius_cm(
            PRIMARY_MASS_MSUN,
            SECONDARY_MASS_MSUN,
            LISA_MIN_FREQUENCY_HZ,
            redshift,
        )
        for redshift in redshifts
    ]
    plot_data = pd.DataFrame(
        {
            "elapsed_time_myr": times,
            "gw_radius_over_initial_entry": (
                np.asarray(orbit_radius_cm) / high_z_entry_radius_cm
            ),
            "moving_entry_radius_over_initial_entry": (
                np.asarray(moving_entry_radius_cm) / high_z_entry_radius_cm
            ),
            "frozen_entry_radius_over_initial_entry": np.ones_like(times),
            "redshift": redshifts,
            "is_moving_boundary_arrival": times
            == moving_high_z.time_to_entry_myr,
            "is_frozen_boundary_arrival": times == frozen_high_z_time_myr,
        }
    )
    return MovingEntryDiagnostic(
        high_z_entry_radius_cm=high_z_entry_radius_cm,
        high_z_start_radius_cm=high_z_start_radius_cm,
        frozen_high_z_time_myr=frozen_high_z_time_myr,
        moving_high_z=moving_high_z,
        frozen_low_z_time_myr=frozen_low_z_time_myr,
        moving_low_z=moving_low_z,
        plot_data=plot_data,
    )


def validate_moving_entry_boundary() -> None:
    diagnostic = build_moving_entry_diagnostic()
    moving_high_z = diagnostic.moving_high_z
    assert moving_high_z.status == "entry_reached"
    assert moving_high_z.time_to_entry_myr < diagnostic.frozen_high_z_time_myr
    assert moving_high_z.entry_radius_cm > 1.5 * diagnostic.high_z_entry_radius_cm
    assert np.isclose(
        moving_high_z.observed_frequency_hz,
        LISA_MIN_FREQUENCY_HZ,
        rtol=2.0e-10,
    )
    moving_low_z = diagnostic.moving_low_z
    assert moving_low_z.status == "entry_reached"
    assert np.isclose(
        moving_low_z.time_to_entry_myr,
        diagnostic.frozen_low_z_time_myr,
        rtol=2.0e-6,
    )


def validate_single_agn_fixed_primary() -> None:
    initial_entry_radius_cm = gw_helpers.lisa_entry_radius_cm(
        PRIMARY_MASS_MSUN,
        SECONDARY_MASS_MSUN,
        LISA_MIN_FREQUENCY_HZ,
        INITIAL_REDSHIFT,
    )
    decision = _detect(
        radius_cm=initial_entry_radius_cm,
        initial_redshift=INITIAL_REDSHIFT,
    )
    assert decision.cosmic_merge_applied is False
    assert decision.final_smbh_mass_msun == PRIMARY_MASS_MSUN
    assert np.isfinite(decision.detection_time_myr)
    expected_entry_radius_cm = gw_helpers.lisa_entry_radius_cm(
        PRIMARY_MASS_MSUN,
        SECONDARY_MASS_MSUN,
        LISA_MIN_FREQUENCY_HZ,
        decision.detection_redshift,
    )
    assert np.isclose(decision.final_entry_radius_cm, expected_entry_radius_cm)
    assert np.isclose(
        decision.observed_frequency_hz,
        LISA_MIN_FREQUENCY_HZ,
        rtol=2.0e-10,
    )

    time_limited = _detect(
        radius_cm=2.0 * initial_entry_radius_cm,
        initial_redshift=INITIAL_REDSHIFT,
        max_gw_time_myr=1.0e-12,
    )
    assert time_limited.detect_rate_reason.endswith("misses_gw_time_budget")


def _plot_crossover(output_dir: Path) -> None:
    diagnostic = build_crossover_diagnostic()
    data = diagnostic.plot_data
    save_plot_data(data, output_dir / "data" / "gas_gw_crossover.csv")
    fig, axis = new_figure("Gas–GW handoff regression: crossover case")
    axis.plot(
        data["radius_over_target"],
        data["gw_remaining_myr"],
        linewidth=2.1,
        label=r"$T_{\rm GW}$ remaining",
    )
    axis.plot(
        data["radius_over_target"],
        data["gas_assisted_remaining_myr"],
        linewidth=2.1,
        label=r"$T_{\rm gas-assisted}$ remaining",
    )
    for radius, label, color in (
        (10.0, "target", "#009E73"),
        (float(diagnostic.crossover[0]), "crossover", "#D55E00"),
        (100.0, "initial", "#0072B2"),
    ):
        axis.axvline(radius / 10.0, color=color, linestyle="--", linewidth=1.2)
        axis.text(
            radius / 10.0,
            0.98,
            label,
            transform=axis.get_xaxis_transform(),
            rotation=90,
            ha="right",
            va="top",
            color=color,
            fontsize=9,
        )
    axis.set_xlabel(r"Controlled radius $r/r_{\rm target}$")
    axis.set_ylabel("Remaining time [Myr]")
    axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir / "gas_gw_crossover")


def _plot_duty_handoff(output_dir: Path) -> None:
    diagnostic = build_duty_handoff_diagnostic()
    data = diagnostic.plot_data
    save_plot_data(data, output_dir / "data" / "duty_cycle_to_gw_trajectory.csv")
    fig, axis = new_figure("Duty-limited gas transport followed by GW entry")
    for phase, label, color in (
        ("pre_embedding", "Before embedding", "0.55"),
        ("gas_active", "Gas-active transport", "#009E73"),
        ("gw_only_after_duty", "GW-only after duty end", "#0072B2"),
    ):
        segment = data[data["phase"].eq(phase)]
        axis.plot(
            segment["elapsed_time_myr"],
            segment["orbital_radius_over_initial_entry"],
            color=color,
            linewidth=2.2,
            label=label,
        )
    axis.plot(
        data["elapsed_time_myr"],
        data["moving_entry_radius_over_initial_entry"],
        color="#CC79A7",
        linestyle="--",
        linewidth=1.8,
        label=r"Moving $r_{\rm entry}[z(t)]$",
    )
    for marker_column, label, marker in (
        ("is_embedding_time", "embedding", "s"),
        ("is_duty_end", "gas ends at duty limit", "D"),
        ("is_physical_entry", "physical LISA entry", "*"),
    ):
        point = data[data[marker_column]].iloc[0]
        axis.scatter(
            [point["elapsed_time_myr"]],
            [point["orbital_radius_over_initial_entry"]],
            marker=marker,
            s=75,
            color="black",
            zorder=5,
            label=label,
        )
    axis.text(
        0.98,
        0.04,
        f"entry redshift = {diagnostic.reaches_lisa.detection_redshift:.3f}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    axis.set_xlabel("Elapsed time [Myr]")
    axis.set_ylabel(r"Radius $r/r_{\rm entry,initial}$")
    axis.set_yscale("log")
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_dir / "duty_cycle_to_gw_trajectory")


def _plot_moving_entry(output_dir: Path) -> None:
    diagnostic = build_moving_entry_diagnostic()
    data = diagnostic.plot_data
    save_plot_data(
        data, output_dir / "data" / "moving_vs_frozen_lisa_entry.csv"
    )
    fig, axis = new_figure("Moving versus frozen LISA-entry boundary: controlled case")
    axis.plot(
        data["elapsed_time_myr"],
        data["gw_radius_over_initial_entry"],
        linewidth=2.2,
        color="#0072B2",
        label=r"Peters-evolved $r_{\rm GW}(t)$",
    )
    axis.plot(
        data["elapsed_time_myr"],
        data["moving_entry_radius_over_initial_entry"],
        linewidth=1.8,
        linestyle="--",
        color="#CC79A7",
        label=r"Moving $r_{\rm entry}[z(t)]$",
    )
    axis.plot(
        data["elapsed_time_myr"],
        data["frozen_entry_radius_over_initial_entry"],
        linewidth=1.6,
        linestyle=":",
        color="#D55E00",
        label=r"Frozen $r_{\rm entry}(z_{\rm initial})$",
    )
    for marker_column, label, marker, color in (
        ("is_moving_boundary_arrival", "moving arrival", "o", "#CC79A7"),
        ("is_frozen_boundary_arrival", "frozen arrival", "x", "#D55E00"),
    ):
        point = data[data[marker_column]].iloc[0]
        axis.scatter(
            [point["elapsed_time_myr"]],
            [point["gw_radius_over_initial_entry"]],
            marker=marker,
            s=70,
            color=color,
            zorder=5,
            label=label,
        )
    axis.set_xlabel("Elapsed time [Myr]")
    axis.set_ylabel(r"Radius $r/r_{\rm entry}(z_{\rm initial})$")
    axis.set_yscale("log")
    axis.legend(frameon=False, fontsize=8)
    arrival_delta_myr = (
        diagnostic.frozen_high_z_time_myr
        - diagnostic.moving_high_z.time_to_entry_myr
    )
    zoom = axis.inset_axes((0.47, 0.47, 0.50, 0.34))
    focused = data[
        data["elapsed_time_myr"].between(
            diagnostic.moving_high_z.time_to_entry_myr,
            diagnostic.frozen_high_z_time_myr,
        )
    ]
    focused_time_years = 1.0e6 * (
        focused["elapsed_time_myr"]
        - diagnostic.moving_high_z.time_to_entry_myr
    )
    zoom.plot(
        focused_time_years,
        focused["gw_radius_over_initial_entry"],
        color="#0072B2",
        linewidth=1.5,
    )
    zoom.plot(
        focused_time_years,
        focused["moving_entry_radius_over_initial_entry"],
        color="#CC79A7",
        linestyle="--",
        linewidth=1.3,
    )
    zoom.plot(
        focused_time_years,
        focused["frozen_entry_radius_over_initial_entry"],
        color="#D55E00",
        linestyle=":",
        linewidth=1.3,
    )
    zoom.set_title(
        rf"Arrival separation: $\Delta t={arrival_delta_myr:.3e}$ Myr",
        fontsize=8,
    )
    zoom.set_xlabel("Time after moving arrival [yr]", fontsize=7)
    zoom.tick_params(labelsize=7)
    zoom.grid(True, color="0.9", linewidth=0.6)
    fig.tight_layout()
    save_figure(fig, output_dir / "moving_vs_frozen_lisa_entry")


def generate_plots(output_dir: Path) -> None:
    _plot_crossover(output_dir)
    _plot_duty_handoff(output_dir)
    _plot_moving_entry(output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate gas delivery and moving LISA-entry handoff."
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="write controlled diagnostic plots and their CSV data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diagnostics/gas_gw_handoff"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    validate_gas_boundaries()
    validate_duty_limited_handoff()
    validate_moving_entry_boundary()
    validate_single_agn_fixed_primary()
    if arguments.plots:
        generate_plots(arguments.output_dir)
    print("\nGas/GW handoff regression checks passed.")


if __name__ == "__main__":
    main()
