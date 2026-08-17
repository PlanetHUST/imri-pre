"""Diagnose active-node damping and the circular-migration handoff contract."""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for path in (SOURCE_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from src.disk_quantity.pagn_disk_adapter import PagnDiskProfile, build_pagn_disk_profile
from src.physical_process import constants as const
from src.physical_process.agn import AGNProperties
from src.physical_process.damp import (
    MAX_EMBEDDED_ECCENTRICITY,
    DampingCalculator,
    DampingOutcome,
    DampingStatus,
    DragProperties,
    OrbitalState,
    TimescaleData,
)
from src.physical_process.nsc_orbital import disk_crossing_mask, orbital_node_radii_cm
from src.workflows.config import DISK_ACCRETION_RATE_G_S_PER_MSUN, PAGN_ALPHA
from src.workflows.models import DampingTask
from src.workflows.workers import init_damping_worker, process_damping_task

SMBH_MASS_MSUN = 1.0e3
OBJECT_MASS_MSUN = 30.0
AGN_LIFETIME_MYR = 10.0
FORMATION_TIME_MYR = 100.0
PAGN_MODEL = "sirko"
PAGN_EDDINGTON_RATIO = 0.1
PAGN_B = 0
PAGN_RESOLUTION = 1000
USE_DISK_INNER_RADIUS = True

HIGH_INCLINATION_DEG = 20.0
LOW_INCLINATION_H_OVER_R_FACTOR = 0.5
BOUNDARY_ECCENTRICITY = 0.002
BOUNDARY_COS_OMEGA = 0.5
FALSE_OVERLAP_ECCENTRICITY = 0.99999999

# Pairwise heatmaps show the median over the third coordinate. Thresholds not
# reached by the production integration remain censored as NaN.
SCAN_AXIS_MIN_RG = 1.0e3
SCAN_AXIS_MAX_OUTER_FRACTION = 0.3
SCAN_AXIS_COUNT = 9
SCAN_ECCENTRICITY_MIN = 0.05
SCAN_ECCENTRICITY_MAX = 0.9
SCAN_INCLINATION_MIN_DEG = 10.0
SCAN_INCLINATION_MAX_DEG = 170.0
SCAN_COS_OMEGA = 0.5
SCAN_MAX_ITER = 20000

OUTPUT_DIR = Path("figs")
OUTPUT_STEM = "damping_check"
HEATMAP_STEM = "damping_timescale_heatmaps"


@dataclass(frozen=True)
class DampingCase:
    name: str
    object_type: str
    semi_major_axis_outer_fraction: float
    eccentricity: float
    cos_omega: float
    expected_node_flags: tuple[bool, bool]
    expected_status: DampingStatus
    max_iter: int
    inclination_deg: float | None = None
    inclination_h_over_r_factor: float | None = None


@dataclass(frozen=True)
class DampingCaseResult:
    case_name: str
    object_type: str
    smbh_mass_msun: float
    object_mass_msun: float
    circularity_threshold: float
    disk_inner_radius_cm: float
    disk_outer_radius_cm: float
    disk_inner_radius_rg: float
    disk_outer_radius_rg: float
    initial_semi_major_axis_cm: float
    initial_semi_major_axis_rg: float
    final_semi_major_axis_cm: float
    final_semi_major_axis_rg: float
    initial_eccentricity: float
    final_eccentricity: float
    initial_inclination_rad: float
    initial_inclination_deg: float
    final_inclination_rad: float
    final_inclination_deg: float
    final_inclination_limit_rad: float
    final_inclination_limit_deg: float
    initial_node_1_radius_rg: float
    initial_node_2_radius_rg: float
    initial_node_1_in_disk: bool
    initial_node_2_in_disk: bool
    final_node_1_radius_rg: float
    final_node_2_radius_rg: float
    final_node_1_in_disk: bool
    final_node_2_in_disk: bool
    final_periapsis_rg: float
    final_apoapsis_rg: float
    elapsed_damping_time_myr: float
    status: str
    low_inclination: bool
    sufficiently_circular: bool
    radially_contained: bool
    torque_migration_eligible: bool


@dataclass(frozen=True)
class DampingScanResult:
    axis_rg: np.ndarray
    eccentricity: np.ndarray
    inclination_deg: np.ndarray
    inclination_time_myr: np.ndarray
    eccentricity_time_myr: np.ndarray
    rows: list[dict[str, float | bool | str]]


@dataclass(frozen=True)
class DiagnosticDisk:
    profile: PagnDiskProfile
    agn: AGNProperties
    calculator: DampingCalculator


CASES = (
    DampingCase(
        name="both_nodes_inside",
        object_type="Star",
        semi_major_axis_outer_fraction=0.1,
        eccentricity=0.1,
        inclination_deg=HIGH_INCLINATION_DEG,
        cos_omega=0.5,
        expected_node_flags=(True, True),
        expected_status=DampingStatus.DAMPING_INCOMPLETE,
        max_iter=0,
    ),
    DampingCase(
        name="only_node_1_inside",
        object_type="Star",
        semi_major_axis_outer_fraction=1.0,
        eccentricity=BOUNDARY_ECCENTRICITY,
        inclination_deg=HIGH_INCLINATION_DEG,
        cos_omega=-BOUNDARY_COS_OMEGA,
        expected_node_flags=(True, False),
        expected_status=DampingStatus.DAMPING_INCOMPLETE,
        max_iter=0,
    ),
    DampingCase(
        name="only_node_2_inside",
        object_type="Star",
        semi_major_axis_outer_fraction=1.0,
        eccentricity=BOUNDARY_ECCENTRICITY,
        inclination_deg=HIGH_INCLINATION_DEG,
        cos_omega=BOUNDARY_COS_OMEGA,
        expected_node_flags=(False, True),
        expected_status=DampingStatus.DAMPING_INCOMPLETE,
        max_iter=0,
    ),
    DampingCase(
        name="false_interval_overlap",
        object_type="Star",
        semi_major_axis_outer_fraction=0.75,
        eccentricity=FALSE_OVERLAP_ECCENTRICITY,
        inclination_deg=HIGH_INCLINATION_DEG,
        cos_omega=1.0,
        expected_node_flags=(False, False),
        expected_status=DampingStatus.OUTSIDE_DISK,
        max_iter=10,
    ),
    DampingCase(
        name="low_i_low_e_inside",
        object_type="Star",
        semi_major_axis_outer_fraction=0.1,
        eccentricity=0.005,
        inclination_h_over_r_factor=LOW_INCLINATION_H_OVER_R_FACTOR,
        cos_omega=0.5,
        expected_node_flags=(True, True),
        expected_status=DampingStatus.EMBEDDED_STAR,
        max_iter=0,
    ),
    DampingCase(
        name="low_i_eccentric",
        object_type="Star",
        semi_major_axis_outer_fraction=0.1,
        eccentricity=0.02,
        inclination_h_over_r_factor=LOW_INCLINATION_H_OVER_R_FACTOR,
        cos_omega=0.5,
        expected_node_flags=(True, True),
        expected_status=DampingStatus.DAMPING_INCOMPLETE,
        max_iter=0,
    ),
    DampingCase(
        name="small_i_beyond_q1",
        object_type="Star",
        semi_major_axis_outer_fraction=1.01,
        eccentricity=0.0,
        inclination_deg=0.1,
        cos_omega=0.5,
        expected_node_flags=(False, False),
        expected_status=DampingStatus.OUTSIDE_DISK,
        max_iter=10,
    ),
    DampingCase(
        name="evolved_trajectory",
        object_type="Star",
        semi_major_axis_outer_fraction=0.1,
        eccentricity=0.3,
        inclination_deg=HIGH_INCLINATION_DEG,
        cos_omega=0.5,
        expected_node_flags=(True, True),
        expected_status=DampingStatus.EMBEDDED_STAR,
        max_iter=50000,
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="directory for diagnostic PNG and CSV files",
    )
    return parser.parse_args()


def _build_disk() -> DiagnosticDisk:
    profile = build_pagn_disk_profile(
        SMBH_MASS_MSUN,
        PAGN_ALPHA,
        model=PAGN_MODEL,
        n_resolution=PAGN_RESOLUTION,
        eddington_ratio=PAGN_EDDINGTON_RATIO,
        b=PAGN_B,
        accretion_rate_g_s=DISK_ACCRETION_RATE_G_S_PER_MSUN * SMBH_MASS_MSUN,
        suppress_pagn_output=True,
    )
    agn = AGNProperties.from_mass(SMBH_MASS_MSUN, AGN_LIFETIME_MYR)
    calculator = DampingCalculator(
        agn,
        verbose=False,
        radius_grid_cm=np.asarray(profile.radius_cm, dtype=float),
        surface_density_g_cm2=np.asarray(profile.surface_density_g_cm2, dtype=float),
        aspect_ratio=np.asarray(profile.aspect_ratio, dtype=float),
        disk_inner_radius_cm=profile.inner_radius_cm,
        disk_outer_radius_cm=profile.outer_radius_cm,
        use_disk_inner_radius=USE_DISK_INNER_RADIUS,
    )
    return DiagnosticDisk(profile=profile, agn=agn, calculator=calculator)


def _case_state(case: DampingCase, disk: DiagnosticDisk) -> OrbitalState:
    semi_major_axis_cm = (
        case.semi_major_axis_outer_fraction * disk.profile.outer_radius_cm
    )
    if case.inclination_h_over_r_factor is not None:
        reference_state = OrbitalState(
            semi_major_axis_cm=semi_major_axis_cm,
            eccentricity=case.eccentricity,
            inclination_rad=0.0,
            cos_omega=case.cos_omega,
        )
        local_h_over_r = disk.calculator.inclination_limit_rad(reference_state)
        if not np.isfinite(local_h_over_r):
            raise ValueError(f"{case.name}: local H/R is unavailable")
        inclination_rad = case.inclination_h_over_r_factor * local_h_over_r
    elif case.inclination_deg is not None:
        inclination_rad = float(np.deg2rad(case.inclination_deg))
    else:
        raise ValueError(f"{case.name}: inclination input is missing")
    return OrbitalState(
        semi_major_axis_cm=semi_major_axis_cm,
        eccentricity=case.eccentricity,
        inclination_rad=float(inclination_rad),
        cos_omega=case.cos_omega,
        time_s=0.0,
    )


def _node_data(
    calculator: DampingCalculator,
    state: OrbitalState,
) -> tuple[tuple[float, float], tuple[bool, bool]]:
    node_1_cm, node_2_cm = orbital_node_radii_cm(
        state.semi_major_axis_cm,
        state.eccentricity,
        state.cos_omega,
    )
    node_radii_cm = (float(node_1_cm), float(node_2_cm))
    return node_radii_cm, calculator.timescale_calc.active_node_mask(*node_radii_cm)


def _result_from_outcome(
    disk: DiagnosticDisk,
    case: DampingCase,
    initial_state: OrbitalState,
    outcome: DampingOutcome,
) -> DampingCaseResult:
    rg_cm = disk.agn.gravitational_radius_cm
    initial_nodes_cm, initial_flags = _node_data(disk.calculator, initial_state)
    final_state = outcome.final_state
    final_nodes_cm, final_flags = _node_data(disk.calculator, final_state)
    criteria = disk.calculator.embedding_criteria(final_state)
    return DampingCaseResult(
        case_name=case.name,
        object_type=case.object_type,
        smbh_mass_msun=SMBH_MASS_MSUN,
        object_mass_msun=OBJECT_MASS_MSUN,
        circularity_threshold=MAX_EMBEDDED_ECCENTRICITY,
        disk_inner_radius_cm=disk.profile.inner_radius_cm,
        disk_outer_radius_cm=disk.profile.outer_radius_cm,
        disk_inner_radius_rg=disk.profile.inner_radius_cm / rg_cm,
        disk_outer_radius_rg=disk.profile.outer_radius_cm / rg_cm,
        initial_semi_major_axis_cm=initial_state.semi_major_axis_cm,
        initial_semi_major_axis_rg=initial_state.semi_major_axis_cm / rg_cm,
        final_semi_major_axis_cm=final_state.semi_major_axis_cm,
        final_semi_major_axis_rg=final_state.semi_major_axis_cm / rg_cm,
        initial_eccentricity=initial_state.eccentricity,
        final_eccentricity=final_state.eccentricity,
        initial_inclination_rad=initial_state.inclination_rad,
        initial_inclination_deg=float(np.degrees(initial_state.inclination_rad)),
        final_inclination_rad=final_state.inclination_rad,
        final_inclination_deg=float(np.degrees(final_state.inclination_rad)),
        final_inclination_limit_rad=criteria.inclination_limit_rad,
        final_inclination_limit_deg=float(np.degrees(criteria.inclination_limit_rad)),
        initial_node_1_radius_rg=initial_nodes_cm[0] / rg_cm,
        initial_node_2_radius_rg=initial_nodes_cm[1] / rg_cm,
        initial_node_1_in_disk=initial_flags[0],
        initial_node_2_in_disk=initial_flags[1],
        final_node_1_radius_rg=final_nodes_cm[0] / rg_cm,
        final_node_2_radius_rg=final_nodes_cm[1] / rg_cm,
        final_node_1_in_disk=final_flags[0],
        final_node_2_in_disk=final_flags[1],
        final_periapsis_rg=criteria.periapsis_cm / rg_cm,
        final_apoapsis_rg=criteria.apoapsis_cm / rg_cm,
        elapsed_damping_time_myr=outcome.elapsed_time_s / const.Myr,
        status=outcome.status.value,
        low_inclination=criteria.low_inclination,
        sufficiently_circular=criteria.sufficiently_circular,
        radially_contained=criteria.radially_contained,
        torque_migration_eligible=criteria.eligible,
    )


def _run_cases(
    disk: DiagnosticDisk,
) -> tuple[list[DampingCaseResult], list[OrbitalState], DampingOutcome]:
    results: list[DampingCaseResult] = []
    trajectory: list[OrbitalState] = []
    trajectory_outcome: DampingOutcome | None = None
    object_mass_g = OBJECT_MASS_MSUN * const.msun
    for case in CASES:
        state = _case_state(case, disk)
        _, initial_flags = _node_data(disk.calculator, state)
        assert initial_flags == case.expected_node_flags
        selected_by_population = disk_crossing_mask(
            np.asarray([state.semi_major_axis_cm]),
            np.asarray([state.eccentricity]),
            np.asarray([state.cos_omega]),
            disk.profile.inner_radius_cm,
            disk.profile.outer_radius_cm,
            USE_DISK_INNER_RADIUS,
        )
        assert bool(selected_by_population[0]) is any(initial_flags)
        history = trajectory if case.name == "evolved_trajectory" else None
        outcome = disk.calculator.damping_simulation(
            object_mass_g,
            state,
            FORMATION_TIME_MYR,
            case.object_type,
            max_iter=case.max_iter,
            history=history,
        )
        assert outcome.status is case.expected_status, (
            f"{case.name}: expected {case.expected_status.value}, "
            f"got {outcome.status.value}"
        )
        results.append(_result_from_outcome(disk, case, state, outcome))
        if case.name == "evolved_trajectory":
            trajectory_outcome = outcome
    assert trajectory_outcome is not None
    return results, trajectory, trajectory_outcome


def _first_passage_times_myr(
    calculator: DampingCalculator,
    history: list[OrbitalState],
) -> tuple[float, float]:
    inclination_time_myr = next(
        (
            state.time_s / const.Myr
            for state in history
            if calculator.inclination_is_damped(state)
        ),
        np.nan,
    )
    eccentricity_time_myr = next(
        (
            state.time_s / const.Myr
            for state in history
            if state.eccentricity <= MAX_EMBEDDED_ECCENTRICITY
        ),
        np.nan,
    )
    return float(inclination_time_myr), float(eccentricity_time_myr)


def _run_full_space_scan(disk: DiagnosticDisk) -> DampingScanResult:
    rg_cm = disk.agn.gravitational_radius_cm
    axis_rg = np.logspace(
        np.log10(SCAN_AXIS_MIN_RG),
        np.log10(SCAN_AXIS_MAX_OUTER_FRACTION * disk.profile.outer_radius_cm / rg_cm),
        SCAN_AXIS_COUNT,
    )
    eccentricity = np.linspace(
        SCAN_ECCENTRICITY_MIN,
        SCAN_ECCENTRICITY_MAX,
        SCAN_AXIS_COUNT,
    )
    inclination_deg = np.linspace(
        SCAN_INCLINATION_MIN_DEG,
        SCAN_INCLINATION_MAX_DEG,
        SCAN_AXIS_COUNT,
    )
    shape = (axis_rg.size, eccentricity.size, inclination_deg.size)
    inclination_time_myr = np.full(shape, np.nan)
    eccentricity_time_myr = np.full(shape, np.nan)
    rows: list[dict[str, float | bool | str]] = []
    object_mass_g = OBJECT_MASS_MSUN * const.msun

    for axis_index, initial_axis_rg in enumerate(axis_rg):
        for eccentricity_index, initial_eccentricity in enumerate(eccentricity):
            for inclination_index, initial_inclination_deg in enumerate(
                inclination_deg
            ):
                state = OrbitalState(
                    semi_major_axis_cm=float(initial_axis_rg * rg_cm),
                    eccentricity=float(initial_eccentricity),
                    inclination_rad=float(np.deg2rad(initial_inclination_deg)),
                    cos_omega=SCAN_COS_OMEGA,
                )
                history: list[OrbitalState] = []
                outcome = disk.calculator.damping_simulation(
                    object_mass_g,
                    state,
                    FORMATION_TIME_MYR,
                    "Star",
                    max_iter=SCAN_MAX_ITER,
                    history=history,
                )
                inclination_time, eccentricity_time = _first_passage_times_myr(
                    disk.calculator,
                    history,
                )
                inclination_time_myr[
                    axis_index, eccentricity_index, inclination_index
                ] = inclination_time
                eccentricity_time_myr[
                    axis_index, eccentricity_index, inclination_index
                ] = eccentricity_time
                rows.append(
                    {
                        "initial_semi_major_axis_rg": float(initial_axis_rg),
                        "initial_eccentricity": float(initial_eccentricity),
                        "initial_inclination_deg": float(initial_inclination_deg),
                        "cos_omega": SCAN_COS_OMEGA,
                        "inclination_time_myr": inclination_time,
                        "eccentricity_time_myr": eccentricity_time,
                        "inclination_reached": bool(np.isfinite(inclination_time)),
                        "eccentricity_reached": bool(np.isfinite(eccentricity_time)),
                        "final_semi_major_axis_rg": (
                            outcome.final_state.semi_major_axis_cm / rg_cm
                        ),
                        "final_eccentricity": outcome.final_state.eccentricity,
                        "final_inclination_deg": float(
                            np.degrees(outcome.final_state.inclination_rad)
                        ),
                        "elapsed_time_myr": outcome.elapsed_time_s / const.Myr,
                        "status": outcome.status.value,
                    }
                )
    return DampingScanResult(
        axis_rg=axis_rg,
        eccentricity=eccentricity,
        inclination_deg=inclination_deg,
        inclination_time_myr=inclination_time_myr,
        eccentricity_time_myr=eccentricity_time_myr,
        rows=rows,
    )


def _assert_single_active_node_behavior(
    disk: DiagnosticDisk,
    case_name: str,
) -> None:
    case = next(case for case in CASES if case.name == case_name)
    state = _case_state(case, disk)
    timescales = disk.calculator.timescale_calc.compute_all_timescales(
        OBJECT_MASS_MSUN * const.msun,
        state,
        case.object_type,
    )
    assert timescales is not None
    active_mask = timescales["active_node_mask"]
    assert sum(active_mask) == 1
    inactive_index = 0 if not active_mask[0] else 1
    disk_properties = timescales["disk"]
    drag = timescales["drag"]
    assert np.isnan(disk_properties.surface_density_g_cm2[inactive_index])
    assert np.isnan(disk_properties.density_g_cm3[inactive_index])
    assert np.isnan(drag.drag_radius_cm[inactive_index])
    assert np.isnan(drag.relative_velocity_cm_s[inactive_index])
    assert np.isinf(drag.drag_time_s[inactive_index])
    assert timescales["crossing_time_s"][inactive_index] == 0.0

    coefficients = disk.calculator.compute_ode_coefficients(
        state,
        *drag.drag_time_s,
        *timescales["crossing_time_s"],
        active_mask,
    )
    inactive_indices = (0, 2, 4) if inactive_index == 0 else (1, 3, 5)
    assert all(coefficients[index] == 0.0 for index in inactive_indices)

    original_step_s = disk.calculator.damping_time_step_s(timescales, 0.01)
    changed_drag_times = list(drag.drag_time_s)
    changed_crossing_times = list(timescales["crossing_time_s"])
    changed_drag_times[inactive_index] = 1.0e-99
    changed_crossing_times[inactive_index] = 1.0e99
    changed_timescales: TimescaleData = {
        **timescales,
        "drag": DragProperties(
            drag_radius_cm=drag.drag_radius_cm,
            relative_velocity_cm_s=drag.relative_velocity_cm_s,
            drag_time_s=(changed_drag_times[0], changed_drag_times[1]),
        ),
        "crossing_time_s": (
            changed_crossing_times[0],
            changed_crossing_times[1],
        ),
    }
    changed_step_s = disk.calculator.damping_time_step_s(changed_timescales, 0.01)
    assert original_step_s == changed_step_s


def _assert_worker_serialization(
    disk: DiagnosticDisk,
    trajectory_outcome: DampingOutcome,
) -> None:
    case = next(case for case in CASES if case.name == "evolved_trajectory")
    initial_state = _case_state(case, disk)
    init_damping_worker(
        disk.agn,
        np.asarray(disk.profile.radius_cm, dtype=float),
        np.asarray(disk.profile.surface_density_g_cm2, dtype=float),
        np.asarray(disk.profile.aspect_ratio, dtype=float),
        disk.profile.inner_radius_cm,
        disk.profile.outer_radius_cm,
        USE_DISK_INNER_RADIUS,
    )
    task: DampingTask = {
        "task_kind": "diagnostic",
        "idx": 0,
        "mass_g": OBJECT_MASS_MSUN * const.msun,
        "semi_major_axis_cm": initial_state.semi_major_axis_cm,
        "eccentricity": initial_state.eccentricity,
        "inclination_rad": initial_state.inclination_rad,
        "cos_omega": initial_state.cos_omega,
        "formation_time_myr": FORMATION_TIME_MYR,
        "state_time_s": initial_state.time_s,
    }
    worker_result = process_damping_task(task)
    final_state = trajectory_outcome.final_state
    assert worker_result["status"] is trajectory_outcome.status
    assert worker_result["final_semi_major_axis_cm"] == final_state.semi_major_axis_cm
    assert worker_result["final_eccentricity"] == final_state.eccentricity
    assert worker_result["final_inclination_rad"] == final_state.inclination_rad
    assert worker_result["final_cos_omega"] == final_state.cos_omega


def _assert_results(results: list[DampingCaseResult]) -> None:
    false_overlap = next(
        result for result in results if result.case_name == "false_interval_overlap"
    )
    old_overlap_predicate = (
        min(
            false_overlap.initial_node_1_radius_rg,
            false_overlap.initial_node_2_radius_rg,
        )
        < false_overlap.disk_outer_radius_rg
        and max(
            false_overlap.initial_node_1_radius_rg,
            false_overlap.initial_node_2_radius_rg,
        )
        > false_overlap.disk_inner_radius_rg
    )
    assert old_overlap_predicate
    assert not (
        false_overlap.initial_node_1_in_disk or false_overlap.initial_node_2_in_disk
    )

    embedded_statuses = {
        DampingStatus.EMBEDDED_STAR.value,
        DampingStatus.EMBEDDED_BH.value,
    }
    for result in results:
        if result.status not in embedded_statuses:
            continue
        assert result.low_inclination
        assert result.final_inclination_rad <= result.final_inclination_limit_rad
        assert result.sufficiently_circular
        assert result.radially_contained
        assert result.torque_migration_eligible
        assert result.disk_inner_radius_cm <= result.final_semi_major_axis_cm
        assert result.final_semi_major_axis_cm <= result.disk_outer_radius_cm
        assert np.all(
            np.isfinite(
                (
                    result.final_semi_major_axis_cm,
                    result.final_eccentricity,
                    result.final_inclination_rad,
                )
            )
        )

    eccentric = next(
        result for result in results if result.case_name == "low_i_eccentric"
    )
    outside_q1 = next(
        result for result in results if result.case_name == "small_i_beyond_q1"
    )
    assert eccentric.low_inclination and not eccentric.sufficiently_circular
    assert eccentric.status not in embedded_statuses
    assert not outside_q1.radially_contained
    assert outside_q1.status not in embedded_statuses


def _assert_scan(scan: DampingScanResult) -> None:
    expected_shape = (SCAN_AXIS_COUNT, SCAN_AXIS_COUNT, SCAN_AXIS_COUNT)
    assert len(scan.rows) == SCAN_AXIS_COUNT**3
    assert scan.inclination_time_myr.shape == expected_shape
    assert scan.eccentricity_time_myr.shape == expected_shape
    for row in scan.rows:
        elapsed_time_myr = float(row["elapsed_time_myr"])
        for key in ("inclination_time_myr", "eccentricity_time_myr"):
            threshold_time_myr = float(row[key])
            if np.isfinite(threshold_time_myr):
                assert 0.0 <= threshold_time_myr <= elapsed_time_myr


def _trajectory_rows(
    disk: DiagnosticDisk,
    trajectory: list[OrbitalState],
) -> list[dict[str, float | bool | int]]:
    rg_cm = disk.agn.gravitational_radius_cm
    rows: list[dict[str, float | bool | int]] = []
    for index, state in enumerate(trajectory):
        nodes_cm, flags = _node_data(disk.calculator, state)
        inclination_limit_rad = disk.calculator.inclination_limit_rad(state)
        rows.append(
            {
                "step": index,
                "time_myr": state.time_s / const.Myr,
                "semi_major_axis_rg": state.semi_major_axis_cm / rg_cm,
                "node_1_radius_rg": nodes_cm[0] / rg_cm,
                "node_2_radius_rg": nodes_cm[1] / rg_cm,
                "node_1_in_disk": flags[0],
                "node_2_in_disk": flags[1],
                "eccentricity": state.eccentricity,
                "eccentricity_over_threshold": (
                    state.eccentricity / MAX_EMBEDDED_ECCENTRICITY
                ),
                "inclination_rad": state.inclination_rad,
                "inclination_deg": float(np.degrees(state.inclination_rad)),
                "local_aspect_ratio": inclination_limit_rad,
                "inclination_over_aspect_ratio": (
                    abs(state.inclination_rad) / inclination_limit_rad
                ),
            }
        )
    return rows


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _plot_trajectory_and_cases(
    output_dir: Path,
    results: list[DampingCaseResult],
    trajectory_rows: list[dict[str, float | bool | int]],
    inclination_time_myr: float,
    eccentricity_time_myr: float,
) -> Path:
    time_myr = np.asarray([row["time_myr"] for row in trajectory_rows], dtype=float)
    outer_radius_rg = results[0].disk_outer_radius_rg
    axis_over_outer = (
        np.asarray([row["semi_major_axis_rg"] for row in trajectory_rows], dtype=float)
        / outer_radius_rg
    )
    node_1_over_outer = (
        np.asarray([row["node_1_radius_rg"] for row in trajectory_rows], dtype=float)
        / outer_radius_rg
    )
    node_2_over_outer = (
        np.asarray([row["node_2_radius_rg"] for row in trajectory_rows], dtype=float)
        / outer_radius_rg
    )
    eccentricity_ratio = np.asarray(
        [row["eccentricity_over_threshold"] for row in trajectory_rows],
        dtype=float,
    )
    inclination_ratio = np.asarray(
        [row["inclination_over_aspect_ratio"] for row in trajectory_rows],
        dtype=float,
    )

    fig = plt.figure(figsize=(13.0, 9.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.45))
    radius_ax = fig.add_subplot(grid[0, 0])
    threshold_ax = fig.add_subplot(grid[0, 1])
    criteria_ax = fig.add_subplot(grid[1, :])

    radius_ax.plot(time_myr, axis_over_outer, label="$a/R_{Q=1}$", linewidth=2.0)
    radius_ax.plot(
        time_myr,
        node_1_over_outer,
        label="$r_{node,1}/R_{Q=1}$",
        linestyle="--",
    )
    radius_ax.plot(
        time_myr,
        node_2_over_outer,
        label="$r_{node,2}/R_{Q=1}$",
        linestyle=":",
    )
    radius_ax.axhline(1.0, color="#D55E00", linestyle="--", label="Q=1 edge")
    radius_ax.set_xlabel("Elapsed damping time [Myr]")
    radius_ax.set_ylabel("Radius / $R_{Q=1}$")
    radius_ax.set_title("A. Evolving orbit: values below 1 are inside Q=1")
    radius_ax.legend(fontsize=8)
    radius_ax.ticklabel_format(
        axis="x",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )
    radius_ax.text(
        0.02,
        0.88,
        f"$R_{{in}}/R_{{Q=1}}={results[0].disk_inner_radius_rg / outer_radius_rg:.2e}$",
        transform=radius_ax.transAxes,
        fontsize=8,
    )

    threshold_ax.plot(
        time_myr,
        eccentricity_ratio,
        color="#1f77b4",
        label="$e/e_{cir}$",
    )
    threshold_ax.plot(
        time_myr,
        inclination_ratio,
        color="#BE9A62",
        label="$|i|/(H/R)(a)$",
    )
    threshold_ax.axhline(1.0, color="black", linestyle="--", label="criterion")
    if np.isfinite(eccentricity_time_myr):
        threshold_ax.axvline(
            eccentricity_time_myr,
            color="#1f77b4",
            linestyle=":",
            label=f"$t_{{ecc}}={eccentricity_time_myr:.2e}$ Myr",
        )
    if np.isfinite(inclination_time_myr):
        threshold_ax.axvline(
            inclination_time_myr,
            color="#BE9A62",
            linestyle=":",
            label=f"$t_{{inc}}={inclination_time_myr:.2e}$ Myr",
        )
    threshold_ax.set_yscale("log")
    threshold_ax.set_ylim(bottom=0.5)
    threshold_ax.set_xlabel("Elapsed damping time [Myr]")
    threshold_ax.set_ylabel("Value / embedding threshold")
    threshold_ax.set_title("B. First passage occurs when each curve reaches 1")
    threshold_ax.legend(fontsize=8)
    threshold_ax.ticklabel_format(
        axis="x",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )

    criteria = np.asarray(
        [
            (
                result.low_inclination,
                result.sufficiently_circular,
                result.radially_contained,
                result.torque_migration_eligible,
            )
            for result in results
        ],
        dtype=float,
    )
    criteria_ax.imshow(criteria, aspect="auto", vmin=0.0, vmax=1.0, cmap="RdYlGn")
    criteria_ax.set_xticks(
        range(4),
        (
            r"$|i|\leq(H/R)(a)$",
            r"$e\leq0.01$",
            "peri/apo inside disk",
            "migration eligible",
        ),
    )
    criteria_ax.set_yticks(
        range(len(results)),
        [f"{result.case_name}  [{result.status}]" for result in results],
    )
    criteria_ax.set_title("C. Final-state predicates: green=true, red=false")
    for row_index in range(criteria.shape[0]):
        for column_index in range(criteria.shape[1]):
            criteria_ax.text(
                column_index,
                row_index,
                "true" if criteria[row_index, column_index] else "false",
                ha="center",
                va="center",
                fontsize=8,
            )

    path = output_dir / f"{OUTPUT_STEM}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _median_projection(values: np.ndarray, axis: int) -> np.ndarray:
    return np.asarray(
        np.ma.median(np.ma.masked_invalid(values), axis=axis).filled(np.nan),
        dtype=float,
    )


def _plot_scan_heatmaps(output_dir: Path, scan: DampingScanResult) -> Path:
    inclination_maps = (
        _median_projection(scan.inclination_time_myr, 2).T,
        _median_projection(scan.inclination_time_myr, 1).T,
        _median_projection(scan.inclination_time_myr, 0).T,
    )
    eccentricity_maps = (
        _median_projection(scan.eccentricity_time_myr, 2).T,
        _median_projection(scan.eccentricity_time_myr, 1).T,
        _median_projection(scan.eccentricity_time_myr, 0).T,
    )
    map_values = (*inclination_maps, *eccentricity_maps)
    finite_values = [
        values[np.isfinite(values) & (values > 0.0)] for values in map_values
    ]
    all_values = np.concatenate(finite_values)
    if all_values.size == 0:
        raise RuntimeError("the damping scan produced no finite threshold times")
    norm = LogNorm(vmin=float(np.min(all_values)), vmax=float(np.max(all_values)))
    color_map = plt.get_cmap("viridis").copy()
    color_map.set_bad("#d9d9d9")

    log_axis_rg = np.log10(scan.axis_rg)
    coordinates = (
        (
            log_axis_rg,
            scan.eccentricity,
            r"$\log_{10}(a_i/r_g)$",
            "$e_i$",
            "median over $i_i$",
        ),
        (
            log_axis_rg,
            scan.inclination_deg,
            r"$\log_{10}(a_i/r_g)$",
            "$i_i$ [deg]",
            "median over $e_i$",
        ),
        (
            scan.eccentricity,
            scan.inclination_deg,
            "$e_i$",
            "$i_i$ [deg]",
            "median over $a_i$",
        ),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.5), constrained_layout=True)
    image = None
    for row_index, (maps, time_label) in enumerate(
        (
            (inclination_maps, r"$t_{damp,inc}$: $|i|\leq(H/R)(a)$"),
            (eccentricity_maps, r"$t_{damp,ecc}$: $e\leq0.01$"),
        )
    ):
        for column_index, (values, coordinate) in enumerate(zip(maps, coordinates)):
            x_values, y_values, x_label, y_label, reduction = coordinate
            image = axes[row_index, column_index].pcolormesh(
                x_values,
                y_values,
                values,
                shading="nearest",
                cmap=color_map,
                norm=norm,
            )
            axes[row_index, column_index].set_xlabel(x_label)
            axes[row_index, column_index].set_ylabel(y_label)
            axes[row_index, column_index].set_title(f"{time_label}\n{reduction}")
    assert image is not None
    fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        label="Median first-passage time [Myr]; gray = censored",
    )

    inclination_reached = np.isfinite(scan.inclination_time_myr)
    eccentricity_reached = np.isfinite(scan.eccentricity_time_myr)
    eccentricity_first = eccentricity_reached & (
        ~inclination_reached | (scan.eccentricity_time_myr <= scan.inclination_time_myr)
    )
    inclination_first = inclination_reached & (
        ~eccentricity_reached | (scan.inclination_time_myr < scan.eccentricity_time_myr)
    )
    fig.suptitle(
        "Full $(a_i,e_i,i_i)$ production damping scan: "
        f"eccentricity first={np.count_nonzero(eccentricity_first)}, "
        f"inclination first={np.count_nonzero(inclination_first)}, "
        f"total={scan.inclination_time_myr.size}",
        fontsize=13,
    )
    path = output_dir / f"{HEATMAP_STEM}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _print_results(
    results: list[DampingCaseResult],
    scan: DampingScanResult,
) -> None:
    print(
        f"SMBH={SMBH_MASS_MSUN:.3e} Msun, object={OBJECT_MASS_MSUN:g} Msun, "
        f"e_cir={MAX_EMBEDDED_ECCENTRICITY:g}"
    )
    print(
        "inclination criterion: |i| <= (H/R)(a); "
        f"Rin={results[0].disk_inner_radius_rg:.3e} rg, "
        f"RQ1={results[0].disk_outer_radius_rg:.3e} rg"
    )
    print(
        f"{'case':25s} {'nodes':7s} {'e_i -> e_f':17s} " f"{'i_f / (H/R)':15s} status"
    )
    for result in results:
        nodes = (
            f"{int(result.initial_node_1_in_disk)}{int(result.initial_node_2_in_disk)}"
        )
        inclination_ratio = (
            result.final_inclination_rad / result.final_inclination_limit_rad
            if np.isfinite(result.final_inclination_limit_rad)
            else np.nan
        )
        print(
            f"{result.case_name:25s} {nodes:7s} "
            f"{result.initial_eccentricity:.3e} -> {result.final_eccentricity:.3e}  "
            f"{inclination_ratio:15.6e} {result.status}"
        )

    inclination_reached = np.isfinite(scan.inclination_time_myr)
    eccentricity_reached = np.isfinite(scan.eccentricity_time_myr)
    eccentricity_first = eccentricity_reached & (
        ~inclination_reached | (scan.eccentricity_time_myr <= scan.inclination_time_myr)
    )
    inclination_first = inclination_reached & (
        ~eccentricity_reached | (scan.inclination_time_myr < scan.eccentricity_time_myr)
    )
    print(
        "full-space scan: "
        f"total={scan.inclination_time_myr.size}, "
        f"eccentricity_first={np.count_nonzero(eccentricity_first)}, "
        f"inclination_first={np.count_nonzero(inclination_first)}, "
        f"neither={np.count_nonzero(~inclination_reached & ~eccentricity_reached)}"
    )


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    disk = _build_disk()
    results, trajectory, trajectory_outcome = _run_cases(disk)
    _assert_single_active_node_behavior(disk, "only_node_1_inside")
    _assert_single_active_node_behavior(disk, "only_node_2_inside")
    _assert_worker_serialization(disk, trajectory_outcome)
    _assert_results(results)

    trajectory_rows = _trajectory_rows(disk, trajectory)
    trajectory_times = _first_passage_times_myr(disk.calculator, trajectory)
    scan = _run_full_space_scan(disk)
    _assert_scan(scan)

    case_csv_path = _write_rows(
        args.output_dir / f"{OUTPUT_STEM}_results.csv",
        [asdict(result) for result in results],
    )
    trajectory_csv_path = _write_rows(
        args.output_dir / f"{OUTPUT_STEM}_trajectory.csv",
        trajectory_rows,
    )
    scan_csv_path = _write_rows(
        args.output_dir / f"{HEATMAP_STEM}.csv",
        scan.rows,
    )
    figure_path = _plot_trajectory_and_cases(
        args.output_dir,
        results,
        trajectory_rows,
        *trajectory_times,
    )
    heatmap_path = _plot_scan_heatmaps(args.output_dir, scan)
    _print_results(results, scan)
    for path in (
        case_csv_path,
        trajectory_csv_path,
        scan_csv_path,
        figure_path,
        heatmap_path,
    ):
        print(f"wrote {path}")
    print("all damping invariants passed")


if __name__ == "__main__":
    main()
