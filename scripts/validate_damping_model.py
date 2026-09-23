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
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.colors import LogNorm, TwoSlopeNorm
import numpy as np

from disk_quantity.pagn_disk_adapter import PagnDiskProfile, build_pagn_disk_profile
from physical_process import constants as const
from physical_process.agn import AGNProperties
from physical_process.damp import (
    MAX_EMBEDDED_ECCENTRICITY,
    DampingCalculator,
    DampingOutcome,
    DampingStatus,
    DragProperties,
    OrbitalState,
    TimescaleData,
)
from physical_process.nsc_orbital import disk_crossing_mask, orbital_node_radii_cm
from workflows.config import DISK_ACCRETION_RATE_G_S_PER_MSUN, PAGN_ALPHA
from workflows.models import DampingTask
from workflows.workers import init_damping_worker, process_damping_task

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

PHASE_AXIS_INITIAL_MIN_RG = 1.0e5
PHASE_AXIS_INITIAL_MAX_RG = 1.0e7
PHASE_AXIS_PLOT_MIN_RG = 3.0e3
PHASE_AXIS_PLOT_MAX_RG = 1.2e7
PHASE_ECCENTRICITY_MIN = 0.0
PHASE_ECCENTRICITY_MAX = 0.95
PHASE_ECCENTRICITY_INITIAL_MIN = 0.6
PHASE_ECCENTRICITY_INITIAL_MAX = 0.9
PHASE_INCLINATION_INITIAL_MIN_DEG = 60.0
PHASE_INCLINATION_INITIAL_MAX_DEG = 120.0
PHASE_RADIUS_CONTOURS_RG = (1.0e5, 1.0e6, 1.0e7)
PHASE_CONTOUR_SAMPLE_COUNT = 800
PHASE_GRID_COUNT = 4
PHASE_INCLINATION_PLOT_MIN_DEG = 0.3
PHASE_INCLINATION_PLOT_MAX_DEG = 150.0
PHASE_MAX_ITER = 50000
PHASE_COS_OMEGA = 0.5
RATIO_LOG_COLOR_FALLBACK_WIDTH = 1.0

OUTPUT_DIR = Path("figs")
OUTPUT_STEM = "damping_check"
PHASE_SPACE_DATA_STEM = "damping_phase_space_trajectories"
PHASE_A_E_STEM = "damping_phase_space_a_e"
PHASE_A_I_STEM = "damping_phase_space_a_i"
PHASE_E_I_STEM = "damping_phase_space_e_i"
RATIO_CUBE_STEM = "damping_timescale_ratio_cube"
RATIO_HEATMAP_STEM = "damping_timescale_ratio_heatmaps"


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
    timescale_ratio_ecc_over_inc: np.ndarray
    log10_timescale_ratio_ecc_over_inc: np.ndarray
    ratio_valid: np.ndarray
    rows: list[dict[str, float | bool | str]]


@dataclass(frozen=True)
class PhaseInitialCondition:
    identifier: str
    phase_plane: str
    semi_major_axis_rg: float
    eccentricity: float
    inclination_deg: float


@dataclass(frozen=True)
class PhaseTrajectory:
    condition: PhaseInitialCondition
    history: list[OrbitalState]
    outcome: DampingOutcome


@dataclass(frozen=True)
class RatioProjection:
    name: str
    x_values: np.ndarray
    y_values: np.ndarray
    x_field: str
    y_field: str
    x_label: str
    y_label: str
    reduction: str
    reduction_axis: int
    median_ratio: np.ndarray
    log10_median_ratio: np.ndarray
    valid_count: np.ndarray


@dataclass(frozen=True)
class DiagnosticDisk:
    profile: PagnDiskProfile
    agn: AGNProperties
    calculator: DampingCalculator


PHASE_AXIS_INITIAL_RG = np.geomspace(
    PHASE_AXIS_INITIAL_MIN_RG,
    PHASE_AXIS_INITIAL_MAX_RG,
    PHASE_GRID_COUNT,
)
PHASE_ECCENTRICITY_INITIAL = np.geomspace(
    PHASE_ECCENTRICITY_INITIAL_MIN,
    PHASE_ECCENTRICITY_INITIAL_MAX,
    PHASE_GRID_COUNT,
)
PHASE_INCLINATION_INITIAL_DEG = np.geomspace(
    PHASE_INCLINATION_INITIAL_MIN_DEG,
    PHASE_INCLINATION_INITIAL_MAX_DEG,
    PHASE_GRID_COUNT,
)
PHASE_FIXED_AXIS_RG = float(
    np.sqrt(PHASE_AXIS_INITIAL_MIN_RG * PHASE_AXIS_INITIAL_MAX_RG)
)
PHASE_FIXED_ECCENTRICITY = float(
    np.sqrt(PHASE_ECCENTRICITY_INITIAL_MIN * PHASE_ECCENTRICITY_INITIAL_MAX)
)
PHASE_FIXED_INCLINATION_DEG = float(
    np.sqrt(
        PHASE_INCLINATION_INITIAL_MIN_DEG
        * PHASE_INCLINATION_INITIAL_MAX_DEG
    )
)
PHASE_INITIAL_CONDITIONS = tuple(
    [
        PhaseInitialCondition(
            f"ae_a{axis_index:02d}_e{eccentricity_index:02d}",
            "a_e",
            float(axis_rg),
            float(eccentricity),
            PHASE_FIXED_INCLINATION_DEG,
        )
        for axis_index, axis_rg in enumerate(PHASE_AXIS_INITIAL_RG)
        for eccentricity_index, eccentricity in enumerate(
            PHASE_ECCENTRICITY_INITIAL
        )
    ]
    + [
        PhaseInitialCondition(
            f"ai_a{axis_index:02d}_i{inclination_index:02d}",
            "a_i",
            float(axis_rg),
            PHASE_FIXED_ECCENTRICITY,
            float(inclination_deg),
        )
        for axis_index, axis_rg in enumerate(PHASE_AXIS_INITIAL_RG)
        for inclination_index, inclination_deg in enumerate(
            PHASE_INCLINATION_INITIAL_DEG
        )
    ]
    + [
        PhaseInitialCondition(
            f"ei_e{eccentricity_index:02d}_i{inclination_index:02d}",
            "e_i",
            PHASE_FIXED_AXIS_RG,
            float(eccentricity),
            float(inclination_deg),
        )
        for eccentricity_index, eccentricity in enumerate(
            PHASE_ECCENTRICITY_INITIAL
        )
        for inclination_index, inclination_deg in enumerate(
            PHASE_INCLINATION_INITIAL_DEG
        )
    ]
)


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
        density_g_cm3=np.asarray(profile.density_g_cm3, dtype=float),
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


def _run_phase_trajectories(
    disk: DiagnosticDisk,
) -> tuple[PhaseTrajectory, ...]:
    trajectories: list[PhaseTrajectory] = []
    object_mass_g = OBJECT_MASS_MSUN * const.msun
    rg_cm = disk.agn.gravitational_radius_cm
    for condition in PHASE_INITIAL_CONDITIONS:
        history: list[OrbitalState] = []
        state = OrbitalState(
            semi_major_axis_cm=condition.semi_major_axis_rg * rg_cm,
            eccentricity=condition.eccentricity,
            inclination_rad=float(np.deg2rad(condition.inclination_deg)),
            cos_omega=PHASE_COS_OMEGA,
        )
        outcome = disk.calculator.damping_simulation(
            object_mass_g,
            state,
            FORMATION_TIME_MYR,
            "Star",
            max_iter=PHASE_MAX_ITER,
            history=history,
        )
        assert history
        trajectories.append(
            PhaseTrajectory(condition=condition, history=history, outcome=outcome)
        )
    return tuple(trajectories)


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
    timescale_ratio_ecc_over_inc = np.full(shape, np.nan)
    log10_timescale_ratio_ecc_over_inc = np.full(shape, np.nan)
    ratio_valid = np.zeros(shape, dtype=bool)
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
                valid_ratio = bool(
                    np.isfinite(inclination_time)
                    and inclination_time > 0.0
                    and np.isfinite(eccentricity_time)
                    and eccentricity_time > 0.0
                )
                timescale_ratio = (
                    eccentricity_time / inclination_time if valid_ratio else np.nan
                )
                log10_timescale_ratio = (
                    float(np.log10(timescale_ratio)) if valid_ratio else np.nan
                )
                ratio_valid[
                    axis_index, eccentricity_index, inclination_index
                ] = valid_ratio
                timescale_ratio_ecc_over_inc[
                    axis_index, eccentricity_index, inclination_index
                ] = timescale_ratio
                log10_timescale_ratio_ecc_over_inc[
                    axis_index, eccentricity_index, inclination_index
                ] = log10_timescale_ratio
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
                        "timescale_ratio_ecc_over_inc": timescale_ratio,
                        "log10_timescale_ratio_ecc_over_inc": (
                            log10_timescale_ratio
                        ),
                        "ratio_valid": valid_ratio,
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
        timescale_ratio_ecc_over_inc=timescale_ratio_ecc_over_inc,
        log10_timescale_ratio_ecc_over_inc=(
            log10_timescale_ratio_ecc_over_inc
        ),
        ratio_valid=ratio_valid,
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
    assert scan.timescale_ratio_ecc_over_inc.shape == expected_shape
    assert scan.log10_timescale_ratio_ecc_over_inc.shape == expected_shape
    assert scan.ratio_valid.shape == expected_shape

    expected_ratio_valid = (
        np.isfinite(scan.inclination_time_myr)
        & (scan.inclination_time_myr > 0.0)
        & np.isfinite(scan.eccentricity_time_myr)
        & (scan.eccentricity_time_myr > 0.0)
    )
    assert np.array_equal(scan.ratio_valid, expected_ratio_valid)
    assert np.all(np.isnan(scan.timescale_ratio_ecc_over_inc[~scan.ratio_valid]))
    assert np.all(
        np.isnan(scan.log10_timescale_ratio_ecc_over_inc[~scan.ratio_valid])
    )
    np.testing.assert_allclose(
        scan.timescale_ratio_ecc_over_inc[scan.ratio_valid],
        scan.eccentricity_time_myr[scan.ratio_valid]
        / scan.inclination_time_myr[scan.ratio_valid],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        scan.log10_timescale_ratio_ecc_over_inc[scan.ratio_valid],
        np.log10(scan.timescale_ratio_ecc_over_inc[scan.ratio_valid]),
        rtol=0.0,
        atol=0.0,
    )

    for row in scan.rows:
        elapsed_time_myr = float(row["elapsed_time_myr"])
        for key in ("inclination_time_myr", "eccentricity_time_myr"):
            threshold_time_myr = float(row[key])
            if np.isfinite(threshold_time_myr):
                assert 0.0 <= threshold_time_myr <= elapsed_time_myr
        valid_ratio = bool(row["ratio_valid"])
        inclination_time_myr = float(row["inclination_time_myr"])
        eccentricity_time_myr = float(row["eccentricity_time_myr"])
        timescale_ratio = float(row["timescale_ratio_ecc_over_inc"])
        log10_timescale_ratio = float(
            row["log10_timescale_ratio_ecc_over_inc"]
        )
        if valid_ratio:
            assert timescale_ratio == eccentricity_time_myr / inclination_time_myr
            assert log10_timescale_ratio == np.log10(timescale_ratio)
        else:
            assert np.isnan(timescale_ratio)
            assert np.isnan(log10_timescale_ratio)


def _trajectory_rows(
    disk: DiagnosticDisk,
    trajectory: list[OrbitalState],
    trajectory_id: str,
    phase_plane: str,
    final_status: str,
) -> list[dict[str, float | bool | int | str]]:
    rg_cm = disk.agn.gravitational_radius_cm
    disk_outer_radius_rg = disk.profile.outer_radius_cm / rg_cm
    initial_state = trajectory[0]
    rows: list[dict[str, float | bool | int | str]] = []
    for index, state in enumerate(trajectory):
        nodes_cm, flags = _node_data(disk.calculator, state)
        inclination_limit_rad = disk.calculator.inclination_limit_rad(state)
        criteria = disk.calculator.embedding_criteria(state)
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "phase_plane": phase_plane,
                "step": index,
                "time_myr": state.time_s / const.Myr,
                "time_yr": state.time_s / const.yr,
                "initial_semi_major_axis_rg": (
                    initial_state.semi_major_axis_cm / rg_cm
                ),
                "initial_semi_major_axis_over_disk_outer": (
                    initial_state.semi_major_axis_cm / disk.profile.outer_radius_cm
                ),
                "initial_eccentricity": initial_state.eccentricity,
                "initial_inclination_deg": float(
                    np.degrees(initial_state.inclination_rad)
                ),
                "semi_major_axis_rg": state.semi_major_axis_cm / rg_cm,
                "semi_major_axis_over_disk_outer": (
                    state.semi_major_axis_cm / rg_cm / disk_outer_radius_rg
                ),
                "periapsis_rg": criteria.periapsis_cm / rg_cm,
                "apoapsis_rg": criteria.apoapsis_cm / rg_cm,
                "periapsis_over_disk_outer": (
                    criteria.periapsis_cm / disk.profile.outer_radius_cm
                ),
                "apoapsis_over_disk_outer": (
                    criteria.apoapsis_cm / disk.profile.outer_radius_cm
                ),
                "node_1_radius_rg": nodes_cm[0] / rg_cm,
                "node_2_radius_rg": nodes_cm[1] / rg_cm,
                "node_1_in_disk": flags[0],
                "node_2_in_disk": flags[1],
                "eccentricity": state.eccentricity,
                "cos_omega": state.cos_omega,
                "eccentricity_over_threshold": (
                    state.eccentricity / MAX_EMBEDDED_ECCENTRICITY
                ),
                "inclination_rad": state.inclination_rad,
                "inclination_deg": float(np.degrees(state.inclination_rad)),
                "local_aspect_ratio": inclination_limit_rad,
                "inclination_over_aspect_ratio": (
                    abs(state.inclination_rad) / inclination_limit_rad
                ),
                "is_initial_state": index == 0,
                "is_final_state": index == len(trajectory) - 1,
                "is_eccentricity_first_passage": False,
                "is_inclination_first_passage": False,
                "final_status": final_status,
            }
        )
    eccentricity_index = next(
        (
            index
            for index, row in enumerate(rows)
            if float(row["eccentricity"]) <= MAX_EMBEDDED_ECCENTRICITY
        ),
        None,
    )
    inclination_index = next(
        (
            index
            for index, row in enumerate(rows)
            if float(row["inclination_over_aspect_ratio"]) <= 1.0
        ),
        None,
    )
    if eccentricity_index is not None:
        rows[eccentricity_index]["is_eccentricity_first_passage"] = True
    if inclination_index is not None:
        rows[inclination_index]["is_inclination_first_passage"] = True
    return rows


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _phase_space_contour(
    radius_rg: float,
    apsis: str,
) -> tuple[np.ndarray, np.ndarray]:
    axis_rg = np.geomspace(
        PHASE_AXIS_PLOT_MIN_RG,
        PHASE_AXIS_PLOT_MAX_RG,
        PHASE_CONTOUR_SAMPLE_COUNT,
    )
    if apsis == "periapsis":
        eccentricity = 1.0 - radius_rg / axis_rg
        reconstructed_radius_rg = axis_rg * (1.0 - eccentricity)
    elif apsis == "apoapsis":
        eccentricity = radius_rg / axis_rg - 1.0
        reconstructed_radius_rg = axis_rg * (1.0 + eccentricity)
    else:
        raise ValueError(f"unknown apsis: {apsis}")
    valid = (
        (eccentricity >= PHASE_ECCENTRICITY_MIN)
        & (eccentricity < 1.0)
        & (eccentricity <= PHASE_ECCENTRICITY_MAX)
    )
    axis_rg = axis_rg[valid]
    eccentricity = eccentricity[valid]
    reconstructed_radius_rg = reconstructed_radius_rg[valid]
    assert axis_rg.size > 0
    assert np.all((eccentricity >= 0.0) & (eccentricity < 1.0))
    np.testing.assert_allclose(
        reconstructed_radius_rg,
        radius_rg,
        rtol=1.0e-12,
        atol=1.0e-8,
    )
    return axis_rg, eccentricity


def _assert_phase_space(
    disk: DiagnosticDisk,
    trajectory: list[OrbitalState],
    trajectory_rows: list[dict[str, float | bool | int | str]],
    trajectory_outcome: DampingOutcome,
) -> None:
    assert len(trajectory_rows) == len(trajectory)
    assert bool(trajectory_rows[0]["is_initial_state"])
    assert bool(trajectory_rows[-1]["is_final_state"])
    assert sum(bool(row["is_initial_state"]) for row in trajectory_rows) == 1
    assert sum(bool(row["is_final_state"]) for row in trajectory_rows) == 1

    times_myr = np.asarray(
        [row["time_myr"] for row in trajectory_rows], dtype=float
    )
    times_yr = np.asarray([row["time_yr"] for row in trajectory_rows], dtype=float)
    assert np.all(np.diff(times_myr) >= 0.0)
    assert np.all(np.diff(times_yr) >= 0.0)
    np.testing.assert_allclose(times_yr, times_myr * 1.0e6, rtol=1.0e-14)
    rg_cm = disk.agn.gravitational_radius_cm
    for state, row in zip(trajectory, trajectory_rows):
        semi_major_axis_rg = state.semi_major_axis_cm / rg_cm
        periapsis_rg = semi_major_axis_rg * (1.0 - state.eccentricity)
        apoapsis_rg = semi_major_axis_rg * (1.0 + state.eccentricity)
        criteria = disk.calculator.embedding_criteria(state)
        assert float(row["time_myr"]) == state.time_s / const.Myr
        assert float(row["time_yr"]) == state.time_s / const.yr
        assert float(row["semi_major_axis_rg"]) == semi_major_axis_rg
        assert float(row["eccentricity"]) == state.eccentricity
        assert float(row["inclination_rad"]) == state.inclination_rad
        np.testing.assert_allclose(
            (float(row["periapsis_rg"]), criteria.periapsis_cm / rg_cm),
            (periapsis_rg, periapsis_rg),
            rtol=1.0e-14,
            atol=0.0,
        )
        np.testing.assert_allclose(
            (float(row["apoapsis_rg"]), criteria.apoapsis_cm / rg_cm),
            (apoapsis_rg, apoapsis_rg),
            rtol=1.0e-14,
            atol=0.0,
        )

    final_state = trajectory_outcome.final_state
    assert float(trajectory_rows[-1]["semi_major_axis_rg"]) == (
        final_state.semi_major_axis_cm / rg_cm
    )
    assert float(trajectory_rows[-1]["eccentricity"]) == final_state.eccentricity

    eccentricity_index = next(
        (
            index
            for index, row in enumerate(trajectory_rows)
            if float(row["eccentricity"]) <= MAX_EMBEDDED_ECCENTRICITY
        ),
        None,
    )
    inclination_index = next(
        (
            index
            for index, row in enumerate(trajectory_rows)
            if float(row["inclination_over_aspect_ratio"]) <= 1.0
        ),
        None,
    )
    assert [
        index
        for index, row in enumerate(trajectory_rows)
        if bool(row["is_eccentricity_first_passage"])
    ] == ([] if eccentricity_index is None else [eccentricity_index])
    assert [
        index
        for index, row in enumerate(trajectory_rows)
        if bool(row["is_inclination_first_passage"])
    ] == ([] if inclination_index is None else [inclination_index])

    axis_rg = np.asarray([row["semi_major_axis_rg"] for row in trajectory_rows])
    eccentricity = np.asarray(
        [row["eccentricity"] for row in trajectory_rows], dtype=float
    )
    assert np.all(
        (axis_rg >= PHASE_AXIS_PLOT_MIN_RG)
        & (axis_rg <= PHASE_AXIS_PLOT_MAX_RG)
    )
    assert np.all(
        (eccentricity >= PHASE_ECCENTRICITY_MIN)
        & (eccentricity <= PHASE_ECCENTRICITY_MAX)
    )
    for level in PHASE_RADIUS_CONTOURS_RG:
        _phase_space_contour(level, "periapsis")
        _phase_space_contour(level, "apoapsis")


def _assert_phase_trajectories(
    disk: DiagnosticDisk,
    trajectories: tuple[PhaseTrajectory, ...],
    rows: list[dict[str, float | bool | int | str]],
) -> None:
    expected_per_plane = PHASE_GRID_COUNT**2
    assert len(trajectories) == 3 * expected_per_plane
    requested_states = {
        (
            trajectory.condition.semi_major_axis_rg,
            trajectory.condition.eccentricity,
            trajectory.condition.inclination_deg,
        )
        for trajectory in trajectories
    }
    assert len(requested_states) == len(trajectories)
    assert {trajectory.condition.identifier for trajectory in trajectories} == {
        condition.identifier for condition in PHASE_INITIAL_CONDITIONS
    }

    for values in (
        PHASE_AXIS_INITIAL_RG,
        PHASE_ECCENTRICITY_INITIAL,
        PHASE_INCLINATION_INITIAL_DEG,
    ):
        logarithmic_steps = np.diff(np.log(values))
        np.testing.assert_allclose(
            logarithmic_steps,
            logarithmic_steps[0],
            rtol=1.0e-14,
            atol=1.0e-14,
        )

    plane_coordinates = {
        "a_e": lambda condition: (
            condition.semi_major_axis_rg,
            condition.eccentricity,
        ),
        "a_i": lambda condition: (
            condition.semi_major_axis_rg,
            condition.inclination_deg,
        ),
        "e_i": lambda condition: (
            condition.eccentricity,
            condition.inclination_deg,
        ),
    }
    for phase_plane, coordinates in plane_coordinates.items():
        plane_conditions = [
            trajectory.condition
            for trajectory in trajectories
            if trajectory.condition.phase_plane == phase_plane
        ]
        assert len(plane_conditions) == expected_per_plane
        assert len({coordinates(condition) for condition in plane_conditions}) == (
            expected_per_plane
        )

    assert all(
        np.isclose(condition.inclination_deg, PHASE_FIXED_INCLINATION_DEG)
        for condition in PHASE_INITIAL_CONDITIONS
        if condition.phase_plane == "a_e"
    )
    assert all(
        np.isclose(condition.eccentricity, PHASE_FIXED_ECCENTRICITY)
        for condition in PHASE_INITIAL_CONDITIONS
        if condition.phase_plane == "a_i"
    )
    assert all(
        np.isclose(condition.semi_major_axis_rg, PHASE_FIXED_AXIS_RG)
        for condition in PHASE_INITIAL_CONDITIONS
        if condition.phase_plane == "e_i"
    )

    for trajectory in trajectories:
        trajectory_rows = [
            row
            for row in rows
            if row["trajectory_id"] == trajectory.condition.identifier
        ]
        _assert_phase_space(
            disk,
            trajectory.history,
            trajectory_rows,
            trajectory.outcome,
        )
        initial_row = trajectory_rows[0]
        assert np.isclose(
            float(initial_row["initial_semi_major_axis_rg"]),
            trajectory.condition.semi_major_axis_rg,
        )
        assert np.isclose(
            float(initial_row["initial_eccentricity"]),
            trajectory.condition.eccentricity,
        )
        assert np.isclose(
            float(initial_row["initial_inclination_deg"]),
            trajectory.condition.inclination_deg,
        )
        assert all(
            float(row["cos_omega"]) == PHASE_COS_OMEGA for row in trajectory_rows
        )
        assert all(
            row["phase_plane"] == trajectory.condition.phase_plane
            for row in trajectory_rows
        )

    positive_time_yr = np.asarray(
        [float(row["time_yr"]) for row in rows if float(row["time_yr"]) > 0.0]
    )
    assert positive_time_yr.size > 0
    assert np.all(np.isfinite(positive_time_yr) & (positive_time_yr > 0.0))
    axis_rg = np.asarray([float(row["semi_major_axis_rg"]) for row in rows])
    assert np.all(
        (axis_rg >= PHASE_AXIS_PLOT_MIN_RG)
        & (axis_rg <= PHASE_AXIS_PLOT_MAX_RG)
    )


def _phase_time_norm(
    trajectory_rows: list[dict[str, float | bool | int | str]],
) -> LogNorm:
    positive_time_yr = np.asarray(
        [float(row["time_yr"]) for row in trajectory_rows if float(row["time_yr"]) > 0]
    )
    assert positive_time_yr.size > 0
    minimum_time_yr = float(np.min(positive_time_yr))
    maximum_time_yr = float(np.max(positive_time_yr))
    assert 0.0 < minimum_time_yr < maximum_time_yr
    return LogNorm(vmin=minimum_time_yr, vmax=maximum_time_yr)


def _plot_phase_paths(
    axis: Axes,
    trajectory_rows: list[dict[str, float | bool | int | str]],
    conditions: Sequence[PhaseInitialCondition],
    x_field: str,
    y_field: str,
    time_norm: LogNorm,
) -> PathCollection:
    points: PathCollection | None = None
    for condition_index, condition in enumerate(conditions):
        rows = [
            row
            for row in trajectory_rows
            if row["trajectory_id"] == condition.identifier
        ]
        x_values = np.asarray([row[x_field] for row in rows], dtype=float)
        y_values = np.asarray([row[y_field] for row in rows], dtype=float)
        time_yr = np.asarray([row["time_yr"] for row in rows], dtype=float)
        positive_time = time_yr > 0.0
        axis.plot(x_values, y_values, color="#444444", linewidth=0.7, alpha=0.45)
        points = axis.scatter(
            x_values[positive_time],
            y_values[positive_time],
            c=time_yr[positive_time],
            cmap="viridis",
            norm=time_norm,
            s=9,
            edgecolors="none",
            alpha=0.72,
            zorder=4,
        )
        axis.scatter(
            x_values[0],
            y_values[0],
            marker="o",
            s=55,
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            label="initial states" if condition_index == 0 else None,
            zorder=6,
        )
        axis.scatter(
            x_values[-1],
            y_values[-1],
            marker="X",
            s=48,
            color="#CC3311",
            edgecolors="black",
            linewidths=0.5,
            label="final states" if condition_index == 0 else None,
            zorder=7,
        )
    assert points is not None
    return points


def _phase_parameter_text(axis: Axes) -> None:
    axis.text(
        0.98,
        -0.14,
        (
            f"$M={SMBH_MASS_MSUN:.1e}\\,M_\\odot$, "
            f"$m={OBJECT_MASS_MSUN:g}\\,M_\\odot$, "
            f"$\\cos\\omega={PHASE_COS_OMEGA:g}$"
        ),
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )


def _plot_phase_spaces(
    output_dir: Path,
    disk: DiagnosticDisk,
    trajectory_rows: list[dict[str, float | bool | int | str]],
) -> tuple[Path, Path, Path]:
    time_norm = _phase_time_norm(trajectory_rows)
    eccentricity_threshold = MAX_EMBEDDED_ECCENTRICITY
    conditions_by_plane = {
        phase_plane: tuple(
            condition
            for condition in PHASE_INITIAL_CONDITIONS
            if condition.phase_plane == phase_plane
        )
        for phase_plane in ("a_e", "a_i", "e_i")
    }

    ae_figure, ae_axis = plt.subplots(figsize=(9.2, 6.2), constrained_layout=True)
    for contour_index, level_rg in enumerate(PHASE_RADIUS_CONTOURS_RG):
        periapsis_axis, periapsis_eccentricity = _phase_space_contour(
            level_rg, "periapsis"
        )
        apoapsis_axis, apoapsis_eccentricity = _phase_space_contour(
            level_rg, "apoapsis"
        )
        levels = ", ".join(
            f"$10^{int(np.log10(value))}$" for value in PHASE_RADIUS_CONTOURS_RG
        )
        ae_axis.plot(
            periapsis_axis,
            periapsis_eccentricity,
            color="#8A8A8A",
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
            label=(r"$r_{in}/r_g=$ " + levels if contour_index == 0 else None),
        )
        ae_axis.plot(
            apoapsis_axis,
            apoapsis_eccentricity,
            color="#666666",
            linestyle=":",
            linewidth=0.9,
            alpha=0.65,
            label=(r"$r_{out}/r_g=$ " + levels if contour_index == 0 else None),
        )
    ae_axis.axhline(
        eccentricity_threshold,
        color="#1f77b4",
        linestyle=":",
        linewidth=1.0,
        label=r"$e_{cir}=0.01$",
    )
    ae_points = _plot_phase_paths(
        ae_axis,
        trajectory_rows,
        conditions_by_plane["a_e"],
        "semi_major_axis_rg",
        "eccentricity",
        time_norm,
    )
    ae_axis.set_xscale("log")
    ae_axis.set_yscale("symlog", linthresh=eccentricity_threshold, base=10)
    ae_axis.set_xlim(PHASE_AXIS_PLOT_MIN_RG, PHASE_AXIS_PLOT_MAX_RG)
    ae_axis.set_ylim(PHASE_ECCENTRICITY_MIN, PHASE_ECCENTRICITY_MAX)
    ae_axis.set_xlabel(r"$a/r_g$")
    ae_axis.set_ylabel("$e$")
    ae_axis.set_title(
        "Damping trajectories: $(a,e)$; "
        rf"fixed $i_i={PHASE_FIXED_INCLINATION_DEG:.1f}^\circ$"
    )
    ae_axis.legend(loc="upper left", fontsize=7, ncols=2)
    _phase_parameter_text(ae_axis)
    ae_figure.colorbar(
        ae_points,
        ax=ae_axis,
        label="Elapsed damping time [yr] (log scale)",
    )
    ae_path = output_dir / f"{PHASE_A_E_STEM}.png"
    ae_figure.savefig(ae_path, dpi=180)
    plt.close(ae_figure)

    ai_figure, ai_axis = plt.subplots(figsize=(9.2, 6.2), constrained_layout=True)
    criterion_axis_rg = np.geomspace(
        PHASE_AXIS_PLOT_MIN_RG,
        PHASE_AXIS_PLOT_MAX_RG,
        PHASE_CONTOUR_SAMPLE_COUNT,
    )
    criterion_inclination_deg = np.asarray(
        [
            np.degrees(
                disk.calculator.inclination_limit_rad(
                    OrbitalState(
                        semi_major_axis_cm=(
                            axis_rg * disk.agn.gravitational_radius_cm
                        ),
                        eccentricity=0.0,
                        inclination_rad=0.0,
                        cos_omega=PHASE_COS_OMEGA,
                    )
                )
            )
            for axis_rg in criterion_axis_rg
        ]
    )
    assert np.all(np.isfinite(criterion_inclination_deg))
    ai_axis.plot(
        criterion_axis_rg,
        criterion_inclination_deg,
        color="#2A7F62",
        linestyle="--",
        linewidth=1.5,
        label=r"$i=(H/R)(a)$",
    )
    ai_points = _plot_phase_paths(
        ai_axis,
        trajectory_rows,
        conditions_by_plane["a_i"],
        "semi_major_axis_rg",
        "inclination_deg",
        time_norm,
    )
    ai_axis.set_xscale("log")
    ai_axis.set_yscale("log")
    ai_axis.set_xlim(PHASE_AXIS_PLOT_MIN_RG, PHASE_AXIS_PLOT_MAX_RG)
    ai_axis.set_ylim(
        PHASE_INCLINATION_PLOT_MIN_DEG,
        PHASE_INCLINATION_PLOT_MAX_DEG,
    )
    ai_axis.set_xlabel(r"$a/r_g$")
    ai_axis.set_ylabel("$i$ [deg]")
    ai_axis.set_title(
        "Damping trajectories: $(a,i)$; "
        rf"fixed $e_i={PHASE_FIXED_ECCENTRICITY:.3f}$"
    )
    ai_axis.legend(loc="upper left", fontsize=8)
    _phase_parameter_text(ai_axis)
    ai_figure.colorbar(
        ai_points,
        ax=ai_axis,
        label="Elapsed damping time [yr] (log scale)",
    )
    ai_path = output_dir / f"{PHASE_A_I_STEM}.png"
    ai_figure.savefig(ai_path, dpi=180)
    plt.close(ai_figure)

    ei_figure, ei_axis = plt.subplots(figsize=(9.2, 6.2), constrained_layout=True)
    ei_axis.axvline(
        eccentricity_threshold,
        color="#1f77b4",
        linestyle=":",
        linewidth=1.2,
        label=r"$e_{cir}=0.01$",
    )
    ei_points = _plot_phase_paths(
        ei_axis,
        trajectory_rows,
        conditions_by_plane["e_i"],
        "eccentricity",
        "inclination_deg",
        time_norm,
    )
    ei_axis.set_xscale("symlog", linthresh=eccentricity_threshold, base=10)
    ei_axis.set_yscale("log")
    ei_axis.set_xlim(PHASE_ECCENTRICITY_MIN, PHASE_ECCENTRICITY_MAX)
    ei_axis.set_ylim(
        PHASE_INCLINATION_PLOT_MIN_DEG,
        PHASE_INCLINATION_PLOT_MAX_DEG,
    )
    ei_axis.set_xlabel("$e$")
    ei_axis.set_ylabel("$i$ [deg]")
    ei_axis.set_title(
        "Damping trajectories: $(e,i)$; "
        rf"fixed $a_i={PHASE_FIXED_AXIS_RG:.1e}\,r_g$"
    )
    ei_axis.legend(loc="lower right", fontsize=8)
    _phase_parameter_text(ei_axis)
    ei_figure.colorbar(
        ei_points,
        ax=ei_axis,
        label="Elapsed damping time [yr] (log scale)",
    )
    ei_path = output_dir / f"{PHASE_E_I_STEM}.png"
    ei_figure.savefig(ei_path, dpi=180)
    plt.close(ei_figure)
    return ae_path, ai_path, ei_path


def _plot_threshold_history(
    output_dir: Path,
    trajectory_rows: list[dict[str, float | bool | int | str]],
    inclination_time_myr: float,
    eccentricity_time_myr: float,
) -> Path:
    time_yr = np.asarray([row["time_yr"] for row in trajectory_rows], dtype=float)
    positive_time = time_yr > 0.0
    eccentricity_ratio = np.asarray(
        [row["eccentricity_over_threshold"] for row in trajectory_rows],
        dtype=float,
    )
    inclination_ratio = np.asarray(
        [row["inclination_over_aspect_ratio"] for row in trajectory_rows],
        dtype=float,
    )
    fig, axis = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    axis.plot(
        time_yr[positive_time],
        eccentricity_ratio[positive_time],
        color="#1f77b4",
        label="$e/e_{cir}$",
    )
    axis.plot(
        time_yr[positive_time],
        inclination_ratio[positive_time],
        color="#BE9A62",
        label="$|i|/(H/R)(a)$",
    )
    axis.axhline(1.0, color="black", linestyle="--", label="criterion")
    if np.isfinite(eccentricity_time_myr):
        eccentricity_time_yr = eccentricity_time_myr * 1.0e6
        axis.axvline(
            eccentricity_time_yr,
            color="#1f77b4",
            linestyle=":",
            label=f"$t_{{ecc}}={eccentricity_time_yr:.2e}$ yr",
        )
    if np.isfinite(inclination_time_myr):
        inclination_time_yr = inclination_time_myr * 1.0e6
        axis.axvline(
            inclination_time_yr,
            color="#BE9A62",
            linestyle=":",
            label=f"$t_{{inc}}={inclination_time_yr:.2e}$ yr",
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_ylim(bottom=0.5)
    axis.set_xlabel("Elapsed damping time [yr] (log scale)")
    axis.set_ylabel("Value / embedding threshold")
    axis.set_title("First passage occurs when each curve reaches 1")
    axis.legend(fontsize=8)
    path = output_dir / f"{OUTPUT_STEM}_thresholds.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_final_predicates(
    output_dir: Path,
    results: list[DampingCaseResult],
) -> Path:
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
    fig, axis = plt.subplots(figsize=(12.0, 6.0), constrained_layout=True)
    axis.imshow(criteria, aspect="auto", vmin=0.0, vmax=1.0, cmap="RdYlGn")
    axis.set_xticks(
        range(4),
        (
            r"$|i|\leq(H/R)(a)$",
            r"$e\leq0.01$",
            "peri/apo inside disk",
            "migration eligible",
        ),
    )
    axis.set_yticks(
        range(len(results)),
        [f"{result.case_name}  [{result.status}]" for result in results],
    )
    axis.set_title("Final-state predicates: green=true, red=false")
    for row_index in range(criteria.shape[0]):
        for column_index in range(criteria.shape[1]):
            axis.text(
                column_index,
                row_index,
                "true" if criteria[row_index, column_index] else "false",
                ha="center",
                va="center",
                fontsize=8,
            )
    path = output_dir / f"{OUTPUT_STEM}_predicates.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _median_projection(values: np.ndarray, axis: int) -> np.ndarray:
    return np.asarray(
        np.ma.median(np.ma.masked_invalid(values), axis=axis).filled(np.nan),
        dtype=float,
    )


def _ratio_projections(scan: DampingScanResult) -> tuple[RatioProjection, ...]:
    ratio = scan.timescale_ratio_ecc_over_inc

    def projection(
        name: str,
        x_values: np.ndarray,
        y_values: np.ndarray,
        x_field: str,
        y_field: str,
        x_label: str,
        y_label: str,
        reduction: str,
        reduction_axis: int,
    ) -> RatioProjection:
        median_ratio = _median_projection(ratio, reduction_axis).T
        log10_median_ratio = np.full(median_ratio.shape, np.nan)
        valid_median = np.isfinite(median_ratio) & (median_ratio > 0.0)
        log10_median_ratio[valid_median] = np.log10(median_ratio[valid_median])
        valid_count = np.count_nonzero(scan.ratio_valid, axis=reduction_axis).T
        return RatioProjection(
            name=name,
            x_values=x_values,
            y_values=y_values,
            x_field=x_field,
            y_field=y_field,
            x_label=x_label,
            y_label=y_label,
            reduction=reduction,
            reduction_axis=reduction_axis,
            median_ratio=median_ratio,
            log10_median_ratio=log10_median_ratio,
            valid_count=valid_count,
        )

    log_axis_rg = np.log10(scan.axis_rg)
    return (
        projection(
            "a_e",
            log_axis_rg,
            scan.eccentricity,
            "log10_initial_semi_major_axis_rg",
            "initial_eccentricity",
            r"$\log_{10}(a_i/r_g)$",
            "$e_i$",
            "median over $i_i$",
            2,
        ),
        projection(
            "a_i",
            log_axis_rg,
            scan.inclination_deg,
            "log10_initial_semi_major_axis_rg",
            "initial_inclination_deg",
            r"$\log_{10}(a_i/r_g)$",
            "$i_i$ [deg]",
            "median over $e_i$",
            1,
        ),
        projection(
            "e_i",
            scan.eccentricity,
            scan.inclination_deg,
            "initial_eccentricity",
            "initial_inclination_deg",
            "$e_i$",
            "$i_i$ [deg]",
            "median over $a_i$",
            0,
        ),
    )


def _projection_source_values(
    scan: DampingScanResult,
    reduction_axis: int,
    x_index: int,
    y_index: int,
) -> np.ndarray:
    if reduction_axis == 2:
        return scan.timescale_ratio_ecc_over_inc[x_index, y_index, :]
    if reduction_axis == 1:
        return scan.timescale_ratio_ecc_over_inc[x_index, :, y_index]
    if reduction_axis == 0:
        return scan.timescale_ratio_ecc_over_inc[:, x_index, y_index]
    raise ValueError(f"invalid reduction axis: {reduction_axis}")


def _assert_ratio_projections(
    scan: DampingScanResult,
    projections: tuple[RatioProjection, ...],
) -> None:
    assert len(projections) == 3
    total_valid = int(np.count_nonzero(scan.ratio_valid))
    for projection in projections:
        expected_shape = (projection.y_values.size, projection.x_values.size)
        assert projection.median_ratio.shape == expected_shape
        assert projection.log10_median_ratio.shape == expected_shape
        assert projection.valid_count.shape == expected_shape
        assert int(np.sum(projection.valid_count)) == total_valid
        for y_index in range(projection.y_values.size):
            for x_index in range(projection.x_values.size):
                source = _projection_source_values(
                    scan,
                    projection.reduction_axis,
                    x_index,
                    y_index,
                )
                finite_source = source[np.isfinite(source)]
                assert projection.valid_count[y_index, x_index] == finite_source.size
                if finite_source.size == 0:
                    assert np.isnan(projection.median_ratio[y_index, x_index])
                    assert np.isnan(projection.log10_median_ratio[y_index, x_index])
                    continue
                expected_median = float(np.median(finite_source))
                assert projection.median_ratio[y_index, x_index] == expected_median
                assert projection.log10_median_ratio[y_index, x_index] == np.log10(
                    expected_median
                )


def _ratio_projection_rows(
    projections: tuple[RatioProjection, ...],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for projection in projections:
        for y_index, y_value in enumerate(projection.y_values):
            for x_index, x_value in enumerate(projection.x_values):
                rows.append(
                    {
                        "projection": projection.name,
                        "x_field": projection.x_field,
                        "x_value": float(x_value),
                        "y_field": projection.y_field,
                        "y_value": float(y_value),
                        "reduction": projection.reduction,
                        "median_timescale_ratio_ecc_over_inc": float(
                            projection.median_ratio[y_index, x_index]
                        ),
                        "log10_median_timescale_ratio_ecc_over_inc": float(
                            projection.log10_median_ratio[y_index, x_index]
                        ),
                        "valid_contributing_count": int(
                            projection.valid_count[y_index, x_index]
                        ),
                    }
                )
    return rows


def _plot_scan_ratio_heatmaps(
    output_dir: Path,
    scan: DampingScanResult,
    projections: tuple[RatioProjection, ...],
) -> Path:
    finite_log_ratios = np.concatenate(
        [
            projection.log10_median_ratio[
                np.isfinite(projection.log10_median_ratio)
            ]
            for projection in projections
        ]
    )
    if finite_log_ratios.size == 0:
        raise RuntimeError("the damping scan produced no valid timescale ratios")
    color_width = float(np.max(np.abs(finite_log_ratios)))
    if color_width == 0.0:
        color_width = RATIO_LOG_COLOR_FALLBACK_WIDTH
    norm = TwoSlopeNorm(vmin=-color_width, vcenter=0.0, vmax=color_width)
    color_map = plt.get_cmap("coolwarm").copy()
    color_map.set_bad("#666666")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), constrained_layout=True)
    image = None
    total_scan_count = scan.ratio_valid.size
    total_valid = int(np.count_nonzero(scan.ratio_valid))
    for axis, projection in zip(axes, projections):
        image = axis.pcolormesh(
            projection.x_values,
            projection.y_values,
            projection.log10_median_ratio,
            shading="nearest",
            cmap=color_map,
            norm=norm,
        )
        axis.set_xlabel(projection.x_label)
        axis.set_ylabel(projection.y_label)
        axis.set_title(
            f"{projection.reduction}\nvalid ratios={total_valid}/{total_scan_count}"
        )
    assert image is not None
    fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        label=r"$\log_{10}(t_{ecc}/t_{inc})$",
    )
    fig.supxlabel(
        "Color meaning: blue (<0) = eccentricity first; white (0) = equal; "
        "red (>0) = inclination first; dark gray = unavailable",
        fontsize=9,
    )

    inclination_reached = np.isfinite(scan.inclination_time_myr)
    eccentricity_reached = np.isfinite(scan.eccentricity_time_myr)
    both_reached = inclination_reached & eccentricity_reached
    eccentricity_only = eccentricity_reached & ~inclination_reached
    inclination_only = inclination_reached & ~eccentricity_reached
    neither_reached = ~inclination_reached & ~eccentricity_reached
    fig.suptitle(
        "Relative first-passage times over the full $(a_i,e_i,i_i)$ scan: "
        f"total={scan.ratio_valid.size}, both={np.count_nonzero(both_reached)}, "
        f"ecc-only={np.count_nonzero(eccentricity_only)}, "
        f"inc-only={np.count_nonzero(inclination_only)}, "
        f"neither={np.count_nonzero(neither_reached)}",
        fontsize=12,
    )
    path = output_dir / f"{RATIO_HEATMAP_STEM}.png"
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
    valid_ratios = scan.timescale_ratio_ecc_over_inc[scan.ratio_valid]
    print(
        "relative timescale ratio t_ecc/t_inc: "
        f"valid={valid_ratios.size}, invalid={scan.ratio_valid.size - valid_ratios.size}, "
        f"min={np.min(valid_ratios):.6e}, "
        f"median={np.median(valid_ratios):.6e}, "
        f"max={np.max(valid_ratios):.6e}"
    )


def _print_phase_trajectories(trajectories: tuple[PhaseTrajectory, ...]) -> None:
    print("phase-space production trajectories:")
    print(
        f"{'plane':5s} {'id':16s} {'a_i[rg]':>10s} "
        f"{'e_i':>6s} {'i_i[deg]':>9s} status"
    )
    for trajectory in trajectories:
        condition = trajectory.condition
        print(
            f"{condition.phase_plane:5s} {condition.identifier:16s} "
            f"{condition.semi_major_axis_rg:10.3e} "
            f"{condition.eccentricity:6.3f} "
            f"{condition.inclination_deg:9.3f} "
            f"{trajectory.outcome.status.value}"
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

    trajectory_rows = _trajectory_rows(
        disk,
        trajectory,
        "central",
        "threshold_reference",
        trajectory_outcome.status.value,
    )
    trajectory_times = _first_passage_times_myr(disk.calculator, trajectory)
    phase_trajectories = _run_phase_trajectories(disk)
    phase_trajectory_rows: list[dict[str, float | bool | int | str]] = []
    for phase_trajectory in phase_trajectories:
        phase_trajectory_rows.extend(
            _trajectory_rows(
                disk,
                phase_trajectory.history,
                phase_trajectory.condition.identifier,
                phase_trajectory.condition.phase_plane,
                phase_trajectory.outcome.status.value,
            )
        )
    _assert_phase_trajectories(disk, phase_trajectories, phase_trajectory_rows)
    scan = _run_full_space_scan(disk)
    _assert_scan(scan)
    ratio_projections = _ratio_projections(scan)
    _assert_ratio_projections(scan, ratio_projections)

    case_csv_path = _write_rows(
        args.output_dir / f"{OUTPUT_STEM}_results.csv",
        [asdict(result) for result in results],
    )
    trajectory_csv_path = _write_rows(
        args.output_dir / f"{PHASE_SPACE_DATA_STEM}.csv",
        phase_trajectory_rows,
    )
    ratio_cube_csv_path = _write_rows(
        args.output_dir / f"{RATIO_CUBE_STEM}.csv",
        scan.rows,
    )
    ratio_heatmap_csv_path = _write_rows(
        args.output_dir / f"{RATIO_HEATMAP_STEM}.csv",
        _ratio_projection_rows(ratio_projections),
    )
    phase_space_paths = _plot_phase_spaces(
        args.output_dir,
        disk,
        phase_trajectory_rows,
    )
    threshold_path = _plot_threshold_history(
        args.output_dir,
        trajectory_rows,
        *trajectory_times,
    )
    predicate_path = _plot_final_predicates(args.output_dir, results)
    ratio_heatmap_path = _plot_scan_ratio_heatmaps(
        args.output_dir,
        scan,
        ratio_projections,
    )
    _print_results(results, scan)
    _print_phase_trajectories(phase_trajectories)
    for path in (
        case_csv_path,
        trajectory_csv_path,
        ratio_cube_csv_path,
        ratio_heatmap_csv_path,
        *phase_space_paths,
        threshold_path,
        predicate_path,
        ratio_heatmap_path,
    ):
        print(f"wrote {path}")
    print("all damping invariants passed")


if __name__ == "__main__":
    main()
