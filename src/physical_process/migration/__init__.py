"""Fixed AGN-disk migration and trap physics.

``gap`` owns continuous depletion and deep-gap diagnostics; ``torque`` owns the
single Jimenez-Masset plus Grishin-2024 torque; ``traps`` classifies signed
zeros; and ``evolution`` integrates that torque without selecting alternatives.
``semi_analytic`` preserves the archived integrated local Type-I/Type-II
scenario for explicit comparison only; production never selects it.
"""

from .evolution import (
    MigrationError,
    MigrationOutcome,
    MigrationStatus,
    TorqueMigrationError,
    TorqueMigrationOutcome,
    find_closest_index,
    migration_time_from_torque,
    torque_migration_outcome,
)
from .diagnostics import inward_only_counterfactual_torque
from .gap import (
    CRIDA_DEEP_GAP_PARAMETER,
    KANAGAWA_CALIBRATED_MAX_PARAMETER,
    KANAGAWA_DEEP_GAP_PARAMETER,
    KANAGAWA_GAP_DEPTH_COEFFICIENT,
    GapDiagnostics,
    crida_gap_parameter,
    gap_diagnostics,
    hill_radius_over_scale_height,
    kanagawa_gap_depth_fraction,
    kanagawa_gap_parameter,
)
from .torque import (
    MigrationParameters,
    ThermalTorqueTerms,
    TorqueProfile,
    compose_migration_torque_si,
    compute_thermal_torque_terms,
    compute_torque_profile,
    torque_si_to_cgs,
)
from .semi_analytic import (
    GAP_OPENING_CONDITIONS,
    TYPE_II_GAP_PARAMETER_THRESHOLD,
    SemiAnalyticMigrationError,
    gap_opening_mask,
    migration_time_from_semi_analytic,
    semi_analytic_migration_outcome,
)
from .traps import (
    TorqueTrapRadii,
    classify_torque_traps,
    stable_blocking_radius_cm,
)

__all__ = [
    "CRIDA_DEEP_GAP_PARAMETER",
    "GAP_OPENING_CONDITIONS",
    "GapDiagnostics",
    "KANAGAWA_CALIBRATED_MAX_PARAMETER",
    "KANAGAWA_DEEP_GAP_PARAMETER",
    "KANAGAWA_GAP_DEPTH_COEFFICIENT",
    "MigrationError",
    "MigrationOutcome",
    "MigrationParameters",
    "MigrationStatus",
    "SemiAnalyticMigrationError",
    "TYPE_II_GAP_PARAMETER_THRESHOLD",
    "ThermalTorqueTerms",
    "TorqueMigrationError",
    "TorqueMigrationOutcome",
    "TorqueProfile",
    "TorqueTrapRadii",
    "classify_torque_traps",
    "compose_migration_torque_si",
    "compute_thermal_torque_terms",
    "compute_torque_profile",
    "crida_gap_parameter",
    "find_closest_index",
    "gap_diagnostics",
    "gap_opening_mask",
    "hill_radius_over_scale_height",
    "inward_only_counterfactual_torque",
    "kanagawa_gap_depth_fraction",
    "kanagawa_gap_parameter",
    "migration_time_from_semi_analytic",
    "migration_time_from_torque",
    "semi_analytic_migration_outcome",
    "stable_blocking_radius_cm",
    "torque_migration_outcome",
    "torque_si_to_cgs",
]
