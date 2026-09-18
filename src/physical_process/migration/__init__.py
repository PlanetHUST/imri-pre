"""Canonical AGN-disk migration physics.

Scientific dependency graph of this package::

    disk state (Sigma, h, Omega, T, rho, kappa)
        |
        v
    torque.py    Lindblad + corotation + thermal + Type-II torque components
        |
        v
    gap.py       dimensionless gap diagnostics -> gap-open decision
        |
        v
    evolution.py effective torque -> dr/dt -> migration time / trap-bound
        |
        v
    traps.py     torque zeros -> stable trap / anti-trap / boundary / unresolved

Module ownership
----------------
``torque``      torque components only; owns no threshold
``gap``         every gap parameter and threshold; owns no torque
``traps``       trap identification from any signed torque profile
``evolution``   regime selection and time integration; owns no equation of the above

Anything that needs a gap parameter, a gap threshold or a trap radius - including
plotting scripts - must import it from here rather than recomputing it.
"""

from .evolution import (
    MigrationError,
    MigrationOutcome,
    MigrationStatus,
    MigrationTorqueSelection,
    SemiAnalyticMigrationError,
    TorqueMigrationError,
    TorqueMigrationOutcome,
    find_closest_index,
    migration_time_from_semi_analytic,
    migration_time_from_torque,
    select_migration_torque_si,
    semi_analytic_migration_outcome,
    torque_migration_outcome,
)
from .gap import (
    DEFAULT_GAP_CRITERION,
    GAP_CRITERION_CRIDA,
    GAP_CRITERION_KANAGAWA,
    GAP_CRITERION_KII,
    GAP_OPENING_CONDITIONS,
    KANAGAWA_GAP_PARAMETER_THRESHOLD,
    KII_GAP_PARAMETER_THRESHOLD,
    TYPE_II_GAP_PARAMETER_THRESHOLD,
    GapCriterion,
    GapDiagnostics,
    crida_gap_parameter,
    depletion_fraction,
    gap_diagnostics,
    gap_opening_mask,
    hill_radius_over_scale_height,
    kanagawa_gap_parameter,
    kii_gap_parameter,
    threshold_for_depletion,
)
from .torque import (
    MigrationParameters,
    TorqueProfile,
    compute_torque_profile,
    torque_si_to_cgs,
)
from .traps import (
    DEFAULT_MIN_TRAP_SEPARATION_CELLS,
    TorqueTrapRadii,
    classify_torque_traps,
    stable_blocking_radius_cm,
)

__all__ = [
    "DEFAULT_GAP_CRITERION",
    "DEFAULT_MIN_TRAP_SEPARATION_CELLS",
    "GAP_CRITERION_CRIDA",
    "GAP_CRITERION_KANAGAWA",
    "GAP_CRITERION_KII",
    "GAP_OPENING_CONDITIONS",
    "GapCriterion",
    "GapDiagnostics",
    "KANAGAWA_GAP_PARAMETER_THRESHOLD",
    "KII_GAP_PARAMETER_THRESHOLD",
    "MigrationError",
    "MigrationOutcome",
    "MigrationParameters",
    "MigrationStatus",
    "MigrationTorqueSelection",
    "SemiAnalyticMigrationError",
    "TYPE_II_GAP_PARAMETER_THRESHOLD",
    "TorqueMigrationError",
    "TorqueMigrationOutcome",
    "TorqueProfile",
    "TorqueTrapRadii",
    "classify_torque_traps",
    "compute_torque_profile",
    "crida_gap_parameter",
    "depletion_fraction",
    "find_closest_index",
    "gap_diagnostics",
    "gap_opening_mask",
    "hill_radius_over_scale_height",
    "kanagawa_gap_parameter",
    "kii_gap_parameter",
    "threshold_for_depletion",
    "migration_time_from_semi_analytic",
    "migration_time_from_torque",
    "select_migration_torque_si",
    "semi_analytic_migration_outcome",
    "stable_blocking_radius_cm",
    "torque_migration_outcome",
    "torque_si_to_cgs",
]
