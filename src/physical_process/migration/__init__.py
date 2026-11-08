"""Default inward Type-I migration and reusable torque-comparison physics.

The single-host workflow uses ``type_i``. Gap/torque/trap functions remain
available to opt-in comparison models, but are not the default prescription.
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
from .type_i import type_i_drift_cm_s, type_i_local_time_s
from .traps import (
    TorqueTrapRadii,
    classify_torque_traps,
    stable_blocking_radius_cm,
)

__all__ = [
    "CRIDA_DEEP_GAP_PARAMETER",
    "GapDiagnostics",
    "KANAGAWA_CALIBRATED_MAX_PARAMETER",
    "KANAGAWA_DEEP_GAP_PARAMETER",
    "KANAGAWA_GAP_DEPTH_COEFFICIENT",
    "MigrationError",
    "MigrationOutcome",
    "MigrationParameters",
    "MigrationStatus",
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
    "hill_radius_over_scale_height",
    "inward_only_counterfactual_torque",
    "kanagawa_gap_depth_fraction",
    "kanagawa_gap_parameter",
    "migration_time_from_torque",
    "stable_blocking_radius_cm",
    "torque_migration_outcome",
    "torque_si_to_cgs",
    "type_i_drift_cm_s",
    "type_i_local_time_s",
]
