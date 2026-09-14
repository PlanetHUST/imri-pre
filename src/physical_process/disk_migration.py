"""Signed torque and integrated semi-analytic disk-migration helpers."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

if __package__ == "src.physical_process":
    from ..disk_quantity.migration_torque import (
        TorqueProfile,
        comp_gap_parameter_dimensionless,
    )
    from . import constants as const
else:
    from disk_quantity.migration_torque import (
        TorqueProfile,
        comp_gap_parameter_dimensionless,
    )
    import physical_process.constants as const

FloatArray = npt.NDArray[np.float64]
MigrationStatus = Literal["already_at_target", "finite", "trap_bound"]

TYPE_II_GAP_PARAMETER_THRESHOLD = 11.0
GAP_DEPTH_COEFFICIENT = 0.04
GAP_OPENING_CONDITIONS = ("kii", "crida")
CONTINUOUS_GAP = "continuous_gap"
CLASSICAL_TYPE2 = "classical_type2"
UNGAPPED = "ungapped"
MIGRATION_MODELS = (CONTINUOUS_GAP, CLASSICAL_TYPE2, UNGAPPED)
DEFAULT_MIGRATION_MODEL = CONTINUOUS_GAP


class MigrationError(ValueError):
    """Raised when a migration path cannot be evaluated numerically."""


class TorqueMigrationError(MigrationError):
    """Raised when a torque migration path cannot be evaluated numerically."""


class SemiAnalyticMigrationError(MigrationError):
    """Raised when a semi-analytic migration path cannot be evaluated."""


@dataclass(frozen=True)
class MigrationOutcome:
    """Physical outcome of one inward migration integration."""

    time_myr: float
    status: MigrationStatus
    blocking_radius_cm: float | None


TorqueMigrationOutcome = MigrationOutcome


@dataclass(frozen=True)
class MigrationTorqueSelection:
    """Selected migration torque in SI and the cells using Type II."""

    torque_si: FloatArray
    type_ii_mask: npt.NDArray[np.bool_]


@dataclass(frozen=True)
class MigrationTorqueDiagnostics:
    """Inspectable dimensionless factors and torque components in SI [N m]."""

    gap_parameter_dimensionless: FloatArray
    gap_depth_factor: FloatArray
    local_disk_mass_ratio: FloatArray
    gamma_grav_ungapped_si: FloatArray
    gamma_grav_gap_si: FloatArray
    gamma_thermal_si: FloatArray
    gamma_type_ii_si: FloatArray
    gamma_total_si: FloatArray
    deep_gap_mask: npt.NDArray[np.bool_]
    type_ii_mask: npt.NDArray[np.bool_]


def gap_depth_factor(gap_parameter_dimensionless: FloatArray) -> FloatArray:
    """Return Sigma_gap/Sigma_0 = (1 + 0.04 K)^-1."""
    gap_parameter = np.asarray(gap_parameter_dimensionless, dtype=float)
    if np.any(~np.isfinite(gap_parameter)) or np.any(gap_parameter < 0.0):
        raise ValueError("gap parameter must be finite and nonnegative")
    return np.asarray(
        1.0 / (1.0 + GAP_DEPTH_COEFFICIENT * gap_parameter),
        dtype=float,
    )


def migration_torque_diagnostics(
    torque_profile: TorqueProfile,
    mass_ratio: float,
    local_aspect_ratio: FloatArray,
    alpha: float,
    gap_opening_condition: str,
    consider_type_ii_migration: bool,
    *,
    migration_model: str = DEFAULT_MIGRATION_MODEL,
    migrator_mass_kg: float | None = None,
) -> MigrationTorqueDiagnostics:
    """Return migration components and gap diagnostics on the torque grid.

    ``migrator_mass_kg`` is optional so legacy synthetic ``TorqueProfile``
    callers can still select a total torque. When omitted, only the local disk
    mass ratio diagnostic is unavailable (NaN); production diagnostics pass it
    explicitly.
    """
    if migration_model not in MIGRATION_MODELS:
        raise ValueError(f"migration model must be one of {MIGRATION_MODELS}")
    if mass_ratio <= 0.0 or alpha <= 0.0:
        raise ValueError("mass ratio and alpha must be positive")
    if (
        consider_type_ii_migration
        and gap_opening_condition not in GAP_OPENING_CONDITIONS
    ):
        raise ValueError(
            f"gap-opening condition must be one of {GAP_OPENING_CONDITIONS}"
        )

    aspect_ratio = np.asarray(local_aspect_ratio, dtype=float)
    gap_parameter = np.asarray(
        comp_gap_parameter_dimensionless(mass_ratio, aspect_ratio, alpha),
        dtype=float,
    )
    f_gap = gap_depth_factor(gap_parameter)
    gamma_grav_ungapped_si = np.asarray(
        torque_profile.gamma_type_i_si,
        dtype=float,
    )
    gamma_thermal_si = np.asarray(torque_profile.gamma_th_si, dtype=float)
    gamma_type_ii_si = np.asarray(torque_profile.gamma_type_ii_si, dtype=float)
    component_shapes = {
        gap_parameter.shape,
        gamma_grav_ungapped_si.shape,
        gamma_thermal_si.shape,
        gamma_type_ii_si.shape,
    }
    if len(component_shapes) != 1:
        raise ValueError("migration torque components must have the same shape")

    gamma_grav_gap_si = f_gap * gamma_grav_ungapped_si
    gamma_ungapped_total_si = gamma_grav_ungapped_si + gamma_thermal_si
    deep_gap_mask = gap_parameter > TYPE_II_GAP_PARAMETER_THRESHOLD
    type_ii_mask = np.zeros(gap_parameter.shape, dtype=bool)

    effective_model = migration_model if consider_type_ii_migration else UNGAPPED
    if effective_model == CONTINUOUS_GAP:
        gamma_total_si = gamma_grav_gap_si + gamma_thermal_si
    elif effective_model == CLASSICAL_TYPE2:
        type_ii_mask = gap_opening_mask(
            mass_ratio,
            aspect_ratio,
            alpha,
            gap_opening_condition,
        )
        gamma_total_si = np.where(
            type_ii_mask,
            gamma_type_ii_si,
            gamma_ungapped_total_si,
        )
    else:
        gamma_total_si = gamma_ungapped_total_si

    if migrator_mass_kg is None:
        local_disk_mass_ratio = np.full(gap_parameter.shape, np.nan, dtype=float)
    else:
        if not np.isfinite(migrator_mass_kg) or migrator_mass_kg <= 0.0:
            raise ValueError("migrator mass must be finite and positive")
        surface_density_kg_m2 = np.asarray(
            torque_profile.surface_density_kg_m2,
            dtype=float,
        )
        radius_m = np.asarray(torque_profile.radius_m, dtype=float)
        if not (
            surface_density_kg_m2.shape == radius_m.shape == gap_parameter.shape
        ):
            raise ValueError("disk-mass diagnostic arrays must have the same shape")
        local_disk_mass_ratio = (
            4.0 * np.pi * surface_density_kg_m2 * radius_m**2 / migrator_mass_kg
        )

    return MigrationTorqueDiagnostics(
        gap_parameter_dimensionless=gap_parameter,
        gap_depth_factor=f_gap,
        local_disk_mass_ratio=np.asarray(local_disk_mass_ratio, dtype=float),
        gamma_grav_ungapped_si=gamma_grav_ungapped_si,
        gamma_grav_gap_si=np.asarray(gamma_grav_gap_si, dtype=float),
        gamma_thermal_si=gamma_thermal_si,
        gamma_type_ii_si=gamma_type_ii_si,
        gamma_total_si=np.asarray(gamma_total_si, dtype=float),
        deep_gap_mask=np.asarray(deep_gap_mask, dtype=bool),
        type_ii_mask=np.asarray(type_ii_mask, dtype=bool),
    )


def gap_opening_mask(
    mass_ratio: float,
    aspect_ratio: FloatArray,
    alpha: float,
    condition: str,
) -> npt.NDArray[np.bool_]:
    """Return cells satisfying the selected dimensionless gap condition."""
    if mass_ratio <= 0.0 or alpha <= 0.0:
        raise ValueError("mass ratio and alpha must be positive")
    if condition not in GAP_OPENING_CONDITIONS:
        raise ValueError(
            f"gap-opening condition must be one of {GAP_OPENING_CONDITIONS}"
        )

    local_aspect_ratio = np.asarray(aspect_ratio, dtype=float)
    valid = np.isfinite(local_aspect_ratio) & (local_aspect_ratio > 0.0)
    if condition == "kii":
        gap_parameter = comp_gap_parameter_dimensionless(
            mass_ratio,
            local_aspect_ratio,
            alpha,
        )
        return valid & (gap_parameter > TYPE_II_GAP_PARAMETER_THRESHOLD)

    hill_radius_over_scale_height = np.cbrt(mass_ratio / 3.0) / local_aspect_ratio
    crida_gap_parameter = (
        0.75 / hill_radius_over_scale_height
        + 50.0 * alpha * local_aspect_ratio**2 / mass_ratio
    )
    return (
        valid
        & np.isfinite(crida_gap_parameter)
        & np.isfinite(hill_radius_over_scale_height)
        & (crida_gap_parameter < 1.0)
        & (hill_radius_over_scale_height > 1.0)
    )


def select_migration_torque_si(
    torque_profile: TorqueProfile,
    mass_ratio: float,
    local_aspect_ratio: FloatArray,
    alpha: float,
    gap_opening_condition: str,
    consider_type_ii_migration: bool,
    *,
    migration_model: str = DEFAULT_MIGRATION_MODEL,
) -> MigrationTorqueSelection:
    """Compose the selected migration torque in SI [N m].

    The required positional arguments and return type are retained for legacy
    callers. ``migration_model`` is a keyword-only extension.
    """
    diagnostics = migration_torque_diagnostics(
        torque_profile,
        mass_ratio,
        local_aspect_ratio,
        alpha,
        gap_opening_condition,
        consider_type_ii_migration,
        migration_model=migration_model,
    )
    return MigrationTorqueSelection(
        torque_si=diagnostics.gamma_total_si,
        type_ii_mask=diagnostics.type_ii_mask,
    )


def find_closest_index(radius_grid_cm: FloatArray, radius_cm: float) -> int:
    """Return the index of the radius [cm] nearest to ``radius_cm``."""
    radius_grid = np.asarray(radius_grid_cm, dtype=float)
    if radius_grid.size == 0:
        raise ValueError("radius grid must not be empty")
    insertion_index = int(np.searchsorted(radius_grid, radius_cm))
    if insertion_index == 0:
        return 0
    if insertion_index == radius_grid.size:
        return radius_grid.size - 1
    left_index = insertion_index - 1
    if abs(radius_grid[left_index] - radius_cm) <= abs(
        radius_grid[insertion_index] - radius_cm
    ):
        return left_index
    return insertion_index


def _stable_blocking_radius_cm(
    radius_cm: FloatArray,
    torque_cgs: FloatArray,
) -> float | None:
    stable_roots_cm: list[float] = []
    for index in range(radius_cm.size - 1):
        inner_torque = torque_cgs[index]
        outer_torque = torque_cgs[index + 1]
        if inner_torque > 0.0 and outer_torque < 0.0:
            inner_radius = radius_cm[index]
            outer_radius = radius_cm[index + 1]
            root_cm = inner_radius - inner_torque * (outer_radius - inner_radius) / (
                outer_torque - inner_torque
            )
            stable_roots_cm.append(float(root_cm))

    zero_indices = np.where(torque_cgs == 0.0)[0]
    for index in zero_indices:
        if (
            index > 0
            and index + 1 < torque_cgs.size
            and torque_cgs[index - 1] > 0.0
            and torque_cgs[index + 1] < 0.0
        ):
            stable_roots_cm.append(float(radius_cm[index]))

    return max(stable_roots_cm) if stable_roots_cm else None


def torque_migration_outcome(
    radius_grid_cm: FloatArray,
    gamma_tot_cgs: FloatArray,
    migrator_mass_g: float,
    r_ini_cm: float,
    r_fin_cm: float,
    smbh_mass_msun: float,
) -> MigrationOutcome:
    """Return the physical outcome of signed inward torque migration.

    Numerical/profile problems raise ``TorqueMigrationError``. ``np.inf`` is
    reserved for a physical torque profile that blocks continuous inward
    delivery to the target.
    """
    if migrator_mass_g <= 0.0 or smbh_mass_msun <= 0.0:
        raise TorqueMigrationError("migrator and SMBH masses must be positive")
    if (
        not np.isfinite(r_ini_cm)
        or not np.isfinite(r_fin_cm)
        or r_ini_cm <= 0.0
        or r_fin_cm <= 0.0
    ):
        raise TorqueMigrationError("migration radii must be finite and positive")
    if r_ini_cm <= r_fin_cm:
        return MigrationOutcome(0.0, "already_at_target", None)

    radius_grid = np.asarray(radius_grid_cm, dtype=float)
    torque_cgs = np.asarray(gamma_tot_cgs, dtype=float)
    if radius_grid.ndim != 1 or torque_cgs.ndim != 1:
        raise TorqueMigrationError("radius and torque grids must be one-dimensional")
    if radius_grid.shape != torque_cgs.shape:
        raise TorqueMigrationError("radius and torque grids must have the same shape")
    if radius_grid.size < 2:
        raise TorqueMigrationError("torque migration requires at least two samples")
    if not np.all(np.isfinite(radius_grid) & (radius_grid > 0.0)):
        raise TorqueMigrationError("radius grid contains invalid samples")
    if not np.all(np.isfinite(torque_cgs)):
        raise TorqueMigrationError("torque grid contains nonfinite samples")

    order = np.argsort(radius_grid)
    radius_grid = radius_grid[order]
    torque_cgs = torque_cgs[order]
    unique_radius_cm, unique_indices = np.unique(radius_grid, return_index=True)
    radius_grid = unique_radius_cm
    torque_cgs = torque_cgs[unique_indices]
    if radius_grid.size < 2:
        raise TorqueMigrationError("torque migration requires two unique radii")

    lower_tolerance_cm = max(np.spacing(radius_grid[0]), radius_grid[0] * 1.0e-12)
    upper_tolerance_cm = max(np.spacing(radius_grid[-1]), radius_grid[-1] * 1.0e-12)
    if (
        r_fin_cm < radius_grid[0] - lower_tolerance_cm
        or r_ini_cm > radius_grid[-1] + upper_tolerance_cm
    ):
        raise TorqueMigrationError(
            "requested migration path lies outside the torque-profile radius domain"
        )

    target_radius_cm = max(r_fin_cm, float(radius_grid[0]))
    start_radius_cm = min(r_ini_cm, float(radius_grid[-1]))
    mask = (radius_grid >= target_radius_cm) & (radius_grid <= start_radius_cm)
    segment_radius_cm = np.concatenate(
        (
            [target_radius_cm],
            radius_grid[
                mask
                & (radius_grid > target_radius_cm)
                & (radius_grid < start_radius_cm)
            ],
            [start_radius_cm],
        )
    )
    segment_torque_cgs = np.interp(segment_radius_cm, radius_grid, torque_cgs)
    if segment_radius_cm.size < 2:
        raise TorqueMigrationError("migration path has insufficient torque samples")

    if np.any(segment_torque_cgs >= 0.0):
        return MigrationOutcome(
            np.inf,
            "trap_bound",
            _stable_blocking_radius_cm(segment_radius_cm, segment_torque_cgs),
        )

    smbh_mass_g = smbh_mass_msun * const.msun
    orbital_frequency_rad_s = np.sqrt(const.G * smbh_mass_g / segment_radius_cm**3)
    integrand = (
        0.5
        * migrator_mass_g
        * segment_radius_cm
        * orbital_frequency_rad_s
        / segment_torque_cgs
    )
    if not np.all(np.isfinite(integrand)):
        raise TorqueMigrationError("migration integrand contains nonfinite samples")

    migration_time_s = float(np.trapezoid(integrand[::-1], segment_radius_cm[::-1]))
    if not np.isfinite(migration_time_s) or migration_time_s <= 0.0:
        raise TorqueMigrationError("signed torque integration returned an invalid time")

    return MigrationOutcome(
        float(migration_time_s / const.Myr),
        "finite",
        None,
    )


def migration_time_from_torque(
    radius_grid_cm: FloatArray,
    gamma_tot_cgs: FloatArray,
    migrator_mass_g: float,
    r_ini_cm: float,
    r_fin_cm: float,
    smbh_mass_msun: float,
) -> float:
    """Return signed torque-integrated migration time [Myr]."""
    return torque_migration_outcome(
        radius_grid_cm,
        gamma_tot_cgs,
        migrator_mass_g,
        r_ini_cm,
        r_fin_cm,
        smbh_mass_msun,
    ).time_myr


def semi_analytic_migration_outcome(
    radius_grid_cm: FloatArray,
    surface_density_g_cm2: FloatArray,
    aspect_ratio: FloatArray,
    migrator_mass_g: float,
    r_ini_cm: float,
    r_fin_cm: float,
    smbh_mass_msun: float,
    alpha: float,
    gap_opening_condition: str,
    consider_type_ii_migration: bool,
) -> MigrationOutcome:
    """Integrate the selected local semi-analytic migration time over ``d ln R``.

    Gap-closed cells use the local Type-I timescale. Selected gap-open cells use
    the local Type-II timescale. Both branches are
    inward and positive; invalid profiles raise ``SemiAnalyticMigrationError``
    rather than producing a physical trap outcome.
    """
    if migrator_mass_g <= 0.0 or smbh_mass_msun <= 0.0 or alpha <= 0.0:
        raise SemiAnalyticMigrationError(
            "migrator mass, SMBH mass, and alpha must be positive"
        )
    if gap_opening_condition not in GAP_OPENING_CONDITIONS:
        raise SemiAnalyticMigrationError(
            f"gap-opening condition must be one of {GAP_OPENING_CONDITIONS}"
        )
    if (
        not np.isfinite(r_ini_cm)
        or not np.isfinite(r_fin_cm)
        or r_ini_cm <= 0.0
        or r_fin_cm <= 0.0
    ):
        raise SemiAnalyticMigrationError("migration radii must be finite and positive")
    if r_ini_cm <= r_fin_cm:
        return MigrationOutcome(0.0, "already_at_target", None)

    radius_cm = np.asarray(radius_grid_cm, dtype=float)
    surface_density = np.asarray(surface_density_g_cm2, dtype=float)
    local_aspect_ratio = np.asarray(aspect_ratio, dtype=float)
    arrays = (radius_cm, surface_density, local_aspect_ratio)
    if any(values.ndim != 1 for values in arrays):
        raise SemiAnalyticMigrationError(
            "semi-analytic profile arrays must be one-dimensional"
        )
    if not (radius_cm.shape == surface_density.shape == local_aspect_ratio.shape):
        raise SemiAnalyticMigrationError(
            "semi-analytic profile arrays must have the same shape"
        )
    if radius_cm.size < 2:
        raise SemiAnalyticMigrationError(
            "semi-analytic migration requires at least two samples"
        )
    if not all(np.all(np.isfinite(values) & (values > 0.0)) for values in arrays):
        raise SemiAnalyticMigrationError(
            "semi-analytic profile contains invalid samples"
        )

    order = np.argsort(radius_cm)
    radius_cm = radius_cm[order]
    surface_density = surface_density[order]
    local_aspect_ratio = local_aspect_ratio[order]
    unique_radius_cm, unique_indices = np.unique(radius_cm, return_index=True)
    radius_cm = unique_radius_cm
    surface_density = surface_density[unique_indices]
    local_aspect_ratio = local_aspect_ratio[unique_indices]
    if radius_cm.size < 2:
        raise SemiAnalyticMigrationError(
            "semi-analytic migration requires two unique radii"
        )

    lower_tolerance_cm = max(np.spacing(radius_cm[0]), radius_cm[0] * 1.0e-12)
    upper_tolerance_cm = max(np.spacing(radius_cm[-1]), radius_cm[-1] * 1.0e-12)
    if (
        r_fin_cm < radius_cm[0] - lower_tolerance_cm
        or r_ini_cm > radius_cm[-1] + upper_tolerance_cm
    ):
        raise SemiAnalyticMigrationError(
            "requested migration path lies outside the semi-analytic profile domain"
        )

    smbh_mass_g = smbh_mass_msun * const.msun
    mass_ratio = migrator_mass_g / smbh_mass_g
    orbital_frequency_rad_s = np.sqrt(const.G * smbh_mass_g / radius_cm**3)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        type_i_time_s = (
            smbh_mass_g**2
            * local_aspect_ratio**2
            / (
                2.0
                * migrator_mass_g
                * surface_density
                * radius_cm**2
                * orbital_frequency_rad_s
            )
        )
        type_ii_time_s = 1.0 / (
            2.0 * alpha * local_aspect_ratio**2 * orbital_frequency_rad_s
        )

    if consider_type_ii_migration:
        type_ii_mask = gap_opening_mask(
            mass_ratio,
            local_aspect_ratio,
            alpha,
            gap_opening_condition,
        )
    else:
        type_ii_mask = np.zeros(radius_cm.shape, dtype=bool)
    local_time_s = np.where(type_ii_mask, type_ii_time_s, type_i_time_s)
    if not np.all(np.isfinite(local_time_s) & (local_time_s > 0.0)):
        raise SemiAnalyticMigrationError(
            "semi-analytic local migration time contains invalid samples"
        )

    target_radius_cm = max(r_fin_cm, float(radius_cm[0]))
    start_radius_cm = min(r_ini_cm, float(radius_cm[-1]))
    mask = (radius_cm >= target_radius_cm) & (radius_cm <= start_radius_cm)
    segment_radius_cm = np.concatenate(
        (
            [target_radius_cm],
            radius_cm[
                mask & (radius_cm > target_radius_cm) & (radius_cm < start_radius_cm)
            ],
            [start_radius_cm],
        )
    )
    if segment_radius_cm.size < 2:
        raise SemiAnalyticMigrationError(
            "migration path has insufficient semi-analytic samples"
        )
    segment_time_s = np.interp(segment_radius_cm, radius_cm, local_time_s)
    migration_time_s = float(np.trapezoid(segment_time_s, np.log(segment_radius_cm)))
    if not np.isfinite(migration_time_s) or migration_time_s <= 0.0:
        raise SemiAnalyticMigrationError(
            "semi-analytic integration returned an invalid time"
        )
    return MigrationOutcome(migration_time_s / const.Myr, "finite", None)


def migration_time_from_semi_analytic(
    radius_grid_cm: FloatArray,
    surface_density_g_cm2: FloatArray,
    aspect_ratio: FloatArray,
    migrator_mass_g: float,
    r_ini_cm: float,
    r_fin_cm: float,
    smbh_mass_msun: float,
    alpha: float,
    gap_opening_condition: str,
    consider_type_ii_migration: bool,
) -> float:
    """Return the integrated semi-analytic migration time [Myr]."""
    return semi_analytic_migration_outcome(
        radius_grid_cm,
        surface_density_g_cm2,
        aspect_ratio,
        migrator_mass_g,
        r_ini_cm,
        r_fin_cm,
        smbh_mass_msun,
        alpha,
        gap_opening_condition,
        consider_type_ii_migration,
    ).time_myr
