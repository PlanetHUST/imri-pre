"""Migration regime selection and radial time integration.

Physical chain implemented here::

    torque components (.torque)
        + gap decision (.gap)
        -> effective total torque   (Type-I+thermal, or Type-II where gap-open)
        -> dr/dt
        -> integrated migration time, or a trap-bound outcome (.traps)

Two independent integrators are provided and both are production paths,
selected by ``IMRIDynConfig.disk_model``:

``torque_migration_outcome`` (``disk_model = "torque_with_trap"``)
    Integrates the *signed* torque profile.  Because the torque is signed, an
    outward-torque region between the start and the target physically blocks
    inward delivery; that returns ``status="trap_bound"`` with ``time = inf``
    and the blocking radius, not a numerical failure.

``semi_analytic_migration_outcome`` (``disk_model = "semi_without_trap"``)
    Integrates local Type-I / Type-II migration timescales over ``d ln R``.
    This branch is unsigned and therefore cannot produce a trap.

Regime transition
-----------------
The Type-I -> Type-II switch is a hard per-cell switch on the gap criterion
(:mod:`.gap`).  It is intentionally discontinuous: the two prescriptions are
different physical models, not two limits of one expression, and the project
does not implement a blending law.  The discontinuity is inherited from the
model rather than introduced here, and Checkpoint E measures its size.

Units
-----
Radii [cm], torques [cgs, g cm^2 s^-2], masses [g] except ``smbh_mass_msun``,
returned times [Myr].
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

if __package__ == "src.physical_process.migration":
    from .. import constants as const
    from .gap import GAP_OPENING_CONDITIONS, gap_opening_mask
    from .torque import TorqueProfile
    from .traps import stable_blocking_radius_cm
else:
    import physical_process.constants as const
    from physical_process.migration.gap import (
        GAP_OPENING_CONDITIONS,
        gap_opening_mask,
    )
    from physical_process.migration.torque import TorqueProfile
    from physical_process.migration.traps import stable_blocking_radius_cm

FloatArray = npt.NDArray[np.float64]
MigrationStatus = Literal["already_at_target", "finite", "trap_bound"]


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


def select_migration_torque_si(
    torque_profile: TorqueProfile,
    mass_ratio: float,
    local_aspect_ratio: FloatArray,
    alpha: float,
    gap_opening_condition: str,
    consider_type_ii_migration: bool,
) -> MigrationTorqueSelection:
    """Compose the effective migration torque in SI units [kg m^2 s^-2].

    Gap-closed cells use ``Gamma_typeI + (1 - gap_attenuation) * Gamma_thermal``;
    cells selected by the gap criterion use ``Gamma_typeII`` instead.  Setting
    ``consider_type_ii_migration=False`` forces every cell to the Type-I branch.
    """
    type_i_thermal_torque_si = np.asarray(
        torque_profile.gamma_type_i_si
        + (1.0 - torque_profile.gap_attenuation_factor) * torque_profile.gamma_th_si,
        dtype=float,
    )
    if consider_type_ii_migration:
        type_ii_mask = gap_opening_mask(
            mass_ratio,
            np.asarray(local_aspect_ratio, dtype=float),
            alpha,
            gap_opening_condition,
        )
    else:
        type_ii_mask = np.zeros(type_i_thermal_torque_si.shape, dtype=bool)
    return MigrationTorqueSelection(
        torque_si=np.where(
            type_ii_mask,
            np.asarray(torque_profile.gamma_type_ii_si, dtype=float),
            type_i_thermal_torque_si,
        ),
        type_ii_mask=type_ii_mask,
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
            stable_blocking_radius_cm(segment_radius_cm, segment_torque_cgs),
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
    if (
        isinstance(gap_opening_condition, str)
        and gap_opening_condition not in GAP_OPENING_CONDITIONS
    ):
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


__all__ = [
    "GAP_OPENING_CONDITIONS",
    "MigrationError",
    "MigrationOutcome",
    "MigrationStatus",
    "MigrationTorqueSelection",
    "SemiAnalyticMigrationError",
    "TorqueMigrationError",
    "TorqueMigrationOutcome",
    "find_closest_index",
    "gap_opening_mask",
    "migration_time_from_semi_analytic",
    "migration_time_from_torque",
    "select_migration_torque_si",
    "semi_analytic_migration_outcome",
    "torque_migration_outcome",
]
