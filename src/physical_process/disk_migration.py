"""Signed torque-based disk migration helpers."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

if __package__ == "src.physical_process":
    from ..disk_quantity.disk_solve import DISK_VISCOSITY_ALPHA
    from . import constants as const
else:
    from disk_quantity.disk_solve import DISK_VISCOSITY_ALPHA
    import physical_process.constants as const

FloatArray = npt.NDArray[np.float64]
MigrationStatus = Literal["already_at_target", "finite", "trap_bound"]

TYPE_II_GAP_PARAMETER_THRESHOLD = 11.0
GAP_OPENING_CONDITIONS = ("kii", "crida")


class TorqueMigrationError(ValueError):
    """Raised when a torque migration path cannot be evaluated numerically."""


@dataclass(frozen=True)
class TorqueMigrationOutcome:
    """Physical outcome of one signed inward torque integration."""

    time_myr: float
    status: MigrationStatus
    blocking_radius_cm: float | None


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
        gap_parameter = mass_ratio**2 / (alpha * local_aspect_ratio**2)
        return valid & (gap_parameter > TYPE_II_GAP_PARAMETER_THRESHOLD)

    hill_radius_over_scale_height = (
        np.cbrt(mass_ratio / 3.0) / local_aspect_ratio
    )
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
            root_cm = inner_radius - inner_torque * (
                outer_radius - inner_radius
            ) / (outer_torque - inner_torque)
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
) -> TorqueMigrationOutcome:
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
        return TorqueMigrationOutcome(0.0, "already_at_target", None)

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
        return TorqueMigrationOutcome(
            np.inf,
            "trap_bound",
            _stable_blocking_radius_cm(segment_radius_cm, segment_torque_cgs),
        )

    smbh_mass_g = smbh_mass_msun * const.msun
    orbital_frequency_rad_s = np.sqrt(
        const.G * smbh_mass_g / segment_radius_cm**3
    )
    integrand = (
        0.5
        * migrator_mass_g
        * segment_radius_cm
        * orbital_frequency_rad_s
        / segment_torque_cgs
    )
    if not np.all(np.isfinite(integrand)):
        raise TorqueMigrationError("migration integrand contains nonfinite samples")

    migration_time_s = float(
        np.trapezoid(integrand[::-1], segment_radius_cm[::-1])
    )
    if not np.isfinite(migration_time_s) or migration_time_s <= 0.0:
        raise TorqueMigrationError("signed torque integration returned an invalid time")

    return TorqueMigrationOutcome(
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
