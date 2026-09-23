"""Signed radial evolution under the fixed production torque profile.

Torque composition belongs to :mod:`.torque`. This module only validates a
radial path, distinguishes physical blockage from numerical failure, and
integrates ``dt = (dL/dR) dR / Gamma`` in CGS units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

if __package__ == "src.physical_process.migration":
    from .. import constants as const
    from .traps import inward_delivery_blocked
else:
    import physical_process.constants as const
    from physical_process.migration.traps import inward_delivery_blocked

FloatArray = npt.NDArray[np.float64]
MigrationStatus = Literal[
    "already_at_target",
    "finite",
    "trap_bound",
    "outward_blocked",
]


class MigrationError(ValueError):
    """Raised when a migration path cannot be evaluated numerically."""


class TorqueMigrationError(MigrationError):
    """Raised when a torque migration path cannot be evaluated numerically."""


@dataclass(frozen=True)
class MigrationOutcome:
    """Physical outcome of one inward migration integration."""

    time_myr: float
    status: MigrationStatus
    blocking_radius_cm: float | None


TorqueMigrationOutcome = MigrationOutcome


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
    gamma_total_cgs: FloatArray,
    migrator_mass_g: float,
    r_ini_cm: float,
    r_fin_cm: float,
    smbh_mass_msun: float,
) -> MigrationOutcome:
    """Integrate the fixed signed torque and return time or physical blockage.

    Malformed, nonfinite, or out-of-domain profiles raise
    :class:`TorqueMigrationError`. ``time_myr = inf`` is reserved for a
    physically non-inward path; a resolved stable trap supplies its radius.
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
    torque_cgs = np.asarray(gamma_total_cgs, dtype=float)
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
    radius_grid, unique_indices = np.unique(radius_grid, return_index=True)
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
    on_path = (radius_grid >= target_radius_cm) & (radius_grid <= start_radius_cm)
    segment_radius_cm = np.concatenate(
        (
            [target_radius_cm],
            radius_grid[
                on_path
                & (radius_grid > target_radius_cm)
                & (radius_grid < start_radius_cm)
            ],
            [start_radius_cm],
        )
    )
    segment_torque_cgs = np.interp(segment_radius_cm, radius_grid, torque_cgs)
    if segment_radius_cm.size < 2:
        raise TorqueMigrationError("migration path has insufficient torque samples")

    # Trap classification uses the full profile so basin width is independent
    # of the object's start radius; the requested path is applied afterwards.
    blocked, blocking_radius_cm = inward_delivery_blocked(
        radius_grid,
        torque_cgs,
        inner_radius_rg=target_radius_cm,
        outer_radius_rg=start_radius_cm,
    )
    if blocked:
        status: MigrationStatus = (
            "trap_bound" if blocking_radius_cm is not None else "outward_blocked"
        )
        return MigrationOutcome(np.inf, status, blocking_radius_cm)

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
    return MigrationOutcome(migration_time_s / const.Myr, "finite", None)


def migration_time_from_torque(
    radius_grid_cm: FloatArray,
    gamma_total_cgs: FloatArray,
    migrator_mass_g: float,
    r_ini_cm: float,
    r_fin_cm: float,
    smbh_mass_msun: float,
) -> float:
    """Return the fixed-torque migration time [Myr]."""
    return torque_migration_outcome(
        radius_grid_cm,
        gamma_total_cgs,
        migrator_mass_g,
        r_ini_cm,
        r_fin_cm,
        smbh_mass_msun,
    ).time_myr


__all__ = [
    "MigrationError",
    "MigrationOutcome",
    "MigrationStatus",
    "TorqueMigrationError",
    "TorqueMigrationOutcome",
    "find_closest_index",
    "migration_time_from_torque",
    "torque_migration_outcome",
]
