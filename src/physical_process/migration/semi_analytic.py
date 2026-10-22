"""Comparison-only integrated local semi-analytic migration.

This module preserves the archived ``semi_without_trap`` scenario for
controlled model comparisons. It is **not** the production migration law:
the production workflow integrates the signed torque in
:mod:`.evolution`, and no CLI option selects this module.

The local timescales are

``t_I(R)  = M^2 h^2 / (2 m Sigma R^2 Omega)``
``t_II(R) = 1 / (2 alpha h^2 Omega)``

and the path time is ``t_mig = integral t_local(R) d ln R``. Cells above the
selected gap-opening threshold use ``t_II``; all other cells use ``t_I``.
Both branches are inward and positive, so this scenario has no trap or
outward-blocking outcome. The same pAGN profile and ``alpha`` are used as in
the production torque comparison.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

from .. import constants as const
from .evolution import MigrationError, MigrationOutcome
from .gap import (
    crida_gap_parameter,
    hill_radius_over_scale_height,
    kanagawa_gap_parameter,
)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

TYPE_II_GAP_PARAMETER_THRESHOLD = 11.0
GAP_OPENING_CONDITIONS = ("kii", "crida")
GapOpeningCondition = Literal["kii", "crida"]


class SemiAnalyticMigrationError(MigrationError):
    """Raised when a semi-analytic migration path cannot be evaluated."""


def gap_opening_mask(
    mass_ratio: float,
    aspect_ratio: FloatArray,
    alpha: float,
    condition: GapOpeningCondition,
) -> BoolArray:
    """Return cells satisfying the selected gap-opening condition.

    ``kii`` uses ``K_II = q^2 / (alpha h^5) > 11``. ``crida`` uses the
    pressure-plus-viscosity criterion ``P < 1`` together with
    ``R_H/H > 1``. The criterion selects the local Type-II timescale only in
    this comparison scenario.
    """
    if not np.isfinite(mass_ratio) or mass_ratio <= 0.0:
        raise ValueError("mass ratio must be finite and positive")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive")
    if condition not in GAP_OPENING_CONDITIONS:
        raise ValueError(
            f"gap-opening condition must be one of {GAP_OPENING_CONDITIONS}"
        )

    local_aspect_ratio = np.asarray(aspect_ratio, dtype=float)
    valid = np.isfinite(local_aspect_ratio) & (local_aspect_ratio > 0.0)
    if condition == "kii":
        gap_parameter = kanagawa_gap_parameter(
            mass_ratio,
            local_aspect_ratio,
            alpha,
        )
        return np.asarray(
            valid
            & np.isfinite(gap_parameter)
            & (gap_parameter > TYPE_II_GAP_PARAMETER_THRESHOLD),
            dtype=bool,
        )

    hill_ratio = hill_radius_over_scale_height(
        mass_ratio,
        local_aspect_ratio,
    )
    crida_parameter = crida_gap_parameter(
        mass_ratio,
        local_aspect_ratio,
        alpha,
    )
    return np.asarray(
        valid
        & np.isfinite(hill_ratio)
        & np.isfinite(crida_parameter)
        & (crida_parameter < 1.0)
        & (hill_ratio > 1.0),
        dtype=bool,
    )


def semi_analytic_local_time_s(
    radius_cm: FloatArray,
    surface_density_g_cm2: FloatArray,
    aspect_ratio: FloatArray,
    migrator_mass_g: float,
    smbh_mass_msun: float,
    alpha: float,
    gap_opening_condition: GapOpeningCondition = "kii",
    consider_type_ii_migration: bool = True,
) -> FloatArray:
    """Return the comparison law's positive local ``r/|dr/dt|`` in seconds."""
    radius = np.asarray(radius_cm, dtype=float)
    surface_density = np.asarray(surface_density_g_cm2, dtype=float)
    local_aspect_ratio = np.asarray(aspect_ratio, dtype=float)
    if not (
        radius.ndim == surface_density.ndim == local_aspect_ratio.ndim == 1
        and radius.shape == surface_density.shape == local_aspect_ratio.shape
        and radius.size >= 2
    ):
        raise SemiAnalyticMigrationError("semi-analytic profile arrays must align")
    if not all(
        np.all(np.isfinite(values) & (values > 0.0))
        for values in (radius, surface_density, local_aspect_ratio)
    ):
        raise SemiAnalyticMigrationError("semi-analytic profile contains invalid samples")
    if migrator_mass_g <= 0.0 or smbh_mass_msun <= 0.0 or alpha <= 0.0:
        raise SemiAnalyticMigrationError("migrator, SMBH, and alpha must be positive")
    smbh_mass_g = smbh_mass_msun * const.msun
    mass_ratio = migrator_mass_g / smbh_mass_g
    orbital_frequency_rad_s = np.sqrt(const.G * smbh_mass_g / radius**3)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        type_i_time_s = (
            smbh_mass_g**2
            * local_aspect_ratio**2
            / (
                2.0
                * migrator_mass_g
                * surface_density
                * radius**2
                * orbital_frequency_rad_s
            )
        )
        type_ii_time_s = 1.0 / (
            2.0 * alpha * local_aspect_ratio**2 * orbital_frequency_rad_s
        )
    if consider_type_ii_migration:
        type_ii_mask = gap_opening_mask(
            mass_ratio, local_aspect_ratio, alpha, gap_opening_condition
        )
    else:
        type_ii_mask = np.zeros(radius.shape, dtype=bool)
    local_time_s = np.where(type_ii_mask, type_ii_time_s, type_i_time_s)
    if not np.all(np.isfinite(local_time_s) & (local_time_s > 0.0)):
        raise SemiAnalyticMigrationError(
            "semi-analytic local migration time contains invalid samples"
        )
    return np.asarray(local_time_s, dtype=float)


def semi_analytic_migration_outcome(
    radius_grid_cm: FloatArray,
    surface_density_g_cm2: FloatArray,
    aspect_ratio: FloatArray,
    migrator_mass_g: float,
    r_ini_cm: float,
    r_fin_cm: float,
    smbh_mass_msun: float,
    alpha: float,
    gap_opening_condition: GapOpeningCondition = "kii",
    consider_type_ii_migration: bool = True,
) -> MigrationOutcome:
    """Integrate the selected local semi-analytic migration time over ``d ln R``.

    Gap-closed cells use the local Type-I timescale. Selected gap-open cells use
    the local Type-II timescale. Both branches are inward and positive; invalid
    profiles raise :class:`SemiAnalyticMigrationError` rather than producing a
    physical trap outcome.
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
        raise SemiAnalyticMigrationError(
            "migration radii must be finite and positive"
        )
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

    local_time_s = semi_analytic_local_time_s(
        radius_cm,
        surface_density,
        local_aspect_ratio,
        migrator_mass_g,
        smbh_mass_msun,
        alpha,
        gap_opening_condition,
        consider_type_ii_migration,
    )

    target_radius_cm = max(r_fin_cm, float(radius_cm[0]))
    start_radius_cm = min(r_ini_cm, float(radius_cm[-1]))
    on_path = (radius_cm >= target_radius_cm) & (radius_cm <= start_radius_cm)
    segment_radius_cm = np.concatenate(
        (
            [target_radius_cm],
            radius_cm[
                on_path
                & (radius_cm > target_radius_cm)
                & (radius_cm < start_radius_cm)
            ],
            [start_radius_cm],
        )
    )
    if segment_radius_cm.size < 2:
        raise SemiAnalyticMigrationError(
            "migration path has insufficient semi-analytic samples"
        )
    segment_time_s = np.interp(segment_radius_cm, radius_cm, local_time_s)
    migration_time_s = float(
        np.trapezoid(segment_time_s, np.log(segment_radius_cm))
    )
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
    gap_opening_condition: GapOpeningCondition = "kii",
    consider_type_ii_migration: bool = True,
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
    "GapOpeningCondition",
    "SemiAnalyticMigrationError",
    "TYPE_II_GAP_PARAMETER_THRESHOLD",
    "gap_opening_mask",
    "migration_time_from_semi_analytic",
    "semi_analytic_local_time_s",
    "semi_analytic_migration_outcome",
]
