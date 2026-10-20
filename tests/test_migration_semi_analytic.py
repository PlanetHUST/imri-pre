"""Correctness tests for the comparison-only semi-analytic migration."""

from __future__ import annotations

import numpy as np

from physical_process import constants as const
from physical_process.migration.semi_analytic import (
    SemiAnalyticMigrationError,
    gap_opening_mask,
    semi_analytic_migration_outcome,
)

ALPHA = 0.1
SMBH_MSUN = 1.0e5
RADIUS_CM = np.logspace(12.0, 14.0, 400)
RADIUS_REFERENCE_CM = 1.0e13
SURFACE_DENSITY_G_CM2 = 1.0e3


def _constant_type_i_profile():
    """Return h(R) making t_I independent of R for constant Sigma."""
    migrator_msun = 0.1
    aspect = 0.05 * (RADIUS_CM / RADIUS_REFERENCE_CM) ** 0.25
    return migrator_msun, aspect


def _constant_type_ii_profile():
    """Return h(R) making t_II independent of R for constant Sigma."""
    migrator_msun = 100.0
    aspect = 0.01 * (RADIUS_CM / RADIUS_REFERENCE_CM) ** 0.75
    return migrator_msun, aspect


def _local_times_myr(migrator_msun: float, aspect):
    smbh_g = SMBH_MSUN * const.msun
    migrator_g = migrator_msun * const.msun
    omega = np.sqrt(const.G * smbh_g / RADIUS_CM**3)
    type_i = (
        smbh_g**2
        * aspect**2
        / (
            2.0
            * migrator_g
            * SURFACE_DENSITY_G_CM2
            * RADIUS_CM**2
            * omega
        )
        / const.Myr
    )
    type_ii = (
        1.0 / (2.0 * ALPHA * aspect**2 * omega) / const.Myr
    )
    return type_i, type_ii


def test_gap_mask_uses_the_kii_threshold() -> None:
    mass_ratio = 1.0e-4
    aspect = np.logspace(-3.0, -1.5, 50)
    mask = gap_opening_mask(mass_ratio, aspect, ALPHA, "kii")
    expected = mass_ratio**2 / (ALPHA * aspect**5) > 11.0
    assert np.array_equal(mask, expected)
    assert mask.any() and not mask.all()


def test_type_i_integral_matches_constant_local_time() -> None:
    migrator_msun, aspect = _constant_type_i_profile()
    type_i, _ = _local_times_myr(migrator_msun, aspect)
    assert np.ptp(type_i) / np.mean(type_i) < 1.0e-12

    outcome = semi_analytic_migration_outcome(
        RADIUS_CM,
        np.full_like(RADIUS_CM, SURFACE_DENSITY_G_CM2),
        aspect,
        migrator_msun * const.msun,
        RADIUS_CM[-1],
        RADIUS_CM[0],
        SMBH_MSUN,
        ALPHA,
        "kii",
        True,
    )
    expected = type_i[0] * np.log(RADIUS_CM[-1] / RADIUS_CM[0])
    assert np.isclose(outcome.time_myr, expected, rtol=1.0e-11, atol=0.0)
    assert outcome.status == "finite"
    assert outcome.blocking_radius_cm is None


def test_type_ii_branch_matches_constant_local_time() -> None:
    migrator_msun, aspect = _constant_type_ii_profile()
    mass_ratio = migrator_msun / SMBH_MSUN
    assert gap_opening_mask(mass_ratio, aspect, ALPHA, "kii").all()
    _, type_ii = _local_times_myr(migrator_msun, aspect)
    assert np.ptp(type_ii) / np.mean(type_ii) < 1.0e-12

    outcome = semi_analytic_migration_outcome(
        RADIUS_CM,
        np.full_like(RADIUS_CM, SURFACE_DENSITY_G_CM2),
        aspect,
        migrator_msun * const.msun,
        RADIUS_CM[-1],
        RADIUS_CM[0],
        SMBH_MSUN,
        ALPHA,
        "kii",
        True,
    )
    expected = type_ii[0] * np.log(RADIUS_CM[-1] / RADIUS_CM[0])
    assert np.isclose(outcome.time_myr, expected, rtol=1.0e-11, atol=0.0)
    assert outcome.blocking_radius_cm is None


def test_short_path_takes_less_time_than_full_path() -> None:
    migrator_msun, aspect = _constant_type_i_profile()
    common = (
        RADIUS_CM,
        np.full_like(RADIUS_CM, SURFACE_DENSITY_G_CM2),
        aspect,
        migrator_msun * const.msun,
        SMBH_MSUN,
        ALPHA,
        "kii",
        True,
    )
    full = semi_analytic_migration_outcome(
        *common[:4], RADIUS_CM[-1], RADIUS_CM[0], *common[4:]
    )
    short = semi_analytic_migration_outcome(
        *common[:4], RADIUS_CM[200], RADIUS_CM[0], *common[4:]
    )
    assert 0.0 < short.time_myr < full.time_myr


def test_invalid_profiles_raise_instead_of_returning_a_trap() -> None:
    migrator_msun, aspect = _constant_type_i_profile()
    bad_surface_density = np.full_like(RADIUS_CM, SURFACE_DENSITY_G_CM2)
    bad_surface_density[10] = 0.0
    for surface_density in (bad_surface_density, -bad_surface_density):
        try:
            semi_analytic_migration_outcome(
                RADIUS_CM,
                surface_density,
                aspect,
                migrator_msun * const.msun,
                RADIUS_CM[-1],
                RADIUS_CM[0],
                SMBH_MSUN,
                ALPHA,
                "kii",
                True,
            )
        except SemiAnalyticMigrationError:
            pass
        else:
            raise AssertionError("invalid surface density was accepted")

    try:
        semi_analytic_migration_outcome(
            RADIUS_CM,
            np.full_like(RADIUS_CM, SURFACE_DENSITY_G_CM2),
            aspect,
            migrator_msun * const.msun,
            2.0 * RADIUS_CM[-1],
            RADIUS_CM[0],
            SMBH_MSUN,
            ALPHA,
            "kii",
            True,
        )
    except SemiAnalyticMigrationError:
        pass
    else:
        raise AssertionError("out-of-domain path was accepted")
