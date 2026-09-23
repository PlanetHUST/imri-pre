"""Focused tests for the fixed JM17 plus Grishin-2024 torque contract."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from physical_process.migration import (
    gap_diagnostics,
    inward_only_counterfactual_torque,
)
from physical_process.migration.torque import (
    MigrationParameters,
    comp_gamma_th_coeff,
    compose_migration_torque_si,
    compute_thermal_torque_terms,
    thermal_diffusivity_m2_s,
)

CELLS = 8
MIGRATOR_MASS_KG = 2.0e31
MASS_RATIO = 1.0e-4


class _Disk:
    """Minimal SI disk stub read by the thermal-torque equations."""

    def __init__(self, optical_depth: float = 1.0e4, opacity: float = 0.1) -> None:
        ones = np.ones(CELLS)
        self.T = ones * 1.0e4
        self.rho = ones * 1.0e-6
        self.h = ones * 1.0e9
        self.R = ones * 1.0e11
        self.Omega = ones * 1.0e-7
        self.cs = ones * 1.0e3
        self.tauV = ones * optical_depth
        self.kappa = ones * opacity


def _gap(disk: _Disk, alpha: float = 0.1):
    return gap_diagnostics(MASS_RATIO, disk.h / disk.R, alpha)


def test_production_torque_has_one_composition() -> None:
    gravitational = np.full(CELLS, 3.0)
    thermal = np.full(CELLS, 5.0)
    depth = np.full(CELLS, 0.25)
    secondary = np.full(CELLS, -2.0)
    total = compose_migration_torque_si(gravitational, thermal, depth, secondary)
    # 0.25*3 + 5 + 0.75*(-2)
    assert np.allclose(total, 4.25)


def test_shallow_and_deep_gap_limits_are_continuous() -> None:
    gravitational = np.full(CELLS, 3.0)
    thermal = np.full(CELLS, 5.0)
    secondary = np.full(CELLS, -2.0)
    shallow = compose_migration_torque_si(
        gravitational, thermal, np.ones(CELLS), secondary
    )
    deep = compose_migration_torque_si(
        gravitational, thermal, np.zeros(CELLS), secondary
    )
    # No gap: pure linear torques, the viscous branch carries no weight.
    assert np.allclose(shallow, 8.0)
    # Deep gap: the linear gravitational torque is gone and its weight has
    # been handed to the gap-limited viscous torque, not to zero.
    assert np.allclose(deep, thermal + secondary)


def test_a_deep_gap_does_not_stall_migration() -> None:
    """A gap stops linear torques; it must not stop viscous transport.

    Regression test for the previous composition, in which ``f_gap -> 0`` drove
    the gravitational contribution to zero with nothing taking its place, so a
    deeply gapped migrator simply stopped moving.
    """
    gravitational = np.full(CELLS, -3.0)
    thermal = np.zeros(CELLS)
    secondary = np.full(CELLS, -2.0)
    deep = compose_migration_torque_si(
        gravitational, thermal, np.zeros(CELLS), secondary
    )
    assert np.all(deep < 0.0), "a deeply gapped migrator must still migrate"
    assert np.allclose(deep, -2.0)


def test_secondary_dominated_torque_is_inertia_limited() -> None:
    """The rate saturates at the viscous value and falls off as B below it."""
    from physical_process.migration.torque import (
        comp_secondary_dominated_torque_si,
    )

    alpha, radius_m, aspect, omega = 0.1, 1.0e12, 0.01, 1.0e-8
    sigma = 1.0e3
    viscous = -1.5 * alpha * aspect**2 * sigma * radius_m**4 * omega**2

    # B >> 1: disk-dominated, classical viscous rate.
    heavy_disk = comp_secondary_dominated_torque_si(
        alpha, sigma, radius_m, aspect, omega, 1.0e20
    )
    assert np.isclose(heavy_disk, viscous, rtol=1.0e-12)

    # B << 1: migrator-dominated, slowed by exactly B.
    light_disk_mass_kg = 1.0e30
    disk_dominance = sigma * radius_m**2 / light_disk_mass_kg
    assert disk_dominance < 1.0
    light_disk = comp_secondary_dominated_torque_si(
        alpha, sigma, radius_m, aspect, omega, light_disk_mass_kg
    )
    assert np.isclose(light_disk, viscous * disk_dominance, rtol=1.0e-12)

    # Always inward: a viscously accreting disk drags the secondary in.
    assert light_disk < 0.0 and heavy_disk < 0.0


def test_no_classical_type_ii_model_is_exposed() -> None:
    from physical_process.migration import torque as torque_module

    for removed_name in (
        "MIGRATION_MODELS",
        "MIGRATION_MODEL_CLASSICAL_TYPE_II",
        "MIGRATION_MODEL_UNGAPPED",
        "comp_gamma_type_ii_si",
    ):
        assert not hasattr(torque_module, removed_name), removed_name


def test_no_trap_extreme_preserves_amplitude_and_forces_inward_sign() -> None:
    fixed_torque = np.array([-4.0, 2.0, -0.5, 8.0])
    counterfactual = inward_only_counterfactual_torque(fixed_torque)
    assert np.array_equal(np.abs(counterfactual), np.abs(fixed_torque))
    assert np.all(counterfactual < 0.0)


def test_no_trap_extreme_rejects_exact_zeros_instead_of_inventing_a_floor() -> None:
    with np.testing.assert_raises(ValueError):
        inward_only_counterfactual_torque(np.array([-1.0, 0.0, 1.0]))


def test_thermal_diffusivity_uses_local_disk_opacity() -> None:
    params = MigrationParameters(alpha=0.1)
    low_opacity = thermal_diffusivity_m2_s(_Disk(opacity=0.01), params.gamma_adiabat)
    high_opacity = thermal_diffusivity_m2_s(_Disk(opacity=0.1), params.gamma_adiabat)
    assert np.allclose(low_opacity / high_opacity, 10.0)


def test_corotation_offset_follows_pressure_gradient_sign() -> None:
    params = MigrationParameters(alpha=0.1)
    disk = _Disk()
    gap = _gap(disk, params.alpha)
    falling = comp_gamma_th_coeff(
        np.full(CELLS, -2.0),
        disk,
        MIGRATOR_MASS_KG,
        MASS_RATIO,
        gap,
        params,
    )
    rising = comp_gamma_th_coeff(
        np.full(CELLS, 2.0),
        disk,
        MIGRATOR_MASS_KG,
        MASS_RATIO,
        gap,
        params,
    )
    assert np.all(np.sign(falling) == -np.sign(rising))
    assert np.all(falling != 0.0)


def test_optically_thin_gas_switches_thermal_torque_off() -> None:
    params = MigrationParameters(alpha=0.1)
    thin_disk = _Disk(optical_depth=0.0)
    thick_disk = _Disk(optical_depth=1.0e12)
    slope = np.full(CELLS, -2.0)
    thin = comp_gamma_th_coeff(
        slope,
        thin_disk,
        MIGRATOR_MASS_KG,
        MASS_RATIO,
        _gap(thin_disk, params.alpha),
        params,
    )
    thick = comp_gamma_th_coeff(
        slope,
        thick_disk,
        MIGRATOR_MASS_KG,
        MASS_RATIO,
        _gap(thick_disk, params.alpha),
        params,
    )
    assert np.allclose(thin, 0.0)
    assert np.all(np.abs(thick) > 0.0)


def test_gap_depletion_reduces_thermal_optical_depth() -> None:
    params = MigrationParameters(alpha=0.1)
    disk = _Disk(optical_depth=100.0)
    base_gap = _gap(disk, params.alpha)
    shallow_gap = replace(base_gap, gap_depth_fraction=np.ones(CELLS))
    deep_gap = replace(base_gap, gap_depth_fraction=np.full(CELLS, 0.01))
    shallow = compute_thermal_torque_terms(
        np.full(CELLS, -2.0),
        disk,
        MIGRATOR_MASS_KG,
        MASS_RATIO,
        shallow_gap,
        params,
    )
    deep = compute_thermal_torque_terms(
        np.full(CELLS, -2.0),
        disk,
        MIGRATOR_MASS_KG,
        MASS_RATIO,
        deep_gap,
        params,
    )
    assert np.allclose(deep.effective_optical_depth, 0.01 * disk.tauV)
    assert np.all(np.abs(deep.coefficient) < np.abs(shallow.coefficient))


def test_bondi_hoyle_saturation_uses_minimum_of_bondi_radius_and_height() -> None:
    import pagn.constants as ct

    params = MigrationParameters(alpha=0.1)
    disk = _Disk()
    terms = compute_thermal_torque_terms(
        np.full(CELLS, -2.0),
        disk,
        MIGRATOR_MASS_KG,
        MASS_RATIO,
        _gap(disk, params.alpha),
        params,
    )
    diffusivity = thermal_diffusivity_m2_s(disk, params.gamma_adiabat)
    bondi_radius = ct.G * MIGRATOR_MASS_KG / disk.cs**2
    expected = diffusivity / (disk.cs * np.minimum(bondi_radius, disk.h))
    assert np.allclose(terms.thermal_mass_ratio, expected)


def test_linear_regime_failure_is_reported_not_hidden() -> None:
    params = MigrationParameters(alpha=0.1)
    disk = _Disk()
    terms = compute_thermal_torque_terms(
        np.full(CELLS, -2.0),
        disk,
        MIGRATOR_MASS_KG,
        1.0e-2,
        gap_diagnostics(1.0e-2, disk.h / disk.R, params.alpha),
        params,
    )
    assert not terms.thermal_torque_regime_valid.any()
    assert np.isfinite(terms.coefficient).all()


def test_composition_rejects_invalid_gap_depth() -> None:
    try:
        compose_migration_torque_si(
            np.ones(CELLS),
            np.ones(CELLS),
            np.full(CELLS, 1.1),
            np.ones(CELLS),
        )
    except ValueError:
        return
    raise AssertionError("a gap depth above unity was accepted")


def test_composition_rejects_mismatched_shapes() -> None:
    try:
        compose_migration_torque_si(
            np.ones(CELLS),
            np.ones(CELLS),
            np.full(CELLS, 0.5),
            np.ones(CELLS + 1),
        )
    except ValueError:
        return
    raise AssertionError("a mismatched secondary-dominated array was accepted")


def _synthetic_disk(cells: int = 64):
    """A pAGN-shaped stub spanning a wide radial range."""

    class _Disk:
        pass

    disk = _Disk()
    disk.R = np.logspace(11.0, 14.0, cells)
    disk.h = 0.01 * disk.R
    disk.rho = np.full(cells, 1.0e-6)
    disk.T = np.full(cells, 3.0e3)
    disk.cs = np.full(cells, 3.0e3)
    disk.Omega = np.sqrt(6.674e-11 * 2.0e35 / disk.R**3)
    disk.tauV = np.full(cells, 1.0e4)
    disk.kappa = np.full(cells, 0.4)
    disk.Teff4 = np.full(cells, 1.0e12)
    disk.Mbh = 2.0e35
    disk.Rs = 2.0 * 6.674e-11 * 2.0e35 / (2.998e8) ** 2
    return disk


def test_thermal_torque_is_dropped_outside_its_validity_regime() -> None:
    """Linear theory must not be extrapolated into the non-linear regime.

    The Grishin-2024 thermal torque assumes ``R_H < H`` among other linear
    conditions. Where those fail the torque is not evaluated rather than
    extrapolated, and the weight records where that happened.
    """
    from physical_process.migration.torque import compute_torque_profile

    disk = _synthetic_disk()
    bounded = compute_torque_profile(
        disk, 10.0, None, params=MigrationParameters(alpha=0.1)
    )
    invalid = ~bounded.thermal_torque_regime_valid
    if not invalid.any():
        raise AssertionError("synthetic disk must exercise the invalid regime")

    assert np.allclose(bounded.gamma_th_si[invalid], 0.0), (
        "the thermal torque must be zero where its derivation does not hold"
    )
    assert np.allclose(bounded.thermal_validity_weight[invalid], 0.0)
    assert np.allclose(
        bounded.thermal_validity_weight[~invalid], 1.0
    ), "the thermal torque must be untouched where the regime is valid"


def test_extrapolation_is_opt_in_and_reproduces_the_old_behaviour() -> None:
    from physical_process.migration.torque import compute_torque_profile

    disk = _synthetic_disk()
    bounded = compute_torque_profile(
        disk, 10.0, None, params=MigrationParameters(alpha=0.1)
    )
    extrapolated = compute_torque_profile(
        disk,
        10.0,
        None,
        params=MigrationParameters(alpha=0.1, extrapolate_thermal_torque=True),
    )
    assert np.allclose(extrapolated.thermal_validity_weight, 1.0)
    invalid = ~bounded.thermal_torque_regime_valid
    # Opting in restores a non-zero torque exactly where bounding removed it.
    assert np.any(np.abs(extrapolated.gamma_th_si[invalid]) > 0.0)
    valid = ~invalid
    assert np.allclose(
        bounded.gamma_th_si[valid], extrapolated.gamma_th_si[valid]
    ), "bounding must not alter the torque inside the valid regime"
