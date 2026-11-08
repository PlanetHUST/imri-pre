"""Correctness tests for the fixed continuous gap contract."""

from __future__ import annotations

import numpy as np

from physical_process.migration import gap as gap_module
from physical_process.migration import gap_diagnostics

ALPHA = 0.1
ASPECT = np.array([0.005, 0.01, 0.02, 0.05, 0.1])
MASS_RATIO = 1.0e-4


def test_kanagawa_parameter_matches_closed_form() -> None:
    values = gap_module.kanagawa_gap_parameter(MASS_RATIO, ASPECT, ALPHA)
    expected = MASS_RATIO**2 / (ALPHA * ASPECT**5)
    assert np.allclose(values, expected, rtol=0.0, atol=0.0)


def test_gap_depth_matches_kanagawa_law() -> None:
    parameter = np.array([0.0, 1.0, 25.0, 100.0])
    depth = gap_module.kanagawa_gap_depth_fraction(parameter)
    assert np.allclose(depth, 1.0 / (1.0 + 0.04 * parameter))
    assert depth[0] == 1.0
    assert depth[2] == 0.5


def test_crida_parameter_matches_pressure_plus_viscosity_form() -> None:
    hill = gap_module.hill_radius_over_scale_height(MASS_RATIO, ASPECT)
    expected = 0.75 / hill + 50.0 * ALPHA * ASPECT**2 / MASS_RATIO
    actual = gap_module.crida_gap_parameter(MASS_RATIO, ASPECT, ALPHA)
    assert np.allclose(actual, expected, rtol=0.0, atol=0.0)


def test_crida_is_diagnostic_not_a_torque_selector() -> None:
    diagnostics = gap_diagnostics(1.0e-3, np.logspace(-4, -1, 40), ALPHA)
    assert np.array_equal(
        diagnostics.deep_gap,
        diagnostics.crida_parameter <= gap_module.CRIDA_DEEP_GAP_PARAMETER,
    )
    for removed_name in (
        "GapCriterion",
        "GAP_OPENING_CONDITIONS",
        "gap_opening_mask",
        "kii_gap_parameter",
    ):
        assert not hasattr(gap_module, removed_name), removed_name


def test_deep_gap_is_monotone_with_disk_thickness() -> None:
    aspect = np.logspace(-4, -1, 200)
    deep_gap = gap_diagnostics(1.0e-3, aspect, ALPHA).deep_gap
    if deep_gap.any() and not deep_gap.all():
        first_closed = int(np.argmin(deep_gap))
        assert not deep_gap[first_closed:].any()


def test_kanagawa_extrapolation_is_explicit() -> None:
    diagnostics = gap_diagnostics(
        1.0e-3,
        np.array([0.003, 0.01, 0.1]),
        ALPHA,
    )
    assert np.array_equal(
        diagnostics.kanagawa_extrapolated,
        diagnostics.kanagawa_parameter
        > gap_module.KANAGAWA_CALIBRATED_MAX_PARAMETER,
    )
    assert diagnostics.kanagawa_extrapolated.any()
    assert not diagnostics.kanagawa_extrapolated.all()


def test_invalid_aspect_ratio_is_reported_not_coerced() -> None:
    diagnostics = gap_diagnostics(
        1.0e-3,
        np.array([np.nan, 0.0, -0.01, np.inf]),
        ALPHA,
    )
    assert np.isnan(diagnostics.kanagawa_parameter).all()
    assert np.isnan(diagnostics.gap_depth_fraction).all()
    assert not diagnostics.deep_gap.any()
    assert not diagnostics.kanagawa_extrapolated.any()


def test_non_positive_mass_ratio_and_alpha_are_rejected() -> None:
    for bad_mass_ratio in (0.0, -1.0, np.nan):
        try:
            gap_diagnostics(bad_mass_ratio, ASPECT, ALPHA)
        except ValueError:
            pass
        else:
            raise AssertionError(f"mass ratio {bad_mass_ratio} was accepted")
    for bad_alpha in (0.0, -0.1, np.nan):
        try:
            gap_diagnostics(MASS_RATIO, ASPECT, bad_alpha)
        except ValueError:
            pass
        else:
            raise AssertionError(f"alpha {bad_alpha} was accepted")


def test_gap_depth_rejects_invalid_parameter() -> None:
    for values in (np.array([-1.0]), np.array([np.nan]), np.array([np.inf])):
        try:
            gap_module.kanagawa_gap_depth_fraction(values)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid Kanagawa parameter was accepted: {values}")


def test_workflow_cli_exposes_no_torque_model_selector() -> None:
    from workflows.cli import build_parser

    parser = build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    for removed_option in (
        "--migration-model",
        "--gap-open-condition",
        "--gap-threshold",
        "--no-type-ii-migration",
        "--disk-model",
    ):
        assert removed_option not in option_strings
