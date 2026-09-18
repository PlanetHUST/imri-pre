"""Correctness tests for the canonical gap criteria (physical_process.migration.gap)."""

from __future__ import annotations

import numpy as np

from physical_process.migration import gap as gap_module
from physical_process.migration import (
    DEFAULT_GAP_CRITERION,
    GAP_OPENING_CONDITIONS,
    GapCriterion,
    gap_diagnostics,
    gap_opening_mask,
)

ALPHA = 0.1
ASPECT = np.array([0.005, 0.01, 0.02, 0.05, 0.1])
MASS_RATIO = 1.0e-4


def test_kii_parameter_matches_closed_form() -> None:
    """K_II must equal q^2 / (alpha h^2) exactly."""
    values = gap_module.kii_gap_parameter(MASS_RATIO, ASPECT, ALPHA)
    expected = MASS_RATIO**2 / (ALPHA * ASPECT**2)
    assert np.allclose(values, expected, rtol=0.0, atol=0.0)


def test_kanagawa_parameter_matches_closed_form() -> None:
    """The Kanagawa-form K must equal q^2 / (alpha h^5) exactly."""
    values = gap_module.kanagawa_gap_parameter(MASS_RATIO, ASPECT, ALPHA)
    expected = MASS_RATIO**2 / (ALPHA * ASPECT**5)
    assert np.allclose(values, expected, rtol=0.0, atol=0.0)


def test_the_two_gap_parameters_differ_by_h_cubed() -> None:
    """Documented ambiguity: the two K definitions differ by exactly h^-3."""
    kii = gap_module.kii_gap_parameter(MASS_RATIO, ASPECT, ALPHA)
    kanagawa = gap_module.kanagawa_gap_parameter(MASS_RATIO, ASPECT, ALPHA)
    assert np.allclose(kanagawa / kii, ASPECT**-3.0)


def test_torque_module_delegates_to_canonical_gap_parameter() -> None:
    """The torque model must not carry its own copy of the gap parameter."""
    from physical_process.migration import torque as torque_module

    delegated = torque_module.comp_gap_parameter_dimensionless(
        MASS_RATIO, ASPECT, ALPHA
    )
    canonical = gap_module.kanagawa_gap_parameter(MASS_RATIO, ASPECT, ALPHA)
    assert np.allclose(delegated, canonical, rtol=0.0, atol=0.0)


def test_the_production_default_is_kanagawa() -> None:
    """The production criterion is a deliberate, single-sourced decision."""
    from workflows.config import DEFAULT_GAP_OPENING_CONDITION

    assert DEFAULT_GAP_CRITERION == "kanagawa"
    assert GapCriterion().name == DEFAULT_GAP_CRITERION
    # The workflow default must never be hard-coded independently of gap.py.
    assert DEFAULT_GAP_OPENING_CONDITION == DEFAULT_GAP_CRITERION


def test_default_criterion_uses_the_torque_models_own_gap_parameter() -> None:
    """The Type-II switch and the Type-I attenuation share one quantity.

    This is the point of the change: before it, the switch used q^2/(alpha h^2)
    while the attenuation used q^2/(alpha h^5).
    """
    from physical_process.migration import torque as torque_module

    diagnostics = gap_diagnostics(MASS_RATIO, ASPECT, ALPHA)
    assert np.allclose(
        diagnostics.parameter_for(DEFAULT_GAP_CRITERION),
        torque_module.comp_gap_parameter_dimensionless(MASS_RATIO, ASPECT, ALPHA),
        rtol=0.0,
        atol=0.0,
    )


def test_historical_kii_criterion_is_still_reachable() -> None:
    """Pre-change results must stay reproducible via an explicit selection."""
    aspect = np.array([1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1])
    mass_ratio = 1.0e-3
    mask = gap_opening_mask(mass_ratio, aspect, ALPHA, "kii")
    expected = (mass_ratio**2 / (ALPHA * aspect**2)) > 11.0
    assert np.array_equal(mask, expected)
    assert "kii" in GAP_OPENING_CONDITIONS


def test_crida_mask_requires_both_conditions() -> None:
    """Crida needs P < 1 AND R_H/H > 1; neither alone is sufficient."""
    aspect = np.logspace(-4, -1, 40)
    mass_ratio = 1.0e-3
    diagnostics = gap_diagnostics(mass_ratio, aspect, ALPHA, "crida")
    expected = (diagnostics.crida_parameter < 1.0) & (
        diagnostics.hill_radius_over_scale_height > 1.0
    )
    assert np.array_equal(diagnostics.gap_open, expected)
    assert np.any(diagnostics.crida_parameter < 1.0)
    assert np.any(diagnostics.hill_radius_over_scale_height <= 1.0)


def test_gap_open_is_monotone_in_aspect_ratio() -> None:
    """Thinner disks open gaps more easily, for every criterion."""
    aspect = np.logspace(-4, -1, 200)
    for name in GAP_OPENING_CONDITIONS:
        mask = gap_opening_mask(1.0e-3, aspect, ALPHA, name)
        if not mask.any() or mask.all():
            continue
        # Gap-open cells must form a single prefix (thin side) of the array.
        first_closed = int(np.argmin(mask))
        assert not mask[first_closed:].any(), f"{name} is not monotone in h"


def test_invalid_aspect_ratio_is_never_gap_open() -> None:
    """NaN, zero and negative aspect ratios must never report a gap."""
    aspect = np.array([np.nan, 0.0, -0.01, np.inf])
    for name in GAP_OPENING_CONDITIONS:
        mask = gap_opening_mask(1.0, aspect, ALPHA, name)
        assert not mask.any(), f"{name} reported a gap for an invalid aspect ratio"


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


def test_unknown_criterion_is_rejected() -> None:
    try:
        GapCriterion(name="not_a_criterion")
    except ValueError:
        return
    raise AssertionError("an unknown gap criterion was accepted")


def test_thresholds_are_configurable() -> None:
    """A consequential threshold must be a parameter, not a magic number."""
    aspect = np.array([0.003])
    mass_ratio = 1.0e-3
    value = float(gap_module.kii_gap_parameter(mass_ratio, aspect, ALPHA)[0])
    strict = GapCriterion(name="kii", kii_threshold=value * 2.0)
    loose = GapCriterion(name="kii", kii_threshold=value / 2.0)
    assert not gap_diagnostics(mass_ratio, aspect, ALPHA, strict).gap_open[0]
    assert gap_diagnostics(mass_ratio, aspect, ALPHA, loose).gap_open[0]


def test_diagnostics_expose_every_parameter() -> None:
    diagnostics = gap_diagnostics(MASS_RATIO, ASPECT, ALPHA)
    for name in GAP_OPENING_CONDITIONS:
        values = diagnostics.parameter_for(name)
        assert values.shape == ASPECT.shape
    assert 0.0 <= diagnostics.gap_open_fraction <= 1.0


def test_default_criterion_opens_more_gap_than_the_historical_one() -> None:
    """Documented consequence of the change: kanagawa is far more permissive.

    For a thin disk the two differ by h^-3, so switching the default moves most
    of the disk from Type I to Type II.  This is asserted so that a future
    threshold change cannot quietly reverse the intended behaviour.
    """
    aspect = np.logspace(-3, -1.5, 200)
    mass_ratio = 1.0e-4
    default_open = gap_opening_mask(mass_ratio, aspect, ALPHA, DEFAULT_GAP_CRITERION)
    historical_open = gap_opening_mask(mass_ratio, aspect, ALPHA, "kii")
    assert default_open.sum() > historical_open.sum()
    assert historical_open.sum() == 0


# --------------------------------------------------------------------------
# Calibration of the production threshold (scripts/calibrate_gap_threshold.py)
# --------------------------------------------------------------------------
def test_type_i_denominator_is_the_kanagawa_gap_depth_law() -> None:
    """The identity the threshold calibration rests on.

    The project's Type-I gap denominator ``1 + K/kappa_norm`` must be the
    Kanagawa gap-depth denominator ``1 + 0.04 K``.  If ``kappa_norm`` is ever
    changed without changing the calibration, this fails.
    """
    from physical_process.migration.torque import MigrationParameters

    params = MigrationParameters()
    K = np.logspace(-2, 4, 400)
    denominator = 1.0 + K / params.kappa_norm
    depth_denominator = 1.0 + gap_module.KANAGAWA_GAP_DEPTH_COEFFICIENT * K
    assert np.max(np.abs(denominator / depth_denominator - 1.0)) < 1.0e-15


def test_threshold_and_depletion_fraction_are_inverses() -> None:
    for fraction in (0.9, 0.5, 0.25, 0.1, 0.01):
        threshold = gap_module.threshold_for_depletion(fraction)
        assert np.isclose(gap_module.depletion_fraction(threshold), fraction)


def test_the_calibrated_threshold_is_the_half_depth_point() -> None:
    """K = 25 must be exactly where the gap is half cleared."""
    assert np.isclose(
        gap_module.depletion_fraction(gap_module.KANAGAWA_GAP_PARAMETER_THRESHOLD),
        0.5,
    )
    assert np.isclose(gap_module.threshold_for_depletion(0.5), 25.0)


def test_depletion_helpers_reject_unphysical_input() -> None:
    for bad in (0.0, -1.0, np.nan):
        for function in (gap_module.depletion_fraction,
                         gap_module.threshold_for_depletion):
            try:
                function(bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{function.__name__} accepted {bad}")
    try:
        gap_module.threshold_for_depletion(1.5)
    except ValueError:
        return
    raise AssertionError("a depletion fraction above 1 was accepted")


def test_threshold_override_reaches_the_simulation_config() -> None:
    """Calibration must be reproducible from the CLI, not only in a notebook."""
    from workflows.cli import build_config_from_arguments, build_parser

    parser = build_parser()
    default = build_config_from_arguments(parser.parse_args(["1", "/tmp/x", "1"]))
    assert default.gap_parameter_threshold is None
    assert default.gap_criterion.kanagawa_threshold == 25.0

    overridden = build_config_from_arguments(
        parser.parse_args(["1", "/tmp/x", "1", "--gap-threshold", "225"])
    )
    assert overridden.gap_criterion.kanagawa_threshold == 225.0
    assert np.isclose(
        gap_module.depletion_fraction(overridden.gap_criterion.kanagawa_threshold), 0.1
    )

    # The override must survive the clone used to ship a config to a worker.
    from workflows.host_mapping import config_with_overrides

    cloned = config_with_overrides(overridden, redshift=8.0)
    assert cloned.gap_criterion.kanagawa_threshold == 225.0
