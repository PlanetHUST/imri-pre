"""Correctness tests for trap identification (physical_process.migration.traps)."""

from __future__ import annotations

import numpy as np

from physical_process.migration import (
    TorqueTrapRadii,
    classify_torque_traps,
    stable_blocking_radius_cm,
)


def _linear_zero_profile(root: float, slope: float, radii: np.ndarray) -> np.ndarray:
    """Torque crossing zero at ``root`` with the given slope."""
    return slope * (radii - root)


def test_stable_trap_is_found_at_the_sign_change() -> None:
    """Positive inside, negative outside => stable trap at the root."""
    radii = np.linspace(10.0, 200.0, 400)
    torque = _linear_zero_profile(100.0, -1.0, radii)
    traps = classify_torque_traps(radii, torque, min_separation_cells=0.0)
    assert traps.stable_radius_rg.size == 1
    assert np.isclose(traps.stable_radius_rg[0], 100.0, rtol=1e-6)
    assert traps.anti_trap_radius_rg.size == 0


def test_anti_trap_is_found_and_distinguished() -> None:
    """Negative inside, positive outside => anti-trap, not a trap."""
    radii = np.linspace(10.0, 200.0, 400)
    torque = _linear_zero_profile(100.0, +1.0, radii)
    traps = classify_torque_traps(radii, torque, min_separation_cells=0.0)
    assert traps.anti_trap_radius_rg.size == 1
    assert np.isclose(traps.anti_trap_radius_rg[0], 100.0, rtol=1e-6)
    assert traps.stable_radius_rg.size == 0


def test_traps_are_derived_not_hardcoded() -> None:
    """Moving the torque zero moves the reported trap by the same amount."""
    radii = np.linspace(10.0, 500.0, 900)
    for root in (50.0, 137.0, 300.0):
        traps = classify_torque_traps(
            radii, _linear_zero_profile(root, -1.0, radii), min_separation_cells=0.0
        )
        assert np.isclose(traps.stable_radius_rg[0], root, rtol=1e-6)


def test_unresolved_wiggle_is_not_reported_as_a_trap() -> None:
    """An opposite-sign pair within a couple of cells is a numerical artefact."""
    radii = np.arange(100.0, 200.0, 1.0)
    torque = -np.ones_like(radii)
    torque[40] = +1.0  # one-cell spike: produces an anti-trap/trap pair
    lenient = classify_torque_traps(radii, torque, min_separation_cells=0.0)
    strict = classify_torque_traps(radii, torque, min_separation_cells=10.0)
    assert lenient.stable_radius_rg.size + lenient.anti_trap_radius_rg.size == 2
    assert strict.stable_radius_rg.size == 0
    assert strict.anti_trap_radius_rg.size == 0
    assert strict.unresolved_radius_rg.size == 2


def test_result_is_independent_of_input_ordering() -> None:
    radii = np.linspace(10.0, 200.0, 301)
    torque = _linear_zero_profile(100.0, -1.0, radii)
    order = np.argsort(np.sin(radii))  # arbitrary shuffle
    shuffled = classify_torque_traps(
        radii[order], torque[order], min_separation_cells=0.0
    )
    straight = classify_torque_traps(radii, torque, min_separation_cells=0.0)
    assert np.allclose(shuffled.stable_radius_rg, straight.stable_radius_rg)


def test_no_sign_change_gives_no_traps() -> None:
    radii = np.linspace(10.0, 200.0, 100)
    traps = classify_torque_traps(radii, -np.ones_like(radii))
    assert isinstance(traps, TorqueTrapRadii)
    assert traps.stable_radius_rg.size == 0
    assert traps.anti_trap_radius_rg.size == 0
    assert np.isnan(traps.innermost_stable_radius_rg)


def test_nonfinite_samples_are_ignored() -> None:
    radii = np.linspace(10.0, 200.0, 400)
    torque = _linear_zero_profile(100.0, -1.0, radii)
    torque[5] = np.nan
    torque[-5] = np.inf
    traps = classify_torque_traps(radii, torque, min_separation_cells=0.0)
    assert traps.stable_radius_rg.size == 1
    assert np.isclose(traps.stable_radius_rg[0], 100.0, rtol=1e-6)


def test_blocking_radius_is_the_outermost_stable_trap() -> None:
    """The integrator must be blocked by the outermost stable trap."""
    radii = np.linspace(10.0, 500.0, 1000)
    # Two stable traps: sin gives +,-,+,- pattern; use an explicit product.
    torque = -(radii - 100.0) * (radii - 300.0) * (radii - 400.0)
    blocking = stable_blocking_radius_cm(radii, torque)
    traps = classify_torque_traps(radii, torque, min_separation_cells=0.0)
    assert blocking is not None
    assert np.isclose(blocking, traps.outermost_stable_radius_rg, rtol=1e-6)


def test_blocking_radius_is_none_without_a_stable_trap() -> None:
    radii = np.linspace(10.0, 200.0, 100)
    assert stable_blocking_radius_cm(radii, -np.ones_like(radii)) is None
