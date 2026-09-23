"""Correctness tests for trap identification (physical_process.migration.traps)."""

from __future__ import annotations

import numpy as np

from physical_process.migration import (
    TorqueTrapRadii,
    classify_torque_traps,
    stable_blocking_radius_cm,
    torque_migration_outcome,
)


def _linear_zero_profile(root: float, slope: float, radii: np.ndarray) -> np.ndarray:
    """Torque crossing zero at ``root`` with the given slope."""
    return slope * (radii - root)


def test_stable_trap_is_found_at_the_sign_change() -> None:
    """Positive inside, negative outside => stable trap at the root."""
    radii = np.linspace(10.0, 200.0, 400)
    torque = _linear_zero_profile(100.0, -1.0, radii)
    traps = classify_torque_traps(radii, torque)
    assert traps.stable_radius_rg.size == 1
    assert np.isclose(traps.stable_radius_rg[0], 100.0, rtol=1e-6)
    assert traps.anti_trap_radius_rg.size == 0


def test_anti_trap_is_found_and_distinguished() -> None:
    """Negative inside, positive outside => anti-trap, not a trap."""
    radii = np.linspace(10.0, 200.0, 400)
    torque = _linear_zero_profile(100.0, +1.0, radii)
    traps = classify_torque_traps(radii, torque)
    assert traps.anti_trap_radius_rg.size == 1
    assert np.isclose(traps.anti_trap_radius_rg[0], 100.0, rtol=1e-6)
    assert traps.stable_radius_rg.size == 0


def test_traps_are_derived_not_hardcoded() -> None:
    """Moving the torque zero moves the reported trap by the same amount."""
    radii = np.linspace(10.0, 500.0, 900)
    for root in (50.0, 137.0, 300.0):
        traps = classify_torque_traps(
            radii, _linear_zero_profile(root, -1.0, radii)
        )
        assert np.isclose(traps.stable_radius_rg[0], root, rtol=1e-6)


def test_unresolved_wiggle_is_not_reported_as_a_trap() -> None:
    """A one-cell opposite-sign pair is a numerical artefact, not a trap.

    Both members of the pair bound the same narrow spike, so both are rejected.
    """
    radii = np.arange(100.0, 200.0, 1.0)
    torque = -np.ones_like(radii)
    torque[40] = +1.0  # one-cell spike: produces an anti-trap/trap pair
    lenient = classify_torque_traps(radii, torque, min_basin_dex=0.0)
    strict = classify_torque_traps(radii, torque)
    assert lenient.stable_radius_rg.size + lenient.anti_trap_radius_rg.size == 2
    assert strict.stable_radius_rg.size == 0
    assert strict.anti_trap_radius_rg.size == 0
    assert strict.unresolved_radius_rg.size == 2


def test_a_narrow_basin_is_rejected_regardless_of_cell_count() -> None:
    """The basin test is resolution independent; the cell test is not.

    A one-cell spike on a *fine* grid passes the cell-count test simply because
    the grid is fine, so the cell count cannot be the primary filter.
    """
    radii = np.linspace(1.0e5, 1.01e5, 4000)  # very fine, spans ~0.004 dex
    torque = -np.ones_like(radii)
    torque[2000:2040] = +1.0  # 40 cells wide, but only ~4e-5 dex
    passes_cell_test = classify_torque_traps(
        radii, torque, min_basin_dex=0.0
    )
    assert passes_cell_test.stable_radius_rg.size == 1, (
        "a 40-cell feature should clear a 10-cell separation test"
    )
    assert float(passes_cell_test.basin_dex[0]) < 0.02, (
        "the basin must be narrower than the default threshold for this test "
        f"to discriminate; got {float(passes_cell_test.basin_dex[0]):.5f} dex"
    )

    rejected = classify_torque_traps(
        radii, torque, min_basin_dex=0.02
    )
    assert rejected.stable_radius_rg.size == 0, (
        "the basin test must reject it: 40 cells here is a negligible width "
        "in decades, and refining the grid would only add cells"
    )


def test_basin_width_is_reported_for_every_stable_trap() -> None:
    radii = np.logspace(5.0, 5.3, 400)
    torque = -np.ones_like(radii)
    torque[:120] = +1.0  # trap at the inner->outer sign change
    traps = classify_torque_traps(radii, torque)
    assert traps.stable_radius_rg.size == traps.basin_dex.size == 1
    # The basin is the width of the outward-torque region the trap bounds,
    # measured inward from the trap, not the outer catchment.
    expected = np.log10(traps.stable_radius_rg[0] / radii[0])
    assert np.isclose(traps.basin_dex[0], expected, rtol=1.0e-6)


def test_blocking_distinguishes_a_trap_from_being_expelled() -> None:
    """Outward torque with no resolved trap is not a trap-bound outcome."""
    from physical_process.migration.traps import inward_delivery_blocked

    radii = np.logspace(4.0, 6.0, 500)

    # A genuine trap: positive inside, negative outside.
    trapped = np.where(radii < 1.0e5, 1.0, -1.0)
    blocked, radius = inward_delivery_blocked(radii, trapped)
    assert blocked and radius is not None
    assert np.isclose(radius, 1.0e5, rtol=0.02)

    # Positive everywhere: the migrator is pushed out, not held.
    expelled = np.ones_like(radii)
    blocked, radius = inward_delivery_blocked(radii, expelled)
    assert blocked, "outward torque still blocks inward delivery"
    assert radius is None, "being expelled is not being trap-bound"

    # Inward everywhere: nothing blocks delivery.
    clear = -np.ones_like(radii)
    assert inward_delivery_blocked(radii, clear) == (False, None)


def test_evolution_status_distinguishes_trap_from_outward_blockage() -> None:
    radii_cm = np.logspace(10.0, 12.0, 500)
    trapped = np.where(radii_cm < 1.0e11, 1.0e40, -1.0e40)
    expelled = np.ones_like(radii_cm) * 1.0e40
    arguments = (1.0e34, 9.0e11, 2.0e10, 1.0e5)

    trap_outcome = torque_migration_outcome(radii_cm, trapped, *arguments)
    outward_outcome = torque_migration_outcome(radii_cm, expelled, *arguments)

    assert trap_outcome.status == "trap_bound"
    assert trap_outcome.blocking_radius_cm is not None
    assert outward_outcome.status == "outward_blocked"
    assert outward_outcome.blocking_radius_cm is None


def test_blocking_only_counts_traps_on_the_requested_path() -> None:
    """A trap outside the migration path must not block it."""
    from physical_process.migration.traps import inward_delivery_blocked

    radii = np.logspace(4.0, 6.0, 500)
    torque = np.where(radii < 5.0e5, -1.0, 1.0)  # anti-trap far outside
    torque[radii > 8.0e5] = -1.0  # trap at ~8e5, beyond any inner path

    blocked, _ = inward_delivery_blocked(
        radii, torque, inner_radius_rg=1.0e4, outer_radius_rg=1.0e5
    )
    assert not blocked, "a trap at 8e5 rg cannot block a path ending at 1e5 rg"


def test_result_is_independent_of_input_ordering() -> None:
    radii = np.linspace(10.0, 200.0, 301)
    torque = _linear_zero_profile(100.0, -1.0, radii)
    order = np.argsort(np.sin(radii))  # arbitrary shuffle
    shuffled = classify_torque_traps(
        radii[order], torque[order]
    )
    straight = classify_torque_traps(radii, torque)
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
    traps = classify_torque_traps(radii, torque)
    assert traps.stable_radius_rg.size == 1
    assert np.isclose(traps.stable_radius_rg[0], 100.0, rtol=1e-6)


def test_blocking_radius_is_the_outermost_stable_trap() -> None:
    """The integrator must be blocked by the outermost stable trap."""
    radii = np.linspace(10.0, 500.0, 1000)
    # Two stable traps: sin gives +,-,+,- pattern; use an explicit product.
    torque = -(radii - 100.0) * (radii - 300.0) * (radii - 400.0)
    blocking = stable_blocking_radius_cm(radii, torque)
    traps = classify_torque_traps(radii, torque)
    assert blocking is not None
    assert np.isclose(blocking, traps.outermost_stable_radius_rg, rtol=1e-6)


def test_blocking_radius_is_none_without_a_stable_trap() -> None:
    radii = np.linspace(10.0, 200.0, 100)
    assert stable_blocking_radius_cm(radii, -np.ones_like(radii)) is None
