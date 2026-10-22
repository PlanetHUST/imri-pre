"""Historical routing tests for the superseded gas / GW handoff.

The ordering under test comes from ``docs/bug_fix_plan.md`` Task G: a gas
torque barrier blocks *gas delivery*, not GW evolution, so a finite direct-GW
route must be allowed to win before the migrator is declared trap-bound.
The production workflow now uses a simultaneous circular gas+GW orbit; these
tests protect the retained standalone diagnostic only.
"""

import numpy as np

from workflows.event_rate import solve_gas_final_radius_cm

R_INI = 1.0e16
R_TARGET = 1.0e14
R_ENTRY = 1.0e13


def _solve(
    migration_time_myr: float,
    gw_time_myr: float,
    available_gas_time_myr: float = 10.0,
    handoff_required: bool = True,
):
    """Run the solver with constant migration and GW timescales."""

    def migration_time(_outer_cm, _inner_cm):
        return migration_time_myr

    def gw_time(radius_cm, _time_myr):
        # Scale with radius so the sampled crossover has something to bite on,
        # anchored so that gw_time(R_INI) is the requested value.
        if not np.isfinite(gw_time_myr):
            return np.inf
        return gw_time_myr * (radius_cm / R_INI) ** 4

    return solve_gas_final_radius_cm(
        r_ini_cm=R_INI,
        r_gas_target_cm=R_TARGET,
        r_entry_cm=R_ENTRY,
        migration_time_myr_call=migration_time,
        gw_time_to_entry_myr=gw_time,
        available_gas_time_myr=available_gas_time_myr,
        gas_start_time_myr=0.0,
        handoff_required=handoff_required,
        num_radius_samples=96,
    )


def _stop_reason(*args, **kwargs) -> str:
    return _solve(*args, **kwargs)[-1]


def test_finite_gas_that_is_faster_is_not_preempted_by_gw() -> None:
    """Gas wins: the solver must not return gw_dominates_at_start."""
    reason = _stop_reason(migration_time_myr=0.01, gw_time_myr=1.0e6)
    assert reason != "gw_dominates_at_start", reason
    assert reason in {"gas_to_gw_handoff", "gw_dominates_after_gas", "duty_cycle_limited"}


def test_finite_gas_that_is_slower_yields_gw_dominates_at_start() -> None:
    reason = _stop_reason(migration_time_myr=1.0e6, gw_time_myr=1.0e-3)
    assert reason == "gw_dominates_at_start", reason


def test_infinite_gas_with_a_finite_gw_route_is_not_trap_bound() -> None:
    """The regression this task exists for.

    A trap blocks gas delivery only.  With a finite direct-GW route the object
    still reaches the band, so it must route as gw_dominates_at_start rather
    than terminating as migration_trap_bound.
    """
    reason = _stop_reason(migration_time_myr=np.inf, gw_time_myr=100.0)
    assert reason == "gw_dominates_at_start", (
        f"a trap-bound gas path must not preempt a finite GW route; got {reason}"
    )


def test_infinite_gas_and_infinite_gw_is_trap_bound() -> None:
    reason = _stop_reason(migration_time_myr=np.inf, gw_time_myr=np.inf)
    assert reason == "migration_trap_bound", reason


def test_duty_cycle_limited_path_is_preserved() -> None:
    """Gas is finite but slower than the remaining AGN time allows."""
    reason = _stop_reason(
        migration_time_myr=100.0, gw_time_myr=np.inf, available_gas_time_myr=1.0
    )
    assert reason == "duty_cycle_limited", reason


def test_no_available_gas_time_still_reports_the_damping_overrun() -> None:
    reason = _stop_reason(
        migration_time_myr=1.0, gw_time_myr=np.inf, available_gas_time_myr=0.0
    )
    assert reason == "damping_misses_duty_cycle", reason


def test_trap_bound_still_reports_the_infinite_migration_time() -> None:
    """Status changes must not hide the underlying timescales."""
    _, _, migration_myr, gw_myr, _, reason = _solve(
        migration_time_myr=np.inf, gw_time_myr=100.0
    )
    assert reason == "gw_dominates_at_start"
    assert not np.isfinite(migration_myr), (
        "the blocked gas time must still be reported as infinite"
    )
    assert np.isfinite(gw_myr)
