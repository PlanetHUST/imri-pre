"""Regression checks for adaptive damping shared with cos_tde_revised."""

from types import SimpleNamespace

import numpy as np

from physical_process import constants as const
from physical_process.agn import AGNProperties
from physical_process.damp import DampingCalculator, DampingStatus, OrbitalState


def _calculator():
    calculator = DampingCalculator(
        AGNProperties.from_mass(3000.0, 100.0), False,
        radius_grid_cm=np.array([1.0e11, 2.0e12]),
        surface_density_g_cm2=np.ones(2),
        aspect_ratio=np.full(2, 0.02),
        disk_inner_radius_cm=1.0e11,
        disk_outer_radius_cm=2.0e12,
        density_g_cm3=np.full(2, 1.0e-12),
    )
    calculator.timescale_calc.compute_all_timescales = lambda *_: {
        "drag": SimpleNamespace(drag_time_s=(1.0, 1.0)),
        "crossing_time_s": (1.0, 1.0),
        "orbital_period_s": 1.0,
        "active_node_mask": (True, True),
    }
    return calculator


def test_steps_shrink_below_the_old_floor_and_reach_embedding() -> None:
    calculator = _calculator()
    calls = []

    def coefficients(*args):
        calls.append(args[0])
        return -1.0e16, 0.0, -1.0e5, 0.0, 0.0, 0.0

    calculator.compute_ode_coefficients = coefficients
    initial = OrbitalState(1.0e12, 0.005, 0.2, 0.5)
    history = []
    outcome = calculator.damping_simulation(
        1.0, initial, 0.0, "BH", max_iter=1000, history=history,
    )
    assert outcome.status is DampingStatus.EMBEDDED_BH
    assert calculator.embedding_criteria(outcome.final_state).eligible
    assert 0.0 < outcome.elapsed_time_s < 1.0e-5
    assert outcome.final_state.semi_major_axis_cm < initial.semi_major_axis_cm
    # Rejected trials have the same state; their derivatives should be reused.
    assert len(calls) == len(history) - 1
    assert np.all(np.diff([s.time_s for s in history]) > 0.0)


def test_iteration_exhaustion_is_a_numerical_failure() -> None:
    calculator = _calculator()
    calculator.compute_ode_coefficients = lambda *_: (0.0,) * 6
    outcome = calculator.damping_simulation(
        1.0, OrbitalState(1.0e12, 0.005, 0.2, 0.5),
        0.0, "BH", max_iter=1,
    )
    assert outcome.status is DampingStatus.CALCULATION_ERROR
    assert outcome.final_state.inclination_rad > 0.02


def test_nonfinite_derivative_cannot_produce_an_embedded_result() -> None:
    calculator = _calculator()
    calculator.compute_ode_coefficients = lambda *_: (np.nan, 0, 0, 0, 0, 0)
    outcome = calculator.damping_simulation(
        1.0, OrbitalState(1.0e12, 0.005, 0.2, 0.5), 0.0, "BH",
    )
    assert outcome.status is DampingStatus.CALCULATION_ERROR
    assert np.isfinite(outcome.final_state.semi_major_axis_cm)


def test_a_stalled_clock_is_a_numerical_failure() -> None:
    calculator = _calculator()
    calculator.compute_ode_coefficients = lambda *_: (0.0,) * 6
    initial = OrbitalState(1.0e12, 0.005, 0.2, 0.5, time_s=const.Myr)
    calculator.damping_time_step_s = lambda *_: 1.0e-20
    outcome = calculator.damping_simulation(1.0, initial, 0.0, "BH")
    assert outcome.status is DampingStatus.CALCULATION_ERROR
    assert outcome.final_state == initial


def test_agn_deadline_is_physical_incompletion() -> None:
    calculator = _calculator()
    calculator.compute_ode_coefficients = lambda *_: (0.0,) * 6
    initial = OrbitalState(1.0e12, 0.005, 0.2, 0.5, time_s=100.0 * const.Myr)
    outcome = calculator.damping_simulation(1.0, initial, 0.0, "BH")
    assert outcome.status is DampingStatus.DAMPING_INCOMPLETE
    assert outcome.elapsed_time_s == 0.0


def test_already_embedded_state_does_not_require_an_integration_step() -> None:
    calculator = _calculator()
    state = OrbitalState(1.0e12, 0.005, 0.01, 0.5)
    outcome = calculator.damping_simulation(1.0, state, 0.0, "BH", max_iter=0)
    assert outcome.status is DampingStatus.EMBEDDED_BH
    assert outcome.elapsed_time_s == 0.0


def test_embedding_on_the_last_allowed_step_is_successful() -> None:
    calculator = _calculator()
    calculator.compute_ode_coefficients = lambda *_: (0.0, 0.0, -0.01, 0.0, 0.0, 0.0)
    initial = OrbitalState(1.0e12, 0.005, 0.02005, 0.5)
    outcome = calculator.damping_simulation(1.0, initial, 0.0, "BH", max_iter=1)
    assert outcome.status is DampingStatus.EMBEDDED_BH
    assert 0.0 < outcome.final_state.inclination_rad <= 0.02


def test_stellar_collapse_stop_clips_damping_step_exactly() -> None:
    calculator = _calculator()
    calculator.compute_ode_coefficients = lambda *_: (0.0,) * 6
    calculator.damping_time_step_s = lambda *_: 2.0e6
    initial = OrbitalState(1.0e12, 0.005, 0.2, 0.5)
    outcome = calculator.damping_simulation(
        1.0, initial, 1.0e6 / const.Myr, "Star", stop_time_s=1.0e6
    )
    assert outcome.status is DampingStatus.COLLAPSE_REACHED
    assert outcome.final_state.time_s == 1.0e6
    assert outcome.final_state.semi_major_axis_cm == initial.semi_major_axis_cm
