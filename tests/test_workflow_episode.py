"""Causal donor scheduling at the single-halo episode boundary."""

import numpy as np

from physical_process.damp import DampingStatus
from workflows.models import IMRIDynSingleResult
from workflows.population import PopulationSelection
from workflows.replenishment import ConsumptionCycle


def test_consumption_cycle_keeps_time_order_and_finite_source_balance() -> None:
    masses = np.array([20.0, 30.0, 40.0, 50.0])
    initial = {
        "mass_msun": masses,
        "formation_time_myr": np.array([5.0, 5.0, 5.0, 5.0]),
        "semi_major_axis_cm": np.full(4, 1.0e15),
        "eccentricity": np.full(4, 0.5),
        "inclination_rad": np.full(4, 0.2),
        "cos_omega": np.ones(4),
        "nsc_radius_cm": 1.0e18,
    }
    selected = np.array([0, 1])
    crossing = {name: values[selected] for name, values in initial.items()
                if name != "nsc_radius_cm"}
    population = PopulationSelection(initial, crossing, 1.0, selected)
    result = IMRIDynSingleResult(1.0e5, 2.0e5, 0, "torque_with_trap", 77)
    primaries = [
        {"idx": idx, "initial_semi_major_axis_cm": 1.0e15,
         "initial_eccentricity": 0.5, "initial_inclination_rad": 0.2,
         "initial_cos_omega": 1.0}
        for idx in (0, 1)
    ]

    def event(idx: int, origin: int, injection: float, consumption: float) -> dict:
        return {
            "event_id": f"event-{idx}", "object_index": idx,
            "origin_primary_idx": origin, "injection_time_myr": injection,
            "consumption_time_myr": consumption,
            "consumption_status": "isco_within_duty_cycle",
        }

    cycle = ConsumptionCycle(
        result=result, population=population, primary_results=primaries,
        primary_events=[event(0, 0, 0.0, 2.0), event(1, 1, 0.0, 1.0)],
        horizon_myr=10.0,
    )
    first = cycle.next_replacement_task()
    assert first is not None and first["origin_primary_idx"] == 1
    assert first["state_time_s"] > 0.0
    cycle.complete_replacement(
        {"status": DampingStatus.EMBEDDED_BH},
        event(first["idx"], 1, 1.0, 3.0),
    )
    second = cycle.next_replacement_task()
    assert second is not None and second["origin_primary_idx"] == 0
    assert second["source_id"] != first["source_id"]
    cycle.complete_replacement({"status": DampingStatus.DAMPING_INCOMPLETE}, None)
    assert cycle.next_replacement_task() is None
    assert not cycle.has_pending_consumption()
    cycle.verify_balance()

    assert [row["consumption_time_myr"] for row in result.replenishment_ledger] == [1.0, 2.0, 3.0]
    assert [row["status"] for row in result.replenishment_ledger] == [
        "injected", "injected", "source_exhausted"
    ]
    assert result.replenishment_injection_count == 2
    assert result.replenishment_source_remaining_count == 0
    assert result.replenishment_source_remaining_mass_msun == 0.0
