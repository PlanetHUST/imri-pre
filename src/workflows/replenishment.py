"""Finite NSC supply channels reopened by completed IMRI inspirals."""

from collections.abc import Sequence
import heapq

import numpy as np

from physical_process import constants as const

from .models import DampingResult, DampingTask, IMRIEventRecord, IMRIDynSingleResult
from .population import PopulationSelection


class ConsumptionCycle:
    """Finite donor pool and chronological ISCO agenda for one episode.

    The episode driver owns the loop and evaluates each replacement through
    the same source actions as a primary. This class owns only donor/channel
    accounting and the stable event priority.
    """

    def __init__(
        self,
        *,
        result: IMRIDynSingleResult,
        population: PopulationSelection,
        primary_results: Sequence[DampingResult],
        primary_events: Sequence[IMRIEventRecord],
        horizon_myr: float,
    ) -> None:
        self.result = result
        self.population = population
        self.horizon_myr = horizon_myr
        self.sampled_count = population.total_count
        selected = np.zeros(self.sampled_count, dtype=bool)
        selected[population.selected_indices] = True
        self.available_source_ids = np.flatnonzero(~selected).tolist()
        self.source_mass_msun = population.initial["mass_msun"]
        result.replenishment_source_initial_count = len(self.available_source_ids)
        result.replenishment_source_remaining_count = len(self.available_source_ids)
        result.replenishment_source_initial_mass_msun = float(
            np.sum(self.source_mass_msun[self.available_source_ids])
        )
        result.replenishment_source_remaining_mass_msun = (
            result.replenishment_source_initial_mass_msun
        )
        self.primary_by_index = {item["idx"]: item for item in primary_results}
        self.channel_occupant: dict[int, int | None] = {
            idx: idx for idx in self.primary_by_index
        }
        self.rng = (
            np.random.default_rng(np.random.SeedSequence([result.task_seed, 918273]))
            if result.task_seed is not None
            else np.random.default_rng()
        )
        self.queue: list[tuple[float, int, int, IMRIEventRecord]] = []
        self.copy_number_by_origin: dict[int, int] = {}
        self.pending_row: dict | None = None
        for event in primary_events:
            self.schedule(event)

    def schedule(self, event: IMRIEventRecord) -> None:
        time_myr = float(event["consumption_time_myr"])
        if (
            event["consumption_status"] == "isco_within_duty_cycle"
            and np.isfinite(time_myr)
            and 0.0 <= time_myr < self.horizon_myr
        ):
            if time_myr <= float(event["injection_time_myr"]):
                raise AssertionError("ISCO consumption must follow channel occupation")
            heapq.heappush(
                self.queue,
                (
                    time_myr,
                    int(event["origin_primary_idx"]),
                    int(event["object_index"]),
                    event,
                ),
            )

    def has_pending_consumption(self) -> bool:
        return bool(self.queue)

    def next_replacement_task(self) -> DampingTask | None:
        """Consume the earliest channel and draw a donor when available."""
        if self.pending_row is not None:
            raise AssertionError("complete the pending replacement before advancing")
        consumption_time_myr, origin_idx, predecessor_idx, predecessor = (
            heapq.heappop(self.queue)
        )
        result = self.result
        if self.channel_occupant.get(origin_idx) != predecessor_idx:
            raise AssertionError("consumed object is not the channel occupant")
        result.replenishment_consumption_count += 1
        row = {
            "m_smbh_msun": result.m_smbh_msun,
            "m_nsc_msun": result.m_nsc_msun,
            "repeat": result.repeat_idx,
            "disk_model": result.disk_model,
            "origin_primary_idx": origin_idx,
            "predecessor_object_index": predecessor_idx,
            "predecessor_event_id": predecessor["event_id"],
            "consumption_time_myr": consumption_time_myr,
            "injection_time_myr": None,
            "source_id": None,
            "source_mass_msun": None,
            "replacement_object_index": None,
            "copy_number": None,
            "replacement_damping_status": None,
            "replacement_event_id": None,
            "replacement_consumption_time_myr": None,
            "status": "source_exhausted",
        }
        if not self.available_source_ids:
            self.channel_occupant[origin_idx] = None
            result.replenishment_source_exhausted_count += 1
            result.replenishment_ledger.append(row)
            return None

        source_position = int(self.rng.integers(0, len(self.available_source_ids)))
        source_id = self.available_source_ids[source_position]
        self.available_source_ids[source_position] = self.available_source_ids[-1]
        self.available_source_ids.pop()
        source_mass = float(self.source_mass_msun[source_id])
        result.replenishment_injection_count += 1
        result.replenishment_source_remaining_count -= 1
        result.replenishment_source_remaining_mass_msun -= source_mass
        copy_number = self.copy_number_by_origin.get(origin_idx, 0) + 1
        self.copy_number_by_origin[origin_idx] = copy_number
        primary = self.primary_by_index[origin_idx]
        replacement_idx = self.sampled_count + result.replenishment_injection_count
        self.channel_occupant[origin_idx] = replacement_idx
        task: DampingTask = {
            "task_kind": "replacement",
            "idx": replacement_idx,
            "source_id": source_id,
            "origin_primary_idx": origin_idx,
            "copy_number": copy_number,
            "mass_g": source_mass * const.msun,
            "semi_major_axis_cm": primary["initial_semi_major_axis_cm"],
            "eccentricity": primary["initial_eccentricity"],
            "inclination_rad": primary["initial_inclination_rad"],
            "cos_omega": primary["initial_cos_omega"],
            "formation_time_myr": float(
                self.population.initial["formation_time_myr"][source_id]
            ),
            "state_time_s": consumption_time_myr * const.Myr,
            "task_seed": result.task_seed,
        }
        row.update(
            injection_time_myr=consumption_time_myr,
            source_id=source_id,
            source_mass_msun=source_mass,
            replacement_object_index=replacement_idx,
            copy_number=copy_number,
            status="injected",
        )
        self.pending_row = row
        return task

    def complete_replacement(
        self, damping_result: DampingResult, event: IMRIEventRecord | None
    ) -> None:
        """Record one replacement outcome and schedule its later consumption."""
        if self.pending_row is None:
            raise AssertionError("no replacement is pending")
        row = self.pending_row
        consumption_time_myr = float(row["consumption_time_myr"])
        row.update(
            replacement_damping_status=damping_result["status"].value,
            replacement_event_id=event["event_id"] if event is not None else None,
            replacement_consumption_time_myr=(
                event["consumption_time_myr"] if event is not None else None
            ),
        )
        self.result.replenishment_ledger.append(row)
        self.pending_row = None
        if event is not None:
            next_consumption_time = float(event["consumption_time_myr"])
            if np.isfinite(next_consumption_time) and (
                next_consumption_time <= consumption_time_myr
            ):
                raise AssertionError("replacement consumption must follow injection")
            self.schedule(event)

    def verify_balance(self) -> None:
        """Check finite reservoir count and mass after the agenda is empty."""
        result = self.result
        if self.queue or self.pending_row is not None:
            raise AssertionError("consumption cycle has unfinished events")
        if (
            result.replenishment_source_initial_count
            != result.replenishment_source_remaining_count
            + result.replenishment_injection_count
        ):
            raise AssertionError("finite NSC source count does not balance")
        expected_remaining_mass_msun = (
            result.replenishment_source_initial_mass_msun
            - sum(
                row["source_mass_msun"]
                for row in result.replenishment_ledger
                if row["status"] == "injected"
            )
        )
        if not np.isclose(
            expected_remaining_mass_msun,
            result.replenishment_source_remaining_mass_msun,
            rtol=1.0e-10,
            atol=1.0e-7,
        ):
            raise AssertionError("finite NSC source mass does not balance")
