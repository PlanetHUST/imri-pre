"""Coordinate the stages of one dynamical IMRI realization."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np

import physical_process.constants as const
from physical_process.agn import AGNProperties
from physical_process.damp import DampingCalculator, DampingStatus

from .config import IMRIDynConfig
from .engine import IMRIDynEngine
from .event_work import IMRIEventEvaluator
from .models import (
    DampingResult,
    DampingTask,
    DiskProfiles,
    IMRIEventRecord,
    IMRIDynSingleResult,
    SourceTransition,
    WorkflowCheckpoint,
)
from .population import sample_disk_crossing_population
from .replenishment import ConsumptionCycle
from .reporting import vprint_population_ranges
from .runtime import vprint
from .workers import (
    build_damping_tasks,
    damp_task_with_calculator,
    get_hpc_cpu_count,
    run_damping_tasks_parallel,
)


@dataclass(frozen=True)
class SourceOutcome:
    """The final damping state and optional event of one source action chain."""

    damping: DampingResult
    event: IMRIEventRecord | None


class SourcePhase(str, Enum):
    DAMPING = "damping"
    STELLAR_HANDOFF = "stellar_handoff"
    ENTRY_DETECTION = "entry_detection"
    COMPLETE = "complete"


@dataclass
class SourceState:
    """Control state for one primary or donor; physics remains in stage outcomes."""

    phase: SourcePhase
    task: DampingTask | None
    damping: DampingResult | None
    event: IMRIEventRecord | None = None


class IMRISingleWorkflow:
    """Run one Monte Carlo realization using an IMRI engine."""

    def __init__(self, engine: IMRIDynEngine) -> None:
        self.engine = engine
        self.config: IMRIDynConfig = engine.config
        self.duty_cycle_myr = engine.duty_cycle_myr
        self.event_evaluator = IMRIEventEvaluator(engine)

    def run_single_simulation(
        self,
        m_smbh_msun: float,
        m_nsc_msun: float,
        repeat_idx: int,
        task_seed: int | None = None,
    ) -> IMRIDynSingleResult:
        """Run one Monte Carlo realization and classify IMRI candidates."""
        result = IMRIDynSingleResult(
            m_smbh_msun,
            m_nsc_msun,
            repeat_idx,
            self.config.disk_model,
            task_seed,
        )
        cpu_count = (
            self.config.damping_cpu_count
            if self.config.damping_cpu_count is not None
            else get_hpc_cpu_count()
        )
        agn_properties = AGNProperties.from_mass(m_smbh_msun, self.duty_cycle_myr)
        vprint("AGN initialized done.")

        disk_data = self.engine.get_disk_profiles(m_smbh_msun)
        result.disk_inner_radius_cm = disk_data["inner_radius_cm"]
        result.disk_outer_radius_cm = disk_data["outer_radius_cm"]
        result.gravitational_radius_cm = const.G * m_smbh_msun * const.msun / const.c**2
        vprint(
            "Disk profiles cached. "
            f"r_out = {disk_data['outer_radius_cm']:.3e} cm = "
            f"{disk_data['outer_radius_cm']/const.parsec:.3e} pc"
        )

        population = sample_disk_crossing_population(
            m_smbh_msun,
            m_nsc_msun,
            disk_data,
            self.config,
        )

        result.nsc_radius_cm = population.initial["nsc_radius_cm"]
        result.total_sampled_objects = population.total_count
        vprint_population_ranges(
            "initial population",
            population.initial["mass_msun"],
            population.initial["formation_time_myr"],
            population.initial["semi_major_axis_cm"],
            population.initial["eccentricity"],
            population.initial["inclination_rad"],
        )

        result.disk_crossing_objects = population.disk_crossing_count
        result.f_geo = population.geometric_fraction
        source_mask = np.ones(population.total_count, dtype=bool)
        source_mask[population.selected_indices] = False
        source_masses_msun = population.initial["mass_msun"][source_mask]
        result.replenishment_source_initial_count = len(source_masses_msun)
        result.replenishment_source_remaining_count = len(source_masses_msun)
        result.replenishment_source_initial_mass_msun = float(
            np.sum(source_masses_msun)
        )
        result.replenishment_source_remaining_mass_msun = (
            result.replenishment_source_initial_mass_msun
        )
        vprint_population_ranges(
            "disk-crossing population",
            population.disk_crossing["mass_msun"],
            population.disk_crossing["formation_time_myr"],
            population.disk_crossing["semi_major_axis_cm"],
            population.disk_crossing["eccentricity"],
            population.disk_crossing["inclination_rad"],
        )
        self._checkpoint(result, "population_selected")

        if population.disk_crossing_count == 0:
            self._checkpoint(result, "primary_damping_complete")
            self._checkpoint(result, "primary_sources_complete")
            self._checkpoint(result, "consumption_cycle_complete")
            self.event_evaluator.finalize_rates(result)
            self._checkpoint(result, "rates_finalized")
            return result
        vprint(
            f"Geometry selection: {population.disk_crossing_count}/"
            f"{result.total_sampled_objects} objects selected "
            f"(f_geo={result.f_geo:.3e})"
        )

        result.mean_relaxation_time_myr_geo = population.mean_relaxation_time_myr
        vprint(
            "Relaxation times initialized, mean_trel_myr_geo = "
            f"{result.mean_relaxation_time_myr_geo:.3e} Myr"
        )

        # Damp the selected orbits, then classify their IMRI outcomes.
        damping_tasks = build_damping_tasks(
            population.disk_crossing, population.selected_indices, task_seed
        )
        damping_results = sorted(
            run_damping_tasks_parallel(
                damping_tasks,
                agn_properties,
                disk_data["radius_cm"],
                disk_data["surface_density_g_cm2"],
                disk_data["aspect_ratio"],
                disk_data["inner_radius_cm"],
                disk_data["outer_radius_cm"],
                self.config.use_disk_inner_radius,
                cpu_count,
                disk_data["density_g_cm3"],
                self.config,
            ),
            key=lambda item: item["idx"],
        )
        result.primary_damped_count = len(damping_results)
        self._checkpoint(result, "primary_damping_complete")
        calculator: DampingCalculator | None = None

        def get_calculator() -> DampingCalculator:
            nonlocal calculator
            if calculator is None:
                calculator = DampingCalculator(
                    agn_properties,
                    verbose=False,
                    radius_grid_cm=disk_data["radius_cm"],
                    surface_density_g_cm2=disk_data["surface_density_g_cm2"],
                    aspect_ratio=disk_data["aspect_ratio"],
                    disk_inner_radius_cm=disk_data["inner_radius_cm"],
                    disk_outer_radius_cm=disk_data["outer_radius_cm"],
                    use_disk_inner_radius=self.config.use_disk_inner_radius,
                    density_g_cm3=disk_data["density_g_cm3"],
                )
            return calculator

        primary_results: list[DampingResult] = []
        primary_events: list[IMRIEventRecord] = []
        for damping_result in damping_results:
            outcome = self._advance_source(
                result, agn_properties, disk_data, get_calculator,
                damping_result=damping_result,
            )
            primary_results.append(outcome.damping)
            if outcome.event is not None:
                primary_events.append(outcome.event)
        self._checkpoint(result, "primary_sources_complete")

        if self.config.consumption_replenishment:
            cycle = ConsumptionCycle(
                result=result,
                population=population,
                primary_results=primary_results,
                primary_events=primary_events,
                horizon_myr=self.duty_cycle_myr,
            )
            while cycle.has_pending_consumption():
                task = cycle.next_replacement_task()
                if task is None:
                    continue
                outcome = self._advance_source(
                    result, agn_properties, disk_data, get_calculator, task=task
                )
                cycle.complete_replacement(outcome.damping, outcome.event)
            cycle.verify_balance()
        self._checkpoint(result, "consumption_cycle_complete")
        self.event_evaluator.finalize_rates(result)
        self._checkpoint(result, "rates_finalized")
        return result

    def _advance_source(
        self,
        result: IMRIDynSingleResult,
        agn_properties: AGNProperties,
        disk_data: DiskProfiles,
        get_calculator: Callable[[], DampingCalculator],
        *,
        damping_result: DampingResult | None = None,
        task: DampingTask | None = None,
    ) -> SourceOutcome:
        """Apply the same guarded source actions to primaries and donors.

        Primary damping is precomputed in a worker batch; a replacement starts
        at its ISCO injection time and runs its damping action here. Both then
        follow the same stellar handoff and entry/detector classification.
        """
        if (damping_result is None) == (task is None):
            raise ValueError("provide exactly one damping result or replacement task")
        state = SourceState(SourcePhase.DAMPING, task, damping_result)
        while state.phase is not SourcePhase.COMPLETE:
            if state.phase is SourcePhase.DAMPING:
                if state.damping is None:
                    if state.task is None:
                        raise AssertionError("damping phase has no source task")
                    state.damping = damp_task_with_calculator(
                        state.task, get_calculator(), self.config
                    )
                self._record_transition(result, state.damping, "damping", None)
                state.phase = SourcePhase.STELLAR_HANDOFF
            elif state.phase is SourcePhase.STELLAR_HANDOFF:
                if state.damping is None:
                    raise AssertionError("stellar handoff has no damping outcome")
                if (
                    self.config.stellar_evolution_model == "sevn"
                    and state.damping["status"] is DampingStatus.EMBEDDED_STAR
                ):
                    state.damping = self._resolve_stellar_event(
                        result, agn_properties, disk_data, state.damping,
                        get_calculator(),
                    )
                    self._record_transition(
                        result, state.damping, "stellar_handoff", None
                    )
                if self.config.stellar_evolution_model == "sevn":
                    result.stellar_ledger.append(
                        self._stellar_ledger_row(result, state.damping)
                    )
                state.phase = SourcePhase.ENTRY_DETECTION
            elif state.phase is SourcePhase.ENTRY_DETECTION:
                if state.damping is None:
                    raise AssertionError("entry classification has no source state")
                events = self.event_evaluator.classify_imri_candidates(
                    result, agn_properties, disk_data, [state.damping]
                )
                state.event = events[0] if events else None
                self._record_transition(
                    result, state.damping, "entry_detection", state.event
                )
                state.phase = SourcePhase.COMPLETE
            else:
                raise AssertionError(f"unknown source phase: {state.phase}")
        result.sources_processed_count += 1
        if state.damping is None:
            raise AssertionError("completed source has no damping outcome")
        return SourceOutcome(state.damping, state.event)

    @staticmethod
    def _record_transition(
        result: IMRIDynSingleResult,
        damping_result: DampingResult,
        stage: str,
        event: IMRIEventRecord | None,
    ) -> None:
        row: SourceTransition = {
            "m_smbh_msun": result.m_smbh_msun,
            "m_nsc_msun": result.m_nsc_msun,
            "repeat": result.repeat_idx,
            "object_index": damping_result["idx"],
            "source_id": damping_result["source_id"],
            "task_kind": damping_result["task_kind"],
            "origin_primary_idx": damping_result["origin_primary_idx"],
            "copy_number": damping_result["copy_number"],
            "stage": stage,
            "injection_time_myr": damping_result["state_time_myr"],
            "time_myr": damping_result["final_time_myr"],
            "collapse_time_myr": damping_result["formation_time_myr"],
            "mass_msun": damping_result["mass_msun"],
            "status": (
                event["event_rate_class"] if event is not None
                else damping_result["status"].value
            ),
            "reason": (
                event["event_rate_reason"] if event is not None
                else damping_result.get("stellar_outcome", damping_result["status"].value)
            ),
            "event_id": event["event_id"] if event is not None else None,
            "band_entry_time_myr": (
                event["detection_time_myr"] if event is not None else None
            ),
            "consumption_time_myr": (
                event["consumption_time_myr"] if event is not None else None
            ),
        }
        result.source_transitions.append(row)

    def _checkpoint(self, result: IMRIDynSingleResult, stage: str) -> None:
        row: WorkflowCheckpoint = {
            "m_smbh_msun": result.m_smbh_msun,
            "m_nsc_msun": result.m_nsc_msun,
            "repeat": result.repeat_idx,
            "task_seed": result.task_seed,
            "disk_model": result.disk_model,
            "stellar_evolution_model": self.config.stellar_evolution_model,
            "natal_kick_model": self.config.natal_kick_model,
            "redshift": self.config.redshift,
            "duty_cycle_myr": self.duty_cycle_myr,
            "stage": stage,
            "n_sampled": result.total_sampled_objects,
            "n_disk_cross": result.disk_crossing_objects,
            "n_primary_damped": result.primary_damped_count,
            "n_sources_processed": result.sources_processed_count,
            "n_stellar_rows": len(result.stellar_ledger),
            "n_event_rows": len(result.imri_events),
            "n_imri": result.imri_count,
            "n_det": result.detectable_imri_count,
            "n_stellar_unavailable": result.stellar_unavailable_count,
            "n_entry_inside_isco": result.band_entry_inside_isco_count,
            "n_profile_error": result.profile_error_count,
            "n_replenishment_unavailable": result.replenishment_unavailable_count,
            "n_detector_unavailable": result.detector_unavailable_imri_count,
            "n_isco_consumed": result.replenishment_consumption_count,
            "n_replenished": result.replenishment_injection_count,
            "n_source_remaining": result.replenishment_source_remaining_count,
            "rate_imri_per_myr": (
                result.imri_rate_per_myr if stage == "rates_finalized" else None
            ),
            "rate_det_per_myr": (
                result.detectable_imri_rate_per_myr if stage == "rates_finalized" else None
            ),
        }
        result.workflow_checkpoints.append(row)

    def _resolve_stellar_event(
        self,
        result: IMRIDynSingleResult,
        agn_properties: AGNProperties,
        disk_data,
        item: dict,
        calculator: DampingCalculator,
    ) -> dict:
        try:
            return self.event_evaluator.resolve_embedded_stellar_event(
                result, agn_properties, disk_data, item, calculator
            )
        except Exception as error:
            unavailable = item.copy()
            unavailable["status"] = DampingStatus.STELLAR_UNAVAILABLE
            unavailable["stellar_outcome"] = "stellar_event_evaluation_failed"
            unavailable["error"] = f"{type(error).__name__}: {error}"
            return unavailable

    @staticmethod
    def _stellar_ledger_row(result: IMRIDynSingleResult, item: dict) -> dict:
        return {
            "m_smbh_msun": result.m_smbh_msun,
            "m_nsc_msun": result.m_nsc_msun,
            "repeat": result.repeat_idx,
            "object_index": item["idx"],
            "source_id": item["source_id"],
            "task_kind": item["task_kind"],
            "injection_time_myr": item["state_time_myr"],
            "final_time_myr": item["final_time_myr"],
            "final_semi_major_axis_cm": item["final_semi_major_axis_cm"],
            "final_eccentricity": item["final_eccentricity"],
            "final_inclination_rad": item["final_inclination_rad"],
            "zams_mass_msun": item.get("zams_mass_msun", item["mass_msun"]),
            "presupernova_mass_msun": item.get("presupernova_mass_msun"),
            "presupernova_phase": item.get("presupernova_phase"),
            "collapse_time_myr": item["formation_time_myr"],
            "remnant_mass_msun": item.get("remnant_mass_msun"),
            "remnant_type": item.get("remnant_type"),
            "sevn_seed": item.get("sevn_seed"),
            "kick_speed_km_s": item.get("kick_speed_km_s"),
            "kick_applied": item.get("kick_applied", False),
            "stellar_outcome": item.get("stellar_outcome", "unavailable"),
            "damping_status": item["status"].value,
            "radius_at_collapse_cm": item.get("radius_at_collapse_cm"),
            "postkick_semi_major_axis_cm": item.get("postkick_semi_major_axis_cm"),
            "postkick_eccentricity": item.get("postkick_eccentricity"),
            "postkick_inclination_rad": item.get("postkick_inclination_rad"),
            "postkick_damping_status": item.get("postkick_damping_status"),
            "postkick_gw_to_band_z0_myr": item.get("postkick_gw_to_band_z0_myr"),
            "error": item.get("error"),
        }
