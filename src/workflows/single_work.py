"""Coordinate the stages of one dynamical IMRI realization."""

import numpy as np

import physical_process.constants as const
from physical_process.agn import AGNProperties
from physical_process.damp import DampingCalculator

from .config import IMRIDynConfig
from .engine import IMRIDynEngine
from .event_work import IMRIEventEvaluator
from .models import (
    IMRIDynSingleResult,
)
from .population import sample_disk_crossing_population
from .replenishment import run_consumption_cycle
from .reporting import vprint_population_ranges
from .runtime import vprint
from .workers import (
    build_damping_tasks,
    damp_task_with_calculator,
    get_hpc_cpu_count,
    run_damping_tasks_parallel,
)


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

        if population.disk_crossing_count == 0:
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
            population.disk_crossing, population.selected_indices
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
            ),
            key=lambda item: item["idx"],
        )
        primary_events = self.event_evaluator.classify_imri_candidates(
            result, agn_properties, disk_data, damping_results
        )
        if self.config.consumption_replenishment:
            calculator: DampingCalculator | None = None

            def evaluate_copy(task):
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
                damping_result = damp_task_with_calculator(task, calculator)
                events = self.event_evaluator.classify_imri_candidates(
                    result, agn_properties, disk_data, [damping_result]
                )
                return damping_result, events[0] if events else None

            run_consumption_cycle(
                result=result,
                population=population,
                primary_results=damping_results,
                primary_events=primary_events,
                horizon_myr=self.duty_cycle_myr,
                evaluate_copy=evaluate_copy,
            )
        return result
