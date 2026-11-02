"""Multiprocessing boundary for IMRI orbital damping tasks."""

import os
import multiprocessing
from multiprocessing import Pool
from typing import Sequence

import physical_process.constants as const
from physical_process import gw_frequency_radius as gw_helpers
from physical_process.agn import AGNProperties
from physical_process.damp import DampingCalculator, DampingStatus, OrbitalState
from physical_process.redshift_evolve import RedshiftEvolution

from .models import DampingResult, DampingTask, FloatArray, GeometricPopulation
from .runtime import vprint
from .config import IMRIDynConfig
from .stellar_evolution import apply_sevn_kick, evolve_stellar_endpoint

_WORKER_CALCULATOR: DampingCalculator | None = None
_WORKER_CONFIG: IMRIDynConfig | None = None


def build_damping_tasks(
    population: GeometricPopulation,
    selected_indices: Sequence[int] | None = None,
    task_seed: int | None = None,
) -> list[DampingTask]:
    """Serialize one selected population for the damping worker boundary."""
    return [
        {
            "task_kind": "primary",
            "idx": index,
            "mass_g": float(population["mass_msun"][index] * const.msun),
            "semi_major_axis_cm": float(population["semi_major_axis_cm"][index]),
            "eccentricity": float(population["eccentricity"][index]),
            "inclination_rad": float(population["inclination_rad"][index]),
            "cos_omega": float(population["cos_omega"][index]),
            "formation_time_myr": float(population["formation_time_myr"][index]),
            "state_time_s": 0.0,
            "source_id": (
                int(selected_indices[index])
                if selected_indices is not None
                else index
            ),
            "origin_primary_idx": index,
            "copy_number": 0,
            "task_seed": task_seed,
        }
        for index in range(len(population["mass_msun"]))
    ]


def get_hpc_cpu_count() -> int:
    """Return scheduler-assigned CPU count, falling back to local CPUs."""
    for variable in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        value = os.environ.get(variable)
        if value is not None:
            try:
                return max(1, int(value))
            except ValueError:
                pass
    try:
        return max(1, multiprocessing.cpu_count() - 1)
    except NotImplementedError:
        return 2


def init_damping_worker(
    agn_properties: AGNProperties,
    radius_cm: FloatArray,
    surface_density_g_cm2: FloatArray,
    aspect_ratio: FloatArray,
    disk_inner_radius_cm: float,
    disk_outer_radius_cm: float,
    use_disk_inner_radius: bool,
    density_g_cm3: FloatArray,
    config: IMRIDynConfig | None = None,
) -> None:
    """Create one damping calculator per worker process."""
    global _WORKER_CALCULATOR, _WORKER_CONFIG
    _WORKER_CONFIG = config
    _WORKER_CALCULATOR = DampingCalculator(
        agn_properties,
        verbose=False,
        radius_grid_cm=radius_cm,
        surface_density_g_cm2=surface_density_g_cm2,
        aspect_ratio=aspect_ratio,
        disk_inner_radius_cm=disk_inner_radius_cm,
        disk_outer_radius_cm=disk_outer_radius_cm,
        use_disk_inner_radius=use_disk_inner_radius,
        density_g_cm3=density_g_cm3,
    )


def process_damping_task(task: DampingTask) -> DampingResult:
    """Run one damping task for a disk-crossing object."""
    if _WORKER_CALCULATOR is None:
        raise RuntimeError("damping worker was not initialized")
    return damp_task_with_calculator(task, _WORKER_CALCULATOR, _WORKER_CONFIG)


def damp_task_with_calculator(
    task: DampingTask,
    calculator: DampingCalculator,
    config: IMRIDynConfig | None = None,
) -> DampingResult:
    """Run one task with a reusable calculator in a causal replacement chain."""

    state = OrbitalState(
        semi_major_axis_cm=task["semi_major_axis_cm"],
        eccentricity=task["eccentricity"],
        inclination_rad=task["inclination_rad"],
        cos_omega=task["cos_omega"],
        time_s=task["state_time_s"],
    )
    stellar_extra: dict[str, float | int | bool | str] = {}
    final_mass_g = task["mass_g"]
    formation_time_myr = task["formation_time_myr"]
    try:
        if config is not None and config.stellar_evolution_model == "sevn":
            if config.sevn_hydrogen_tables is None or config.sevn_helium_tables is None:
                raise RuntimeError("SEVN table paths are not configured")
            endpoint = evolve_stellar_endpoint(
                source_id=task.get("source_id", task["idx"]),
                zams_mass_msun=task["mass_g"] / const.msun,
                metallicity_zsun=config.metallicity,
                task_seed=task.get("task_seed"),
                hydrogen_tables=config.sevn_hydrogen_tables,
                helium_tables=config.sevn_helium_tables,
            )
            collapse_s = endpoint.collapse_time_myr * const.Myr
            formation_time_myr = endpoint.collapse_time_myr
            stellar_extra = {
                "zams_mass_msun": endpoint.zams_mass_msun,
                "presupernova_mass_msun": endpoint.presupernova_mass_msun,
                "presupernova_phase": endpoint.presupernova_phase,
                "remnant_mass_msun": endpoint.remnant_mass_msun,
                "remnant_type": endpoint.remnant_type,
                "sevn_seed": endpoint.seed,
                "kick_speed_km_s": endpoint.kick_speed_km_s,
                "kick_applied": False,
                "stellar_outcome": "star_during_damping",
            }
            if collapse_s <= state.time_s:
                # The source was already a remnant when its assumed channel
                # orbit was assigned. Its historical kick is not repeated.
                stellar_extra["stellar_outcome"] = "remnant_at_injection"
                if not endpoint.is_black_hole:
                    final_state, status = state, DampingStatus.NO_BH_REMNANT
                else:
                    final_mass_g = endpoint.remnant_mass_msun * const.msun
                    evolved = calculator.damping_simulation(
                        final_mass_g, state, endpoint.collapse_time_myr, "BH"
                    )
                    final_state, status = evolved.final_state, evolved.status
            else:
                pre = calculator.damping_simulation(
                    task["mass_g"], state, endpoint.collapse_time_myr, "Star",
                    stop_time_s=(
                        collapse_s
                        if collapse_s < config.duty_cycle_myr * const.Myr
                        else None
                    ),
                )
                final_state, status = pre.final_state, pre.status
                if status is DampingStatus.COLLAPSE_REACHED:
                    stellar_extra["stellar_outcome"] = "collapse_during_damping"
                    stellar_extra["radius_at_collapse_cm"] = final_state.semi_major_axis_cm
                    if not endpoint.is_black_hole:
                        status = DampingStatus.NO_BH_REMNANT
                    else:
                        if config.natal_kick_model == "sevn":
                            kicked = apply_sevn_kick(
                                final_state,
                                kick_speed_km_s=endpoint.kick_speed_km_s,
                                smbh_mass_msun=calculator.agn_properties.smbh_mass_msun,
                                seed=endpoint.seed,
                            )
                            stellar_extra["kick_applied"] = endpoint.kick_speed_km_s > 0.0
                            if kicked is None:
                                status = DampingStatus.KICK_UNBOUND
                            else:
                                final_state = kicked
                                stellar_extra["postkick_semi_major_axis_cm"] = (
                                    kicked.semi_major_axis_cm
                                )
                                stellar_extra["postkick_eccentricity"] = kicked.eccentricity
                                stellar_extra["postkick_inclination_rad"] = kicked.inclination_rad
                        final_mass_g = endpoint.remnant_mass_msun * const.msun
                        if status is not DampingStatus.KICK_UNBOUND:
                            if calculator.embedding_criteria(final_state).eligible:
                                status = DampingStatus.EMBEDDED_BH
                            else:
                                post = calculator.damping_simulation(
                                    final_mass_g, final_state,
                                    endpoint.collapse_time_myr, "BH",
                                )
                                final_state, status = post.final_state, post.status
                elif status is DampingStatus.EMBEDDED_STAR:
                    stellar_extra["stellar_outcome"] = "collapse_after_embedding"
                if (
                    endpoint.is_black_hole
                    and stellar_extra["kick_applied"]
                    and status in (
                        DampingStatus.DAMPING_INCOMPLETE,
                        DampingStatus.OUTSIDE_DISK,
                    )
                ):
                    gw_to_band_myr = gw_helpers.optimistic_eccentric_band_time_myr(
                        calculator.agn_properties.smbh_mass_msun,
                        endpoint.remnant_mass_msun,
                        final_state.semi_major_axis_cm,
                        final_state.eccentricity,
                        config.lisa_min_gw_frequency_hz,
                    )
                    stellar_extra["postkick_gw_to_band_z0_myr"] = gw_to_band_myr
                    redshift_model = RedshiftEvolution(
                        config.detector_hubble_constant_km_s_mpc,
                        config.detector_omega_m,
                    )
                    cosmic_remaining_myr = (
                        redshift_model.remaining_time_to_zero_redshift_myr(
                            config.redshift
                        ) - final_state.time_s / const.Myr
                    )
                    if gw_to_band_myr > cosmic_remaining_myr:
                        stellar_extra["stellar_outcome"] = "postkick_no_band_before_z0"
                    else:
                        status = DampingStatus.STELLAR_UNAVAILABLE
                        stellar_extra["stellar_outcome"] = (
                            "postkick_possible_eccentric_entry"
                        )
        else:
            outcome = calculator.damping_simulation(
                task["mass_g"], state, task["formation_time_myr"], "Star"
            )
            final_state, status = outcome.final_state, outcome.status
        damping_time_myr = max(0.0, final_state.time_s - state.time_s) / const.Myr
        error = None
    except Exception as exc:
        final_state = state
        status = (
            DampingStatus.STELLAR_UNAVAILABLE
            if config is not None and config.stellar_evolution_model == "sevn"
            else DampingStatus.CALCULATION_ERROR
        )
        damping_time_myr = float("nan")
        error = f"{type(exc).__name__}: {exc}"

    result: DampingResult = {
        "task_kind": task["task_kind"],
        "idx": task["idx"],
        "final_time_s": final_state.time_s,
        "final_time_myr": final_state.time_s / const.Myr,
        "final_semi_major_axis_cm": final_state.semi_major_axis_cm,
        "final_eccentricity": final_state.eccentricity,
        "final_inclination_rad": final_state.inclination_rad,
        "final_cos_omega": final_state.cos_omega,
        "status": status,
        "formation_time_myr": formation_time_myr,
        "state_time_myr": task["state_time_s"] / const.Myr,
        "damping_time_myr": damping_time_myr,
        "initial_semi_major_axis_cm": task["semi_major_axis_cm"],
        "initial_eccentricity": task["eccentricity"],
        "initial_inclination_rad": task["inclination_rad"],
        "initial_cos_omega": task["cos_omega"],
        "mass_g": final_mass_g,
        "mass_msun": final_mass_g / const.msun,
        "source_id": task.get("source_id", task["idx"]),
        "origin_primary_idx": task.get("origin_primary_idx", task["idx"]),
        "copy_number": task.get("copy_number", 0),
    }
    if error is not None:
        result["error"] = error
    result.update(stellar_extra)
    return result


def run_damping_tasks_parallel(
    tasks: Sequence[DampingTask],
    agn_properties: AGNProperties,
    radius_cm: FloatArray,
    surface_density_g_cm2: FloatArray,
    aspect_ratio: FloatArray,
    disk_inner_radius_cm: float,
    disk_outer_radius_cm: float,
    use_disk_inner_radius: bool,
    cpu_count: int,
    density_g_cm3: FloatArray,
    config: IMRIDynConfig | None = None,
) -> list[DampingResult]:
    """Run damping tasks in a worker pool."""
    if not tasks:
        return []
    process_count = max(1, min(cpu_count, len(tasks)))
    chunk_size = max(1, min(8, len(tasks) // max(1, 4 * process_count)))
    vprint(
        f"Running {len(tasks)} damping tasks with "
        f"cpus={process_count}, chunksize={chunk_size}"
    )
    with Pool(
        processes=process_count,
        initializer=init_damping_worker,
        initargs=(
            agn_properties,
            radius_cm,
            surface_density_g_cm2,
            aspect_ratio,
            disk_inner_radius_cm,
            disk_outer_radius_cm,
            use_disk_inner_radius,
            density_g_cm3,
            config,
        ),
    ) as pool:
        return list(
            pool.imap_unordered(process_damping_task, tasks, chunksize=chunk_size)
        )
