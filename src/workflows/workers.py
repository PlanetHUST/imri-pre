"""Multiprocessing boundary for IMRI orbital damping tasks."""

import os
import multiprocessing
from multiprocessing import Pool
from typing import Sequence

import physical_process.constants as const
from physical_process.agn import AGNProperties
from physical_process.damp import DampingCalculator, DampingStatus, OrbitalState

from .models import DampingResult, DampingTask, FloatArray, GeometricPopulation
from .runtime import vprint

_WORKER_CALCULATOR: DampingCalculator | None = None


def build_damping_tasks(
    population: GeometricPopulation, selected_indices: Sequence[int] | None = None
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
) -> None:
    """Create one damping calculator per worker process."""
    global _WORKER_CALCULATOR
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
    return damp_task_with_calculator(task, _WORKER_CALCULATOR)


def damp_task_with_calculator(
    task: DampingTask, calculator: DampingCalculator
) -> DampingResult:
    """Run one task with a reusable calculator in a causal replacement chain."""

    state = OrbitalState(
        semi_major_axis_cm=task["semi_major_axis_cm"],
        eccentricity=task["eccentricity"],
        inclination_rad=task["inclination_rad"],
        cos_omega=task["cos_omega"],
        time_s=task["state_time_s"],
    )
    try:
        outcome = calculator.damping_simulation(
            task["mass_g"],
            state,
            task["formation_time_myr"],
            "Star",
        )
        final_state = outcome.final_state
        status = outcome.status
        damping_time_myr = outcome.elapsed_time_s / const.Myr
        error = None
    except Exception as exc:
        final_state = state
        status = DampingStatus.CALCULATION_ERROR
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
        "formation_time_myr": task["formation_time_myr"],
        "state_time_myr": task["state_time_s"] / const.Myr,
        "damping_time_myr": damping_time_myr,
        "initial_semi_major_axis_cm": task["semi_major_axis_cm"],
        "initial_eccentricity": task["eccentricity"],
        "initial_inclination_rad": task["inclination_rad"],
        "initial_cos_omega": task["cos_omega"],
        "mass_g": task["mass_g"],
        "mass_msun": task["mass_g"] / const.msun,
        "source_id": task.get("source_id", task["idx"]),
        "origin_primary_idx": task.get("origin_primary_idx", task["idx"]),
        "copy_number": task.get("copy_number", 0),
    }
    if error is not None:
        result["error"] = error
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
        ),
    ) as pool:
        return list(
            pool.imap_unordered(process_damping_task, tasks, chunksize=chunk_size)
        )
