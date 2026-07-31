"""Multiprocessing boundary for IMRI orbital damping tasks."""

import os
from multiprocessing import Pool
from typing import Sequence

import physical_process.constants as const
import physical_process.nsc_orbital as nsc_orbital
from physical_process.damp import DampingCalculator, OrbitalState
from physical_process.general_class import AGN_properties

from .models import DampingResult, DampingTask, FloatArray
from .runtime import vprint

_WORKER_CALCULATOR: DampingCalculator | None = None


def get_hpc_cpu_count() -> int:
    """Return scheduler-assigned CPU count, falling back to local CPUs."""
    for variable in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        value = os.environ.get(variable)
        if value is not None:
            try:
                return max(1, int(value))
            except ValueError:
                pass
    return nsc_orbital.get_cpu_count()


def init_damping_worker(
    agn_properties: AGN_properties,
    radius_cm: FloatArray,
    surface_density_g_cm2: FloatArray,
    aspect_ratio: FloatArray,
) -> None:
    """Create one damping calculator per worker process."""
    global _WORKER_CALCULATOR
    _WORKER_CALCULATOR = DampingCalculator(
        agn_properties,
        verbose=False,
        r_list=radius_cm,
        sigma_list=surface_density_g_cm2,
        h_list=aspect_ratio,
    )


def process_damping_task(task: DampingTask) -> DampingResult:
    """Run one damping task for a disk-crossing object."""
    if _WORKER_CALCULATOR is None:
        raise RuntimeError("damping worker was not initialized")

    try:
        state = OrbitalState(
            task["semi_major_axis_cm"],
            task["eccentricity"],
            task["inclination_rad"],
            task["cos_omega"],
            task["state_time_s"],
        )
        final_time_s, final_axis_cm, object_type = (
            _WORKER_CALCULATOR.damping_simulation(
                task["mass_g"],
                state,
                task["formation_time_myr"],
                "Star",
                task["type_flag"],
            )
        )
        if object_type == 2 and final_time_s == 0.0:
            damping_time_myr = float("nan")
        else:
            damping_time_myr = max(
                0.0,
                final_time_s / const.Myr - task["state_time_s"] / const.Myr,
            )
    except Exception:
        final_time_s = 0.0
        final_axis_cm = 0.0
        object_type = -1
        damping_time_myr = float("nan")

    return {
        "task_kind": task["task_kind"],
        "idx": task["idx"],
        "final_time_s": final_time_s,
        "final_time_myr": final_time_s / const.Myr,
        "final_semi_major_axis_cm": final_axis_cm,
        "obj_type": object_type,
        "formation_time_myr": task["formation_time_myr"],
        "state_time_myr": task["state_time_s"] / const.Myr,
        "damping_time_myr": damping_time_myr,
        "initial_semi_major_axis_cm": task["semi_major_axis_cm"],
        "eccentricity": task["eccentricity"],
        "inclination_rad": task["inclination_rad"],
        "cos_omega": task["cos_omega"],
        "mass_g": task["mass_g"],
        "mass_msun": task["mass_g"] / const.msun,
    }


def run_damping_tasks_parallel(
    tasks: Sequence[DampingTask],
    agn_properties: AGN_properties,
    radius_cm: FloatArray,
    surface_density_g_cm2: FloatArray,
    aspect_ratio: FloatArray,
    cpu_count: int,
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
        ),
    ) as pool:
        return list(
            pool.imap_unordered(process_damping_task, tasks, chunksize=chunk_size)
        )
