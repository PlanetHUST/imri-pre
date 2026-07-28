"""Top-level grid workflow for dynamical IMRI-rate calculations."""

import time

import numpy as np
import pandas as pd

from .config import IMRIDynConfig, SECONDS_PER_MINUTE
from .engine import IMRIDynEngine
from .models import IMRIDynAveragedResult, IMRIDynSingleResult
from .outputs import save_events, save_metadata, save_results


def print_header(text: str, width: int = 72, char: str = "=") -> None:
    """Print a centered workflow heading."""
    print(f"\n{char * width}")
    print(f"{text:^{width}}")
    print(f"{char * width}")


def debug_print_result(result: IMRIDynSingleResult) -> None:
    """Print detailed counters for one realization."""
    print("\n========== DEBUG INFO ==========")
    print(f"Sampled objects: {result.total_sampled_objects}")
    print(f"Disk crossing: {result.disk_crossing_objects}")
    print(f"Embedded stars: {result.embedded_star_count}")
    print(f"Embedded BHs: {result.embedded_bh_count}")
    print(f"BH candidates: {result.bh_candidate_count}")
    print(f"IMRI count: {result.imri_count}")
    print(f"IMRI mass: {result.imri_mass_msun:.3e} Msun")
    print("================================\n")


def run_simulation(config: IMRIDynConfig | None = None) -> pd.DataFrame:
    """Run the configured IMRI mass grid and return averaged results."""
    if config is None:
        config = IMRIDynConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print_header("IMRI DYN SIMULATION", 72, "#")
    smbh_masses_msun, nsc_masses_msun = config.get_param_grid()
    grid_size = len(smbh_masses_msun) * len(nsc_masses_msun)
    print(f"Grid size: {len(smbh_masses_msun)} x {len(nsc_masses_msun)} = {grid_size}")
    print(f"Repeats per point: {config.num_repeats}")
    print(f"Duty cycle: {config.duty_cycle_myr:.3e} Myr")
    print(f"Metallicity: {config.metallicity:.3e} Zsun")
    print(f"Type-II migration: {config.consider_type_ii_migration}")
    print(
        "LISA GW band: "
        f"{config.lisa_min_gw_frequency_hz:.3e} to "
        f"{config.lisa_max_gw_frequency_hz:.3e} Hz"
    )
    print(f"Maximum GW time: {config.max_gw_time_myr:.3e} Myr")
    print(f"Output directory: {config.output_dir}")
    print(
        "Damping CPU count: "
        f"{config.damping_cpu_count if config.damping_cpu_count is not None else 'auto'}"
    )

    engine = IMRIDynEngine(config)
    averaged_results: list[IMRIDynAveragedResult] = []
    simulation_count = 0
    start_time_s = time.time()

    for smbh_index, smbh_mass_msun in enumerate(smbh_masses_msun):
        for nsc_index, nsc_mass_msun in enumerate(nsc_masses_msun):
            grid_index = smbh_index * len(nsc_masses_msun) + nsc_index + 1
            print_header(
                f"Grid Point {grid_index}/{grid_size}: "
                f"M_SMBH={smbh_mass_msun:.2e}, "
                f"M_NSC={nsc_mass_msun:.2e}",
                72,
                "-",
            )
            repeat_results: list[IMRIDynSingleResult] = []
            for repeat_index in range(config.num_repeats):
                print(f"Repeat {repeat_index + 1}/{config.num_repeats}")
                result = engine.run_single_simulation(
                    float(smbh_mass_msun),
                    float(nsc_mass_msun),
                    repeat_index,
                )
                _print_result_summary(result)
                if config.print_debug_results:
                    debug_print_result(result)
                repeat_results.append(result)
                simulation_count += 1

            averaged_result = IMRIDynAveragedResult(
                float(smbh_mass_msun),
                float(nsc_mass_msun),
                repeat_results,
            )
            averaged_results.append(averaged_result)
            if (
                grid_index % config.checkpoint_grid_interval == 0
                or grid_index == grid_size
            ):
                save_results(averaged_results, config.output_dir)
                save_events(averaged_results, config.output_dir)

    runtime_s = time.time() - start_time_s
    print_header("SIMULATION COMPLETE", 72, "#")
    print(f"Total simulations: {simulation_count}")
    print(f"Runtime: {runtime_s:.2f} s " f"({runtime_s/SECONDS_PER_MINUTE:.2f} min)")
    frame = save_results(averaged_results, config.output_dir)
    save_events(averaged_results, config.output_dir)
    save_metadata(
        config.output_dir,
        config,
        {
            "workflow": "imri",
            "total_simulations": simulation_count,
            "total_runtime_seconds": runtime_s,
            "total_runtime_minutes": runtime_s / SECONDS_PER_MINUTE,
            "average_time_per_simulation_seconds": (
                runtime_s / simulation_count if simulation_count > 0 else np.nan
            ),
        },
    )
    return frame


def _print_result_summary(result: IMRIDynSingleResult) -> None:
    print(f"  total sampled objects : {result.total_sampled_objects:.3e}")
    print(f"  disk-crossing objects : {result.disk_crossing_objects:.3e}")
    print(f"  f_geo                 : {result.f_geo:.3e}")
    print(f"  embedded BHs          : {result.embedded_bh_count:.3e}")
    print(f"  BH candidates         : {result.bh_candidate_count:.3e}")
    print(f"  IMRI count            : {result.imri_count:.3e}")
    print(f"  IMRI rate             : {result.imri_rate_per_myr:.3e} /Myr")
    print(
        f"  IMRI mass rate        : "
        f"{result.imri_mass_rate_msun_per_myr:.3e} Msun/Myr"
    )


main = run_simulation
